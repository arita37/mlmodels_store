
  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_all 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '0635d2a358ad260f77f69ce3b3238ee806f53e4b', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/0635d2a358ad260f77f69ce3b3238ee806f53e4b

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/0635d2a358ad260f77f69ce3b3238ee806f53e4b

 ************************************************************************************************************************

  ############Check model ################################ 

  ['model_keras.keras_gan', 'model_keras.nbeats', 'model_keras.01_deepctr', 'model_keras.textvae', 'model_keras.namentity_crm_bilstm_dataloader', 'model_keras.Autokeras', 'model_keras.charcnn_zhang', 'model_keras.charcnn', 'model_keras.namentity_crm_bilstm', 'model_keras.textcnn', 'model_keras.armdn', 'model_keras.02_cnn', 'model_tf.1_lstm', 'model_tf.temporal_fusion_google', 'model_gluon.gluon_automl', 'model_gluon.fb_prophet', 'model_gluon.gluonts_model_old', 'model_gluon.gluonts_model', 'model_sklearn.model_sklearn', 'model_sklearn.model_lightgbm', 'model_tch.nbeats', 'model_tch.transformer_classifier', 'model_tch.matchzoo_models', 'model_tch.torchhub', 'model_tch.03_nbeats_dataloader', 'model_tch.transformer_sentence', 'model_tch.pytorch_vae', 'model_tch.pplm', 'model_tch.textcnn', 'model_tch.mlp'] 

  Used ['model_keras.keras_gan', 'model_keras.nbeats', 'model_keras.01_deepctr', 'model_keras.textvae', 'model_keras.namentity_crm_bilstm_dataloader', 'model_keras.Autokeras', 'model_keras.charcnn_zhang', 'model_keras.charcnn', 'model_keras.namentity_crm_bilstm', 'model_keras.textcnn', 'model_keras.armdn', 'model_keras.02_cnn', 'model_tf.1_lstm', 'model_tf.temporal_fusion_google', 'model_gluon.gluon_automl', 'model_gluon.fb_prophet', 'model_gluon.gluonts_model_old', 'model_gluon.gluonts_model', 'model_sklearn.model_sklearn', 'model_sklearn.model_lightgbm', 'model_tch.nbeats', 'model_tch.transformer_classifier', 'model_tch.matchzoo_models', 'model_tch.torchhub', 'model_tch.03_nbeats_dataloader', 'model_tch.transformer_sentence', 'model_tch.pytorch_vae', 'model_tch.pplm', 'model_tch.textcnn', 'model_tch.mlp'] 





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
[master 17857a2] ml_store  && git pull --all
 2 files changed, 61 insertions(+), 9936 deletions(-)
 rewrite log_testall/log_testall.py (99%)
To github.com:arita37/mlmodels_store.git
   0c3e7d5..17857a2  master -> master





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
[master 0203c77] ml_store  && git pull --all
 1 file changed, 47 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   17857a2..0203c77  master -> master





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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         36          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         20          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-27 20:15:13.954548: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-27 20:15:13.959989: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-27 20:15:13.960236: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ab6905cb50 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-27 20:15:13.960252: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 198
Trainable params: 198
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2475 - binary_crossentropy: 0.6862500/500 [==============================] - 1s 1ms/sample - loss: 0.2507 - binary_crossentropy: 0.7459 - val_loss: 0.2504 - val_binary_crossentropy: 0.7448

  #### metrics   #################################################### 
{'MSE': 0.2502253251743453}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         36          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
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
Total params: 198
Trainable params: 198
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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_max[0][0]               
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
Total params: 468
Trainable params: 468
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2572 - binary_crossentropy: 0.8380500/500 [==============================] - 1s 2ms/sample - loss: 0.2558 - binary_crossentropy: 0.8098 - val_loss: 0.2530 - val_binary_crossentropy: 0.8027

  #### metrics   #################################################### 
{'MSE': 0.2542341258550538}

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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_max[0][0]               
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
Total params: 468
Trainable params: 468
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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
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
Total params: 607
Trainable params: 607
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.5100 - binary_crossentropy: 7.8667500/500 [==============================] - 1s 2ms/sample - loss: 0.4940 - binary_crossentropy: 7.6199 - val_loss: 0.5040 - val_binary_crossentropy: 7.7742

  #### metrics   #################################################### 
{'MSE': 0.499}

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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
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
Total params: 607
Trainable params: 607
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         24          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_2[0][0]           
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
Total params: 398
Trainable params: 398
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2736 - binary_crossentropy: 0.7420500/500 [==============================] - 1s 3ms/sample - loss: 0.2674 - binary_crossentropy: 0.7291 - val_loss: 0.2522 - val_binary_crossentropy: 0.6977

  #### metrics   #################################################### 
{'MSE': 0.25512496296167314}

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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         24          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_2[0][0]           
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
Total params: 398
Trainable params: 398
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
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
100/500 [=====>........................] - ETA: 2s - loss: 0.3470 - binary_crossentropy: 1.9553500/500 [==============================] - 2s 3ms/sample - loss: 0.3183 - binary_crossentropy: 1.8336 - val_loss: 0.3327 - val_binary_crossentropy: 1.9431

  #### metrics   #################################################### 
{'MSE': 0.3248551296970074}

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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-27 20:16:38.219041: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 20:16:38.221291: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 20:16:38.227432: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-27 20:16:38.238122: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-27 20:16:38.240122: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-27 20:16:38.241993: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 20:16:38.243631: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2490 - val_binary_crossentropy: 0.6911
2020-05-27 20:16:39.571566: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 20:16:39.573506: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 20:16:39.578191: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-27 20:16:39.587597: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-27 20:16:39.589201: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-27 20:16:39.591094: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 20:16:39.592627: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2484955222909543}

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
2020-05-27 20:17:04.040890: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 20:17:04.042263: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 20:17:04.046085: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-27 20:17:04.052535: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-27 20:17:04.053673: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-27 20:17:04.054710: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 20:17:04.055685: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
2020-05-27 20:17:05.555428: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 20:17:05.556618: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 20:17:05.559664: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-27 20:17:05.565301: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-27 20:17:05.566662: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-27 20:17:05.567571: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 20:17:05.568408: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2506844111489806}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-27 20:17:40.171704: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 20:17:40.179349: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 20:17:40.195265: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-27 20:17:40.223705: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-27 20:17:40.228190: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-27 20:17:40.232345: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 20:17:40.236311: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.4716 - binary_crossentropy: 1.1607 - val_loss: 0.2512 - val_binary_crossentropy: 0.6955
2020-05-27 20:17:42.383703: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 20:17:42.388351: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 20:17:42.400097: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-27 20:17:42.422555: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-27 20:17:42.426930: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-27 20:17:42.431192: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 20:17:42.434933: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24000337613759606}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
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
Total params: 700
Trainable params: 700
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.4893 - binary_crossentropy: 5.3871500/500 [==============================] - 4s 8ms/sample - loss: 0.4179 - binary_crossentropy: 4.5180 - val_loss: 0.4150 - val_binary_crossentropy: 4.6877

  #### metrics   #################################################### 
{'MSE': 0.4158230234365396}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
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
Total params: 700
Trainable params: 700
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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 2)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 2)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 2)         10          sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         4           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         10          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         14          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_2[0][0]           
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
Total params: 227
Trainable params: 227
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2927 - binary_crossentropy: 1.3074500/500 [==============================] - 5s 9ms/sample - loss: 0.2782 - binary_crossentropy: 1.1186 - val_loss: 0.2649 - val_binary_crossentropy: 0.9847

  #### metrics   #################################################### 
{'MSE': 0.2710375996144835}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 2)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 2)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 2)         10          sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         4           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         10          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         14          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_2[0][0]           
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
Total params: 227
Trainable params: 227
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         8           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         2           sequence_max[0][0]               
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
100/500 [=====>........................] - ETA: 5s - loss: 0.4239 - binary_crossentropy: 4.8380500/500 [==============================] - 4s 9ms/sample - loss: 0.4131 - binary_crossentropy: 4.6037 - val_loss: 0.3856 - val_binary_crossentropy: 4.1222

  #### metrics   #################################################### 
{'MSE': 0.3988739757626838}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         8           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         2           sequence_max[0][0]               
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
regionsequence_sum (InputLayer) [(None, 6)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 8)]          0                                            
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
region_10sparse_seq_emb_regions (None, 6, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 6, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 6, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 6, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 6, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 6, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 6, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 6, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
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
Total params: 168
Trainable params: 168
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2628 - binary_crossentropy: 0.8510500/500 [==============================] - 6s 12ms/sample - loss: 0.2531 - binary_crossentropy: 0.7257 - val_loss: 0.2529 - val_binary_crossentropy: 0.6988

  #### metrics   #################################################### 
{'MSE': 0.2527825162193837}

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
regionsequence_sum (InputLayer) [(None, 6)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 8)]          0                                            
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
region_10sparse_seq_emb_regions (None, 6, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 6, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 6, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 6, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 6, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 6, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 6, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 6, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
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
Total params: 168
Trainable params: 168
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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
Total params: 1,412
Trainable params: 1,412
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2482 - binary_crossentropy: 0.6897500/500 [==============================] - 5s 11ms/sample - loss: 0.2485 - binary_crossentropy: 0.6900 - val_loss: 0.2505 - val_binary_crossentropy: 0.6941

  #### metrics   #################################################### 
{'MSE': 0.25057743797915577}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
Total params: 1,412
Trainable params: 1,412
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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         32          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         16          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 7, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 9, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 1, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 7, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 9, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 1, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 7, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 9, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 7, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 1, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 9, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 1, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           hash_11[0][0]                    
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
Total params: 3,069
Trainable params: 2,989
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.3350 - binary_crossentropy: 2.3003500/500 [==============================] - 6s 12ms/sample - loss: 0.3028 - binary_crossentropy: 2.0235 - val_loss: 0.3160 - val_binary_crossentropy: 2.0838

  #### metrics   #################################################### 
{'MSE': 0.307182377070609}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         32          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         16          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 7, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 9, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 1, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 7, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 9, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 1, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 7, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 9, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 7, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 1, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 9, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 1, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           hash_11[0][0]                    
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
Total params: 3,069
Trainable params: 2,989
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
[master 2af541c] ml_store  && git pull --all
 1 file changed, 4946 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
 + 7cfc87d...2af541c master -> master (forced update)





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
[master b30052a] ml_store  && git pull --all
 1 file changed, 50 insertions(+)
To github.com:arita37/mlmodels_store.git
   2af541c..b30052a  master -> master





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
[master e49890e] ml_store  && git pull --all
 1 file changed, 45 insertions(+)
To github.com:arita37/mlmodels_store.git
   b30052a..e49890e  master -> master





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
[master 5e863bc] ml_store  && git pull --all
 1 file changed, 34 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   e49890e..5e863bc  master -> master





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
[master 93889e4] ml_store  && git pull --all
 1 file changed, 48 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.113.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   5e863bc..93889e4  master -> master





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
[master ad29b51] ml_store  && git pull --all
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   93889e4..ad29b51  master -> master





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
[master ed2be59] ml_store  && git pull --all
 1 file changed, 43 insertions(+)
To github.com:arita37/mlmodels_store.git
   ad29b51..ed2be59  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3244032/17464789 [====>.........................] - ETA: 0s
10764288/17464789 [=================>............] - ETA: 0s
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
2020-05-27 20:27:14.925287: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-27 20:27:14.930048: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-27 20:27:14.930232: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ff5a431c90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-27 20:27:14.930250: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 13s - loss: 7.5440 - accuracy: 0.5080
 2000/25000 [=>............................] - ETA: 10s - loss: 7.5210 - accuracy: 0.5095
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5133 - accuracy: 0.5100 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6973 - accuracy: 0.4980
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6574 - accuracy: 0.5006
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6462 - accuracy: 0.5013
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6732 - accuracy: 0.4996
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7069 - accuracy: 0.4974
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7399 - accuracy: 0.4952
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7157 - accuracy: 0.4968
11000/25000 [============>.................] - ETA: 4s - loss: 7.7140 - accuracy: 0.4969
12000/25000 [=============>................] - ETA: 4s - loss: 7.6858 - accuracy: 0.4988
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6820 - accuracy: 0.4990
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6787 - accuracy: 0.4992
15000/25000 [=================>............] - ETA: 3s - loss: 7.7096 - accuracy: 0.4972
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7356 - accuracy: 0.4955
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7180 - accuracy: 0.4966
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7254 - accuracy: 0.4962
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7126 - accuracy: 0.4970
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7149 - accuracy: 0.4969
21000/25000 [========================>.....] - ETA: 1s - loss: 7.7075 - accuracy: 0.4973
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6994 - accuracy: 0.4979
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6926 - accuracy: 0.4983
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6800 - accuracy: 0.4991
25000/25000 [==============================] - 9s 379us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7f0f0a7421d0>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f0f071704e0> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.8046 - accuracy: 0.4910
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8583 - accuracy: 0.4875 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.8046 - accuracy: 0.4910
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6743 - accuracy: 0.4995
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6237 - accuracy: 0.5028
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6794 - accuracy: 0.4992
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6798 - accuracy: 0.4991
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7011 - accuracy: 0.4978
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6581 - accuracy: 0.5006
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6927 - accuracy: 0.4983
11000/25000 [============>.................] - ETA: 4s - loss: 7.6973 - accuracy: 0.4980
12000/25000 [=============>................] - ETA: 4s - loss: 7.7203 - accuracy: 0.4965
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6902 - accuracy: 0.4985
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7247 - accuracy: 0.4962
15000/25000 [=================>............] - ETA: 3s - loss: 7.6912 - accuracy: 0.4984
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6954 - accuracy: 0.4981
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7099 - accuracy: 0.4972
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7024 - accuracy: 0.4977
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6828 - accuracy: 0.4989
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6720 - accuracy: 0.4997
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6593 - accuracy: 0.5005
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6527 - accuracy: 0.5009
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6640 - accuracy: 0.5002
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6730 - accuracy: 0.4996
25000/25000 [==============================] - 9s 379us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 13s - loss: 7.8966 - accuracy: 0.4850
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8430 - accuracy: 0.4885 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6666 - accuracy: 0.5000
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7740 - accuracy: 0.4930
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7678 - accuracy: 0.4934
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7305 - accuracy: 0.4958
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6732 - accuracy: 0.4996
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6609 - accuracy: 0.5004
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6717 - accuracy: 0.4997
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6237 - accuracy: 0.5028
11000/25000 [============>.................] - ETA: 4s - loss: 7.6025 - accuracy: 0.5042
12000/25000 [=============>................] - ETA: 4s - loss: 7.6078 - accuracy: 0.5038
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6348 - accuracy: 0.5021
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6480 - accuracy: 0.5012
15000/25000 [=================>............] - ETA: 3s - loss: 7.6411 - accuracy: 0.5017
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6513 - accuracy: 0.5010
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6540 - accuracy: 0.5008
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6453 - accuracy: 0.5014
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6642 - accuracy: 0.5002
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6659 - accuracy: 0.5001
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6549 - accuracy: 0.5008
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6771 - accuracy: 0.4993
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6700 - accuracy: 0.4998
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6788 - accuracy: 0.4992
25000/25000 [==============================] - 9s 376us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
[master 3ce8f13] ml_store  && git pull --all
 1 file changed, 315 insertions(+)
To github.com:arita37/mlmodels_store.git
 + 9ede53e...3ce8f13 master -> master (forced update)





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

13/13 [==============================] - 1s 115ms/step - loss: nan
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
[master 0358107] ml_store  && git pull --all
 1 file changed, 124 insertions(+)
To github.com:arita37/mlmodels_store.git
   3ce8f13..0358107  master -> master





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
 1933312/11490434 [====>.........................] - ETA: 0s
10321920/11490434 [=========================>....] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:40 - loss: 2.2969 - categorical_accuracy: 0.1562
   64/60000 [..............................] - ETA: 4:52 - loss: 2.3369 - categorical_accuracy: 0.1250
   96/60000 [..............................] - ETA: 3:53 - loss: 2.3168 - categorical_accuracy: 0.1146
  128/60000 [..............................] - ETA: 3:23 - loss: 2.3045 - categorical_accuracy: 0.1172
  160/60000 [..............................] - ETA: 3:04 - loss: 2.2707 - categorical_accuracy: 0.1562
  192/60000 [..............................] - ETA: 2:51 - loss: 2.2475 - categorical_accuracy: 0.1667
  224/60000 [..............................] - ETA: 2:43 - loss: 2.2019 - categorical_accuracy: 0.1920
  256/60000 [..............................] - ETA: 2:36 - loss: 2.1670 - categorical_accuracy: 0.2109
  288/60000 [..............................] - ETA: 2:31 - loss: 2.1102 - categorical_accuracy: 0.2569
  320/60000 [..............................] - ETA: 2:26 - loss: 2.0706 - categorical_accuracy: 0.2844
  352/60000 [..............................] - ETA: 2:22 - loss: 2.0263 - categorical_accuracy: 0.3068
  384/60000 [..............................] - ETA: 2:19 - loss: 1.9836 - categorical_accuracy: 0.3203
  416/60000 [..............................] - ETA: 2:18 - loss: 1.9374 - categorical_accuracy: 0.3389
  448/60000 [..............................] - ETA: 2:15 - loss: 1.8838 - categorical_accuracy: 0.3549
  480/60000 [..............................] - ETA: 2:13 - loss: 1.8577 - categorical_accuracy: 0.3667
  512/60000 [..............................] - ETA: 2:12 - loss: 1.8688 - categorical_accuracy: 0.3691
  544/60000 [..............................] - ETA: 2:10 - loss: 1.8378 - categorical_accuracy: 0.3879
  576/60000 [..............................] - ETA: 2:09 - loss: 1.8144 - categorical_accuracy: 0.3906
  608/60000 [..............................] - ETA: 2:08 - loss: 1.7901 - categorical_accuracy: 0.3947
  640/60000 [..............................] - ETA: 2:07 - loss: 1.7461 - categorical_accuracy: 0.4078
  672/60000 [..............................] - ETA: 2:06 - loss: 1.7152 - categorical_accuracy: 0.4182
  704/60000 [..............................] - ETA: 2:06 - loss: 1.7180 - categorical_accuracy: 0.4219
  736/60000 [..............................] - ETA: 2:05 - loss: 1.7024 - categorical_accuracy: 0.4266
  768/60000 [..............................] - ETA: 2:04 - loss: 1.6895 - categorical_accuracy: 0.4349
  800/60000 [..............................] - ETA: 2:03 - loss: 1.6714 - categorical_accuracy: 0.4437
  832/60000 [..............................] - ETA: 2:03 - loss: 1.6447 - categorical_accuracy: 0.4531
  864/60000 [..............................] - ETA: 2:02 - loss: 1.6350 - categorical_accuracy: 0.4549
  896/60000 [..............................] - ETA: 2:02 - loss: 1.6109 - categorical_accuracy: 0.4643
  928/60000 [..............................] - ETA: 2:02 - loss: 1.5844 - categorical_accuracy: 0.4752
  960/60000 [..............................] - ETA: 2:01 - loss: 1.5695 - categorical_accuracy: 0.4802
  992/60000 [..............................] - ETA: 2:01 - loss: 1.5445 - categorical_accuracy: 0.4899
 1024/60000 [..............................] - ETA: 2:01 - loss: 1.5172 - categorical_accuracy: 0.4980
 1056/60000 [..............................] - ETA: 2:00 - loss: 1.4926 - categorical_accuracy: 0.5066
 1088/60000 [..............................] - ETA: 2:00 - loss: 1.4784 - categorical_accuracy: 0.5129
 1120/60000 [..............................] - ETA: 1:59 - loss: 1.4619 - categorical_accuracy: 0.5170
 1152/60000 [..............................] - ETA: 1:59 - loss: 1.4403 - categorical_accuracy: 0.5243
 1184/60000 [..............................] - ETA: 1:59 - loss: 1.4247 - categorical_accuracy: 0.5287
 1216/60000 [..............................] - ETA: 1:59 - loss: 1.4104 - categorical_accuracy: 0.5329
 1248/60000 [..............................] - ETA: 1:58 - loss: 1.3991 - categorical_accuracy: 0.5377
 1280/60000 [..............................] - ETA: 1:58 - loss: 1.3846 - categorical_accuracy: 0.5445
 1312/60000 [..............................] - ETA: 1:58 - loss: 1.3718 - categorical_accuracy: 0.5503
 1344/60000 [..............................] - ETA: 1:57 - loss: 1.3549 - categorical_accuracy: 0.5551
 1376/60000 [..............................] - ETA: 1:58 - loss: 1.3418 - categorical_accuracy: 0.5589
 1408/60000 [..............................] - ETA: 1:58 - loss: 1.3308 - categorical_accuracy: 0.5618
 1440/60000 [..............................] - ETA: 1:57 - loss: 1.3127 - categorical_accuracy: 0.5674
 1472/60000 [..............................] - ETA: 1:57 - loss: 1.3023 - categorical_accuracy: 0.5720
 1504/60000 [..............................] - ETA: 1:57 - loss: 1.2978 - categorical_accuracy: 0.5731
 1536/60000 [..............................] - ETA: 1:56 - loss: 1.2879 - categorical_accuracy: 0.5768
 1568/60000 [..............................] - ETA: 1:56 - loss: 1.2725 - categorical_accuracy: 0.5823
 1600/60000 [..............................] - ETA: 1:56 - loss: 1.2589 - categorical_accuracy: 0.5863
 1632/60000 [..............................] - ETA: 1:56 - loss: 1.2550 - categorical_accuracy: 0.5858
 1664/60000 [..............................] - ETA: 1:55 - loss: 1.2453 - categorical_accuracy: 0.5877
 1696/60000 [..............................] - ETA: 1:55 - loss: 1.2346 - categorical_accuracy: 0.5920
 1728/60000 [..............................] - ETA: 1:55 - loss: 1.2187 - categorical_accuracy: 0.5978
 1760/60000 [..............................] - ETA: 1:54 - loss: 1.2055 - categorical_accuracy: 0.6034
 1792/60000 [..............................] - ETA: 1:54 - loss: 1.1959 - categorical_accuracy: 0.6060
 1824/60000 [..............................] - ETA: 1:54 - loss: 1.1873 - categorical_accuracy: 0.6091
 1856/60000 [..............................] - ETA: 1:54 - loss: 1.1718 - categorical_accuracy: 0.6158
 1888/60000 [..............................] - ETA: 1:53 - loss: 1.1590 - categorical_accuracy: 0.6202
 1920/60000 [..............................] - ETA: 1:53 - loss: 1.1578 - categorical_accuracy: 0.6224
 1952/60000 [..............................] - ETA: 1:53 - loss: 1.1503 - categorical_accuracy: 0.6255
 1984/60000 [..............................] - ETA: 1:53 - loss: 1.1383 - categorical_accuracy: 0.6310
 2016/60000 [>.............................] - ETA: 1:52 - loss: 1.1318 - categorical_accuracy: 0.6329
 2048/60000 [>.............................] - ETA: 1:52 - loss: 1.1299 - categorical_accuracy: 0.6338
 2080/60000 [>.............................] - ETA: 1:52 - loss: 1.1203 - categorical_accuracy: 0.6370
 2112/60000 [>.............................] - ETA: 1:52 - loss: 1.1126 - categorical_accuracy: 0.6397
 2144/60000 [>.............................] - ETA: 1:51 - loss: 1.1044 - categorical_accuracy: 0.6427
 2176/60000 [>.............................] - ETA: 1:52 - loss: 1.0957 - categorical_accuracy: 0.6452
 2208/60000 [>.............................] - ETA: 1:51 - loss: 1.0849 - categorical_accuracy: 0.6490
 2240/60000 [>.............................] - ETA: 1:51 - loss: 1.0747 - categorical_accuracy: 0.6522
 2272/60000 [>.............................] - ETA: 1:51 - loss: 1.0699 - categorical_accuracy: 0.6536
 2304/60000 [>.............................] - ETA: 1:51 - loss: 1.0625 - categorical_accuracy: 0.6567
 2336/60000 [>.............................] - ETA: 1:51 - loss: 1.0539 - categorical_accuracy: 0.6592
 2368/60000 [>.............................] - ETA: 1:51 - loss: 1.0480 - categorical_accuracy: 0.6613
 2400/60000 [>.............................] - ETA: 1:50 - loss: 1.0391 - categorical_accuracy: 0.6646
 2432/60000 [>.............................] - ETA: 1:50 - loss: 1.0296 - categorical_accuracy: 0.6682
 2464/60000 [>.............................] - ETA: 1:50 - loss: 1.0271 - categorical_accuracy: 0.6696
 2496/60000 [>.............................] - ETA: 1:50 - loss: 1.0163 - categorical_accuracy: 0.6731
 2528/60000 [>.............................] - ETA: 1:50 - loss: 1.0119 - categorical_accuracy: 0.6748
 2560/60000 [>.............................] - ETA: 1:50 - loss: 1.0050 - categorical_accuracy: 0.6773
 2592/60000 [>.............................] - ETA: 1:50 - loss: 0.9958 - categorical_accuracy: 0.6802
 2624/60000 [>.............................] - ETA: 1:49 - loss: 0.9866 - categorical_accuracy: 0.6837
 2656/60000 [>.............................] - ETA: 1:49 - loss: 0.9775 - categorical_accuracy: 0.6864
 2688/60000 [>.............................] - ETA: 1:49 - loss: 0.9698 - categorical_accuracy: 0.6890
 2720/60000 [>.............................] - ETA: 1:49 - loss: 0.9639 - categorical_accuracy: 0.6904
 2752/60000 [>.............................] - ETA: 1:49 - loss: 0.9558 - categorical_accuracy: 0.6933
 2784/60000 [>.............................] - ETA: 1:49 - loss: 0.9470 - categorical_accuracy: 0.6961
 2816/60000 [>.............................] - ETA: 1:49 - loss: 0.9435 - categorical_accuracy: 0.6971
 2848/60000 [>.............................] - ETA: 1:49 - loss: 0.9381 - categorical_accuracy: 0.6987
 2880/60000 [>.............................] - ETA: 1:49 - loss: 0.9340 - categorical_accuracy: 0.6997
 2912/60000 [>.............................] - ETA: 1:49 - loss: 0.9276 - categorical_accuracy: 0.7016
 2944/60000 [>.............................] - ETA: 1:49 - loss: 0.9230 - categorical_accuracy: 0.7035
 2976/60000 [>.............................] - ETA: 1:48 - loss: 0.9178 - categorical_accuracy: 0.7046
 3008/60000 [>.............................] - ETA: 1:48 - loss: 0.9111 - categorical_accuracy: 0.7071
 3040/60000 [>.............................] - ETA: 1:48 - loss: 0.9060 - categorical_accuracy: 0.7089
 3072/60000 [>.............................] - ETA: 1:48 - loss: 0.9014 - categorical_accuracy: 0.7109
 3104/60000 [>.............................] - ETA: 1:48 - loss: 0.8976 - categorical_accuracy: 0.7123
 3136/60000 [>.............................] - ETA: 1:48 - loss: 0.8942 - categorical_accuracy: 0.7133
 3168/60000 [>.............................] - ETA: 1:48 - loss: 0.8912 - categorical_accuracy: 0.7146
 3200/60000 [>.............................] - ETA: 1:48 - loss: 0.8859 - categorical_accuracy: 0.7166
 3232/60000 [>.............................] - ETA: 1:48 - loss: 0.8799 - categorical_accuracy: 0.7181
 3264/60000 [>.............................] - ETA: 1:48 - loss: 0.8753 - categorical_accuracy: 0.7197
 3296/60000 [>.............................] - ETA: 1:48 - loss: 0.8703 - categorical_accuracy: 0.7218
 3328/60000 [>.............................] - ETA: 1:48 - loss: 0.8642 - categorical_accuracy: 0.7236
 3360/60000 [>.............................] - ETA: 1:47 - loss: 0.8593 - categorical_accuracy: 0.7259
 3392/60000 [>.............................] - ETA: 1:47 - loss: 0.8564 - categorical_accuracy: 0.7270
 3424/60000 [>.............................] - ETA: 1:47 - loss: 0.8523 - categorical_accuracy: 0.7284
 3456/60000 [>.............................] - ETA: 1:47 - loss: 0.8483 - categorical_accuracy: 0.7292
 3488/60000 [>.............................] - ETA: 1:47 - loss: 0.8439 - categorical_accuracy: 0.7308
 3520/60000 [>.............................] - ETA: 1:47 - loss: 0.8437 - categorical_accuracy: 0.7315
 3552/60000 [>.............................] - ETA: 1:47 - loss: 0.8408 - categorical_accuracy: 0.7323
 3584/60000 [>.............................] - ETA: 1:47 - loss: 0.8372 - categorical_accuracy: 0.7333
 3616/60000 [>.............................] - ETA: 1:47 - loss: 0.8336 - categorical_accuracy: 0.7345
 3648/60000 [>.............................] - ETA: 1:46 - loss: 0.8289 - categorical_accuracy: 0.7357
 3680/60000 [>.............................] - ETA: 1:46 - loss: 0.8241 - categorical_accuracy: 0.7370
 3712/60000 [>.............................] - ETA: 1:46 - loss: 0.8205 - categorical_accuracy: 0.7387
 3744/60000 [>.............................] - ETA: 1:46 - loss: 0.8167 - categorical_accuracy: 0.7399
 3776/60000 [>.............................] - ETA: 1:46 - loss: 0.8118 - categorical_accuracy: 0.7415
 3808/60000 [>.............................] - ETA: 1:46 - loss: 0.8075 - categorical_accuracy: 0.7426
 3840/60000 [>.............................] - ETA: 1:46 - loss: 0.8055 - categorical_accuracy: 0.7437
 3872/60000 [>.............................] - ETA: 1:46 - loss: 0.8015 - categorical_accuracy: 0.7448
 3904/60000 [>.............................] - ETA: 1:46 - loss: 0.7985 - categorical_accuracy: 0.7454
 3936/60000 [>.............................] - ETA: 1:45 - loss: 0.7959 - categorical_accuracy: 0.7462
 3968/60000 [>.............................] - ETA: 1:45 - loss: 0.7958 - categorical_accuracy: 0.7462
 4000/60000 [=>............................] - ETA: 1:45 - loss: 0.7935 - categorical_accuracy: 0.7467
 4032/60000 [=>............................] - ETA: 1:45 - loss: 0.7908 - categorical_accuracy: 0.7478
 4064/60000 [=>............................] - ETA: 1:45 - loss: 0.7869 - categorical_accuracy: 0.7490
 4096/60000 [=>............................] - ETA: 1:45 - loss: 0.7835 - categorical_accuracy: 0.7498
 4128/60000 [=>............................] - ETA: 1:45 - loss: 0.7789 - categorical_accuracy: 0.7515
 4160/60000 [=>............................] - ETA: 1:45 - loss: 0.7746 - categorical_accuracy: 0.7529
 4192/60000 [=>............................] - ETA: 1:45 - loss: 0.7715 - categorical_accuracy: 0.7543
 4224/60000 [=>............................] - ETA: 1:45 - loss: 0.7691 - categorical_accuracy: 0.7554
 4256/60000 [=>............................] - ETA: 1:45 - loss: 0.7678 - categorical_accuracy: 0.7556
 4288/60000 [=>............................] - ETA: 1:45 - loss: 0.7652 - categorical_accuracy: 0.7565
 4320/60000 [=>............................] - ETA: 1:45 - loss: 0.7627 - categorical_accuracy: 0.7572
 4352/60000 [=>............................] - ETA: 1:44 - loss: 0.7631 - categorical_accuracy: 0.7583
 4384/60000 [=>............................] - ETA: 1:44 - loss: 0.7598 - categorical_accuracy: 0.7596
 4416/60000 [=>............................] - ETA: 1:44 - loss: 0.7564 - categorical_accuracy: 0.7609
 4448/60000 [=>............................] - ETA: 1:44 - loss: 0.7527 - categorical_accuracy: 0.7619
 4480/60000 [=>............................] - ETA: 1:44 - loss: 0.7494 - categorical_accuracy: 0.7627
 4512/60000 [=>............................] - ETA: 1:44 - loss: 0.7465 - categorical_accuracy: 0.7640
 4544/60000 [=>............................] - ETA: 1:44 - loss: 0.7436 - categorical_accuracy: 0.7652
 4576/60000 [=>............................] - ETA: 1:44 - loss: 0.7421 - categorical_accuracy: 0.7660
 4608/60000 [=>............................] - ETA: 1:44 - loss: 0.7395 - categorical_accuracy: 0.7665
 4640/60000 [=>............................] - ETA: 1:43 - loss: 0.7371 - categorical_accuracy: 0.7672
 4672/60000 [=>............................] - ETA: 1:43 - loss: 0.7333 - categorical_accuracy: 0.7684
 4704/60000 [=>............................] - ETA: 1:43 - loss: 0.7294 - categorical_accuracy: 0.7693
 4736/60000 [=>............................] - ETA: 1:43 - loss: 0.7280 - categorical_accuracy: 0.7703
 4768/60000 [=>............................] - ETA: 1:43 - loss: 0.7238 - categorical_accuracy: 0.7718
 4800/60000 [=>............................] - ETA: 1:43 - loss: 0.7209 - categorical_accuracy: 0.7731
 4832/60000 [=>............................] - ETA: 1:43 - loss: 0.7182 - categorical_accuracy: 0.7738
 4864/60000 [=>............................] - ETA: 1:43 - loss: 0.7154 - categorical_accuracy: 0.7745
 4896/60000 [=>............................] - ETA: 1:43 - loss: 0.7127 - categorical_accuracy: 0.7753
 4928/60000 [=>............................] - ETA: 1:43 - loss: 0.7104 - categorical_accuracy: 0.7762
 4960/60000 [=>............................] - ETA: 1:43 - loss: 0.7076 - categorical_accuracy: 0.7770
 4992/60000 [=>............................] - ETA: 1:43 - loss: 0.7045 - categorical_accuracy: 0.7778
 5024/60000 [=>............................] - ETA: 1:43 - loss: 0.7017 - categorical_accuracy: 0.7789
 5056/60000 [=>............................] - ETA: 1:42 - loss: 0.6991 - categorical_accuracy: 0.7795
 5088/60000 [=>............................] - ETA: 1:42 - loss: 0.6957 - categorical_accuracy: 0.7805
 5120/60000 [=>............................] - ETA: 1:42 - loss: 0.6951 - categorical_accuracy: 0.7807
 5152/60000 [=>............................] - ETA: 1:42 - loss: 0.6954 - categorical_accuracy: 0.7803
 5184/60000 [=>............................] - ETA: 1:42 - loss: 0.6930 - categorical_accuracy: 0.7812
 5216/60000 [=>............................] - ETA: 1:42 - loss: 0.6919 - categorical_accuracy: 0.7816
 5248/60000 [=>............................] - ETA: 1:42 - loss: 0.6904 - categorical_accuracy: 0.7820
 5280/60000 [=>............................] - ETA: 1:42 - loss: 0.6872 - categorical_accuracy: 0.7831
 5312/60000 [=>............................] - ETA: 1:42 - loss: 0.6845 - categorical_accuracy: 0.7839
 5344/60000 [=>............................] - ETA: 1:42 - loss: 0.6817 - categorical_accuracy: 0.7848
 5376/60000 [=>............................] - ETA: 1:41 - loss: 0.6786 - categorical_accuracy: 0.7857
 5408/60000 [=>............................] - ETA: 1:41 - loss: 0.6769 - categorical_accuracy: 0.7864
 5440/60000 [=>............................] - ETA: 1:41 - loss: 0.6740 - categorical_accuracy: 0.7875
 5472/60000 [=>............................] - ETA: 1:41 - loss: 0.6720 - categorical_accuracy: 0.7884
 5504/60000 [=>............................] - ETA: 1:41 - loss: 0.6695 - categorical_accuracy: 0.7892
 5536/60000 [=>............................] - ETA: 1:41 - loss: 0.6665 - categorical_accuracy: 0.7903
 5568/60000 [=>............................] - ETA: 1:41 - loss: 0.6649 - categorical_accuracy: 0.7909
 5600/60000 [=>............................] - ETA: 1:41 - loss: 0.6620 - categorical_accuracy: 0.7920
 5632/60000 [=>............................] - ETA: 1:41 - loss: 0.6610 - categorical_accuracy: 0.7921
 5664/60000 [=>............................] - ETA: 1:41 - loss: 0.6592 - categorical_accuracy: 0.7924
 5696/60000 [=>............................] - ETA: 1:41 - loss: 0.6575 - categorical_accuracy: 0.7930
 5728/60000 [=>............................] - ETA: 1:41 - loss: 0.6550 - categorical_accuracy: 0.7940
 5760/60000 [=>............................] - ETA: 1:41 - loss: 0.6535 - categorical_accuracy: 0.7943
 5792/60000 [=>............................] - ETA: 1:40 - loss: 0.6515 - categorical_accuracy: 0.7945
 5824/60000 [=>............................] - ETA: 1:40 - loss: 0.6494 - categorical_accuracy: 0.7952
 5856/60000 [=>............................] - ETA: 1:40 - loss: 0.6476 - categorical_accuracy: 0.7959
 5888/60000 [=>............................] - ETA: 1:40 - loss: 0.6468 - categorical_accuracy: 0.7964
 5920/60000 [=>............................] - ETA: 1:40 - loss: 0.6458 - categorical_accuracy: 0.7968
 5952/60000 [=>............................] - ETA: 1:40 - loss: 0.6435 - categorical_accuracy: 0.7974
 5984/60000 [=>............................] - ETA: 1:40 - loss: 0.6435 - categorical_accuracy: 0.7976
 6016/60000 [==>...........................] - ETA: 1:40 - loss: 0.6419 - categorical_accuracy: 0.7984
 6048/60000 [==>...........................] - ETA: 1:40 - loss: 0.6409 - categorical_accuracy: 0.7988
 6080/60000 [==>...........................] - ETA: 1:40 - loss: 0.6409 - categorical_accuracy: 0.7993
 6112/60000 [==>...........................] - ETA: 1:40 - loss: 0.6395 - categorical_accuracy: 0.7999
 6144/60000 [==>...........................] - ETA: 1:40 - loss: 0.6378 - categorical_accuracy: 0.8003
 6176/60000 [==>...........................] - ETA: 1:40 - loss: 0.6362 - categorical_accuracy: 0.8008
 6208/60000 [==>...........................] - ETA: 1:40 - loss: 0.6346 - categorical_accuracy: 0.8012
 6240/60000 [==>...........................] - ETA: 1:40 - loss: 0.6321 - categorical_accuracy: 0.8019
 6272/60000 [==>...........................] - ETA: 1:40 - loss: 0.6296 - categorical_accuracy: 0.8029
 6304/60000 [==>...........................] - ETA: 1:40 - loss: 0.6279 - categorical_accuracy: 0.8035
 6336/60000 [==>...........................] - ETA: 1:39 - loss: 0.6258 - categorical_accuracy: 0.8038
 6368/60000 [==>...........................] - ETA: 1:39 - loss: 0.6241 - categorical_accuracy: 0.8043
 6400/60000 [==>...........................] - ETA: 1:39 - loss: 0.6219 - categorical_accuracy: 0.8047
 6432/60000 [==>...........................] - ETA: 1:39 - loss: 0.6195 - categorical_accuracy: 0.8055
 6464/60000 [==>...........................] - ETA: 1:39 - loss: 0.6185 - categorical_accuracy: 0.8058
 6496/60000 [==>...........................] - ETA: 1:39 - loss: 0.6168 - categorical_accuracy: 0.8063
 6528/60000 [==>...........................] - ETA: 1:39 - loss: 0.6151 - categorical_accuracy: 0.8068
 6560/60000 [==>...........................] - ETA: 1:39 - loss: 0.6135 - categorical_accuracy: 0.8072
 6592/60000 [==>...........................] - ETA: 1:39 - loss: 0.6121 - categorical_accuracy: 0.8075
 6624/60000 [==>...........................] - ETA: 1:39 - loss: 0.6113 - categorical_accuracy: 0.8080
 6656/60000 [==>...........................] - ETA: 1:39 - loss: 0.6104 - categorical_accuracy: 0.8086
 6688/60000 [==>...........................] - ETA: 1:39 - loss: 0.6093 - categorical_accuracy: 0.8088
 6720/60000 [==>...........................] - ETA: 1:39 - loss: 0.6079 - categorical_accuracy: 0.8092
 6752/60000 [==>...........................] - ETA: 1:39 - loss: 0.6055 - categorical_accuracy: 0.8098
 6784/60000 [==>...........................] - ETA: 1:39 - loss: 0.6032 - categorical_accuracy: 0.8106
 6816/60000 [==>...........................] - ETA: 1:39 - loss: 0.6012 - categorical_accuracy: 0.8113
 6848/60000 [==>...........................] - ETA: 1:39 - loss: 0.5997 - categorical_accuracy: 0.8118
 6880/60000 [==>...........................] - ETA: 1:39 - loss: 0.5978 - categorical_accuracy: 0.8124
 6912/60000 [==>...........................] - ETA: 1:38 - loss: 0.5962 - categorical_accuracy: 0.8126
 6944/60000 [==>...........................] - ETA: 1:39 - loss: 0.5947 - categorical_accuracy: 0.8131
 6976/60000 [==>...........................] - ETA: 1:38 - loss: 0.5938 - categorical_accuracy: 0.8132
 7008/60000 [==>...........................] - ETA: 1:38 - loss: 0.5914 - categorical_accuracy: 0.8141
 7040/60000 [==>...........................] - ETA: 1:38 - loss: 0.5902 - categorical_accuracy: 0.8145
 7072/60000 [==>...........................] - ETA: 1:38 - loss: 0.5888 - categorical_accuracy: 0.8149
 7104/60000 [==>...........................] - ETA: 1:38 - loss: 0.5871 - categorical_accuracy: 0.8152
 7136/60000 [==>...........................] - ETA: 1:38 - loss: 0.5869 - categorical_accuracy: 0.8154
 7168/60000 [==>...........................] - ETA: 1:38 - loss: 0.5863 - categorical_accuracy: 0.8158
 7200/60000 [==>...........................] - ETA: 1:38 - loss: 0.5854 - categorical_accuracy: 0.8160
 7232/60000 [==>...........................] - ETA: 1:38 - loss: 0.5837 - categorical_accuracy: 0.8165
 7264/60000 [==>...........................] - ETA: 1:38 - loss: 0.5816 - categorical_accuracy: 0.8172
 7296/60000 [==>...........................] - ETA: 1:38 - loss: 0.5809 - categorical_accuracy: 0.8172
 7328/60000 [==>...........................] - ETA: 1:38 - loss: 0.5793 - categorical_accuracy: 0.8177
 7360/60000 [==>...........................] - ETA: 1:38 - loss: 0.5781 - categorical_accuracy: 0.8178
 7392/60000 [==>...........................] - ETA: 1:38 - loss: 0.5766 - categorical_accuracy: 0.8183
 7424/60000 [==>...........................] - ETA: 1:37 - loss: 0.5755 - categorical_accuracy: 0.8187
 7456/60000 [==>...........................] - ETA: 1:37 - loss: 0.5739 - categorical_accuracy: 0.8191
 7488/60000 [==>...........................] - ETA: 1:37 - loss: 0.5726 - categorical_accuracy: 0.8196
 7520/60000 [==>...........................] - ETA: 1:37 - loss: 0.5704 - categorical_accuracy: 0.8203
 7552/60000 [==>...........................] - ETA: 1:37 - loss: 0.5689 - categorical_accuracy: 0.8208
 7584/60000 [==>...........................] - ETA: 1:37 - loss: 0.5679 - categorical_accuracy: 0.8215
 7616/60000 [==>...........................] - ETA: 1:37 - loss: 0.5683 - categorical_accuracy: 0.8216
 7648/60000 [==>...........................] - ETA: 1:37 - loss: 0.5666 - categorical_accuracy: 0.8222
 7680/60000 [==>...........................] - ETA: 1:37 - loss: 0.5650 - categorical_accuracy: 0.8228
 7712/60000 [==>...........................] - ETA: 1:37 - loss: 0.5642 - categorical_accuracy: 0.8230
 7744/60000 [==>...........................] - ETA: 1:37 - loss: 0.5627 - categorical_accuracy: 0.8235
 7776/60000 [==>...........................] - ETA: 1:37 - loss: 0.5614 - categorical_accuracy: 0.8239
 7808/60000 [==>...........................] - ETA: 1:37 - loss: 0.5606 - categorical_accuracy: 0.8240
 7840/60000 [==>...........................] - ETA: 1:37 - loss: 0.5591 - categorical_accuracy: 0.8245
 7872/60000 [==>...........................] - ETA: 1:36 - loss: 0.5585 - categorical_accuracy: 0.8248
 7904/60000 [==>...........................] - ETA: 1:36 - loss: 0.5576 - categorical_accuracy: 0.8252
 7936/60000 [==>...........................] - ETA: 1:36 - loss: 0.5558 - categorical_accuracy: 0.8257
 7968/60000 [==>...........................] - ETA: 1:36 - loss: 0.5541 - categorical_accuracy: 0.8264
 8000/60000 [===>..........................] - ETA: 1:36 - loss: 0.5522 - categorical_accuracy: 0.8271
 8032/60000 [===>..........................] - ETA: 1:36 - loss: 0.5503 - categorical_accuracy: 0.8277
 8064/60000 [===>..........................] - ETA: 1:36 - loss: 0.5488 - categorical_accuracy: 0.8281
 8096/60000 [===>..........................] - ETA: 1:36 - loss: 0.5483 - categorical_accuracy: 0.8284
 8128/60000 [===>..........................] - ETA: 1:36 - loss: 0.5468 - categorical_accuracy: 0.8287
 8160/60000 [===>..........................] - ETA: 1:36 - loss: 0.5450 - categorical_accuracy: 0.8293
 8192/60000 [===>..........................] - ETA: 1:36 - loss: 0.5442 - categorical_accuracy: 0.8296
 8224/60000 [===>..........................] - ETA: 1:36 - loss: 0.5424 - categorical_accuracy: 0.8301
 8256/60000 [===>..........................] - ETA: 1:36 - loss: 0.5407 - categorical_accuracy: 0.8307
 8288/60000 [===>..........................] - ETA: 1:36 - loss: 0.5393 - categorical_accuracy: 0.8311
 8320/60000 [===>..........................] - ETA: 1:36 - loss: 0.5379 - categorical_accuracy: 0.8316
 8352/60000 [===>..........................] - ETA: 1:35 - loss: 0.5365 - categorical_accuracy: 0.8318
 8384/60000 [===>..........................] - ETA: 1:35 - loss: 0.5350 - categorical_accuracy: 0.8323
 8416/60000 [===>..........................] - ETA: 1:35 - loss: 0.5344 - categorical_accuracy: 0.8327
 8448/60000 [===>..........................] - ETA: 1:35 - loss: 0.5338 - categorical_accuracy: 0.8331
 8480/60000 [===>..........................] - ETA: 1:35 - loss: 0.5326 - categorical_accuracy: 0.8335
 8512/60000 [===>..........................] - ETA: 1:35 - loss: 0.5314 - categorical_accuracy: 0.8339
 8544/60000 [===>..........................] - ETA: 1:35 - loss: 0.5299 - categorical_accuracy: 0.8343
 8576/60000 [===>..........................] - ETA: 1:35 - loss: 0.5288 - categorical_accuracy: 0.8347
 8608/60000 [===>..........................] - ETA: 1:35 - loss: 0.5290 - categorical_accuracy: 0.8349
 8640/60000 [===>..........................] - ETA: 1:35 - loss: 0.5282 - categorical_accuracy: 0.8353
 8672/60000 [===>..........................] - ETA: 1:35 - loss: 0.5274 - categorical_accuracy: 0.8357
 8704/60000 [===>..........................] - ETA: 1:35 - loss: 0.5257 - categorical_accuracy: 0.8363
 8736/60000 [===>..........................] - ETA: 1:35 - loss: 0.5244 - categorical_accuracy: 0.8368
 8768/60000 [===>..........................] - ETA: 1:35 - loss: 0.5233 - categorical_accuracy: 0.8372
 8800/60000 [===>..........................] - ETA: 1:35 - loss: 0.5218 - categorical_accuracy: 0.8377
 8832/60000 [===>..........................] - ETA: 1:35 - loss: 0.5203 - categorical_accuracy: 0.8383
 8864/60000 [===>..........................] - ETA: 1:34 - loss: 0.5188 - categorical_accuracy: 0.8389
 8896/60000 [===>..........................] - ETA: 1:34 - loss: 0.5173 - categorical_accuracy: 0.8394
 8928/60000 [===>..........................] - ETA: 1:34 - loss: 0.5160 - categorical_accuracy: 0.8398
 8960/60000 [===>..........................] - ETA: 1:34 - loss: 0.5145 - categorical_accuracy: 0.8402
 8992/60000 [===>..........................] - ETA: 1:34 - loss: 0.5131 - categorical_accuracy: 0.8406
 9024/60000 [===>..........................] - ETA: 1:34 - loss: 0.5123 - categorical_accuracy: 0.8410
 9056/60000 [===>..........................] - ETA: 1:34 - loss: 0.5120 - categorical_accuracy: 0.8412
 9088/60000 [===>..........................] - ETA: 1:34 - loss: 0.5108 - categorical_accuracy: 0.8417
 9120/60000 [===>..........................] - ETA: 1:34 - loss: 0.5092 - categorical_accuracy: 0.8422
 9152/60000 [===>..........................] - ETA: 1:34 - loss: 0.5077 - categorical_accuracy: 0.8427
 9184/60000 [===>..........................] - ETA: 1:34 - loss: 0.5070 - categorical_accuracy: 0.8430
 9216/60000 [===>..........................] - ETA: 1:34 - loss: 0.5059 - categorical_accuracy: 0.8433
 9248/60000 [===>..........................] - ETA: 1:34 - loss: 0.5049 - categorical_accuracy: 0.8438
 9280/60000 [===>..........................] - ETA: 1:34 - loss: 0.5042 - categorical_accuracy: 0.8440
 9312/60000 [===>..........................] - ETA: 1:34 - loss: 0.5034 - categorical_accuracy: 0.8442
 9344/60000 [===>..........................] - ETA: 1:33 - loss: 0.5033 - categorical_accuracy: 0.8442
 9376/60000 [===>..........................] - ETA: 1:33 - loss: 0.5021 - categorical_accuracy: 0.8444
 9408/60000 [===>..........................] - ETA: 1:33 - loss: 0.5014 - categorical_accuracy: 0.8447
 9440/60000 [===>..........................] - ETA: 1:33 - loss: 0.5000 - categorical_accuracy: 0.8452
 9472/60000 [===>..........................] - ETA: 1:33 - loss: 0.4984 - categorical_accuracy: 0.8458
 9504/60000 [===>..........................] - ETA: 1:33 - loss: 0.4975 - categorical_accuracy: 0.8462
 9536/60000 [===>..........................] - ETA: 1:33 - loss: 0.4968 - categorical_accuracy: 0.8464
 9568/60000 [===>..........................] - ETA: 1:33 - loss: 0.4955 - categorical_accuracy: 0.8468
 9600/60000 [===>..........................] - ETA: 1:33 - loss: 0.4945 - categorical_accuracy: 0.8472
 9632/60000 [===>..........................] - ETA: 1:33 - loss: 0.4931 - categorical_accuracy: 0.8476
 9664/60000 [===>..........................] - ETA: 1:33 - loss: 0.4936 - categorical_accuracy: 0.8474
 9696/60000 [===>..........................] - ETA: 1:33 - loss: 0.4924 - categorical_accuracy: 0.8479
 9728/60000 [===>..........................] - ETA: 1:33 - loss: 0.4927 - categorical_accuracy: 0.8479
 9760/60000 [===>..........................] - ETA: 1:33 - loss: 0.4916 - categorical_accuracy: 0.8483
 9792/60000 [===>..........................] - ETA: 1:33 - loss: 0.4905 - categorical_accuracy: 0.8484
 9824/60000 [===>..........................] - ETA: 1:33 - loss: 0.4891 - categorical_accuracy: 0.8489
 9856/60000 [===>..........................] - ETA: 1:32 - loss: 0.4879 - categorical_accuracy: 0.8493
 9888/60000 [===>..........................] - ETA: 1:32 - loss: 0.4868 - categorical_accuracy: 0.8497
 9920/60000 [===>..........................] - ETA: 1:32 - loss: 0.4865 - categorical_accuracy: 0.8500
 9952/60000 [===>..........................] - ETA: 1:32 - loss: 0.4852 - categorical_accuracy: 0.8505
 9984/60000 [===>..........................] - ETA: 1:32 - loss: 0.4848 - categorical_accuracy: 0.8506
10016/60000 [====>.........................] - ETA: 1:32 - loss: 0.4835 - categorical_accuracy: 0.8509
10048/60000 [====>.........................] - ETA: 1:32 - loss: 0.4827 - categorical_accuracy: 0.8512
10080/60000 [====>.........................] - ETA: 1:32 - loss: 0.4813 - categorical_accuracy: 0.8517
10112/60000 [====>.........................] - ETA: 1:32 - loss: 0.4807 - categorical_accuracy: 0.8518
10144/60000 [====>.........................] - ETA: 1:32 - loss: 0.4792 - categorical_accuracy: 0.8522
10176/60000 [====>.........................] - ETA: 1:32 - loss: 0.4780 - categorical_accuracy: 0.8527
10208/60000 [====>.........................] - ETA: 1:32 - loss: 0.4775 - categorical_accuracy: 0.8527
10240/60000 [====>.........................] - ETA: 1:32 - loss: 0.4765 - categorical_accuracy: 0.8529
10272/60000 [====>.........................] - ETA: 1:32 - loss: 0.4753 - categorical_accuracy: 0.8532
10304/60000 [====>.........................] - ETA: 1:32 - loss: 0.4747 - categorical_accuracy: 0.8534
10336/60000 [====>.........................] - ETA: 1:32 - loss: 0.4736 - categorical_accuracy: 0.8538
10368/60000 [====>.........................] - ETA: 1:31 - loss: 0.4725 - categorical_accuracy: 0.8541
10400/60000 [====>.........................] - ETA: 1:31 - loss: 0.4719 - categorical_accuracy: 0.8542
10432/60000 [====>.........................] - ETA: 1:31 - loss: 0.4706 - categorical_accuracy: 0.8547
10464/60000 [====>.........................] - ETA: 1:31 - loss: 0.4701 - categorical_accuracy: 0.8549
10496/60000 [====>.........................] - ETA: 1:31 - loss: 0.4697 - categorical_accuracy: 0.8551
10528/60000 [====>.........................] - ETA: 1:31 - loss: 0.4698 - categorical_accuracy: 0.8551
10560/60000 [====>.........................] - ETA: 1:31 - loss: 0.4697 - categorical_accuracy: 0.8552
10592/60000 [====>.........................] - ETA: 1:31 - loss: 0.4694 - categorical_accuracy: 0.8554
10624/60000 [====>.........................] - ETA: 1:31 - loss: 0.4685 - categorical_accuracy: 0.8556
10656/60000 [====>.........................] - ETA: 1:31 - loss: 0.4674 - categorical_accuracy: 0.8560
10688/60000 [====>.........................] - ETA: 1:31 - loss: 0.4667 - categorical_accuracy: 0.8564
10720/60000 [====>.........................] - ETA: 1:31 - loss: 0.4654 - categorical_accuracy: 0.8568
10752/60000 [====>.........................] - ETA: 1:31 - loss: 0.4644 - categorical_accuracy: 0.8570
10784/60000 [====>.........................] - ETA: 1:31 - loss: 0.4638 - categorical_accuracy: 0.8573
10816/60000 [====>.........................] - ETA: 1:31 - loss: 0.4635 - categorical_accuracy: 0.8573
10848/60000 [====>.........................] - ETA: 1:31 - loss: 0.4624 - categorical_accuracy: 0.8578
10880/60000 [====>.........................] - ETA: 1:30 - loss: 0.4624 - categorical_accuracy: 0.8577
10912/60000 [====>.........................] - ETA: 1:30 - loss: 0.4618 - categorical_accuracy: 0.8580
10944/60000 [====>.........................] - ETA: 1:30 - loss: 0.4611 - categorical_accuracy: 0.8581
10976/60000 [====>.........................] - ETA: 1:30 - loss: 0.4603 - categorical_accuracy: 0.8583
11008/60000 [====>.........................] - ETA: 1:30 - loss: 0.4599 - categorical_accuracy: 0.8586
11040/60000 [====>.........................] - ETA: 1:30 - loss: 0.4595 - categorical_accuracy: 0.8589
11072/60000 [====>.........................] - ETA: 1:30 - loss: 0.4586 - categorical_accuracy: 0.8591
11104/60000 [====>.........................] - ETA: 1:30 - loss: 0.4580 - categorical_accuracy: 0.8593
11136/60000 [====>.........................] - ETA: 1:30 - loss: 0.4576 - categorical_accuracy: 0.8595
11168/60000 [====>.........................] - ETA: 1:30 - loss: 0.4569 - categorical_accuracy: 0.8597
11200/60000 [====>.........................] - ETA: 1:30 - loss: 0.4561 - categorical_accuracy: 0.8599
11232/60000 [====>.........................] - ETA: 1:30 - loss: 0.4555 - categorical_accuracy: 0.8601
11264/60000 [====>.........................] - ETA: 1:30 - loss: 0.4561 - categorical_accuracy: 0.8601
11296/60000 [====>.........................] - ETA: 1:30 - loss: 0.4552 - categorical_accuracy: 0.8603
11328/60000 [====>.........................] - ETA: 1:30 - loss: 0.4544 - categorical_accuracy: 0.8605
11360/60000 [====>.........................] - ETA: 1:30 - loss: 0.4536 - categorical_accuracy: 0.8607
11392/60000 [====>.........................] - ETA: 1:30 - loss: 0.4530 - categorical_accuracy: 0.8608
11424/60000 [====>.........................] - ETA: 1:30 - loss: 0.4521 - categorical_accuracy: 0.8610
11456/60000 [====>.........................] - ETA: 1:30 - loss: 0.4511 - categorical_accuracy: 0.8613
11488/60000 [====>.........................] - ETA: 1:30 - loss: 0.4510 - categorical_accuracy: 0.8612
11520/60000 [====>.........................] - ETA: 1:29 - loss: 0.4504 - categorical_accuracy: 0.8615
11552/60000 [====>.........................] - ETA: 1:29 - loss: 0.4498 - categorical_accuracy: 0.8616
11584/60000 [====>.........................] - ETA: 1:29 - loss: 0.4487 - categorical_accuracy: 0.8620
11616/60000 [====>.........................] - ETA: 1:29 - loss: 0.4478 - categorical_accuracy: 0.8623
11648/60000 [====>.........................] - ETA: 1:29 - loss: 0.4475 - categorical_accuracy: 0.8624
11680/60000 [====>.........................] - ETA: 1:29 - loss: 0.4464 - categorical_accuracy: 0.8628
11712/60000 [====>.........................] - ETA: 1:29 - loss: 0.4454 - categorical_accuracy: 0.8631
11744/60000 [====>.........................] - ETA: 1:29 - loss: 0.4444 - categorical_accuracy: 0.8634
11776/60000 [====>.........................] - ETA: 1:29 - loss: 0.4435 - categorical_accuracy: 0.8638
11808/60000 [====>.........................] - ETA: 1:29 - loss: 0.4430 - categorical_accuracy: 0.8641
11840/60000 [====>.........................] - ETA: 1:29 - loss: 0.4425 - categorical_accuracy: 0.8642
11872/60000 [====>.........................] - ETA: 1:29 - loss: 0.4417 - categorical_accuracy: 0.8645
11904/60000 [====>.........................] - ETA: 1:29 - loss: 0.4411 - categorical_accuracy: 0.8648
11936/60000 [====>.........................] - ETA: 1:29 - loss: 0.4401 - categorical_accuracy: 0.8651
11968/60000 [====>.........................] - ETA: 1:29 - loss: 0.4392 - categorical_accuracy: 0.8654
12000/60000 [=====>........................] - ETA: 1:29 - loss: 0.4382 - categorical_accuracy: 0.8658
12032/60000 [=====>........................] - ETA: 1:28 - loss: 0.4379 - categorical_accuracy: 0.8658
12064/60000 [=====>........................] - ETA: 1:28 - loss: 0.4371 - categorical_accuracy: 0.8660
12096/60000 [=====>........................] - ETA: 1:28 - loss: 0.4368 - categorical_accuracy: 0.8662
12128/60000 [=====>........................] - ETA: 1:28 - loss: 0.4366 - categorical_accuracy: 0.8662
12160/60000 [=====>........................] - ETA: 1:28 - loss: 0.4357 - categorical_accuracy: 0.8664
12192/60000 [=====>........................] - ETA: 1:28 - loss: 0.4354 - categorical_accuracy: 0.8666
12224/60000 [=====>........................] - ETA: 1:28 - loss: 0.4346 - categorical_accuracy: 0.8668
12256/60000 [=====>........................] - ETA: 1:28 - loss: 0.4339 - categorical_accuracy: 0.8671
12288/60000 [=====>........................] - ETA: 1:28 - loss: 0.4338 - categorical_accuracy: 0.8672
12320/60000 [=====>........................] - ETA: 1:28 - loss: 0.4328 - categorical_accuracy: 0.8675
12352/60000 [=====>........................] - ETA: 1:28 - loss: 0.4319 - categorical_accuracy: 0.8677
12384/60000 [=====>........................] - ETA: 1:28 - loss: 0.4312 - categorical_accuracy: 0.8679
12416/60000 [=====>........................] - ETA: 1:28 - loss: 0.4303 - categorical_accuracy: 0.8682
12448/60000 [=====>........................] - ETA: 1:28 - loss: 0.4296 - categorical_accuracy: 0.8684
12480/60000 [=====>........................] - ETA: 1:28 - loss: 0.4300 - categorical_accuracy: 0.8686
12512/60000 [=====>........................] - ETA: 1:28 - loss: 0.4296 - categorical_accuracy: 0.8687
12544/60000 [=====>........................] - ETA: 1:28 - loss: 0.4287 - categorical_accuracy: 0.8689
12576/60000 [=====>........................] - ETA: 1:28 - loss: 0.4285 - categorical_accuracy: 0.8691
12608/60000 [=====>........................] - ETA: 1:28 - loss: 0.4281 - categorical_accuracy: 0.8691
12640/60000 [=====>........................] - ETA: 1:27 - loss: 0.4275 - categorical_accuracy: 0.8693
12672/60000 [=====>........................] - ETA: 1:27 - loss: 0.4272 - categorical_accuracy: 0.8694
12704/60000 [=====>........................] - ETA: 1:27 - loss: 0.4267 - categorical_accuracy: 0.8696
12736/60000 [=====>........................] - ETA: 1:27 - loss: 0.4260 - categorical_accuracy: 0.8697
12768/60000 [=====>........................] - ETA: 1:27 - loss: 0.4256 - categorical_accuracy: 0.8698
12800/60000 [=====>........................] - ETA: 1:27 - loss: 0.4248 - categorical_accuracy: 0.8702
12832/60000 [=====>........................] - ETA: 1:27 - loss: 0.4241 - categorical_accuracy: 0.8704
12864/60000 [=====>........................] - ETA: 1:27 - loss: 0.4232 - categorical_accuracy: 0.8707
12896/60000 [=====>........................] - ETA: 1:27 - loss: 0.4230 - categorical_accuracy: 0.8707
12928/60000 [=====>........................] - ETA: 1:27 - loss: 0.4221 - categorical_accuracy: 0.8711
12960/60000 [=====>........................] - ETA: 1:27 - loss: 0.4212 - categorical_accuracy: 0.8713
12992/60000 [=====>........................] - ETA: 1:27 - loss: 0.4210 - categorical_accuracy: 0.8715
13024/60000 [=====>........................] - ETA: 1:27 - loss: 0.4203 - categorical_accuracy: 0.8716
13056/60000 [=====>........................] - ETA: 1:27 - loss: 0.4198 - categorical_accuracy: 0.8717
13088/60000 [=====>........................] - ETA: 1:27 - loss: 0.4190 - categorical_accuracy: 0.8719
13120/60000 [=====>........................] - ETA: 1:26 - loss: 0.4181 - categorical_accuracy: 0.8722
13152/60000 [=====>........................] - ETA: 1:26 - loss: 0.4172 - categorical_accuracy: 0.8724
13184/60000 [=====>........................] - ETA: 1:26 - loss: 0.4172 - categorical_accuracy: 0.8724
13216/60000 [=====>........................] - ETA: 1:26 - loss: 0.4168 - categorical_accuracy: 0.8725
13248/60000 [=====>........................] - ETA: 1:26 - loss: 0.4166 - categorical_accuracy: 0.8725
13280/60000 [=====>........................] - ETA: 1:26 - loss: 0.4157 - categorical_accuracy: 0.8727
13312/60000 [=====>........................] - ETA: 1:26 - loss: 0.4149 - categorical_accuracy: 0.8730
13344/60000 [=====>........................] - ETA: 1:26 - loss: 0.4141 - categorical_accuracy: 0.8733
13376/60000 [=====>........................] - ETA: 1:26 - loss: 0.4135 - categorical_accuracy: 0.8734
13408/60000 [=====>........................] - ETA: 1:26 - loss: 0.4126 - categorical_accuracy: 0.8737
13440/60000 [=====>........................] - ETA: 1:26 - loss: 0.4123 - categorical_accuracy: 0.8738
13472/60000 [=====>........................] - ETA: 1:26 - loss: 0.4116 - categorical_accuracy: 0.8740
13504/60000 [=====>........................] - ETA: 1:26 - loss: 0.4107 - categorical_accuracy: 0.8743
13536/60000 [=====>........................] - ETA: 1:26 - loss: 0.4101 - categorical_accuracy: 0.8744
13568/60000 [=====>........................] - ETA: 1:26 - loss: 0.4096 - categorical_accuracy: 0.8746
13600/60000 [=====>........................] - ETA: 1:26 - loss: 0.4092 - categorical_accuracy: 0.8748
13632/60000 [=====>........................] - ETA: 1:26 - loss: 0.4083 - categorical_accuracy: 0.8750
13664/60000 [=====>........................] - ETA: 1:26 - loss: 0.4075 - categorical_accuracy: 0.8752
13696/60000 [=====>........................] - ETA: 1:25 - loss: 0.4069 - categorical_accuracy: 0.8754
13728/60000 [=====>........................] - ETA: 1:25 - loss: 0.4065 - categorical_accuracy: 0.8755
13760/60000 [=====>........................] - ETA: 1:25 - loss: 0.4057 - categorical_accuracy: 0.8757
13792/60000 [=====>........................] - ETA: 1:25 - loss: 0.4056 - categorical_accuracy: 0.8758
13824/60000 [=====>........................] - ETA: 1:25 - loss: 0.4049 - categorical_accuracy: 0.8761
13856/60000 [=====>........................] - ETA: 1:25 - loss: 0.4045 - categorical_accuracy: 0.8762
13888/60000 [=====>........................] - ETA: 1:25 - loss: 0.4050 - categorical_accuracy: 0.8761
13920/60000 [=====>........................] - ETA: 1:25 - loss: 0.4043 - categorical_accuracy: 0.8763
13952/60000 [=====>........................] - ETA: 1:25 - loss: 0.4036 - categorical_accuracy: 0.8764
13984/60000 [=====>........................] - ETA: 1:25 - loss: 0.4030 - categorical_accuracy: 0.8766
14016/60000 [======>.......................] - ETA: 1:25 - loss: 0.4024 - categorical_accuracy: 0.8768
14048/60000 [======>.......................] - ETA: 1:25 - loss: 0.4016 - categorical_accuracy: 0.8771
14080/60000 [======>.......................] - ETA: 1:25 - loss: 0.4012 - categorical_accuracy: 0.8773
14112/60000 [======>.......................] - ETA: 1:25 - loss: 0.4004 - categorical_accuracy: 0.8776
14144/60000 [======>.......................] - ETA: 1:25 - loss: 0.4000 - categorical_accuracy: 0.8775
14176/60000 [======>.......................] - ETA: 1:25 - loss: 0.3993 - categorical_accuracy: 0.8778
14208/60000 [======>.......................] - ETA: 1:24 - loss: 0.3987 - categorical_accuracy: 0.8780
14240/60000 [======>.......................] - ETA: 1:24 - loss: 0.3979 - categorical_accuracy: 0.8782
14272/60000 [======>.......................] - ETA: 1:24 - loss: 0.3974 - categorical_accuracy: 0.8784
14304/60000 [======>.......................] - ETA: 1:24 - loss: 0.3969 - categorical_accuracy: 0.8786
14336/60000 [======>.......................] - ETA: 1:24 - loss: 0.3966 - categorical_accuracy: 0.8786
14368/60000 [======>.......................] - ETA: 1:24 - loss: 0.3964 - categorical_accuracy: 0.8787
14400/60000 [======>.......................] - ETA: 1:24 - loss: 0.3960 - categorical_accuracy: 0.8788
14432/60000 [======>.......................] - ETA: 1:24 - loss: 0.3952 - categorical_accuracy: 0.8790
14464/60000 [======>.......................] - ETA: 1:24 - loss: 0.3946 - categorical_accuracy: 0.8791
14496/60000 [======>.......................] - ETA: 1:24 - loss: 0.3942 - categorical_accuracy: 0.8792
14528/60000 [======>.......................] - ETA: 1:24 - loss: 0.3936 - categorical_accuracy: 0.8793
14560/60000 [======>.......................] - ETA: 1:24 - loss: 0.3928 - categorical_accuracy: 0.8795
14592/60000 [======>.......................] - ETA: 1:24 - loss: 0.3923 - categorical_accuracy: 0.8796
14624/60000 [======>.......................] - ETA: 1:24 - loss: 0.3916 - categorical_accuracy: 0.8798
14656/60000 [======>.......................] - ETA: 1:24 - loss: 0.3908 - categorical_accuracy: 0.8800
14688/60000 [======>.......................] - ETA: 1:24 - loss: 0.3906 - categorical_accuracy: 0.8802
14720/60000 [======>.......................] - ETA: 1:23 - loss: 0.3901 - categorical_accuracy: 0.8804
14752/60000 [======>.......................] - ETA: 1:23 - loss: 0.3896 - categorical_accuracy: 0.8805
14784/60000 [======>.......................] - ETA: 1:23 - loss: 0.3898 - categorical_accuracy: 0.8805
14816/60000 [======>.......................] - ETA: 1:23 - loss: 0.3901 - categorical_accuracy: 0.8805
14848/60000 [======>.......................] - ETA: 1:23 - loss: 0.3894 - categorical_accuracy: 0.8807
14880/60000 [======>.......................] - ETA: 1:23 - loss: 0.3892 - categorical_accuracy: 0.8808
14912/60000 [======>.......................] - ETA: 1:23 - loss: 0.3885 - categorical_accuracy: 0.8810
14944/60000 [======>.......................] - ETA: 1:23 - loss: 0.3882 - categorical_accuracy: 0.8811
14976/60000 [======>.......................] - ETA: 1:23 - loss: 0.3880 - categorical_accuracy: 0.8811
15008/60000 [======>.......................] - ETA: 1:23 - loss: 0.3882 - categorical_accuracy: 0.8811
15040/60000 [======>.......................] - ETA: 1:23 - loss: 0.3885 - categorical_accuracy: 0.8811
15072/60000 [======>.......................] - ETA: 1:23 - loss: 0.3881 - categorical_accuracy: 0.8811
15104/60000 [======>.......................] - ETA: 1:23 - loss: 0.3875 - categorical_accuracy: 0.8813
15136/60000 [======>.......................] - ETA: 1:23 - loss: 0.3869 - categorical_accuracy: 0.8815
15168/60000 [======>.......................] - ETA: 1:23 - loss: 0.3863 - categorical_accuracy: 0.8817
15200/60000 [======>.......................] - ETA: 1:23 - loss: 0.3863 - categorical_accuracy: 0.8818
15232/60000 [======>.......................] - ETA: 1:22 - loss: 0.3855 - categorical_accuracy: 0.8820
15264/60000 [======>.......................] - ETA: 1:22 - loss: 0.3854 - categorical_accuracy: 0.8821
15296/60000 [======>.......................] - ETA: 1:22 - loss: 0.3852 - categorical_accuracy: 0.8822
15328/60000 [======>.......................] - ETA: 1:22 - loss: 0.3847 - categorical_accuracy: 0.8824
15360/60000 [======>.......................] - ETA: 1:22 - loss: 0.3841 - categorical_accuracy: 0.8826
15392/60000 [======>.......................] - ETA: 1:22 - loss: 0.3840 - categorical_accuracy: 0.8827
15424/60000 [======>.......................] - ETA: 1:22 - loss: 0.3837 - categorical_accuracy: 0.8827
15456/60000 [======>.......................] - ETA: 1:22 - loss: 0.3832 - categorical_accuracy: 0.8828
15488/60000 [======>.......................] - ETA: 1:22 - loss: 0.3831 - categorical_accuracy: 0.8829
15520/60000 [======>.......................] - ETA: 1:22 - loss: 0.3824 - categorical_accuracy: 0.8831
15552/60000 [======>.......................] - ETA: 1:22 - loss: 0.3817 - categorical_accuracy: 0.8834
15584/60000 [======>.......................] - ETA: 1:22 - loss: 0.3811 - categorical_accuracy: 0.8836
15616/60000 [======>.......................] - ETA: 1:22 - loss: 0.3809 - categorical_accuracy: 0.8835
15648/60000 [======>.......................] - ETA: 1:22 - loss: 0.3803 - categorical_accuracy: 0.8838
15680/60000 [======>.......................] - ETA: 1:22 - loss: 0.3797 - categorical_accuracy: 0.8839
15712/60000 [======>.......................] - ETA: 1:22 - loss: 0.3790 - categorical_accuracy: 0.8841
15744/60000 [======>.......................] - ETA: 1:22 - loss: 0.3784 - categorical_accuracy: 0.8843
15776/60000 [======>.......................] - ETA: 1:22 - loss: 0.3777 - categorical_accuracy: 0.8845
15808/60000 [======>.......................] - ETA: 1:21 - loss: 0.3780 - categorical_accuracy: 0.8846
15840/60000 [======>.......................] - ETA: 1:21 - loss: 0.3775 - categorical_accuracy: 0.8847
15872/60000 [======>.......................] - ETA: 1:21 - loss: 0.3769 - categorical_accuracy: 0.8849
15904/60000 [======>.......................] - ETA: 1:21 - loss: 0.3767 - categorical_accuracy: 0.8850
15936/60000 [======>.......................] - ETA: 1:21 - loss: 0.3764 - categorical_accuracy: 0.8851
15968/60000 [======>.......................] - ETA: 1:21 - loss: 0.3759 - categorical_accuracy: 0.8853
16000/60000 [=======>......................] - ETA: 1:21 - loss: 0.3753 - categorical_accuracy: 0.8855
16032/60000 [=======>......................] - ETA: 1:21 - loss: 0.3748 - categorical_accuracy: 0.8856
16064/60000 [=======>......................] - ETA: 1:21 - loss: 0.3745 - categorical_accuracy: 0.8856
16096/60000 [=======>......................] - ETA: 1:21 - loss: 0.3739 - categorical_accuracy: 0.8858
16128/60000 [=======>......................] - ETA: 1:21 - loss: 0.3732 - categorical_accuracy: 0.8860
16160/60000 [=======>......................] - ETA: 1:21 - loss: 0.3730 - categorical_accuracy: 0.8861
16192/60000 [=======>......................] - ETA: 1:21 - loss: 0.3723 - categorical_accuracy: 0.8863
16224/60000 [=======>......................] - ETA: 1:21 - loss: 0.3716 - categorical_accuracy: 0.8865
16256/60000 [=======>......................] - ETA: 1:21 - loss: 0.3714 - categorical_accuracy: 0.8865
16288/60000 [=======>......................] - ETA: 1:21 - loss: 0.3710 - categorical_accuracy: 0.8866
16320/60000 [=======>......................] - ETA: 1:21 - loss: 0.3705 - categorical_accuracy: 0.8866
16352/60000 [=======>......................] - ETA: 1:21 - loss: 0.3701 - categorical_accuracy: 0.8867
16384/60000 [=======>......................] - ETA: 1:21 - loss: 0.3698 - categorical_accuracy: 0.8868
16416/60000 [=======>......................] - ETA: 1:21 - loss: 0.3692 - categorical_accuracy: 0.8870
16448/60000 [=======>......................] - ETA: 1:20 - loss: 0.3691 - categorical_accuracy: 0.8869
16480/60000 [=======>......................] - ETA: 1:20 - loss: 0.3686 - categorical_accuracy: 0.8870
16512/60000 [=======>......................] - ETA: 1:20 - loss: 0.3682 - categorical_accuracy: 0.8871
16544/60000 [=======>......................] - ETA: 1:20 - loss: 0.3679 - categorical_accuracy: 0.8873
16576/60000 [=======>......................] - ETA: 1:20 - loss: 0.3678 - categorical_accuracy: 0.8874
16608/60000 [=======>......................] - ETA: 1:20 - loss: 0.3676 - categorical_accuracy: 0.8875
16640/60000 [=======>......................] - ETA: 1:20 - loss: 0.3671 - categorical_accuracy: 0.8876
16672/60000 [=======>......................] - ETA: 1:20 - loss: 0.3666 - categorical_accuracy: 0.8878
16704/60000 [=======>......................] - ETA: 1:20 - loss: 0.3662 - categorical_accuracy: 0.8878
16736/60000 [=======>......................] - ETA: 1:20 - loss: 0.3657 - categorical_accuracy: 0.8880
16768/60000 [=======>......................] - ETA: 1:20 - loss: 0.3655 - categorical_accuracy: 0.8881
16800/60000 [=======>......................] - ETA: 1:20 - loss: 0.3654 - categorical_accuracy: 0.8882
16832/60000 [=======>......................] - ETA: 1:20 - loss: 0.3648 - categorical_accuracy: 0.8884
16864/60000 [=======>......................] - ETA: 1:20 - loss: 0.3645 - categorical_accuracy: 0.8885
16896/60000 [=======>......................] - ETA: 1:20 - loss: 0.3639 - categorical_accuracy: 0.8887
16928/60000 [=======>......................] - ETA: 1:20 - loss: 0.3634 - categorical_accuracy: 0.8888
16960/60000 [=======>......................] - ETA: 1:20 - loss: 0.3629 - categorical_accuracy: 0.8890
16992/60000 [=======>......................] - ETA: 1:19 - loss: 0.3626 - categorical_accuracy: 0.8890
17024/60000 [=======>......................] - ETA: 1:19 - loss: 0.3621 - categorical_accuracy: 0.8892
17056/60000 [=======>......................] - ETA: 1:19 - loss: 0.3618 - categorical_accuracy: 0.8892
17088/60000 [=======>......................] - ETA: 1:19 - loss: 0.3614 - categorical_accuracy: 0.8893
17120/60000 [=======>......................] - ETA: 1:19 - loss: 0.3610 - categorical_accuracy: 0.8894
17152/60000 [=======>......................] - ETA: 1:19 - loss: 0.3608 - categorical_accuracy: 0.8894
17184/60000 [=======>......................] - ETA: 1:19 - loss: 0.3602 - categorical_accuracy: 0.8896
17216/60000 [=======>......................] - ETA: 1:19 - loss: 0.3597 - categorical_accuracy: 0.8898
17248/60000 [=======>......................] - ETA: 1:19 - loss: 0.3595 - categorical_accuracy: 0.8898
17280/60000 [=======>......................] - ETA: 1:19 - loss: 0.3589 - categorical_accuracy: 0.8900
17312/60000 [=======>......................] - ETA: 1:19 - loss: 0.3583 - categorical_accuracy: 0.8902
17344/60000 [=======>......................] - ETA: 1:19 - loss: 0.3578 - categorical_accuracy: 0.8903
17376/60000 [=======>......................] - ETA: 1:19 - loss: 0.3575 - categorical_accuracy: 0.8903
17408/60000 [=======>......................] - ETA: 1:19 - loss: 0.3569 - categorical_accuracy: 0.8905
17440/60000 [=======>......................] - ETA: 1:19 - loss: 0.3567 - categorical_accuracy: 0.8905
17472/60000 [=======>......................] - ETA: 1:19 - loss: 0.3563 - categorical_accuracy: 0.8905
17504/60000 [=======>......................] - ETA: 1:19 - loss: 0.3557 - categorical_accuracy: 0.8907
17536/60000 [=======>......................] - ETA: 1:18 - loss: 0.3556 - categorical_accuracy: 0.8907
17568/60000 [=======>......................] - ETA: 1:18 - loss: 0.3550 - categorical_accuracy: 0.8909
17600/60000 [=======>......................] - ETA: 1:18 - loss: 0.3545 - categorical_accuracy: 0.8911
17632/60000 [=======>......................] - ETA: 1:18 - loss: 0.3542 - categorical_accuracy: 0.8913
17664/60000 [=======>......................] - ETA: 1:18 - loss: 0.3540 - categorical_accuracy: 0.8914
17696/60000 [=======>......................] - ETA: 1:18 - loss: 0.3534 - categorical_accuracy: 0.8916
17728/60000 [=======>......................] - ETA: 1:18 - loss: 0.3534 - categorical_accuracy: 0.8916
17760/60000 [=======>......................] - ETA: 1:18 - loss: 0.3530 - categorical_accuracy: 0.8917
17792/60000 [=======>......................] - ETA: 1:18 - loss: 0.3525 - categorical_accuracy: 0.8919
17824/60000 [=======>......................] - ETA: 1:18 - loss: 0.3522 - categorical_accuracy: 0.8920
17856/60000 [=======>......................] - ETA: 1:18 - loss: 0.3520 - categorical_accuracy: 0.8921
17888/60000 [=======>......................] - ETA: 1:18 - loss: 0.3515 - categorical_accuracy: 0.8922
17920/60000 [=======>......................] - ETA: 1:18 - loss: 0.3513 - categorical_accuracy: 0.8923
17952/60000 [=======>......................] - ETA: 1:18 - loss: 0.3507 - categorical_accuracy: 0.8925
17984/60000 [=======>......................] - ETA: 1:18 - loss: 0.3504 - categorical_accuracy: 0.8926
18016/60000 [========>.....................] - ETA: 1:18 - loss: 0.3502 - categorical_accuracy: 0.8927
18048/60000 [========>.....................] - ETA: 1:17 - loss: 0.3498 - categorical_accuracy: 0.8928
18080/60000 [========>.....................] - ETA: 1:17 - loss: 0.3495 - categorical_accuracy: 0.8930
18112/60000 [========>.....................] - ETA: 1:17 - loss: 0.3489 - categorical_accuracy: 0.8932
18144/60000 [========>.....................] - ETA: 1:17 - loss: 0.3484 - categorical_accuracy: 0.8934
18176/60000 [========>.....................] - ETA: 1:17 - loss: 0.3482 - categorical_accuracy: 0.8934
18208/60000 [========>.....................] - ETA: 1:17 - loss: 0.3477 - categorical_accuracy: 0.8935
18240/60000 [========>.....................] - ETA: 1:17 - loss: 0.3476 - categorical_accuracy: 0.8935
18272/60000 [========>.....................] - ETA: 1:17 - loss: 0.3473 - categorical_accuracy: 0.8936
18304/60000 [========>.....................] - ETA: 1:17 - loss: 0.3469 - categorical_accuracy: 0.8937
18336/60000 [========>.....................] - ETA: 1:17 - loss: 0.3465 - categorical_accuracy: 0.8938
18368/60000 [========>.....................] - ETA: 1:17 - loss: 0.3460 - categorical_accuracy: 0.8939
18400/60000 [========>.....................] - ETA: 1:17 - loss: 0.3456 - categorical_accuracy: 0.8940
18432/60000 [========>.....................] - ETA: 1:17 - loss: 0.3456 - categorical_accuracy: 0.8940
18464/60000 [========>.....................] - ETA: 1:17 - loss: 0.3453 - categorical_accuracy: 0.8942
18496/60000 [========>.....................] - ETA: 1:17 - loss: 0.3451 - categorical_accuracy: 0.8942
18528/60000 [========>.....................] - ETA: 1:17 - loss: 0.3448 - categorical_accuracy: 0.8944
18560/60000 [========>.....................] - ETA: 1:17 - loss: 0.3444 - categorical_accuracy: 0.8945
18592/60000 [========>.....................] - ETA: 1:16 - loss: 0.3442 - categorical_accuracy: 0.8944
18624/60000 [========>.....................] - ETA: 1:16 - loss: 0.3437 - categorical_accuracy: 0.8946
18656/60000 [========>.....................] - ETA: 1:16 - loss: 0.3435 - categorical_accuracy: 0.8946
18688/60000 [========>.....................] - ETA: 1:16 - loss: 0.3431 - categorical_accuracy: 0.8947
18720/60000 [========>.....................] - ETA: 1:16 - loss: 0.3427 - categorical_accuracy: 0.8948
18752/60000 [========>.....................] - ETA: 1:16 - loss: 0.3426 - categorical_accuracy: 0.8948
18784/60000 [========>.....................] - ETA: 1:16 - loss: 0.3423 - categorical_accuracy: 0.8949
18816/60000 [========>.....................] - ETA: 1:16 - loss: 0.3421 - categorical_accuracy: 0.8948
18848/60000 [========>.....................] - ETA: 1:16 - loss: 0.3417 - categorical_accuracy: 0.8949
18880/60000 [========>.....................] - ETA: 1:16 - loss: 0.3414 - categorical_accuracy: 0.8950
18912/60000 [========>.....................] - ETA: 1:16 - loss: 0.3409 - categorical_accuracy: 0.8951
18944/60000 [========>.....................] - ETA: 1:16 - loss: 0.3403 - categorical_accuracy: 0.8953
18976/60000 [========>.....................] - ETA: 1:16 - loss: 0.3402 - categorical_accuracy: 0.8953
19008/60000 [========>.....................] - ETA: 1:16 - loss: 0.3399 - categorical_accuracy: 0.8954
19040/60000 [========>.....................] - ETA: 1:16 - loss: 0.3394 - categorical_accuracy: 0.8955
19072/60000 [========>.....................] - ETA: 1:16 - loss: 0.3393 - categorical_accuracy: 0.8956
19104/60000 [========>.....................] - ETA: 1:16 - loss: 0.3389 - categorical_accuracy: 0.8956
19136/60000 [========>.....................] - ETA: 1:15 - loss: 0.3386 - categorical_accuracy: 0.8957
19168/60000 [========>.....................] - ETA: 1:15 - loss: 0.3382 - categorical_accuracy: 0.8958
19200/60000 [========>.....................] - ETA: 1:15 - loss: 0.3382 - categorical_accuracy: 0.8958
19232/60000 [========>.....................] - ETA: 1:15 - loss: 0.3379 - categorical_accuracy: 0.8959
19264/60000 [========>.....................] - ETA: 1:15 - loss: 0.3376 - categorical_accuracy: 0.8960
19296/60000 [========>.....................] - ETA: 1:15 - loss: 0.3372 - categorical_accuracy: 0.8961
19328/60000 [========>.....................] - ETA: 1:15 - loss: 0.3367 - categorical_accuracy: 0.8963
19360/60000 [========>.....................] - ETA: 1:15 - loss: 0.3366 - categorical_accuracy: 0.8963
19392/60000 [========>.....................] - ETA: 1:15 - loss: 0.3361 - categorical_accuracy: 0.8965
19424/60000 [========>.....................] - ETA: 1:15 - loss: 0.3358 - categorical_accuracy: 0.8965
19456/60000 [========>.....................] - ETA: 1:15 - loss: 0.3357 - categorical_accuracy: 0.8966
19488/60000 [========>.....................] - ETA: 1:15 - loss: 0.3352 - categorical_accuracy: 0.8968
19520/60000 [========>.....................] - ETA: 1:15 - loss: 0.3352 - categorical_accuracy: 0.8969
19552/60000 [========>.....................] - ETA: 1:15 - loss: 0.3353 - categorical_accuracy: 0.8969
19584/60000 [========>.....................] - ETA: 1:15 - loss: 0.3350 - categorical_accuracy: 0.8971
19616/60000 [========>.....................] - ETA: 1:15 - loss: 0.3346 - categorical_accuracy: 0.8972
19648/60000 [========>.....................] - ETA: 1:15 - loss: 0.3342 - categorical_accuracy: 0.8973
19680/60000 [========>.....................] - ETA: 1:14 - loss: 0.3337 - categorical_accuracy: 0.8975
19712/60000 [========>.....................] - ETA: 1:14 - loss: 0.3333 - categorical_accuracy: 0.8976
19744/60000 [========>.....................] - ETA: 1:14 - loss: 0.3331 - categorical_accuracy: 0.8976
19776/60000 [========>.....................] - ETA: 1:14 - loss: 0.3327 - categorical_accuracy: 0.8978
19808/60000 [========>.....................] - ETA: 1:14 - loss: 0.3322 - categorical_accuracy: 0.8979
19840/60000 [========>.....................] - ETA: 1:14 - loss: 0.3320 - categorical_accuracy: 0.8980
19872/60000 [========>.....................] - ETA: 1:14 - loss: 0.3318 - categorical_accuracy: 0.8980
19904/60000 [========>.....................] - ETA: 1:14 - loss: 0.3314 - categorical_accuracy: 0.8982
19936/60000 [========>.....................] - ETA: 1:14 - loss: 0.3310 - categorical_accuracy: 0.8983
19968/60000 [========>.....................] - ETA: 1:14 - loss: 0.3307 - categorical_accuracy: 0.8984
20000/60000 [=========>....................] - ETA: 1:14 - loss: 0.3304 - categorical_accuracy: 0.8985
20032/60000 [=========>....................] - ETA: 1:14 - loss: 0.3301 - categorical_accuracy: 0.8986
20064/60000 [=========>....................] - ETA: 1:14 - loss: 0.3298 - categorical_accuracy: 0.8986
20096/60000 [=========>....................] - ETA: 1:14 - loss: 0.3295 - categorical_accuracy: 0.8986
20128/60000 [=========>....................] - ETA: 1:14 - loss: 0.3293 - categorical_accuracy: 0.8987
20160/60000 [=========>....................] - ETA: 1:13 - loss: 0.3289 - categorical_accuracy: 0.8988
20192/60000 [=========>....................] - ETA: 1:13 - loss: 0.3287 - categorical_accuracy: 0.8989
20224/60000 [=========>....................] - ETA: 1:13 - loss: 0.3282 - categorical_accuracy: 0.8990
20256/60000 [=========>....................] - ETA: 1:13 - loss: 0.3278 - categorical_accuracy: 0.8992
20288/60000 [=========>....................] - ETA: 1:13 - loss: 0.3276 - categorical_accuracy: 0.8992
20320/60000 [=========>....................] - ETA: 1:13 - loss: 0.3273 - categorical_accuracy: 0.8993
20352/60000 [=========>....................] - ETA: 1:13 - loss: 0.3268 - categorical_accuracy: 0.8994
20384/60000 [=========>....................] - ETA: 1:13 - loss: 0.3265 - categorical_accuracy: 0.8995
20416/60000 [=========>....................] - ETA: 1:13 - loss: 0.3263 - categorical_accuracy: 0.8995
20448/60000 [=========>....................] - ETA: 1:13 - loss: 0.3260 - categorical_accuracy: 0.8996
20480/60000 [=========>....................] - ETA: 1:13 - loss: 0.3255 - categorical_accuracy: 0.8998
20512/60000 [=========>....................] - ETA: 1:13 - loss: 0.3252 - categorical_accuracy: 0.8999
20544/60000 [=========>....................] - ETA: 1:13 - loss: 0.3247 - categorical_accuracy: 0.9001
20576/60000 [=========>....................] - ETA: 1:13 - loss: 0.3244 - categorical_accuracy: 0.9002
20608/60000 [=========>....................] - ETA: 1:13 - loss: 0.3245 - categorical_accuracy: 0.9002
20640/60000 [=========>....................] - ETA: 1:13 - loss: 0.3242 - categorical_accuracy: 0.9003
20672/60000 [=========>....................] - ETA: 1:12 - loss: 0.3240 - categorical_accuracy: 0.9003
20704/60000 [=========>....................] - ETA: 1:12 - loss: 0.3238 - categorical_accuracy: 0.9004
20736/60000 [=========>....................] - ETA: 1:12 - loss: 0.3236 - categorical_accuracy: 0.9004
20768/60000 [=========>....................] - ETA: 1:12 - loss: 0.3234 - categorical_accuracy: 0.9003
20800/60000 [=========>....................] - ETA: 1:12 - loss: 0.3234 - categorical_accuracy: 0.9004
20832/60000 [=========>....................] - ETA: 1:12 - loss: 0.3229 - categorical_accuracy: 0.9005
20864/60000 [=========>....................] - ETA: 1:12 - loss: 0.3225 - categorical_accuracy: 0.9006
20896/60000 [=========>....................] - ETA: 1:12 - loss: 0.3221 - categorical_accuracy: 0.9007
20928/60000 [=========>....................] - ETA: 1:12 - loss: 0.3219 - categorical_accuracy: 0.9009
20960/60000 [=========>....................] - ETA: 1:12 - loss: 0.3217 - categorical_accuracy: 0.9009
20992/60000 [=========>....................] - ETA: 1:12 - loss: 0.3214 - categorical_accuracy: 0.9010
21024/60000 [=========>....................] - ETA: 1:12 - loss: 0.3212 - categorical_accuracy: 0.9010
21056/60000 [=========>....................] - ETA: 1:12 - loss: 0.3208 - categorical_accuracy: 0.9011
21088/60000 [=========>....................] - ETA: 1:12 - loss: 0.3204 - categorical_accuracy: 0.9012
21120/60000 [=========>....................] - ETA: 1:12 - loss: 0.3205 - categorical_accuracy: 0.9013
21152/60000 [=========>....................] - ETA: 1:12 - loss: 0.3201 - categorical_accuracy: 0.9014
21184/60000 [=========>....................] - ETA: 1:12 - loss: 0.3199 - categorical_accuracy: 0.9014
21216/60000 [=========>....................] - ETA: 1:11 - loss: 0.3196 - categorical_accuracy: 0.9015
21248/60000 [=========>....................] - ETA: 1:11 - loss: 0.3198 - categorical_accuracy: 0.9014
21280/60000 [=========>....................] - ETA: 1:11 - loss: 0.3196 - categorical_accuracy: 0.9015
21312/60000 [=========>....................] - ETA: 1:11 - loss: 0.3193 - categorical_accuracy: 0.9016
21344/60000 [=========>....................] - ETA: 1:11 - loss: 0.3193 - categorical_accuracy: 0.9016
21376/60000 [=========>....................] - ETA: 1:11 - loss: 0.3189 - categorical_accuracy: 0.9017
21408/60000 [=========>....................] - ETA: 1:11 - loss: 0.3185 - categorical_accuracy: 0.9018
21440/60000 [=========>....................] - ETA: 1:11 - loss: 0.3184 - categorical_accuracy: 0.9018
21472/60000 [=========>....................] - ETA: 1:11 - loss: 0.3180 - categorical_accuracy: 0.9020
21504/60000 [=========>....................] - ETA: 1:11 - loss: 0.3180 - categorical_accuracy: 0.9020
21536/60000 [=========>....................] - ETA: 1:11 - loss: 0.3178 - categorical_accuracy: 0.9020
21568/60000 [=========>....................] - ETA: 1:11 - loss: 0.3179 - categorical_accuracy: 0.9020
21600/60000 [=========>....................] - ETA: 1:11 - loss: 0.3175 - categorical_accuracy: 0.9021
21632/60000 [=========>....................] - ETA: 1:11 - loss: 0.3172 - categorical_accuracy: 0.9022
21664/60000 [=========>....................] - ETA: 1:11 - loss: 0.3169 - categorical_accuracy: 0.9023
21696/60000 [=========>....................] - ETA: 1:11 - loss: 0.3168 - categorical_accuracy: 0.9023
21728/60000 [=========>....................] - ETA: 1:10 - loss: 0.3165 - categorical_accuracy: 0.9024
21760/60000 [=========>....................] - ETA: 1:10 - loss: 0.3161 - categorical_accuracy: 0.9025
21792/60000 [=========>....................] - ETA: 1:10 - loss: 0.3157 - categorical_accuracy: 0.9026
21824/60000 [=========>....................] - ETA: 1:10 - loss: 0.3153 - categorical_accuracy: 0.9028
21856/60000 [=========>....................] - ETA: 1:10 - loss: 0.3153 - categorical_accuracy: 0.9028
21888/60000 [=========>....................] - ETA: 1:10 - loss: 0.3150 - categorical_accuracy: 0.9028
21920/60000 [=========>....................] - ETA: 1:10 - loss: 0.3146 - categorical_accuracy: 0.9030
21952/60000 [=========>....................] - ETA: 1:10 - loss: 0.3145 - categorical_accuracy: 0.9031
21984/60000 [=========>....................] - ETA: 1:10 - loss: 0.3140 - categorical_accuracy: 0.9032
22016/60000 [==========>...................] - ETA: 1:10 - loss: 0.3139 - categorical_accuracy: 0.9033
22048/60000 [==========>...................] - ETA: 1:10 - loss: 0.3135 - categorical_accuracy: 0.9034
22080/60000 [==========>...................] - ETA: 1:10 - loss: 0.3132 - categorical_accuracy: 0.9035
22112/60000 [==========>...................] - ETA: 1:10 - loss: 0.3129 - categorical_accuracy: 0.9035
22144/60000 [==========>...................] - ETA: 1:10 - loss: 0.3128 - categorical_accuracy: 0.9036
22176/60000 [==========>...................] - ETA: 1:10 - loss: 0.3128 - categorical_accuracy: 0.9036
22208/60000 [==========>...................] - ETA: 1:10 - loss: 0.3125 - categorical_accuracy: 0.9037
22240/60000 [==========>...................] - ETA: 1:10 - loss: 0.3122 - categorical_accuracy: 0.9037
22272/60000 [==========>...................] - ETA: 1:09 - loss: 0.3118 - categorical_accuracy: 0.9039
22304/60000 [==========>...................] - ETA: 1:09 - loss: 0.3115 - categorical_accuracy: 0.9039
22336/60000 [==========>...................] - ETA: 1:09 - loss: 0.3111 - categorical_accuracy: 0.9041
22368/60000 [==========>...................] - ETA: 1:09 - loss: 0.3108 - categorical_accuracy: 0.9041
22400/60000 [==========>...................] - ETA: 1:09 - loss: 0.3108 - categorical_accuracy: 0.9042
22432/60000 [==========>...................] - ETA: 1:09 - loss: 0.3107 - categorical_accuracy: 0.9042
22464/60000 [==========>...................] - ETA: 1:09 - loss: 0.3104 - categorical_accuracy: 0.9042
22496/60000 [==========>...................] - ETA: 1:09 - loss: 0.3104 - categorical_accuracy: 0.9043
22528/60000 [==========>...................] - ETA: 1:09 - loss: 0.3100 - categorical_accuracy: 0.9044
22560/60000 [==========>...................] - ETA: 1:09 - loss: 0.3097 - categorical_accuracy: 0.9045
22592/60000 [==========>...................] - ETA: 1:09 - loss: 0.3093 - categorical_accuracy: 0.9046
22624/60000 [==========>...................] - ETA: 1:09 - loss: 0.3090 - categorical_accuracy: 0.9047
22656/60000 [==========>...................] - ETA: 1:09 - loss: 0.3086 - categorical_accuracy: 0.9048
22688/60000 [==========>...................] - ETA: 1:09 - loss: 0.3084 - categorical_accuracy: 0.9049
22720/60000 [==========>...................] - ETA: 1:09 - loss: 0.3085 - categorical_accuracy: 0.9049
22752/60000 [==========>...................] - ETA: 1:09 - loss: 0.3083 - categorical_accuracy: 0.9049
22784/60000 [==========>...................] - ETA: 1:08 - loss: 0.3083 - categorical_accuracy: 0.9049
22816/60000 [==========>...................] - ETA: 1:08 - loss: 0.3082 - categorical_accuracy: 0.9050
22848/60000 [==========>...................] - ETA: 1:08 - loss: 0.3078 - categorical_accuracy: 0.9051
22880/60000 [==========>...................] - ETA: 1:08 - loss: 0.3075 - categorical_accuracy: 0.9052
22912/60000 [==========>...................] - ETA: 1:08 - loss: 0.3073 - categorical_accuracy: 0.9052
22944/60000 [==========>...................] - ETA: 1:08 - loss: 0.3070 - categorical_accuracy: 0.9053
22976/60000 [==========>...................] - ETA: 1:08 - loss: 0.3067 - categorical_accuracy: 0.9053
23008/60000 [==========>...................] - ETA: 1:08 - loss: 0.3068 - categorical_accuracy: 0.9054
23040/60000 [==========>...................] - ETA: 1:08 - loss: 0.3065 - categorical_accuracy: 0.9054
23072/60000 [==========>...................] - ETA: 1:08 - loss: 0.3061 - categorical_accuracy: 0.9056
23104/60000 [==========>...................] - ETA: 1:08 - loss: 0.3058 - categorical_accuracy: 0.9057
23136/60000 [==========>...................] - ETA: 1:08 - loss: 0.3056 - categorical_accuracy: 0.9057
23168/60000 [==========>...................] - ETA: 1:08 - loss: 0.3052 - categorical_accuracy: 0.9059
23200/60000 [==========>...................] - ETA: 1:08 - loss: 0.3052 - categorical_accuracy: 0.9059
23232/60000 [==========>...................] - ETA: 1:08 - loss: 0.3049 - categorical_accuracy: 0.9060
23264/60000 [==========>...................] - ETA: 1:08 - loss: 0.3049 - categorical_accuracy: 0.9060
23296/60000 [==========>...................] - ETA: 1:08 - loss: 0.3045 - categorical_accuracy: 0.9061
23328/60000 [==========>...................] - ETA: 1:07 - loss: 0.3042 - categorical_accuracy: 0.9062
23360/60000 [==========>...................] - ETA: 1:07 - loss: 0.3039 - categorical_accuracy: 0.9063
23392/60000 [==========>...................] - ETA: 1:07 - loss: 0.3035 - categorical_accuracy: 0.9064
23424/60000 [==========>...................] - ETA: 1:07 - loss: 0.3032 - categorical_accuracy: 0.9065
23456/60000 [==========>...................] - ETA: 1:07 - loss: 0.3031 - categorical_accuracy: 0.9065
23488/60000 [==========>...................] - ETA: 1:07 - loss: 0.3028 - categorical_accuracy: 0.9066
23520/60000 [==========>...................] - ETA: 1:07 - loss: 0.3026 - categorical_accuracy: 0.9067
23552/60000 [==========>...................] - ETA: 1:07 - loss: 0.3022 - categorical_accuracy: 0.9068
23584/60000 [==========>...................] - ETA: 1:07 - loss: 0.3019 - categorical_accuracy: 0.9069
23616/60000 [==========>...................] - ETA: 1:07 - loss: 0.3016 - categorical_accuracy: 0.9070
23648/60000 [==========>...................] - ETA: 1:07 - loss: 0.3014 - categorical_accuracy: 0.9071
23680/60000 [==========>...................] - ETA: 1:07 - loss: 0.3011 - categorical_accuracy: 0.9071
23712/60000 [==========>...................] - ETA: 1:07 - loss: 0.3009 - categorical_accuracy: 0.9072
23744/60000 [==========>...................] - ETA: 1:07 - loss: 0.3006 - categorical_accuracy: 0.9073
23776/60000 [==========>...................] - ETA: 1:07 - loss: 0.3004 - categorical_accuracy: 0.9073
23808/60000 [==========>...................] - ETA: 1:07 - loss: 0.3003 - categorical_accuracy: 0.9074
23840/60000 [==========>...................] - ETA: 1:07 - loss: 0.3000 - categorical_accuracy: 0.9074
23872/60000 [==========>...................] - ETA: 1:06 - loss: 0.2998 - categorical_accuracy: 0.9075
23904/60000 [==========>...................] - ETA: 1:06 - loss: 0.2995 - categorical_accuracy: 0.9076
23936/60000 [==========>...................] - ETA: 1:06 - loss: 0.2991 - categorical_accuracy: 0.9077
23968/60000 [==========>...................] - ETA: 1:06 - loss: 0.2993 - categorical_accuracy: 0.9077
24000/60000 [===========>..................] - ETA: 1:06 - loss: 0.2995 - categorical_accuracy: 0.9077
24032/60000 [===========>..................] - ETA: 1:06 - loss: 0.2993 - categorical_accuracy: 0.9077
24064/60000 [===========>..................] - ETA: 1:06 - loss: 0.2992 - categorical_accuracy: 0.9078
24096/60000 [===========>..................] - ETA: 1:06 - loss: 0.2990 - categorical_accuracy: 0.9078
24128/60000 [===========>..................] - ETA: 1:06 - loss: 0.2988 - categorical_accuracy: 0.9079
24160/60000 [===========>..................] - ETA: 1:06 - loss: 0.2985 - categorical_accuracy: 0.9080
24192/60000 [===========>..................] - ETA: 1:06 - loss: 0.2982 - categorical_accuracy: 0.9081
24224/60000 [===========>..................] - ETA: 1:06 - loss: 0.2979 - categorical_accuracy: 0.9081
24256/60000 [===========>..................] - ETA: 1:06 - loss: 0.2978 - categorical_accuracy: 0.9082
24288/60000 [===========>..................] - ETA: 1:06 - loss: 0.2976 - categorical_accuracy: 0.9082
24320/60000 [===========>..................] - ETA: 1:06 - loss: 0.2972 - categorical_accuracy: 0.9083
24352/60000 [===========>..................] - ETA: 1:06 - loss: 0.2970 - categorical_accuracy: 0.9084
24384/60000 [===========>..................] - ETA: 1:06 - loss: 0.2967 - categorical_accuracy: 0.9085
24416/60000 [===========>..................] - ETA: 1:05 - loss: 0.2968 - categorical_accuracy: 0.9085
24448/60000 [===========>..................] - ETA: 1:05 - loss: 0.2966 - categorical_accuracy: 0.9085
24480/60000 [===========>..................] - ETA: 1:05 - loss: 0.2966 - categorical_accuracy: 0.9085
24512/60000 [===========>..................] - ETA: 1:05 - loss: 0.2963 - categorical_accuracy: 0.9086
24544/60000 [===========>..................] - ETA: 1:05 - loss: 0.2960 - categorical_accuracy: 0.9087
24576/60000 [===========>..................] - ETA: 1:05 - loss: 0.2958 - categorical_accuracy: 0.9087
24608/60000 [===========>..................] - ETA: 1:05 - loss: 0.2956 - categorical_accuracy: 0.9088
24640/60000 [===========>..................] - ETA: 1:05 - loss: 0.2955 - categorical_accuracy: 0.9088
24672/60000 [===========>..................] - ETA: 1:05 - loss: 0.2952 - categorical_accuracy: 0.9089
24704/60000 [===========>..................] - ETA: 1:05 - loss: 0.2950 - categorical_accuracy: 0.9090
24736/60000 [===========>..................] - ETA: 1:05 - loss: 0.2954 - categorical_accuracy: 0.9090
24768/60000 [===========>..................] - ETA: 1:05 - loss: 0.2957 - categorical_accuracy: 0.9089
24800/60000 [===========>..................] - ETA: 1:05 - loss: 0.2956 - categorical_accuracy: 0.9090
24832/60000 [===========>..................] - ETA: 1:05 - loss: 0.2953 - categorical_accuracy: 0.9090
24864/60000 [===========>..................] - ETA: 1:05 - loss: 0.2950 - categorical_accuracy: 0.9091
24896/60000 [===========>..................] - ETA: 1:05 - loss: 0.2949 - categorical_accuracy: 0.9091
24928/60000 [===========>..................] - ETA: 1:05 - loss: 0.2946 - categorical_accuracy: 0.9092
24960/60000 [===========>..................] - ETA: 1:04 - loss: 0.2943 - categorical_accuracy: 0.9093
24992/60000 [===========>..................] - ETA: 1:04 - loss: 0.2943 - categorical_accuracy: 0.9093
25024/60000 [===========>..................] - ETA: 1:04 - loss: 0.2941 - categorical_accuracy: 0.9094
25056/60000 [===========>..................] - ETA: 1:04 - loss: 0.2937 - categorical_accuracy: 0.9095
25088/60000 [===========>..................] - ETA: 1:04 - loss: 0.2934 - categorical_accuracy: 0.9096
25120/60000 [===========>..................] - ETA: 1:04 - loss: 0.2932 - categorical_accuracy: 0.9097
25152/60000 [===========>..................] - ETA: 1:04 - loss: 0.2929 - categorical_accuracy: 0.9098
25184/60000 [===========>..................] - ETA: 1:04 - loss: 0.2927 - categorical_accuracy: 0.9099
25216/60000 [===========>..................] - ETA: 1:04 - loss: 0.2926 - categorical_accuracy: 0.9099
25248/60000 [===========>..................] - ETA: 1:04 - loss: 0.2926 - categorical_accuracy: 0.9099
25280/60000 [===========>..................] - ETA: 1:04 - loss: 0.2924 - categorical_accuracy: 0.9100
25312/60000 [===========>..................] - ETA: 1:04 - loss: 0.2927 - categorical_accuracy: 0.9100
25344/60000 [===========>..................] - ETA: 1:04 - loss: 0.2925 - categorical_accuracy: 0.9100
25376/60000 [===========>..................] - ETA: 1:04 - loss: 0.2922 - categorical_accuracy: 0.9101
25408/60000 [===========>..................] - ETA: 1:04 - loss: 0.2919 - categorical_accuracy: 0.9102
25440/60000 [===========>..................] - ETA: 1:04 - loss: 0.2917 - categorical_accuracy: 0.9103
25472/60000 [===========>..................] - ETA: 1:04 - loss: 0.2914 - categorical_accuracy: 0.9104
25504/60000 [===========>..................] - ETA: 1:03 - loss: 0.2911 - categorical_accuracy: 0.9104
25536/60000 [===========>..................] - ETA: 1:03 - loss: 0.2909 - categorical_accuracy: 0.9105
25568/60000 [===========>..................] - ETA: 1:03 - loss: 0.2908 - categorical_accuracy: 0.9106
25600/60000 [===========>..................] - ETA: 1:03 - loss: 0.2905 - categorical_accuracy: 0.9106
25632/60000 [===========>..................] - ETA: 1:03 - loss: 0.2902 - categorical_accuracy: 0.9107
25664/60000 [===========>..................] - ETA: 1:03 - loss: 0.2899 - categorical_accuracy: 0.9108
25696/60000 [===========>..................] - ETA: 1:03 - loss: 0.2902 - categorical_accuracy: 0.9108
25728/60000 [===========>..................] - ETA: 1:03 - loss: 0.2900 - categorical_accuracy: 0.9109
25760/60000 [===========>..................] - ETA: 1:03 - loss: 0.2897 - categorical_accuracy: 0.9110
25792/60000 [===========>..................] - ETA: 1:03 - loss: 0.2893 - categorical_accuracy: 0.9111
25824/60000 [===========>..................] - ETA: 1:03 - loss: 0.2890 - categorical_accuracy: 0.9112
25856/60000 [===========>..................] - ETA: 1:03 - loss: 0.2888 - categorical_accuracy: 0.9112
25888/60000 [===========>..................] - ETA: 1:03 - loss: 0.2885 - categorical_accuracy: 0.9113
25920/60000 [===========>..................] - ETA: 1:03 - loss: 0.2883 - categorical_accuracy: 0.9114
25952/60000 [===========>..................] - ETA: 1:03 - loss: 0.2880 - categorical_accuracy: 0.9115
25984/60000 [===========>..................] - ETA: 1:03 - loss: 0.2878 - categorical_accuracy: 0.9116
26016/60000 [============>.................] - ETA: 1:02 - loss: 0.2875 - categorical_accuracy: 0.9117
26048/60000 [============>.................] - ETA: 1:02 - loss: 0.2872 - categorical_accuracy: 0.9118
26080/60000 [============>.................] - ETA: 1:02 - loss: 0.2870 - categorical_accuracy: 0.9118
26112/60000 [============>.................] - ETA: 1:02 - loss: 0.2870 - categorical_accuracy: 0.9118
26144/60000 [============>.................] - ETA: 1:02 - loss: 0.2867 - categorical_accuracy: 0.9119
26176/60000 [============>.................] - ETA: 1:02 - loss: 0.2865 - categorical_accuracy: 0.9119
26208/60000 [============>.................] - ETA: 1:02 - loss: 0.2863 - categorical_accuracy: 0.9120
26240/60000 [============>.................] - ETA: 1:02 - loss: 0.2859 - categorical_accuracy: 0.9121
26272/60000 [============>.................] - ETA: 1:02 - loss: 0.2860 - categorical_accuracy: 0.9121
26304/60000 [============>.................] - ETA: 1:02 - loss: 0.2859 - categorical_accuracy: 0.9121
26336/60000 [============>.................] - ETA: 1:02 - loss: 0.2858 - categorical_accuracy: 0.9122
26368/60000 [============>.................] - ETA: 1:02 - loss: 0.2855 - categorical_accuracy: 0.9123
26400/60000 [============>.................] - ETA: 1:02 - loss: 0.2853 - categorical_accuracy: 0.9123
26432/60000 [============>.................] - ETA: 1:02 - loss: 0.2852 - categorical_accuracy: 0.9124
26464/60000 [============>.................] - ETA: 1:02 - loss: 0.2851 - categorical_accuracy: 0.9124
26496/60000 [============>.................] - ETA: 1:02 - loss: 0.2849 - categorical_accuracy: 0.9125
26528/60000 [============>.................] - ETA: 1:01 - loss: 0.2847 - categorical_accuracy: 0.9125
26560/60000 [============>.................] - ETA: 1:01 - loss: 0.2846 - categorical_accuracy: 0.9125
26592/60000 [============>.................] - ETA: 1:01 - loss: 0.2843 - categorical_accuracy: 0.9126
26624/60000 [============>.................] - ETA: 1:01 - loss: 0.2841 - categorical_accuracy: 0.9127
26656/60000 [============>.................] - ETA: 1:01 - loss: 0.2840 - categorical_accuracy: 0.9127
26688/60000 [============>.................] - ETA: 1:01 - loss: 0.2838 - categorical_accuracy: 0.9127
26720/60000 [============>.................] - ETA: 1:01 - loss: 0.2838 - categorical_accuracy: 0.9127
26752/60000 [============>.................] - ETA: 1:01 - loss: 0.2840 - categorical_accuracy: 0.9127
26784/60000 [============>.................] - ETA: 1:01 - loss: 0.2839 - categorical_accuracy: 0.9128
26816/60000 [============>.................] - ETA: 1:01 - loss: 0.2838 - categorical_accuracy: 0.9128
26848/60000 [============>.................] - ETA: 1:01 - loss: 0.2835 - categorical_accuracy: 0.9129
26880/60000 [============>.................] - ETA: 1:01 - loss: 0.2834 - categorical_accuracy: 0.9130
26912/60000 [============>.................] - ETA: 1:01 - loss: 0.2830 - categorical_accuracy: 0.9131
26944/60000 [============>.................] - ETA: 1:01 - loss: 0.2830 - categorical_accuracy: 0.9131
26976/60000 [============>.................] - ETA: 1:01 - loss: 0.2828 - categorical_accuracy: 0.9131
27008/60000 [============>.................] - ETA: 1:01 - loss: 0.2825 - categorical_accuracy: 0.9132
27040/60000 [============>.................] - ETA: 1:01 - loss: 0.2822 - categorical_accuracy: 0.9134
27072/60000 [============>.................] - ETA: 1:00 - loss: 0.2819 - categorical_accuracy: 0.9135
27104/60000 [============>.................] - ETA: 1:00 - loss: 0.2816 - categorical_accuracy: 0.9136
27136/60000 [============>.................] - ETA: 1:00 - loss: 0.2814 - categorical_accuracy: 0.9136
27168/60000 [============>.................] - ETA: 1:00 - loss: 0.2811 - categorical_accuracy: 0.9137
27200/60000 [============>.................] - ETA: 1:00 - loss: 0.2809 - categorical_accuracy: 0.9138
27232/60000 [============>.................] - ETA: 1:00 - loss: 0.2806 - categorical_accuracy: 0.9139
27264/60000 [============>.................] - ETA: 1:00 - loss: 0.2803 - categorical_accuracy: 0.9140
27296/60000 [============>.................] - ETA: 1:00 - loss: 0.2802 - categorical_accuracy: 0.9140
27328/60000 [============>.................] - ETA: 1:00 - loss: 0.2799 - categorical_accuracy: 0.9141
27360/60000 [============>.................] - ETA: 1:00 - loss: 0.2801 - categorical_accuracy: 0.9141
27392/60000 [============>.................] - ETA: 1:00 - loss: 0.2798 - categorical_accuracy: 0.9142
27424/60000 [============>.................] - ETA: 1:00 - loss: 0.2795 - categorical_accuracy: 0.9143
27456/60000 [============>.................] - ETA: 1:00 - loss: 0.2793 - categorical_accuracy: 0.9144
27488/60000 [============>.................] - ETA: 1:00 - loss: 0.2792 - categorical_accuracy: 0.9144
27520/60000 [============>.................] - ETA: 1:00 - loss: 0.2790 - categorical_accuracy: 0.9145
27552/60000 [============>.................] - ETA: 1:00 - loss: 0.2787 - categorical_accuracy: 0.9146
27584/60000 [============>.................] - ETA: 1:00 - loss: 0.2787 - categorical_accuracy: 0.9146
27616/60000 [============>.................] - ETA: 1:00 - loss: 0.2785 - categorical_accuracy: 0.9146
27648/60000 [============>.................] - ETA: 59s - loss: 0.2784 - categorical_accuracy: 0.9146 
27680/60000 [============>.................] - ETA: 59s - loss: 0.2781 - categorical_accuracy: 0.9147
27712/60000 [============>.................] - ETA: 59s - loss: 0.2779 - categorical_accuracy: 0.9148
27744/60000 [============>.................] - ETA: 59s - loss: 0.2778 - categorical_accuracy: 0.9148
27776/60000 [============>.................] - ETA: 59s - loss: 0.2777 - categorical_accuracy: 0.9148
27808/60000 [============>.................] - ETA: 59s - loss: 0.2774 - categorical_accuracy: 0.9149
27840/60000 [============>.................] - ETA: 59s - loss: 0.2772 - categorical_accuracy: 0.9150
27872/60000 [============>.................] - ETA: 59s - loss: 0.2770 - categorical_accuracy: 0.9151
27904/60000 [============>.................] - ETA: 59s - loss: 0.2767 - categorical_accuracy: 0.9151
27936/60000 [============>.................] - ETA: 59s - loss: 0.2768 - categorical_accuracy: 0.9152
27968/60000 [============>.................] - ETA: 59s - loss: 0.2765 - categorical_accuracy: 0.9153
28000/60000 [=============>................] - ETA: 59s - loss: 0.2762 - categorical_accuracy: 0.9154
28032/60000 [=============>................] - ETA: 59s - loss: 0.2762 - categorical_accuracy: 0.9155
28064/60000 [=============>................] - ETA: 59s - loss: 0.2763 - categorical_accuracy: 0.9155
28096/60000 [=============>................] - ETA: 59s - loss: 0.2760 - categorical_accuracy: 0.9156
28128/60000 [=============>................] - ETA: 59s - loss: 0.2757 - categorical_accuracy: 0.9157
28160/60000 [=============>................] - ETA: 59s - loss: 0.2756 - categorical_accuracy: 0.9157
28192/60000 [=============>................] - ETA: 58s - loss: 0.2753 - categorical_accuracy: 0.9158
28224/60000 [=============>................] - ETA: 58s - loss: 0.2750 - categorical_accuracy: 0.9159
28256/60000 [=============>................] - ETA: 58s - loss: 0.2747 - categorical_accuracy: 0.9160
28288/60000 [=============>................] - ETA: 58s - loss: 0.2745 - categorical_accuracy: 0.9161
28320/60000 [=============>................] - ETA: 58s - loss: 0.2746 - categorical_accuracy: 0.9161
28352/60000 [=============>................] - ETA: 58s - loss: 0.2746 - categorical_accuracy: 0.9161
28384/60000 [=============>................] - ETA: 58s - loss: 0.2744 - categorical_accuracy: 0.9162
28416/60000 [=============>................] - ETA: 58s - loss: 0.2742 - categorical_accuracy: 0.9162
28448/60000 [=============>................] - ETA: 58s - loss: 0.2741 - categorical_accuracy: 0.9163
28480/60000 [=============>................] - ETA: 58s - loss: 0.2739 - categorical_accuracy: 0.9164
28512/60000 [=============>................] - ETA: 58s - loss: 0.2736 - categorical_accuracy: 0.9165
28544/60000 [=============>................] - ETA: 58s - loss: 0.2734 - categorical_accuracy: 0.9165
28576/60000 [=============>................] - ETA: 58s - loss: 0.2731 - categorical_accuracy: 0.9166
28608/60000 [=============>................] - ETA: 58s - loss: 0.2730 - categorical_accuracy: 0.9166
28640/60000 [=============>................] - ETA: 58s - loss: 0.2729 - categorical_accuracy: 0.9167
28672/60000 [=============>................] - ETA: 58s - loss: 0.2728 - categorical_accuracy: 0.9167
28704/60000 [=============>................] - ETA: 57s - loss: 0.2727 - categorical_accuracy: 0.9167
28736/60000 [=============>................] - ETA: 57s - loss: 0.2725 - categorical_accuracy: 0.9168
28768/60000 [=============>................] - ETA: 57s - loss: 0.2723 - categorical_accuracy: 0.9169
28800/60000 [=============>................] - ETA: 57s - loss: 0.2722 - categorical_accuracy: 0.9169
28832/60000 [=============>................] - ETA: 57s - loss: 0.2719 - categorical_accuracy: 0.9170
28864/60000 [=============>................] - ETA: 57s - loss: 0.2717 - categorical_accuracy: 0.9170
28896/60000 [=============>................] - ETA: 57s - loss: 0.2716 - categorical_accuracy: 0.9171
28928/60000 [=============>................] - ETA: 57s - loss: 0.2713 - categorical_accuracy: 0.9172
28960/60000 [=============>................] - ETA: 57s - loss: 0.2713 - categorical_accuracy: 0.9172
28992/60000 [=============>................] - ETA: 57s - loss: 0.2712 - categorical_accuracy: 0.9173
29024/60000 [=============>................] - ETA: 57s - loss: 0.2709 - categorical_accuracy: 0.9174
29056/60000 [=============>................] - ETA: 57s - loss: 0.2709 - categorical_accuracy: 0.9174
29088/60000 [=============>................] - ETA: 57s - loss: 0.2707 - categorical_accuracy: 0.9175
29120/60000 [=============>................] - ETA: 57s - loss: 0.2705 - categorical_accuracy: 0.9175
29152/60000 [=============>................] - ETA: 57s - loss: 0.2703 - categorical_accuracy: 0.9176
29184/60000 [=============>................] - ETA: 57s - loss: 0.2702 - categorical_accuracy: 0.9176
29216/60000 [=============>................] - ETA: 57s - loss: 0.2699 - categorical_accuracy: 0.9177
29248/60000 [=============>................] - ETA: 56s - loss: 0.2697 - categorical_accuracy: 0.9178
29280/60000 [=============>................] - ETA: 56s - loss: 0.2694 - categorical_accuracy: 0.9179
29312/60000 [=============>................] - ETA: 56s - loss: 0.2692 - categorical_accuracy: 0.9179
29344/60000 [=============>................] - ETA: 56s - loss: 0.2690 - categorical_accuracy: 0.9180
29376/60000 [=============>................] - ETA: 56s - loss: 0.2687 - categorical_accuracy: 0.9181
29408/60000 [=============>................] - ETA: 56s - loss: 0.2685 - categorical_accuracy: 0.9182
29440/60000 [=============>................] - ETA: 56s - loss: 0.2684 - categorical_accuracy: 0.9182
29472/60000 [=============>................] - ETA: 56s - loss: 0.2683 - categorical_accuracy: 0.9183
29504/60000 [=============>................] - ETA: 56s - loss: 0.2682 - categorical_accuracy: 0.9183
29536/60000 [=============>................] - ETA: 56s - loss: 0.2679 - categorical_accuracy: 0.9184
29568/60000 [=============>................] - ETA: 56s - loss: 0.2678 - categorical_accuracy: 0.9184
29600/60000 [=============>................] - ETA: 56s - loss: 0.2676 - categorical_accuracy: 0.9185
29632/60000 [=============>................] - ETA: 56s - loss: 0.2673 - categorical_accuracy: 0.9186
29664/60000 [=============>................] - ETA: 56s - loss: 0.2670 - categorical_accuracy: 0.9187
29696/60000 [=============>................] - ETA: 56s - loss: 0.2668 - categorical_accuracy: 0.9187
29728/60000 [=============>................] - ETA: 56s - loss: 0.2666 - categorical_accuracy: 0.9188
29760/60000 [=============>................] - ETA: 56s - loss: 0.2664 - categorical_accuracy: 0.9189
29792/60000 [=============>................] - ETA: 55s - loss: 0.2662 - categorical_accuracy: 0.9189
29824/60000 [=============>................] - ETA: 55s - loss: 0.2662 - categorical_accuracy: 0.9190
29856/60000 [=============>................] - ETA: 55s - loss: 0.2659 - categorical_accuracy: 0.9190
29888/60000 [=============>................] - ETA: 55s - loss: 0.2657 - categorical_accuracy: 0.9191
29920/60000 [=============>................] - ETA: 55s - loss: 0.2655 - categorical_accuracy: 0.9192
29952/60000 [=============>................] - ETA: 55s - loss: 0.2654 - categorical_accuracy: 0.9192
29984/60000 [=============>................] - ETA: 55s - loss: 0.2652 - categorical_accuracy: 0.9193
30016/60000 [==============>...............] - ETA: 55s - loss: 0.2650 - categorical_accuracy: 0.9194
30048/60000 [==============>...............] - ETA: 55s - loss: 0.2651 - categorical_accuracy: 0.9194
30080/60000 [==============>...............] - ETA: 55s - loss: 0.2651 - categorical_accuracy: 0.9194
30112/60000 [==============>...............] - ETA: 55s - loss: 0.2649 - categorical_accuracy: 0.9194
30144/60000 [==============>...............] - ETA: 55s - loss: 0.2648 - categorical_accuracy: 0.9195
30176/60000 [==============>...............] - ETA: 55s - loss: 0.2647 - categorical_accuracy: 0.9195
30208/60000 [==============>...............] - ETA: 55s - loss: 0.2648 - categorical_accuracy: 0.9196
30240/60000 [==============>...............] - ETA: 55s - loss: 0.2647 - categorical_accuracy: 0.9196
30272/60000 [==============>...............] - ETA: 55s - loss: 0.2647 - categorical_accuracy: 0.9197
30304/60000 [==============>...............] - ETA: 55s - loss: 0.2645 - categorical_accuracy: 0.9197
30336/60000 [==============>...............] - ETA: 54s - loss: 0.2643 - categorical_accuracy: 0.9198
30368/60000 [==============>...............] - ETA: 54s - loss: 0.2641 - categorical_accuracy: 0.9198
30400/60000 [==============>...............] - ETA: 54s - loss: 0.2640 - categorical_accuracy: 0.9198
30432/60000 [==============>...............] - ETA: 54s - loss: 0.2637 - categorical_accuracy: 0.9199
30464/60000 [==============>...............] - ETA: 54s - loss: 0.2635 - categorical_accuracy: 0.9200
30496/60000 [==============>...............] - ETA: 54s - loss: 0.2632 - categorical_accuracy: 0.9201
30528/60000 [==============>...............] - ETA: 54s - loss: 0.2635 - categorical_accuracy: 0.9200
30560/60000 [==============>...............] - ETA: 54s - loss: 0.2633 - categorical_accuracy: 0.9200
30592/60000 [==============>...............] - ETA: 54s - loss: 0.2630 - categorical_accuracy: 0.9200
30624/60000 [==============>...............] - ETA: 54s - loss: 0.2629 - categorical_accuracy: 0.9201
30656/60000 [==============>...............] - ETA: 54s - loss: 0.2627 - categorical_accuracy: 0.9201
30688/60000 [==============>...............] - ETA: 54s - loss: 0.2627 - categorical_accuracy: 0.9202
30720/60000 [==============>...............] - ETA: 54s - loss: 0.2627 - categorical_accuracy: 0.9202
30752/60000 [==============>...............] - ETA: 54s - loss: 0.2625 - categorical_accuracy: 0.9202
30784/60000 [==============>...............] - ETA: 54s - loss: 0.2623 - categorical_accuracy: 0.9203
30816/60000 [==============>...............] - ETA: 54s - loss: 0.2621 - categorical_accuracy: 0.9204
30848/60000 [==============>...............] - ETA: 54s - loss: 0.2620 - categorical_accuracy: 0.9204
30880/60000 [==============>...............] - ETA: 53s - loss: 0.2618 - categorical_accuracy: 0.9205
30912/60000 [==============>...............] - ETA: 53s - loss: 0.2616 - categorical_accuracy: 0.9205
30944/60000 [==============>...............] - ETA: 53s - loss: 0.2618 - categorical_accuracy: 0.9205
30976/60000 [==============>...............] - ETA: 53s - loss: 0.2615 - categorical_accuracy: 0.9206
31008/60000 [==============>...............] - ETA: 53s - loss: 0.2613 - categorical_accuracy: 0.9206
31040/60000 [==============>...............] - ETA: 53s - loss: 0.2611 - categorical_accuracy: 0.9207
31072/60000 [==============>...............] - ETA: 53s - loss: 0.2609 - categorical_accuracy: 0.9208
31104/60000 [==============>...............] - ETA: 53s - loss: 0.2608 - categorical_accuracy: 0.9208
31136/60000 [==============>...............] - ETA: 53s - loss: 0.2606 - categorical_accuracy: 0.9208
31168/60000 [==============>...............] - ETA: 53s - loss: 0.2605 - categorical_accuracy: 0.9208
31200/60000 [==============>...............] - ETA: 53s - loss: 0.2606 - categorical_accuracy: 0.9208
31232/60000 [==============>...............] - ETA: 53s - loss: 0.2603 - categorical_accuracy: 0.9209
31264/60000 [==============>...............] - ETA: 53s - loss: 0.2601 - categorical_accuracy: 0.9209
31296/60000 [==============>...............] - ETA: 53s - loss: 0.2599 - categorical_accuracy: 0.9209
31328/60000 [==============>...............] - ETA: 53s - loss: 0.2599 - categorical_accuracy: 0.9210
31360/60000 [==============>...............] - ETA: 53s - loss: 0.2598 - categorical_accuracy: 0.9210
31392/60000 [==============>...............] - ETA: 52s - loss: 0.2597 - categorical_accuracy: 0.9210
31424/60000 [==============>...............] - ETA: 52s - loss: 0.2596 - categorical_accuracy: 0.9210
31456/60000 [==============>...............] - ETA: 52s - loss: 0.2596 - categorical_accuracy: 0.9210
31488/60000 [==============>...............] - ETA: 52s - loss: 0.2598 - categorical_accuracy: 0.9210
31520/60000 [==============>...............] - ETA: 52s - loss: 0.2596 - categorical_accuracy: 0.9210
31552/60000 [==============>...............] - ETA: 52s - loss: 0.2594 - categorical_accuracy: 0.9211
31584/60000 [==============>...............] - ETA: 52s - loss: 0.2592 - categorical_accuracy: 0.9212
31616/60000 [==============>...............] - ETA: 52s - loss: 0.2590 - categorical_accuracy: 0.9213
31648/60000 [==============>...............] - ETA: 52s - loss: 0.2588 - categorical_accuracy: 0.9214
31680/60000 [==============>...............] - ETA: 52s - loss: 0.2586 - categorical_accuracy: 0.9214
31712/60000 [==============>...............] - ETA: 52s - loss: 0.2586 - categorical_accuracy: 0.9214
31744/60000 [==============>...............] - ETA: 52s - loss: 0.2585 - categorical_accuracy: 0.9214
31776/60000 [==============>...............] - ETA: 52s - loss: 0.2583 - categorical_accuracy: 0.9215
31808/60000 [==============>...............] - ETA: 52s - loss: 0.2581 - categorical_accuracy: 0.9216
31840/60000 [==============>...............] - ETA: 52s - loss: 0.2579 - categorical_accuracy: 0.9216
31872/60000 [==============>...............] - ETA: 52s - loss: 0.2576 - categorical_accuracy: 0.9217
31904/60000 [==============>...............] - ETA: 52s - loss: 0.2575 - categorical_accuracy: 0.9218
31936/60000 [==============>...............] - ETA: 51s - loss: 0.2573 - categorical_accuracy: 0.9218
31968/60000 [==============>...............] - ETA: 51s - loss: 0.2573 - categorical_accuracy: 0.9219
32000/60000 [===============>..............] - ETA: 51s - loss: 0.2571 - categorical_accuracy: 0.9219
32032/60000 [===============>..............] - ETA: 51s - loss: 0.2568 - categorical_accuracy: 0.9220
32064/60000 [===============>..............] - ETA: 51s - loss: 0.2566 - categorical_accuracy: 0.9221
32096/60000 [===============>..............] - ETA: 51s - loss: 0.2564 - categorical_accuracy: 0.9221
32128/60000 [===============>..............] - ETA: 51s - loss: 0.2567 - categorical_accuracy: 0.9221
32160/60000 [===============>..............] - ETA: 51s - loss: 0.2565 - categorical_accuracy: 0.9221
32192/60000 [===============>..............] - ETA: 51s - loss: 0.2565 - categorical_accuracy: 0.9222
32224/60000 [===============>..............] - ETA: 51s - loss: 0.2563 - categorical_accuracy: 0.9222
32256/60000 [===============>..............] - ETA: 51s - loss: 0.2564 - categorical_accuracy: 0.9222
32288/60000 [===============>..............] - ETA: 51s - loss: 0.2562 - categorical_accuracy: 0.9223
32320/60000 [===============>..............] - ETA: 51s - loss: 0.2561 - categorical_accuracy: 0.9223
32352/60000 [===============>..............] - ETA: 51s - loss: 0.2558 - categorical_accuracy: 0.9224
32384/60000 [===============>..............] - ETA: 51s - loss: 0.2557 - categorical_accuracy: 0.9224
32416/60000 [===============>..............] - ETA: 51s - loss: 0.2556 - categorical_accuracy: 0.9224
32448/60000 [===============>..............] - ETA: 51s - loss: 0.2555 - categorical_accuracy: 0.9224
32480/60000 [===============>..............] - ETA: 50s - loss: 0.2554 - categorical_accuracy: 0.9225
32512/60000 [===============>..............] - ETA: 50s - loss: 0.2552 - categorical_accuracy: 0.9225
32544/60000 [===============>..............] - ETA: 50s - loss: 0.2551 - categorical_accuracy: 0.9226
32576/60000 [===============>..............] - ETA: 50s - loss: 0.2551 - categorical_accuracy: 0.9226
32608/60000 [===============>..............] - ETA: 50s - loss: 0.2550 - categorical_accuracy: 0.9226
32640/60000 [===============>..............] - ETA: 50s - loss: 0.2548 - categorical_accuracy: 0.9226
32672/60000 [===============>..............] - ETA: 50s - loss: 0.2547 - categorical_accuracy: 0.9227
32704/60000 [===============>..............] - ETA: 50s - loss: 0.2544 - categorical_accuracy: 0.9227
32736/60000 [===============>..............] - ETA: 50s - loss: 0.2543 - categorical_accuracy: 0.9228
32768/60000 [===============>..............] - ETA: 50s - loss: 0.2542 - categorical_accuracy: 0.9228
32800/60000 [===============>..............] - ETA: 50s - loss: 0.2541 - categorical_accuracy: 0.9228
32832/60000 [===============>..............] - ETA: 50s - loss: 0.2540 - categorical_accuracy: 0.9229
32864/60000 [===============>..............] - ETA: 50s - loss: 0.2538 - categorical_accuracy: 0.9229
32896/60000 [===============>..............] - ETA: 50s - loss: 0.2538 - categorical_accuracy: 0.9229
32928/60000 [===============>..............] - ETA: 50s - loss: 0.2536 - categorical_accuracy: 0.9230
32960/60000 [===============>..............] - ETA: 50s - loss: 0.2534 - categorical_accuracy: 0.9231
32992/60000 [===============>..............] - ETA: 50s - loss: 0.2532 - categorical_accuracy: 0.9231
33024/60000 [===============>..............] - ETA: 49s - loss: 0.2532 - categorical_accuracy: 0.9231
33056/60000 [===============>..............] - ETA: 49s - loss: 0.2531 - categorical_accuracy: 0.9232
33088/60000 [===============>..............] - ETA: 49s - loss: 0.2529 - categorical_accuracy: 0.9232
33120/60000 [===============>..............] - ETA: 49s - loss: 0.2527 - categorical_accuracy: 0.9233
33152/60000 [===============>..............] - ETA: 49s - loss: 0.2525 - categorical_accuracy: 0.9233
33184/60000 [===============>..............] - ETA: 49s - loss: 0.2523 - categorical_accuracy: 0.9234
33216/60000 [===============>..............] - ETA: 49s - loss: 0.2522 - categorical_accuracy: 0.9234
33248/60000 [===============>..............] - ETA: 49s - loss: 0.2520 - categorical_accuracy: 0.9235
33280/60000 [===============>..............] - ETA: 49s - loss: 0.2519 - categorical_accuracy: 0.9235
33312/60000 [===============>..............] - ETA: 49s - loss: 0.2517 - categorical_accuracy: 0.9236
33344/60000 [===============>..............] - ETA: 49s - loss: 0.2517 - categorical_accuracy: 0.9236
33376/60000 [===============>..............] - ETA: 49s - loss: 0.2515 - categorical_accuracy: 0.9236
33408/60000 [===============>..............] - ETA: 49s - loss: 0.2513 - categorical_accuracy: 0.9236
33440/60000 [===============>..............] - ETA: 49s - loss: 0.2513 - categorical_accuracy: 0.9236
33472/60000 [===============>..............] - ETA: 49s - loss: 0.2514 - categorical_accuracy: 0.9236
33504/60000 [===============>..............] - ETA: 49s - loss: 0.2512 - categorical_accuracy: 0.9237
33536/60000 [===============>..............] - ETA: 49s - loss: 0.2509 - categorical_accuracy: 0.9238
33568/60000 [===============>..............] - ETA: 48s - loss: 0.2508 - categorical_accuracy: 0.9238
33600/60000 [===============>..............] - ETA: 48s - loss: 0.2506 - categorical_accuracy: 0.9239
33632/60000 [===============>..............] - ETA: 48s - loss: 0.2504 - categorical_accuracy: 0.9239
33664/60000 [===============>..............] - ETA: 48s - loss: 0.2502 - categorical_accuracy: 0.9240
33696/60000 [===============>..............] - ETA: 48s - loss: 0.2501 - categorical_accuracy: 0.9240
33728/60000 [===============>..............] - ETA: 48s - loss: 0.2499 - categorical_accuracy: 0.9241
33760/60000 [===============>..............] - ETA: 48s - loss: 0.2497 - categorical_accuracy: 0.9241
33792/60000 [===============>..............] - ETA: 48s - loss: 0.2496 - categorical_accuracy: 0.9242
33824/60000 [===============>..............] - ETA: 48s - loss: 0.2495 - categorical_accuracy: 0.9242
33856/60000 [===============>..............] - ETA: 48s - loss: 0.2493 - categorical_accuracy: 0.9243
33888/60000 [===============>..............] - ETA: 48s - loss: 0.2497 - categorical_accuracy: 0.9242
33920/60000 [===============>..............] - ETA: 48s - loss: 0.2494 - categorical_accuracy: 0.9243
33952/60000 [===============>..............] - ETA: 48s - loss: 0.2495 - categorical_accuracy: 0.9243
33984/60000 [===============>..............] - ETA: 48s - loss: 0.2495 - categorical_accuracy: 0.9243
34016/60000 [================>.............] - ETA: 48s - loss: 0.2495 - categorical_accuracy: 0.9244
34048/60000 [================>.............] - ETA: 48s - loss: 0.2493 - categorical_accuracy: 0.9244
34080/60000 [================>.............] - ETA: 48s - loss: 0.2492 - categorical_accuracy: 0.9244
34112/60000 [================>.............] - ETA: 47s - loss: 0.2491 - categorical_accuracy: 0.9244
34144/60000 [================>.............] - ETA: 47s - loss: 0.2490 - categorical_accuracy: 0.9245
34176/60000 [================>.............] - ETA: 47s - loss: 0.2491 - categorical_accuracy: 0.9244
34208/60000 [================>.............] - ETA: 47s - loss: 0.2489 - categorical_accuracy: 0.9245
34240/60000 [================>.............] - ETA: 47s - loss: 0.2491 - categorical_accuracy: 0.9245
34272/60000 [================>.............] - ETA: 47s - loss: 0.2493 - categorical_accuracy: 0.9244
34304/60000 [================>.............] - ETA: 47s - loss: 0.2491 - categorical_accuracy: 0.9245
34336/60000 [================>.............] - ETA: 47s - loss: 0.2490 - categorical_accuracy: 0.9245
34368/60000 [================>.............] - ETA: 47s - loss: 0.2489 - categorical_accuracy: 0.9246
34400/60000 [================>.............] - ETA: 47s - loss: 0.2487 - categorical_accuracy: 0.9246
34432/60000 [================>.............] - ETA: 47s - loss: 0.2486 - categorical_accuracy: 0.9246
34464/60000 [================>.............] - ETA: 47s - loss: 0.2485 - categorical_accuracy: 0.9246
34496/60000 [================>.............] - ETA: 47s - loss: 0.2483 - categorical_accuracy: 0.9247
34528/60000 [================>.............] - ETA: 47s - loss: 0.2484 - categorical_accuracy: 0.9247
34560/60000 [================>.............] - ETA: 47s - loss: 0.2482 - categorical_accuracy: 0.9247
34592/60000 [================>.............] - ETA: 47s - loss: 0.2480 - categorical_accuracy: 0.9248
34624/60000 [================>.............] - ETA: 47s - loss: 0.2479 - categorical_accuracy: 0.9248
34656/60000 [================>.............] - ETA: 46s - loss: 0.2478 - categorical_accuracy: 0.9248
34688/60000 [================>.............] - ETA: 46s - loss: 0.2478 - categorical_accuracy: 0.9248
34720/60000 [================>.............] - ETA: 46s - loss: 0.2477 - categorical_accuracy: 0.9248
34752/60000 [================>.............] - ETA: 46s - loss: 0.2476 - categorical_accuracy: 0.9249
34784/60000 [================>.............] - ETA: 46s - loss: 0.2474 - categorical_accuracy: 0.9249
34816/60000 [================>.............] - ETA: 46s - loss: 0.2472 - categorical_accuracy: 0.9249
34848/60000 [================>.............] - ETA: 46s - loss: 0.2471 - categorical_accuracy: 0.9250
34880/60000 [================>.............] - ETA: 46s - loss: 0.2469 - categorical_accuracy: 0.9250
34912/60000 [================>.............] - ETA: 46s - loss: 0.2467 - categorical_accuracy: 0.9251
34944/60000 [================>.............] - ETA: 46s - loss: 0.2466 - categorical_accuracy: 0.9251
34976/60000 [================>.............] - ETA: 46s - loss: 0.2464 - categorical_accuracy: 0.9251
35008/60000 [================>.............] - ETA: 46s - loss: 0.2465 - categorical_accuracy: 0.9251
35040/60000 [================>.............] - ETA: 46s - loss: 0.2464 - categorical_accuracy: 0.9251
35072/60000 [================>.............] - ETA: 46s - loss: 0.2463 - categorical_accuracy: 0.9252
35104/60000 [================>.............] - ETA: 46s - loss: 0.2461 - categorical_accuracy: 0.9253
35136/60000 [================>.............] - ETA: 46s - loss: 0.2460 - categorical_accuracy: 0.9253
35168/60000 [================>.............] - ETA: 45s - loss: 0.2459 - categorical_accuracy: 0.9253
35200/60000 [================>.............] - ETA: 45s - loss: 0.2457 - categorical_accuracy: 0.9254
35232/60000 [================>.............] - ETA: 45s - loss: 0.2456 - categorical_accuracy: 0.9255
35264/60000 [================>.............] - ETA: 45s - loss: 0.2454 - categorical_accuracy: 0.9255
35296/60000 [================>.............] - ETA: 45s - loss: 0.2452 - categorical_accuracy: 0.9255
35328/60000 [================>.............] - ETA: 45s - loss: 0.2451 - categorical_accuracy: 0.9256
35360/60000 [================>.............] - ETA: 45s - loss: 0.2452 - categorical_accuracy: 0.9255
35392/60000 [================>.............] - ETA: 45s - loss: 0.2451 - categorical_accuracy: 0.9255
35424/60000 [================>.............] - ETA: 45s - loss: 0.2449 - categorical_accuracy: 0.9256
35456/60000 [================>.............] - ETA: 45s - loss: 0.2448 - categorical_accuracy: 0.9256
35488/60000 [================>.............] - ETA: 45s - loss: 0.2447 - categorical_accuracy: 0.9256
35520/60000 [================>.............] - ETA: 45s - loss: 0.2447 - categorical_accuracy: 0.9256
35552/60000 [================>.............] - ETA: 45s - loss: 0.2445 - categorical_accuracy: 0.9257
35584/60000 [================>.............] - ETA: 45s - loss: 0.2443 - categorical_accuracy: 0.9258
35616/60000 [================>.............] - ETA: 45s - loss: 0.2442 - categorical_accuracy: 0.9258
35648/60000 [================>.............] - ETA: 45s - loss: 0.2440 - categorical_accuracy: 0.9259
35680/60000 [================>.............] - ETA: 45s - loss: 0.2438 - categorical_accuracy: 0.9260
35712/60000 [================>.............] - ETA: 44s - loss: 0.2436 - categorical_accuracy: 0.9260
35744/60000 [================>.............] - ETA: 44s - loss: 0.2435 - categorical_accuracy: 0.9261
35776/60000 [================>.............] - ETA: 44s - loss: 0.2434 - categorical_accuracy: 0.9261
35808/60000 [================>.............] - ETA: 44s - loss: 0.2432 - categorical_accuracy: 0.9262
35840/60000 [================>.............] - ETA: 44s - loss: 0.2431 - categorical_accuracy: 0.9262
35872/60000 [================>.............] - ETA: 44s - loss: 0.2428 - categorical_accuracy: 0.9262
35904/60000 [================>.............] - ETA: 44s - loss: 0.2427 - categorical_accuracy: 0.9263
35936/60000 [================>.............] - ETA: 44s - loss: 0.2426 - categorical_accuracy: 0.9263
35968/60000 [================>.............] - ETA: 44s - loss: 0.2428 - categorical_accuracy: 0.9264
36000/60000 [=================>............] - ETA: 44s - loss: 0.2427 - categorical_accuracy: 0.9264
36032/60000 [=================>............] - ETA: 44s - loss: 0.2430 - categorical_accuracy: 0.9263
36064/60000 [=================>............] - ETA: 44s - loss: 0.2428 - categorical_accuracy: 0.9264
36096/60000 [=================>............] - ETA: 44s - loss: 0.2426 - categorical_accuracy: 0.9264
36128/60000 [=================>............] - ETA: 44s - loss: 0.2426 - categorical_accuracy: 0.9264
36160/60000 [=================>............] - ETA: 44s - loss: 0.2425 - categorical_accuracy: 0.9265
36192/60000 [=================>............] - ETA: 44s - loss: 0.2423 - categorical_accuracy: 0.9265
36224/60000 [=================>............] - ETA: 44s - loss: 0.2421 - categorical_accuracy: 0.9266
36256/60000 [=================>............] - ETA: 43s - loss: 0.2419 - categorical_accuracy: 0.9266
36288/60000 [=================>............] - ETA: 43s - loss: 0.2418 - categorical_accuracy: 0.9267
36320/60000 [=================>............] - ETA: 43s - loss: 0.2416 - categorical_accuracy: 0.9267
36352/60000 [=================>............] - ETA: 43s - loss: 0.2417 - categorical_accuracy: 0.9267
36384/60000 [=================>............] - ETA: 43s - loss: 0.2415 - categorical_accuracy: 0.9268
36416/60000 [=================>............] - ETA: 43s - loss: 0.2414 - categorical_accuracy: 0.9268
36448/60000 [=================>............] - ETA: 43s - loss: 0.2417 - categorical_accuracy: 0.9268
36480/60000 [=================>............] - ETA: 43s - loss: 0.2415 - categorical_accuracy: 0.9269
36512/60000 [=================>............] - ETA: 43s - loss: 0.2414 - categorical_accuracy: 0.9269
36544/60000 [=================>............] - ETA: 43s - loss: 0.2412 - categorical_accuracy: 0.9270
36576/60000 [=================>............] - ETA: 43s - loss: 0.2412 - categorical_accuracy: 0.9269
36608/60000 [=================>............] - ETA: 43s - loss: 0.2410 - categorical_accuracy: 0.9270
36640/60000 [=================>............] - ETA: 43s - loss: 0.2412 - categorical_accuracy: 0.9270
36672/60000 [=================>............] - ETA: 43s - loss: 0.2414 - categorical_accuracy: 0.9270
36704/60000 [=================>............] - ETA: 43s - loss: 0.2412 - categorical_accuracy: 0.9270
36736/60000 [=================>............] - ETA: 43s - loss: 0.2410 - categorical_accuracy: 0.9271
36768/60000 [=================>............] - ETA: 43s - loss: 0.2409 - categorical_accuracy: 0.9271
36800/60000 [=================>............] - ETA: 42s - loss: 0.2408 - categorical_accuracy: 0.9271
36832/60000 [=================>............] - ETA: 42s - loss: 0.2407 - categorical_accuracy: 0.9272
36864/60000 [=================>............] - ETA: 42s - loss: 0.2406 - categorical_accuracy: 0.9272
36896/60000 [=================>............] - ETA: 42s - loss: 0.2404 - categorical_accuracy: 0.9273
36928/60000 [=================>............] - ETA: 42s - loss: 0.2403 - categorical_accuracy: 0.9273
36960/60000 [=================>............] - ETA: 42s - loss: 0.2401 - categorical_accuracy: 0.9274
36992/60000 [=================>............] - ETA: 42s - loss: 0.2399 - categorical_accuracy: 0.9274
37024/60000 [=================>............] - ETA: 42s - loss: 0.2398 - categorical_accuracy: 0.9275
37056/60000 [=================>............] - ETA: 42s - loss: 0.2396 - categorical_accuracy: 0.9275
37088/60000 [=================>............] - ETA: 42s - loss: 0.2394 - categorical_accuracy: 0.9276
37120/60000 [=================>............] - ETA: 42s - loss: 0.2393 - categorical_accuracy: 0.9276
37152/60000 [=================>............] - ETA: 42s - loss: 0.2392 - categorical_accuracy: 0.9276
37184/60000 [=================>............] - ETA: 42s - loss: 0.2391 - categorical_accuracy: 0.9277
37216/60000 [=================>............] - ETA: 42s - loss: 0.2390 - categorical_accuracy: 0.9277
37248/60000 [=================>............] - ETA: 42s - loss: 0.2389 - categorical_accuracy: 0.9277
37280/60000 [=================>............] - ETA: 42s - loss: 0.2388 - categorical_accuracy: 0.9278
37312/60000 [=================>............] - ETA: 41s - loss: 0.2386 - categorical_accuracy: 0.9278
37344/60000 [=================>............] - ETA: 41s - loss: 0.2386 - categorical_accuracy: 0.9278
37376/60000 [=================>............] - ETA: 41s - loss: 0.2384 - categorical_accuracy: 0.9279
37408/60000 [=================>............] - ETA: 41s - loss: 0.2382 - categorical_accuracy: 0.9280
37440/60000 [=================>............] - ETA: 41s - loss: 0.2380 - categorical_accuracy: 0.9280
37472/60000 [=================>............] - ETA: 41s - loss: 0.2380 - categorical_accuracy: 0.9280
37504/60000 [=================>............] - ETA: 41s - loss: 0.2378 - categorical_accuracy: 0.9280
37536/60000 [=================>............] - ETA: 41s - loss: 0.2378 - categorical_accuracy: 0.9281
37568/60000 [=================>............] - ETA: 41s - loss: 0.2378 - categorical_accuracy: 0.9281
37600/60000 [=================>............] - ETA: 41s - loss: 0.2376 - categorical_accuracy: 0.9281
37632/60000 [=================>............] - ETA: 41s - loss: 0.2374 - categorical_accuracy: 0.9282
37664/60000 [=================>............] - ETA: 41s - loss: 0.2375 - categorical_accuracy: 0.9282
37696/60000 [=================>............] - ETA: 41s - loss: 0.2374 - categorical_accuracy: 0.9282
37728/60000 [=================>............] - ETA: 41s - loss: 0.2372 - categorical_accuracy: 0.9283
37760/60000 [=================>............] - ETA: 41s - loss: 0.2372 - categorical_accuracy: 0.9283
37792/60000 [=================>............] - ETA: 41s - loss: 0.2370 - categorical_accuracy: 0.9284
37824/60000 [=================>............] - ETA: 41s - loss: 0.2369 - categorical_accuracy: 0.9284
37856/60000 [=================>............] - ETA: 40s - loss: 0.2367 - categorical_accuracy: 0.9285
37888/60000 [=================>............] - ETA: 40s - loss: 0.2365 - categorical_accuracy: 0.9285
37920/60000 [=================>............] - ETA: 40s - loss: 0.2365 - categorical_accuracy: 0.9285
37952/60000 [=================>............] - ETA: 40s - loss: 0.2364 - categorical_accuracy: 0.9286
37984/60000 [=================>............] - ETA: 40s - loss: 0.2362 - categorical_accuracy: 0.9286
38016/60000 [==================>...........] - ETA: 40s - loss: 0.2360 - categorical_accuracy: 0.9286
38048/60000 [==================>...........] - ETA: 40s - loss: 0.2359 - categorical_accuracy: 0.9287
38080/60000 [==================>...........] - ETA: 40s - loss: 0.2357 - categorical_accuracy: 0.9288
38112/60000 [==================>...........] - ETA: 40s - loss: 0.2356 - categorical_accuracy: 0.9288
38144/60000 [==================>...........] - ETA: 40s - loss: 0.2354 - categorical_accuracy: 0.9288
38176/60000 [==================>...........] - ETA: 40s - loss: 0.2354 - categorical_accuracy: 0.9289
38208/60000 [==================>...........] - ETA: 40s - loss: 0.2352 - categorical_accuracy: 0.9289
38240/60000 [==================>...........] - ETA: 40s - loss: 0.2353 - categorical_accuracy: 0.9289
38272/60000 [==================>...........] - ETA: 40s - loss: 0.2352 - categorical_accuracy: 0.9290
38304/60000 [==================>...........] - ETA: 40s - loss: 0.2351 - categorical_accuracy: 0.9290
38336/60000 [==================>...........] - ETA: 40s - loss: 0.2352 - categorical_accuracy: 0.9289
38368/60000 [==================>...........] - ETA: 40s - loss: 0.2351 - categorical_accuracy: 0.9290
38400/60000 [==================>...........] - ETA: 39s - loss: 0.2351 - categorical_accuracy: 0.9290
38432/60000 [==================>...........] - ETA: 39s - loss: 0.2350 - categorical_accuracy: 0.9290
38464/60000 [==================>...........] - ETA: 39s - loss: 0.2348 - categorical_accuracy: 0.9291
38496/60000 [==================>...........] - ETA: 39s - loss: 0.2347 - categorical_accuracy: 0.9291
38528/60000 [==================>...........] - ETA: 39s - loss: 0.2346 - categorical_accuracy: 0.9291
38560/60000 [==================>...........] - ETA: 39s - loss: 0.2344 - categorical_accuracy: 0.9292
38592/60000 [==================>...........] - ETA: 39s - loss: 0.2344 - categorical_accuracy: 0.9292
38624/60000 [==================>...........] - ETA: 39s - loss: 0.2342 - categorical_accuracy: 0.9293
38656/60000 [==================>...........] - ETA: 39s - loss: 0.2341 - categorical_accuracy: 0.9293
38688/60000 [==================>...........] - ETA: 39s - loss: 0.2340 - categorical_accuracy: 0.9293
38720/60000 [==================>...........] - ETA: 39s - loss: 0.2339 - categorical_accuracy: 0.9293
38752/60000 [==================>...........] - ETA: 39s - loss: 0.2337 - categorical_accuracy: 0.9294
38784/60000 [==================>...........] - ETA: 39s - loss: 0.2336 - categorical_accuracy: 0.9294
38816/60000 [==================>...........] - ETA: 39s - loss: 0.2335 - categorical_accuracy: 0.9294
38848/60000 [==================>...........] - ETA: 39s - loss: 0.2335 - categorical_accuracy: 0.9294
38880/60000 [==================>...........] - ETA: 39s - loss: 0.2334 - categorical_accuracy: 0.9294
38912/60000 [==================>...........] - ETA: 39s - loss: 0.2333 - categorical_accuracy: 0.9295
38944/60000 [==================>...........] - ETA: 39s - loss: 0.2332 - categorical_accuracy: 0.9295
38976/60000 [==================>...........] - ETA: 38s - loss: 0.2330 - categorical_accuracy: 0.9296
39008/60000 [==================>...........] - ETA: 38s - loss: 0.2328 - categorical_accuracy: 0.9296
39040/60000 [==================>...........] - ETA: 38s - loss: 0.2328 - categorical_accuracy: 0.9296
39072/60000 [==================>...........] - ETA: 38s - loss: 0.2326 - categorical_accuracy: 0.9297
39104/60000 [==================>...........] - ETA: 38s - loss: 0.2326 - categorical_accuracy: 0.9297
39136/60000 [==================>...........] - ETA: 38s - loss: 0.2326 - categorical_accuracy: 0.9297
39168/60000 [==================>...........] - ETA: 38s - loss: 0.2325 - categorical_accuracy: 0.9297
39200/60000 [==================>...........] - ETA: 38s - loss: 0.2323 - categorical_accuracy: 0.9298
39232/60000 [==================>...........] - ETA: 38s - loss: 0.2322 - categorical_accuracy: 0.9298
39264/60000 [==================>...........] - ETA: 38s - loss: 0.2320 - categorical_accuracy: 0.9298
39296/60000 [==================>...........] - ETA: 38s - loss: 0.2319 - categorical_accuracy: 0.9299
39328/60000 [==================>...........] - ETA: 38s - loss: 0.2318 - categorical_accuracy: 0.9299
39360/60000 [==================>...........] - ETA: 38s - loss: 0.2316 - categorical_accuracy: 0.9300
39392/60000 [==================>...........] - ETA: 38s - loss: 0.2314 - categorical_accuracy: 0.9300
39424/60000 [==================>...........] - ETA: 38s - loss: 0.2314 - categorical_accuracy: 0.9300
39456/60000 [==================>...........] - ETA: 38s - loss: 0.2313 - categorical_accuracy: 0.9301
39488/60000 [==================>...........] - ETA: 38s - loss: 0.2312 - categorical_accuracy: 0.9301
39520/60000 [==================>...........] - ETA: 37s - loss: 0.2311 - categorical_accuracy: 0.9301
39552/60000 [==================>...........] - ETA: 37s - loss: 0.2310 - categorical_accuracy: 0.9301
39584/60000 [==================>...........] - ETA: 37s - loss: 0.2310 - categorical_accuracy: 0.9301
39616/60000 [==================>...........] - ETA: 37s - loss: 0.2308 - categorical_accuracy: 0.9301
39648/60000 [==================>...........] - ETA: 37s - loss: 0.2307 - categorical_accuracy: 0.9301
39680/60000 [==================>...........] - ETA: 37s - loss: 0.2306 - categorical_accuracy: 0.9302
39712/60000 [==================>...........] - ETA: 37s - loss: 0.2305 - categorical_accuracy: 0.9302
39744/60000 [==================>...........] - ETA: 37s - loss: 0.2303 - categorical_accuracy: 0.9302
39776/60000 [==================>...........] - ETA: 37s - loss: 0.2302 - categorical_accuracy: 0.9303
39808/60000 [==================>...........] - ETA: 37s - loss: 0.2302 - categorical_accuracy: 0.9303
39840/60000 [==================>...........] - ETA: 37s - loss: 0.2301 - categorical_accuracy: 0.9303
39872/60000 [==================>...........] - ETA: 37s - loss: 0.2300 - categorical_accuracy: 0.9303
39904/60000 [==================>...........] - ETA: 37s - loss: 0.2300 - categorical_accuracy: 0.9303
39936/60000 [==================>...........] - ETA: 37s - loss: 0.2299 - categorical_accuracy: 0.9304
39968/60000 [==================>...........] - ETA: 37s - loss: 0.2297 - categorical_accuracy: 0.9304
40000/60000 [===================>..........] - ETA: 37s - loss: 0.2296 - categorical_accuracy: 0.9305
40032/60000 [===================>..........] - ETA: 37s - loss: 0.2295 - categorical_accuracy: 0.9305
40064/60000 [===================>..........] - ETA: 36s - loss: 0.2295 - categorical_accuracy: 0.9305
40096/60000 [===================>..........] - ETA: 36s - loss: 0.2293 - categorical_accuracy: 0.9305
40128/60000 [===================>..........] - ETA: 36s - loss: 0.2294 - categorical_accuracy: 0.9305
40160/60000 [===================>..........] - ETA: 36s - loss: 0.2292 - categorical_accuracy: 0.9306
40192/60000 [===================>..........] - ETA: 36s - loss: 0.2292 - categorical_accuracy: 0.9306
40224/60000 [===================>..........] - ETA: 36s - loss: 0.2291 - categorical_accuracy: 0.9306
40256/60000 [===================>..........] - ETA: 36s - loss: 0.2289 - categorical_accuracy: 0.9307
40288/60000 [===================>..........] - ETA: 36s - loss: 0.2288 - categorical_accuracy: 0.9307
40320/60000 [===================>..........] - ETA: 36s - loss: 0.2286 - categorical_accuracy: 0.9308
40352/60000 [===================>..........] - ETA: 36s - loss: 0.2285 - categorical_accuracy: 0.9309
40384/60000 [===================>..........] - ETA: 36s - loss: 0.2283 - categorical_accuracy: 0.9309
40416/60000 [===================>..........] - ETA: 36s - loss: 0.2281 - categorical_accuracy: 0.9310
40448/60000 [===================>..........] - ETA: 36s - loss: 0.2281 - categorical_accuracy: 0.9310
40480/60000 [===================>..........] - ETA: 36s - loss: 0.2279 - categorical_accuracy: 0.9310
40512/60000 [===================>..........] - ETA: 36s - loss: 0.2278 - categorical_accuracy: 0.9310
40544/60000 [===================>..........] - ETA: 36s - loss: 0.2277 - categorical_accuracy: 0.9311
40576/60000 [===================>..........] - ETA: 35s - loss: 0.2275 - categorical_accuracy: 0.9311
40608/60000 [===================>..........] - ETA: 35s - loss: 0.2273 - categorical_accuracy: 0.9312
40640/60000 [===================>..........] - ETA: 35s - loss: 0.2272 - categorical_accuracy: 0.9312
40672/60000 [===================>..........] - ETA: 35s - loss: 0.2272 - categorical_accuracy: 0.9312
40704/60000 [===================>..........] - ETA: 35s - loss: 0.2271 - categorical_accuracy: 0.9313
40736/60000 [===================>..........] - ETA: 35s - loss: 0.2270 - categorical_accuracy: 0.9313
40768/60000 [===================>..........] - ETA: 35s - loss: 0.2268 - categorical_accuracy: 0.9314
40800/60000 [===================>..........] - ETA: 35s - loss: 0.2268 - categorical_accuracy: 0.9314
40832/60000 [===================>..........] - ETA: 35s - loss: 0.2266 - categorical_accuracy: 0.9314
40864/60000 [===================>..........] - ETA: 35s - loss: 0.2265 - categorical_accuracy: 0.9315
40896/60000 [===================>..........] - ETA: 35s - loss: 0.2264 - categorical_accuracy: 0.9315
40928/60000 [===================>..........] - ETA: 35s - loss: 0.2263 - categorical_accuracy: 0.9315
40960/60000 [===================>..........] - ETA: 35s - loss: 0.2263 - categorical_accuracy: 0.9316
40992/60000 [===================>..........] - ETA: 35s - loss: 0.2263 - categorical_accuracy: 0.9316
41024/60000 [===================>..........] - ETA: 35s - loss: 0.2262 - categorical_accuracy: 0.9316
41056/60000 [===================>..........] - ETA: 35s - loss: 0.2262 - categorical_accuracy: 0.9316
41088/60000 [===================>..........] - ETA: 35s - loss: 0.2260 - categorical_accuracy: 0.9316
41120/60000 [===================>..........] - ETA: 34s - loss: 0.2259 - categorical_accuracy: 0.9316
41152/60000 [===================>..........] - ETA: 34s - loss: 0.2258 - categorical_accuracy: 0.9316
41184/60000 [===================>..........] - ETA: 34s - loss: 0.2258 - categorical_accuracy: 0.9316
41216/60000 [===================>..........] - ETA: 34s - loss: 0.2257 - categorical_accuracy: 0.9317
41248/60000 [===================>..........] - ETA: 34s - loss: 0.2256 - categorical_accuracy: 0.9317
41280/60000 [===================>..........] - ETA: 34s - loss: 0.2254 - categorical_accuracy: 0.9317
41312/60000 [===================>..........] - ETA: 34s - loss: 0.2252 - categorical_accuracy: 0.9317
41344/60000 [===================>..........] - ETA: 34s - loss: 0.2251 - categorical_accuracy: 0.9318
41376/60000 [===================>..........] - ETA: 34s - loss: 0.2251 - categorical_accuracy: 0.9318
41408/60000 [===================>..........] - ETA: 34s - loss: 0.2250 - categorical_accuracy: 0.9318
41440/60000 [===================>..........] - ETA: 34s - loss: 0.2249 - categorical_accuracy: 0.9319
41472/60000 [===================>..........] - ETA: 34s - loss: 0.2248 - categorical_accuracy: 0.9319
41504/60000 [===================>..........] - ETA: 34s - loss: 0.2248 - categorical_accuracy: 0.9319
41536/60000 [===================>..........] - ETA: 34s - loss: 0.2246 - categorical_accuracy: 0.9319
41568/60000 [===================>..........] - ETA: 34s - loss: 0.2245 - categorical_accuracy: 0.9320
41600/60000 [===================>..........] - ETA: 34s - loss: 0.2244 - categorical_accuracy: 0.9320
41632/60000 [===================>..........] - ETA: 34s - loss: 0.2243 - categorical_accuracy: 0.9320
41664/60000 [===================>..........] - ETA: 33s - loss: 0.2242 - categorical_accuracy: 0.9320
41696/60000 [===================>..........] - ETA: 33s - loss: 0.2240 - categorical_accuracy: 0.9321
41728/60000 [===================>..........] - ETA: 33s - loss: 0.2239 - categorical_accuracy: 0.9321
41760/60000 [===================>..........] - ETA: 33s - loss: 0.2237 - categorical_accuracy: 0.9321
41792/60000 [===================>..........] - ETA: 33s - loss: 0.2237 - categorical_accuracy: 0.9321
41824/60000 [===================>..........] - ETA: 33s - loss: 0.2236 - categorical_accuracy: 0.9322
41856/60000 [===================>..........] - ETA: 33s - loss: 0.2236 - categorical_accuracy: 0.9322
41888/60000 [===================>..........] - ETA: 33s - loss: 0.2235 - categorical_accuracy: 0.9322
41920/60000 [===================>..........] - ETA: 33s - loss: 0.2234 - categorical_accuracy: 0.9322
41952/60000 [===================>..........] - ETA: 33s - loss: 0.2233 - categorical_accuracy: 0.9323
41984/60000 [===================>..........] - ETA: 33s - loss: 0.2232 - categorical_accuracy: 0.9323
42016/60000 [====================>.........] - ETA: 33s - loss: 0.2230 - categorical_accuracy: 0.9323
42048/60000 [====================>.........] - ETA: 33s - loss: 0.2229 - categorical_accuracy: 0.9324
42080/60000 [====================>.........] - ETA: 33s - loss: 0.2228 - categorical_accuracy: 0.9324
42112/60000 [====================>.........] - ETA: 33s - loss: 0.2228 - categorical_accuracy: 0.9324
42144/60000 [====================>.........] - ETA: 33s - loss: 0.2228 - categorical_accuracy: 0.9324
42176/60000 [====================>.........] - ETA: 33s - loss: 0.2227 - categorical_accuracy: 0.9324
42208/60000 [====================>.........] - ETA: 32s - loss: 0.2226 - categorical_accuracy: 0.9325
42240/60000 [====================>.........] - ETA: 32s - loss: 0.2226 - categorical_accuracy: 0.9325
42272/60000 [====================>.........] - ETA: 32s - loss: 0.2225 - categorical_accuracy: 0.9325
42304/60000 [====================>.........] - ETA: 32s - loss: 0.2224 - categorical_accuracy: 0.9325
42336/60000 [====================>.........] - ETA: 32s - loss: 0.2223 - categorical_accuracy: 0.9325
42368/60000 [====================>.........] - ETA: 32s - loss: 0.2222 - categorical_accuracy: 0.9326
42400/60000 [====================>.........] - ETA: 32s - loss: 0.2222 - categorical_accuracy: 0.9326
42432/60000 [====================>.........] - ETA: 32s - loss: 0.2221 - categorical_accuracy: 0.9326
42464/60000 [====================>.........] - ETA: 32s - loss: 0.2220 - categorical_accuracy: 0.9327
42496/60000 [====================>.........] - ETA: 32s - loss: 0.2219 - categorical_accuracy: 0.9327
42528/60000 [====================>.........] - ETA: 32s - loss: 0.2217 - categorical_accuracy: 0.9328
42560/60000 [====================>.........] - ETA: 32s - loss: 0.2216 - categorical_accuracy: 0.9328
42592/60000 [====================>.........] - ETA: 32s - loss: 0.2214 - categorical_accuracy: 0.9329
42624/60000 [====================>.........] - ETA: 32s - loss: 0.2214 - categorical_accuracy: 0.9329
42656/60000 [====================>.........] - ETA: 32s - loss: 0.2214 - categorical_accuracy: 0.9329
42688/60000 [====================>.........] - ETA: 32s - loss: 0.2213 - categorical_accuracy: 0.9329
42720/60000 [====================>.........] - ETA: 32s - loss: 0.2211 - categorical_accuracy: 0.9329
42752/60000 [====================>.........] - ETA: 31s - loss: 0.2210 - categorical_accuracy: 0.9330
42784/60000 [====================>.........] - ETA: 31s - loss: 0.2210 - categorical_accuracy: 0.9329
42816/60000 [====================>.........] - ETA: 31s - loss: 0.2209 - categorical_accuracy: 0.9330
42848/60000 [====================>.........] - ETA: 31s - loss: 0.2210 - categorical_accuracy: 0.9330
42880/60000 [====================>.........] - ETA: 31s - loss: 0.2209 - categorical_accuracy: 0.9330
42912/60000 [====================>.........] - ETA: 31s - loss: 0.2209 - categorical_accuracy: 0.9331
42944/60000 [====================>.........] - ETA: 31s - loss: 0.2208 - categorical_accuracy: 0.9331
42976/60000 [====================>.........] - ETA: 31s - loss: 0.2207 - categorical_accuracy: 0.9331
43008/60000 [====================>.........] - ETA: 31s - loss: 0.2206 - categorical_accuracy: 0.9331
43040/60000 [====================>.........] - ETA: 31s - loss: 0.2205 - categorical_accuracy: 0.9332
43072/60000 [====================>.........] - ETA: 31s - loss: 0.2204 - categorical_accuracy: 0.9332
43104/60000 [====================>.........] - ETA: 31s - loss: 0.2204 - categorical_accuracy: 0.9332
43136/60000 [====================>.........] - ETA: 31s - loss: 0.2203 - categorical_accuracy: 0.9332
43168/60000 [====================>.........] - ETA: 31s - loss: 0.2201 - categorical_accuracy: 0.9333
43200/60000 [====================>.........] - ETA: 31s - loss: 0.2200 - categorical_accuracy: 0.9333
43232/60000 [====================>.........] - ETA: 31s - loss: 0.2200 - categorical_accuracy: 0.9333
43264/60000 [====================>.........] - ETA: 31s - loss: 0.2199 - categorical_accuracy: 0.9333
43296/60000 [====================>.........] - ETA: 30s - loss: 0.2198 - categorical_accuracy: 0.9334
43328/60000 [====================>.........] - ETA: 30s - loss: 0.2198 - categorical_accuracy: 0.9334
43360/60000 [====================>.........] - ETA: 30s - loss: 0.2198 - categorical_accuracy: 0.9334
43392/60000 [====================>.........] - ETA: 30s - loss: 0.2197 - categorical_accuracy: 0.9335
43424/60000 [====================>.........] - ETA: 30s - loss: 0.2195 - categorical_accuracy: 0.9335
43456/60000 [====================>.........] - ETA: 30s - loss: 0.2195 - categorical_accuracy: 0.9335
43488/60000 [====================>.........] - ETA: 30s - loss: 0.2195 - categorical_accuracy: 0.9335
43520/60000 [====================>.........] - ETA: 30s - loss: 0.2194 - categorical_accuracy: 0.9335
43552/60000 [====================>.........] - ETA: 30s - loss: 0.2192 - categorical_accuracy: 0.9336
43584/60000 [====================>.........] - ETA: 30s - loss: 0.2191 - categorical_accuracy: 0.9336
43616/60000 [====================>.........] - ETA: 30s - loss: 0.2190 - categorical_accuracy: 0.9337
43648/60000 [====================>.........] - ETA: 30s - loss: 0.2189 - categorical_accuracy: 0.9337
43680/60000 [====================>.........] - ETA: 30s - loss: 0.2188 - categorical_accuracy: 0.9337
43712/60000 [====================>.........] - ETA: 30s - loss: 0.2187 - categorical_accuracy: 0.9338
43744/60000 [====================>.........] - ETA: 30s - loss: 0.2185 - categorical_accuracy: 0.9338
43776/60000 [====================>.........] - ETA: 30s - loss: 0.2185 - categorical_accuracy: 0.9338
43808/60000 [====================>.........] - ETA: 30s - loss: 0.2184 - categorical_accuracy: 0.9339
43840/60000 [====================>.........] - ETA: 29s - loss: 0.2183 - categorical_accuracy: 0.9339
43872/60000 [====================>.........] - ETA: 29s - loss: 0.2181 - categorical_accuracy: 0.9339
43904/60000 [====================>.........] - ETA: 29s - loss: 0.2181 - categorical_accuracy: 0.9339
43936/60000 [====================>.........] - ETA: 29s - loss: 0.2179 - categorical_accuracy: 0.9340
43968/60000 [====================>.........] - ETA: 29s - loss: 0.2179 - categorical_accuracy: 0.9340
44000/60000 [=====================>........] - ETA: 29s - loss: 0.2178 - categorical_accuracy: 0.9340
44032/60000 [=====================>........] - ETA: 29s - loss: 0.2179 - categorical_accuracy: 0.9340
44064/60000 [=====================>........] - ETA: 29s - loss: 0.2178 - categorical_accuracy: 0.9340
44096/60000 [=====================>........] - ETA: 29s - loss: 0.2179 - categorical_accuracy: 0.9340
44128/60000 [=====================>........] - ETA: 29s - loss: 0.2178 - categorical_accuracy: 0.9340
44160/60000 [=====================>........] - ETA: 29s - loss: 0.2176 - categorical_accuracy: 0.9341
44192/60000 [=====================>........] - ETA: 29s - loss: 0.2176 - categorical_accuracy: 0.9341
44224/60000 [=====================>........] - ETA: 29s - loss: 0.2175 - categorical_accuracy: 0.9341
44256/60000 [=====================>........] - ETA: 29s - loss: 0.2174 - categorical_accuracy: 0.9341
44288/60000 [=====================>........] - ETA: 29s - loss: 0.2173 - categorical_accuracy: 0.9342
44320/60000 [=====================>........] - ETA: 29s - loss: 0.2173 - categorical_accuracy: 0.9342
44352/60000 [=====================>........] - ETA: 29s - loss: 0.2172 - categorical_accuracy: 0.9342
44384/60000 [=====================>........] - ETA: 28s - loss: 0.2171 - categorical_accuracy: 0.9342
44416/60000 [=====================>........] - ETA: 28s - loss: 0.2170 - categorical_accuracy: 0.9342
44448/60000 [=====================>........] - ETA: 28s - loss: 0.2169 - categorical_accuracy: 0.9343
44480/60000 [=====================>........] - ETA: 28s - loss: 0.2169 - categorical_accuracy: 0.9343
44512/60000 [=====================>........] - ETA: 28s - loss: 0.2168 - categorical_accuracy: 0.9343
44544/60000 [=====================>........] - ETA: 28s - loss: 0.2166 - categorical_accuracy: 0.9344
44576/60000 [=====================>........] - ETA: 28s - loss: 0.2166 - categorical_accuracy: 0.9344
44608/60000 [=====================>........] - ETA: 28s - loss: 0.2165 - categorical_accuracy: 0.9344
44640/60000 [=====================>........] - ETA: 28s - loss: 0.2165 - categorical_accuracy: 0.9345
44672/60000 [=====================>........] - ETA: 28s - loss: 0.2164 - categorical_accuracy: 0.9345
44704/60000 [=====================>........] - ETA: 28s - loss: 0.2163 - categorical_accuracy: 0.9345
44736/60000 [=====================>........] - ETA: 28s - loss: 0.2163 - categorical_accuracy: 0.9345
44768/60000 [=====================>........] - ETA: 28s - loss: 0.2163 - categorical_accuracy: 0.9345
44800/60000 [=====================>........] - ETA: 28s - loss: 0.2162 - categorical_accuracy: 0.9346
44832/60000 [=====================>........] - ETA: 28s - loss: 0.2161 - categorical_accuracy: 0.9346
44864/60000 [=====================>........] - ETA: 28s - loss: 0.2161 - categorical_accuracy: 0.9346
44896/60000 [=====================>........] - ETA: 27s - loss: 0.2162 - categorical_accuracy: 0.9346
44928/60000 [=====================>........] - ETA: 27s - loss: 0.2162 - categorical_accuracy: 0.9346
44960/60000 [=====================>........] - ETA: 27s - loss: 0.2161 - categorical_accuracy: 0.9346
44992/60000 [=====================>........] - ETA: 27s - loss: 0.2160 - categorical_accuracy: 0.9346
45024/60000 [=====================>........] - ETA: 27s - loss: 0.2159 - categorical_accuracy: 0.9346
45056/60000 [=====================>........] - ETA: 27s - loss: 0.2158 - categorical_accuracy: 0.9346
45088/60000 [=====================>........] - ETA: 27s - loss: 0.2157 - categorical_accuracy: 0.9347
45120/60000 [=====================>........] - ETA: 27s - loss: 0.2156 - categorical_accuracy: 0.9347
45152/60000 [=====================>........] - ETA: 27s - loss: 0.2155 - categorical_accuracy: 0.9347
45184/60000 [=====================>........] - ETA: 27s - loss: 0.2154 - categorical_accuracy: 0.9348
45216/60000 [=====================>........] - ETA: 27s - loss: 0.2154 - categorical_accuracy: 0.9348
45248/60000 [=====================>........] - ETA: 27s - loss: 0.2153 - categorical_accuracy: 0.9348
45280/60000 [=====================>........] - ETA: 27s - loss: 0.2152 - categorical_accuracy: 0.9348
45312/60000 [=====================>........] - ETA: 27s - loss: 0.2150 - categorical_accuracy: 0.9349
45344/60000 [=====================>........] - ETA: 27s - loss: 0.2151 - categorical_accuracy: 0.9349
45376/60000 [=====================>........] - ETA: 27s - loss: 0.2151 - categorical_accuracy: 0.9349
45408/60000 [=====================>........] - ETA: 27s - loss: 0.2150 - categorical_accuracy: 0.9349
45440/60000 [=====================>........] - ETA: 26s - loss: 0.2149 - categorical_accuracy: 0.9350
45472/60000 [=====================>........] - ETA: 26s - loss: 0.2148 - categorical_accuracy: 0.9350
45504/60000 [=====================>........] - ETA: 26s - loss: 0.2146 - categorical_accuracy: 0.9351
45536/60000 [=====================>........] - ETA: 26s - loss: 0.2145 - categorical_accuracy: 0.9351
45568/60000 [=====================>........] - ETA: 26s - loss: 0.2145 - categorical_accuracy: 0.9351
45600/60000 [=====================>........] - ETA: 26s - loss: 0.2144 - categorical_accuracy: 0.9351
45632/60000 [=====================>........] - ETA: 26s - loss: 0.2144 - categorical_accuracy: 0.9351
45664/60000 [=====================>........] - ETA: 26s - loss: 0.2143 - categorical_accuracy: 0.9351
45696/60000 [=====================>........] - ETA: 26s - loss: 0.2142 - categorical_accuracy: 0.9352
45728/60000 [=====================>........] - ETA: 26s - loss: 0.2141 - categorical_accuracy: 0.9352
45760/60000 [=====================>........] - ETA: 26s - loss: 0.2140 - categorical_accuracy: 0.9352
45792/60000 [=====================>........] - ETA: 26s - loss: 0.2140 - categorical_accuracy: 0.9352
45824/60000 [=====================>........] - ETA: 26s - loss: 0.2139 - categorical_accuracy: 0.9352
45856/60000 [=====================>........] - ETA: 26s - loss: 0.2137 - categorical_accuracy: 0.9353
45888/60000 [=====================>........] - ETA: 26s - loss: 0.2136 - categorical_accuracy: 0.9353
45920/60000 [=====================>........] - ETA: 26s - loss: 0.2134 - categorical_accuracy: 0.9354
45952/60000 [=====================>........] - ETA: 26s - loss: 0.2134 - categorical_accuracy: 0.9354
45984/60000 [=====================>........] - ETA: 25s - loss: 0.2133 - categorical_accuracy: 0.9353
46016/60000 [======================>.......] - ETA: 25s - loss: 0.2132 - categorical_accuracy: 0.9354
46048/60000 [======================>.......] - ETA: 25s - loss: 0.2131 - categorical_accuracy: 0.9354
46080/60000 [======================>.......] - ETA: 25s - loss: 0.2131 - categorical_accuracy: 0.9354
46112/60000 [======================>.......] - ETA: 25s - loss: 0.2130 - categorical_accuracy: 0.9355
46144/60000 [======================>.......] - ETA: 25s - loss: 0.2129 - categorical_accuracy: 0.9355
46176/60000 [======================>.......] - ETA: 25s - loss: 0.2128 - categorical_accuracy: 0.9356
46208/60000 [======================>.......] - ETA: 25s - loss: 0.2126 - categorical_accuracy: 0.9356
46240/60000 [======================>.......] - ETA: 25s - loss: 0.2125 - categorical_accuracy: 0.9356
46272/60000 [======================>.......] - ETA: 25s - loss: 0.2124 - categorical_accuracy: 0.9356
46304/60000 [======================>.......] - ETA: 25s - loss: 0.2123 - categorical_accuracy: 0.9357
46336/60000 [======================>.......] - ETA: 25s - loss: 0.2122 - categorical_accuracy: 0.9357
46368/60000 [======================>.......] - ETA: 25s - loss: 0.2121 - categorical_accuracy: 0.9358
46400/60000 [======================>.......] - ETA: 25s - loss: 0.2120 - categorical_accuracy: 0.9358
46432/60000 [======================>.......] - ETA: 25s - loss: 0.2120 - categorical_accuracy: 0.9358
46464/60000 [======================>.......] - ETA: 25s - loss: 0.2119 - categorical_accuracy: 0.9358
46496/60000 [======================>.......] - ETA: 25s - loss: 0.2119 - categorical_accuracy: 0.9358
46528/60000 [======================>.......] - ETA: 24s - loss: 0.2119 - categorical_accuracy: 0.9358
46560/60000 [======================>.......] - ETA: 24s - loss: 0.2118 - categorical_accuracy: 0.9358
46592/60000 [======================>.......] - ETA: 24s - loss: 0.2117 - categorical_accuracy: 0.9359
46624/60000 [======================>.......] - ETA: 24s - loss: 0.2115 - categorical_accuracy: 0.9359
46656/60000 [======================>.......] - ETA: 24s - loss: 0.2114 - categorical_accuracy: 0.9360
46688/60000 [======================>.......] - ETA: 24s - loss: 0.2114 - categorical_accuracy: 0.9360
46720/60000 [======================>.......] - ETA: 24s - loss: 0.2113 - categorical_accuracy: 0.9360
46752/60000 [======================>.......] - ETA: 24s - loss: 0.2112 - categorical_accuracy: 0.9360
46784/60000 [======================>.......] - ETA: 24s - loss: 0.2112 - categorical_accuracy: 0.9360
46816/60000 [======================>.......] - ETA: 24s - loss: 0.2111 - categorical_accuracy: 0.9360
46848/60000 [======================>.......] - ETA: 24s - loss: 0.2110 - categorical_accuracy: 0.9361
46880/60000 [======================>.......] - ETA: 24s - loss: 0.2110 - categorical_accuracy: 0.9361
46912/60000 [======================>.......] - ETA: 24s - loss: 0.2109 - categorical_accuracy: 0.9361
46944/60000 [======================>.......] - ETA: 24s - loss: 0.2108 - categorical_accuracy: 0.9362
46976/60000 [======================>.......] - ETA: 24s - loss: 0.2107 - categorical_accuracy: 0.9362
47008/60000 [======================>.......] - ETA: 24s - loss: 0.2105 - categorical_accuracy: 0.9362
47040/60000 [======================>.......] - ETA: 24s - loss: 0.2105 - categorical_accuracy: 0.9362
47072/60000 [======================>.......] - ETA: 23s - loss: 0.2104 - categorical_accuracy: 0.9362
47104/60000 [======================>.......] - ETA: 23s - loss: 0.2104 - categorical_accuracy: 0.9363
47136/60000 [======================>.......] - ETA: 23s - loss: 0.2105 - categorical_accuracy: 0.9362
47168/60000 [======================>.......] - ETA: 23s - loss: 0.2105 - categorical_accuracy: 0.9363
47200/60000 [======================>.......] - ETA: 23s - loss: 0.2104 - categorical_accuracy: 0.9363
47232/60000 [======================>.......] - ETA: 23s - loss: 0.2103 - categorical_accuracy: 0.9363
47264/60000 [======================>.......] - ETA: 23s - loss: 0.2102 - categorical_accuracy: 0.9364
47296/60000 [======================>.......] - ETA: 23s - loss: 0.2100 - categorical_accuracy: 0.9364
47328/60000 [======================>.......] - ETA: 23s - loss: 0.2099 - categorical_accuracy: 0.9365
47360/60000 [======================>.......] - ETA: 23s - loss: 0.2099 - categorical_accuracy: 0.9364
47392/60000 [======================>.......] - ETA: 23s - loss: 0.2099 - categorical_accuracy: 0.9364
47424/60000 [======================>.......] - ETA: 23s - loss: 0.2098 - categorical_accuracy: 0.9365
47456/60000 [======================>.......] - ETA: 23s - loss: 0.2099 - categorical_accuracy: 0.9364
47488/60000 [======================>.......] - ETA: 23s - loss: 0.2098 - categorical_accuracy: 0.9365
47520/60000 [======================>.......] - ETA: 23s - loss: 0.2097 - categorical_accuracy: 0.9365
47552/60000 [======================>.......] - ETA: 23s - loss: 0.2097 - categorical_accuracy: 0.9365
47584/60000 [======================>.......] - ETA: 23s - loss: 0.2096 - categorical_accuracy: 0.9365
47616/60000 [======================>.......] - ETA: 22s - loss: 0.2095 - categorical_accuracy: 0.9365
47648/60000 [======================>.......] - ETA: 22s - loss: 0.2094 - categorical_accuracy: 0.9365
47680/60000 [======================>.......] - ETA: 22s - loss: 0.2093 - categorical_accuracy: 0.9366
47712/60000 [======================>.......] - ETA: 22s - loss: 0.2092 - categorical_accuracy: 0.9366
47744/60000 [======================>.......] - ETA: 22s - loss: 0.2092 - categorical_accuracy: 0.9366
47776/60000 [======================>.......] - ETA: 22s - loss: 0.2091 - categorical_accuracy: 0.9366
47808/60000 [======================>.......] - ETA: 22s - loss: 0.2090 - categorical_accuracy: 0.9366
47840/60000 [======================>.......] - ETA: 22s - loss: 0.2089 - categorical_accuracy: 0.9367
47872/60000 [======================>.......] - ETA: 22s - loss: 0.2088 - categorical_accuracy: 0.9367
47904/60000 [======================>.......] - ETA: 22s - loss: 0.2087 - categorical_accuracy: 0.9367
47936/60000 [======================>.......] - ETA: 22s - loss: 0.2087 - categorical_accuracy: 0.9367
47968/60000 [======================>.......] - ETA: 22s - loss: 0.2086 - categorical_accuracy: 0.9368
48000/60000 [=======================>......] - ETA: 22s - loss: 0.2085 - categorical_accuracy: 0.9368
48032/60000 [=======================>......] - ETA: 22s - loss: 0.2085 - categorical_accuracy: 0.9368
48064/60000 [=======================>......] - ETA: 22s - loss: 0.2083 - categorical_accuracy: 0.9369
48096/60000 [=======================>......] - ETA: 22s - loss: 0.2082 - categorical_accuracy: 0.9369
48128/60000 [=======================>......] - ETA: 22s - loss: 0.2081 - categorical_accuracy: 0.9369
48160/60000 [=======================>......] - ETA: 21s - loss: 0.2080 - categorical_accuracy: 0.9370
48192/60000 [=======================>......] - ETA: 21s - loss: 0.2079 - categorical_accuracy: 0.9370
48224/60000 [=======================>......] - ETA: 21s - loss: 0.2079 - categorical_accuracy: 0.9370
48256/60000 [=======================>......] - ETA: 21s - loss: 0.2078 - categorical_accuracy: 0.9370
48288/60000 [=======================>......] - ETA: 21s - loss: 0.2077 - categorical_accuracy: 0.9371
48320/60000 [=======================>......] - ETA: 21s - loss: 0.2076 - categorical_accuracy: 0.9371
48352/60000 [=======================>......] - ETA: 21s - loss: 0.2075 - categorical_accuracy: 0.9371
48384/60000 [=======================>......] - ETA: 21s - loss: 0.2074 - categorical_accuracy: 0.9372
48416/60000 [=======================>......] - ETA: 21s - loss: 0.2072 - categorical_accuracy: 0.9372
48448/60000 [=======================>......] - ETA: 21s - loss: 0.2071 - categorical_accuracy: 0.9373
48480/60000 [=======================>......] - ETA: 21s - loss: 0.2070 - categorical_accuracy: 0.9373
48512/60000 [=======================>......] - ETA: 21s - loss: 0.2068 - categorical_accuracy: 0.9373
48544/60000 [=======================>......] - ETA: 21s - loss: 0.2068 - categorical_accuracy: 0.9373
48576/60000 [=======================>......] - ETA: 21s - loss: 0.2066 - categorical_accuracy: 0.9374
48608/60000 [=======================>......] - ETA: 21s - loss: 0.2067 - categorical_accuracy: 0.9374
48640/60000 [=======================>......] - ETA: 21s - loss: 0.2066 - categorical_accuracy: 0.9374
48672/60000 [=======================>......] - ETA: 21s - loss: 0.2065 - categorical_accuracy: 0.9374
48704/60000 [=======================>......] - ETA: 20s - loss: 0.2064 - categorical_accuracy: 0.9374
48736/60000 [=======================>......] - ETA: 20s - loss: 0.2064 - categorical_accuracy: 0.9374
48768/60000 [=======================>......] - ETA: 20s - loss: 0.2062 - categorical_accuracy: 0.9375
48800/60000 [=======================>......] - ETA: 20s - loss: 0.2062 - categorical_accuracy: 0.9375
48832/60000 [=======================>......] - ETA: 20s - loss: 0.2061 - categorical_accuracy: 0.9375
48864/60000 [=======================>......] - ETA: 20s - loss: 0.2059 - categorical_accuracy: 0.9376
48896/60000 [=======================>......] - ETA: 20s - loss: 0.2058 - categorical_accuracy: 0.9376
48928/60000 [=======================>......] - ETA: 20s - loss: 0.2058 - categorical_accuracy: 0.9376
48960/60000 [=======================>......] - ETA: 20s - loss: 0.2056 - categorical_accuracy: 0.9376
48992/60000 [=======================>......] - ETA: 20s - loss: 0.2056 - categorical_accuracy: 0.9376
49024/60000 [=======================>......] - ETA: 20s - loss: 0.2055 - categorical_accuracy: 0.9377
49056/60000 [=======================>......] - ETA: 20s - loss: 0.2054 - categorical_accuracy: 0.9377
49088/60000 [=======================>......] - ETA: 20s - loss: 0.2053 - categorical_accuracy: 0.9377
49120/60000 [=======================>......] - ETA: 20s - loss: 0.2053 - categorical_accuracy: 0.9377
49152/60000 [=======================>......] - ETA: 20s - loss: 0.2052 - categorical_accuracy: 0.9377
49184/60000 [=======================>......] - ETA: 20s - loss: 0.2052 - categorical_accuracy: 0.9377
49216/60000 [=======================>......] - ETA: 19s - loss: 0.2052 - categorical_accuracy: 0.9377
49248/60000 [=======================>......] - ETA: 19s - loss: 0.2052 - categorical_accuracy: 0.9377
49280/60000 [=======================>......] - ETA: 19s - loss: 0.2050 - categorical_accuracy: 0.9378
49312/60000 [=======================>......] - ETA: 19s - loss: 0.2049 - categorical_accuracy: 0.9378
49344/60000 [=======================>......] - ETA: 19s - loss: 0.2049 - categorical_accuracy: 0.9378
49376/60000 [=======================>......] - ETA: 19s - loss: 0.2048 - categorical_accuracy: 0.9378
49408/60000 [=======================>......] - ETA: 19s - loss: 0.2047 - categorical_accuracy: 0.9379
49440/60000 [=======================>......] - ETA: 19s - loss: 0.2047 - categorical_accuracy: 0.9379
49472/60000 [=======================>......] - ETA: 19s - loss: 0.2047 - categorical_accuracy: 0.9379
49504/60000 [=======================>......] - ETA: 19s - loss: 0.2047 - categorical_accuracy: 0.9379
49536/60000 [=======================>......] - ETA: 19s - loss: 0.2047 - categorical_accuracy: 0.9378
49568/60000 [=======================>......] - ETA: 19s - loss: 0.2046 - categorical_accuracy: 0.9378
49600/60000 [=======================>......] - ETA: 19s - loss: 0.2045 - categorical_accuracy: 0.9379
49632/60000 [=======================>......] - ETA: 19s - loss: 0.2047 - categorical_accuracy: 0.9379
49664/60000 [=======================>......] - ETA: 19s - loss: 0.2045 - categorical_accuracy: 0.9379
49696/60000 [=======================>......] - ETA: 19s - loss: 0.2045 - categorical_accuracy: 0.9379
49728/60000 [=======================>......] - ETA: 19s - loss: 0.2044 - categorical_accuracy: 0.9379
49760/60000 [=======================>......] - ETA: 18s - loss: 0.2043 - categorical_accuracy: 0.9380
49792/60000 [=======================>......] - ETA: 18s - loss: 0.2043 - categorical_accuracy: 0.9380
49824/60000 [=======================>......] - ETA: 18s - loss: 0.2044 - categorical_accuracy: 0.9380
49856/60000 [=======================>......] - ETA: 18s - loss: 0.2044 - categorical_accuracy: 0.9380
49888/60000 [=======================>......] - ETA: 18s - loss: 0.2043 - categorical_accuracy: 0.9380
49920/60000 [=======================>......] - ETA: 18s - loss: 0.2042 - categorical_accuracy: 0.9380
49952/60000 [=======================>......] - ETA: 18s - loss: 0.2041 - categorical_accuracy: 0.9380
49984/60000 [=======================>......] - ETA: 18s - loss: 0.2040 - categorical_accuracy: 0.9381
50016/60000 [========================>.....] - ETA: 18s - loss: 0.2039 - categorical_accuracy: 0.9381
50048/60000 [========================>.....] - ETA: 18s - loss: 0.2039 - categorical_accuracy: 0.9381
50080/60000 [========================>.....] - ETA: 18s - loss: 0.2038 - categorical_accuracy: 0.9381
50112/60000 [========================>.....] - ETA: 18s - loss: 0.2037 - categorical_accuracy: 0.9382
50144/60000 [========================>.....] - ETA: 18s - loss: 0.2038 - categorical_accuracy: 0.9381
50176/60000 [========================>.....] - ETA: 18s - loss: 0.2038 - categorical_accuracy: 0.9381
50208/60000 [========================>.....] - ETA: 18s - loss: 0.2037 - categorical_accuracy: 0.9382
50240/60000 [========================>.....] - ETA: 18s - loss: 0.2036 - categorical_accuracy: 0.9382
50272/60000 [========================>.....] - ETA: 18s - loss: 0.2035 - categorical_accuracy: 0.9382
50304/60000 [========================>.....] - ETA: 17s - loss: 0.2034 - categorical_accuracy: 0.9382
50336/60000 [========================>.....] - ETA: 17s - loss: 0.2034 - categorical_accuracy: 0.9383
50368/60000 [========================>.....] - ETA: 17s - loss: 0.2033 - categorical_accuracy: 0.9383
50400/60000 [========================>.....] - ETA: 17s - loss: 0.2032 - categorical_accuracy: 0.9383
50432/60000 [========================>.....] - ETA: 17s - loss: 0.2032 - categorical_accuracy: 0.9383
50464/60000 [========================>.....] - ETA: 17s - loss: 0.2031 - categorical_accuracy: 0.9384
50496/60000 [========================>.....] - ETA: 17s - loss: 0.2030 - categorical_accuracy: 0.9384
50528/60000 [========================>.....] - ETA: 17s - loss: 0.2029 - categorical_accuracy: 0.9384
50560/60000 [========================>.....] - ETA: 17s - loss: 0.2028 - categorical_accuracy: 0.9385
50592/60000 [========================>.....] - ETA: 17s - loss: 0.2027 - categorical_accuracy: 0.9385
50624/60000 [========================>.....] - ETA: 17s - loss: 0.2029 - categorical_accuracy: 0.9384
50656/60000 [========================>.....] - ETA: 17s - loss: 0.2028 - categorical_accuracy: 0.9385
50688/60000 [========================>.....] - ETA: 17s - loss: 0.2027 - categorical_accuracy: 0.9385
50720/60000 [========================>.....] - ETA: 17s - loss: 0.2026 - categorical_accuracy: 0.9385
50752/60000 [========================>.....] - ETA: 17s - loss: 0.2025 - categorical_accuracy: 0.9385
50784/60000 [========================>.....] - ETA: 17s - loss: 0.2024 - categorical_accuracy: 0.9386
50816/60000 [========================>.....] - ETA: 17s - loss: 0.2023 - categorical_accuracy: 0.9386
50848/60000 [========================>.....] - ETA: 16s - loss: 0.2023 - categorical_accuracy: 0.9386
50880/60000 [========================>.....] - ETA: 16s - loss: 0.2022 - categorical_accuracy: 0.9386
50912/60000 [========================>.....] - ETA: 16s - loss: 0.2024 - categorical_accuracy: 0.9386
50944/60000 [========================>.....] - ETA: 16s - loss: 0.2023 - categorical_accuracy: 0.9387
50976/60000 [========================>.....] - ETA: 16s - loss: 0.2022 - categorical_accuracy: 0.9387
51008/60000 [========================>.....] - ETA: 16s - loss: 0.2020 - categorical_accuracy: 0.9387
51040/60000 [========================>.....] - ETA: 16s - loss: 0.2020 - categorical_accuracy: 0.9387
51072/60000 [========================>.....] - ETA: 16s - loss: 0.2020 - categorical_accuracy: 0.9387
51104/60000 [========================>.....] - ETA: 16s - loss: 0.2019 - categorical_accuracy: 0.9388
51136/60000 [========================>.....] - ETA: 16s - loss: 0.2018 - categorical_accuracy: 0.9388
51168/60000 [========================>.....] - ETA: 16s - loss: 0.2017 - categorical_accuracy: 0.9388
51200/60000 [========================>.....] - ETA: 16s - loss: 0.2017 - categorical_accuracy: 0.9388
51232/60000 [========================>.....] - ETA: 16s - loss: 0.2016 - categorical_accuracy: 0.9388
51264/60000 [========================>.....] - ETA: 16s - loss: 0.2015 - categorical_accuracy: 0.9389
51296/60000 [========================>.....] - ETA: 16s - loss: 0.2014 - categorical_accuracy: 0.9389
51328/60000 [========================>.....] - ETA: 16s - loss: 0.2013 - categorical_accuracy: 0.9389
51360/60000 [========================>.....] - ETA: 16s - loss: 0.2012 - categorical_accuracy: 0.9390
51392/60000 [========================>.....] - ETA: 15s - loss: 0.2011 - categorical_accuracy: 0.9390
51424/60000 [========================>.....] - ETA: 15s - loss: 0.2011 - categorical_accuracy: 0.9390
51456/60000 [========================>.....] - ETA: 15s - loss: 0.2010 - categorical_accuracy: 0.9391
51488/60000 [========================>.....] - ETA: 15s - loss: 0.2009 - categorical_accuracy: 0.9391
51520/60000 [========================>.....] - ETA: 15s - loss: 0.2009 - categorical_accuracy: 0.9391
51552/60000 [========================>.....] - ETA: 15s - loss: 0.2008 - categorical_accuracy: 0.9391
51584/60000 [========================>.....] - ETA: 15s - loss: 0.2007 - categorical_accuracy: 0.9391
51616/60000 [========================>.....] - ETA: 15s - loss: 0.2007 - categorical_accuracy: 0.9392
51648/60000 [========================>.....] - ETA: 15s - loss: 0.2006 - categorical_accuracy: 0.9392
51680/60000 [========================>.....] - ETA: 15s - loss: 0.2005 - categorical_accuracy: 0.9392
51712/60000 [========================>.....] - ETA: 15s - loss: 0.2004 - categorical_accuracy: 0.9392
51744/60000 [========================>.....] - ETA: 15s - loss: 0.2004 - categorical_accuracy: 0.9392
51776/60000 [========================>.....] - ETA: 15s - loss: 0.2003 - categorical_accuracy: 0.9393
51808/60000 [========================>.....] - ETA: 15s - loss: 0.2002 - categorical_accuracy: 0.9393
51840/60000 [========================>.....] - ETA: 15s - loss: 0.2001 - categorical_accuracy: 0.9393
51872/60000 [========================>.....] - ETA: 15s - loss: 0.2002 - categorical_accuracy: 0.9393
51904/60000 [========================>.....] - ETA: 15s - loss: 0.2001 - categorical_accuracy: 0.9393
51936/60000 [========================>.....] - ETA: 14s - loss: 0.2000 - categorical_accuracy: 0.9394
51968/60000 [========================>.....] - ETA: 14s - loss: 0.2001 - categorical_accuracy: 0.9393
52000/60000 [=========================>....] - ETA: 14s - loss: 0.2001 - categorical_accuracy: 0.9393
52032/60000 [=========================>....] - ETA: 14s - loss: 0.2000 - categorical_accuracy: 0.9394
52064/60000 [=========================>....] - ETA: 14s - loss: 0.1999 - categorical_accuracy: 0.9394
52096/60000 [=========================>....] - ETA: 14s - loss: 0.1998 - categorical_accuracy: 0.9394
52128/60000 [=========================>....] - ETA: 14s - loss: 0.1997 - categorical_accuracy: 0.9395
52160/60000 [=========================>....] - ETA: 14s - loss: 0.1997 - categorical_accuracy: 0.9395
52192/60000 [=========================>....] - ETA: 14s - loss: 0.1997 - categorical_accuracy: 0.9395
52224/60000 [=========================>....] - ETA: 14s - loss: 0.1996 - categorical_accuracy: 0.9395
52256/60000 [=========================>....] - ETA: 14s - loss: 0.1995 - categorical_accuracy: 0.9395
52288/60000 [=========================>....] - ETA: 14s - loss: 0.1994 - categorical_accuracy: 0.9395
52320/60000 [=========================>....] - ETA: 14s - loss: 0.1993 - categorical_accuracy: 0.9395
52352/60000 [=========================>....] - ETA: 14s - loss: 0.1992 - categorical_accuracy: 0.9396
52384/60000 [=========================>....] - ETA: 14s - loss: 0.1991 - categorical_accuracy: 0.9396
52416/60000 [=========================>....] - ETA: 14s - loss: 0.1990 - categorical_accuracy: 0.9397
52448/60000 [=========================>....] - ETA: 13s - loss: 0.1989 - categorical_accuracy: 0.9397
52480/60000 [=========================>....] - ETA: 13s - loss: 0.1988 - categorical_accuracy: 0.9397
52512/60000 [=========================>....] - ETA: 13s - loss: 0.1988 - categorical_accuracy: 0.9397
52544/60000 [=========================>....] - ETA: 13s - loss: 0.1987 - categorical_accuracy: 0.9397
52576/60000 [=========================>....] - ETA: 13s - loss: 0.1986 - categorical_accuracy: 0.9397
52608/60000 [=========================>....] - ETA: 13s - loss: 0.1985 - categorical_accuracy: 0.9398
52640/60000 [=========================>....] - ETA: 13s - loss: 0.1985 - categorical_accuracy: 0.9398
52672/60000 [=========================>....] - ETA: 13s - loss: 0.1986 - categorical_accuracy: 0.9398
52704/60000 [=========================>....] - ETA: 13s - loss: 0.1985 - categorical_accuracy: 0.9398
52736/60000 [=========================>....] - ETA: 13s - loss: 0.1984 - categorical_accuracy: 0.9398
52768/60000 [=========================>....] - ETA: 13s - loss: 0.1984 - categorical_accuracy: 0.9398
52800/60000 [=========================>....] - ETA: 13s - loss: 0.1983 - categorical_accuracy: 0.9399
52832/60000 [=========================>....] - ETA: 13s - loss: 0.1982 - categorical_accuracy: 0.9399
52864/60000 [=========================>....] - ETA: 13s - loss: 0.1982 - categorical_accuracy: 0.9399
52896/60000 [=========================>....] - ETA: 13s - loss: 0.1983 - categorical_accuracy: 0.9399
52928/60000 [=========================>....] - ETA: 13s - loss: 0.1984 - categorical_accuracy: 0.9399
52960/60000 [=========================>....] - ETA: 13s - loss: 0.1983 - categorical_accuracy: 0.9399
52992/60000 [=========================>....] - ETA: 12s - loss: 0.1982 - categorical_accuracy: 0.9399
53024/60000 [=========================>....] - ETA: 12s - loss: 0.1982 - categorical_accuracy: 0.9399
53056/60000 [=========================>....] - ETA: 12s - loss: 0.1981 - categorical_accuracy: 0.9400
53088/60000 [=========================>....] - ETA: 12s - loss: 0.1980 - categorical_accuracy: 0.9400
53120/60000 [=========================>....] - ETA: 12s - loss: 0.1980 - categorical_accuracy: 0.9400
53152/60000 [=========================>....] - ETA: 12s - loss: 0.1979 - categorical_accuracy: 0.9400
53184/60000 [=========================>....] - ETA: 12s - loss: 0.1978 - categorical_accuracy: 0.9400
53216/60000 [=========================>....] - ETA: 12s - loss: 0.1978 - categorical_accuracy: 0.9401
53248/60000 [=========================>....] - ETA: 12s - loss: 0.1976 - categorical_accuracy: 0.9401
53280/60000 [=========================>....] - ETA: 12s - loss: 0.1975 - categorical_accuracy: 0.9401
53312/60000 [=========================>....] - ETA: 12s - loss: 0.1974 - categorical_accuracy: 0.9402
53344/60000 [=========================>....] - ETA: 12s - loss: 0.1975 - categorical_accuracy: 0.9401
53376/60000 [=========================>....] - ETA: 12s - loss: 0.1974 - categorical_accuracy: 0.9402
53408/60000 [=========================>....] - ETA: 12s - loss: 0.1973 - categorical_accuracy: 0.9402
53440/60000 [=========================>....] - ETA: 12s - loss: 0.1973 - categorical_accuracy: 0.9402
53472/60000 [=========================>....] - ETA: 12s - loss: 0.1972 - categorical_accuracy: 0.9402
53504/60000 [=========================>....] - ETA: 12s - loss: 0.1972 - categorical_accuracy: 0.9402
53536/60000 [=========================>....] - ETA: 11s - loss: 0.1971 - categorical_accuracy: 0.9403
53568/60000 [=========================>....] - ETA: 11s - loss: 0.1971 - categorical_accuracy: 0.9403
53600/60000 [=========================>....] - ETA: 11s - loss: 0.1970 - categorical_accuracy: 0.9403
53632/60000 [=========================>....] - ETA: 11s - loss: 0.1971 - categorical_accuracy: 0.9403
53664/60000 [=========================>....] - ETA: 11s - loss: 0.1969 - categorical_accuracy: 0.9403
53696/60000 [=========================>....] - ETA: 11s - loss: 0.1968 - categorical_accuracy: 0.9404
53728/60000 [=========================>....] - ETA: 11s - loss: 0.1967 - categorical_accuracy: 0.9404
53760/60000 [=========================>....] - ETA: 11s - loss: 0.1967 - categorical_accuracy: 0.9404
53792/60000 [=========================>....] - ETA: 11s - loss: 0.1967 - categorical_accuracy: 0.9404
53824/60000 [=========================>....] - ETA: 11s - loss: 0.1965 - categorical_accuracy: 0.9405
53856/60000 [=========================>....] - ETA: 11s - loss: 0.1964 - categorical_accuracy: 0.9405
53888/60000 [=========================>....] - ETA: 11s - loss: 0.1963 - categorical_accuracy: 0.9405
53920/60000 [=========================>....] - ETA: 11s - loss: 0.1962 - categorical_accuracy: 0.9406
53952/60000 [=========================>....] - ETA: 11s - loss: 0.1962 - categorical_accuracy: 0.9406
53984/60000 [=========================>....] - ETA: 11s - loss: 0.1962 - categorical_accuracy: 0.9406
54016/60000 [==========================>...] - ETA: 11s - loss: 0.1962 - categorical_accuracy: 0.9406
54048/60000 [==========================>...] - ETA: 11s - loss: 0.1961 - categorical_accuracy: 0.9406
54080/60000 [==========================>...] - ETA: 10s - loss: 0.1960 - categorical_accuracy: 0.9406
54112/60000 [==========================>...] - ETA: 10s - loss: 0.1959 - categorical_accuracy: 0.9407
54144/60000 [==========================>...] - ETA: 10s - loss: 0.1958 - categorical_accuracy: 0.9407
54176/60000 [==========================>...] - ETA: 10s - loss: 0.1957 - categorical_accuracy: 0.9407
54208/60000 [==========================>...] - ETA: 10s - loss: 0.1957 - categorical_accuracy: 0.9407
54240/60000 [==========================>...] - ETA: 10s - loss: 0.1959 - categorical_accuracy: 0.9407
54272/60000 [==========================>...] - ETA: 10s - loss: 0.1960 - categorical_accuracy: 0.9407
54304/60000 [==========================>...] - ETA: 10s - loss: 0.1959 - categorical_accuracy: 0.9407
54336/60000 [==========================>...] - ETA: 10s - loss: 0.1959 - categorical_accuracy: 0.9407
54368/60000 [==========================>...] - ETA: 10s - loss: 0.1959 - categorical_accuracy: 0.9407
54400/60000 [==========================>...] - ETA: 10s - loss: 0.1958 - categorical_accuracy: 0.9407
54432/60000 [==========================>...] - ETA: 10s - loss: 0.1958 - categorical_accuracy: 0.9407
54464/60000 [==========================>...] - ETA: 10s - loss: 0.1959 - categorical_accuracy: 0.9407
54496/60000 [==========================>...] - ETA: 10s - loss: 0.1958 - categorical_accuracy: 0.9407
54528/60000 [==========================>...] - ETA: 10s - loss: 0.1957 - categorical_accuracy: 0.9408
54560/60000 [==========================>...] - ETA: 10s - loss: 0.1957 - categorical_accuracy: 0.9408
54592/60000 [==========================>...] - ETA: 10s - loss: 0.1956 - categorical_accuracy: 0.9408
54624/60000 [==========================>...] - ETA: 9s - loss: 0.1955 - categorical_accuracy: 0.9408 
54656/60000 [==========================>...] - ETA: 9s - loss: 0.1954 - categorical_accuracy: 0.9408
54688/60000 [==========================>...] - ETA: 9s - loss: 0.1954 - categorical_accuracy: 0.9409
54720/60000 [==========================>...] - ETA: 9s - loss: 0.1953 - categorical_accuracy: 0.9409
54752/60000 [==========================>...] - ETA: 9s - loss: 0.1953 - categorical_accuracy: 0.9409
54784/60000 [==========================>...] - ETA: 9s - loss: 0.1952 - categorical_accuracy: 0.9409
54816/60000 [==========================>...] - ETA: 9s - loss: 0.1951 - categorical_accuracy: 0.9409
54848/60000 [==========================>...] - ETA: 9s - loss: 0.1950 - categorical_accuracy: 0.9409
54880/60000 [==========================>...] - ETA: 9s - loss: 0.1949 - categorical_accuracy: 0.9410
54912/60000 [==========================>...] - ETA: 9s - loss: 0.1948 - categorical_accuracy: 0.9410
54944/60000 [==========================>...] - ETA: 9s - loss: 0.1948 - categorical_accuracy: 0.9410
54976/60000 [==========================>...] - ETA: 9s - loss: 0.1947 - categorical_accuracy: 0.9410
55008/60000 [==========================>...] - ETA: 9s - loss: 0.1946 - categorical_accuracy: 0.9411
55040/60000 [==========================>...] - ETA: 9s - loss: 0.1945 - categorical_accuracy: 0.9411
55072/60000 [==========================>...] - ETA: 9s - loss: 0.1944 - categorical_accuracy: 0.9411
55104/60000 [==========================>...] - ETA: 9s - loss: 0.1944 - categorical_accuracy: 0.9411
55136/60000 [==========================>...] - ETA: 9s - loss: 0.1943 - categorical_accuracy: 0.9412
55168/60000 [==========================>...] - ETA: 8s - loss: 0.1942 - categorical_accuracy: 0.9412
55200/60000 [==========================>...] - ETA: 8s - loss: 0.1941 - categorical_accuracy: 0.9412
55232/60000 [==========================>...] - ETA: 8s - loss: 0.1940 - categorical_accuracy: 0.9413
55264/60000 [==========================>...] - ETA: 8s - loss: 0.1939 - categorical_accuracy: 0.9413
55296/60000 [==========================>...] - ETA: 8s - loss: 0.1938 - categorical_accuracy: 0.9413
55328/60000 [==========================>...] - ETA: 8s - loss: 0.1938 - categorical_accuracy: 0.9413
55360/60000 [==========================>...] - ETA: 8s - loss: 0.1937 - categorical_accuracy: 0.9414
55392/60000 [==========================>...] - ETA: 8s - loss: 0.1937 - categorical_accuracy: 0.9414
55424/60000 [==========================>...] - ETA: 8s - loss: 0.1936 - categorical_accuracy: 0.9414
55456/60000 [==========================>...] - ETA: 8s - loss: 0.1935 - categorical_accuracy: 0.9414
55488/60000 [==========================>...] - ETA: 8s - loss: 0.1934 - categorical_accuracy: 0.9414
55520/60000 [==========================>...] - ETA: 8s - loss: 0.1935 - categorical_accuracy: 0.9414
55552/60000 [==========================>...] - ETA: 8s - loss: 0.1936 - categorical_accuracy: 0.9414
55584/60000 [==========================>...] - ETA: 8s - loss: 0.1935 - categorical_accuracy: 0.9414
55616/60000 [==========================>...] - ETA: 8s - loss: 0.1935 - categorical_accuracy: 0.9415
55648/60000 [==========================>...] - ETA: 8s - loss: 0.1935 - categorical_accuracy: 0.9415
55680/60000 [==========================>...] - ETA: 8s - loss: 0.1935 - categorical_accuracy: 0.9415
55712/60000 [==========================>...] - ETA: 7s - loss: 0.1934 - categorical_accuracy: 0.9415
55744/60000 [==========================>...] - ETA: 7s - loss: 0.1934 - categorical_accuracy: 0.9415
55776/60000 [==========================>...] - ETA: 7s - loss: 0.1933 - categorical_accuracy: 0.9415
55808/60000 [==========================>...] - ETA: 7s - loss: 0.1932 - categorical_accuracy: 0.9415
55840/60000 [==========================>...] - ETA: 7s - loss: 0.1931 - categorical_accuracy: 0.9416
55872/60000 [==========================>...] - ETA: 7s - loss: 0.1930 - categorical_accuracy: 0.9416
55904/60000 [==========================>...] - ETA: 7s - loss: 0.1930 - categorical_accuracy: 0.9416
55936/60000 [==========================>...] - ETA: 7s - loss: 0.1929 - categorical_accuracy: 0.9416
55968/60000 [==========================>...] - ETA: 7s - loss: 0.1930 - categorical_accuracy: 0.9416
56000/60000 [===========================>..] - ETA: 7s - loss: 0.1931 - categorical_accuracy: 0.9416
56032/60000 [===========================>..] - ETA: 7s - loss: 0.1930 - categorical_accuracy: 0.9416
56064/60000 [===========================>..] - ETA: 7s - loss: 0.1931 - categorical_accuracy: 0.9416
56096/60000 [===========================>..] - ETA: 7s - loss: 0.1930 - categorical_accuracy: 0.9416
56128/60000 [===========================>..] - ETA: 7s - loss: 0.1929 - categorical_accuracy: 0.9417
56160/60000 [===========================>..] - ETA: 7s - loss: 0.1929 - categorical_accuracy: 0.9416
56192/60000 [===========================>..] - ETA: 7s - loss: 0.1929 - categorical_accuracy: 0.9416
56224/60000 [===========================>..] - ETA: 6s - loss: 0.1928 - categorical_accuracy: 0.9416
56256/60000 [===========================>..] - ETA: 6s - loss: 0.1927 - categorical_accuracy: 0.9417
56288/60000 [===========================>..] - ETA: 6s - loss: 0.1927 - categorical_accuracy: 0.9417
56320/60000 [===========================>..] - ETA: 6s - loss: 0.1927 - categorical_accuracy: 0.9417
56352/60000 [===========================>..] - ETA: 6s - loss: 0.1926 - categorical_accuracy: 0.9417
56384/60000 [===========================>..] - ETA: 6s - loss: 0.1926 - categorical_accuracy: 0.9417
56416/60000 [===========================>..] - ETA: 6s - loss: 0.1925 - categorical_accuracy: 0.9418
56448/60000 [===========================>..] - ETA: 6s - loss: 0.1925 - categorical_accuracy: 0.9418
56480/60000 [===========================>..] - ETA: 6s - loss: 0.1924 - categorical_accuracy: 0.9418
56512/60000 [===========================>..] - ETA: 6s - loss: 0.1924 - categorical_accuracy: 0.9418
56544/60000 [===========================>..] - ETA: 6s - loss: 0.1923 - categorical_accuracy: 0.9419
56576/60000 [===========================>..] - ETA: 6s - loss: 0.1922 - categorical_accuracy: 0.9419
56608/60000 [===========================>..] - ETA: 6s - loss: 0.1921 - categorical_accuracy: 0.9419
56640/60000 [===========================>..] - ETA: 6s - loss: 0.1920 - categorical_accuracy: 0.9419
56672/60000 [===========================>..] - ETA: 6s - loss: 0.1919 - categorical_accuracy: 0.9420
56704/60000 [===========================>..] - ETA: 6s - loss: 0.1920 - categorical_accuracy: 0.9420
56736/60000 [===========================>..] - ETA: 6s - loss: 0.1919 - categorical_accuracy: 0.9420
56768/60000 [===========================>..] - ETA: 5s - loss: 0.1918 - categorical_accuracy: 0.9420
56800/60000 [===========================>..] - ETA: 5s - loss: 0.1917 - categorical_accuracy: 0.9420
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1918 - categorical_accuracy: 0.9420
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1919 - categorical_accuracy: 0.9420
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1918 - categorical_accuracy: 0.9421
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1917 - categorical_accuracy: 0.9421
56960/60000 [===========================>..] - ETA: 5s - loss: 0.1917 - categorical_accuracy: 0.9421
56992/60000 [===========================>..] - ETA: 5s - loss: 0.1916 - categorical_accuracy: 0.9421
57024/60000 [===========================>..] - ETA: 5s - loss: 0.1915 - categorical_accuracy: 0.9421
57056/60000 [===========================>..] - ETA: 5s - loss: 0.1915 - categorical_accuracy: 0.9421
57088/60000 [===========================>..] - ETA: 5s - loss: 0.1914 - categorical_accuracy: 0.9422
57120/60000 [===========================>..] - ETA: 5s - loss: 0.1914 - categorical_accuracy: 0.9422
57152/60000 [===========================>..] - ETA: 5s - loss: 0.1913 - categorical_accuracy: 0.9422
57184/60000 [===========================>..] - ETA: 5s - loss: 0.1913 - categorical_accuracy: 0.9422
57216/60000 [===========================>..] - ETA: 5s - loss: 0.1913 - categorical_accuracy: 0.9422
57248/60000 [===========================>..] - ETA: 5s - loss: 0.1912 - categorical_accuracy: 0.9422
57280/60000 [===========================>..] - ETA: 5s - loss: 0.1912 - categorical_accuracy: 0.9422
57312/60000 [===========================>..] - ETA: 4s - loss: 0.1911 - categorical_accuracy: 0.9423
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1910 - categorical_accuracy: 0.9423
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1909 - categorical_accuracy: 0.9423
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1908 - categorical_accuracy: 0.9423
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1908 - categorical_accuracy: 0.9424
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1907 - categorical_accuracy: 0.9424
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1907 - categorical_accuracy: 0.9424
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1907 - categorical_accuracy: 0.9424
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1906 - categorical_accuracy: 0.9424
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1906 - categorical_accuracy: 0.9424
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1905 - categorical_accuracy: 0.9424
57664/60000 [===========================>..] - ETA: 4s - loss: 0.1905 - categorical_accuracy: 0.9424
57696/60000 [===========================>..] - ETA: 4s - loss: 0.1905 - categorical_accuracy: 0.9424
57728/60000 [===========================>..] - ETA: 4s - loss: 0.1904 - categorical_accuracy: 0.9424
57760/60000 [===========================>..] - ETA: 4s - loss: 0.1905 - categorical_accuracy: 0.9424
57792/60000 [===========================>..] - ETA: 4s - loss: 0.1905 - categorical_accuracy: 0.9424
57824/60000 [===========================>..] - ETA: 4s - loss: 0.1904 - categorical_accuracy: 0.9424
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1903 - categorical_accuracy: 0.9425
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1903 - categorical_accuracy: 0.9425
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1902 - categorical_accuracy: 0.9425
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1902 - categorical_accuracy: 0.9425
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1901 - categorical_accuracy: 0.9425
58016/60000 [============================>.] - ETA: 3s - loss: 0.1900 - categorical_accuracy: 0.9426
58048/60000 [============================>.] - ETA: 3s - loss: 0.1899 - categorical_accuracy: 0.9426
58080/60000 [============================>.] - ETA: 3s - loss: 0.1898 - categorical_accuracy: 0.9426
58112/60000 [============================>.] - ETA: 3s - loss: 0.1897 - categorical_accuracy: 0.9426
58144/60000 [============================>.] - ETA: 3s - loss: 0.1897 - categorical_accuracy: 0.9426
58176/60000 [============================>.] - ETA: 3s - loss: 0.1896 - categorical_accuracy: 0.9427
58208/60000 [============================>.] - ETA: 3s - loss: 0.1895 - categorical_accuracy: 0.9427
58240/60000 [============================>.] - ETA: 3s - loss: 0.1896 - categorical_accuracy: 0.9427
58272/60000 [============================>.] - ETA: 3s - loss: 0.1895 - categorical_accuracy: 0.9427
58304/60000 [============================>.] - ETA: 3s - loss: 0.1894 - categorical_accuracy: 0.9427
58336/60000 [============================>.] - ETA: 3s - loss: 0.1893 - categorical_accuracy: 0.9428
58368/60000 [============================>.] - ETA: 3s - loss: 0.1892 - categorical_accuracy: 0.9428
58400/60000 [============================>.] - ETA: 2s - loss: 0.1891 - categorical_accuracy: 0.9428
58432/60000 [============================>.] - ETA: 2s - loss: 0.1891 - categorical_accuracy: 0.9429
58464/60000 [============================>.] - ETA: 2s - loss: 0.1891 - categorical_accuracy: 0.9429
58496/60000 [============================>.] - ETA: 2s - loss: 0.1891 - categorical_accuracy: 0.9429
58528/60000 [============================>.] - ETA: 2s - loss: 0.1890 - categorical_accuracy: 0.9429
58560/60000 [============================>.] - ETA: 2s - loss: 0.1889 - categorical_accuracy: 0.9429
58592/60000 [============================>.] - ETA: 2s - loss: 0.1888 - categorical_accuracy: 0.9429
58624/60000 [============================>.] - ETA: 2s - loss: 0.1888 - categorical_accuracy: 0.9429
58656/60000 [============================>.] - ETA: 2s - loss: 0.1888 - categorical_accuracy: 0.9429
58688/60000 [============================>.] - ETA: 2s - loss: 0.1888 - categorical_accuracy: 0.9429
58720/60000 [============================>.] - ETA: 2s - loss: 0.1888 - categorical_accuracy: 0.9429
58752/60000 [============================>.] - ETA: 2s - loss: 0.1889 - categorical_accuracy: 0.9429
58784/60000 [============================>.] - ETA: 2s - loss: 0.1889 - categorical_accuracy: 0.9429
58816/60000 [============================>.] - ETA: 2s - loss: 0.1891 - categorical_accuracy: 0.9429
58848/60000 [============================>.] - ETA: 2s - loss: 0.1890 - categorical_accuracy: 0.9429
58880/60000 [============================>.] - ETA: 2s - loss: 0.1890 - categorical_accuracy: 0.9430
58912/60000 [============================>.] - ETA: 2s - loss: 0.1889 - categorical_accuracy: 0.9430
58944/60000 [============================>.] - ETA: 1s - loss: 0.1888 - categorical_accuracy: 0.9430
58976/60000 [============================>.] - ETA: 1s - loss: 0.1887 - categorical_accuracy: 0.9430
59008/60000 [============================>.] - ETA: 1s - loss: 0.1886 - categorical_accuracy: 0.9431
59040/60000 [============================>.] - ETA: 1s - loss: 0.1886 - categorical_accuracy: 0.9431
59072/60000 [============================>.] - ETA: 1s - loss: 0.1885 - categorical_accuracy: 0.9431
59104/60000 [============================>.] - ETA: 1s - loss: 0.1884 - categorical_accuracy: 0.9432
59136/60000 [============================>.] - ETA: 1s - loss: 0.1883 - categorical_accuracy: 0.9432
59168/60000 [============================>.] - ETA: 1s - loss: 0.1882 - categorical_accuracy: 0.9432
59200/60000 [============================>.] - ETA: 1s - loss: 0.1882 - categorical_accuracy: 0.9432
59232/60000 [============================>.] - ETA: 1s - loss: 0.1881 - categorical_accuracy: 0.9432
59264/60000 [============================>.] - ETA: 1s - loss: 0.1882 - categorical_accuracy: 0.9432
59296/60000 [============================>.] - ETA: 1s - loss: 0.1882 - categorical_accuracy: 0.9432
59328/60000 [============================>.] - ETA: 1s - loss: 0.1882 - categorical_accuracy: 0.9432
59360/60000 [============================>.] - ETA: 1s - loss: 0.1881 - categorical_accuracy: 0.9432
59392/60000 [============================>.] - ETA: 1s - loss: 0.1881 - categorical_accuracy: 0.9433
59424/60000 [============================>.] - ETA: 1s - loss: 0.1881 - categorical_accuracy: 0.9433
59456/60000 [============================>.] - ETA: 1s - loss: 0.1880 - categorical_accuracy: 0.9433
59488/60000 [============================>.] - ETA: 0s - loss: 0.1879 - categorical_accuracy: 0.9433
59520/60000 [============================>.] - ETA: 0s - loss: 0.1878 - categorical_accuracy: 0.9434
59552/60000 [============================>.] - ETA: 0s - loss: 0.1878 - categorical_accuracy: 0.9434
59584/60000 [============================>.] - ETA: 0s - loss: 0.1877 - categorical_accuracy: 0.9434
59616/60000 [============================>.] - ETA: 0s - loss: 0.1876 - categorical_accuracy: 0.9434
59648/60000 [============================>.] - ETA: 0s - loss: 0.1876 - categorical_accuracy: 0.9434
59680/60000 [============================>.] - ETA: 0s - loss: 0.1875 - categorical_accuracy: 0.9434
59712/60000 [============================>.] - ETA: 0s - loss: 0.1875 - categorical_accuracy: 0.9434
59744/60000 [============================>.] - ETA: 0s - loss: 0.1874 - categorical_accuracy: 0.9435
59776/60000 [============================>.] - ETA: 0s - loss: 0.1873 - categorical_accuracy: 0.9435
59808/60000 [============================>.] - ETA: 0s - loss: 0.1874 - categorical_accuracy: 0.9435
59840/60000 [============================>.] - ETA: 0s - loss: 0.1873 - categorical_accuracy: 0.9435
59872/60000 [============================>.] - ETA: 0s - loss: 0.1872 - categorical_accuracy: 0.9435
59904/60000 [============================>.] - ETA: 0s - loss: 0.1874 - categorical_accuracy: 0.9435
59936/60000 [============================>.] - ETA: 0s - loss: 0.1873 - categorical_accuracy: 0.9435
59968/60000 [============================>.] - ETA: 0s - loss: 0.1873 - categorical_accuracy: 0.9435
60000/60000 [==============================] - 115s 2ms/step - loss: 0.1872 - categorical_accuracy: 0.9435 - val_loss: 0.0464 - val_categorical_accuracy: 0.9838

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 16s
  192/10000 [..............................] - ETA: 5s 
  320/10000 [..............................] - ETA: 5s
  480/10000 [>.............................] - ETA: 4s
  640/10000 [>.............................] - ETA: 4s
  800/10000 [=>............................] - ETA: 3s
  960/10000 [=>............................] - ETA: 3s
 1120/10000 [==>...........................] - ETA: 3s
 1248/10000 [==>...........................] - ETA: 3s
 1408/10000 [===>..........................] - ETA: 3s
 1568/10000 [===>..........................] - ETA: 3s
 1728/10000 [====>.........................] - ETA: 3s
 1888/10000 [====>.........................] - ETA: 3s
 2048/10000 [=====>........................] - ETA: 3s
 2208/10000 [=====>........................] - ETA: 2s
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
 4608/10000 [============>.................] - ETA: 1s
 4768/10000 [=============>................] - ETA: 1s
 4928/10000 [=============>................] - ETA: 1s
 5088/10000 [==============>...............] - ETA: 1s
 5248/10000 [==============>...............] - ETA: 1s
 5408/10000 [===============>..............] - ETA: 1s
 5568/10000 [===============>..............] - ETA: 1s
 5728/10000 [================>.............] - ETA: 1s
 5888/10000 [================>.............] - ETA: 1s
 6048/10000 [=================>............] - ETA: 1s
 6208/10000 [=================>............] - ETA: 1s
 6368/10000 [==================>...........] - ETA: 1s
 6528/10000 [==================>...........] - ETA: 1s
 6688/10000 [===================>..........] - ETA: 1s
 6848/10000 [===================>..........] - ETA: 1s
 7008/10000 [====================>.........] - ETA: 1s
 7168/10000 [====================>.........] - ETA: 1s
 7328/10000 [====================>.........] - ETA: 0s
 7488/10000 [=====================>........] - ETA: 0s
 7648/10000 [=====================>........] - ETA: 0s
 7808/10000 [======================>.......] - ETA: 0s
 7968/10000 [======================>.......] - ETA: 0s
 8128/10000 [=======================>......] - ETA: 0s
 8288/10000 [=======================>......] - ETA: 0s
 8448/10000 [========================>.....] - ETA: 0s
 8608/10000 [========================>.....] - ETA: 0s
 8768/10000 [=========================>....] - ETA: 0s
 8928/10000 [=========================>....] - ETA: 0s
 9056/10000 [==========================>...] - ETA: 0s
 9216/10000 [==========================>...] - ETA: 0s
 9376/10000 [===========================>..] - ETA: 0s
 9536/10000 [===========================>..] - ETA: 0s
 9696/10000 [============================>.] - ETA: 0s
 9856/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 4s 357us/step
[[5.1014261e-08 1.7572885e-08 1.3402486e-06 ... 9.9999046e-01
  3.1680489e-08 1.8039075e-06]
 [1.7323311e-04 8.6133277e-06 9.9974650e-01 ... 2.0183464e-07
  2.0202049e-05 2.6191702e-09]
 [8.6617074e-06 9.9942076e-01 8.4280939e-05 ... 1.5025777e-04
  1.3383843e-04 3.9640649e-06]
 ...
 [8.2287437e-07 8.3703371e-06 3.4210186e-07 ... 6.8882633e-05
  1.8514083e-05 2.9952938e-04]
 [2.6901675e-06 2.0200322e-07 2.6178779e-07 ... 1.4242389e-07
  3.1200569e-04 1.4881736e-05]
 [5.0323091e-05 1.5960482e-06 3.6947902e-05 ... 2.7638583e-08
  2.7492108e-06 1.2863786e-07]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.04638703968073241, 'accuracy_test:': 0.9837999939918518}

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
[master 3b68d73] ml_store  && git pull --all
 1 file changed, 2036 insertions(+)
To github.com:arita37/mlmodels_store.git
 ! [remote rejected] master -> master (cannot lock ref 'refs/heads/master': is at 10b44c5e0edeb25f8a0030d24162faeb9840154d but expected fb12526c6b16b3662d5efd14e2386ff579142ebf)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'





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
{'loss': 0.4850614219903946, 'loss_history': []}

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
[master 9a8549b] ml_store  && git pull --all
 1 file changed, 112 insertions(+)
To github.com:arita37/mlmodels_store.git
 + 10b44c5...9a8549b master -> master (forced update)





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
[master 2a5078f] ml_store  && git pull --all
 1 file changed, 34 insertions(+)
To github.com:arita37/mlmodels_store.git
   9a8549b..2a5078f  master -> master





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
 40%|████      | 2/5 [00:21<00:31, 10.59s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.8417656029293349, 'learning_rate': 0.00655542373249228, 'min_data_in_leaf': 28, 'num_leaves': 58} and reward: 0.3832
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xea\xef\xbej\xefU\xbbX\r\x00\x00\x00learning_rateq\x02G?z\xd9\xdc(\xad\xfb\xf0X\x10\x00\x00\x00min_data_in_leafq\x03K\x1cX\n\x00\x00\x00num_leavesq\x04K:u.' and reward: 0.3832
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xea\xef\xbej\xefU\xbbX\r\x00\x00\x00learning_rateq\x02G?z\xd9\xdc(\xad\xfb\xf0X\x10\x00\x00\x00min_data_in_leafq\x03K\x1cX\n\x00\x00\x00num_leavesq\x04K:u.' and reward: 0.3832
 60%|██████    | 3/5 [00:50<00:32, 16.34s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.9437796858896423, 'learning_rate': 0.020311129939778735, 'min_data_in_leaf': 6, 'num_leaves': 39} and reward: 0.3922
Finished Task with config: b"\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xee3qt\xb0\xce*X\r\x00\x00\x00learning_rateq\x02G?\x94\xccp\xdbX9OX\x10\x00\x00\x00min_data_in_leafq\x03K\x06X\n\x00\x00\x00num_leavesq\x04K'u." and reward: 0.3922
Finished Task with config: b"\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xee3qt\xb0\xce*X\r\x00\x00\x00learning_rateq\x02G?\x94\xccp\xdbX9OX\x10\x00\x00\x00min_data_in_leafq\x03K\x06X\n\x00\x00\x00num_leavesq\x04K'u." and reward: 0.3922
 80%|████████  | 4/5 [01:13<00:18, 18.31s/it] 80%|████████  | 4/5 [01:13<00:18, 18.46s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.9497076501544541, 'learning_rate': 0.04390163835867459, 'min_data_in_leaf': 9, 'num_leaves': 42} and reward: 0.3948
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeed\x01LE\x94\x9aX\r\x00\x00\x00learning_rateq\x02G?\xa6zF\x89\xfb\x8f%X\x10\x00\x00\x00min_data_in_leafq\x03K\tX\n\x00\x00\x00num_leavesq\x04K*u.' and reward: 0.3948
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeed\x01LE\x94\x9aX\r\x00\x00\x00learning_rateq\x02G?\xa6zF\x89\xfb\x8f%X\x10\x00\x00\x00min_data_in_leafq\x03K\tX\n\x00\x00\x00num_leavesq\x04K*u.' and reward: 0.3948

Time for Gradient Boosting hyperparameter optimization: 97.88580012321472
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.9497076501544541, 'learning_rate': 0.04390163835867459, 'min_data_in_leaf': 9, 'num_leaves': 42}
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
 40%|████      | 2/5 [00:50<01:16, 25.49s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.04455392573474277, 'embedding_size_factor': 0.7077797252513578, 'layers.choice': 1, 'learning_rate': 0.002714288967545856, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 9.23992690230209e-07} and reward: 0.3946
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xa6\xcf\xc5\xab\xe0\xd6\tX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe6\xa6!\xaa\x97?\x0eX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?f<F\xcb\x1e\xbe\xbbX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xaf\x01\ti\xed>\x1bu.' and reward: 0.3946
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xa6\xcf\xc5\xab\xe0\xd6\tX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe6\xa6!\xaa\x97?\x0eX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?f<F\xcb\x1e\xbe\xbbX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xaf\x01\ti\xed>\x1bu.' and reward: 0.3946
 60%|██████    | 3/5 [01:59<01:16, 38.47s/it] 60%|██████    | 3/5 [01:59<01:19, 39.91s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.3317600138545141, 'embedding_size_factor': 0.7356635934369176, 'layers.choice': 2, 'learning_rate': 0.0031096256213106166, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 2.3712504342997544e-08} and reward: 0.3876
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd5;\x8eZh\n\x91X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe7\x8a\x8e`Uk\xbcX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?iy[\x8b\x15({X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>Yv\x0b&\xe4\x80 u.' and reward: 0.3876
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd5;\x8eZh\n\x91X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe7\x8a\x8e`Uk\xbcX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?iy[\x8b\x15({X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>Yv\x0b&\xe4\x80 u.' and reward: 0.3876
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 171.4141731262207
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.04455392573474277, 'embedding_size_factor': 0.7077797252513578, 'layers.choice': 1, 'learning_rate': 0.002714288967545856, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 9.23992690230209e-07}
Saving dataset/models/trainer.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.76s of the -153.89s of remaining time.
Ensemble size: 98
Ensemble weights: 
[0.03061224 0.31632653 0.13265306 0.09183673 0.07142857 0.09183673
 0.26530612]
	0.4006	 = Validation accuracy score
	1.61s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 275.55s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f5e11755da0>

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
[master 6653379] ml_store  && git pull --all
 1 file changed, 208 insertions(+)
To github.com:arita37/mlmodels_store.git
 + e267ed9...6653379 master -> master (forced update)





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
[master 3213580] ml_store  && git pull --all
 1 file changed, 34 insertions(+)
To github.com:arita37/mlmodels_store.git
   6653379..3213580  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon//gluonts_model_old.py 
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

  model_gluon.gluonts_model_old 
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
100%|██████████| 10/10 [00:02<00:00,  3.84it/s, avg_epoch_loss=5.25]
INFO:root:Epoch[0] Elapsed time 2.605 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.250747
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.2507470607757565 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model_old' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model_old.py'> <mlmodels.model_gluon.gluonts_model_old.Model object at 0x7f32dc7443c8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model_old.Model object at 0x7f32dc7443c8>

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
Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 101.61it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 1069.0757649739583,
    "abs_error": 371.38720703125,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 2.4607929982683263,
    "sMAPE": 0.5138812107171418,
    "MSIS": 98.43171346007892,
    "QuantileLoss[0.5]": 371.38719177246094,
    "Coverage[0.5]": 1.0,
    "RMSE": 32.69672407098238,
    "NRMSE": 0.6883520857048921,
    "ND": 0.6515565035635965,
    "wQuantileLoss[0.5]": 0.6515564767937911,
    "mean_wQuantileLoss": 0.6515564767937911,
    "MAE_Coverage": 0.5
}

  #### Plot   ####################################################### 

  #### Loading params   ############################################## 

  model_gluon.gluonts_model_old 
{'model_name': 'deepfactor', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_hidden_global': 50, 'num_layers_global': 1, 'num_factors': 10, 'num_hidden_local': 5, 'num_layers_local': 1, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'embedding_dimension': 10}, '_comment': {'distr_output': 'StudentTOutput()', 'cardinality': 'List[int] = list([1])', 'context_length': 'None'}} {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}} {'path': 'ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]}

  #### Loading dataset   ############################################# 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### Model init, fit   ############################################# 

INFO:root:Using CPU
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in DeepFactorTrainingNetwork: 12466
100%|██████████| 10/10 [00:01<00:00,  7.76it/s, avg_epoch_loss=2.71e+3]
INFO:root:Epoch[0] Elapsed time 1.289 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=2713.411247
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 2713.4112467447917 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model_old' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model_old.py'> <mlmodels.model_gluon.gluonts_model_old.Model object at 0x7f32b026bd30>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model_old.Model object at 0x7f32b026bd30>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 131.95it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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

  model_gluon.gluonts_model_old 
{'model_name': 'transformer', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}} {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}} {'path': 'ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]}

  #### Loading dataset   ############################################# 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### Model init, fit   ############################################# 

INFO:root:Using CPU
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in TransformerTrainingNetwork: 33911
100%|██████████| 10/10 [00:01<00:00,  5.10it/s, avg_epoch_loss=5.25]
INFO:root:Epoch[0] Elapsed time 1.960 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.252879
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.25287880897522 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model_old' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model_old.py'> <mlmodels.model_gluon.gluonts_model_old.Model object at 0x7f32b01d6668>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model_old.Model object at 0x7f32b01d6668>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 142.15it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 232.7195027669271,
    "abs_error": 161.52691650390625,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.0702692436340144,
    "sMAPE": 0.2723951184292836,
    "MSIS": 42.81076651003351,
    "QuantileLoss[0.5]": 161.52692031860352,
    "Coverage[0.5]": 0.75,
    "RMSE": 15.255146763205103,
    "NRMSE": 0.3211609844885285,
    "ND": 0.283380555270011,
    "wQuantileLoss[0.5]": 0.2833805619624623,
    "mean_wQuantileLoss": 0.2833805619624623,
    "MAE_Coverage": 0.25
}

  #### Plot   ####################################################### 

  #### Loading params   ############################################## 

  model_gluon.gluonts_model_old 
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
 30%|███       | 3/10 [00:12<00:29,  4.23s/it, avg_epoch_loss=6.93] 60%|██████    | 6/10 [00:23<00:16,  4.07s/it, avg_epoch_loss=6.9]  90%|█████████ | 9/10 [00:34<00:03,  3.96s/it, avg_epoch_loss=6.87]100%|██████████| 10/10 [00:38<00:00,  3.86s/it, avg_epoch_loss=6.86]
INFO:root:Epoch[0] Elapsed time 38.645 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.857960
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.85796046257019 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model_old' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model_old.py'> <mlmodels.model_gluon.gluonts_model_old.Model object at 0x7f32b014a198>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model_old.Model object at 0x7f32b014a198>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 159.36it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 53039.708333333336,
    "abs_error": 2709.94775390625,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 17.955977837234315,
    "sMAPE": 1.410798063537338,
    "MSIS": 718.2389840762902,
    "QuantileLoss[0.5]": 2709.9474334716797,
    "Coverage[0.5]": 1.0,
    "RMSE": 230.3035135062714,
    "NRMSE": 4.848495021184661,
    "ND": 4.754294305098684,
    "wQuantileLoss[0.5]": 4.754293742932771,
    "mean_wQuantileLoss": 4.754293742932771,
    "MAE_Coverage": 0.5
}

  #### Plot   ####################################################### 

  #### Loading params   ############################################## 

  model_gluon.gluonts_model_old 
{'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}} {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}} {'path': 'ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}

  #### Loading dataset   ############################################# 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### Model init, fit   ############################################# 

INFO:root:Using CPU
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in SimpleFeedForwardTrainingNetwork: 20323
100%|██████████| 10/10 [00:00<00:00, 55.72it/s, avg_epoch_loss=5.16]
INFO:root:Epoch[0] Elapsed time 0.180 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.163656
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.1636559009552006 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model_old' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model_old.py'> <mlmodels.model_gluon.gluonts_model_old.Model object at 0x7f32a0f925c0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model_old.Model object at 0x7f32a0f925c0>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 145.06it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 541.433837890625,
    "abs_error": 195.23114013671875,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.2935917382096447,
    "sMAPE": 0.3250679338426852,
    "MSIS": 51.743669528385794,
    "QuantileLoss[0.5]": 195.23114395141602,
    "Coverage[0.5]": 0.6666666666666666,
    "RMSE": 23.268730904168905,
    "NRMSE": 0.4898680190351348,
    "ND": 0.342510772169682,
    "wQuantileLoss[0.5]": 0.3425107788621334,
    "mean_wQuantileLoss": 0.3425107788621334,
    "MAE_Coverage": 0.16666666666666663
}

  #### Plot   ####################################################### 

  #### Loading params   ############################################## 

  model_gluon.gluonts_model_old 
{'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}} {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}} {'path': 'ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]}

  #### Loading dataset   ############################################# 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### Model init, fit   ############################################# 

INFO:root:Using CPU
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in GaussianProcessTrainingNetwork: 14
100%|██████████| 10/10 [00:01<00:00,  8.88it/s, avg_epoch_loss=161]
INFO:root:Epoch[0] Elapsed time 1.126 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=161.108291
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 161.1082910709855 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model_old' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model_old.py'> <mlmodels.model_gluon.gluonts_model_old.Model object at 0x7f32a0f925c0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model_old.Model object at 0x7f32a0f925c0>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 152.27it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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

  model_gluon.gluonts_model_old 
{'model_name': 'deepstate', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}} {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}} {'path': 'ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]}

  #### Loading dataset   ############################################# 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### Model init, fit   ############################################# 

INFO:root:Using CPU
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in DeepStateTrainingNetwork: 28054
 10%|█         | 1/10 [02:18<20:42, 138.08s/it, avg_epoch_loss=0.582] 20%|██        | 2/10 [05:27<20:26, 153.34s/it, avg_epoch_loss=0.565] 30%|███       | 3/10 [09:11<20:23, 174.78s/it, avg_epoch_loss=0.548] 40%|████      | 4/10 [13:07<19:17, 192.91s/it, avg_epoch_loss=0.531] 50%|█████     | 5/10 [17:07<17:16, 207.27s/it, avg_epoch_loss=0.515] 60%|██████    | 6/10 [20:21<13:32, 203.09s/it, avg_epoch_loss=0.499] 70%|███████   | 7/10 [23:36<10:02, 200.67s/it, avg_epoch_loss=0.484] 80%|████████  | 8/10 [27:18<06:54, 207.28s/it, avg_epoch_loss=0.47]  90%|█████████ | 9/10 [30:53<03:29, 209.53s/it, avg_epoch_loss=0.457]100%|██████████| 10/10 [34:50<00:00, 217.73s/it, avg_epoch_loss=0.447]100%|██████████| 10/10 [34:50<00:00, 209.06s/it, avg_epoch_loss=0.447]
INFO:root:Epoch[0] Elapsed time 2090.562 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.446552
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.4465524971485138 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model_old' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model_old.py'> <mlmodels.model_gluon.gluonts_model_old.Model object at 0x7f32a10469e8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model_old.Model object at 0x7f32a10469e8>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 14.56it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
[master 9bb2aef] ml_store  && git pull --all
 1 file changed, 498 insertions(+)
To github.com:arita37/mlmodels_store.git
 + c9b1cc9...9bb2aef master -> master (forced update)





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
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon//gluonts_model.py", line 54, in <module>
    from mlmodels.util import load_function_uri
ImportError: cannot import name 'load_function_uri'

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
[master 0f6c10d] ml_store  && git pull --all
 1 file changed, 49 insertions(+)
To github.com:arita37/mlmodels_store.git
   9bb2aef..0f6c10d  master -> master





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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7f0b37843710> 

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
[master 24d65bc] ml_store  && git pull --all
 1 file changed, 107 insertions(+)
To github.com:arita37/mlmodels_store.git
   0f6c10d..24d65bc  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 6.10942600e-01 -2.79099641e+00 -1.33520272e+00 -4.56117555e-01
  -9.44959948e-01 -9.79890252e-01 -1.56993672e-01  6.92574348e-01
  -4.78672356e-01 -1.06460122e-01]
 [ 9.97855163e-01 -6.00138799e-01  4.57947076e-01  1.46765263e-01
  -9.33557290e-01  5.71804879e-01  5.72962726e-01 -3.68176565e-02
   1.12368489e-01 -1.78175491e-02]
 [ 1.06040861e+00  5.10307597e-01  5.01725109e-01 -9.15791849e-01
  -9.07318361e-01 -4.07252043e-01 -1.79612295e-01  9.84951672e-01
   1.07125243e+00 -5.93343754e-01]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 1.34728643e+00 -3.64538050e-01  8.07509886e-02 -4.59717681e-01
  -8.89487596e-01  1.70548352e+00  9.49961101e-02  2.40505552e-01
  -9.99426501e-01 -7.67803746e-01]
 [ 1.25704434e+00 -1.82391985e+00 -6.12406973e-01  1.16707517e+00
  -6.23732812e-01 -3.96687001e-02  8.16043684e-01  8.85825799e-01
   1.89861649e-01  3.93109245e-01]
 [ 8.71225789e-01 -2.09752935e-01 -4.56987858e-01  9.35147780e-01
  -8.73535822e-01  1.81252782e+00  9.25501215e-01  1.40109881e-01
  -1.41914878e+00  1.06898597e+00]
 [ 1.13545112e+00  8.61623101e-01  4.90616924e-02 -2.08639057e+00
  -1.11469020e+00  3.61801641e-01 -8.06178212e-01  4.25920177e-01
   4.90803971e-02 -5.96086335e-01]
 [ 1.18468624e+00 -1.00016919e+00 -5.93843067e-01  1.04499441e+00
   9.65482331e-01  6.08514698e-01 -6.25342001e-01 -6.93286967e-02
  -1.08392067e-01 -3.43900709e-01]
 [ 8.95623122e-01 -2.29820588e+00 -1.95225583e-02  1.45652739e+00
  -1.85064099e+00  3.16637236e-01  1.11337266e-01 -2.66412594e+00
  -4.26428618e-01 -8.39988915e-01]
 [ 8.15836116e-01 -1.39169388e+00  2.50598029e+00  4.50217742e-01
  -8.82869820e-01  6.27437083e-01 -1.19586151e+00  7.51337235e-01
   1.40395436e-01  1.91979229e+00]
 [ 1.01195228e+00 -1.88141087e+00  1.70018815e+00  4.97269099e-01
  -9.17664624e-01  2.37332699e-01 -1.09033833e+00 -2.14444405e+00
  -3.69562425e-01  6.08783659e-01]
 [ 4.46895161e-01  3.86539145e-01  1.35010682e+00 -8.51455657e-01
   8.50637963e-01  1.00088142e+00 -1.16017010e+00 -3.84832249e-01
   1.45810824e+00 -3.31283170e-01]
 [ 8.48069266e-01  4.51946037e-01  6.30195671e-01 -1.57915629e+00
   8.27987371e-01 -8.28627979e-01 -1.05344713e-01  5.28879746e-01
  -2.23708651e+00 -4.14846901e-01]
 [ 1.36586461e+00  3.95860270e+00  5.48129585e-01  6.48643644e-01
   8.49176066e-01  1.07343294e-01  1.38631426e+00 -1.39881282e+00
   8.17678188e-02 -1.63744959e+00]
 [ 1.24549398e+00 -7.22391905e-01  1.11813340e+00  1.09899633e+00
   1.00277655e+00 -9.01634490e-01 -5.32234021e-01 -8.22467189e-01
   7.21711292e-01  6.74396105e-01]
 [ 1.05936450e-01 -7.37289628e-01  6.50323214e-01  1.64665066e-01
  -1.53556118e+00  7.78174179e-01  5.03170861e-02  3.09816759e-01
   1.05132077e+00  6.06548400e-01]
 [ 3.54133613e-01  2.11124755e-01  9.21450069e-01  1.65275673e-02
   9.03945451e-01  1.77187720e-01  9.54250872e-02 -1.11647002e+00
   8.09271010e-02  6.07501958e-02]
 [ 2.07582971e+00 -1.40232915e+00 -4.79184915e-01  4.51122939e-01
   1.03436581e+00 -6.94920901e-01 -4.18937898e-01  5.15413802e-01
  -1.11487105e+00 -1.95210529e+00]
 [ 4.41189807e-01  4.79852371e-01 -1.92003697e-01 -1.55269878e+00
  -1.88873982e+00  5.78464420e-01  3.98598388e-01 -9.61263599e-01
  -1.45832446e+00 -3.05376438e+00]
 [ 8.78740711e-01 -1.92316341e-02  3.19656942e-01  1.50016279e-01
  -1.46662161e+00  4.63534322e-01 -8.98683193e-01  3.97880425e-01
  -9.96010889e-01  3.18154200e-01]
 [ 7.22978007e-01  1.85535621e-01  9.15499268e-01  3.94428030e-01
  -8.49830738e-01  7.25522558e-01 -1.50504326e-01  1.49588477e+00
   6.75453809e-01 -4.38200267e-01]
 [ 8.72267394e-01 -2.51630386e+00 -7.75070287e-01 -5.95667881e-01
   1.02600767e+00 -3.09121319e-01  1.74643509e+00  5.10937774e-01
   1.71066184e+00  1.41640538e-01]
 [ 1.83829400e+00  5.02740882e-01  1.29101580e-01  1.55880554e+00
   1.32551412e+00  1.09402696e-01  1.40754000e+00 -1.21974440e+00
   2.44936865e+00  1.61694960e+00]
 [ 1.22867367e+00  1.34373116e-01 -1.82420406e-01 -2.68371304e-01
  -1.73963799e+00 -1.31675626e-01 -9.26871939e-01  1.01855247e+00
   1.23055820e+00 -4.91125138e-01]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7ff204558cf8>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7ff2255c6f98> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 8.48069266e-01  4.51946037e-01  6.30195671e-01 -1.57915629e+00
   8.27987371e-01 -8.28627979e-01 -1.05344713e-01  5.28879746e-01
  -2.23708651e+00 -4.14846901e-01]
 [ 1.37661405e+00 -6.00225330e-01  7.25916853e-01 -3.79517516e-01
  -6.27546260e-01 -1.01480369e+00  9.66220863e-01  4.35986196e-01
  -6.87487393e-01  3.32107876e+00]
 [ 7.00175710e-01  5.56073510e-01  8.96864073e-02  1.69380911e+00
   8.82393314e-01  1.96869779e-01 -5.63788735e-01  1.69869255e-01
  -1.16400797e+00 -6.01156801e-01]
 [ 7.88018455e-01  3.01960045e-01  7.00982122e-01 -3.94689681e-01
  -1.20376927e+00 -1.17181338e+00  7.55392029e-01  9.84012237e-01
  -5.59681422e-01 -1.98937450e-01]
 [ 9.29250600e-01 -1.10657307e+00 -1.95816909e+00 -3.59224096e-01
  -1.21258781e+00  5.05381903e-01  5.42645295e-01  1.21794090e+00
  -1.94068096e+00  6.77807571e-01]
 [ 4.67397905e-01 -2.37875265e-01 -1.54491194e-01 -7.55662765e-01
  -5.47062239e-01  1.85143789e+00 -1.46405357e+00  2.09096677e-01
   1.55501599e+00 -9.24323185e-02]
 [ 1.17867274e+00 -5.99804531e-01 -6.94693595e-01  1.12341216e+00
   1.17899425e+00  3.05267040e-01  1.33526763e-02  1.38877940e+00
  -6.61344243e-01  6.21803504e-01]
 [ 8.95623122e-01 -2.29820588e+00 -1.95225583e-02  1.45652739e+00
  -1.85064099e+00  3.16637236e-01  1.11337266e-01 -2.66412594e+00
  -4.26428618e-01 -8.39988915e-01]
 [ 8.15836116e-01 -1.39169388e+00  2.50598029e+00  4.50217742e-01
  -8.82869820e-01  6.27437083e-01 -1.19586151e+00  7.51337235e-01
   1.40395436e-01  1.91979229e+00]
 [ 1.36586461e+00  3.95860270e+00  5.48129585e-01  6.48643644e-01
   8.49176066e-01  1.07343294e-01  1.38631426e+00 -1.39881282e+00
   8.17678188e-02 -1.63744959e+00]
 [ 1.25704434e+00 -1.82391985e+00 -6.12406973e-01  1.16707517e+00
  -6.23732812e-01 -3.96687001e-02  8.16043684e-01  8.85825799e-01
   1.89861649e-01  3.93109245e-01]
 [ 8.76994650e-01  1.23225307e+00 -8.67787223e-01 -2.54179868e-01
   8.91891405e-01  1.39984394e+00 -8.77281519e-01 -7.81911683e-01
  -4.37508983e-01 -1.44087602e+00]
 [ 1.03967316e+00 -7.31530982e-01  3.61847316e-01 -1.56573815e+00
   9.59288190e-01  1.01382247e+00 -1.78791289e+00 -2.22711263e+00
  -1.69933360e+00 -4.24492791e-01]
 [ 1.34740825e+00  7.33023232e-01  8.38634747e-01 -1.89881206e+00
  -5.42459922e-01 -1.11711069e+00 -1.09715436e+00 -5.08972278e-01
  -1.66485955e-01 -1.03918232e+00]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 1.58463774e+00  5.71209961e-02 -1.77183179e-02 -7.99547491e-01
   1.32970299e+00 -2.91594596e-01 -1.10771250e+00 -2.58982853e-01
   1.89293198e-01 -1.71939447e+00]
 [ 1.05936450e-01 -7.37289628e-01  6.50323214e-01  1.64665066e-01
  -1.53556118e+00  7.78174179e-01  5.03170861e-02  3.09816759e-01
   1.05132077e+00  6.06548400e-01]
 [ 1.16755486e+00  3.53600971e-02  7.14789597e-01 -1.53879325e+00
   1.10863359e+00 -4.47895185e-01 -1.75592564e+00  6.17985534e-01
  -1.84176326e-01  8.52704062e-01]
 [ 9.36211246e-01  2.04377395e-01 -1.49419377e+00  6.12232523e-01
  -9.84377246e-01  7.44884536e-01  4.94341651e-01 -3.62812886e-02
  -8.32395348e-01 -4.46699203e-01]
 [ 1.12062155e+00 -7.02920403e-01 -1.22957425e+00  7.25550518e-01
  -1.18013412e+00 -3.24204219e-01  1.10223673e+00  8.14343129e-01
   7.80469930e-01  1.10861676e+00]
 [ 9.80427414e-01  1.93752881e+00 -2.30839743e-01  3.66332015e-01
   1.10018476e+00 -1.04458938e+00 -3.44987210e-01  2.05117344e+00
   5.85662000e-01 -2.79308500e+00]
 [ 1.66752297e+00  1.22372221e+00 -4.59930104e-01 -5.93679025e-02
  -4.93856997e-01  1.44898940e+00 -1.18110317e+00 -4.77580855e-01
   2.59999942e-02 -7.90799954e-01]
 [ 1.39198128e+00 -1.90221025e-01 -5.37223024e-01 -4.48738033e-01
   7.04557071e-01 -6.72448039e-01 -7.01344426e-01 -5.57494722e-01
   9.39168744e-01  1.56263850e-01]
 [ 1.34728643e+00 -3.64538050e-01  8.07509886e-02 -4.59717681e-01
  -8.89487596e-01  1.70548352e+00  9.49961101e-02  2.40505552e-01
  -9.99426501e-01 -7.67803746e-01]
 [ 6.67591795e-01 -4.52524973e-01 -6.05981321e-01  1.16128569e+00
  -1.44620987e+00  1.06996554e+00  1.92381543e+00 -1.04553425e+00
   3.55284507e-01  1.80358898e+00]]
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
[[ 6.13636707e-01  3.16658895e-01  1.34710546e+00 -1.89526695e+00
  -7.60458095e-01  8.97291174e-02 -3.29051549e-01  4.10265745e-01
   8.59870972e-01 -1.04906775e+00]
 [ 9.36211246e-01  2.04377395e-01 -1.49419377e+00  6.12232523e-01
  -9.84377246e-01  7.44884536e-01  4.94341651e-01 -3.62812886e-02
  -8.32395348e-01 -4.46699203e-01]
 [ 5.58538729e-01 -5.16347909e-01 -5.18145552e-01  3.51116897e-01
   8.25506954e-01 -6.87704631e-02 -9.52062101e-01 -1.34776494e+00
   1.47073986e+00 -1.46140360e+00]
 [ 1.03967316e+00 -7.31530982e-01  3.61847316e-01 -1.56573815e+00
   9.59288190e-01  1.01382247e+00 -1.78791289e+00 -2.22711263e+00
  -1.69933360e+00 -4.24492791e-01]
 [ 1.46893146e+00 -1.47115693e+00  5.85910431e-01 -8.30171895e-01
   1.03345052e+00 -8.80577600e-01 -9.55425262e-01 -2.79097722e-01
   1.62284909e+00  2.06578332e+00]
 [ 1.98519313e+00  6.74711526e-01 -1.39662042e+00  6.18539131e-01
   1.22382712e+00 -4.43171931e-01 -1.89148284e-03  1.81053491e+00
  -1.30572692e+00 -8.61316361e-01]
 [ 1.14809657e+00 -7.33271604e-01  2.62467445e-01  8.36004719e-01
   1.17353145e+00  1.54335911e+00  2.84748111e-01  7.58805660e-01
   8.84908814e-01  2.76499305e-01]
 [ 6.67591795e-01 -4.52524973e-01 -6.05981321e-01  1.16128569e+00
  -1.44620987e+00  1.06996554e+00  1.92381543e+00 -1.04553425e+00
   3.55284507e-01  1.80358898e+00]
 [ 1.39198128e+00 -1.90221025e-01 -5.37223024e-01 -4.48738033e-01
   7.04557071e-01 -6.72448039e-01 -7.01344426e-01 -5.57494722e-01
   9.39168744e-01  1.56263850e-01]
 [ 8.78643802e-01  1.03703898e+00 -4.77124206e-01  6.72619748e-01
  -1.04948638e+00  2.42887697e+00  5.24750492e-01  1.00568668e+00
   3.53567216e-01 -3.59901817e-02]
 [ 9.83799588e-01 -4.07240024e-01  9.32721414e-01  1.60564992e-01
  -1.27861800e+00 -1.20149976e-01  1.99759555e-01  3.85602292e-01
   7.18290736e-01 -5.30119800e-01]
 [ 8.71225789e-01 -2.09752935e-01 -4.56987858e-01  9.35147780e-01
  -8.73535822e-01  1.81252782e+00  9.25501215e-01  1.40109881e-01
  -1.41914878e+00  1.06898597e+00]
 [ 4.46895161e-01  3.86539145e-01  1.35010682e+00 -8.51455657e-01
   8.50637963e-01  1.00088142e+00 -1.16017010e+00 -3.84832249e-01
   1.45810824e+00 -3.31283170e-01]
 [ 1.16777676e+00 -6.65754518e-01 -1.23312074e+00 -1.67419581e+00
   1.01313574e+00  8.25029824e-01 -1.20464572e-01 -4.98213564e-01
  -3.10984978e-01 -1.18231813e+00]
 [ 6.25673373e-01  5.92472801e-01  6.74570707e-01  1.19783084e+00
   1.23187251e+00  1.70459417e+00 -7.67309826e-01  1.04008915e+00
  -9.18440038e-01  1.46089238e+00]
 [ 1.37661405e+00 -6.00225330e-01  7.25916853e-01 -3.79517516e-01
  -6.27546260e-01 -1.01480369e+00  9.66220863e-01  4.35986196e-01
  -6.87487393e-01  3.32107876e+00]
 [ 1.83829400e+00  5.02740882e-01  1.29101580e-01  1.55880554e+00
   1.32551412e+00  1.09402696e-01  1.40754000e+00 -1.21974440e+00
   2.44936865e+00  1.61694960e+00]
 [ 6.23629500e-01  9.86352180e-01  1.45391758e+00 -4.66154857e-01
   9.36403332e-01  1.38499134e+00  3.49435894e-02 -1.07296428e+00
   4.95158611e-01  6.61681076e-01]
 [ 1.34740825e+00  7.33023232e-01  8.38634747e-01 -1.89881206e+00
  -5.42459922e-01 -1.11711069e+00 -1.09715436e+00 -5.08972278e-01
  -1.66485955e-01 -1.03918232e+00]
 [ 6.21530991e-01 -1.50957268e+00 -1.01932039e-01 -1.08071069e+00
  -1.13742855e+00  7.25474004e-01  7.98063795e-01 -3.91782562e-02
  -2.28754171e-01  7.43356544e-01]
 [ 1.34728643e+00 -3.64538050e-01  8.07509886e-02 -4.59717681e-01
  -8.89487596e-01  1.70548352e+00  9.49961101e-02  2.40505552e-01
  -9.99426501e-01 -7.67803746e-01]
 [ 7.75285326e-01  1.47016034e+00  1.03298378e+00 -8.70008223e-01
   7.86556511e-01  3.69190470e-01 -1.43195745e-01  8.53282186e-01
  -1.39711730e-01 -2.22414029e-01]
 [ 6.92114488e-01 -6.06524918e-02  2.05635552e+00 -2.41350300e+00
   1.17456965e+00 -1.77756638e+00 -2.81736269e-01 -7.77858827e-01
   1.11584111e+00  1.76024923e+00]
 [ 8.53355545e-01 -7.04350332e-01 -6.79383783e-01 -4.58666861e-02
  -1.29936179e+00 -2.18733459e-01  5.90039464e-01  1.53920701e+00
  -1.14870423e+00 -9.50909251e-01]
 [ 7.88018455e-01  3.01960045e-01  7.00982122e-01 -3.94689681e-01
  -1.20376927e+00 -1.17181338e+00  7.55392029e-01  9.84012237e-01
  -5.59681422e-01 -1.98937450e-01]]
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
[master b075e9c] ml_store  && git pull --all
 1 file changed, 320 insertions(+)
To github.com:arita37/mlmodels_store.git
   24d65bc..b075e9c  master -> master





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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140316603400144
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140316603399920
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140316603398688
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140316603398240
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140316603397736
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140316603397400

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
grad_step = 000000, loss = 1.874554
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 1.632913
grad_step = 000002, loss = 1.418207
grad_step = 000003, loss = 1.163768
grad_step = 000004, loss = 0.851063
grad_step = 000005, loss = 0.498983
grad_step = 000006, loss = 0.182492
grad_step = 000007, loss = 0.111840
grad_step = 000008, loss = 0.332893
grad_step = 000009, loss = 0.303975
grad_step = 000010, loss = 0.136363
grad_step = 000011, loss = 0.028663
grad_step = 000012, loss = 0.012792
grad_step = 000013, loss = 0.041973
grad_step = 000014, loss = 0.076018
grad_step = 000015, loss = 0.095265
grad_step = 000016, loss = 0.094977
grad_step = 000017, loss = 0.078995
grad_step = 000018, loss = 0.055638
grad_step = 000019, loss = 0.034907
grad_step = 000020, loss = 0.024346
grad_step = 000021, loss = 0.025008
grad_step = 000022, loss = 0.030083
grad_step = 000023, loss = 0.030896
grad_step = 000024, loss = 0.025249
grad_step = 000025, loss = 0.017862
grad_step = 000026, loss = 0.014263
grad_step = 000027, loss = 0.015507
grad_step = 000028, loss = 0.018820
grad_step = 000029, loss = 0.021018
grad_step = 000030, loss = 0.020434
grad_step = 000031, loss = 0.017295
grad_step = 000032, loss = 0.013099
grad_step = 000033, loss = 0.009651
grad_step = 000034, loss = 0.007967
grad_step = 000035, loss = 0.007945
grad_step = 000036, loss = 0.008606
grad_step = 000037, loss = 0.009032
grad_step = 000038, loss = 0.008953
grad_step = 000039, loss = 0.008623
grad_step = 000040, loss = 0.008372
grad_step = 000041, loss = 0.008178
grad_step = 000042, loss = 0.007886
grad_step = 000043, loss = 0.007349
grad_step = 000044, loss = 0.006599
grad_step = 000045, loss = 0.005865
grad_step = 000046, loss = 0.005430
grad_step = 000047, loss = 0.005440
grad_step = 000048, loss = 0.005742
grad_step = 000049, loss = 0.006019
grad_step = 000050, loss = 0.006017
grad_step = 000051, loss = 0.005736
grad_step = 000052, loss = 0.005390
grad_step = 000053, loss = 0.005191
grad_step = 000054, loss = 0.005165
grad_step = 000055, loss = 0.005179
grad_step = 000056, loss = 0.005086
grad_step = 000057, loss = 0.004857
grad_step = 000058, loss = 0.004595
grad_step = 000059, loss = 0.004448
grad_step = 000060, loss = 0.004485
grad_step = 000061, loss = 0.004637
grad_step = 000062, loss = 0.004751
grad_step = 000063, loss = 0.004718
grad_step = 000064, loss = 0.004549
grad_step = 000065, loss = 0.004359
grad_step = 000066, loss = 0.004257
grad_step = 000067, loss = 0.004263
grad_step = 000068, loss = 0.004307
grad_step = 000069, loss = 0.004306
grad_step = 000070, loss = 0.004238
grad_step = 000071, loss = 0.004146
grad_step = 000072, loss = 0.004093
grad_step = 000073, loss = 0.004100
grad_step = 000074, loss = 0.004132
grad_step = 000075, loss = 0.004134
grad_step = 000076, loss = 0.004080
grad_step = 000077, loss = 0.003996
grad_step = 000078, loss = 0.003927
grad_step = 000079, loss = 0.003898
grad_step = 000080, loss = 0.003897
grad_step = 000081, loss = 0.003896
grad_step = 000082, loss = 0.003876
grad_step = 000083, loss = 0.003840
grad_step = 000084, loss = 0.003804
grad_step = 000085, loss = 0.003781
grad_step = 000086, loss = 0.003764
grad_step = 000087, loss = 0.003744
grad_step = 000088, loss = 0.003714
grad_step = 000089, loss = 0.003681
grad_step = 000090, loss = 0.003653
grad_step = 000091, loss = 0.003631
grad_step = 000092, loss = 0.003613
grad_step = 000093, loss = 0.003593
grad_step = 000094, loss = 0.003570
grad_step = 000095, loss = 0.003546
grad_step = 000096, loss = 0.003522
grad_step = 000097, loss = 0.003499
grad_step = 000098, loss = 0.003474
grad_step = 000099, loss = 0.003449
grad_step = 000100, loss = 0.003424
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.003402
grad_step = 000102, loss = 0.003381
grad_step = 000103, loss = 0.003360
grad_step = 000104, loss = 0.003337
grad_step = 000105, loss = 0.003313
grad_step = 000106, loss = 0.003290
grad_step = 000107, loss = 0.003268
grad_step = 000108, loss = 0.003245
grad_step = 000109, loss = 0.003221
grad_step = 000110, loss = 0.003197
grad_step = 000111, loss = 0.003173
grad_step = 000112, loss = 0.003152
grad_step = 000113, loss = 0.003130
grad_step = 000114, loss = 0.003107
grad_step = 000115, loss = 0.003083
grad_step = 000116, loss = 0.003060
grad_step = 000117, loss = 0.003037
grad_step = 000118, loss = 0.003014
grad_step = 000119, loss = 0.002991
grad_step = 000120, loss = 0.002968
grad_step = 000121, loss = 0.002944
grad_step = 000122, loss = 0.002922
grad_step = 000123, loss = 0.002899
grad_step = 000124, loss = 0.002877
grad_step = 000125, loss = 0.002853
grad_step = 000126, loss = 0.002830
grad_step = 000127, loss = 0.002807
grad_step = 000128, loss = 0.002785
grad_step = 000129, loss = 0.002762
grad_step = 000130, loss = 0.002739
grad_step = 000131, loss = 0.002716
grad_step = 000132, loss = 0.002693
grad_step = 000133, loss = 0.002670
grad_step = 000134, loss = 0.002647
grad_step = 000135, loss = 0.002624
grad_step = 000136, loss = 0.002600
grad_step = 000137, loss = 0.002576
grad_step = 000138, loss = 0.002552
grad_step = 000139, loss = 0.002528
grad_step = 000140, loss = 0.002504
grad_step = 000141, loss = 0.002479
grad_step = 000142, loss = 0.002453
grad_step = 000143, loss = 0.002427
grad_step = 000144, loss = 0.002401
grad_step = 000145, loss = 0.002375
grad_step = 000146, loss = 0.002348
grad_step = 000147, loss = 0.002321
grad_step = 000148, loss = 0.002294
grad_step = 000149, loss = 0.002266
grad_step = 000150, loss = 0.002239
grad_step = 000151, loss = 0.002211
grad_step = 000152, loss = 0.002183
grad_step = 000153, loss = 0.002156
grad_step = 000154, loss = 0.002127
grad_step = 000155, loss = 0.002099
grad_step = 000156, loss = 0.002071
grad_step = 000157, loss = 0.002042
grad_step = 000158, loss = 0.002014
grad_step = 000159, loss = 0.001985
grad_step = 000160, loss = 0.001957
grad_step = 000161, loss = 0.001929
grad_step = 000162, loss = 0.001901
grad_step = 000163, loss = 0.001873
grad_step = 000164, loss = 0.001845
grad_step = 000165, loss = 0.001818
grad_step = 000166, loss = 0.001791
grad_step = 000167, loss = 0.001764
grad_step = 000168, loss = 0.001738
grad_step = 000169, loss = 0.001712
grad_step = 000170, loss = 0.001687
grad_step = 000171, loss = 0.001661
grad_step = 000172, loss = 0.001637
grad_step = 000173, loss = 0.001613
grad_step = 000174, loss = 0.001589
grad_step = 000175, loss = 0.001567
grad_step = 000176, loss = 0.001545
grad_step = 000177, loss = 0.001523
grad_step = 000178, loss = 0.001503
grad_step = 000179, loss = 0.001482
grad_step = 000180, loss = 0.001462
grad_step = 000181, loss = 0.001443
grad_step = 000182, loss = 0.001425
grad_step = 000183, loss = 0.001408
grad_step = 000184, loss = 0.001391
grad_step = 000185, loss = 0.001375
grad_step = 000186, loss = 0.001359
grad_step = 000187, loss = 0.001345
grad_step = 000188, loss = 0.001331
grad_step = 000189, loss = 0.001317
grad_step = 000190, loss = 0.001304
grad_step = 000191, loss = 0.001291
grad_step = 000192, loss = 0.001278
grad_step = 000193, loss = 0.001266
grad_step = 000194, loss = 0.001253
grad_step = 000195, loss = 0.001240
grad_step = 000196, loss = 0.001229
grad_step = 000197, loss = 0.001217
grad_step = 000198, loss = 0.001207
grad_step = 000199, loss = 0.001196
grad_step = 000200, loss = 0.001185
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001174
grad_step = 000202, loss = 0.001163
grad_step = 000203, loss = 0.001153
grad_step = 000204, loss = 0.001144
grad_step = 000205, loss = 0.001134
grad_step = 000206, loss = 0.001125
grad_step = 000207, loss = 0.001115
grad_step = 000208, loss = 0.001106
grad_step = 000209, loss = 0.001096
grad_step = 000210, loss = 0.001087
grad_step = 000211, loss = 0.001077
grad_step = 000212, loss = 0.001067
grad_step = 000213, loss = 0.001057
grad_step = 000214, loss = 0.001047
grad_step = 000215, loss = 0.001037
grad_step = 000216, loss = 0.001029
grad_step = 000217, loss = 0.001019
grad_step = 000218, loss = 0.001008
grad_step = 000219, loss = 0.000999
grad_step = 000220, loss = 0.000990
grad_step = 000221, loss = 0.000980
grad_step = 000222, loss = 0.000971
grad_step = 000223, loss = 0.000963
grad_step = 000224, loss = 0.000953
grad_step = 000225, loss = 0.000945
grad_step = 000226, loss = 0.000936
grad_step = 000227, loss = 0.000928
grad_step = 000228, loss = 0.000919
grad_step = 000229, loss = 0.000911
grad_step = 000230, loss = 0.000903
grad_step = 000231, loss = 0.000895
grad_step = 000232, loss = 0.000888
grad_step = 000233, loss = 0.000880
grad_step = 000234, loss = 0.000873
grad_step = 000235, loss = 0.000865
grad_step = 000236, loss = 0.000858
grad_step = 000237, loss = 0.000851
grad_step = 000238, loss = 0.000845
grad_step = 000239, loss = 0.000838
grad_step = 000240, loss = 0.000832
grad_step = 000241, loss = 0.000826
grad_step = 000242, loss = 0.000819
grad_step = 000243, loss = 0.000813
grad_step = 000244, loss = 0.000808
grad_step = 000245, loss = 0.000802
grad_step = 000246, loss = 0.000796
grad_step = 000247, loss = 0.000791
grad_step = 000248, loss = 0.000786
grad_step = 000249, loss = 0.000780
grad_step = 000250, loss = 0.000775
grad_step = 000251, loss = 0.000770
grad_step = 000252, loss = 0.000765
grad_step = 000253, loss = 0.000760
grad_step = 000254, loss = 0.000755
grad_step = 000255, loss = 0.000751
grad_step = 000256, loss = 0.000747
grad_step = 000257, loss = 0.000743
grad_step = 000258, loss = 0.000739
grad_step = 000259, loss = 0.000736
grad_step = 000260, loss = 0.000731
grad_step = 000261, loss = 0.000728
grad_step = 000262, loss = 0.000724
grad_step = 000263, loss = 0.000720
grad_step = 000264, loss = 0.000717
grad_step = 000265, loss = 0.000714
grad_step = 000266, loss = 0.000711
grad_step = 000267, loss = 0.000707
grad_step = 000268, loss = 0.000704
grad_step = 000269, loss = 0.000702
grad_step = 000270, loss = 0.000699
grad_step = 000271, loss = 0.000696
grad_step = 000272, loss = 0.000693
grad_step = 000273, loss = 0.000690
grad_step = 000274, loss = 0.000687
grad_step = 000275, loss = 0.000684
grad_step = 000276, loss = 0.000681
grad_step = 000277, loss = 0.000678
grad_step = 000278, loss = 0.000676
grad_step = 000279, loss = 0.000673
grad_step = 000280, loss = 0.000671
grad_step = 000281, loss = 0.000668
grad_step = 000282, loss = 0.000665
grad_step = 000283, loss = 0.000663
grad_step = 000284, loss = 0.000660
grad_step = 000285, loss = 0.000657
grad_step = 000286, loss = 0.000654
grad_step = 000287, loss = 0.000651
grad_step = 000288, loss = 0.000649
grad_step = 000289, loss = 0.000646
grad_step = 000290, loss = 0.000644
grad_step = 000291, loss = 0.000641
grad_step = 000292, loss = 0.000638
grad_step = 000293, loss = 0.000635
grad_step = 000294, loss = 0.000632
grad_step = 000295, loss = 0.000629
grad_step = 000296, loss = 0.000627
grad_step = 000297, loss = 0.000624
grad_step = 000298, loss = 0.000621
grad_step = 000299, loss = 0.000618
grad_step = 000300, loss = 0.000615
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000612
grad_step = 000302, loss = 0.000609
grad_step = 000303, loss = 0.000607
grad_step = 000304, loss = 0.000605
grad_step = 000305, loss = 0.000603
grad_step = 000306, loss = 0.000603
grad_step = 000307, loss = 0.000602
grad_step = 000308, loss = 0.000599
grad_step = 000309, loss = 0.000593
grad_step = 000310, loss = 0.000587
grad_step = 000311, loss = 0.000584
grad_step = 000312, loss = 0.000584
grad_step = 000313, loss = 0.000582
grad_step = 000314, loss = 0.000578
grad_step = 000315, loss = 0.000573
grad_step = 000316, loss = 0.000571
grad_step = 000317, loss = 0.000570
grad_step = 000318, loss = 0.000568
grad_step = 000319, loss = 0.000564
grad_step = 000320, loss = 0.000560
grad_step = 000321, loss = 0.000557
grad_step = 000322, loss = 0.000555
grad_step = 000323, loss = 0.000553
grad_step = 000324, loss = 0.000551
grad_step = 000325, loss = 0.000547
grad_step = 000326, loss = 0.000544
grad_step = 000327, loss = 0.000542
grad_step = 000328, loss = 0.000540
grad_step = 000329, loss = 0.000537
grad_step = 000330, loss = 0.000535
grad_step = 000331, loss = 0.000532
grad_step = 000332, loss = 0.000529
grad_step = 000333, loss = 0.000527
grad_step = 000334, loss = 0.000525
grad_step = 000335, loss = 0.000523
grad_step = 000336, loss = 0.000521
grad_step = 000337, loss = 0.000519
grad_step = 000338, loss = 0.000515
grad_step = 000339, loss = 0.000513
grad_step = 000340, loss = 0.000511
grad_step = 000341, loss = 0.000508
grad_step = 000342, loss = 0.000507
grad_step = 000343, loss = 0.000505
grad_step = 000344, loss = 0.000502
grad_step = 000345, loss = 0.000501
grad_step = 000346, loss = 0.000498
grad_step = 000347, loss = 0.000496
grad_step = 000348, loss = 0.000494
grad_step = 000349, loss = 0.000493
grad_step = 000350, loss = 0.000491
grad_step = 000351, loss = 0.000489
grad_step = 000352, loss = 0.000488
grad_step = 000353, loss = 0.000488
grad_step = 000354, loss = 0.000489
grad_step = 000355, loss = 0.000488
grad_step = 000356, loss = 0.000487
grad_step = 000357, loss = 0.000484
grad_step = 000358, loss = 0.000480
grad_step = 000359, loss = 0.000475
grad_step = 000360, loss = 0.000472
grad_step = 000361, loss = 0.000471
grad_step = 000362, loss = 0.000470
grad_step = 000363, loss = 0.000469
grad_step = 000364, loss = 0.000468
grad_step = 000365, loss = 0.000467
grad_step = 000366, loss = 0.000464
grad_step = 000367, loss = 0.000462
grad_step = 000368, loss = 0.000461
grad_step = 000369, loss = 0.000459
grad_step = 000370, loss = 0.000458
grad_step = 000371, loss = 0.000458
grad_step = 000372, loss = 0.000457
grad_step = 000373, loss = 0.000456
grad_step = 000374, loss = 0.000454
grad_step = 000375, loss = 0.000452
grad_step = 000376, loss = 0.000450
grad_step = 000377, loss = 0.000449
grad_step = 000378, loss = 0.000447
grad_step = 000379, loss = 0.000446
grad_step = 000380, loss = 0.000445
grad_step = 000381, loss = 0.000444
grad_step = 000382, loss = 0.000443
grad_step = 000383, loss = 0.000442
grad_step = 000384, loss = 0.000441
grad_step = 000385, loss = 0.000440
grad_step = 000386, loss = 0.000438
grad_step = 000387, loss = 0.000437
grad_step = 000388, loss = 0.000436
grad_step = 000389, loss = 0.000434
grad_step = 000390, loss = 0.000433
grad_step = 000391, loss = 0.000432
grad_step = 000392, loss = 0.000430
grad_step = 000393, loss = 0.000430
grad_step = 000394, loss = 0.000429
grad_step = 000395, loss = 0.000428
grad_step = 000396, loss = 0.000427
grad_step = 000397, loss = 0.000426
grad_step = 000398, loss = 0.000425
grad_step = 000399, loss = 0.000424
grad_step = 000400, loss = 0.000423
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000422
grad_step = 000402, loss = 0.000421
grad_step = 000403, loss = 0.000419
grad_step = 000404, loss = 0.000418
grad_step = 000405, loss = 0.000417
grad_step = 000406, loss = 0.000416
grad_step = 000407, loss = 0.000416
grad_step = 000408, loss = 0.000415
grad_step = 000409, loss = 0.000415
grad_step = 000410, loss = 0.000414
grad_step = 000411, loss = 0.000412
grad_step = 000412, loss = 0.000411
grad_step = 000413, loss = 0.000409
grad_step = 000414, loss = 0.000408
grad_step = 000415, loss = 0.000407
grad_step = 000416, loss = 0.000406
grad_step = 000417, loss = 0.000405
grad_step = 000418, loss = 0.000405
grad_step = 000419, loss = 0.000403
grad_step = 000420, loss = 0.000402
grad_step = 000421, loss = 0.000402
grad_step = 000422, loss = 0.000401
grad_step = 000423, loss = 0.000400
grad_step = 000424, loss = 0.000400
grad_step = 000425, loss = 0.000402
grad_step = 000426, loss = 0.000406
grad_step = 000427, loss = 0.000413
grad_step = 000428, loss = 0.000421
grad_step = 000429, loss = 0.000428
grad_step = 000430, loss = 0.000422
grad_step = 000431, loss = 0.000406
grad_step = 000432, loss = 0.000393
grad_step = 000433, loss = 0.000397
grad_step = 000434, loss = 0.000409
grad_step = 000435, loss = 0.000410
grad_step = 000436, loss = 0.000398
grad_step = 000437, loss = 0.000388
grad_step = 000438, loss = 0.000390
grad_step = 000439, loss = 0.000396
grad_step = 000440, loss = 0.000396
grad_step = 000441, loss = 0.000389
grad_step = 000442, loss = 0.000384
grad_step = 000443, loss = 0.000385
grad_step = 000444, loss = 0.000389
grad_step = 000445, loss = 0.000388
grad_step = 000446, loss = 0.000383
grad_step = 000447, loss = 0.000379
grad_step = 000448, loss = 0.000380
grad_step = 000449, loss = 0.000382
grad_step = 000450, loss = 0.000381
grad_step = 000451, loss = 0.000378
grad_step = 000452, loss = 0.000375
grad_step = 000453, loss = 0.000375
grad_step = 000454, loss = 0.000376
grad_step = 000455, loss = 0.000376
grad_step = 000456, loss = 0.000373
grad_step = 000457, loss = 0.000372
grad_step = 000458, loss = 0.000371
grad_step = 000459, loss = 0.000371
grad_step = 000460, loss = 0.000371
grad_step = 000461, loss = 0.000370
grad_step = 000462, loss = 0.000368
grad_step = 000463, loss = 0.000367
grad_step = 000464, loss = 0.000367
grad_step = 000465, loss = 0.000367
grad_step = 000466, loss = 0.000366
grad_step = 000467, loss = 0.000365
grad_step = 000468, loss = 0.000364
grad_step = 000469, loss = 0.000363
grad_step = 000470, loss = 0.000363
grad_step = 000471, loss = 0.000362
grad_step = 000472, loss = 0.000362
grad_step = 000473, loss = 0.000361
grad_step = 000474, loss = 0.000360
grad_step = 000475, loss = 0.000359
grad_step = 000476, loss = 0.000358
grad_step = 000477, loss = 0.000358
grad_step = 000478, loss = 0.000357
grad_step = 000479, loss = 0.000357
grad_step = 000480, loss = 0.000356
grad_step = 000481, loss = 0.000355
grad_step = 000482, loss = 0.000354
grad_step = 000483, loss = 0.000353
grad_step = 000484, loss = 0.000353
grad_step = 000485, loss = 0.000352
grad_step = 000486, loss = 0.000351
grad_step = 000487, loss = 0.000351
grad_step = 000488, loss = 0.000350
grad_step = 000489, loss = 0.000349
grad_step = 000490, loss = 0.000349
grad_step = 000491, loss = 0.000348
grad_step = 000492, loss = 0.000348
grad_step = 000493, loss = 0.000347
grad_step = 000494, loss = 0.000347
grad_step = 000495, loss = 0.000347
grad_step = 000496, loss = 0.000347
grad_step = 000497, loss = 0.000347
grad_step = 000498, loss = 0.000347
grad_step = 000499, loss = 0.000347
grad_step = 000500, loss = 0.000346
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000344
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
[[0.8371633  0.85437715 0.9216315  0.9537374  1.0071701 ]
 [0.83878726 0.8850957  0.96086144 1.0157355  0.9914766 ]
 [0.8846834  0.92493796 0.99684274 1.005268   0.937876  ]
 [0.92194325 0.9581772  0.9923235  0.9723804  0.9045591 ]
 [0.98987633 0.9859184  0.9595052  0.90845144 0.8608098 ]
 [0.996481   0.96819806 0.91016984 0.8619337  0.8567972 ]
 [0.95904124 0.9009813  0.8731424  0.8408834  0.83773506]
 [0.89624286 0.82445884 0.85010415 0.8183424  0.8289795 ]
 [0.83307505 0.81369007 0.82078606 0.83398163 0.8520293 ]
 [0.807021   0.78945136 0.85802394 0.83782005 0.8289663 ]
 [0.8037404  0.8054138  0.8486498  0.8615781  0.89899224]
 [0.81386447 0.8548703  0.84421486 0.90346575 0.9538846 ]
 [0.827261   0.8525042  0.91825366 0.9594507  1.0027378 ]
 [0.8434947  0.89537275 0.9613263  1.0205845  0.98416775]
 [0.90009487 0.941406   0.9995545  1.0012395  0.91996646]
 [0.93245953 0.9620433  0.980314   0.9515122  0.8879918 ]
 [1.0003985  0.9865777  0.94650644 0.88215214 0.8383492 ]
 [0.9884364  0.9485558  0.89139247 0.8451116  0.8361509 ]
 [0.95643806 0.8892287  0.85445493 0.8295847  0.8302148 ]
 [0.9013354  0.8360567  0.8407565  0.8240502  0.83205163]
 [0.8516884  0.8327386  0.8194793  0.8437215  0.8626492 ]
 [0.82579356 0.80729854 0.8625647  0.8489324  0.8361247 ]
 [0.8179322  0.81851083 0.8579508  0.87265635 0.905467  ]
 [0.8207829  0.85996914 0.8520958  0.9114889  0.95383483]
 [0.84215033 0.8593292  0.9232657  0.95624286 1.0129864 ]
 [0.84599745 0.8921096  0.9654765  1.0243984  1.0047214 ]
 [0.8943852  0.9357824  1.0059558  1.0206177  0.95263773]
 [0.92916846 0.96689653 1.0047846  0.9876958  0.91460466]
 [0.997147   0.9948973  0.9751623  0.922034   0.86416227]
 [1.0080695  0.9787574  0.92103136 0.8697213  0.8603818 ]
 [0.9695845  0.90964174 0.8790678  0.8476509  0.845392  ]]

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
[master 4016711] ml_store  && git pull --all
 1 file changed, 1121 insertions(+)
To github.com:arita37/mlmodels_store.git
   b075e9c..4016711  master -> master





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
