
  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_all 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '7d2329693089c1f82c9643c24694005c94b5ebed', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/7d2329693089c1f82c9643c24694005c94b5ebed

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed

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
Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
Already up to date.
[master 94525e6] ml_store
 2 files changed, 68 insertions(+), 10053 deletions(-)
 rewrite log_testall/log_testall.py (99%)
To github.com:arita37/mlmodels_store.git
   959598d..94525e6  master -> master





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
[master b6d869a] ml_store
 1 file changed, 49 insertions(+)
To github.com:arita37/mlmodels_store.git
   94525e6..b6d869a  master -> master





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
sequence_max (InputLayer)       [(None, 3)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_mean (E (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         20          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-20 16:12:27.189053: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-20 16:12:27.194175: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-20 16:12:27.194357: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56409590e010 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-20 16:12:27.194374: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 228
Trainable params: 228
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 2ms/sample - loss: 0.2516 - binary_crossentropy: 0.7226 - val_loss: 0.2521 - val_binary_crossentropy: 0.7494

  #### metrics   #################################################### 
{'MSE': 0.2515830159883276}

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
sequence_max (InputLayer)       [(None, 3)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_mean (E (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
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
Total params: 228
Trainable params: 228
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_max[0][0]               
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
Total params: 463
Trainable params: 463
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2502 - binary_crossentropy: 0.6936500/500 [==============================] - 1s 2ms/sample - loss: 0.2498 - binary_crossentropy: 0.6928 - val_loss: 0.2509 - val_binary_crossentropy: 0.6949

  #### metrics   #################################################### 
{'MSE': 0.25024204845150194}

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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_max[0][0]               
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
Total params: 463
Trainable params: 463
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
weighted_sequence_layer_6 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
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
Total params: 582
Trainable params: 582
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 3ms/sample - loss: 0.2500 - binary_crossentropy: 0.6932 - val_loss: 0.2500 - val_binary_crossentropy: 0.6932

  #### metrics   #################################################### 
{'MSE': 0.24991806995197252}

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
weighted_sequence_layer_6 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
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
Total params: 582
Trainable params: 582
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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_mean (E (None, 6, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         28          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         12          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_2[0][0]           
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
100/500 [=====>........................] - ETA: 2s - loss: 0.2400 - binary_crossentropy: 0.6720500/500 [==============================] - 2s 3ms/sample - loss: 0.2566 - binary_crossentropy: 0.7077 - val_loss: 0.2715 - val_binary_crossentropy: 0.7379

  #### metrics   #################################################### 
{'MSE': 0.2627861320544804}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_mean (E (None, 6, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         28          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         12          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_2[0][0]           
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_mean[0][0]              
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
Total params: 138
Trainable params: 138
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 3s - loss: 0.2577 - binary_crossentropy: 0.7091500/500 [==============================] - 2s 4ms/sample - loss: 0.2550 - binary_crossentropy: 0.7036 - val_loss: 0.2538 - val_binary_crossentropy: 0.7008

  #### metrics   #################################################### 
{'MSE': 0.25280213669194507}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_mean[0][0]              
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
Total params: 138
Trainable params: 138
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-20 16:14:01.074369: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 16:14:01.076841: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 16:14:01.082334: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-20 16:14:01.092438: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-20 16:14:01.094253: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-20 16:14:01.095898: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 16:14:01.097664: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2580 - val_binary_crossentropy: 0.7092
2020-05-20 16:14:02.520173: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 16:14:02.521989: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 16:14:02.526272: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-20 16:14:02.535065: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-20 16:14:02.536686: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-20 16:14:02.538126: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 16:14:02.539557: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2598732395322478}

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
2020-05-20 16:14:25.564879: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 16:14:25.566518: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 16:14:25.570580: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-20 16:14:25.577959: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-20 16:14:25.579180: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-20 16:14:25.580241: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 16:14:25.581290: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2516 - val_binary_crossentropy: 0.6964
2020-05-20 16:14:27.123550: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 16:14:27.124750: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 16:14:27.127818: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-20 16:14:27.133463: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-20 16:14:27.134612: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-20 16:14:27.135627: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 16:14:27.136417: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2520601511545877}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-20 16:14:59.508204: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 16:14:59.513232: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 16:14:59.530141: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-20 16:14:59.555569: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-20 16:14:59.560745: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-20 16:14:59.564874: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 16:14:59.568808: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 4s 4s/sample - loss: 0.5690 - binary_crossentropy: 1.4036 - val_loss: 0.2675 - val_binary_crossentropy: 0.7295
2020-05-20 16:15:01.790112: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 16:15:01.796094: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 16:15:01.809182: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-20 16:15:01.834946: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-20 16:15:01.839305: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-20 16:15:01.853823: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 16:15:01.858300: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.31102896079143755}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_mean (E (None, 6, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
Total params: 710
Trainable params: 710
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2443 - binary_crossentropy: 0.6817500/500 [==============================] - 4s 8ms/sample - loss: 0.2515 - binary_crossentropy: 0.6962 - val_loss: 0.2569 - val_binary_crossentropy: 0.7071

  #### metrics   #################################################### 
{'MSE': 0.2537422191317477}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_mean (E (None, 6, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
Total params: 710
Trainable params: 710
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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 2)         14          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 2)         18          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 2)         14          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         14          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         14          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_2[0][0]           
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
Total params: 254
Trainable params: 254
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2796 - binary_crossentropy: 0.7586500/500 [==============================] - 5s 9ms/sample - loss: 0.2679 - binary_crossentropy: 0.7338 - val_loss: 0.2563 - val_binary_crossentropy: 0.7070

  #### metrics   #################################################### 
{'MSE': 0.2602967587126479}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 2)         14          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 2)         18          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 2)         14          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         14          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         14          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_2[0][0]           
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
Total params: 254
Trainable params: 254
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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
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
Total params: 1,904
Trainable params: 1,904
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2541 - binary_crossentropy: 0.7014500/500 [==============================] - 5s 10ms/sample - loss: 0.2578 - binary_crossentropy: 0.7090 - val_loss: 0.2518 - val_binary_crossentropy: 0.6967

  #### metrics   #################################################### 
{'MSE': 0.2516493760742491}

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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
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
Total params: 1,904
Trainable params: 1,904
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
regionsequence_sum (InputLayer) [(None, 8)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 5)]          0                                            
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
region_10sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         9           regionsequence_max[0][0]         
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
100/500 [=====>........................] - ETA: 9s - loss: 0.2631 - binary_crossentropy: 0.7197500/500 [==============================] - 6s 13ms/sample - loss: 0.2605 - binary_crossentropy: 0.7147 - val_loss: 0.2600 - val_binary_crossentropy: 0.7134

  #### metrics   #################################################### 
{'MSE': 0.25949113051114875}

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
regionsequence_sum (InputLayer) [(None, 8)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 5)]          0                                            
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
region_10sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         9           regionsequence_max[0][0]         
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_mean (E (None, 1, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_max[0][0]               
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
Total params: 1,382
Trainable params: 1,382
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2900 - binary_crossentropy: 0.7873500/500 [==============================] - 5s 11ms/sample - loss: 0.2842 - binary_crossentropy: 0.7731 - val_loss: 0.2710 - val_binary_crossentropy: 0.7402

  #### metrics   #################################################### 
{'MSE': 0.2737142779495717}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_mean (E (None, 1, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_max[0][0]               
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
Total params: 1,382
Trainable params: 1,382
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         8           hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         36          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 8, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 8, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 8, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 8, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 2, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         12          sequence_mean[0][0]              
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           hash_10[0][0]                    
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
Total params: 3,086
Trainable params: 3,006
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2680 - binary_crossentropy: 0.7309500/500 [==============================] - 6s 12ms/sample - loss: 0.2666 - binary_crossentropy: 0.8565 - val_loss: 0.2580 - val_binary_crossentropy: 0.8129

  #### metrics   #################################################### 
{'MSE': 0.26055262009562385}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         8           hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         36          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 8, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 8, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 8, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 8, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 2, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         12          sequence_mean[0][0]              
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           hash_10[0][0]                    
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
Total params: 3,086
Trainable params: 3,006
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
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
From github.com:arita37/mlmodels_store
   b6d869a..7e15b76  master     -> origin/master
Updating b6d869a..7e15b76
Fast-forward
 error_list/20200520/list_log_testall_20200520.md | 752 +----------------------
 1 file changed, 5 insertions(+), 747 deletions(-)
[master a1deeb5] ml_store
 1 file changed, 4953 insertions(+)
To github.com:arita37/mlmodels_store.git
   7e15b76..a1deeb5  master -> master





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
[master e7b4bc7] ml_store
 1 file changed, 51 insertions(+)
To github.com:arita37/mlmodels_store.git
   a1deeb5..e7b4bc7  master -> master





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
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
Already up to date.
[master 950e1f4] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   e7b4bc7..950e1f4  master -> master





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
[master c7f5500] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   950e1f4..c7f5500  master -> master





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

2020-05-20 16:23:51.836568: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-20 16:23:51.841227: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-20 16:23:51.841386: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55bc57299760 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-20 16:23:51.841401: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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
256/354 [====================>.........] - ETA: 3s - loss: 1.2618
354/354 [==============================] - 15s 42ms/step - loss: 1.2771 - val_loss: 1.9349

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
Warning: Permanently added the RSA host key for IP address '140.82.113.4' to the list of known hosts.
Already up to date.
[master e300f9d] ml_store
 1 file changed, 151 insertions(+)
To github.com:arita37/mlmodels_store.git
   c7f5500..e300f9d  master -> master





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
[master ed9f690] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   e300f9d..ed9f690  master -> master





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
[master be0a209] ml_store
 1 file changed, 45 insertions(+)
To github.com:arita37/mlmodels_store.git
   ed9f690..be0a209  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 4079616/17464789 [======>.......................] - ETA: 0s
11362304/17464789 [==================>...........] - ETA: 0s
16506880/17464789 [===========================>..] - ETA: 0s
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
2020-05-20 16:24:53.970186: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-20 16:24:53.974083: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-20 16:24:53.974226: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c0cdd00d30 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-20 16:24:53.974241: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.7586 - accuracy: 0.4940
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8123 - accuracy: 0.4905 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7280 - accuracy: 0.4960
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7855 - accuracy: 0.4922
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.8322 - accuracy: 0.4892
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.8378 - accuracy: 0.4888
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7740 - accuracy: 0.4930
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7395 - accuracy: 0.4952
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7348 - accuracy: 0.4956
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7188 - accuracy: 0.4966
11000/25000 [============>.................] - ETA: 4s - loss: 7.7029 - accuracy: 0.4976
12000/25000 [=============>................] - ETA: 3s - loss: 7.6858 - accuracy: 0.4988
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7091 - accuracy: 0.4972
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7039 - accuracy: 0.4976
15000/25000 [=================>............] - ETA: 3s - loss: 7.7300 - accuracy: 0.4959
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7030 - accuracy: 0.4976
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6883 - accuracy: 0.4986
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6811 - accuracy: 0.4991
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6916 - accuracy: 0.4984
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6820 - accuracy: 0.4990
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6856 - accuracy: 0.4988
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6868 - accuracy: 0.4987
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6920 - accuracy: 0.4983
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6839 - accuracy: 0.4989
25000/25000 [==============================] - 9s 368us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7ff597998208>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7ff597ac07f0> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.7893 - accuracy: 0.4920
 2000/25000 [=>............................] - ETA: 9s - loss: 8.1343 - accuracy: 0.4695 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.9171 - accuracy: 0.4837
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.8583 - accuracy: 0.4875
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.8752 - accuracy: 0.4864
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.8506 - accuracy: 0.4880
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.8419 - accuracy: 0.4886
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.8008 - accuracy: 0.4913
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7757 - accuracy: 0.4929
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7755 - accuracy: 0.4929
11000/25000 [============>.................] - ETA: 4s - loss: 7.7447 - accuracy: 0.4949
12000/25000 [=============>................] - ETA: 4s - loss: 7.7510 - accuracy: 0.4945
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7256 - accuracy: 0.4962
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7159 - accuracy: 0.4968
15000/25000 [=================>............] - ETA: 3s - loss: 7.6952 - accuracy: 0.4981
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7193 - accuracy: 0.4966
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6955 - accuracy: 0.4981
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6888 - accuracy: 0.4986
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6537 - accuracy: 0.5008
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6536 - accuracy: 0.5009
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6498 - accuracy: 0.5011
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6548 - accuracy: 0.5008
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6640 - accuracy: 0.5002
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6717 - accuracy: 0.4997
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

 1000/25000 [>.............................] - ETA: 11s - loss: 7.9120 - accuracy: 0.4840
 2000/25000 [=>............................] - ETA: 8s - loss: 7.9580 - accuracy: 0.4810 
 3000/25000 [==>...........................] - ETA: 7s - loss: 8.0244 - accuracy: 0.4767
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.8698 - accuracy: 0.4868
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.8169 - accuracy: 0.4902
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7663 - accuracy: 0.4935
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7761 - accuracy: 0.4929
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6973 - accuracy: 0.4980
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6683 - accuracy: 0.4999
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6835 - accuracy: 0.4989
11000/25000 [============>.................] - ETA: 4s - loss: 7.6652 - accuracy: 0.5001
12000/25000 [=============>................] - ETA: 3s - loss: 7.6577 - accuracy: 0.5006
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6242 - accuracy: 0.5028
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6403 - accuracy: 0.5017
15000/25000 [=================>............] - ETA: 3s - loss: 7.6482 - accuracy: 0.5012
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6427 - accuracy: 0.5016
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6621 - accuracy: 0.5003
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6709 - accuracy: 0.4997
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6779 - accuracy: 0.4993
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6613 - accuracy: 0.5003
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6557 - accuracy: 0.5007
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6485 - accuracy: 0.5012
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6553 - accuracy: 0.5007
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
25000/25000 [==============================] - 9s 364us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
   be0a209..dbb23a4  master     -> origin/master
Updating be0a209..dbb23a4
Fast-forward
 error_list/20200520/list_log_testall_20200520.md | 103 +++++++++++++++++++++++
 1 file changed, 103 insertions(+)
[master df7f74d] ml_store
 1 file changed, 323 insertions(+)
To github.com:arita37/mlmodels_store.git
   dbb23a4..df7f74d  master -> master





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

13/13 [==============================] - 1s 102ms/step - loss: nan
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

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 8/10

13/13 [==============================] - 0s 6ms/step - loss: nan
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
[master d05bfb2] ml_store
 1 file changed, 126 insertions(+)
To github.com:arita37/mlmodels_store.git
   df7f74d..d05bfb2  master -> master





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
  614400/11490434 [>.............................] - ETA: 0s
 1482752/11490434 [==>...........................] - ETA: 0s
 2408448/11490434 [=====>........................] - ETA: 0s
 3366912/11490434 [=======>......................] - ETA: 0s
 4407296/11490434 [==========>...................] - ETA: 0s
 5505024/11490434 [=============>................] - ETA: 0s
 6635520/11490434 [================>.............] - ETA: 0s
 7856128/11490434 [===================>..........] - ETA: 0s
 9109504/11490434 [======================>.......] - ETA: 0s
10420224/11490434 [==========================>...] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:15 - loss: 2.2753 - categorical_accuracy: 0.1562
   64/60000 [..............................] - ETA: 4:31 - loss: 2.2999 - categorical_accuracy: 0.1406
   96/60000 [..............................] - ETA: 3:34 - loss: 2.2686 - categorical_accuracy: 0.1562
  128/60000 [..............................] - ETA: 3:06 - loss: 2.2539 - categorical_accuracy: 0.1641
  160/60000 [..............................] - ETA: 2:49 - loss: 2.2251 - categorical_accuracy: 0.1875
  192/60000 [..............................] - ETA: 2:37 - loss: 2.1900 - categorical_accuracy: 0.2344
  224/60000 [..............................] - ETA: 2:29 - loss: 2.1749 - categorical_accuracy: 0.2455
  256/60000 [..............................] - ETA: 2:23 - loss: 2.1265 - categorical_accuracy: 0.2656
  288/60000 [..............................] - ETA: 2:18 - loss: 2.0968 - categorical_accuracy: 0.2674
  320/60000 [..............................] - ETA: 2:15 - loss: 2.0787 - categorical_accuracy: 0.2656
  352/60000 [..............................] - ETA: 2:12 - loss: 2.0359 - categorical_accuracy: 0.2926
  384/60000 [..............................] - ETA: 2:09 - loss: 1.9720 - categorical_accuracy: 0.3151
  416/60000 [..............................] - ETA: 2:07 - loss: 1.9370 - categorical_accuracy: 0.3293
  448/60000 [..............................] - ETA: 2:05 - loss: 1.9061 - categorical_accuracy: 0.3438
  480/60000 [..............................] - ETA: 2:03 - loss: 1.8954 - categorical_accuracy: 0.3521
  512/60000 [..............................] - ETA: 2:02 - loss: 1.8694 - categorical_accuracy: 0.3672
  544/60000 [..............................] - ETA: 2:01 - loss: 1.8094 - categorical_accuracy: 0.3879
  576/60000 [..............................] - ETA: 2:00 - loss: 1.7867 - categorical_accuracy: 0.3906
  608/60000 [..............................] - ETA: 2:00 - loss: 1.7693 - categorical_accuracy: 0.3964
  640/60000 [..............................] - ETA: 1:59 - loss: 1.7441 - categorical_accuracy: 0.4047
  672/60000 [..............................] - ETA: 1:57 - loss: 1.7037 - categorical_accuracy: 0.4182
  704/60000 [..............................] - ETA: 1:56 - loss: 1.6704 - categorical_accuracy: 0.4304
  736/60000 [..............................] - ETA: 1:56 - loss: 1.6631 - categorical_accuracy: 0.4375
  768/60000 [..............................] - ETA: 1:55 - loss: 1.6441 - categorical_accuracy: 0.4440
  800/60000 [..............................] - ETA: 1:54 - loss: 1.6097 - categorical_accuracy: 0.4563
  832/60000 [..............................] - ETA: 1:54 - loss: 1.5787 - categorical_accuracy: 0.4700
  864/60000 [..............................] - ETA: 1:53 - loss: 1.5434 - categorical_accuracy: 0.4838
  896/60000 [..............................] - ETA: 1:53 - loss: 1.5218 - categorical_accuracy: 0.4922
  928/60000 [..............................] - ETA: 1:53 - loss: 1.5159 - categorical_accuracy: 0.4957
  960/60000 [..............................] - ETA: 1:52 - loss: 1.5085 - categorical_accuracy: 0.5010
  992/60000 [..............................] - ETA: 1:52 - loss: 1.4958 - categorical_accuracy: 0.5060
 1024/60000 [..............................] - ETA: 1:51 - loss: 1.4794 - categorical_accuracy: 0.5127
 1056/60000 [..............................] - ETA: 1:51 - loss: 1.4620 - categorical_accuracy: 0.5199
 1088/60000 [..............................] - ETA: 1:50 - loss: 1.4438 - categorical_accuracy: 0.5276
 1120/60000 [..............................] - ETA: 1:51 - loss: 1.4235 - categorical_accuracy: 0.5339
 1152/60000 [..............................] - ETA: 1:51 - loss: 1.4087 - categorical_accuracy: 0.5365
 1184/60000 [..............................] - ETA: 1:50 - loss: 1.3834 - categorical_accuracy: 0.5448
 1216/60000 [..............................] - ETA: 1:50 - loss: 1.3620 - categorical_accuracy: 0.5526
 1248/60000 [..............................] - ETA: 1:49 - loss: 1.3479 - categorical_accuracy: 0.5577
 1280/60000 [..............................] - ETA: 1:49 - loss: 1.3348 - categorical_accuracy: 0.5625
 1312/60000 [..............................] - ETA: 1:49 - loss: 1.3175 - categorical_accuracy: 0.5678
 1344/60000 [..............................] - ETA: 1:49 - loss: 1.3119 - categorical_accuracy: 0.5692
 1376/60000 [..............................] - ETA: 1:49 - loss: 1.3040 - categorical_accuracy: 0.5712
 1408/60000 [..............................] - ETA: 1:48 - loss: 1.2851 - categorical_accuracy: 0.5781
 1440/60000 [..............................] - ETA: 1:48 - loss: 1.2707 - categorical_accuracy: 0.5826
 1472/60000 [..............................] - ETA: 1:48 - loss: 1.2573 - categorical_accuracy: 0.5870
 1504/60000 [..............................] - ETA: 1:47 - loss: 1.2431 - categorical_accuracy: 0.5918
 1536/60000 [..............................] - ETA: 1:47 - loss: 1.2264 - categorical_accuracy: 0.5970
 1568/60000 [..............................] - ETA: 1:47 - loss: 1.2173 - categorical_accuracy: 0.5982
 1600/60000 [..............................] - ETA: 1:46 - loss: 1.2105 - categorical_accuracy: 0.6000
 1632/60000 [..............................] - ETA: 1:46 - loss: 1.2083 - categorical_accuracy: 0.5999
 1664/60000 [..............................] - ETA: 1:46 - loss: 1.1977 - categorical_accuracy: 0.6052
 1696/60000 [..............................] - ETA: 1:46 - loss: 1.1886 - categorical_accuracy: 0.6079
 1728/60000 [..............................] - ETA: 1:46 - loss: 1.1815 - categorical_accuracy: 0.6111
 1760/60000 [..............................] - ETA: 1:45 - loss: 1.1764 - categorical_accuracy: 0.6136
 1792/60000 [..............................] - ETA: 1:45 - loss: 1.1684 - categorical_accuracy: 0.6166
 1824/60000 [..............................] - ETA: 1:45 - loss: 1.1592 - categorical_accuracy: 0.6201
 1856/60000 [..............................] - ETA: 1:45 - loss: 1.1467 - categorical_accuracy: 0.6250
 1888/60000 [..............................] - ETA: 1:45 - loss: 1.1383 - categorical_accuracy: 0.6271
 1920/60000 [..............................] - ETA: 1:45 - loss: 1.1286 - categorical_accuracy: 0.6313
 1952/60000 [..............................] - ETA: 1:45 - loss: 1.1208 - categorical_accuracy: 0.6337
 1984/60000 [..............................] - ETA: 1:45 - loss: 1.1126 - categorical_accuracy: 0.6361
 2016/60000 [>.............................] - ETA: 1:45 - loss: 1.1081 - categorical_accuracy: 0.6384
 2048/60000 [>.............................] - ETA: 1:44 - loss: 1.1020 - categorical_accuracy: 0.6411
 2080/60000 [>.............................] - ETA: 1:44 - loss: 1.0899 - categorical_accuracy: 0.6462
 2112/60000 [>.............................] - ETA: 1:44 - loss: 1.0788 - categorical_accuracy: 0.6496
 2144/60000 [>.............................] - ETA: 1:44 - loss: 1.0744 - categorical_accuracy: 0.6516
 2176/60000 [>.............................] - ETA: 1:44 - loss: 1.0651 - categorical_accuracy: 0.6540
 2208/60000 [>.............................] - ETA: 1:44 - loss: 1.0591 - categorical_accuracy: 0.6558
 2240/60000 [>.............................] - ETA: 1:44 - loss: 1.0522 - categorical_accuracy: 0.6580
 2272/60000 [>.............................] - ETA: 1:43 - loss: 1.0472 - categorical_accuracy: 0.6585
 2304/60000 [>.............................] - ETA: 1:43 - loss: 1.0391 - categorical_accuracy: 0.6615
 2336/60000 [>.............................] - ETA: 1:43 - loss: 1.0297 - categorical_accuracy: 0.6652
 2368/60000 [>.............................] - ETA: 1:43 - loss: 1.0219 - categorical_accuracy: 0.6672
 2400/60000 [>.............................] - ETA: 1:43 - loss: 1.0136 - categorical_accuracy: 0.6692
 2432/60000 [>.............................] - ETA: 1:43 - loss: 1.0035 - categorical_accuracy: 0.6727
 2464/60000 [>.............................] - ETA: 1:42 - loss: 0.9966 - categorical_accuracy: 0.6749
 2496/60000 [>.............................] - ETA: 1:42 - loss: 0.9877 - categorical_accuracy: 0.6775
 2528/60000 [>.............................] - ETA: 1:42 - loss: 0.9794 - categorical_accuracy: 0.6804
 2560/60000 [>.............................] - ETA: 1:42 - loss: 0.9730 - categorical_accuracy: 0.6816
 2592/60000 [>.............................] - ETA: 1:42 - loss: 0.9659 - categorical_accuracy: 0.6844
 2624/60000 [>.............................] - ETA: 1:42 - loss: 0.9594 - categorical_accuracy: 0.6860
 2656/60000 [>.............................] - ETA: 1:42 - loss: 0.9608 - categorical_accuracy: 0.6879
 2688/60000 [>.............................] - ETA: 1:42 - loss: 0.9547 - categorical_accuracy: 0.6897
 2720/60000 [>.............................] - ETA: 1:41 - loss: 0.9505 - categorical_accuracy: 0.6912
 2752/60000 [>.............................] - ETA: 1:41 - loss: 0.9412 - categorical_accuracy: 0.6948
 2784/60000 [>.............................] - ETA: 1:41 - loss: 0.9339 - categorical_accuracy: 0.6972
 2816/60000 [>.............................] - ETA: 1:41 - loss: 0.9273 - categorical_accuracy: 0.6996
 2848/60000 [>.............................] - ETA: 1:41 - loss: 0.9191 - categorical_accuracy: 0.7022
 2880/60000 [>.............................] - ETA: 1:41 - loss: 0.9108 - categorical_accuracy: 0.7052
 2912/60000 [>.............................] - ETA: 1:41 - loss: 0.9063 - categorical_accuracy: 0.7067
 2944/60000 [>.............................] - ETA: 1:41 - loss: 0.9028 - categorical_accuracy: 0.7075
 2976/60000 [>.............................] - ETA: 1:41 - loss: 0.8980 - categorical_accuracy: 0.7083
 3008/60000 [>.............................] - ETA: 1:41 - loss: 0.8952 - categorical_accuracy: 0.7101
 3040/60000 [>.............................] - ETA: 1:41 - loss: 0.8912 - categorical_accuracy: 0.7112
 3072/60000 [>.............................] - ETA: 1:41 - loss: 0.8839 - categorical_accuracy: 0.7135
 3104/60000 [>.............................] - ETA: 1:41 - loss: 0.8758 - categorical_accuracy: 0.7162
 3136/60000 [>.............................] - ETA: 1:40 - loss: 0.8740 - categorical_accuracy: 0.7175
 3168/60000 [>.............................] - ETA: 1:40 - loss: 0.8699 - categorical_accuracy: 0.7184
 3200/60000 [>.............................] - ETA: 1:40 - loss: 0.8633 - categorical_accuracy: 0.7203
 3232/60000 [>.............................] - ETA: 1:40 - loss: 0.8556 - categorical_accuracy: 0.7231
 3264/60000 [>.............................] - ETA: 1:40 - loss: 0.8495 - categorical_accuracy: 0.7255
 3296/60000 [>.............................] - ETA: 1:40 - loss: 0.8500 - categorical_accuracy: 0.7257
 3328/60000 [>.............................] - ETA: 1:40 - loss: 0.8456 - categorical_accuracy: 0.7272
 3360/60000 [>.............................] - ETA: 1:40 - loss: 0.8409 - categorical_accuracy: 0.7292
 3392/60000 [>.............................] - ETA: 1:40 - loss: 0.8380 - categorical_accuracy: 0.7288
 3424/60000 [>.............................] - ETA: 1:40 - loss: 0.8317 - categorical_accuracy: 0.7304
 3456/60000 [>.............................] - ETA: 1:40 - loss: 0.8284 - categorical_accuracy: 0.7318
 3488/60000 [>.............................] - ETA: 1:40 - loss: 0.8226 - categorical_accuracy: 0.7339
 3520/60000 [>.............................] - ETA: 1:40 - loss: 0.8197 - categorical_accuracy: 0.7355
 3552/60000 [>.............................] - ETA: 1:40 - loss: 0.8170 - categorical_accuracy: 0.7362
 3584/60000 [>.............................] - ETA: 1:40 - loss: 0.8125 - categorical_accuracy: 0.7374
 3616/60000 [>.............................] - ETA: 1:40 - loss: 0.8093 - categorical_accuracy: 0.7378
 3648/60000 [>.............................] - ETA: 1:40 - loss: 0.8053 - categorical_accuracy: 0.7390
 3680/60000 [>.............................] - ETA: 1:39 - loss: 0.8038 - categorical_accuracy: 0.7402
 3712/60000 [>.............................] - ETA: 1:39 - loss: 0.7982 - categorical_accuracy: 0.7419
 3744/60000 [>.............................] - ETA: 1:39 - loss: 0.7931 - categorical_accuracy: 0.7436
 3776/60000 [>.............................] - ETA: 1:39 - loss: 0.7912 - categorical_accuracy: 0.7447
 3808/60000 [>.............................] - ETA: 1:39 - loss: 0.7869 - categorical_accuracy: 0.7461
 3840/60000 [>.............................] - ETA: 1:39 - loss: 0.7857 - categorical_accuracy: 0.7466
 3872/60000 [>.............................] - ETA: 1:39 - loss: 0.7827 - categorical_accuracy: 0.7482
 3904/60000 [>.............................] - ETA: 1:39 - loss: 0.7793 - categorical_accuracy: 0.7492
 3936/60000 [>.............................] - ETA: 1:39 - loss: 0.7794 - categorical_accuracy: 0.7500
 3968/60000 [>.............................] - ETA: 1:39 - loss: 0.7766 - categorical_accuracy: 0.7510
 4000/60000 [=>............................] - ETA: 1:39 - loss: 0.7750 - categorical_accuracy: 0.7517
 4032/60000 [=>............................] - ETA: 1:39 - loss: 0.7727 - categorical_accuracy: 0.7525
 4064/60000 [=>............................] - ETA: 1:39 - loss: 0.7715 - categorical_accuracy: 0.7527
 4096/60000 [=>............................] - ETA: 1:39 - loss: 0.7666 - categorical_accuracy: 0.7546
 4128/60000 [=>............................] - ETA: 1:39 - loss: 0.7627 - categorical_accuracy: 0.7561
 4160/60000 [=>............................] - ETA: 1:39 - loss: 0.7602 - categorical_accuracy: 0.7565
 4192/60000 [=>............................] - ETA: 1:39 - loss: 0.7583 - categorical_accuracy: 0.7574
 4224/60000 [=>............................] - ETA: 1:38 - loss: 0.7544 - categorical_accuracy: 0.7590
 4256/60000 [=>............................] - ETA: 1:38 - loss: 0.7501 - categorical_accuracy: 0.7606
 4288/60000 [=>............................] - ETA: 1:38 - loss: 0.7456 - categorical_accuracy: 0.7619
 4320/60000 [=>............................] - ETA: 1:38 - loss: 0.7433 - categorical_accuracy: 0.7627
 4352/60000 [=>............................] - ETA: 1:38 - loss: 0.7386 - categorical_accuracy: 0.7640
 4384/60000 [=>............................] - ETA: 1:38 - loss: 0.7345 - categorical_accuracy: 0.7653
 4416/60000 [=>............................] - ETA: 1:38 - loss: 0.7317 - categorical_accuracy: 0.7661
 4448/60000 [=>............................] - ETA: 1:38 - loss: 0.7300 - categorical_accuracy: 0.7671
 4480/60000 [=>............................] - ETA: 1:38 - loss: 0.7267 - categorical_accuracy: 0.7685
 4512/60000 [=>............................] - ETA: 1:38 - loss: 0.7231 - categorical_accuracy: 0.7697
 4544/60000 [=>............................] - ETA: 1:38 - loss: 0.7194 - categorical_accuracy: 0.7709
 4576/60000 [=>............................] - ETA: 1:37 - loss: 0.7163 - categorical_accuracy: 0.7719
 4608/60000 [=>............................] - ETA: 1:37 - loss: 0.7151 - categorical_accuracy: 0.7726
 4640/60000 [=>............................] - ETA: 1:38 - loss: 0.7163 - categorical_accuracy: 0.7724
 4672/60000 [=>............................] - ETA: 1:38 - loss: 0.7146 - categorical_accuracy: 0.7727
 4704/60000 [=>............................] - ETA: 1:37 - loss: 0.7112 - categorical_accuracy: 0.7736
 4736/60000 [=>............................] - ETA: 1:37 - loss: 0.7116 - categorical_accuracy: 0.7736
 4768/60000 [=>............................] - ETA: 1:37 - loss: 0.7086 - categorical_accuracy: 0.7747
 4800/60000 [=>............................] - ETA: 1:37 - loss: 0.7057 - categorical_accuracy: 0.7758
 4832/60000 [=>............................] - ETA: 1:37 - loss: 0.7021 - categorical_accuracy: 0.7771
 4864/60000 [=>............................] - ETA: 1:37 - loss: 0.6988 - categorical_accuracy: 0.7782
 4896/60000 [=>............................] - ETA: 1:37 - loss: 0.6955 - categorical_accuracy: 0.7792
 4928/60000 [=>............................] - ETA: 1:37 - loss: 0.6919 - categorical_accuracy: 0.7802
 4960/60000 [=>............................] - ETA: 1:37 - loss: 0.6891 - categorical_accuracy: 0.7810
 4992/60000 [=>............................] - ETA: 1:37 - loss: 0.6858 - categorical_accuracy: 0.7821
 5024/60000 [=>............................] - ETA: 1:37 - loss: 0.6843 - categorical_accuracy: 0.7826
 5056/60000 [=>............................] - ETA: 1:37 - loss: 0.6824 - categorical_accuracy: 0.7834
 5088/60000 [=>............................] - ETA: 1:37 - loss: 0.6798 - categorical_accuracy: 0.7844
 5120/60000 [=>............................] - ETA: 1:37 - loss: 0.6780 - categorical_accuracy: 0.7852
 5152/60000 [=>............................] - ETA: 1:37 - loss: 0.6752 - categorical_accuracy: 0.7861
 5184/60000 [=>............................] - ETA: 1:37 - loss: 0.6724 - categorical_accuracy: 0.7870
 5216/60000 [=>............................] - ETA: 1:37 - loss: 0.6701 - categorical_accuracy: 0.7876
 5248/60000 [=>............................] - ETA: 1:37 - loss: 0.6674 - categorical_accuracy: 0.7885
 5280/60000 [=>............................] - ETA: 1:37 - loss: 0.6649 - categorical_accuracy: 0.7892
 5312/60000 [=>............................] - ETA: 1:37 - loss: 0.6621 - categorical_accuracy: 0.7903
 5344/60000 [=>............................] - ETA: 1:37 - loss: 0.6604 - categorical_accuracy: 0.7906
 5376/60000 [=>............................] - ETA: 1:36 - loss: 0.6580 - categorical_accuracy: 0.7913
 5408/60000 [=>............................] - ETA: 1:36 - loss: 0.6550 - categorical_accuracy: 0.7923
 5440/60000 [=>............................] - ETA: 1:36 - loss: 0.6538 - categorical_accuracy: 0.7928
 5472/60000 [=>............................] - ETA: 1:36 - loss: 0.6514 - categorical_accuracy: 0.7935
 5504/60000 [=>............................] - ETA: 1:36 - loss: 0.6488 - categorical_accuracy: 0.7943
 5536/60000 [=>............................] - ETA: 1:36 - loss: 0.6463 - categorical_accuracy: 0.7953
 5568/60000 [=>............................] - ETA: 1:36 - loss: 0.6440 - categorical_accuracy: 0.7963
 5600/60000 [=>............................] - ETA: 1:36 - loss: 0.6421 - categorical_accuracy: 0.7968
 5632/60000 [=>............................] - ETA: 1:36 - loss: 0.6390 - categorical_accuracy: 0.7978
 5664/60000 [=>............................] - ETA: 1:36 - loss: 0.6380 - categorical_accuracy: 0.7984
 5696/60000 [=>............................] - ETA: 1:36 - loss: 0.6357 - categorical_accuracy: 0.7993
 5728/60000 [=>............................] - ETA: 1:36 - loss: 0.6338 - categorical_accuracy: 0.7998
 5760/60000 [=>............................] - ETA: 1:36 - loss: 0.6314 - categorical_accuracy: 0.8005
 5792/60000 [=>............................] - ETA: 1:36 - loss: 0.6308 - categorical_accuracy: 0.8009
 5824/60000 [=>............................] - ETA: 1:36 - loss: 0.6294 - categorical_accuracy: 0.8017
 5856/60000 [=>............................] - ETA: 1:36 - loss: 0.6270 - categorical_accuracy: 0.8028
 5888/60000 [=>............................] - ETA: 1:36 - loss: 0.6245 - categorical_accuracy: 0.8035
 5920/60000 [=>............................] - ETA: 1:35 - loss: 0.6234 - categorical_accuracy: 0.8039
 5952/60000 [=>............................] - ETA: 1:35 - loss: 0.6224 - categorical_accuracy: 0.8043
 5984/60000 [=>............................] - ETA: 1:35 - loss: 0.6207 - categorical_accuracy: 0.8048
 6016/60000 [==>...........................] - ETA: 1:35 - loss: 0.6194 - categorical_accuracy: 0.8054
 6048/60000 [==>...........................] - ETA: 1:35 - loss: 0.6172 - categorical_accuracy: 0.8059
 6080/60000 [==>...........................] - ETA: 1:35 - loss: 0.6152 - categorical_accuracy: 0.8066
 6112/60000 [==>...........................] - ETA: 1:35 - loss: 0.6138 - categorical_accuracy: 0.8069
 6144/60000 [==>...........................] - ETA: 1:35 - loss: 0.6134 - categorical_accuracy: 0.8075
 6176/60000 [==>...........................] - ETA: 1:35 - loss: 0.6110 - categorical_accuracy: 0.8081
 6208/60000 [==>...........................] - ETA: 1:35 - loss: 0.6086 - categorical_accuracy: 0.8091
 6240/60000 [==>...........................] - ETA: 1:35 - loss: 0.6062 - categorical_accuracy: 0.8099
 6272/60000 [==>...........................] - ETA: 1:35 - loss: 0.6043 - categorical_accuracy: 0.8104
 6304/60000 [==>...........................] - ETA: 1:35 - loss: 0.6017 - categorical_accuracy: 0.8112
 6336/60000 [==>...........................] - ETA: 1:35 - loss: 0.6011 - categorical_accuracy: 0.8116
 6368/60000 [==>...........................] - ETA: 1:35 - loss: 0.5996 - categorical_accuracy: 0.8120
 6400/60000 [==>...........................] - ETA: 1:35 - loss: 0.5991 - categorical_accuracy: 0.8120
 6432/60000 [==>...........................] - ETA: 1:35 - loss: 0.5972 - categorical_accuracy: 0.8125
 6464/60000 [==>...........................] - ETA: 1:35 - loss: 0.5952 - categorical_accuracy: 0.8131
 6496/60000 [==>...........................] - ETA: 1:35 - loss: 0.5936 - categorical_accuracy: 0.8136
 6528/60000 [==>...........................] - ETA: 1:35 - loss: 0.5918 - categorical_accuracy: 0.8140
 6560/60000 [==>...........................] - ETA: 1:35 - loss: 0.5893 - categorical_accuracy: 0.8148
 6592/60000 [==>...........................] - ETA: 1:35 - loss: 0.5878 - categorical_accuracy: 0.8151
 6624/60000 [==>...........................] - ETA: 1:35 - loss: 0.5861 - categorical_accuracy: 0.8155
 6656/60000 [==>...........................] - ETA: 1:34 - loss: 0.5847 - categorical_accuracy: 0.8158
 6688/60000 [==>...........................] - ETA: 1:34 - loss: 0.5838 - categorical_accuracy: 0.8159
 6720/60000 [==>...........................] - ETA: 1:34 - loss: 0.5828 - categorical_accuracy: 0.8161
 6752/60000 [==>...........................] - ETA: 1:34 - loss: 0.5812 - categorical_accuracy: 0.8166
 6784/60000 [==>...........................] - ETA: 1:34 - loss: 0.5795 - categorical_accuracy: 0.8171
 6816/60000 [==>...........................] - ETA: 1:34 - loss: 0.5776 - categorical_accuracy: 0.8176
 6848/60000 [==>...........................] - ETA: 1:34 - loss: 0.5756 - categorical_accuracy: 0.8182
 6880/60000 [==>...........................] - ETA: 1:34 - loss: 0.5737 - categorical_accuracy: 0.8189
 6912/60000 [==>...........................] - ETA: 1:34 - loss: 0.5717 - categorical_accuracy: 0.8196
 6944/60000 [==>...........................] - ETA: 1:34 - loss: 0.5700 - categorical_accuracy: 0.8201
 6976/60000 [==>...........................] - ETA: 1:34 - loss: 0.5688 - categorical_accuracy: 0.8204
 7008/60000 [==>...........................] - ETA: 1:34 - loss: 0.5667 - categorical_accuracy: 0.8212
 7040/60000 [==>...........................] - ETA: 1:34 - loss: 0.5646 - categorical_accuracy: 0.8219
 7072/60000 [==>...........................] - ETA: 1:34 - loss: 0.5632 - categorical_accuracy: 0.8223
 7104/60000 [==>...........................] - ETA: 1:34 - loss: 0.5617 - categorical_accuracy: 0.8228
 7136/60000 [==>...........................] - ETA: 1:34 - loss: 0.5598 - categorical_accuracy: 0.8234
 7168/60000 [==>...........................] - ETA: 1:33 - loss: 0.5577 - categorical_accuracy: 0.8242
 7200/60000 [==>...........................] - ETA: 1:33 - loss: 0.5564 - categorical_accuracy: 0.8243
 7232/60000 [==>...........................] - ETA: 1:33 - loss: 0.5548 - categorical_accuracy: 0.8249
 7264/60000 [==>...........................] - ETA: 1:33 - loss: 0.5536 - categorical_accuracy: 0.8254
 7296/60000 [==>...........................] - ETA: 1:33 - loss: 0.5520 - categorical_accuracy: 0.8258
 7328/60000 [==>...........................] - ETA: 1:33 - loss: 0.5513 - categorical_accuracy: 0.8260
 7360/60000 [==>...........................] - ETA: 1:33 - loss: 0.5496 - categorical_accuracy: 0.8266
 7392/60000 [==>...........................] - ETA: 1:33 - loss: 0.5478 - categorical_accuracy: 0.8271
 7424/60000 [==>...........................] - ETA: 1:33 - loss: 0.5463 - categorical_accuracy: 0.8276
 7456/60000 [==>...........................] - ETA: 1:33 - loss: 0.5460 - categorical_accuracy: 0.8278
 7488/60000 [==>...........................] - ETA: 1:33 - loss: 0.5446 - categorical_accuracy: 0.8280
 7520/60000 [==>...........................] - ETA: 1:33 - loss: 0.5430 - categorical_accuracy: 0.8285
 7552/60000 [==>...........................] - ETA: 1:33 - loss: 0.5411 - categorical_accuracy: 0.8291
 7584/60000 [==>...........................] - ETA: 1:33 - loss: 0.5405 - categorical_accuracy: 0.8294
 7616/60000 [==>...........................] - ETA: 1:33 - loss: 0.5390 - categorical_accuracy: 0.8297
 7648/60000 [==>...........................] - ETA: 1:33 - loss: 0.5371 - categorical_accuracy: 0.8303
 7680/60000 [==>...........................] - ETA: 1:33 - loss: 0.5364 - categorical_accuracy: 0.8306
 7712/60000 [==>...........................] - ETA: 1:32 - loss: 0.5349 - categorical_accuracy: 0.8312
 7744/60000 [==>...........................] - ETA: 1:32 - loss: 0.5333 - categorical_accuracy: 0.8317
 7776/60000 [==>...........................] - ETA: 1:32 - loss: 0.5326 - categorical_accuracy: 0.8319
 7808/60000 [==>...........................] - ETA: 1:32 - loss: 0.5312 - categorical_accuracy: 0.8324
 7840/60000 [==>...........................] - ETA: 1:32 - loss: 0.5301 - categorical_accuracy: 0.8328
 7872/60000 [==>...........................] - ETA: 1:32 - loss: 0.5282 - categorical_accuracy: 0.8335
 7904/60000 [==>...........................] - ETA: 1:32 - loss: 0.5266 - categorical_accuracy: 0.8339
 7936/60000 [==>...........................] - ETA: 1:32 - loss: 0.5251 - categorical_accuracy: 0.8344
 7968/60000 [==>...........................] - ETA: 1:32 - loss: 0.5239 - categorical_accuracy: 0.8348
 8000/60000 [===>..........................] - ETA: 1:32 - loss: 0.5224 - categorical_accuracy: 0.8354
 8032/60000 [===>..........................] - ETA: 1:32 - loss: 0.5219 - categorical_accuracy: 0.8355
 8064/60000 [===>..........................] - ETA: 1:32 - loss: 0.5202 - categorical_accuracy: 0.8361
 8096/60000 [===>..........................] - ETA: 1:32 - loss: 0.5188 - categorical_accuracy: 0.8365
 8128/60000 [===>..........................] - ETA: 1:32 - loss: 0.5173 - categorical_accuracy: 0.8370
 8160/60000 [===>..........................] - ETA: 1:32 - loss: 0.5161 - categorical_accuracy: 0.8373
 8192/60000 [===>..........................] - ETA: 1:32 - loss: 0.5147 - categorical_accuracy: 0.8375
 8224/60000 [===>..........................] - ETA: 1:32 - loss: 0.5142 - categorical_accuracy: 0.8378
 8256/60000 [===>..........................] - ETA: 1:32 - loss: 0.5130 - categorical_accuracy: 0.8381
 8288/60000 [===>..........................] - ETA: 1:31 - loss: 0.5118 - categorical_accuracy: 0.8384
 8320/60000 [===>..........................] - ETA: 1:31 - loss: 0.5106 - categorical_accuracy: 0.8387
 8352/60000 [===>..........................] - ETA: 1:31 - loss: 0.5092 - categorical_accuracy: 0.8391
 8384/60000 [===>..........................] - ETA: 1:31 - loss: 0.5092 - categorical_accuracy: 0.8393
 8416/60000 [===>..........................] - ETA: 1:31 - loss: 0.5076 - categorical_accuracy: 0.8398
 8448/60000 [===>..........................] - ETA: 1:31 - loss: 0.5070 - categorical_accuracy: 0.8401
 8480/60000 [===>..........................] - ETA: 1:31 - loss: 0.5077 - categorical_accuracy: 0.8401
 8512/60000 [===>..........................] - ETA: 1:31 - loss: 0.5065 - categorical_accuracy: 0.8405
 8544/60000 [===>..........................] - ETA: 1:31 - loss: 0.5053 - categorical_accuracy: 0.8407
 8576/60000 [===>..........................] - ETA: 1:31 - loss: 0.5046 - categorical_accuracy: 0.8410
 8608/60000 [===>..........................] - ETA: 1:31 - loss: 0.5037 - categorical_accuracy: 0.8413
 8640/60000 [===>..........................] - ETA: 1:31 - loss: 0.5023 - categorical_accuracy: 0.8418
 8672/60000 [===>..........................] - ETA: 1:31 - loss: 0.5006 - categorical_accuracy: 0.8423
 8704/60000 [===>..........................] - ETA: 1:31 - loss: 0.4999 - categorical_accuracy: 0.8427
 8736/60000 [===>..........................] - ETA: 1:31 - loss: 0.4991 - categorical_accuracy: 0.8431
 8768/60000 [===>..........................] - ETA: 1:31 - loss: 0.4998 - categorical_accuracy: 0.8430
 8800/60000 [===>..........................] - ETA: 1:31 - loss: 0.4983 - categorical_accuracy: 0.8435
 8832/60000 [===>..........................] - ETA: 1:31 - loss: 0.4969 - categorical_accuracy: 0.8440
 8864/60000 [===>..........................] - ETA: 1:30 - loss: 0.4959 - categorical_accuracy: 0.8442
 8896/60000 [===>..........................] - ETA: 1:30 - loss: 0.4942 - categorical_accuracy: 0.8448
 8928/60000 [===>..........................] - ETA: 1:30 - loss: 0.4935 - categorical_accuracy: 0.8446
 8960/60000 [===>..........................] - ETA: 1:30 - loss: 0.4920 - categorical_accuracy: 0.8452
 8992/60000 [===>..........................] - ETA: 1:30 - loss: 0.4913 - categorical_accuracy: 0.8455
 9024/60000 [===>..........................] - ETA: 1:30 - loss: 0.4903 - categorical_accuracy: 0.8459
 9056/60000 [===>..........................] - ETA: 1:30 - loss: 0.4898 - categorical_accuracy: 0.8460
 9088/60000 [===>..........................] - ETA: 1:30 - loss: 0.4883 - categorical_accuracy: 0.8464
 9120/60000 [===>..........................] - ETA: 1:30 - loss: 0.4874 - categorical_accuracy: 0.8468
 9152/60000 [===>..........................] - ETA: 1:30 - loss: 0.4863 - categorical_accuracy: 0.8470
 9184/60000 [===>..........................] - ETA: 1:30 - loss: 0.4857 - categorical_accuracy: 0.8473
 9216/60000 [===>..........................] - ETA: 1:30 - loss: 0.4842 - categorical_accuracy: 0.8479
 9248/60000 [===>..........................] - ETA: 1:30 - loss: 0.4832 - categorical_accuracy: 0.8482
 9280/60000 [===>..........................] - ETA: 1:30 - loss: 0.4823 - categorical_accuracy: 0.8485
 9312/60000 [===>..........................] - ETA: 1:30 - loss: 0.4810 - categorical_accuracy: 0.8488
 9344/60000 [===>..........................] - ETA: 1:30 - loss: 0.4802 - categorical_accuracy: 0.8490
 9376/60000 [===>..........................] - ETA: 1:30 - loss: 0.4793 - categorical_accuracy: 0.8493
 9408/60000 [===>..........................] - ETA: 1:30 - loss: 0.4780 - categorical_accuracy: 0.8497
 9440/60000 [===>..........................] - ETA: 1:30 - loss: 0.4768 - categorical_accuracy: 0.8501
 9472/60000 [===>..........................] - ETA: 1:30 - loss: 0.4759 - categorical_accuracy: 0.8503
 9504/60000 [===>..........................] - ETA: 1:30 - loss: 0.4750 - categorical_accuracy: 0.8506
 9536/60000 [===>..........................] - ETA: 1:29 - loss: 0.4742 - categorical_accuracy: 0.8507
 9568/60000 [===>..........................] - ETA: 1:29 - loss: 0.4732 - categorical_accuracy: 0.8510
 9600/60000 [===>..........................] - ETA: 1:29 - loss: 0.4720 - categorical_accuracy: 0.8512
 9632/60000 [===>..........................] - ETA: 1:29 - loss: 0.4719 - categorical_accuracy: 0.8514
 9664/60000 [===>..........................] - ETA: 1:29 - loss: 0.4705 - categorical_accuracy: 0.8519
 9696/60000 [===>..........................] - ETA: 1:29 - loss: 0.4696 - categorical_accuracy: 0.8522
 9728/60000 [===>..........................] - ETA: 1:29 - loss: 0.4682 - categorical_accuracy: 0.8527
 9760/60000 [===>..........................] - ETA: 1:29 - loss: 0.4680 - categorical_accuracy: 0.8529
 9792/60000 [===>..........................] - ETA: 1:29 - loss: 0.4672 - categorical_accuracy: 0.8529
 9824/60000 [===>..........................] - ETA: 1:29 - loss: 0.4664 - categorical_accuracy: 0.8533
 9856/60000 [===>..........................] - ETA: 1:29 - loss: 0.4654 - categorical_accuracy: 0.8536
 9888/60000 [===>..........................] - ETA: 1:29 - loss: 0.4645 - categorical_accuracy: 0.8540
 9920/60000 [===>..........................] - ETA: 1:29 - loss: 0.4634 - categorical_accuracy: 0.8542
 9952/60000 [===>..........................] - ETA: 1:29 - loss: 0.4632 - categorical_accuracy: 0.8544
 9984/60000 [===>..........................] - ETA: 1:29 - loss: 0.4621 - categorical_accuracy: 0.8548
10016/60000 [====>.........................] - ETA: 1:29 - loss: 0.4612 - categorical_accuracy: 0.8550
10048/60000 [====>.........................] - ETA: 1:29 - loss: 0.4613 - categorical_accuracy: 0.8552
10080/60000 [====>.........................] - ETA: 1:29 - loss: 0.4611 - categorical_accuracy: 0.8554
10112/60000 [====>.........................] - ETA: 1:28 - loss: 0.4601 - categorical_accuracy: 0.8556
10144/60000 [====>.........................] - ETA: 1:28 - loss: 0.4591 - categorical_accuracy: 0.8559
10176/60000 [====>.........................] - ETA: 1:28 - loss: 0.4581 - categorical_accuracy: 0.8562
10208/60000 [====>.........................] - ETA: 1:28 - loss: 0.4570 - categorical_accuracy: 0.8565
10240/60000 [====>.........................] - ETA: 1:28 - loss: 0.4559 - categorical_accuracy: 0.8568
10272/60000 [====>.........................] - ETA: 1:28 - loss: 0.4548 - categorical_accuracy: 0.8572
10304/60000 [====>.........................] - ETA: 1:28 - loss: 0.4544 - categorical_accuracy: 0.8573
10336/60000 [====>.........................] - ETA: 1:28 - loss: 0.4534 - categorical_accuracy: 0.8575
10368/60000 [====>.........................] - ETA: 1:28 - loss: 0.4523 - categorical_accuracy: 0.8578
10400/60000 [====>.........................] - ETA: 1:28 - loss: 0.4516 - categorical_accuracy: 0.8579
10432/60000 [====>.........................] - ETA: 1:28 - loss: 0.4504 - categorical_accuracy: 0.8582
10464/60000 [====>.........................] - ETA: 1:28 - loss: 0.4496 - categorical_accuracy: 0.8585
10496/60000 [====>.........................] - ETA: 1:28 - loss: 0.4486 - categorical_accuracy: 0.8587
10528/60000 [====>.........................] - ETA: 1:28 - loss: 0.4480 - categorical_accuracy: 0.8589
10560/60000 [====>.........................] - ETA: 1:28 - loss: 0.4473 - categorical_accuracy: 0.8591
10592/60000 [====>.........................] - ETA: 1:28 - loss: 0.4466 - categorical_accuracy: 0.8592
10624/60000 [====>.........................] - ETA: 1:28 - loss: 0.4455 - categorical_accuracy: 0.8596
10656/60000 [====>.........................] - ETA: 1:27 - loss: 0.4451 - categorical_accuracy: 0.8595
10688/60000 [====>.........................] - ETA: 1:27 - loss: 0.4443 - categorical_accuracy: 0.8598
10720/60000 [====>.........................] - ETA: 1:27 - loss: 0.4432 - categorical_accuracy: 0.8603
10752/60000 [====>.........................] - ETA: 1:27 - loss: 0.4421 - categorical_accuracy: 0.8606
10784/60000 [====>.........................] - ETA: 1:27 - loss: 0.4423 - categorical_accuracy: 0.8607
10816/60000 [====>.........................] - ETA: 1:27 - loss: 0.4414 - categorical_accuracy: 0.8610
10848/60000 [====>.........................] - ETA: 1:27 - loss: 0.4405 - categorical_accuracy: 0.8614
10880/60000 [====>.........................] - ETA: 1:27 - loss: 0.4395 - categorical_accuracy: 0.8618
10912/60000 [====>.........................] - ETA: 1:27 - loss: 0.4387 - categorical_accuracy: 0.8620
10944/60000 [====>.........................] - ETA: 1:27 - loss: 0.4376 - categorical_accuracy: 0.8624
10976/60000 [====>.........................] - ETA: 1:27 - loss: 0.4370 - categorical_accuracy: 0.8627
11008/60000 [====>.........................] - ETA: 1:27 - loss: 0.4359 - categorical_accuracy: 0.8631
11040/60000 [====>.........................] - ETA: 1:27 - loss: 0.4353 - categorical_accuracy: 0.8634
11072/60000 [====>.........................] - ETA: 1:27 - loss: 0.4349 - categorical_accuracy: 0.8635
11104/60000 [====>.........................] - ETA: 1:27 - loss: 0.4345 - categorical_accuracy: 0.8635
11136/60000 [====>.........................] - ETA: 1:27 - loss: 0.4336 - categorical_accuracy: 0.8637
11168/60000 [====>.........................] - ETA: 1:27 - loss: 0.4325 - categorical_accuracy: 0.8641
11200/60000 [====>.........................] - ETA: 1:26 - loss: 0.4319 - categorical_accuracy: 0.8641
11232/60000 [====>.........................] - ETA: 1:26 - loss: 0.4309 - categorical_accuracy: 0.8644
11264/60000 [====>.........................] - ETA: 1:26 - loss: 0.4300 - categorical_accuracy: 0.8647
11296/60000 [====>.........................] - ETA: 1:26 - loss: 0.4294 - categorical_accuracy: 0.8649
11328/60000 [====>.........................] - ETA: 1:26 - loss: 0.4285 - categorical_accuracy: 0.8651
11360/60000 [====>.........................] - ETA: 1:26 - loss: 0.4282 - categorical_accuracy: 0.8652
11392/60000 [====>.........................] - ETA: 1:26 - loss: 0.4275 - categorical_accuracy: 0.8655
11424/60000 [====>.........................] - ETA: 1:26 - loss: 0.4265 - categorical_accuracy: 0.8659
11456/60000 [====>.........................] - ETA: 1:26 - loss: 0.4257 - categorical_accuracy: 0.8662
11488/60000 [====>.........................] - ETA: 1:26 - loss: 0.4258 - categorical_accuracy: 0.8662
11520/60000 [====>.........................] - ETA: 1:26 - loss: 0.4254 - categorical_accuracy: 0.8665
11552/60000 [====>.........................] - ETA: 1:26 - loss: 0.4243 - categorical_accuracy: 0.8669
11584/60000 [====>.........................] - ETA: 1:26 - loss: 0.4233 - categorical_accuracy: 0.8672
11616/60000 [====>.........................] - ETA: 1:26 - loss: 0.4224 - categorical_accuracy: 0.8674
11648/60000 [====>.........................] - ETA: 1:26 - loss: 0.4217 - categorical_accuracy: 0.8676
11680/60000 [====>.........................] - ETA: 1:26 - loss: 0.4208 - categorical_accuracy: 0.8680
11712/60000 [====>.........................] - ETA: 1:26 - loss: 0.4209 - categorical_accuracy: 0.8681
11744/60000 [====>.........................] - ETA: 1:26 - loss: 0.4214 - categorical_accuracy: 0.8679
11776/60000 [====>.........................] - ETA: 1:26 - loss: 0.4213 - categorical_accuracy: 0.8678
11808/60000 [====>.........................] - ETA: 1:26 - loss: 0.4208 - categorical_accuracy: 0.8679
11840/60000 [====>.........................] - ETA: 1:25 - loss: 0.4200 - categorical_accuracy: 0.8682
11872/60000 [====>.........................] - ETA: 1:25 - loss: 0.4194 - categorical_accuracy: 0.8684
11904/60000 [====>.........................] - ETA: 1:25 - loss: 0.4198 - categorical_accuracy: 0.8685
11936/60000 [====>.........................] - ETA: 1:25 - loss: 0.4192 - categorical_accuracy: 0.8686
11968/60000 [====>.........................] - ETA: 1:25 - loss: 0.4183 - categorical_accuracy: 0.8689
12000/60000 [=====>........................] - ETA: 1:25 - loss: 0.4179 - categorical_accuracy: 0.8691
12032/60000 [=====>........................] - ETA: 1:25 - loss: 0.4175 - categorical_accuracy: 0.8693
12064/60000 [=====>........................] - ETA: 1:25 - loss: 0.4169 - categorical_accuracy: 0.8694
12096/60000 [=====>........................] - ETA: 1:25 - loss: 0.4159 - categorical_accuracy: 0.8698
12128/60000 [=====>........................] - ETA: 1:25 - loss: 0.4154 - categorical_accuracy: 0.8698
12160/60000 [=====>........................] - ETA: 1:25 - loss: 0.4146 - categorical_accuracy: 0.8701
12192/60000 [=====>........................] - ETA: 1:25 - loss: 0.4139 - categorical_accuracy: 0.8702
12224/60000 [=====>........................] - ETA: 1:25 - loss: 0.4134 - categorical_accuracy: 0.8704
12256/60000 [=====>........................] - ETA: 1:25 - loss: 0.4128 - categorical_accuracy: 0.8707
12288/60000 [=====>........................] - ETA: 1:25 - loss: 0.4123 - categorical_accuracy: 0.8708
12320/60000 [=====>........................] - ETA: 1:25 - loss: 0.4114 - categorical_accuracy: 0.8712
12352/60000 [=====>........................] - ETA: 1:25 - loss: 0.4104 - categorical_accuracy: 0.8715
12384/60000 [=====>........................] - ETA: 1:25 - loss: 0.4097 - categorical_accuracy: 0.8717
12416/60000 [=====>........................] - ETA: 1:25 - loss: 0.4090 - categorical_accuracy: 0.8719
12448/60000 [=====>........................] - ETA: 1:25 - loss: 0.4084 - categorical_accuracy: 0.8719
12480/60000 [=====>........................] - ETA: 1:24 - loss: 0.4076 - categorical_accuracy: 0.8722
12512/60000 [=====>........................] - ETA: 1:24 - loss: 0.4070 - categorical_accuracy: 0.8723
12544/60000 [=====>........................] - ETA: 1:24 - loss: 0.4067 - categorical_accuracy: 0.8724
12576/60000 [=====>........................] - ETA: 1:24 - loss: 0.4063 - categorical_accuracy: 0.8725
12608/60000 [=====>........................] - ETA: 1:24 - loss: 0.4054 - categorical_accuracy: 0.8729
12640/60000 [=====>........................] - ETA: 1:24 - loss: 0.4048 - categorical_accuracy: 0.8730
12672/60000 [=====>........................] - ETA: 1:24 - loss: 0.4040 - categorical_accuracy: 0.8732
12704/60000 [=====>........................] - ETA: 1:24 - loss: 0.4034 - categorical_accuracy: 0.8733
12736/60000 [=====>........................] - ETA: 1:24 - loss: 0.4040 - categorical_accuracy: 0.8734
12768/60000 [=====>........................] - ETA: 1:24 - loss: 0.4035 - categorical_accuracy: 0.8734
12800/60000 [=====>........................] - ETA: 1:24 - loss: 0.4026 - categorical_accuracy: 0.8737
12832/60000 [=====>........................] - ETA: 1:24 - loss: 0.4022 - categorical_accuracy: 0.8739
12864/60000 [=====>........................] - ETA: 1:24 - loss: 0.4014 - categorical_accuracy: 0.8741
12896/60000 [=====>........................] - ETA: 1:24 - loss: 0.4009 - categorical_accuracy: 0.8744
12928/60000 [=====>........................] - ETA: 1:24 - loss: 0.4003 - categorical_accuracy: 0.8745
12960/60000 [=====>........................] - ETA: 1:24 - loss: 0.3997 - categorical_accuracy: 0.8747
12992/60000 [=====>........................] - ETA: 1:23 - loss: 0.3992 - categorical_accuracy: 0.8748
13024/60000 [=====>........................] - ETA: 1:23 - loss: 0.3987 - categorical_accuracy: 0.8749
13056/60000 [=====>........................] - ETA: 1:23 - loss: 0.3978 - categorical_accuracy: 0.8752
13088/60000 [=====>........................] - ETA: 1:23 - loss: 0.3970 - categorical_accuracy: 0.8755
13120/60000 [=====>........................] - ETA: 1:23 - loss: 0.3961 - categorical_accuracy: 0.8758
13152/60000 [=====>........................] - ETA: 1:23 - loss: 0.3959 - categorical_accuracy: 0.8758
13184/60000 [=====>........................] - ETA: 1:23 - loss: 0.3956 - categorical_accuracy: 0.8759
13216/60000 [=====>........................] - ETA: 1:23 - loss: 0.3952 - categorical_accuracy: 0.8761
13248/60000 [=====>........................] - ETA: 1:23 - loss: 0.3948 - categorical_accuracy: 0.8763
13280/60000 [=====>........................] - ETA: 1:23 - loss: 0.3944 - categorical_accuracy: 0.8765
13312/60000 [=====>........................] - ETA: 1:23 - loss: 0.3943 - categorical_accuracy: 0.8766
13344/60000 [=====>........................] - ETA: 1:23 - loss: 0.3941 - categorical_accuracy: 0.8766
13376/60000 [=====>........................] - ETA: 1:23 - loss: 0.3937 - categorical_accuracy: 0.8769
13408/60000 [=====>........................] - ETA: 1:23 - loss: 0.3931 - categorical_accuracy: 0.8771
13440/60000 [=====>........................] - ETA: 1:23 - loss: 0.3924 - categorical_accuracy: 0.8773
13472/60000 [=====>........................] - ETA: 1:23 - loss: 0.3919 - categorical_accuracy: 0.8774
13504/60000 [=====>........................] - ETA: 1:23 - loss: 0.3911 - categorical_accuracy: 0.8777
13536/60000 [=====>........................] - ETA: 1:23 - loss: 0.3904 - categorical_accuracy: 0.8780
13568/60000 [=====>........................] - ETA: 1:23 - loss: 0.3895 - categorical_accuracy: 0.8782
13600/60000 [=====>........................] - ETA: 1:22 - loss: 0.3887 - categorical_accuracy: 0.8785
13632/60000 [=====>........................] - ETA: 1:22 - loss: 0.3887 - categorical_accuracy: 0.8786
13664/60000 [=====>........................] - ETA: 1:22 - loss: 0.3880 - categorical_accuracy: 0.8788
13696/60000 [=====>........................] - ETA: 1:22 - loss: 0.3879 - categorical_accuracy: 0.8788
13728/60000 [=====>........................] - ETA: 1:22 - loss: 0.3879 - categorical_accuracy: 0.8789
13760/60000 [=====>........................] - ETA: 1:22 - loss: 0.3880 - categorical_accuracy: 0.8791
13792/60000 [=====>........................] - ETA: 1:22 - loss: 0.3874 - categorical_accuracy: 0.8793
13824/60000 [=====>........................] - ETA: 1:22 - loss: 0.3869 - categorical_accuracy: 0.8795
13856/60000 [=====>........................] - ETA: 1:22 - loss: 0.3865 - categorical_accuracy: 0.8796
13888/60000 [=====>........................] - ETA: 1:22 - loss: 0.3859 - categorical_accuracy: 0.8798
13920/60000 [=====>........................] - ETA: 1:22 - loss: 0.3853 - categorical_accuracy: 0.8799
13952/60000 [=====>........................] - ETA: 1:22 - loss: 0.3850 - categorical_accuracy: 0.8800
13984/60000 [=====>........................] - ETA: 1:22 - loss: 0.3847 - categorical_accuracy: 0.8801
14016/60000 [======>.......................] - ETA: 1:22 - loss: 0.3841 - categorical_accuracy: 0.8803
14048/60000 [======>.......................] - ETA: 1:22 - loss: 0.3834 - categorical_accuracy: 0.8806
14080/60000 [======>.......................] - ETA: 1:22 - loss: 0.3828 - categorical_accuracy: 0.8808
14112/60000 [======>.......................] - ETA: 1:22 - loss: 0.3822 - categorical_accuracy: 0.8810
14144/60000 [======>.......................] - ETA: 1:21 - loss: 0.3815 - categorical_accuracy: 0.8812
14176/60000 [======>.......................] - ETA: 1:21 - loss: 0.3809 - categorical_accuracy: 0.8814
14208/60000 [======>.......................] - ETA: 1:21 - loss: 0.3803 - categorical_accuracy: 0.8816
14240/60000 [======>.......................] - ETA: 1:21 - loss: 0.3799 - categorical_accuracy: 0.8817
14272/60000 [======>.......................] - ETA: 1:21 - loss: 0.3792 - categorical_accuracy: 0.8820
14304/60000 [======>.......................] - ETA: 1:21 - loss: 0.3789 - categorical_accuracy: 0.8821
14336/60000 [======>.......................] - ETA: 1:21 - loss: 0.3788 - categorical_accuracy: 0.8822
14368/60000 [======>.......................] - ETA: 1:21 - loss: 0.3784 - categorical_accuracy: 0.8822
14400/60000 [======>.......................] - ETA: 1:21 - loss: 0.3777 - categorical_accuracy: 0.8824
14432/60000 [======>.......................] - ETA: 1:21 - loss: 0.3773 - categorical_accuracy: 0.8826
14464/60000 [======>.......................] - ETA: 1:21 - loss: 0.3771 - categorical_accuracy: 0.8827
14496/60000 [======>.......................] - ETA: 1:21 - loss: 0.3770 - categorical_accuracy: 0.8829
14528/60000 [======>.......................] - ETA: 1:21 - loss: 0.3766 - categorical_accuracy: 0.8830
14560/60000 [======>.......................] - ETA: 1:21 - loss: 0.3759 - categorical_accuracy: 0.8832
14592/60000 [======>.......................] - ETA: 1:21 - loss: 0.3755 - categorical_accuracy: 0.8833
14624/60000 [======>.......................] - ETA: 1:21 - loss: 0.3750 - categorical_accuracy: 0.8834
14656/60000 [======>.......................] - ETA: 1:21 - loss: 0.3750 - categorical_accuracy: 0.8835
14688/60000 [======>.......................] - ETA: 1:21 - loss: 0.3748 - categorical_accuracy: 0.8836
14720/60000 [======>.......................] - ETA: 1:21 - loss: 0.3750 - categorical_accuracy: 0.8837
14752/60000 [======>.......................] - ETA: 1:20 - loss: 0.3749 - categorical_accuracy: 0.8837
14784/60000 [======>.......................] - ETA: 1:20 - loss: 0.3749 - categorical_accuracy: 0.8838
14816/60000 [======>.......................] - ETA: 1:20 - loss: 0.3741 - categorical_accuracy: 0.8840
14848/60000 [======>.......................] - ETA: 1:20 - loss: 0.3738 - categorical_accuracy: 0.8841
14880/60000 [======>.......................] - ETA: 1:20 - loss: 0.3738 - categorical_accuracy: 0.8841
14912/60000 [======>.......................] - ETA: 1:20 - loss: 0.3737 - categorical_accuracy: 0.8843
14944/60000 [======>.......................] - ETA: 1:20 - loss: 0.3732 - categorical_accuracy: 0.8844
14976/60000 [======>.......................] - ETA: 1:20 - loss: 0.3730 - categorical_accuracy: 0.8845
15008/60000 [======>.......................] - ETA: 1:20 - loss: 0.3728 - categorical_accuracy: 0.8846
15040/60000 [======>.......................] - ETA: 1:20 - loss: 0.3723 - categorical_accuracy: 0.8848
15072/60000 [======>.......................] - ETA: 1:20 - loss: 0.3721 - categorical_accuracy: 0.8849
15104/60000 [======>.......................] - ETA: 1:20 - loss: 0.3716 - categorical_accuracy: 0.8851
15136/60000 [======>.......................] - ETA: 1:20 - loss: 0.3710 - categorical_accuracy: 0.8852
15168/60000 [======>.......................] - ETA: 1:20 - loss: 0.3708 - categorical_accuracy: 0.8854
15200/60000 [======>.......................] - ETA: 1:20 - loss: 0.3711 - categorical_accuracy: 0.8854
15232/60000 [======>.......................] - ETA: 1:20 - loss: 0.3707 - categorical_accuracy: 0.8855
15264/60000 [======>.......................] - ETA: 1:20 - loss: 0.3706 - categorical_accuracy: 0.8855
15296/60000 [======>.......................] - ETA: 1:20 - loss: 0.3700 - categorical_accuracy: 0.8857
15328/60000 [======>.......................] - ETA: 1:20 - loss: 0.3694 - categorical_accuracy: 0.8858
15360/60000 [======>.......................] - ETA: 1:19 - loss: 0.3696 - categorical_accuracy: 0.8859
15392/60000 [======>.......................] - ETA: 1:19 - loss: 0.3690 - categorical_accuracy: 0.8861
15424/60000 [======>.......................] - ETA: 1:19 - loss: 0.3687 - categorical_accuracy: 0.8862
15456/60000 [======>.......................] - ETA: 1:19 - loss: 0.3680 - categorical_accuracy: 0.8865
15488/60000 [======>.......................] - ETA: 1:19 - loss: 0.3677 - categorical_accuracy: 0.8864
15520/60000 [======>.......................] - ETA: 1:19 - loss: 0.3671 - categorical_accuracy: 0.8867
15552/60000 [======>.......................] - ETA: 1:19 - loss: 0.3669 - categorical_accuracy: 0.8866
15584/60000 [======>.......................] - ETA: 1:19 - loss: 0.3662 - categorical_accuracy: 0.8868
15616/60000 [======>.......................] - ETA: 1:19 - loss: 0.3658 - categorical_accuracy: 0.8869
15648/60000 [======>.......................] - ETA: 1:19 - loss: 0.3654 - categorical_accuracy: 0.8870
15680/60000 [======>.......................] - ETA: 1:19 - loss: 0.3652 - categorical_accuracy: 0.8870
15712/60000 [======>.......................] - ETA: 1:19 - loss: 0.3647 - categorical_accuracy: 0.8871
15744/60000 [======>.......................] - ETA: 1:19 - loss: 0.3651 - categorical_accuracy: 0.8872
15776/60000 [======>.......................] - ETA: 1:19 - loss: 0.3644 - categorical_accuracy: 0.8874
15808/60000 [======>.......................] - ETA: 1:19 - loss: 0.3639 - categorical_accuracy: 0.8875
15840/60000 [======>.......................] - ETA: 1:19 - loss: 0.3638 - categorical_accuracy: 0.8875
15872/60000 [======>.......................] - ETA: 1:19 - loss: 0.3631 - categorical_accuracy: 0.8877
15904/60000 [======>.......................] - ETA: 1:19 - loss: 0.3626 - categorical_accuracy: 0.8878
15936/60000 [======>.......................] - ETA: 1:18 - loss: 0.3620 - categorical_accuracy: 0.8881
15968/60000 [======>.......................] - ETA: 1:18 - loss: 0.3614 - categorical_accuracy: 0.8882
16000/60000 [=======>......................] - ETA: 1:18 - loss: 0.3610 - categorical_accuracy: 0.8884
16032/60000 [=======>......................] - ETA: 1:18 - loss: 0.3604 - categorical_accuracy: 0.8885
16064/60000 [=======>......................] - ETA: 1:18 - loss: 0.3600 - categorical_accuracy: 0.8886
16096/60000 [=======>......................] - ETA: 1:18 - loss: 0.3594 - categorical_accuracy: 0.8889
16128/60000 [=======>......................] - ETA: 1:18 - loss: 0.3587 - categorical_accuracy: 0.8891
16160/60000 [=======>......................] - ETA: 1:18 - loss: 0.3581 - categorical_accuracy: 0.8893
16192/60000 [=======>......................] - ETA: 1:18 - loss: 0.3576 - categorical_accuracy: 0.8895
16224/60000 [=======>......................] - ETA: 1:18 - loss: 0.3571 - categorical_accuracy: 0.8897
16256/60000 [=======>......................] - ETA: 1:18 - loss: 0.3564 - categorical_accuracy: 0.8899
16288/60000 [=======>......................] - ETA: 1:18 - loss: 0.3558 - categorical_accuracy: 0.8900
16320/60000 [=======>......................] - ETA: 1:18 - loss: 0.3553 - categorical_accuracy: 0.8902
16352/60000 [=======>......................] - ETA: 1:18 - loss: 0.3550 - categorical_accuracy: 0.8903
16384/60000 [=======>......................] - ETA: 1:18 - loss: 0.3558 - categorical_accuracy: 0.8904
16416/60000 [=======>......................] - ETA: 1:18 - loss: 0.3556 - categorical_accuracy: 0.8904
16448/60000 [=======>......................] - ETA: 1:17 - loss: 0.3551 - categorical_accuracy: 0.8906
16480/60000 [=======>......................] - ETA: 1:17 - loss: 0.3547 - categorical_accuracy: 0.8907
16512/60000 [=======>......................] - ETA: 1:17 - loss: 0.3542 - categorical_accuracy: 0.8908
16544/60000 [=======>......................] - ETA: 1:17 - loss: 0.3537 - categorical_accuracy: 0.8909
16576/60000 [=======>......................] - ETA: 1:17 - loss: 0.3532 - categorical_accuracy: 0.8911
16608/60000 [=======>......................] - ETA: 1:17 - loss: 0.3526 - categorical_accuracy: 0.8913
16640/60000 [=======>......................] - ETA: 1:17 - loss: 0.3523 - categorical_accuracy: 0.8913
16672/60000 [=======>......................] - ETA: 1:17 - loss: 0.3518 - categorical_accuracy: 0.8916
16704/60000 [=======>......................] - ETA: 1:17 - loss: 0.3520 - categorical_accuracy: 0.8915
16736/60000 [=======>......................] - ETA: 1:17 - loss: 0.3520 - categorical_accuracy: 0.8916
16768/60000 [=======>......................] - ETA: 1:17 - loss: 0.3516 - categorical_accuracy: 0.8917
16800/60000 [=======>......................] - ETA: 1:17 - loss: 0.3510 - categorical_accuracy: 0.8918
16832/60000 [=======>......................] - ETA: 1:17 - loss: 0.3505 - categorical_accuracy: 0.8920
16864/60000 [=======>......................] - ETA: 1:17 - loss: 0.3499 - categorical_accuracy: 0.8922
16896/60000 [=======>......................] - ETA: 1:17 - loss: 0.3499 - categorical_accuracy: 0.8922
16928/60000 [=======>......................] - ETA: 1:17 - loss: 0.3496 - categorical_accuracy: 0.8923
16960/60000 [=======>......................] - ETA: 1:17 - loss: 0.3492 - categorical_accuracy: 0.8924
16992/60000 [=======>......................] - ETA: 1:17 - loss: 0.3487 - categorical_accuracy: 0.8925
17024/60000 [=======>......................] - ETA: 1:16 - loss: 0.3486 - categorical_accuracy: 0.8924
17056/60000 [=======>......................] - ETA: 1:16 - loss: 0.3482 - categorical_accuracy: 0.8925
17088/60000 [=======>......................] - ETA: 1:16 - loss: 0.3476 - categorical_accuracy: 0.8927
17120/60000 [=======>......................] - ETA: 1:16 - loss: 0.3470 - categorical_accuracy: 0.8929
17152/60000 [=======>......................] - ETA: 1:16 - loss: 0.3465 - categorical_accuracy: 0.8931
17184/60000 [=======>......................] - ETA: 1:16 - loss: 0.3461 - categorical_accuracy: 0.8932
17216/60000 [=======>......................] - ETA: 1:16 - loss: 0.3457 - categorical_accuracy: 0.8934
17248/60000 [=======>......................] - ETA: 1:16 - loss: 0.3452 - categorical_accuracy: 0.8935
17280/60000 [=======>......................] - ETA: 1:16 - loss: 0.3447 - categorical_accuracy: 0.8936
17312/60000 [=======>......................] - ETA: 1:16 - loss: 0.3442 - categorical_accuracy: 0.8938
17344/60000 [=======>......................] - ETA: 1:16 - loss: 0.3438 - categorical_accuracy: 0.8939
17376/60000 [=======>......................] - ETA: 1:16 - loss: 0.3434 - categorical_accuracy: 0.8940
17408/60000 [=======>......................] - ETA: 1:16 - loss: 0.3428 - categorical_accuracy: 0.8942
17440/60000 [=======>......................] - ETA: 1:16 - loss: 0.3423 - categorical_accuracy: 0.8944
17472/60000 [=======>......................] - ETA: 1:16 - loss: 0.3420 - categorical_accuracy: 0.8944
17504/60000 [=======>......................] - ETA: 1:16 - loss: 0.3421 - categorical_accuracy: 0.8945
17536/60000 [=======>......................] - ETA: 1:16 - loss: 0.3418 - categorical_accuracy: 0.8947
17568/60000 [=======>......................] - ETA: 1:15 - loss: 0.3417 - categorical_accuracy: 0.8946
17600/60000 [=======>......................] - ETA: 1:15 - loss: 0.3412 - categorical_accuracy: 0.8948
17632/60000 [=======>......................] - ETA: 1:15 - loss: 0.3409 - categorical_accuracy: 0.8949
17664/60000 [=======>......................] - ETA: 1:15 - loss: 0.3405 - categorical_accuracy: 0.8950
17696/60000 [=======>......................] - ETA: 1:15 - loss: 0.3401 - categorical_accuracy: 0.8951
17728/60000 [=======>......................] - ETA: 1:15 - loss: 0.3397 - categorical_accuracy: 0.8953
17760/60000 [=======>......................] - ETA: 1:15 - loss: 0.3394 - categorical_accuracy: 0.8953
17792/60000 [=======>......................] - ETA: 1:15 - loss: 0.3391 - categorical_accuracy: 0.8954
17824/60000 [=======>......................] - ETA: 1:15 - loss: 0.3387 - categorical_accuracy: 0.8955
17856/60000 [=======>......................] - ETA: 1:15 - loss: 0.3386 - categorical_accuracy: 0.8954
17888/60000 [=======>......................] - ETA: 1:15 - loss: 0.3381 - categorical_accuracy: 0.8955
17920/60000 [=======>......................] - ETA: 1:15 - loss: 0.3376 - categorical_accuracy: 0.8957
17952/60000 [=======>......................] - ETA: 1:15 - loss: 0.3371 - categorical_accuracy: 0.8959
17984/60000 [=======>......................] - ETA: 1:15 - loss: 0.3372 - categorical_accuracy: 0.8959
18016/60000 [========>.....................] - ETA: 1:15 - loss: 0.3368 - categorical_accuracy: 0.8960
18048/60000 [========>.....................] - ETA: 1:15 - loss: 0.3364 - categorical_accuracy: 0.8961
18080/60000 [========>.....................] - ETA: 1:15 - loss: 0.3362 - categorical_accuracy: 0.8961
18112/60000 [========>.....................] - ETA: 1:15 - loss: 0.3358 - categorical_accuracy: 0.8962
18144/60000 [========>.....................] - ETA: 1:14 - loss: 0.3355 - categorical_accuracy: 0.8963
18176/60000 [========>.....................] - ETA: 1:14 - loss: 0.3354 - categorical_accuracy: 0.8963
18208/60000 [========>.....................] - ETA: 1:14 - loss: 0.3352 - categorical_accuracy: 0.8964
18240/60000 [========>.....................] - ETA: 1:14 - loss: 0.3350 - categorical_accuracy: 0.8964
18272/60000 [========>.....................] - ETA: 1:14 - loss: 0.3348 - categorical_accuracy: 0.8965
18304/60000 [========>.....................] - ETA: 1:14 - loss: 0.3348 - categorical_accuracy: 0.8966
18336/60000 [========>.....................] - ETA: 1:14 - loss: 0.3351 - categorical_accuracy: 0.8966
18368/60000 [========>.....................] - ETA: 1:14 - loss: 0.3346 - categorical_accuracy: 0.8968
18400/60000 [========>.....................] - ETA: 1:14 - loss: 0.3345 - categorical_accuracy: 0.8968
18432/60000 [========>.....................] - ETA: 1:14 - loss: 0.3341 - categorical_accuracy: 0.8969
18464/60000 [========>.....................] - ETA: 1:14 - loss: 0.3341 - categorical_accuracy: 0.8968
18496/60000 [========>.....................] - ETA: 1:14 - loss: 0.3341 - categorical_accuracy: 0.8967
18528/60000 [========>.....................] - ETA: 1:14 - loss: 0.3336 - categorical_accuracy: 0.8969
18560/60000 [========>.....................] - ETA: 1:14 - loss: 0.3333 - categorical_accuracy: 0.8969
18592/60000 [========>.....................] - ETA: 1:14 - loss: 0.3330 - categorical_accuracy: 0.8971
18624/60000 [========>.....................] - ETA: 1:14 - loss: 0.3327 - categorical_accuracy: 0.8972
18656/60000 [========>.....................] - ETA: 1:14 - loss: 0.3323 - categorical_accuracy: 0.8973
18688/60000 [========>.....................] - ETA: 1:14 - loss: 0.3319 - categorical_accuracy: 0.8975
18720/60000 [========>.....................] - ETA: 1:14 - loss: 0.3315 - categorical_accuracy: 0.8976
18752/60000 [========>.....................] - ETA: 1:13 - loss: 0.3309 - categorical_accuracy: 0.8978
18784/60000 [========>.....................] - ETA: 1:13 - loss: 0.3308 - categorical_accuracy: 0.8979
18816/60000 [========>.....................] - ETA: 1:13 - loss: 0.3305 - categorical_accuracy: 0.8981
18848/60000 [========>.....................] - ETA: 1:13 - loss: 0.3300 - categorical_accuracy: 0.8982
18880/60000 [========>.....................] - ETA: 1:13 - loss: 0.3296 - categorical_accuracy: 0.8984
18912/60000 [========>.....................] - ETA: 1:13 - loss: 0.3295 - categorical_accuracy: 0.8984
18944/60000 [========>.....................] - ETA: 1:13 - loss: 0.3296 - categorical_accuracy: 0.8985
18976/60000 [========>.....................] - ETA: 1:13 - loss: 0.3292 - categorical_accuracy: 0.8986
19008/60000 [========>.....................] - ETA: 1:13 - loss: 0.3287 - categorical_accuracy: 0.8987
19040/60000 [========>.....................] - ETA: 1:13 - loss: 0.3282 - categorical_accuracy: 0.8989
19072/60000 [========>.....................] - ETA: 1:13 - loss: 0.3278 - categorical_accuracy: 0.8990
19104/60000 [========>.....................] - ETA: 1:13 - loss: 0.3280 - categorical_accuracy: 0.8990
19136/60000 [========>.....................] - ETA: 1:13 - loss: 0.3277 - categorical_accuracy: 0.8991
19168/60000 [========>.....................] - ETA: 1:13 - loss: 0.3274 - categorical_accuracy: 0.8993
19200/60000 [========>.....................] - ETA: 1:13 - loss: 0.3269 - categorical_accuracy: 0.8994
19232/60000 [========>.....................] - ETA: 1:13 - loss: 0.3264 - categorical_accuracy: 0.8995
19264/60000 [========>.....................] - ETA: 1:12 - loss: 0.3262 - categorical_accuracy: 0.8996
19296/60000 [========>.....................] - ETA: 1:12 - loss: 0.3258 - categorical_accuracy: 0.8997
19328/60000 [========>.....................] - ETA: 1:12 - loss: 0.3253 - categorical_accuracy: 0.8998
19360/60000 [========>.....................] - ETA: 1:12 - loss: 0.3249 - categorical_accuracy: 0.8999
19392/60000 [========>.....................] - ETA: 1:12 - loss: 0.3245 - categorical_accuracy: 0.9001
19424/60000 [========>.....................] - ETA: 1:12 - loss: 0.3242 - categorical_accuracy: 0.9002
19456/60000 [========>.....................] - ETA: 1:12 - loss: 0.3238 - categorical_accuracy: 0.9003
19488/60000 [========>.....................] - ETA: 1:12 - loss: 0.3237 - categorical_accuracy: 0.9003
19520/60000 [========>.....................] - ETA: 1:12 - loss: 0.3233 - categorical_accuracy: 0.9005
19552/60000 [========>.....................] - ETA: 1:12 - loss: 0.3232 - categorical_accuracy: 0.9005
19584/60000 [========>.....................] - ETA: 1:12 - loss: 0.3229 - categorical_accuracy: 0.9006
19616/60000 [========>.....................] - ETA: 1:12 - loss: 0.3226 - categorical_accuracy: 0.9007
19648/60000 [========>.....................] - ETA: 1:12 - loss: 0.3221 - categorical_accuracy: 0.9009
19680/60000 [========>.....................] - ETA: 1:12 - loss: 0.3217 - categorical_accuracy: 0.9010
19712/60000 [========>.....................] - ETA: 1:12 - loss: 0.3217 - categorical_accuracy: 0.9010
19744/60000 [========>.....................] - ETA: 1:12 - loss: 0.3212 - categorical_accuracy: 0.9012
19776/60000 [========>.....................] - ETA: 1:12 - loss: 0.3207 - categorical_accuracy: 0.9013
19808/60000 [========>.....................] - ETA: 1:12 - loss: 0.3203 - categorical_accuracy: 0.9015
19840/60000 [========>.....................] - ETA: 1:11 - loss: 0.3200 - categorical_accuracy: 0.9016
19872/60000 [========>.....................] - ETA: 1:11 - loss: 0.3197 - categorical_accuracy: 0.9017
19904/60000 [========>.....................] - ETA: 1:11 - loss: 0.3195 - categorical_accuracy: 0.9018
19936/60000 [========>.....................] - ETA: 1:11 - loss: 0.3193 - categorical_accuracy: 0.9018
19968/60000 [========>.....................] - ETA: 1:11 - loss: 0.3191 - categorical_accuracy: 0.9017
20000/60000 [=========>....................] - ETA: 1:11 - loss: 0.3187 - categorical_accuracy: 0.9019
20032/60000 [=========>....................] - ETA: 1:11 - loss: 0.3186 - categorical_accuracy: 0.9020
20064/60000 [=========>....................] - ETA: 1:11 - loss: 0.3182 - categorical_accuracy: 0.9021
20096/60000 [=========>....................] - ETA: 1:11 - loss: 0.3181 - categorical_accuracy: 0.9021
20128/60000 [=========>....................] - ETA: 1:11 - loss: 0.3176 - categorical_accuracy: 0.9022
20160/60000 [=========>....................] - ETA: 1:11 - loss: 0.3172 - categorical_accuracy: 0.9023
20192/60000 [=========>....................] - ETA: 1:11 - loss: 0.3169 - categorical_accuracy: 0.9024
20224/60000 [=========>....................] - ETA: 1:11 - loss: 0.3165 - categorical_accuracy: 0.9026
20256/60000 [=========>....................] - ETA: 1:11 - loss: 0.3162 - categorical_accuracy: 0.9027
20288/60000 [=========>....................] - ETA: 1:11 - loss: 0.3159 - categorical_accuracy: 0.9028
20320/60000 [=========>....................] - ETA: 1:11 - loss: 0.3155 - categorical_accuracy: 0.9029
20352/60000 [=========>....................] - ETA: 1:11 - loss: 0.3151 - categorical_accuracy: 0.9030
20384/60000 [=========>....................] - ETA: 1:11 - loss: 0.3148 - categorical_accuracy: 0.9031
20416/60000 [=========>....................] - ETA: 1:10 - loss: 0.3143 - categorical_accuracy: 0.9033
20448/60000 [=========>....................] - ETA: 1:10 - loss: 0.3141 - categorical_accuracy: 0.9034
20480/60000 [=========>....................] - ETA: 1:10 - loss: 0.3141 - categorical_accuracy: 0.9033
20544/60000 [=========>....................] - ETA: 1:10 - loss: 0.3135 - categorical_accuracy: 0.9034
20576/60000 [=========>....................] - ETA: 1:10 - loss: 0.3133 - categorical_accuracy: 0.9034
20608/60000 [=========>....................] - ETA: 1:10 - loss: 0.3131 - categorical_accuracy: 0.9034
20640/60000 [=========>....................] - ETA: 1:10 - loss: 0.3129 - categorical_accuracy: 0.9035
20672/60000 [=========>....................] - ETA: 1:10 - loss: 0.3126 - categorical_accuracy: 0.9035
20704/60000 [=========>....................] - ETA: 1:10 - loss: 0.3126 - categorical_accuracy: 0.9035
20736/60000 [=========>....................] - ETA: 1:10 - loss: 0.3123 - categorical_accuracy: 0.9036
20768/60000 [=========>....................] - ETA: 1:10 - loss: 0.3120 - categorical_accuracy: 0.9037
20800/60000 [=========>....................] - ETA: 1:10 - loss: 0.3120 - categorical_accuracy: 0.9037
20832/60000 [=========>....................] - ETA: 1:10 - loss: 0.3117 - categorical_accuracy: 0.9038
20864/60000 [=========>....................] - ETA: 1:10 - loss: 0.3115 - categorical_accuracy: 0.9038
20896/60000 [=========>....................] - ETA: 1:10 - loss: 0.3111 - categorical_accuracy: 0.9039
20928/60000 [=========>....................] - ETA: 1:10 - loss: 0.3112 - categorical_accuracy: 0.9039
20960/60000 [=========>....................] - ETA: 1:09 - loss: 0.3110 - categorical_accuracy: 0.9040
20992/60000 [=========>....................] - ETA: 1:09 - loss: 0.3107 - categorical_accuracy: 0.9041
21024/60000 [=========>....................] - ETA: 1:09 - loss: 0.3103 - categorical_accuracy: 0.9042
21056/60000 [=========>....................] - ETA: 1:09 - loss: 0.3099 - categorical_accuracy: 0.9043
21088/60000 [=========>....................] - ETA: 1:09 - loss: 0.3098 - categorical_accuracy: 0.9043
21120/60000 [=========>....................] - ETA: 1:09 - loss: 0.3093 - categorical_accuracy: 0.9044
21152/60000 [=========>....................] - ETA: 1:09 - loss: 0.3090 - categorical_accuracy: 0.9045
21184/60000 [=========>....................] - ETA: 1:09 - loss: 0.3092 - categorical_accuracy: 0.9045
21216/60000 [=========>....................] - ETA: 1:09 - loss: 0.3088 - categorical_accuracy: 0.9046
21248/60000 [=========>....................] - ETA: 1:09 - loss: 0.3085 - categorical_accuracy: 0.9046
21312/60000 [=========>....................] - ETA: 1:09 - loss: 0.3078 - categorical_accuracy: 0.9048
21344/60000 [=========>....................] - ETA: 1:09 - loss: 0.3075 - categorical_accuracy: 0.9049
21376/60000 [=========>....................] - ETA: 1:09 - loss: 0.3074 - categorical_accuracy: 0.9049
21408/60000 [=========>....................] - ETA: 1:09 - loss: 0.3071 - categorical_accuracy: 0.9049
21440/60000 [=========>....................] - ETA: 1:09 - loss: 0.3071 - categorical_accuracy: 0.9049
21472/60000 [=========>....................] - ETA: 1:08 - loss: 0.3068 - categorical_accuracy: 0.9051
21504/60000 [=========>....................] - ETA: 1:08 - loss: 0.3065 - categorical_accuracy: 0.9052
21536/60000 [=========>....................] - ETA: 1:08 - loss: 0.3062 - categorical_accuracy: 0.9053
21568/60000 [=========>....................] - ETA: 1:08 - loss: 0.3059 - categorical_accuracy: 0.9054
21600/60000 [=========>....................] - ETA: 1:08 - loss: 0.3056 - categorical_accuracy: 0.9055
21632/60000 [=========>....................] - ETA: 1:08 - loss: 0.3053 - categorical_accuracy: 0.9056
21664/60000 [=========>....................] - ETA: 1:08 - loss: 0.3049 - categorical_accuracy: 0.9057
21696/60000 [=========>....................] - ETA: 1:08 - loss: 0.3047 - categorical_accuracy: 0.9058
21728/60000 [=========>....................] - ETA: 1:08 - loss: 0.3044 - categorical_accuracy: 0.9059
21760/60000 [=========>....................] - ETA: 1:08 - loss: 0.3040 - categorical_accuracy: 0.9060
21792/60000 [=========>....................] - ETA: 1:08 - loss: 0.3038 - categorical_accuracy: 0.9061
21856/60000 [=========>....................] - ETA: 1:08 - loss: 0.3034 - categorical_accuracy: 0.9062
21920/60000 [=========>....................] - ETA: 1:08 - loss: 0.3028 - categorical_accuracy: 0.9064
21952/60000 [=========>....................] - ETA: 1:08 - loss: 0.3024 - categorical_accuracy: 0.9066
21984/60000 [=========>....................] - ETA: 1:08 - loss: 0.3023 - categorical_accuracy: 0.9066
22016/60000 [==========>...................] - ETA: 1:07 - loss: 0.3019 - categorical_accuracy: 0.9067
22048/60000 [==========>...................] - ETA: 1:07 - loss: 0.3016 - categorical_accuracy: 0.9068
22080/60000 [==========>...................] - ETA: 1:07 - loss: 0.3014 - categorical_accuracy: 0.9069
22112/60000 [==========>...................] - ETA: 1:07 - loss: 0.3012 - categorical_accuracy: 0.9070
22144/60000 [==========>...................] - ETA: 1:07 - loss: 0.3016 - categorical_accuracy: 0.9070
22176/60000 [==========>...................] - ETA: 1:07 - loss: 0.3016 - categorical_accuracy: 0.9069
22208/60000 [==========>...................] - ETA: 1:07 - loss: 0.3012 - categorical_accuracy: 0.9071
22240/60000 [==========>...................] - ETA: 1:07 - loss: 0.3014 - categorical_accuracy: 0.9070
22272/60000 [==========>...................] - ETA: 1:07 - loss: 0.3010 - categorical_accuracy: 0.9071
22304/60000 [==========>...................] - ETA: 1:07 - loss: 0.3008 - categorical_accuracy: 0.9072
22336/60000 [==========>...................] - ETA: 1:07 - loss: 0.3004 - categorical_accuracy: 0.9074
22368/60000 [==========>...................] - ETA: 1:07 - loss: 0.3001 - categorical_accuracy: 0.9075
22400/60000 [==========>...................] - ETA: 1:07 - loss: 0.2999 - categorical_accuracy: 0.9075
22432/60000 [==========>...................] - ETA: 1:07 - loss: 0.2995 - categorical_accuracy: 0.9076
22464/60000 [==========>...................] - ETA: 1:07 - loss: 0.2995 - categorical_accuracy: 0.9077
22496/60000 [==========>...................] - ETA: 1:07 - loss: 0.2993 - categorical_accuracy: 0.9078
22528/60000 [==========>...................] - ETA: 1:07 - loss: 0.2989 - categorical_accuracy: 0.9079
22560/60000 [==========>...................] - ETA: 1:07 - loss: 0.2989 - categorical_accuracy: 0.9079
22592/60000 [==========>...................] - ETA: 1:06 - loss: 0.2985 - categorical_accuracy: 0.9081
22624/60000 [==========>...................] - ETA: 1:06 - loss: 0.2983 - categorical_accuracy: 0.9082
22656/60000 [==========>...................] - ETA: 1:06 - loss: 0.2983 - categorical_accuracy: 0.9081
22688/60000 [==========>...................] - ETA: 1:06 - loss: 0.2980 - categorical_accuracy: 0.9082
22720/60000 [==========>...................] - ETA: 1:06 - loss: 0.2977 - categorical_accuracy: 0.9083
22752/60000 [==========>...................] - ETA: 1:06 - loss: 0.2974 - categorical_accuracy: 0.9084
22784/60000 [==========>...................] - ETA: 1:06 - loss: 0.2974 - categorical_accuracy: 0.9084
22816/60000 [==========>...................] - ETA: 1:06 - loss: 0.2972 - categorical_accuracy: 0.9084
22848/60000 [==========>...................] - ETA: 1:06 - loss: 0.2969 - categorical_accuracy: 0.9086
22880/60000 [==========>...................] - ETA: 1:06 - loss: 0.2968 - categorical_accuracy: 0.9086
22912/60000 [==========>...................] - ETA: 1:06 - loss: 0.2966 - categorical_accuracy: 0.9087
22944/60000 [==========>...................] - ETA: 1:06 - loss: 0.2962 - categorical_accuracy: 0.9088
22976/60000 [==========>...................] - ETA: 1:06 - loss: 0.2959 - categorical_accuracy: 0.9089
23008/60000 [==========>...................] - ETA: 1:06 - loss: 0.2958 - categorical_accuracy: 0.9089
23040/60000 [==========>...................] - ETA: 1:06 - loss: 0.2955 - categorical_accuracy: 0.9089
23072/60000 [==========>...................] - ETA: 1:06 - loss: 0.2952 - categorical_accuracy: 0.9091
23104/60000 [==========>...................] - ETA: 1:06 - loss: 0.2951 - categorical_accuracy: 0.9091
23136/60000 [==========>...................] - ETA: 1:05 - loss: 0.2950 - categorical_accuracy: 0.9091
23168/60000 [==========>...................] - ETA: 1:05 - loss: 0.2947 - categorical_accuracy: 0.9092
23200/60000 [==========>...................] - ETA: 1:05 - loss: 0.2944 - categorical_accuracy: 0.9093
23232/60000 [==========>...................] - ETA: 1:05 - loss: 0.2941 - categorical_accuracy: 0.9093
23264/60000 [==========>...................] - ETA: 1:05 - loss: 0.2938 - categorical_accuracy: 0.9095
23296/60000 [==========>...................] - ETA: 1:05 - loss: 0.2934 - categorical_accuracy: 0.9096
23328/60000 [==========>...................] - ETA: 1:05 - loss: 0.2931 - categorical_accuracy: 0.9097
23360/60000 [==========>...................] - ETA: 1:05 - loss: 0.2933 - categorical_accuracy: 0.9097
23392/60000 [==========>...................] - ETA: 1:05 - loss: 0.2929 - categorical_accuracy: 0.9098
23424/60000 [==========>...................] - ETA: 1:05 - loss: 0.2926 - categorical_accuracy: 0.9099
23456/60000 [==========>...................] - ETA: 1:05 - loss: 0.2922 - categorical_accuracy: 0.9100
23488/60000 [==========>...................] - ETA: 1:05 - loss: 0.2919 - categorical_accuracy: 0.9102
23520/60000 [==========>...................] - ETA: 1:05 - loss: 0.2916 - categorical_accuracy: 0.9102
23552/60000 [==========>...................] - ETA: 1:05 - loss: 0.2912 - categorical_accuracy: 0.9103
23584/60000 [==========>...................] - ETA: 1:05 - loss: 0.2911 - categorical_accuracy: 0.9104
23616/60000 [==========>...................] - ETA: 1:05 - loss: 0.2910 - categorical_accuracy: 0.9104
23648/60000 [==========>...................] - ETA: 1:05 - loss: 0.2907 - categorical_accuracy: 0.9104
23680/60000 [==========>...................] - ETA: 1:05 - loss: 0.2906 - categorical_accuracy: 0.9104
23712/60000 [==========>...................] - ETA: 1:04 - loss: 0.2906 - categorical_accuracy: 0.9104
23744/60000 [==========>...................] - ETA: 1:04 - loss: 0.2902 - categorical_accuracy: 0.9105
23776/60000 [==========>...................] - ETA: 1:04 - loss: 0.2902 - categorical_accuracy: 0.9106
23808/60000 [==========>...................] - ETA: 1:04 - loss: 0.2900 - categorical_accuracy: 0.9106
23840/60000 [==========>...................] - ETA: 1:04 - loss: 0.2897 - categorical_accuracy: 0.9107
23872/60000 [==========>...................] - ETA: 1:04 - loss: 0.2894 - categorical_accuracy: 0.9108
23904/60000 [==========>...................] - ETA: 1:04 - loss: 0.2892 - categorical_accuracy: 0.9109
23936/60000 [==========>...................] - ETA: 1:04 - loss: 0.2889 - categorical_accuracy: 0.9110
23968/60000 [==========>...................] - ETA: 1:04 - loss: 0.2887 - categorical_accuracy: 0.9110
24000/60000 [===========>..................] - ETA: 1:04 - loss: 0.2885 - categorical_accuracy: 0.9110
24032/60000 [===========>..................] - ETA: 1:04 - loss: 0.2883 - categorical_accuracy: 0.9111
24096/60000 [===========>..................] - ETA: 1:04 - loss: 0.2877 - categorical_accuracy: 0.9112
24128/60000 [===========>..................] - ETA: 1:04 - loss: 0.2876 - categorical_accuracy: 0.9113
24160/60000 [===========>..................] - ETA: 1:04 - loss: 0.2874 - categorical_accuracy: 0.9114
24192/60000 [===========>..................] - ETA: 1:04 - loss: 0.2872 - categorical_accuracy: 0.9114
24224/60000 [===========>..................] - ETA: 1:03 - loss: 0.2871 - categorical_accuracy: 0.9114
24256/60000 [===========>..................] - ETA: 1:03 - loss: 0.2873 - categorical_accuracy: 0.9114
24288/60000 [===========>..................] - ETA: 1:03 - loss: 0.2872 - categorical_accuracy: 0.9114
24320/60000 [===========>..................] - ETA: 1:03 - loss: 0.2873 - categorical_accuracy: 0.9114
24352/60000 [===========>..................] - ETA: 1:03 - loss: 0.2871 - categorical_accuracy: 0.9115
24384/60000 [===========>..................] - ETA: 1:03 - loss: 0.2869 - categorical_accuracy: 0.9115
24416/60000 [===========>..................] - ETA: 1:03 - loss: 0.2866 - categorical_accuracy: 0.9116
24448/60000 [===========>..................] - ETA: 1:03 - loss: 0.2865 - categorical_accuracy: 0.9116
24480/60000 [===========>..................] - ETA: 1:03 - loss: 0.2863 - categorical_accuracy: 0.9117
24512/60000 [===========>..................] - ETA: 1:03 - loss: 0.2860 - categorical_accuracy: 0.9118
24544/60000 [===========>..................] - ETA: 1:03 - loss: 0.2856 - categorical_accuracy: 0.9119
24576/60000 [===========>..................] - ETA: 1:03 - loss: 0.2856 - categorical_accuracy: 0.9119
24608/60000 [===========>..................] - ETA: 1:03 - loss: 0.2854 - categorical_accuracy: 0.9120
24640/60000 [===========>..................] - ETA: 1:03 - loss: 0.2852 - categorical_accuracy: 0.9120
24672/60000 [===========>..................] - ETA: 1:03 - loss: 0.2849 - categorical_accuracy: 0.9120
24704/60000 [===========>..................] - ETA: 1:03 - loss: 0.2848 - categorical_accuracy: 0.9121
24736/60000 [===========>..................] - ETA: 1:03 - loss: 0.2847 - categorical_accuracy: 0.9121
24768/60000 [===========>..................] - ETA: 1:02 - loss: 0.2845 - categorical_accuracy: 0.9121
24800/60000 [===========>..................] - ETA: 1:02 - loss: 0.2842 - categorical_accuracy: 0.9122
24832/60000 [===========>..................] - ETA: 1:02 - loss: 0.2839 - categorical_accuracy: 0.9123
24864/60000 [===========>..................] - ETA: 1:02 - loss: 0.2836 - categorical_accuracy: 0.9124
24896/60000 [===========>..................] - ETA: 1:02 - loss: 0.2834 - categorical_accuracy: 0.9125
24928/60000 [===========>..................] - ETA: 1:02 - loss: 0.2833 - categorical_accuracy: 0.9125
24960/60000 [===========>..................] - ETA: 1:02 - loss: 0.2830 - categorical_accuracy: 0.9125
24992/60000 [===========>..................] - ETA: 1:02 - loss: 0.2827 - categorical_accuracy: 0.9127
25024/60000 [===========>..................] - ETA: 1:02 - loss: 0.2826 - categorical_accuracy: 0.9127
25056/60000 [===========>..................] - ETA: 1:02 - loss: 0.2823 - categorical_accuracy: 0.9128
25088/60000 [===========>..................] - ETA: 1:02 - loss: 0.2821 - categorical_accuracy: 0.9129
25120/60000 [===========>..................] - ETA: 1:02 - loss: 0.2818 - categorical_accuracy: 0.9129
25152/60000 [===========>..................] - ETA: 1:02 - loss: 0.2818 - categorical_accuracy: 0.9130
25184/60000 [===========>..................] - ETA: 1:02 - loss: 0.2818 - categorical_accuracy: 0.9130
25216/60000 [===========>..................] - ETA: 1:02 - loss: 0.2816 - categorical_accuracy: 0.9131
25248/60000 [===========>..................] - ETA: 1:02 - loss: 0.2814 - categorical_accuracy: 0.9131
25280/60000 [===========>..................] - ETA: 1:02 - loss: 0.2813 - categorical_accuracy: 0.9132
25312/60000 [===========>..................] - ETA: 1:01 - loss: 0.2812 - categorical_accuracy: 0.9132
25344/60000 [===========>..................] - ETA: 1:01 - loss: 0.2811 - categorical_accuracy: 0.9132
25376/60000 [===========>..................] - ETA: 1:01 - loss: 0.2810 - categorical_accuracy: 0.9133
25408/60000 [===========>..................] - ETA: 1:01 - loss: 0.2808 - categorical_accuracy: 0.9133
25440/60000 [===========>..................] - ETA: 1:01 - loss: 0.2805 - categorical_accuracy: 0.9134
25472/60000 [===========>..................] - ETA: 1:01 - loss: 0.2803 - categorical_accuracy: 0.9135
25504/60000 [===========>..................] - ETA: 1:01 - loss: 0.2800 - categorical_accuracy: 0.9136
25536/60000 [===========>..................] - ETA: 1:01 - loss: 0.2798 - categorical_accuracy: 0.9137
25568/60000 [===========>..................] - ETA: 1:01 - loss: 0.2797 - categorical_accuracy: 0.9136
25600/60000 [===========>..................] - ETA: 1:01 - loss: 0.2794 - categorical_accuracy: 0.9137
25632/60000 [===========>..................] - ETA: 1:01 - loss: 0.2791 - categorical_accuracy: 0.9138
25664/60000 [===========>..................] - ETA: 1:01 - loss: 0.2789 - categorical_accuracy: 0.9138
25696/60000 [===========>..................] - ETA: 1:01 - loss: 0.2788 - categorical_accuracy: 0.9139
25728/60000 [===========>..................] - ETA: 1:01 - loss: 0.2785 - categorical_accuracy: 0.9139
25760/60000 [===========>..................] - ETA: 1:01 - loss: 0.2784 - categorical_accuracy: 0.9139
25792/60000 [===========>..................] - ETA: 1:01 - loss: 0.2783 - categorical_accuracy: 0.9139
25824/60000 [===========>..................] - ETA: 1:01 - loss: 0.2780 - categorical_accuracy: 0.9140
25856/60000 [===========>..................] - ETA: 1:01 - loss: 0.2779 - categorical_accuracy: 0.9140
25888/60000 [===========>..................] - ETA: 1:00 - loss: 0.2777 - categorical_accuracy: 0.9141
25920/60000 [===========>..................] - ETA: 1:00 - loss: 0.2776 - categorical_accuracy: 0.9140
25952/60000 [===========>..................] - ETA: 1:00 - loss: 0.2779 - categorical_accuracy: 0.9140
25984/60000 [===========>..................] - ETA: 1:00 - loss: 0.2776 - categorical_accuracy: 0.9141
26016/60000 [============>.................] - ETA: 1:00 - loss: 0.2774 - categorical_accuracy: 0.9141
26048/60000 [============>.................] - ETA: 1:00 - loss: 0.2774 - categorical_accuracy: 0.9141
26080/60000 [============>.................] - ETA: 1:00 - loss: 0.2771 - categorical_accuracy: 0.9141
26112/60000 [============>.................] - ETA: 1:00 - loss: 0.2769 - categorical_accuracy: 0.9142
26144/60000 [============>.................] - ETA: 1:00 - loss: 0.2766 - categorical_accuracy: 0.9143
26176/60000 [============>.................] - ETA: 1:00 - loss: 0.2763 - categorical_accuracy: 0.9144
26208/60000 [============>.................] - ETA: 1:00 - loss: 0.2760 - categorical_accuracy: 0.9145
26240/60000 [============>.................] - ETA: 1:00 - loss: 0.2757 - categorical_accuracy: 0.9146
26272/60000 [============>.................] - ETA: 1:00 - loss: 0.2756 - categorical_accuracy: 0.9146
26304/60000 [============>.................] - ETA: 1:00 - loss: 0.2755 - categorical_accuracy: 0.9147
26336/60000 [============>.................] - ETA: 1:00 - loss: 0.2752 - categorical_accuracy: 0.9148
26368/60000 [============>.................] - ETA: 1:00 - loss: 0.2751 - categorical_accuracy: 0.9148
26400/60000 [============>.................] - ETA: 1:00 - loss: 0.2748 - categorical_accuracy: 0.9149
26432/60000 [============>.................] - ETA: 59s - loss: 0.2746 - categorical_accuracy: 0.9150 
26464/60000 [============>.................] - ETA: 59s - loss: 0.2743 - categorical_accuracy: 0.9151
26496/60000 [============>.................] - ETA: 59s - loss: 0.2742 - categorical_accuracy: 0.9150
26528/60000 [============>.................] - ETA: 59s - loss: 0.2739 - categorical_accuracy: 0.9151
26560/60000 [============>.................] - ETA: 59s - loss: 0.2736 - categorical_accuracy: 0.9152
26592/60000 [============>.................] - ETA: 59s - loss: 0.2733 - categorical_accuracy: 0.9154
26624/60000 [============>.................] - ETA: 59s - loss: 0.2730 - categorical_accuracy: 0.9154
26656/60000 [============>.................] - ETA: 59s - loss: 0.2731 - categorical_accuracy: 0.9155
26688/60000 [============>.................] - ETA: 59s - loss: 0.2729 - categorical_accuracy: 0.9155
26720/60000 [============>.................] - ETA: 59s - loss: 0.2728 - categorical_accuracy: 0.9156
26752/60000 [============>.................] - ETA: 59s - loss: 0.2726 - categorical_accuracy: 0.9156
26784/60000 [============>.................] - ETA: 59s - loss: 0.2724 - categorical_accuracy: 0.9157
26816/60000 [============>.................] - ETA: 59s - loss: 0.2723 - categorical_accuracy: 0.9157
26848/60000 [============>.................] - ETA: 59s - loss: 0.2721 - categorical_accuracy: 0.9158
26880/60000 [============>.................] - ETA: 59s - loss: 0.2718 - categorical_accuracy: 0.9159
26912/60000 [============>.................] - ETA: 59s - loss: 0.2719 - categorical_accuracy: 0.9159
26944/60000 [============>.................] - ETA: 59s - loss: 0.2716 - categorical_accuracy: 0.9159
26976/60000 [============>.................] - ETA: 59s - loss: 0.2715 - categorical_accuracy: 0.9159
27008/60000 [============>.................] - ETA: 58s - loss: 0.2712 - categorical_accuracy: 0.9160
27040/60000 [============>.................] - ETA: 58s - loss: 0.2710 - categorical_accuracy: 0.9161
27072/60000 [============>.................] - ETA: 58s - loss: 0.2708 - categorical_accuracy: 0.9161
27104/60000 [============>.................] - ETA: 58s - loss: 0.2706 - categorical_accuracy: 0.9162
27136/60000 [============>.................] - ETA: 58s - loss: 0.2703 - categorical_accuracy: 0.9163
27168/60000 [============>.................] - ETA: 58s - loss: 0.2701 - categorical_accuracy: 0.9164
27200/60000 [============>.................] - ETA: 58s - loss: 0.2698 - categorical_accuracy: 0.9164
27232/60000 [============>.................] - ETA: 58s - loss: 0.2695 - categorical_accuracy: 0.9165
27264/60000 [============>.................] - ETA: 58s - loss: 0.2693 - categorical_accuracy: 0.9166
27296/60000 [============>.................] - ETA: 58s - loss: 0.2693 - categorical_accuracy: 0.9166
27328/60000 [============>.................] - ETA: 58s - loss: 0.2691 - categorical_accuracy: 0.9167
27360/60000 [============>.................] - ETA: 58s - loss: 0.2690 - categorical_accuracy: 0.9167
27392/60000 [============>.................] - ETA: 58s - loss: 0.2689 - categorical_accuracy: 0.9167
27424/60000 [============>.................] - ETA: 58s - loss: 0.2687 - categorical_accuracy: 0.9168
27456/60000 [============>.................] - ETA: 58s - loss: 0.2687 - categorical_accuracy: 0.9168
27488/60000 [============>.................] - ETA: 58s - loss: 0.2684 - categorical_accuracy: 0.9169
27520/60000 [============>.................] - ETA: 58s - loss: 0.2687 - categorical_accuracy: 0.9169
27552/60000 [============>.................] - ETA: 57s - loss: 0.2685 - categorical_accuracy: 0.9169
27584/60000 [============>.................] - ETA: 57s - loss: 0.2683 - categorical_accuracy: 0.9170
27616/60000 [============>.................] - ETA: 57s - loss: 0.2683 - categorical_accuracy: 0.9170
27648/60000 [============>.................] - ETA: 57s - loss: 0.2680 - categorical_accuracy: 0.9171
27680/60000 [============>.................] - ETA: 57s - loss: 0.2678 - categorical_accuracy: 0.9171
27712/60000 [============>.................] - ETA: 57s - loss: 0.2675 - categorical_accuracy: 0.9172
27744/60000 [============>.................] - ETA: 57s - loss: 0.2674 - categorical_accuracy: 0.9172
27776/60000 [============>.................] - ETA: 57s - loss: 0.2671 - categorical_accuracy: 0.9173
27840/60000 [============>.................] - ETA: 57s - loss: 0.2668 - categorical_accuracy: 0.9173
27872/60000 [============>.................] - ETA: 57s - loss: 0.2666 - categorical_accuracy: 0.9174
27904/60000 [============>.................] - ETA: 57s - loss: 0.2663 - categorical_accuracy: 0.9175
27936/60000 [============>.................] - ETA: 57s - loss: 0.2661 - categorical_accuracy: 0.9176
27968/60000 [============>.................] - ETA: 57s - loss: 0.2658 - categorical_accuracy: 0.9177
28000/60000 [=============>................] - ETA: 57s - loss: 0.2656 - categorical_accuracy: 0.9178
28032/60000 [=============>................] - ETA: 57s - loss: 0.2654 - categorical_accuracy: 0.9178
28064/60000 [=============>................] - ETA: 56s - loss: 0.2654 - categorical_accuracy: 0.9178
28096/60000 [=============>................] - ETA: 56s - loss: 0.2651 - categorical_accuracy: 0.9179
28128/60000 [=============>................] - ETA: 56s - loss: 0.2649 - categorical_accuracy: 0.9180
28160/60000 [=============>................] - ETA: 56s - loss: 0.2646 - categorical_accuracy: 0.9181
28192/60000 [=============>................] - ETA: 56s - loss: 0.2643 - categorical_accuracy: 0.9182
28224/60000 [=============>................] - ETA: 56s - loss: 0.2641 - categorical_accuracy: 0.9182
28256/60000 [=============>................] - ETA: 56s - loss: 0.2644 - categorical_accuracy: 0.9181
28288/60000 [=============>................] - ETA: 56s - loss: 0.2643 - categorical_accuracy: 0.9182
28320/60000 [=============>................] - ETA: 56s - loss: 0.2645 - categorical_accuracy: 0.9181
28352/60000 [=============>................] - ETA: 56s - loss: 0.2643 - categorical_accuracy: 0.9182
28384/60000 [=============>................] - ETA: 56s - loss: 0.2641 - categorical_accuracy: 0.9183
28416/60000 [=============>................] - ETA: 56s - loss: 0.2638 - categorical_accuracy: 0.9184
28448/60000 [=============>................] - ETA: 56s - loss: 0.2635 - categorical_accuracy: 0.9184
28480/60000 [=============>................] - ETA: 56s - loss: 0.2632 - categorical_accuracy: 0.9185
28512/60000 [=============>................] - ETA: 56s - loss: 0.2632 - categorical_accuracy: 0.9185
28544/60000 [=============>................] - ETA: 56s - loss: 0.2633 - categorical_accuracy: 0.9185
28576/60000 [=============>................] - ETA: 56s - loss: 0.2631 - categorical_accuracy: 0.9185
28608/60000 [=============>................] - ETA: 56s - loss: 0.2632 - categorical_accuracy: 0.9186
28640/60000 [=============>................] - ETA: 55s - loss: 0.2630 - categorical_accuracy: 0.9186
28672/60000 [=============>................] - ETA: 55s - loss: 0.2628 - categorical_accuracy: 0.9187
28704/60000 [=============>................] - ETA: 55s - loss: 0.2626 - categorical_accuracy: 0.9187
28736/60000 [=============>................] - ETA: 55s - loss: 0.2623 - categorical_accuracy: 0.9188
28768/60000 [=============>................] - ETA: 55s - loss: 0.2622 - categorical_accuracy: 0.9188
28800/60000 [=============>................] - ETA: 55s - loss: 0.2619 - categorical_accuracy: 0.9189
28832/60000 [=============>................] - ETA: 55s - loss: 0.2617 - categorical_accuracy: 0.9190
28864/60000 [=============>................] - ETA: 55s - loss: 0.2615 - categorical_accuracy: 0.9191
28896/60000 [=============>................] - ETA: 55s - loss: 0.2614 - categorical_accuracy: 0.9191
28928/60000 [=============>................] - ETA: 55s - loss: 0.2614 - categorical_accuracy: 0.9191
28960/60000 [=============>................] - ETA: 55s - loss: 0.2613 - categorical_accuracy: 0.9192
28992/60000 [=============>................] - ETA: 55s - loss: 0.2611 - categorical_accuracy: 0.9192
29024/60000 [=============>................] - ETA: 55s - loss: 0.2610 - categorical_accuracy: 0.9193
29056/60000 [=============>................] - ETA: 55s - loss: 0.2608 - categorical_accuracy: 0.9194
29088/60000 [=============>................] - ETA: 55s - loss: 0.2605 - categorical_accuracy: 0.9195
29120/60000 [=============>................] - ETA: 55s - loss: 0.2603 - categorical_accuracy: 0.9195
29152/60000 [=============>................] - ETA: 55s - loss: 0.2604 - categorical_accuracy: 0.9195
29184/60000 [=============>................] - ETA: 55s - loss: 0.2604 - categorical_accuracy: 0.9195
29216/60000 [=============>................] - ETA: 54s - loss: 0.2602 - categorical_accuracy: 0.9196
29248/60000 [=============>................] - ETA: 54s - loss: 0.2599 - categorical_accuracy: 0.9197
29280/60000 [=============>................] - ETA: 54s - loss: 0.2597 - categorical_accuracy: 0.9197
29312/60000 [=============>................] - ETA: 54s - loss: 0.2597 - categorical_accuracy: 0.9197
29344/60000 [=============>................] - ETA: 54s - loss: 0.2598 - categorical_accuracy: 0.9196
29376/60000 [=============>................] - ETA: 54s - loss: 0.2596 - categorical_accuracy: 0.9197
29408/60000 [=============>................] - ETA: 54s - loss: 0.2596 - categorical_accuracy: 0.9197
29440/60000 [=============>................] - ETA: 54s - loss: 0.2601 - categorical_accuracy: 0.9197
29472/60000 [=============>................] - ETA: 54s - loss: 0.2600 - categorical_accuracy: 0.9197
29504/60000 [=============>................] - ETA: 54s - loss: 0.2598 - categorical_accuracy: 0.9198
29536/60000 [=============>................] - ETA: 54s - loss: 0.2597 - categorical_accuracy: 0.9198
29568/60000 [=============>................] - ETA: 54s - loss: 0.2595 - categorical_accuracy: 0.9198
29600/60000 [=============>................] - ETA: 54s - loss: 0.2593 - categorical_accuracy: 0.9199
29632/60000 [=============>................] - ETA: 54s - loss: 0.2591 - categorical_accuracy: 0.9200
29664/60000 [=============>................] - ETA: 54s - loss: 0.2589 - categorical_accuracy: 0.9200
29696/60000 [=============>................] - ETA: 54s - loss: 0.2589 - categorical_accuracy: 0.9200
29728/60000 [=============>................] - ETA: 54s - loss: 0.2586 - categorical_accuracy: 0.9201
29760/60000 [=============>................] - ETA: 53s - loss: 0.2585 - categorical_accuracy: 0.9201
29792/60000 [=============>................] - ETA: 53s - loss: 0.2585 - categorical_accuracy: 0.9201
29824/60000 [=============>................] - ETA: 53s - loss: 0.2583 - categorical_accuracy: 0.9202
29856/60000 [=============>................] - ETA: 53s - loss: 0.2581 - categorical_accuracy: 0.9203
29888/60000 [=============>................] - ETA: 53s - loss: 0.2582 - categorical_accuracy: 0.9203
29920/60000 [=============>................] - ETA: 53s - loss: 0.2579 - categorical_accuracy: 0.9204
29952/60000 [=============>................] - ETA: 53s - loss: 0.2579 - categorical_accuracy: 0.9204
29984/60000 [=============>................] - ETA: 53s - loss: 0.2577 - categorical_accuracy: 0.9204
30016/60000 [==============>...............] - ETA: 53s - loss: 0.2575 - categorical_accuracy: 0.9204
30048/60000 [==============>...............] - ETA: 53s - loss: 0.2574 - categorical_accuracy: 0.9205
30080/60000 [==============>...............] - ETA: 53s - loss: 0.2572 - categorical_accuracy: 0.9205
30112/60000 [==============>...............] - ETA: 53s - loss: 0.2571 - categorical_accuracy: 0.9205
30144/60000 [==============>...............] - ETA: 53s - loss: 0.2569 - categorical_accuracy: 0.9206
30176/60000 [==============>...............] - ETA: 53s - loss: 0.2569 - categorical_accuracy: 0.9206
30208/60000 [==============>...............] - ETA: 53s - loss: 0.2568 - categorical_accuracy: 0.9206
30240/60000 [==============>...............] - ETA: 53s - loss: 0.2567 - categorical_accuracy: 0.9206
30272/60000 [==============>...............] - ETA: 53s - loss: 0.2566 - categorical_accuracy: 0.9206
30304/60000 [==============>...............] - ETA: 53s - loss: 0.2564 - categorical_accuracy: 0.9207
30336/60000 [==============>...............] - ETA: 52s - loss: 0.2562 - categorical_accuracy: 0.9208
30368/60000 [==============>...............] - ETA: 52s - loss: 0.2560 - categorical_accuracy: 0.9208
30400/60000 [==============>...............] - ETA: 52s - loss: 0.2559 - categorical_accuracy: 0.9209
30432/60000 [==============>...............] - ETA: 52s - loss: 0.2558 - categorical_accuracy: 0.9209
30464/60000 [==============>...............] - ETA: 52s - loss: 0.2558 - categorical_accuracy: 0.9209
30496/60000 [==============>...............] - ETA: 52s - loss: 0.2556 - categorical_accuracy: 0.9210
30528/60000 [==============>...............] - ETA: 52s - loss: 0.2554 - categorical_accuracy: 0.9210
30560/60000 [==============>...............] - ETA: 52s - loss: 0.2553 - categorical_accuracy: 0.9210
30592/60000 [==============>...............] - ETA: 52s - loss: 0.2551 - categorical_accuracy: 0.9211
30624/60000 [==============>...............] - ETA: 52s - loss: 0.2549 - categorical_accuracy: 0.9211
30656/60000 [==============>...............] - ETA: 52s - loss: 0.2547 - categorical_accuracy: 0.9212
30688/60000 [==============>...............] - ETA: 52s - loss: 0.2545 - categorical_accuracy: 0.9213
30720/60000 [==============>...............] - ETA: 52s - loss: 0.2544 - categorical_accuracy: 0.9213
30752/60000 [==============>...............] - ETA: 52s - loss: 0.2545 - categorical_accuracy: 0.9213
30784/60000 [==============>...............] - ETA: 52s - loss: 0.2543 - categorical_accuracy: 0.9213
30816/60000 [==============>...............] - ETA: 52s - loss: 0.2541 - categorical_accuracy: 0.9214
30848/60000 [==============>...............] - ETA: 52s - loss: 0.2539 - categorical_accuracy: 0.9215
30880/60000 [==============>...............] - ETA: 51s - loss: 0.2539 - categorical_accuracy: 0.9215
30912/60000 [==============>...............] - ETA: 51s - loss: 0.2538 - categorical_accuracy: 0.9215
30944/60000 [==============>...............] - ETA: 51s - loss: 0.2536 - categorical_accuracy: 0.9216
30976/60000 [==============>...............] - ETA: 51s - loss: 0.2534 - categorical_accuracy: 0.9217
31008/60000 [==============>...............] - ETA: 51s - loss: 0.2532 - categorical_accuracy: 0.9218
31040/60000 [==============>...............] - ETA: 51s - loss: 0.2530 - categorical_accuracy: 0.9218
31072/60000 [==============>...............] - ETA: 51s - loss: 0.2527 - categorical_accuracy: 0.9219
31104/60000 [==============>...............] - ETA: 51s - loss: 0.2526 - categorical_accuracy: 0.9220
31136/60000 [==============>...............] - ETA: 51s - loss: 0.2524 - categorical_accuracy: 0.9220
31168/60000 [==============>...............] - ETA: 51s - loss: 0.2526 - categorical_accuracy: 0.9220
31200/60000 [==============>...............] - ETA: 51s - loss: 0.2525 - categorical_accuracy: 0.9221
31232/60000 [==============>...............] - ETA: 51s - loss: 0.2523 - categorical_accuracy: 0.9222
31264/60000 [==============>...............] - ETA: 51s - loss: 0.2521 - categorical_accuracy: 0.9222
31296/60000 [==============>...............] - ETA: 51s - loss: 0.2518 - categorical_accuracy: 0.9223
31328/60000 [==============>...............] - ETA: 51s - loss: 0.2516 - categorical_accuracy: 0.9224
31360/60000 [==============>...............] - ETA: 51s - loss: 0.2515 - categorical_accuracy: 0.9224
31392/60000 [==============>...............] - ETA: 51s - loss: 0.2512 - categorical_accuracy: 0.9225
31424/60000 [==============>...............] - ETA: 50s - loss: 0.2511 - categorical_accuracy: 0.9225
31456/60000 [==============>...............] - ETA: 50s - loss: 0.2509 - categorical_accuracy: 0.9225
31488/60000 [==============>...............] - ETA: 50s - loss: 0.2509 - categorical_accuracy: 0.9225
31520/60000 [==============>...............] - ETA: 50s - loss: 0.2508 - categorical_accuracy: 0.9225
31552/60000 [==============>...............] - ETA: 50s - loss: 0.2509 - categorical_accuracy: 0.9225
31584/60000 [==============>...............] - ETA: 50s - loss: 0.2507 - categorical_accuracy: 0.9225
31616/60000 [==============>...............] - ETA: 50s - loss: 0.2506 - categorical_accuracy: 0.9225
31648/60000 [==============>...............] - ETA: 50s - loss: 0.2505 - categorical_accuracy: 0.9226
31680/60000 [==============>...............] - ETA: 50s - loss: 0.2503 - categorical_accuracy: 0.9227
31712/60000 [==============>...............] - ETA: 50s - loss: 0.2501 - categorical_accuracy: 0.9227
31744/60000 [==============>...............] - ETA: 50s - loss: 0.2499 - categorical_accuracy: 0.9228
31776/60000 [==============>...............] - ETA: 50s - loss: 0.2498 - categorical_accuracy: 0.9228
31808/60000 [==============>...............] - ETA: 50s - loss: 0.2496 - categorical_accuracy: 0.9229
31840/60000 [==============>...............] - ETA: 50s - loss: 0.2493 - categorical_accuracy: 0.9230
31872/60000 [==============>...............] - ETA: 50s - loss: 0.2494 - categorical_accuracy: 0.9230
31904/60000 [==============>...............] - ETA: 50s - loss: 0.2492 - categorical_accuracy: 0.9231
31936/60000 [==============>...............] - ETA: 50s - loss: 0.2490 - categorical_accuracy: 0.9231
31968/60000 [==============>...............] - ETA: 49s - loss: 0.2489 - categorical_accuracy: 0.9231
32000/60000 [===============>..............] - ETA: 49s - loss: 0.2488 - categorical_accuracy: 0.9232
32032/60000 [===============>..............] - ETA: 49s - loss: 0.2488 - categorical_accuracy: 0.9232
32064/60000 [===============>..............] - ETA: 49s - loss: 0.2488 - categorical_accuracy: 0.9232
32096/60000 [===============>..............] - ETA: 49s - loss: 0.2486 - categorical_accuracy: 0.9232
32128/60000 [===============>..............] - ETA: 49s - loss: 0.2488 - categorical_accuracy: 0.9232
32160/60000 [===============>..............] - ETA: 49s - loss: 0.2486 - categorical_accuracy: 0.9232
32192/60000 [===============>..............] - ETA: 49s - loss: 0.2487 - categorical_accuracy: 0.9232
32224/60000 [===============>..............] - ETA: 49s - loss: 0.2487 - categorical_accuracy: 0.9232
32256/60000 [===============>..............] - ETA: 49s - loss: 0.2484 - categorical_accuracy: 0.9233
32288/60000 [===============>..............] - ETA: 49s - loss: 0.2484 - categorical_accuracy: 0.9233
32320/60000 [===============>..............] - ETA: 49s - loss: 0.2485 - categorical_accuracy: 0.9233
32352/60000 [===============>..............] - ETA: 49s - loss: 0.2484 - categorical_accuracy: 0.9233
32384/60000 [===============>..............] - ETA: 49s - loss: 0.2482 - categorical_accuracy: 0.9234
32416/60000 [===============>..............] - ETA: 49s - loss: 0.2480 - categorical_accuracy: 0.9235
32448/60000 [===============>..............] - ETA: 49s - loss: 0.2478 - categorical_accuracy: 0.9235
32480/60000 [===============>..............] - ETA: 49s - loss: 0.2478 - categorical_accuracy: 0.9235
32512/60000 [===============>..............] - ETA: 48s - loss: 0.2477 - categorical_accuracy: 0.9235
32544/60000 [===============>..............] - ETA: 48s - loss: 0.2476 - categorical_accuracy: 0.9236
32576/60000 [===============>..............] - ETA: 48s - loss: 0.2475 - categorical_accuracy: 0.9236
32608/60000 [===============>..............] - ETA: 48s - loss: 0.2473 - categorical_accuracy: 0.9237
32640/60000 [===============>..............] - ETA: 48s - loss: 0.2471 - categorical_accuracy: 0.9237
32672/60000 [===============>..............] - ETA: 48s - loss: 0.2472 - categorical_accuracy: 0.9237
32704/60000 [===============>..............] - ETA: 48s - loss: 0.2476 - categorical_accuracy: 0.9237
32736/60000 [===============>..............] - ETA: 48s - loss: 0.2474 - categorical_accuracy: 0.9238
32768/60000 [===============>..............] - ETA: 48s - loss: 0.2473 - categorical_accuracy: 0.9238
32800/60000 [===============>..............] - ETA: 48s - loss: 0.2473 - categorical_accuracy: 0.9238
32832/60000 [===============>..............] - ETA: 48s - loss: 0.2471 - categorical_accuracy: 0.9239
32864/60000 [===============>..............] - ETA: 48s - loss: 0.2472 - categorical_accuracy: 0.9239
32896/60000 [===============>..............] - ETA: 48s - loss: 0.2469 - categorical_accuracy: 0.9239
32928/60000 [===============>..............] - ETA: 48s - loss: 0.2468 - categorical_accuracy: 0.9240
32960/60000 [===============>..............] - ETA: 48s - loss: 0.2466 - categorical_accuracy: 0.9240
32992/60000 [===============>..............] - ETA: 48s - loss: 0.2465 - categorical_accuracy: 0.9241
33024/60000 [===============>..............] - ETA: 48s - loss: 0.2463 - categorical_accuracy: 0.9241
33056/60000 [===============>..............] - ETA: 47s - loss: 0.2461 - categorical_accuracy: 0.9242
33088/60000 [===============>..............] - ETA: 47s - loss: 0.2458 - categorical_accuracy: 0.9243
33120/60000 [===============>..............] - ETA: 47s - loss: 0.2459 - categorical_accuracy: 0.9243
33152/60000 [===============>..............] - ETA: 47s - loss: 0.2458 - categorical_accuracy: 0.9243
33184/60000 [===============>..............] - ETA: 47s - loss: 0.2457 - categorical_accuracy: 0.9244
33216/60000 [===============>..............] - ETA: 47s - loss: 0.2456 - categorical_accuracy: 0.9244
33248/60000 [===============>..............] - ETA: 47s - loss: 0.2455 - categorical_accuracy: 0.9244
33280/60000 [===============>..............] - ETA: 47s - loss: 0.2454 - categorical_accuracy: 0.9244
33312/60000 [===============>..............] - ETA: 47s - loss: 0.2452 - categorical_accuracy: 0.9245
33344/60000 [===============>..............] - ETA: 47s - loss: 0.2451 - categorical_accuracy: 0.9245
33376/60000 [===============>..............] - ETA: 47s - loss: 0.2451 - categorical_accuracy: 0.9245
33408/60000 [===============>..............] - ETA: 47s - loss: 0.2450 - categorical_accuracy: 0.9245
33440/60000 [===============>..............] - ETA: 47s - loss: 0.2449 - categorical_accuracy: 0.9246
33472/60000 [===============>..............] - ETA: 47s - loss: 0.2449 - categorical_accuracy: 0.9246
33504/60000 [===============>..............] - ETA: 47s - loss: 0.2449 - categorical_accuracy: 0.9246
33536/60000 [===============>..............] - ETA: 47s - loss: 0.2449 - categorical_accuracy: 0.9246
33568/60000 [===============>..............] - ETA: 47s - loss: 0.2447 - categorical_accuracy: 0.9247
33600/60000 [===============>..............] - ETA: 47s - loss: 0.2447 - categorical_accuracy: 0.9246
33632/60000 [===============>..............] - ETA: 46s - loss: 0.2445 - categorical_accuracy: 0.9247
33664/60000 [===============>..............] - ETA: 46s - loss: 0.2445 - categorical_accuracy: 0.9247
33696/60000 [===============>..............] - ETA: 46s - loss: 0.2443 - categorical_accuracy: 0.9248
33728/60000 [===============>..............] - ETA: 46s - loss: 0.2441 - categorical_accuracy: 0.9248
33760/60000 [===============>..............] - ETA: 46s - loss: 0.2442 - categorical_accuracy: 0.9248
33792/60000 [===============>..............] - ETA: 46s - loss: 0.2440 - categorical_accuracy: 0.9249
33824/60000 [===============>..............] - ETA: 46s - loss: 0.2438 - categorical_accuracy: 0.9249
33856/60000 [===============>..............] - ETA: 46s - loss: 0.2439 - categorical_accuracy: 0.9250
33888/60000 [===============>..............] - ETA: 46s - loss: 0.2437 - categorical_accuracy: 0.9250
33920/60000 [===============>..............] - ETA: 46s - loss: 0.2435 - categorical_accuracy: 0.9251
33952/60000 [===============>..............] - ETA: 46s - loss: 0.2434 - categorical_accuracy: 0.9251
33984/60000 [===============>..............] - ETA: 46s - loss: 0.2433 - categorical_accuracy: 0.9251
34016/60000 [================>.............] - ETA: 46s - loss: 0.2431 - categorical_accuracy: 0.9252
34048/60000 [================>.............] - ETA: 46s - loss: 0.2429 - categorical_accuracy: 0.9253
34080/60000 [================>.............] - ETA: 46s - loss: 0.2427 - categorical_accuracy: 0.9253
34112/60000 [================>.............] - ETA: 46s - loss: 0.2426 - categorical_accuracy: 0.9253
34144/60000 [================>.............] - ETA: 46s - loss: 0.2424 - categorical_accuracy: 0.9254
34176/60000 [================>.............] - ETA: 46s - loss: 0.2423 - categorical_accuracy: 0.9254
34208/60000 [================>.............] - ETA: 45s - loss: 0.2421 - categorical_accuracy: 0.9255
34240/60000 [================>.............] - ETA: 45s - loss: 0.2419 - categorical_accuracy: 0.9256
34272/60000 [================>.............] - ETA: 45s - loss: 0.2418 - categorical_accuracy: 0.9256
34304/60000 [================>.............] - ETA: 45s - loss: 0.2419 - categorical_accuracy: 0.9256
34336/60000 [================>.............] - ETA: 45s - loss: 0.2417 - categorical_accuracy: 0.9256
34368/60000 [================>.............] - ETA: 45s - loss: 0.2417 - categorical_accuracy: 0.9257
34400/60000 [================>.............] - ETA: 45s - loss: 0.2415 - categorical_accuracy: 0.9257
34432/60000 [================>.............] - ETA: 45s - loss: 0.2414 - categorical_accuracy: 0.9257
34464/60000 [================>.............] - ETA: 45s - loss: 0.2413 - categorical_accuracy: 0.9257
34496/60000 [================>.............] - ETA: 45s - loss: 0.2412 - categorical_accuracy: 0.9258
34528/60000 [================>.............] - ETA: 45s - loss: 0.2411 - categorical_accuracy: 0.9258
34560/60000 [================>.............] - ETA: 45s - loss: 0.2408 - categorical_accuracy: 0.9259
34592/60000 [================>.............] - ETA: 45s - loss: 0.2406 - categorical_accuracy: 0.9260
34624/60000 [================>.............] - ETA: 45s - loss: 0.2405 - categorical_accuracy: 0.9260
34656/60000 [================>.............] - ETA: 45s - loss: 0.2406 - categorical_accuracy: 0.9260
34688/60000 [================>.............] - ETA: 45s - loss: 0.2406 - categorical_accuracy: 0.9260
34720/60000 [================>.............] - ETA: 45s - loss: 0.2404 - categorical_accuracy: 0.9261
34752/60000 [================>.............] - ETA: 44s - loss: 0.2405 - categorical_accuracy: 0.9261
34784/60000 [================>.............] - ETA: 44s - loss: 0.2403 - categorical_accuracy: 0.9261
34816/60000 [================>.............] - ETA: 44s - loss: 0.2402 - categorical_accuracy: 0.9262
34848/60000 [================>.............] - ETA: 44s - loss: 0.2400 - categorical_accuracy: 0.9262
34880/60000 [================>.............] - ETA: 44s - loss: 0.2399 - categorical_accuracy: 0.9263
34912/60000 [================>.............] - ETA: 44s - loss: 0.2398 - categorical_accuracy: 0.9263
34944/60000 [================>.............] - ETA: 44s - loss: 0.2397 - categorical_accuracy: 0.9263
34976/60000 [================>.............] - ETA: 44s - loss: 0.2396 - categorical_accuracy: 0.9263
35008/60000 [================>.............] - ETA: 44s - loss: 0.2394 - categorical_accuracy: 0.9264
35040/60000 [================>.............] - ETA: 44s - loss: 0.2392 - categorical_accuracy: 0.9265
35072/60000 [================>.............] - ETA: 44s - loss: 0.2390 - categorical_accuracy: 0.9265
35104/60000 [================>.............] - ETA: 44s - loss: 0.2389 - categorical_accuracy: 0.9265
35136/60000 [================>.............] - ETA: 44s - loss: 0.2388 - categorical_accuracy: 0.9265
35168/60000 [================>.............] - ETA: 44s - loss: 0.2386 - categorical_accuracy: 0.9266
35200/60000 [================>.............] - ETA: 44s - loss: 0.2386 - categorical_accuracy: 0.9266
35232/60000 [================>.............] - ETA: 44s - loss: 0.2386 - categorical_accuracy: 0.9266
35264/60000 [================>.............] - ETA: 44s - loss: 0.2385 - categorical_accuracy: 0.9267
35296/60000 [================>.............] - ETA: 44s - loss: 0.2384 - categorical_accuracy: 0.9267
35328/60000 [================>.............] - ETA: 43s - loss: 0.2384 - categorical_accuracy: 0.9267
35360/60000 [================>.............] - ETA: 43s - loss: 0.2385 - categorical_accuracy: 0.9267
35392/60000 [================>.............] - ETA: 43s - loss: 0.2384 - categorical_accuracy: 0.9267
35424/60000 [================>.............] - ETA: 43s - loss: 0.2383 - categorical_accuracy: 0.9267
35456/60000 [================>.............] - ETA: 43s - loss: 0.2383 - categorical_accuracy: 0.9267
35488/60000 [================>.............] - ETA: 43s - loss: 0.2385 - categorical_accuracy: 0.9266
35520/60000 [================>.............] - ETA: 43s - loss: 0.2383 - categorical_accuracy: 0.9267
35552/60000 [================>.............] - ETA: 43s - loss: 0.2382 - categorical_accuracy: 0.9267
35584/60000 [================>.............] - ETA: 43s - loss: 0.2381 - categorical_accuracy: 0.9268
35616/60000 [================>.............] - ETA: 43s - loss: 0.2380 - categorical_accuracy: 0.9268
35648/60000 [================>.............] - ETA: 43s - loss: 0.2379 - categorical_accuracy: 0.9268
35680/60000 [================>.............] - ETA: 43s - loss: 0.2378 - categorical_accuracy: 0.9269
35712/60000 [================>.............] - ETA: 43s - loss: 0.2377 - categorical_accuracy: 0.9269
35744/60000 [================>.............] - ETA: 43s - loss: 0.2376 - categorical_accuracy: 0.9269
35776/60000 [================>.............] - ETA: 43s - loss: 0.2376 - categorical_accuracy: 0.9269
35808/60000 [================>.............] - ETA: 43s - loss: 0.2374 - categorical_accuracy: 0.9270
35840/60000 [================>.............] - ETA: 43s - loss: 0.2372 - categorical_accuracy: 0.9271
35872/60000 [================>.............] - ETA: 43s - loss: 0.2372 - categorical_accuracy: 0.9270
35904/60000 [================>.............] - ETA: 42s - loss: 0.2370 - categorical_accuracy: 0.9271
35936/60000 [================>.............] - ETA: 42s - loss: 0.2368 - categorical_accuracy: 0.9271
35968/60000 [================>.............] - ETA: 42s - loss: 0.2367 - categorical_accuracy: 0.9272
36000/60000 [=================>............] - ETA: 42s - loss: 0.2365 - categorical_accuracy: 0.9273
36032/60000 [=================>............] - ETA: 42s - loss: 0.2365 - categorical_accuracy: 0.9273
36064/60000 [=================>............] - ETA: 42s - loss: 0.2363 - categorical_accuracy: 0.9274
36096/60000 [=================>............] - ETA: 42s - loss: 0.2362 - categorical_accuracy: 0.9274
36128/60000 [=================>............] - ETA: 42s - loss: 0.2360 - categorical_accuracy: 0.9275
36160/60000 [=================>............] - ETA: 42s - loss: 0.2359 - categorical_accuracy: 0.9275
36192/60000 [=================>............] - ETA: 42s - loss: 0.2358 - categorical_accuracy: 0.9274
36224/60000 [=================>............] - ETA: 42s - loss: 0.2357 - categorical_accuracy: 0.9275
36256/60000 [=================>............] - ETA: 42s - loss: 0.2358 - categorical_accuracy: 0.9274
36288/60000 [=================>............] - ETA: 42s - loss: 0.2357 - categorical_accuracy: 0.9274
36320/60000 [=================>............] - ETA: 42s - loss: 0.2356 - categorical_accuracy: 0.9275
36352/60000 [=================>............] - ETA: 42s - loss: 0.2354 - categorical_accuracy: 0.9275
36384/60000 [=================>............] - ETA: 42s - loss: 0.2352 - categorical_accuracy: 0.9276
36416/60000 [=================>............] - ETA: 42s - loss: 0.2351 - categorical_accuracy: 0.9276
36448/60000 [=================>............] - ETA: 42s - loss: 0.2349 - categorical_accuracy: 0.9277
36480/60000 [=================>............] - ETA: 41s - loss: 0.2347 - categorical_accuracy: 0.9277
36512/60000 [=================>............] - ETA: 41s - loss: 0.2348 - categorical_accuracy: 0.9277
36544/60000 [=================>............] - ETA: 41s - loss: 0.2346 - categorical_accuracy: 0.9278
36576/60000 [=================>............] - ETA: 41s - loss: 0.2346 - categorical_accuracy: 0.9278
36608/60000 [=================>............] - ETA: 41s - loss: 0.2344 - categorical_accuracy: 0.9278
36640/60000 [=================>............] - ETA: 41s - loss: 0.2343 - categorical_accuracy: 0.9279
36672/60000 [=================>............] - ETA: 41s - loss: 0.2341 - categorical_accuracy: 0.9279
36704/60000 [=================>............] - ETA: 41s - loss: 0.2340 - categorical_accuracy: 0.9279
36736/60000 [=================>............] - ETA: 41s - loss: 0.2339 - categorical_accuracy: 0.9280
36768/60000 [=================>............] - ETA: 41s - loss: 0.2337 - categorical_accuracy: 0.9280
36800/60000 [=================>............] - ETA: 41s - loss: 0.2337 - categorical_accuracy: 0.9280
36832/60000 [=================>............] - ETA: 41s - loss: 0.2336 - categorical_accuracy: 0.9281
36864/60000 [=================>............] - ETA: 41s - loss: 0.2334 - categorical_accuracy: 0.9282
36896/60000 [=================>............] - ETA: 41s - loss: 0.2332 - categorical_accuracy: 0.9282
36928/60000 [=================>............] - ETA: 41s - loss: 0.2330 - categorical_accuracy: 0.9283
36960/60000 [=================>............] - ETA: 41s - loss: 0.2329 - categorical_accuracy: 0.9283
36992/60000 [=================>............] - ETA: 41s - loss: 0.2327 - categorical_accuracy: 0.9283
37024/60000 [=================>............] - ETA: 41s - loss: 0.2326 - categorical_accuracy: 0.9284
37056/60000 [=================>............] - ETA: 40s - loss: 0.2325 - categorical_accuracy: 0.9284
37088/60000 [=================>............] - ETA: 40s - loss: 0.2323 - categorical_accuracy: 0.9285
37120/60000 [=================>............] - ETA: 40s - loss: 0.2323 - categorical_accuracy: 0.9285
37152/60000 [=================>............] - ETA: 40s - loss: 0.2322 - categorical_accuracy: 0.9286
37184/60000 [=================>............] - ETA: 40s - loss: 0.2322 - categorical_accuracy: 0.9285
37216/60000 [=================>............] - ETA: 40s - loss: 0.2322 - categorical_accuracy: 0.9286
37248/60000 [=================>............] - ETA: 40s - loss: 0.2322 - categorical_accuracy: 0.9286
37280/60000 [=================>............] - ETA: 40s - loss: 0.2321 - categorical_accuracy: 0.9286
37312/60000 [=================>............] - ETA: 40s - loss: 0.2319 - categorical_accuracy: 0.9287
37344/60000 [=================>............] - ETA: 40s - loss: 0.2318 - categorical_accuracy: 0.9287
37376/60000 [=================>............] - ETA: 40s - loss: 0.2317 - categorical_accuracy: 0.9288
37408/60000 [=================>............] - ETA: 40s - loss: 0.2315 - categorical_accuracy: 0.9288
37440/60000 [=================>............] - ETA: 40s - loss: 0.2315 - categorical_accuracy: 0.9288
37472/60000 [=================>............] - ETA: 40s - loss: 0.2316 - categorical_accuracy: 0.9288
37504/60000 [=================>............] - ETA: 40s - loss: 0.2315 - categorical_accuracy: 0.9289
37536/60000 [=================>............] - ETA: 40s - loss: 0.2313 - categorical_accuracy: 0.9289
37568/60000 [=================>............] - ETA: 40s - loss: 0.2311 - categorical_accuracy: 0.9290
37600/60000 [=================>............] - ETA: 40s - loss: 0.2311 - categorical_accuracy: 0.9290
37632/60000 [=================>............] - ETA: 39s - loss: 0.2310 - categorical_accuracy: 0.9290
37664/60000 [=================>............] - ETA: 39s - loss: 0.2308 - categorical_accuracy: 0.9291
37696/60000 [=================>............] - ETA: 39s - loss: 0.2307 - categorical_accuracy: 0.9291
37728/60000 [=================>............] - ETA: 39s - loss: 0.2305 - categorical_accuracy: 0.9292
37760/60000 [=================>............] - ETA: 39s - loss: 0.2306 - categorical_accuracy: 0.9292
37792/60000 [=================>............] - ETA: 39s - loss: 0.2304 - categorical_accuracy: 0.9292
37824/60000 [=================>............] - ETA: 39s - loss: 0.2303 - categorical_accuracy: 0.9292
37856/60000 [=================>............] - ETA: 39s - loss: 0.2302 - categorical_accuracy: 0.9292
37888/60000 [=================>............] - ETA: 39s - loss: 0.2301 - categorical_accuracy: 0.9292
37920/60000 [=================>............] - ETA: 39s - loss: 0.2300 - categorical_accuracy: 0.9292
37952/60000 [=================>............] - ETA: 39s - loss: 0.2300 - categorical_accuracy: 0.9293
37984/60000 [=================>............] - ETA: 39s - loss: 0.2298 - categorical_accuracy: 0.9293
38016/60000 [==================>...........] - ETA: 39s - loss: 0.2298 - categorical_accuracy: 0.9293
38048/60000 [==================>...........] - ETA: 39s - loss: 0.2299 - categorical_accuracy: 0.9293
38080/60000 [==================>...........] - ETA: 39s - loss: 0.2298 - categorical_accuracy: 0.9293
38112/60000 [==================>...........] - ETA: 39s - loss: 0.2296 - categorical_accuracy: 0.9293
38144/60000 [==================>...........] - ETA: 39s - loss: 0.2295 - categorical_accuracy: 0.9294
38176/60000 [==================>...........] - ETA: 39s - loss: 0.2294 - categorical_accuracy: 0.9294
38208/60000 [==================>...........] - ETA: 38s - loss: 0.2292 - categorical_accuracy: 0.9295
38240/60000 [==================>...........] - ETA: 38s - loss: 0.2291 - categorical_accuracy: 0.9295
38272/60000 [==================>...........] - ETA: 38s - loss: 0.2290 - categorical_accuracy: 0.9295
38304/60000 [==================>...........] - ETA: 38s - loss: 0.2289 - categorical_accuracy: 0.9296
38336/60000 [==================>...........] - ETA: 38s - loss: 0.2289 - categorical_accuracy: 0.9295
38368/60000 [==================>...........] - ETA: 38s - loss: 0.2288 - categorical_accuracy: 0.9296
38400/60000 [==================>...........] - ETA: 38s - loss: 0.2287 - categorical_accuracy: 0.9296
38432/60000 [==================>...........] - ETA: 38s - loss: 0.2285 - categorical_accuracy: 0.9297
38464/60000 [==================>...........] - ETA: 38s - loss: 0.2284 - categorical_accuracy: 0.9297
38496/60000 [==================>...........] - ETA: 38s - loss: 0.2283 - categorical_accuracy: 0.9297
38528/60000 [==================>...........] - ETA: 38s - loss: 0.2282 - categorical_accuracy: 0.9297
38560/60000 [==================>...........] - ETA: 38s - loss: 0.2282 - categorical_accuracy: 0.9298
38592/60000 [==================>...........] - ETA: 38s - loss: 0.2281 - categorical_accuracy: 0.9298
38624/60000 [==================>...........] - ETA: 38s - loss: 0.2281 - categorical_accuracy: 0.9298
38656/60000 [==================>...........] - ETA: 38s - loss: 0.2280 - categorical_accuracy: 0.9298
38688/60000 [==================>...........] - ETA: 38s - loss: 0.2279 - categorical_accuracy: 0.9298
38720/60000 [==================>...........] - ETA: 38s - loss: 0.2278 - categorical_accuracy: 0.9299
38752/60000 [==================>...........] - ETA: 38s - loss: 0.2276 - categorical_accuracy: 0.9299
38784/60000 [==================>...........] - ETA: 37s - loss: 0.2275 - categorical_accuracy: 0.9299
38816/60000 [==================>...........] - ETA: 37s - loss: 0.2274 - categorical_accuracy: 0.9300
38848/60000 [==================>...........] - ETA: 37s - loss: 0.2272 - categorical_accuracy: 0.9301
38880/60000 [==================>...........] - ETA: 37s - loss: 0.2271 - categorical_accuracy: 0.9301
38912/60000 [==================>...........] - ETA: 37s - loss: 0.2270 - categorical_accuracy: 0.9301
38944/60000 [==================>...........] - ETA: 37s - loss: 0.2269 - categorical_accuracy: 0.9302
38976/60000 [==================>...........] - ETA: 37s - loss: 0.2268 - categorical_accuracy: 0.9302
39008/60000 [==================>...........] - ETA: 37s - loss: 0.2267 - categorical_accuracy: 0.9302
39040/60000 [==================>...........] - ETA: 37s - loss: 0.2266 - categorical_accuracy: 0.9303
39072/60000 [==================>...........] - ETA: 37s - loss: 0.2266 - categorical_accuracy: 0.9302
39104/60000 [==================>...........] - ETA: 37s - loss: 0.2264 - categorical_accuracy: 0.9303
39136/60000 [==================>...........] - ETA: 37s - loss: 0.2265 - categorical_accuracy: 0.9302
39168/60000 [==================>...........] - ETA: 37s - loss: 0.2263 - categorical_accuracy: 0.9303
39200/60000 [==================>...........] - ETA: 37s - loss: 0.2262 - categorical_accuracy: 0.9303
39232/60000 [==================>...........] - ETA: 37s - loss: 0.2260 - categorical_accuracy: 0.9304
39264/60000 [==================>...........] - ETA: 37s - loss: 0.2259 - categorical_accuracy: 0.9304
39296/60000 [==================>...........] - ETA: 37s - loss: 0.2261 - categorical_accuracy: 0.9304
39328/60000 [==================>...........] - ETA: 36s - loss: 0.2263 - categorical_accuracy: 0.9304
39360/60000 [==================>...........] - ETA: 36s - loss: 0.2262 - categorical_accuracy: 0.9304
39392/60000 [==================>...........] - ETA: 36s - loss: 0.2262 - categorical_accuracy: 0.9304
39424/60000 [==================>...........] - ETA: 36s - loss: 0.2260 - categorical_accuracy: 0.9304
39456/60000 [==================>...........] - ETA: 36s - loss: 0.2259 - categorical_accuracy: 0.9304
39488/60000 [==================>...........] - ETA: 36s - loss: 0.2258 - categorical_accuracy: 0.9305
39520/60000 [==================>...........] - ETA: 36s - loss: 0.2257 - categorical_accuracy: 0.9305
39552/60000 [==================>...........] - ETA: 36s - loss: 0.2256 - categorical_accuracy: 0.9305
39584/60000 [==================>...........] - ETA: 36s - loss: 0.2255 - categorical_accuracy: 0.9306
39616/60000 [==================>...........] - ETA: 36s - loss: 0.2253 - categorical_accuracy: 0.9307
39648/60000 [==================>...........] - ETA: 36s - loss: 0.2252 - categorical_accuracy: 0.9307
39680/60000 [==================>...........] - ETA: 36s - loss: 0.2250 - categorical_accuracy: 0.9308
39712/60000 [==================>...........] - ETA: 36s - loss: 0.2249 - categorical_accuracy: 0.9308
39744/60000 [==================>...........] - ETA: 36s - loss: 0.2247 - categorical_accuracy: 0.9309
39776/60000 [==================>...........] - ETA: 36s - loss: 0.2246 - categorical_accuracy: 0.9309
39808/60000 [==================>...........] - ETA: 36s - loss: 0.2245 - categorical_accuracy: 0.9309
39840/60000 [==================>...........] - ETA: 36s - loss: 0.2244 - categorical_accuracy: 0.9309
39872/60000 [==================>...........] - ETA: 36s - loss: 0.2242 - categorical_accuracy: 0.9310
39904/60000 [==================>...........] - ETA: 35s - loss: 0.2242 - categorical_accuracy: 0.9310
39936/60000 [==================>...........] - ETA: 35s - loss: 0.2240 - categorical_accuracy: 0.9311
39968/60000 [==================>...........] - ETA: 35s - loss: 0.2239 - categorical_accuracy: 0.9311
40000/60000 [===================>..........] - ETA: 35s - loss: 0.2238 - categorical_accuracy: 0.9312
40032/60000 [===================>..........] - ETA: 35s - loss: 0.2236 - categorical_accuracy: 0.9312
40064/60000 [===================>..........] - ETA: 35s - loss: 0.2235 - categorical_accuracy: 0.9312
40096/60000 [===================>..........] - ETA: 35s - loss: 0.2233 - categorical_accuracy: 0.9313
40128/60000 [===================>..........] - ETA: 35s - loss: 0.2232 - categorical_accuracy: 0.9313
40160/60000 [===================>..........] - ETA: 35s - loss: 0.2231 - categorical_accuracy: 0.9314
40192/60000 [===================>..........] - ETA: 35s - loss: 0.2233 - categorical_accuracy: 0.9314
40224/60000 [===================>..........] - ETA: 35s - loss: 0.2232 - categorical_accuracy: 0.9314
40256/60000 [===================>..........] - ETA: 35s - loss: 0.2232 - categorical_accuracy: 0.9314
40288/60000 [===================>..........] - ETA: 35s - loss: 0.2232 - categorical_accuracy: 0.9314
40320/60000 [===================>..........] - ETA: 35s - loss: 0.2230 - categorical_accuracy: 0.9314
40352/60000 [===================>..........] - ETA: 35s - loss: 0.2229 - categorical_accuracy: 0.9314
40384/60000 [===================>..........] - ETA: 35s - loss: 0.2230 - categorical_accuracy: 0.9314
40416/60000 [===================>..........] - ETA: 35s - loss: 0.2228 - categorical_accuracy: 0.9315
40448/60000 [===================>..........] - ETA: 34s - loss: 0.2228 - categorical_accuracy: 0.9315
40480/60000 [===================>..........] - ETA: 34s - loss: 0.2226 - categorical_accuracy: 0.9316
40512/60000 [===================>..........] - ETA: 34s - loss: 0.2226 - categorical_accuracy: 0.9316
40544/60000 [===================>..........] - ETA: 34s - loss: 0.2226 - categorical_accuracy: 0.9316
40576/60000 [===================>..........] - ETA: 34s - loss: 0.2224 - categorical_accuracy: 0.9317
40608/60000 [===================>..........] - ETA: 34s - loss: 0.2224 - categorical_accuracy: 0.9317
40640/60000 [===================>..........] - ETA: 34s - loss: 0.2224 - categorical_accuracy: 0.9317
40672/60000 [===================>..........] - ETA: 34s - loss: 0.2222 - categorical_accuracy: 0.9317
40704/60000 [===================>..........] - ETA: 34s - loss: 0.2221 - categorical_accuracy: 0.9317
40736/60000 [===================>..........] - ETA: 34s - loss: 0.2222 - categorical_accuracy: 0.9317
40768/60000 [===================>..........] - ETA: 34s - loss: 0.2221 - categorical_accuracy: 0.9317
40800/60000 [===================>..........] - ETA: 34s - loss: 0.2219 - categorical_accuracy: 0.9318
40832/60000 [===================>..........] - ETA: 34s - loss: 0.2218 - categorical_accuracy: 0.9318
40864/60000 [===================>..........] - ETA: 34s - loss: 0.2217 - categorical_accuracy: 0.9318
40896/60000 [===================>..........] - ETA: 34s - loss: 0.2216 - categorical_accuracy: 0.9319
40928/60000 [===================>..........] - ETA: 34s - loss: 0.2214 - categorical_accuracy: 0.9319
40960/60000 [===================>..........] - ETA: 34s - loss: 0.2213 - categorical_accuracy: 0.9320
40992/60000 [===================>..........] - ETA: 34s - loss: 0.2212 - categorical_accuracy: 0.9320
41024/60000 [===================>..........] - ETA: 33s - loss: 0.2211 - categorical_accuracy: 0.9320
41056/60000 [===================>..........] - ETA: 33s - loss: 0.2211 - categorical_accuracy: 0.9320
41088/60000 [===================>..........] - ETA: 33s - loss: 0.2209 - categorical_accuracy: 0.9320
41120/60000 [===================>..........] - ETA: 33s - loss: 0.2208 - categorical_accuracy: 0.9321
41152/60000 [===================>..........] - ETA: 33s - loss: 0.2209 - categorical_accuracy: 0.9321
41184/60000 [===================>..........] - ETA: 33s - loss: 0.2207 - categorical_accuracy: 0.9322
41216/60000 [===================>..........] - ETA: 33s - loss: 0.2206 - categorical_accuracy: 0.9322
41248/60000 [===================>..........] - ETA: 33s - loss: 0.2205 - categorical_accuracy: 0.9322
41280/60000 [===================>..........] - ETA: 33s - loss: 0.2204 - categorical_accuracy: 0.9323
41312/60000 [===================>..........] - ETA: 33s - loss: 0.2204 - categorical_accuracy: 0.9323
41344/60000 [===================>..........] - ETA: 33s - loss: 0.2203 - categorical_accuracy: 0.9323
41376/60000 [===================>..........] - ETA: 33s - loss: 0.2202 - categorical_accuracy: 0.9323
41408/60000 [===================>..........] - ETA: 33s - loss: 0.2200 - categorical_accuracy: 0.9324
41440/60000 [===================>..........] - ETA: 33s - loss: 0.2200 - categorical_accuracy: 0.9324
41472/60000 [===================>..........] - ETA: 33s - loss: 0.2200 - categorical_accuracy: 0.9324
41504/60000 [===================>..........] - ETA: 33s - loss: 0.2198 - categorical_accuracy: 0.9324
41536/60000 [===================>..........] - ETA: 33s - loss: 0.2197 - categorical_accuracy: 0.9325
41568/60000 [===================>..........] - ETA: 33s - loss: 0.2196 - categorical_accuracy: 0.9325
41600/60000 [===================>..........] - ETA: 32s - loss: 0.2195 - categorical_accuracy: 0.9325
41632/60000 [===================>..........] - ETA: 32s - loss: 0.2194 - categorical_accuracy: 0.9326
41664/60000 [===================>..........] - ETA: 32s - loss: 0.2192 - categorical_accuracy: 0.9326
41696/60000 [===================>..........] - ETA: 32s - loss: 0.2192 - categorical_accuracy: 0.9326
41728/60000 [===================>..........] - ETA: 32s - loss: 0.2190 - categorical_accuracy: 0.9327
41760/60000 [===================>..........] - ETA: 32s - loss: 0.2189 - categorical_accuracy: 0.9327
41792/60000 [===================>..........] - ETA: 32s - loss: 0.2188 - categorical_accuracy: 0.9328
41824/60000 [===================>..........] - ETA: 32s - loss: 0.2186 - categorical_accuracy: 0.9328
41856/60000 [===================>..........] - ETA: 32s - loss: 0.2185 - categorical_accuracy: 0.9329
41888/60000 [===================>..........] - ETA: 32s - loss: 0.2183 - categorical_accuracy: 0.9329
41920/60000 [===================>..........] - ETA: 32s - loss: 0.2182 - categorical_accuracy: 0.9329
41952/60000 [===================>..........] - ETA: 32s - loss: 0.2181 - categorical_accuracy: 0.9329
41984/60000 [===================>..........] - ETA: 32s - loss: 0.2180 - categorical_accuracy: 0.9329
42016/60000 [====================>.........] - ETA: 32s - loss: 0.2179 - categorical_accuracy: 0.9330
42048/60000 [====================>.........] - ETA: 32s - loss: 0.2177 - categorical_accuracy: 0.9330
42080/60000 [====================>.........] - ETA: 32s - loss: 0.2176 - categorical_accuracy: 0.9331
42112/60000 [====================>.........] - ETA: 32s - loss: 0.2176 - categorical_accuracy: 0.9331
42144/60000 [====================>.........] - ETA: 31s - loss: 0.2176 - categorical_accuracy: 0.9331
42176/60000 [====================>.........] - ETA: 31s - loss: 0.2177 - categorical_accuracy: 0.9331
42208/60000 [====================>.........] - ETA: 31s - loss: 0.2176 - categorical_accuracy: 0.9331
42240/60000 [====================>.........] - ETA: 31s - loss: 0.2175 - categorical_accuracy: 0.9331
42272/60000 [====================>.........] - ETA: 31s - loss: 0.2174 - categorical_accuracy: 0.9331
42304/60000 [====================>.........] - ETA: 31s - loss: 0.2173 - categorical_accuracy: 0.9332
42336/60000 [====================>.........] - ETA: 31s - loss: 0.2173 - categorical_accuracy: 0.9332
42368/60000 [====================>.........] - ETA: 31s - loss: 0.2173 - categorical_accuracy: 0.9332
42400/60000 [====================>.........] - ETA: 31s - loss: 0.2173 - categorical_accuracy: 0.9332
42432/60000 [====================>.........] - ETA: 31s - loss: 0.2172 - categorical_accuracy: 0.9333
42464/60000 [====================>.........] - ETA: 31s - loss: 0.2171 - categorical_accuracy: 0.9333
42496/60000 [====================>.........] - ETA: 31s - loss: 0.2170 - categorical_accuracy: 0.9333
42528/60000 [====================>.........] - ETA: 31s - loss: 0.2169 - categorical_accuracy: 0.9333
42560/60000 [====================>.........] - ETA: 31s - loss: 0.2169 - categorical_accuracy: 0.9333
42592/60000 [====================>.........] - ETA: 31s - loss: 0.2169 - categorical_accuracy: 0.9333
42624/60000 [====================>.........] - ETA: 31s - loss: 0.2167 - categorical_accuracy: 0.9334
42656/60000 [====================>.........] - ETA: 31s - loss: 0.2167 - categorical_accuracy: 0.9334
42688/60000 [====================>.........] - ETA: 31s - loss: 0.2166 - categorical_accuracy: 0.9334
42720/60000 [====================>.........] - ETA: 30s - loss: 0.2165 - categorical_accuracy: 0.9335
42752/60000 [====================>.........] - ETA: 30s - loss: 0.2164 - categorical_accuracy: 0.9335
42784/60000 [====================>.........] - ETA: 30s - loss: 0.2163 - categorical_accuracy: 0.9335
42816/60000 [====================>.........] - ETA: 30s - loss: 0.2164 - categorical_accuracy: 0.9335
42848/60000 [====================>.........] - ETA: 30s - loss: 0.2163 - categorical_accuracy: 0.9336
42880/60000 [====================>.........] - ETA: 30s - loss: 0.2162 - categorical_accuracy: 0.9336
42912/60000 [====================>.........] - ETA: 30s - loss: 0.2163 - categorical_accuracy: 0.9336
42944/60000 [====================>.........] - ETA: 30s - loss: 0.2162 - categorical_accuracy: 0.9336
42976/60000 [====================>.........] - ETA: 30s - loss: 0.2161 - categorical_accuracy: 0.9336
43008/60000 [====================>.........] - ETA: 30s - loss: 0.2161 - categorical_accuracy: 0.9336
43040/60000 [====================>.........] - ETA: 30s - loss: 0.2161 - categorical_accuracy: 0.9336
43072/60000 [====================>.........] - ETA: 30s - loss: 0.2161 - categorical_accuracy: 0.9336
43104/60000 [====================>.........] - ETA: 30s - loss: 0.2162 - categorical_accuracy: 0.9336
43136/60000 [====================>.........] - ETA: 30s - loss: 0.2162 - categorical_accuracy: 0.9336
43168/60000 [====================>.........] - ETA: 30s - loss: 0.2161 - categorical_accuracy: 0.9336
43200/60000 [====================>.........] - ETA: 30s - loss: 0.2162 - categorical_accuracy: 0.9336
43232/60000 [====================>.........] - ETA: 30s - loss: 0.2161 - categorical_accuracy: 0.9336
43264/60000 [====================>.........] - ETA: 29s - loss: 0.2160 - categorical_accuracy: 0.9337
43296/60000 [====================>.........] - ETA: 29s - loss: 0.2158 - categorical_accuracy: 0.9337
43328/60000 [====================>.........] - ETA: 29s - loss: 0.2157 - categorical_accuracy: 0.9338
43360/60000 [====================>.........] - ETA: 29s - loss: 0.2156 - categorical_accuracy: 0.9338
43392/60000 [====================>.........] - ETA: 29s - loss: 0.2155 - categorical_accuracy: 0.9338
43424/60000 [====================>.........] - ETA: 29s - loss: 0.2154 - categorical_accuracy: 0.9338
43456/60000 [====================>.........] - ETA: 29s - loss: 0.2153 - categorical_accuracy: 0.9339
43488/60000 [====================>.........] - ETA: 29s - loss: 0.2151 - categorical_accuracy: 0.9339
43520/60000 [====================>.........] - ETA: 29s - loss: 0.2152 - categorical_accuracy: 0.9339
43552/60000 [====================>.........] - ETA: 29s - loss: 0.2150 - categorical_accuracy: 0.9339
43584/60000 [====================>.........] - ETA: 29s - loss: 0.2149 - categorical_accuracy: 0.9340
43616/60000 [====================>.........] - ETA: 29s - loss: 0.2148 - categorical_accuracy: 0.9340
43648/60000 [====================>.........] - ETA: 29s - loss: 0.2147 - categorical_accuracy: 0.9340
43680/60000 [====================>.........] - ETA: 29s - loss: 0.2145 - categorical_accuracy: 0.9341
43712/60000 [====================>.........] - ETA: 29s - loss: 0.2144 - categorical_accuracy: 0.9341
43744/60000 [====================>.........] - ETA: 29s - loss: 0.2147 - categorical_accuracy: 0.9341
43776/60000 [====================>.........] - ETA: 29s - loss: 0.2146 - categorical_accuracy: 0.9341
43808/60000 [====================>.........] - ETA: 29s - loss: 0.2145 - categorical_accuracy: 0.9342
43840/60000 [====================>.........] - ETA: 28s - loss: 0.2143 - categorical_accuracy: 0.9342
43872/60000 [====================>.........] - ETA: 28s - loss: 0.2143 - categorical_accuracy: 0.9342
43904/60000 [====================>.........] - ETA: 28s - loss: 0.2141 - categorical_accuracy: 0.9343
43936/60000 [====================>.........] - ETA: 28s - loss: 0.2141 - categorical_accuracy: 0.9343
43968/60000 [====================>.........] - ETA: 28s - loss: 0.2141 - categorical_accuracy: 0.9343
44000/60000 [=====================>........] - ETA: 28s - loss: 0.2139 - categorical_accuracy: 0.9343
44032/60000 [=====================>........] - ETA: 28s - loss: 0.2138 - categorical_accuracy: 0.9344
44064/60000 [=====================>........] - ETA: 28s - loss: 0.2137 - categorical_accuracy: 0.9344
44096/60000 [=====================>........] - ETA: 28s - loss: 0.2136 - categorical_accuracy: 0.9345
44128/60000 [=====================>........] - ETA: 28s - loss: 0.2135 - categorical_accuracy: 0.9345
44160/60000 [=====================>........] - ETA: 28s - loss: 0.2134 - categorical_accuracy: 0.9345
44192/60000 [=====================>........] - ETA: 28s - loss: 0.2134 - categorical_accuracy: 0.9345
44224/60000 [=====================>........] - ETA: 28s - loss: 0.2133 - categorical_accuracy: 0.9345
44256/60000 [=====================>........] - ETA: 28s - loss: 0.2131 - categorical_accuracy: 0.9346
44288/60000 [=====================>........] - ETA: 28s - loss: 0.2132 - categorical_accuracy: 0.9346
44320/60000 [=====================>........] - ETA: 28s - loss: 0.2130 - categorical_accuracy: 0.9346
44352/60000 [=====================>........] - ETA: 28s - loss: 0.2131 - categorical_accuracy: 0.9346
44384/60000 [=====================>........] - ETA: 28s - loss: 0.2130 - categorical_accuracy: 0.9347
44416/60000 [=====================>........] - ETA: 27s - loss: 0.2130 - categorical_accuracy: 0.9347
44448/60000 [=====================>........] - ETA: 27s - loss: 0.2129 - categorical_accuracy: 0.9347
44480/60000 [=====================>........] - ETA: 27s - loss: 0.2128 - categorical_accuracy: 0.9347
44512/60000 [=====================>........] - ETA: 27s - loss: 0.2126 - categorical_accuracy: 0.9348
44544/60000 [=====================>........] - ETA: 27s - loss: 0.2126 - categorical_accuracy: 0.9348
44576/60000 [=====================>........] - ETA: 27s - loss: 0.2125 - categorical_accuracy: 0.9348
44608/60000 [=====================>........] - ETA: 27s - loss: 0.2125 - categorical_accuracy: 0.9348
44640/60000 [=====================>........] - ETA: 27s - loss: 0.2123 - categorical_accuracy: 0.9349
44672/60000 [=====================>........] - ETA: 27s - loss: 0.2122 - categorical_accuracy: 0.9349
44704/60000 [=====================>........] - ETA: 27s - loss: 0.2121 - categorical_accuracy: 0.9349
44736/60000 [=====================>........] - ETA: 27s - loss: 0.2120 - categorical_accuracy: 0.9349
44768/60000 [=====================>........] - ETA: 27s - loss: 0.2119 - categorical_accuracy: 0.9350
44800/60000 [=====================>........] - ETA: 27s - loss: 0.2118 - categorical_accuracy: 0.9350
44832/60000 [=====================>........] - ETA: 27s - loss: 0.2117 - categorical_accuracy: 0.9350
44864/60000 [=====================>........] - ETA: 27s - loss: 0.2115 - categorical_accuracy: 0.9350
44896/60000 [=====================>........] - ETA: 27s - loss: 0.2114 - categorical_accuracy: 0.9351
44928/60000 [=====================>........] - ETA: 27s - loss: 0.2114 - categorical_accuracy: 0.9351
44960/60000 [=====================>........] - ETA: 26s - loss: 0.2114 - categorical_accuracy: 0.9351
44992/60000 [=====================>........] - ETA: 26s - loss: 0.2112 - categorical_accuracy: 0.9352
45024/60000 [=====================>........] - ETA: 26s - loss: 0.2111 - categorical_accuracy: 0.9352
45056/60000 [=====================>........] - ETA: 26s - loss: 0.2110 - categorical_accuracy: 0.9352
45088/60000 [=====================>........] - ETA: 26s - loss: 0.2109 - categorical_accuracy: 0.9352
45120/60000 [=====================>........] - ETA: 26s - loss: 0.2109 - categorical_accuracy: 0.9353
45152/60000 [=====================>........] - ETA: 26s - loss: 0.2108 - categorical_accuracy: 0.9353
45184/60000 [=====================>........] - ETA: 26s - loss: 0.2107 - categorical_accuracy: 0.9353
45216/60000 [=====================>........] - ETA: 26s - loss: 0.2107 - categorical_accuracy: 0.9353
45248/60000 [=====================>........] - ETA: 26s - loss: 0.2106 - categorical_accuracy: 0.9353
45280/60000 [=====================>........] - ETA: 26s - loss: 0.2106 - categorical_accuracy: 0.9353
45312/60000 [=====================>........] - ETA: 26s - loss: 0.2105 - categorical_accuracy: 0.9353
45344/60000 [=====================>........] - ETA: 26s - loss: 0.2104 - categorical_accuracy: 0.9354
45376/60000 [=====================>........] - ETA: 26s - loss: 0.2102 - categorical_accuracy: 0.9354
45408/60000 [=====================>........] - ETA: 26s - loss: 0.2104 - categorical_accuracy: 0.9354
45440/60000 [=====================>........] - ETA: 26s - loss: 0.2103 - categorical_accuracy: 0.9354
45472/60000 [=====================>........] - ETA: 26s - loss: 0.2103 - categorical_accuracy: 0.9354
45504/60000 [=====================>........] - ETA: 26s - loss: 0.2102 - categorical_accuracy: 0.9354
45536/60000 [=====================>........] - ETA: 25s - loss: 0.2101 - categorical_accuracy: 0.9354
45568/60000 [=====================>........] - ETA: 25s - loss: 0.2100 - categorical_accuracy: 0.9354
45600/60000 [=====================>........] - ETA: 25s - loss: 0.2099 - categorical_accuracy: 0.9355
45632/60000 [=====================>........] - ETA: 25s - loss: 0.2099 - categorical_accuracy: 0.9355
45664/60000 [=====================>........] - ETA: 25s - loss: 0.2098 - categorical_accuracy: 0.9355
45696/60000 [=====================>........] - ETA: 25s - loss: 0.2096 - categorical_accuracy: 0.9356
45728/60000 [=====================>........] - ETA: 25s - loss: 0.2095 - categorical_accuracy: 0.9356
45760/60000 [=====================>........] - ETA: 25s - loss: 0.2095 - categorical_accuracy: 0.9356
45792/60000 [=====================>........] - ETA: 25s - loss: 0.2095 - categorical_accuracy: 0.9356
45824/60000 [=====================>........] - ETA: 25s - loss: 0.2094 - categorical_accuracy: 0.9356
45856/60000 [=====================>........] - ETA: 25s - loss: 0.2093 - categorical_accuracy: 0.9357
45888/60000 [=====================>........] - ETA: 25s - loss: 0.2095 - categorical_accuracy: 0.9357
45920/60000 [=====================>........] - ETA: 25s - loss: 0.2095 - categorical_accuracy: 0.9357
45952/60000 [=====================>........] - ETA: 25s - loss: 0.2094 - categorical_accuracy: 0.9357
45984/60000 [=====================>........] - ETA: 25s - loss: 0.2093 - categorical_accuracy: 0.9357
46016/60000 [======================>.......] - ETA: 25s - loss: 0.2093 - categorical_accuracy: 0.9357
46048/60000 [======================>.......] - ETA: 25s - loss: 0.2092 - categorical_accuracy: 0.9357
46080/60000 [======================>.......] - ETA: 24s - loss: 0.2091 - categorical_accuracy: 0.9357
46112/60000 [======================>.......] - ETA: 24s - loss: 0.2091 - categorical_accuracy: 0.9357
46144/60000 [======================>.......] - ETA: 24s - loss: 0.2091 - categorical_accuracy: 0.9358
46176/60000 [======================>.......] - ETA: 24s - loss: 0.2092 - categorical_accuracy: 0.9358
46208/60000 [======================>.......] - ETA: 24s - loss: 0.2091 - categorical_accuracy: 0.9358
46240/60000 [======================>.......] - ETA: 24s - loss: 0.2090 - categorical_accuracy: 0.9358
46272/60000 [======================>.......] - ETA: 24s - loss: 0.2090 - categorical_accuracy: 0.9358
46304/60000 [======================>.......] - ETA: 24s - loss: 0.2088 - categorical_accuracy: 0.9359
46336/60000 [======================>.......] - ETA: 24s - loss: 0.2087 - categorical_accuracy: 0.9359
46368/60000 [======================>.......] - ETA: 24s - loss: 0.2086 - categorical_accuracy: 0.9359
46400/60000 [======================>.......] - ETA: 24s - loss: 0.2085 - categorical_accuracy: 0.9359
46432/60000 [======================>.......] - ETA: 24s - loss: 0.2084 - categorical_accuracy: 0.9360
46464/60000 [======================>.......] - ETA: 24s - loss: 0.2082 - categorical_accuracy: 0.9360
46496/60000 [======================>.......] - ETA: 24s - loss: 0.2082 - categorical_accuracy: 0.9360
46528/60000 [======================>.......] - ETA: 24s - loss: 0.2082 - categorical_accuracy: 0.9360
46560/60000 [======================>.......] - ETA: 24s - loss: 0.2081 - categorical_accuracy: 0.9360
46592/60000 [======================>.......] - ETA: 24s - loss: 0.2080 - categorical_accuracy: 0.9361
46624/60000 [======================>.......] - ETA: 23s - loss: 0.2079 - categorical_accuracy: 0.9361
46656/60000 [======================>.......] - ETA: 23s - loss: 0.2078 - categorical_accuracy: 0.9361
46688/60000 [======================>.......] - ETA: 23s - loss: 0.2077 - categorical_accuracy: 0.9362
46720/60000 [======================>.......] - ETA: 23s - loss: 0.2077 - categorical_accuracy: 0.9362
46752/60000 [======================>.......] - ETA: 23s - loss: 0.2076 - categorical_accuracy: 0.9362
46784/60000 [======================>.......] - ETA: 23s - loss: 0.2076 - categorical_accuracy: 0.9362
46816/60000 [======================>.......] - ETA: 23s - loss: 0.2076 - categorical_accuracy: 0.9362
46848/60000 [======================>.......] - ETA: 23s - loss: 0.2074 - categorical_accuracy: 0.9362
46880/60000 [======================>.......] - ETA: 23s - loss: 0.2073 - categorical_accuracy: 0.9363
46912/60000 [======================>.......] - ETA: 23s - loss: 0.2072 - categorical_accuracy: 0.9363
46944/60000 [======================>.......] - ETA: 23s - loss: 0.2071 - categorical_accuracy: 0.9363
46976/60000 [======================>.......] - ETA: 23s - loss: 0.2070 - categorical_accuracy: 0.9364
47008/60000 [======================>.......] - ETA: 23s - loss: 0.2069 - categorical_accuracy: 0.9364
47040/60000 [======================>.......] - ETA: 23s - loss: 0.2068 - categorical_accuracy: 0.9364
47072/60000 [======================>.......] - ETA: 23s - loss: 0.2066 - categorical_accuracy: 0.9365
47104/60000 [======================>.......] - ETA: 23s - loss: 0.2066 - categorical_accuracy: 0.9365
47136/60000 [======================>.......] - ETA: 23s - loss: 0.2065 - categorical_accuracy: 0.9365
47168/60000 [======================>.......] - ETA: 23s - loss: 0.2063 - categorical_accuracy: 0.9365
47200/60000 [======================>.......] - ETA: 22s - loss: 0.2062 - categorical_accuracy: 0.9366
47232/60000 [======================>.......] - ETA: 22s - loss: 0.2061 - categorical_accuracy: 0.9366
47264/60000 [======================>.......] - ETA: 22s - loss: 0.2060 - categorical_accuracy: 0.9367
47296/60000 [======================>.......] - ETA: 22s - loss: 0.2059 - categorical_accuracy: 0.9367
47328/60000 [======================>.......] - ETA: 22s - loss: 0.2058 - categorical_accuracy: 0.9367
47360/60000 [======================>.......] - ETA: 22s - loss: 0.2057 - categorical_accuracy: 0.9367
47392/60000 [======================>.......] - ETA: 22s - loss: 0.2056 - categorical_accuracy: 0.9368
47424/60000 [======================>.......] - ETA: 22s - loss: 0.2056 - categorical_accuracy: 0.9368
47456/60000 [======================>.......] - ETA: 22s - loss: 0.2055 - categorical_accuracy: 0.9368
47488/60000 [======================>.......] - ETA: 22s - loss: 0.2055 - categorical_accuracy: 0.9368
47520/60000 [======================>.......] - ETA: 22s - loss: 0.2054 - categorical_accuracy: 0.9368
47552/60000 [======================>.......] - ETA: 22s - loss: 0.2053 - categorical_accuracy: 0.9368
47584/60000 [======================>.......] - ETA: 22s - loss: 0.2052 - categorical_accuracy: 0.9369
47616/60000 [======================>.......] - ETA: 22s - loss: 0.2052 - categorical_accuracy: 0.9369
47648/60000 [======================>.......] - ETA: 22s - loss: 0.2050 - categorical_accuracy: 0.9369
47680/60000 [======================>.......] - ETA: 22s - loss: 0.2050 - categorical_accuracy: 0.9369
47712/60000 [======================>.......] - ETA: 22s - loss: 0.2049 - categorical_accuracy: 0.9369
47744/60000 [======================>.......] - ETA: 21s - loss: 0.2050 - categorical_accuracy: 0.9369
47776/60000 [======================>.......] - ETA: 21s - loss: 0.2049 - categorical_accuracy: 0.9369
47808/60000 [======================>.......] - ETA: 21s - loss: 0.2050 - categorical_accuracy: 0.9369
47872/60000 [======================>.......] - ETA: 21s - loss: 0.2048 - categorical_accuracy: 0.9370
47904/60000 [======================>.......] - ETA: 21s - loss: 0.2047 - categorical_accuracy: 0.9370
47936/60000 [======================>.......] - ETA: 21s - loss: 0.2046 - categorical_accuracy: 0.9370
47968/60000 [======================>.......] - ETA: 21s - loss: 0.2045 - categorical_accuracy: 0.9371
48000/60000 [=======================>......] - ETA: 21s - loss: 0.2044 - categorical_accuracy: 0.9371
48032/60000 [=======================>......] - ETA: 21s - loss: 0.2043 - categorical_accuracy: 0.9371
48064/60000 [=======================>......] - ETA: 21s - loss: 0.2043 - categorical_accuracy: 0.9372
48096/60000 [=======================>......] - ETA: 21s - loss: 0.2042 - categorical_accuracy: 0.9372
48128/60000 [=======================>......] - ETA: 21s - loss: 0.2042 - categorical_accuracy: 0.9372
48160/60000 [=======================>......] - ETA: 21s - loss: 0.2042 - categorical_accuracy: 0.9372
48192/60000 [=======================>......] - ETA: 21s - loss: 0.2040 - categorical_accuracy: 0.9372
48224/60000 [=======================>......] - ETA: 21s - loss: 0.2039 - categorical_accuracy: 0.9373
48256/60000 [=======================>......] - ETA: 21s - loss: 0.2039 - categorical_accuracy: 0.9373
48288/60000 [=======================>......] - ETA: 20s - loss: 0.2038 - categorical_accuracy: 0.9373
48320/60000 [=======================>......] - ETA: 20s - loss: 0.2037 - categorical_accuracy: 0.9373
48352/60000 [=======================>......] - ETA: 20s - loss: 0.2036 - categorical_accuracy: 0.9373
48384/60000 [=======================>......] - ETA: 20s - loss: 0.2036 - categorical_accuracy: 0.9373
48416/60000 [=======================>......] - ETA: 20s - loss: 0.2036 - categorical_accuracy: 0.9373
48448/60000 [=======================>......] - ETA: 20s - loss: 0.2035 - categorical_accuracy: 0.9373
48480/60000 [=======================>......] - ETA: 20s - loss: 0.2034 - categorical_accuracy: 0.9373
48512/60000 [=======================>......] - ETA: 20s - loss: 0.2033 - categorical_accuracy: 0.9374
48544/60000 [=======================>......] - ETA: 20s - loss: 0.2034 - categorical_accuracy: 0.9374
48576/60000 [=======================>......] - ETA: 20s - loss: 0.2033 - categorical_accuracy: 0.9374
48608/60000 [=======================>......] - ETA: 20s - loss: 0.2031 - categorical_accuracy: 0.9374
48640/60000 [=======================>......] - ETA: 20s - loss: 0.2031 - categorical_accuracy: 0.9374
48672/60000 [=======================>......] - ETA: 20s - loss: 0.2030 - categorical_accuracy: 0.9375
48704/60000 [=======================>......] - ETA: 20s - loss: 0.2029 - categorical_accuracy: 0.9375
48736/60000 [=======================>......] - ETA: 20s - loss: 0.2029 - categorical_accuracy: 0.9375
48768/60000 [=======================>......] - ETA: 20s - loss: 0.2029 - categorical_accuracy: 0.9375
48800/60000 [=======================>......] - ETA: 20s - loss: 0.2029 - categorical_accuracy: 0.9375
48832/60000 [=======================>......] - ETA: 20s - loss: 0.2028 - categorical_accuracy: 0.9376
48864/60000 [=======================>......] - ETA: 19s - loss: 0.2027 - categorical_accuracy: 0.9376
48896/60000 [=======================>......] - ETA: 19s - loss: 0.2026 - categorical_accuracy: 0.9376
48928/60000 [=======================>......] - ETA: 19s - loss: 0.2025 - categorical_accuracy: 0.9376
48960/60000 [=======================>......] - ETA: 19s - loss: 0.2024 - categorical_accuracy: 0.9377
48992/60000 [=======================>......] - ETA: 19s - loss: 0.2024 - categorical_accuracy: 0.9376
49024/60000 [=======================>......] - ETA: 19s - loss: 0.2023 - categorical_accuracy: 0.9377
49056/60000 [=======================>......] - ETA: 19s - loss: 0.2023 - categorical_accuracy: 0.9377
49088/60000 [=======================>......] - ETA: 19s - loss: 0.2022 - categorical_accuracy: 0.9377
49120/60000 [=======================>......] - ETA: 19s - loss: 0.2021 - categorical_accuracy: 0.9377
49152/60000 [=======================>......] - ETA: 19s - loss: 0.2020 - categorical_accuracy: 0.9378
49184/60000 [=======================>......] - ETA: 19s - loss: 0.2019 - categorical_accuracy: 0.9378
49216/60000 [=======================>......] - ETA: 19s - loss: 0.2018 - categorical_accuracy: 0.9379
49248/60000 [=======================>......] - ETA: 19s - loss: 0.2016 - categorical_accuracy: 0.9379
49280/60000 [=======================>......] - ETA: 19s - loss: 0.2015 - categorical_accuracy: 0.9379
49312/60000 [=======================>......] - ETA: 19s - loss: 0.2016 - categorical_accuracy: 0.9379
49344/60000 [=======================>......] - ETA: 19s - loss: 0.2016 - categorical_accuracy: 0.9380
49376/60000 [=======================>......] - ETA: 19s - loss: 0.2015 - categorical_accuracy: 0.9380
49408/60000 [=======================>......] - ETA: 19s - loss: 0.2014 - categorical_accuracy: 0.9380
49440/60000 [=======================>......] - ETA: 18s - loss: 0.2013 - categorical_accuracy: 0.9381
49472/60000 [=======================>......] - ETA: 18s - loss: 0.2011 - categorical_accuracy: 0.9381
49504/60000 [=======================>......] - ETA: 18s - loss: 0.2011 - categorical_accuracy: 0.9381
49536/60000 [=======================>......] - ETA: 18s - loss: 0.2011 - categorical_accuracy: 0.9381
49568/60000 [=======================>......] - ETA: 18s - loss: 0.2010 - categorical_accuracy: 0.9381
49600/60000 [=======================>......] - ETA: 18s - loss: 0.2009 - categorical_accuracy: 0.9382
49632/60000 [=======================>......] - ETA: 18s - loss: 0.2008 - categorical_accuracy: 0.9382
49664/60000 [=======================>......] - ETA: 18s - loss: 0.2007 - categorical_accuracy: 0.9382
49696/60000 [=======================>......] - ETA: 18s - loss: 0.2007 - categorical_accuracy: 0.9382
49728/60000 [=======================>......] - ETA: 18s - loss: 0.2006 - categorical_accuracy: 0.9382
49760/60000 [=======================>......] - ETA: 18s - loss: 0.2006 - categorical_accuracy: 0.9382
49792/60000 [=======================>......] - ETA: 18s - loss: 0.2005 - categorical_accuracy: 0.9383
49824/60000 [=======================>......] - ETA: 18s - loss: 0.2004 - categorical_accuracy: 0.9383
49856/60000 [=======================>......] - ETA: 18s - loss: 0.2003 - categorical_accuracy: 0.9383
49888/60000 [=======================>......] - ETA: 18s - loss: 0.2002 - categorical_accuracy: 0.9384
49920/60000 [=======================>......] - ETA: 18s - loss: 0.2001 - categorical_accuracy: 0.9384
49952/60000 [=======================>......] - ETA: 18s - loss: 0.2000 - categorical_accuracy: 0.9384
49984/60000 [=======================>......] - ETA: 17s - loss: 0.2001 - categorical_accuracy: 0.9384
50016/60000 [========================>.....] - ETA: 17s - loss: 0.2000 - categorical_accuracy: 0.9384
50048/60000 [========================>.....] - ETA: 17s - loss: 0.2000 - categorical_accuracy: 0.9384
50080/60000 [========================>.....] - ETA: 17s - loss: 0.1999 - categorical_accuracy: 0.9385
50112/60000 [========================>.....] - ETA: 17s - loss: 0.1999 - categorical_accuracy: 0.9385
50144/60000 [========================>.....] - ETA: 17s - loss: 0.1998 - categorical_accuracy: 0.9385
50176/60000 [========================>.....] - ETA: 17s - loss: 0.1997 - categorical_accuracy: 0.9385
50208/60000 [========================>.....] - ETA: 17s - loss: 0.1996 - categorical_accuracy: 0.9386
50240/60000 [========================>.....] - ETA: 17s - loss: 0.1995 - categorical_accuracy: 0.9386
50272/60000 [========================>.....] - ETA: 17s - loss: 0.1994 - categorical_accuracy: 0.9386
50304/60000 [========================>.....] - ETA: 17s - loss: 0.1994 - categorical_accuracy: 0.9386
50336/60000 [========================>.....] - ETA: 17s - loss: 0.1993 - categorical_accuracy: 0.9387
50368/60000 [========================>.....] - ETA: 17s - loss: 0.1992 - categorical_accuracy: 0.9387
50400/60000 [========================>.....] - ETA: 17s - loss: 0.1991 - categorical_accuracy: 0.9387
50432/60000 [========================>.....] - ETA: 17s - loss: 0.1990 - categorical_accuracy: 0.9387
50464/60000 [========================>.....] - ETA: 17s - loss: 0.1989 - categorical_accuracy: 0.9388
50496/60000 [========================>.....] - ETA: 17s - loss: 0.1988 - categorical_accuracy: 0.9388
50528/60000 [========================>.....] - ETA: 17s - loss: 0.1987 - categorical_accuracy: 0.9388
50560/60000 [========================>.....] - ETA: 16s - loss: 0.1987 - categorical_accuracy: 0.9389
50592/60000 [========================>.....] - ETA: 16s - loss: 0.1986 - categorical_accuracy: 0.9389
50624/60000 [========================>.....] - ETA: 16s - loss: 0.1985 - categorical_accuracy: 0.9389
50656/60000 [========================>.....] - ETA: 16s - loss: 0.1986 - categorical_accuracy: 0.9389
50688/60000 [========================>.....] - ETA: 16s - loss: 0.1985 - categorical_accuracy: 0.9389
50720/60000 [========================>.....] - ETA: 16s - loss: 0.1984 - categorical_accuracy: 0.9390
50752/60000 [========================>.....] - ETA: 16s - loss: 0.1984 - categorical_accuracy: 0.9390
50784/60000 [========================>.....] - ETA: 16s - loss: 0.1983 - categorical_accuracy: 0.9390
50816/60000 [========================>.....] - ETA: 16s - loss: 0.1983 - categorical_accuracy: 0.9390
50848/60000 [========================>.....] - ETA: 16s - loss: 0.1982 - categorical_accuracy: 0.9390
50880/60000 [========================>.....] - ETA: 16s - loss: 0.1982 - categorical_accuracy: 0.9390
50912/60000 [========================>.....] - ETA: 16s - loss: 0.1981 - categorical_accuracy: 0.9391
50944/60000 [========================>.....] - ETA: 16s - loss: 0.1979 - categorical_accuracy: 0.9391
50976/60000 [========================>.....] - ETA: 16s - loss: 0.1978 - categorical_accuracy: 0.9391
51008/60000 [========================>.....] - ETA: 16s - loss: 0.1978 - categorical_accuracy: 0.9391
51040/60000 [========================>.....] - ETA: 16s - loss: 0.1979 - categorical_accuracy: 0.9392
51072/60000 [========================>.....] - ETA: 16s - loss: 0.1978 - categorical_accuracy: 0.9392
51104/60000 [========================>.....] - ETA: 15s - loss: 0.1978 - categorical_accuracy: 0.9392
51136/60000 [========================>.....] - ETA: 15s - loss: 0.1977 - categorical_accuracy: 0.9392
51168/60000 [========================>.....] - ETA: 15s - loss: 0.1976 - categorical_accuracy: 0.9392
51200/60000 [========================>.....] - ETA: 15s - loss: 0.1975 - categorical_accuracy: 0.9392
51232/60000 [========================>.....] - ETA: 15s - loss: 0.1975 - categorical_accuracy: 0.9393
51264/60000 [========================>.....] - ETA: 15s - loss: 0.1975 - categorical_accuracy: 0.9393
51296/60000 [========================>.....] - ETA: 15s - loss: 0.1975 - categorical_accuracy: 0.9393
51328/60000 [========================>.....] - ETA: 15s - loss: 0.1974 - categorical_accuracy: 0.9393
51360/60000 [========================>.....] - ETA: 15s - loss: 0.1973 - categorical_accuracy: 0.9393
51392/60000 [========================>.....] - ETA: 15s - loss: 0.1972 - categorical_accuracy: 0.9394
51424/60000 [========================>.....] - ETA: 15s - loss: 0.1971 - categorical_accuracy: 0.9394
51456/60000 [========================>.....] - ETA: 15s - loss: 0.1971 - categorical_accuracy: 0.9394
51488/60000 [========================>.....] - ETA: 15s - loss: 0.1970 - categorical_accuracy: 0.9394
51520/60000 [========================>.....] - ETA: 15s - loss: 0.1969 - categorical_accuracy: 0.9394
51552/60000 [========================>.....] - ETA: 15s - loss: 0.1970 - categorical_accuracy: 0.9395
51584/60000 [========================>.....] - ETA: 15s - loss: 0.1970 - categorical_accuracy: 0.9395
51616/60000 [========================>.....] - ETA: 15s - loss: 0.1969 - categorical_accuracy: 0.9395
51648/60000 [========================>.....] - ETA: 15s - loss: 0.1968 - categorical_accuracy: 0.9395
51680/60000 [========================>.....] - ETA: 14s - loss: 0.1967 - categorical_accuracy: 0.9396
51712/60000 [========================>.....] - ETA: 14s - loss: 0.1966 - categorical_accuracy: 0.9396
51744/60000 [========================>.....] - ETA: 14s - loss: 0.1966 - categorical_accuracy: 0.9396
51776/60000 [========================>.....] - ETA: 14s - loss: 0.1965 - categorical_accuracy: 0.9396
51808/60000 [========================>.....] - ETA: 14s - loss: 0.1964 - categorical_accuracy: 0.9397
51840/60000 [========================>.....] - ETA: 14s - loss: 0.1963 - categorical_accuracy: 0.9397
51872/60000 [========================>.....] - ETA: 14s - loss: 0.1962 - categorical_accuracy: 0.9397
51904/60000 [========================>.....] - ETA: 14s - loss: 0.1961 - categorical_accuracy: 0.9397
51936/60000 [========================>.....] - ETA: 14s - loss: 0.1961 - categorical_accuracy: 0.9397
51968/60000 [========================>.....] - ETA: 14s - loss: 0.1960 - categorical_accuracy: 0.9398
52000/60000 [=========================>....] - ETA: 14s - loss: 0.1958 - categorical_accuracy: 0.9398
52032/60000 [=========================>....] - ETA: 14s - loss: 0.1957 - categorical_accuracy: 0.9398
52064/60000 [=========================>....] - ETA: 14s - loss: 0.1956 - categorical_accuracy: 0.9399
52096/60000 [=========================>....] - ETA: 14s - loss: 0.1955 - categorical_accuracy: 0.9399
52128/60000 [=========================>....] - ETA: 14s - loss: 0.1956 - categorical_accuracy: 0.9399
52160/60000 [=========================>....] - ETA: 14s - loss: 0.1955 - categorical_accuracy: 0.9399
52192/60000 [=========================>....] - ETA: 14s - loss: 0.1954 - categorical_accuracy: 0.9399
52224/60000 [=========================>....] - ETA: 13s - loss: 0.1954 - categorical_accuracy: 0.9399
52256/60000 [=========================>....] - ETA: 13s - loss: 0.1954 - categorical_accuracy: 0.9399
52288/60000 [=========================>....] - ETA: 13s - loss: 0.1953 - categorical_accuracy: 0.9399
52320/60000 [=========================>....] - ETA: 13s - loss: 0.1952 - categorical_accuracy: 0.9400
52352/60000 [=========================>....] - ETA: 13s - loss: 0.1951 - categorical_accuracy: 0.9400
52384/60000 [=========================>....] - ETA: 13s - loss: 0.1950 - categorical_accuracy: 0.9400
52416/60000 [=========================>....] - ETA: 13s - loss: 0.1950 - categorical_accuracy: 0.9400
52448/60000 [=========================>....] - ETA: 13s - loss: 0.1949 - categorical_accuracy: 0.9401
52480/60000 [=========================>....] - ETA: 13s - loss: 0.1948 - categorical_accuracy: 0.9401
52512/60000 [=========================>....] - ETA: 13s - loss: 0.1949 - categorical_accuracy: 0.9401
52544/60000 [=========================>....] - ETA: 13s - loss: 0.1948 - categorical_accuracy: 0.9401
52576/60000 [=========================>....] - ETA: 13s - loss: 0.1946 - categorical_accuracy: 0.9402
52608/60000 [=========================>....] - ETA: 13s - loss: 0.1945 - categorical_accuracy: 0.9402
52640/60000 [=========================>....] - ETA: 13s - loss: 0.1944 - categorical_accuracy: 0.9402
52704/60000 [=========================>....] - ETA: 13s - loss: 0.1943 - categorical_accuracy: 0.9403
52736/60000 [=========================>....] - ETA: 13s - loss: 0.1943 - categorical_accuracy: 0.9403
52768/60000 [=========================>....] - ETA: 13s - loss: 0.1942 - categorical_accuracy: 0.9403
52800/60000 [=========================>....] - ETA: 12s - loss: 0.1942 - categorical_accuracy: 0.9403
52832/60000 [=========================>....] - ETA: 12s - loss: 0.1941 - categorical_accuracy: 0.9403
52864/60000 [=========================>....] - ETA: 12s - loss: 0.1941 - categorical_accuracy: 0.9403
52896/60000 [=========================>....] - ETA: 12s - loss: 0.1940 - categorical_accuracy: 0.9404
52928/60000 [=========================>....] - ETA: 12s - loss: 0.1939 - categorical_accuracy: 0.9404
52960/60000 [=========================>....] - ETA: 12s - loss: 0.1938 - categorical_accuracy: 0.9404
52992/60000 [=========================>....] - ETA: 12s - loss: 0.1938 - categorical_accuracy: 0.9404
53024/60000 [=========================>....] - ETA: 12s - loss: 0.1937 - categorical_accuracy: 0.9404
53056/60000 [=========================>....] - ETA: 12s - loss: 0.1938 - categorical_accuracy: 0.9404
53088/60000 [=========================>....] - ETA: 12s - loss: 0.1937 - categorical_accuracy: 0.9405
53120/60000 [=========================>....] - ETA: 12s - loss: 0.1937 - categorical_accuracy: 0.9405
53152/60000 [=========================>....] - ETA: 12s - loss: 0.1937 - categorical_accuracy: 0.9405
53184/60000 [=========================>....] - ETA: 12s - loss: 0.1939 - categorical_accuracy: 0.9405
53216/60000 [=========================>....] - ETA: 12s - loss: 0.1938 - categorical_accuracy: 0.9405
53248/60000 [=========================>....] - ETA: 12s - loss: 0.1937 - categorical_accuracy: 0.9405
53280/60000 [=========================>....] - ETA: 12s - loss: 0.1936 - categorical_accuracy: 0.9405
53312/60000 [=========================>....] - ETA: 12s - loss: 0.1935 - categorical_accuracy: 0.9406
53344/60000 [=========================>....] - ETA: 11s - loss: 0.1937 - categorical_accuracy: 0.9405
53376/60000 [=========================>....] - ETA: 11s - loss: 0.1936 - categorical_accuracy: 0.9406
53408/60000 [=========================>....] - ETA: 11s - loss: 0.1936 - categorical_accuracy: 0.9406
53440/60000 [=========================>....] - ETA: 11s - loss: 0.1935 - categorical_accuracy: 0.9406
53472/60000 [=========================>....] - ETA: 11s - loss: 0.1934 - categorical_accuracy: 0.9406
53504/60000 [=========================>....] - ETA: 11s - loss: 0.1934 - categorical_accuracy: 0.9406
53536/60000 [=========================>....] - ETA: 11s - loss: 0.1935 - categorical_accuracy: 0.9406
53568/60000 [=========================>....] - ETA: 11s - loss: 0.1935 - categorical_accuracy: 0.9407
53600/60000 [=========================>....] - ETA: 11s - loss: 0.1934 - categorical_accuracy: 0.9407
53632/60000 [=========================>....] - ETA: 11s - loss: 0.1933 - categorical_accuracy: 0.9407
53664/60000 [=========================>....] - ETA: 11s - loss: 0.1932 - categorical_accuracy: 0.9407
53696/60000 [=========================>....] - ETA: 11s - loss: 0.1934 - categorical_accuracy: 0.9407
53728/60000 [=========================>....] - ETA: 11s - loss: 0.1933 - categorical_accuracy: 0.9407
53760/60000 [=========================>....] - ETA: 11s - loss: 0.1933 - categorical_accuracy: 0.9407
53792/60000 [=========================>....] - ETA: 11s - loss: 0.1932 - categorical_accuracy: 0.9407
53824/60000 [=========================>....] - ETA: 11s - loss: 0.1931 - categorical_accuracy: 0.9408
53856/60000 [=========================>....] - ETA: 11s - loss: 0.1932 - categorical_accuracy: 0.9408
53888/60000 [=========================>....] - ETA: 10s - loss: 0.1931 - categorical_accuracy: 0.9408
53920/60000 [=========================>....] - ETA: 10s - loss: 0.1930 - categorical_accuracy: 0.9408
53952/60000 [=========================>....] - ETA: 10s - loss: 0.1929 - categorical_accuracy: 0.9408
53984/60000 [=========================>....] - ETA: 10s - loss: 0.1929 - categorical_accuracy: 0.9408
54016/60000 [==========================>...] - ETA: 10s - loss: 0.1929 - categorical_accuracy: 0.9409
54048/60000 [==========================>...] - ETA: 10s - loss: 0.1928 - categorical_accuracy: 0.9409
54080/60000 [==========================>...] - ETA: 10s - loss: 0.1928 - categorical_accuracy: 0.9409
54112/60000 [==========================>...] - ETA: 10s - loss: 0.1928 - categorical_accuracy: 0.9409
54144/60000 [==========================>...] - ETA: 10s - loss: 0.1927 - categorical_accuracy: 0.9409
54176/60000 [==========================>...] - ETA: 10s - loss: 0.1927 - categorical_accuracy: 0.9409
54208/60000 [==========================>...] - ETA: 10s - loss: 0.1927 - categorical_accuracy: 0.9409
54240/60000 [==========================>...] - ETA: 10s - loss: 0.1927 - categorical_accuracy: 0.9409
54272/60000 [==========================>...] - ETA: 10s - loss: 0.1926 - categorical_accuracy: 0.9409
54304/60000 [==========================>...] - ETA: 10s - loss: 0.1925 - categorical_accuracy: 0.9409
54336/60000 [==========================>...] - ETA: 10s - loss: 0.1925 - categorical_accuracy: 0.9409
54368/60000 [==========================>...] - ETA: 10s - loss: 0.1925 - categorical_accuracy: 0.9409
54400/60000 [==========================>...] - ETA: 10s - loss: 0.1925 - categorical_accuracy: 0.9409
54432/60000 [==========================>...] - ETA: 9s - loss: 0.1925 - categorical_accuracy: 0.9409 
54464/60000 [==========================>...] - ETA: 9s - loss: 0.1926 - categorical_accuracy: 0.9409
54496/60000 [==========================>...] - ETA: 9s - loss: 0.1925 - categorical_accuracy: 0.9409
54528/60000 [==========================>...] - ETA: 9s - loss: 0.1924 - categorical_accuracy: 0.9409
54560/60000 [==========================>...] - ETA: 9s - loss: 0.1924 - categorical_accuracy: 0.9410
54592/60000 [==========================>...] - ETA: 9s - loss: 0.1923 - categorical_accuracy: 0.9410
54624/60000 [==========================>...] - ETA: 9s - loss: 0.1922 - categorical_accuracy: 0.9410
54656/60000 [==========================>...] - ETA: 9s - loss: 0.1921 - categorical_accuracy: 0.9410
54688/60000 [==========================>...] - ETA: 9s - loss: 0.1921 - categorical_accuracy: 0.9410
54720/60000 [==========================>...] - ETA: 9s - loss: 0.1920 - categorical_accuracy: 0.9410
54752/60000 [==========================>...] - ETA: 9s - loss: 0.1919 - categorical_accuracy: 0.9411
54784/60000 [==========================>...] - ETA: 9s - loss: 0.1918 - categorical_accuracy: 0.9411
54816/60000 [==========================>...] - ETA: 9s - loss: 0.1918 - categorical_accuracy: 0.9411
54848/60000 [==========================>...] - ETA: 9s - loss: 0.1917 - categorical_accuracy: 0.9411
54880/60000 [==========================>...] - ETA: 9s - loss: 0.1917 - categorical_accuracy: 0.9411
54912/60000 [==========================>...] - ETA: 9s - loss: 0.1916 - categorical_accuracy: 0.9411
54944/60000 [==========================>...] - ETA: 9s - loss: 0.1915 - categorical_accuracy: 0.9412
54976/60000 [==========================>...] - ETA: 9s - loss: 0.1914 - categorical_accuracy: 0.9412
55008/60000 [==========================>...] - ETA: 8s - loss: 0.1913 - categorical_accuracy: 0.9412
55040/60000 [==========================>...] - ETA: 8s - loss: 0.1914 - categorical_accuracy: 0.9412
55072/60000 [==========================>...] - ETA: 8s - loss: 0.1913 - categorical_accuracy: 0.9412
55104/60000 [==========================>...] - ETA: 8s - loss: 0.1912 - categorical_accuracy: 0.9412
55136/60000 [==========================>...] - ETA: 8s - loss: 0.1912 - categorical_accuracy: 0.9413
55168/60000 [==========================>...] - ETA: 8s - loss: 0.1912 - categorical_accuracy: 0.9413
55200/60000 [==========================>...] - ETA: 8s - loss: 0.1911 - categorical_accuracy: 0.9413
55232/60000 [==========================>...] - ETA: 8s - loss: 0.1910 - categorical_accuracy: 0.9413
55264/60000 [==========================>...] - ETA: 8s - loss: 0.1910 - categorical_accuracy: 0.9413
55296/60000 [==========================>...] - ETA: 8s - loss: 0.1909 - categorical_accuracy: 0.9414
55328/60000 [==========================>...] - ETA: 8s - loss: 0.1909 - categorical_accuracy: 0.9414
55360/60000 [==========================>...] - ETA: 8s - loss: 0.1908 - categorical_accuracy: 0.9414
55392/60000 [==========================>...] - ETA: 8s - loss: 0.1907 - categorical_accuracy: 0.9414
55424/60000 [==========================>...] - ETA: 8s - loss: 0.1907 - categorical_accuracy: 0.9415
55456/60000 [==========================>...] - ETA: 8s - loss: 0.1906 - categorical_accuracy: 0.9415
55488/60000 [==========================>...] - ETA: 8s - loss: 0.1906 - categorical_accuracy: 0.9415
55520/60000 [==========================>...] - ETA: 8s - loss: 0.1906 - categorical_accuracy: 0.9415
55552/60000 [==========================>...] - ETA: 7s - loss: 0.1905 - categorical_accuracy: 0.9415
55584/60000 [==========================>...] - ETA: 7s - loss: 0.1905 - categorical_accuracy: 0.9415
55616/60000 [==========================>...] - ETA: 7s - loss: 0.1907 - categorical_accuracy: 0.9415
55648/60000 [==========================>...] - ETA: 7s - loss: 0.1906 - categorical_accuracy: 0.9416
55680/60000 [==========================>...] - ETA: 7s - loss: 0.1905 - categorical_accuracy: 0.9416
55712/60000 [==========================>...] - ETA: 7s - loss: 0.1905 - categorical_accuracy: 0.9416
55744/60000 [==========================>...] - ETA: 7s - loss: 0.1905 - categorical_accuracy: 0.9416
55776/60000 [==========================>...] - ETA: 7s - loss: 0.1904 - categorical_accuracy: 0.9417
55808/60000 [==========================>...] - ETA: 7s - loss: 0.1904 - categorical_accuracy: 0.9417
55840/60000 [==========================>...] - ETA: 7s - loss: 0.1903 - categorical_accuracy: 0.9417
55872/60000 [==========================>...] - ETA: 7s - loss: 0.1902 - categorical_accuracy: 0.9417
55904/60000 [==========================>...] - ETA: 7s - loss: 0.1902 - categorical_accuracy: 0.9417
55936/60000 [==========================>...] - ETA: 7s - loss: 0.1901 - categorical_accuracy: 0.9418
55968/60000 [==========================>...] - ETA: 7s - loss: 0.1901 - categorical_accuracy: 0.9418
56000/60000 [===========================>..] - ETA: 7s - loss: 0.1900 - categorical_accuracy: 0.9418
56032/60000 [===========================>..] - ETA: 7s - loss: 0.1900 - categorical_accuracy: 0.9418
56064/60000 [===========================>..] - ETA: 7s - loss: 0.1900 - categorical_accuracy: 0.9418
56096/60000 [===========================>..] - ETA: 7s - loss: 0.1900 - categorical_accuracy: 0.9418
56128/60000 [===========================>..] - ETA: 6s - loss: 0.1900 - categorical_accuracy: 0.9418
56160/60000 [===========================>..] - ETA: 6s - loss: 0.1899 - categorical_accuracy: 0.9419
56192/60000 [===========================>..] - ETA: 6s - loss: 0.1899 - categorical_accuracy: 0.9419
56224/60000 [===========================>..] - ETA: 6s - loss: 0.1898 - categorical_accuracy: 0.9419
56256/60000 [===========================>..] - ETA: 6s - loss: 0.1897 - categorical_accuracy: 0.9419
56288/60000 [===========================>..] - ETA: 6s - loss: 0.1897 - categorical_accuracy: 0.9419
56320/60000 [===========================>..] - ETA: 6s - loss: 0.1896 - categorical_accuracy: 0.9420
56352/60000 [===========================>..] - ETA: 6s - loss: 0.1898 - categorical_accuracy: 0.9419
56384/60000 [===========================>..] - ETA: 6s - loss: 0.1899 - categorical_accuracy: 0.9419
56416/60000 [===========================>..] - ETA: 6s - loss: 0.1898 - categorical_accuracy: 0.9419
56480/60000 [===========================>..] - ETA: 6s - loss: 0.1897 - categorical_accuracy: 0.9420
56512/60000 [===========================>..] - ETA: 6s - loss: 0.1896 - categorical_accuracy: 0.9420
56544/60000 [===========================>..] - ETA: 6s - loss: 0.1897 - categorical_accuracy: 0.9420
56576/60000 [===========================>..] - ETA: 6s - loss: 0.1896 - categorical_accuracy: 0.9420
56608/60000 [===========================>..] - ETA: 6s - loss: 0.1895 - categorical_accuracy: 0.9420
56640/60000 [===========================>..] - ETA: 6s - loss: 0.1895 - categorical_accuracy: 0.9420
56672/60000 [===========================>..] - ETA: 5s - loss: 0.1894 - categorical_accuracy: 0.9421
56704/60000 [===========================>..] - ETA: 5s - loss: 0.1893 - categorical_accuracy: 0.9421
56736/60000 [===========================>..] - ETA: 5s - loss: 0.1892 - categorical_accuracy: 0.9421
56768/60000 [===========================>..] - ETA: 5s - loss: 0.1891 - categorical_accuracy: 0.9422
56800/60000 [===========================>..] - ETA: 5s - loss: 0.1890 - categorical_accuracy: 0.9422
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1891 - categorical_accuracy: 0.9422
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1890 - categorical_accuracy: 0.9422
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1890 - categorical_accuracy: 0.9422
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1889 - categorical_accuracy: 0.9422
56960/60000 [===========================>..] - ETA: 5s - loss: 0.1888 - categorical_accuracy: 0.9423
56992/60000 [===========================>..] - ETA: 5s - loss: 0.1887 - categorical_accuracy: 0.9423
57024/60000 [===========================>..] - ETA: 5s - loss: 0.1887 - categorical_accuracy: 0.9423
57056/60000 [===========================>..] - ETA: 5s - loss: 0.1888 - categorical_accuracy: 0.9423
57088/60000 [===========================>..] - ETA: 5s - loss: 0.1887 - categorical_accuracy: 0.9423
57120/60000 [===========================>..] - ETA: 5s - loss: 0.1887 - categorical_accuracy: 0.9423
57152/60000 [===========================>..] - ETA: 5s - loss: 0.1886 - categorical_accuracy: 0.9424
57184/60000 [===========================>..] - ETA: 5s - loss: 0.1886 - categorical_accuracy: 0.9424
57216/60000 [===========================>..] - ETA: 4s - loss: 0.1885 - categorical_accuracy: 0.9424
57248/60000 [===========================>..] - ETA: 4s - loss: 0.1885 - categorical_accuracy: 0.9424
57280/60000 [===========================>..] - ETA: 4s - loss: 0.1884 - categorical_accuracy: 0.9424
57312/60000 [===========================>..] - ETA: 4s - loss: 0.1884 - categorical_accuracy: 0.9424
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1883 - categorical_accuracy: 0.9425
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1883 - categorical_accuracy: 0.9425
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1882 - categorical_accuracy: 0.9425
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1881 - categorical_accuracy: 0.9425
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1880 - categorical_accuracy: 0.9425
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1880 - categorical_accuracy: 0.9425
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1881 - categorical_accuracy: 0.9426
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1880 - categorical_accuracy: 0.9426
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1879 - categorical_accuracy: 0.9426
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1879 - categorical_accuracy: 0.9426
57664/60000 [===========================>..] - ETA: 4s - loss: 0.1879 - categorical_accuracy: 0.9426
57696/60000 [===========================>..] - ETA: 4s - loss: 0.1878 - categorical_accuracy: 0.9426
57728/60000 [===========================>..] - ETA: 4s - loss: 0.1877 - categorical_accuracy: 0.9427
57760/60000 [===========================>..] - ETA: 4s - loss: 0.1877 - categorical_accuracy: 0.9427
57792/60000 [===========================>..] - ETA: 3s - loss: 0.1877 - categorical_accuracy: 0.9427
57824/60000 [===========================>..] - ETA: 3s - loss: 0.1877 - categorical_accuracy: 0.9427
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1877 - categorical_accuracy: 0.9426
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1877 - categorical_accuracy: 0.9426
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1876 - categorical_accuracy: 0.9427
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1876 - categorical_accuracy: 0.9426
58016/60000 [============================>.] - ETA: 3s - loss: 0.1876 - categorical_accuracy: 0.9427
58048/60000 [============================>.] - ETA: 3s - loss: 0.1875 - categorical_accuracy: 0.9427
58080/60000 [============================>.] - ETA: 3s - loss: 0.1874 - categorical_accuracy: 0.9427
58112/60000 [============================>.] - ETA: 3s - loss: 0.1873 - categorical_accuracy: 0.9427
58144/60000 [============================>.] - ETA: 3s - loss: 0.1872 - categorical_accuracy: 0.9428
58176/60000 [============================>.] - ETA: 3s - loss: 0.1871 - categorical_accuracy: 0.9428
58208/60000 [============================>.] - ETA: 3s - loss: 0.1872 - categorical_accuracy: 0.9428
58240/60000 [============================>.] - ETA: 3s - loss: 0.1871 - categorical_accuracy: 0.9428
58272/60000 [============================>.] - ETA: 3s - loss: 0.1870 - categorical_accuracy: 0.9428
58304/60000 [============================>.] - ETA: 3s - loss: 0.1870 - categorical_accuracy: 0.9429
58336/60000 [============================>.] - ETA: 2s - loss: 0.1870 - categorical_accuracy: 0.9428
58368/60000 [============================>.] - ETA: 2s - loss: 0.1870 - categorical_accuracy: 0.9428
58432/60000 [============================>.] - ETA: 2s - loss: 0.1870 - categorical_accuracy: 0.9428
58464/60000 [============================>.] - ETA: 2s - loss: 0.1869 - categorical_accuracy: 0.9429
58496/60000 [============================>.] - ETA: 2s - loss: 0.1869 - categorical_accuracy: 0.9429
58560/60000 [============================>.] - ETA: 2s - loss: 0.1869 - categorical_accuracy: 0.9429
58624/60000 [============================>.] - ETA: 2s - loss: 0.1868 - categorical_accuracy: 0.9429
58656/60000 [============================>.] - ETA: 2s - loss: 0.1868 - categorical_accuracy: 0.9429
58688/60000 [============================>.] - ETA: 2s - loss: 0.1868 - categorical_accuracy: 0.9429
58720/60000 [============================>.] - ETA: 2s - loss: 0.1867 - categorical_accuracy: 0.9429
58752/60000 [============================>.] - ETA: 2s - loss: 0.1868 - categorical_accuracy: 0.9429
58784/60000 [============================>.] - ETA: 2s - loss: 0.1867 - categorical_accuracy: 0.9430
58816/60000 [============================>.] - ETA: 2s - loss: 0.1866 - categorical_accuracy: 0.9430
58880/60000 [============================>.] - ETA: 2s - loss: 0.1867 - categorical_accuracy: 0.9430
58912/60000 [============================>.] - ETA: 1s - loss: 0.1866 - categorical_accuracy: 0.9430
58944/60000 [============================>.] - ETA: 1s - loss: 0.1866 - categorical_accuracy: 0.9430
58976/60000 [============================>.] - ETA: 1s - loss: 0.1865 - categorical_accuracy: 0.9430
59008/60000 [============================>.] - ETA: 1s - loss: 0.1866 - categorical_accuracy: 0.9430
59072/60000 [============================>.] - ETA: 1s - loss: 0.1865 - categorical_accuracy: 0.9430
59136/60000 [============================>.] - ETA: 1s - loss: 0.1864 - categorical_accuracy: 0.9431
59168/60000 [============================>.] - ETA: 1s - loss: 0.1864 - categorical_accuracy: 0.9431
59232/60000 [============================>.] - ETA: 1s - loss: 0.1863 - categorical_accuracy: 0.9431
59264/60000 [============================>.] - ETA: 1s - loss: 0.1862 - categorical_accuracy: 0.9431
59296/60000 [============================>.] - ETA: 1s - loss: 0.1862 - categorical_accuracy: 0.9431
59328/60000 [============================>.] - ETA: 1s - loss: 0.1861 - categorical_accuracy: 0.9432
59360/60000 [============================>.] - ETA: 1s - loss: 0.1861 - categorical_accuracy: 0.9432
59392/60000 [============================>.] - ETA: 1s - loss: 0.1860 - categorical_accuracy: 0.9432
59424/60000 [============================>.] - ETA: 1s - loss: 0.1860 - categorical_accuracy: 0.9432
59456/60000 [============================>.] - ETA: 0s - loss: 0.1859 - categorical_accuracy: 0.9432
59488/60000 [============================>.] - ETA: 0s - loss: 0.1859 - categorical_accuracy: 0.9432
59520/60000 [============================>.] - ETA: 0s - loss: 0.1858 - categorical_accuracy: 0.9432
59552/60000 [============================>.] - ETA: 0s - loss: 0.1858 - categorical_accuracy: 0.9432
59584/60000 [============================>.] - ETA: 0s - loss: 0.1857 - categorical_accuracy: 0.9433
59616/60000 [============================>.] - ETA: 0s - loss: 0.1856 - categorical_accuracy: 0.9433
59648/60000 [============================>.] - ETA: 0s - loss: 0.1856 - categorical_accuracy: 0.9433
59680/60000 [============================>.] - ETA: 0s - loss: 0.1855 - categorical_accuracy: 0.9433
59712/60000 [============================>.] - ETA: 0s - loss: 0.1854 - categorical_accuracy: 0.9434
59744/60000 [============================>.] - ETA: 0s - loss: 0.1854 - categorical_accuracy: 0.9434
59776/60000 [============================>.] - ETA: 0s - loss: 0.1853 - categorical_accuracy: 0.9434
59808/60000 [============================>.] - ETA: 0s - loss: 0.1852 - categorical_accuracy: 0.9434
59840/60000 [============================>.] - ETA: 0s - loss: 0.1852 - categorical_accuracy: 0.9434
59872/60000 [============================>.] - ETA: 0s - loss: 0.1851 - categorical_accuracy: 0.9434
59904/60000 [============================>.] - ETA: 0s - loss: 0.1850 - categorical_accuracy: 0.9435
59936/60000 [============================>.] - ETA: 0s - loss: 0.1850 - categorical_accuracy: 0.9435
59968/60000 [============================>.] - ETA: 0s - loss: 0.1851 - categorical_accuracy: 0.9435
60000/60000 [==============================] - 111s 2ms/step - loss: 0.1851 - categorical_accuracy: 0.9435 - val_loss: 0.0484 - val_categorical_accuracy: 0.9839

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
 1632/10000 [===>..........................] - ETA: 3s
 1792/10000 [====>.........................] - ETA: 3s
 1952/10000 [====>.........................] - ETA: 3s
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
 4352/10000 [============>.................] - ETA: 2s
 4512/10000 [============>.................] - ETA: 1s
 4672/10000 [=============>................] - ETA: 1s
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
 6368/10000 [==================>...........] - ETA: 1s
 6528/10000 [==================>...........] - ETA: 1s
 6688/10000 [===================>..........] - ETA: 1s
 6784/10000 [===================>..........] - ETA: 1s
 6944/10000 [===================>..........] - ETA: 1s
 7104/10000 [====================>.........] - ETA: 1s
 7264/10000 [====================>.........] - ETA: 1s
 7424/10000 [=====================>........] - ETA: 0s
 7584/10000 [=====================>........] - ETA: 0s
 7744/10000 [======================>.......] - ETA: 0s
 7904/10000 [======================>.......] - ETA: 0s
 8064/10000 [=======================>......] - ETA: 0s
 8224/10000 [=======================>......] - ETA: 0s
 8384/10000 [========================>.....] - ETA: 0s
 8544/10000 [========================>.....] - ETA: 0s
 8704/10000 [=========================>....] - ETA: 0s
 8832/10000 [=========================>....] - ETA: 0s
 8992/10000 [=========================>....] - ETA: 0s
 9152/10000 [==========================>...] - ETA: 0s
 9312/10000 [==========================>...] - ETA: 0s
 9472/10000 [===========================>..] - ETA: 0s
 9632/10000 [===========================>..] - ETA: 0s
 9792/10000 [============================>.] - ETA: 0s
 9952/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 4s 362us/step
[[2.3236691e-08 3.3168412e-08 1.4068573e-06 ... 9.9999464e-01
  1.8955324e-08 2.5538432e-06]
 [4.0087201e-05 2.7408467e-05 9.9991715e-01 ... 2.8097077e-07
  4.1331373e-06 3.6345376e-09]
 [2.1961903e-06 9.9964082e-01 8.8583365e-05 ... 6.5411281e-05
  2.9948178e-05 6.5863001e-06]
 ...
 [3.6398220e-08 2.0905568e-06 6.2079948e-08 ... 9.3570197e-06
  7.5054231e-06 5.0156767e-05]
 [4.7555076e-07 6.3403675e-08 4.5883439e-08 ... 5.2522708e-08
  8.9503745e-05 6.3449988e-07]
 [2.5404852e-05 7.6592278e-06 3.5255423e-04 ... 1.0632584e-06
  1.6873362e-06 8.2581998e-08]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.04841053597284481, 'accuracy_test:': 0.9839000105857849}

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
   d05bfb2..5d944d9  master     -> origin/master
Updating d05bfb2..5d944d9
Fast-forward
 error_list/20200520/list_log_testall_20200520.md | 432 +++++++++++++++++++++++
 1 file changed, 432 insertions(+)
[master 5eae1aa] ml_store
 1 file changed, 2035 insertions(+)
To github.com:arita37/mlmodels_store.git
   5d944d9..5eae1aa  master -> master





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
{'loss': 0.5164135843515396, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-20 16:28:35.840476: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
[master fc4a822] ml_store
 1 file changed, 234 insertions(+)
To github.com:arita37/mlmodels_store.git
   5eae1aa..fc4a822  master -> master





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
[master 80631b5] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   fc4a822..80631b5  master -> master





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
	Data preprocessing and feature engineering runtime = 0.21s ...
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
 40%|████      | 2/5 [00:19<00:29,  9.67s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.9706566400175755, 'learning_rate': 0.04552924364721135, 'min_data_in_leaf': 20, 'num_leaves': 50} and reward: 0.3934
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xef\x0f\x9e\x83\x90\xa9\xd5X\r\x00\x00\x00learning_rateq\x02G?\xa7O\x9b\xe8\xf3\xe92X\x10\x00\x00\x00min_data_in_leafq\x03K\x14X\n\x00\x00\x00num_leavesq\x04K2u.' and reward: 0.3934
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xef\x0f\x9e\x83\x90\xa9\xd5X\r\x00\x00\x00learning_rateq\x02G?\xa7O\x9b\xe8\xf3\xe92X\x10\x00\x00\x00min_data_in_leafq\x03K\x14X\n\x00\x00\x00num_leavesq\x04K2u.' and reward: 0.3934
 60%|██████    | 3/5 [00:45<00:29, 14.54s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.8034960272699736, 'learning_rate': 0.01375088427432148, 'min_data_in_leaf': 8, 'num_leaves': 59} and reward: 0.3912
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9\xb6=L\xf2\xe5\x0eX\r\x00\x00\x00learning_rateq\x02G?\x8c)lq\xfe\x8c\x84X\x10\x00\x00\x00min_data_in_leafq\x03K\x08X\n\x00\x00\x00num_leavesq\x04K;u.' and reward: 0.3912
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9\xb6=L\xf2\xe5\x0eX\r\x00\x00\x00learning_rateq\x02G?\x8c)lq\xfe\x8c\x84X\x10\x00\x00\x00min_data_in_leafq\x03K\x08X\n\x00\x00\x00num_leavesq\x04K;u.' and reward: 0.3912
 80%|████████  | 4/5 [01:12<00:18, 18.48s/it] 80%|████████  | 4/5 [01:12<00:18, 18.23s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.8759049901496346, 'learning_rate': 0.01327718600009142, 'min_data_in_leaf': 10, 'num_leaves': 64} and reward: 0.3916
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xec\x07i\xe6\xe3\x11yX\r\x00\x00\x00learning_rateq\x02G?\x8b1\x11\xbd9\xf6>X\x10\x00\x00\x00min_data_in_leafq\x03K\nX\n\x00\x00\x00num_leavesq\x04K@u.' and reward: 0.3916
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xec\x07i\xe6\xe3\x11yX\r\x00\x00\x00learning_rateq\x02G?\x8b1\x11\xbd9\xf6>X\x10\x00\x00\x00min_data_in_leafq\x03K\nX\n\x00\x00\x00num_leavesq\x04K@u.' and reward: 0.3916
Time for Gradient Boosting hyperparameter optimization: 103.43209385871887
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.9706566400175755, 'learning_rate': 0.04552924364721135, 'min_data_in_leaf': 20, 'num_leaves': 50}
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
 40%|████      | 2/5 [00:46<01:09, 23.33s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.22721314854819896, 'embedding_size_factor': 1.372676257895295, 'layers.choice': 2, 'learning_rate': 0.002949285703459095, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 1.3624261381594738e-07} and reward: 0.3856
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xcd\x15R\t\x1e+\x90X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5\xf6{a:~\xc9X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?h)\x19\xb4\x92\xc7\x06X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\x82IB\xb4h 6u.' and reward: 0.3856
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xcd\x15R\t\x1e+\x90X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5\xf6{a:~\xc9X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?h)\x19\xb4\x92\xc7\x06X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\x82IB\xb4h 6u.' and reward: 0.3856
 60%|██████    | 3/5 [01:31<00:59, 29.74s/it] 60%|██████    | 3/5 [01:31<01:00, 30.45s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.00403066969173388, 'embedding_size_factor': 1.2989171703337345, 'layers.choice': 1, 'learning_rate': 0.0024867993591072227, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 0.0015483303570301313} and reward: 0.3656
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?p\x82v\xa8\x1c\x92AX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf4\xc8]^\xec\xbdrX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?d_2=i\x18\xf4X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?Y^+\x0f\xcc]Nu.' and reward: 0.3656
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?p\x82v\xa8\x1c\x92AX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf4\xc8]^\xec\xbdrX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?d_2=i\x18\xf4X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?Y^+\x0f\xcc]Nu.' and reward: 0.3656
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 162.2438461780548
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
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.78s of the -149.92s of remaining time.
Ensemble size: 42
Ensemble weights: 
[0.11904762 0.07142857 0.11904762 0.19047619 0.11904762 0.33333333
 0.04761905]
	0.3994	 = Validation accuracy score
	1.37s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 271.33s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7fcfdfdd95c0>

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
   80631b5..6cbf4cc  master     -> origin/master
Updating 80631b5..6cbf4cc
Fast-forward
 error_list/20200520/list_log_testall_20200520.md | 175 +++++++++++++++++++++++
 1 file changed, 175 insertions(+)
[master ffd88e7] ml_store
 1 file changed, 218 insertions(+)
To github.com:arita37/mlmodels_store.git
   6cbf4cc..ffd88e7  master -> master





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
[master 62c3769] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   ffd88e7..62c3769  master -> master





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
100%|██████████| 10/10 [00:02<00:00,  4.19it/s, avg_epoch_loss=5.27]
INFO:root:Epoch[0] Elapsed time 2.386 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.267541
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.267541456222534 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fae32646400>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fae32646400>

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
Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 115.89it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 1124.7246907552083,
    "abs_error": 382.03289794921875,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 2.531330812109776,
    "sMAPE": 0.524123908041526,
    "MSIS": 101.2532357197181,
    "QuantileLoss[0.5]": 382.03289794921875,
    "Coverage[0.5]": 1.0,
    "RMSE": 33.53691534347201,
    "NRMSE": 0.7060403230204634,
    "ND": 0.670233154296875,
    "wQuantileLoss[0.5]": 0.670233154296875,
    "mean_wQuantileLoss": 0.670233154296875,
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
100%|██████████| 10/10 [00:01<00:00,  8.62it/s, avg_epoch_loss=2.71e+3]
INFO:root:Epoch[0] Elapsed time 1.160 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=2713.411247
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 2713.4112467447917 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fae060d6278>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fae060d6278>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 134.61it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
100%|██████████| 10/10 [00:01<00:00,  6.38it/s, avg_epoch_loss=5.24]
INFO:root:Epoch[0] Elapsed time 1.568 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.237665
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.2376645565032955 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fae04090668>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fae04090668>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 179.29it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 260.19704182942706,
    "abs_error": 169.43527221679688,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.1226696117610162,
    "sMAPE": 0.2819248733058988,
    "MSIS": 44.906786088104184,
    "QuantileLoss[0.5]": 169.4352684020996,
    "Coverage[0.5]": 0.75,
    "RMSE": 16.130624347167316,
    "NRMSE": 0.3395920915193119,
    "ND": 0.29725486353824015,
    "wQuantileLoss[0.5]": 0.2972548568457888,
    "mean_wQuantileLoss": 0.2972548568457888,
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
 30%|███       | 3/10 [00:11<00:25,  3.70s/it, avg_epoch_loss=6.93] 70%|███████   | 7/10 [00:24<00:10,  3.58s/it, avg_epoch_loss=6.9] 100%|██████████| 10/10 [00:34<00:00,  3.41s/it, avg_epoch_loss=6.86]
INFO:root:Epoch[0] Elapsed time 34.056 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.863243
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.863243055343628 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fae0404bd30>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fae0404bd30>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 173.58it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 54197.776041666664,
    "abs_error": 2749.05517578125,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 18.215101652240218,
    "sMAPE": 1.4168554771361555,
    "MSIS": 728.6040919722253,
    "QuantileLoss[0.5]": 2749.0551147460938,
    "Coverage[0.5]": 1.0,
    "RMSE": 232.8041581279567,
    "NRMSE": 4.901140171114878,
    "ND": 4.822903817160087,
    "wQuantileLoss[0.5]": 4.822903710080866,
    "mean_wQuantileLoss": 4.822903710080866,
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
100%|██████████| 10/10 [00:00<00:00, 59.08it/s, avg_epoch_loss=5.04]
INFO:root:Epoch[0] Elapsed time 0.170 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.044121
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.044120502471924 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fadeef36048>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fadeef36048>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 144.87it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 348.1607259114583,
    "abs_error": 204.6475830078125,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.3559846161738862,
    "sMAPE": 0.33284584312642707,
    "MSIS": 54.239384646955436,
    "QuantileLoss[0.5]": 204.64759063720703,
    "Coverage[0.5]": 0.75,
    "RMSE": 18.659065515492955,
    "NRMSE": 0.39282243190511484,
    "ND": 0.3590308473821272,
    "wQuantileLoss[0.5]": 0.3590308607670299,
    "mean_wQuantileLoss": 0.3590308607670299,
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
100%|██████████| 10/10 [00:01<00:00,  9.77it/s, avg_epoch_loss=144]
INFO:root:Epoch[0] Elapsed time 1.025 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=144.412806
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 144.41280616614463 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fadeefebb38>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fadeefebb38>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 176.51it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
 10%|█         | 1/10 [01:49<16:28, 109.83s/it, avg_epoch_loss=0.488] 20%|██        | 2/10 [04:46<17:20, 130.00s/it, avg_epoch_loss=0.47]  30%|███       | 3/10 [08:06<17:36, 150.99s/it, avg_epoch_loss=0.453] 40%|████      | 4/10 [11:06<15:56, 159.48s/it, avg_epoch_loss=0.437] 50%|█████     | 5/10 [14:23<14:13, 170.74s/it, avg_epoch_loss=0.424] 60%|██████    | 6/10 [17:28<11:41, 175.25s/it, avg_epoch_loss=0.414] 70%|███████   | 7/10 [20:51<09:10, 183.53s/it, avg_epoch_loss=0.407] 80%|████████  | 8/10 [23:51<06:04, 182.38s/it, avg_epoch_loss=0.404] 90%|█████████ | 9/10 [26:59<03:04, 184.20s/it, avg_epoch_loss=0.401]100%|██████████| 10/10 [30:15<00:00, 187.61s/it, avg_epoch_loss=0.399]100%|██████████| 10/10 [30:15<00:00, 181.56s/it, avg_epoch_loss=0.399]
INFO:root:Epoch[0] Elapsed time 1815.587 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.398938
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.3989381194114685 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fadeef67198>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fadeef67198>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 20.78it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
   62c3769..315b974  master     -> origin/master
Updating 62c3769..315b974
Fast-forward
 error_list/20200520/list_log_pullrequest_20200520.md | 2 +-
 error_list/20200520/list_log_testall_20200520.md     | 7 +++++++
 2 files changed, 8 insertions(+), 1 deletion(-)
[master 18b631c] ml_store
 1 file changed, 506 insertions(+)
To github.com:arita37/mlmodels_store.git
   315b974..18b631c  master -> master





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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7f8ace16c4a8> 

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
