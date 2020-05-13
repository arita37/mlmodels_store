
  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_all 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2

 ************************************************************************************************************************

  ############Check model ################################ 

  ['model_keras.keras_gan', 'model_keras.textcnn_dataloader', 'model_keras.nbeats', 'model_keras.01_deepctr', 'model_keras.textvae', 'model_keras.namentity_crm_bilstm_dataloader', 'model_keras.Autokeras', 'model_keras.charcnn_zhang', 'model_keras.charcnn', 'model_keras.namentity_crm_bilstm', 'model_keras.textcnn', 'model_keras.armdn', 'model_keras.02_cnn', 'model_tf.1_lstm', 'model_tf.temporal_fusion_google', 'model_gluon.gluon_automl', 'model_gluon.fb_prophet', 'model_gluon.gluonts_model', 'model_sklearn.model_sklearn', 'model_sklearn.model_lightgbm', 'model_tch.nbeats', 'model_tch.transformer_classifier', 'model_tch.matchzoo_models', 'model_tch.torchhub', 'model_tch.03_nbeats_dataloader', 'model_tch.transformer_sentence', 'model_tch.pytorch_vae', 'model_tch.pplm', 'model_tch.textcnn', 'model_tch.mlp'] 

  Used ['model_keras.keras_gan', 'model_keras.textcnn_dataloader', 'model_keras.nbeats', 'model_keras.01_deepctr', 'model_keras.textvae', 'model_keras.namentity_crm_bilstm_dataloader', 'model_keras.Autokeras', 'model_keras.charcnn_zhang', 'model_keras.charcnn', 'model_keras.namentity_crm_bilstm', 'model_keras.textcnn', 'model_keras.armdn', 'model_keras.02_cnn', 'model_tf.1_lstm', 'model_tf.temporal_fusion_google', 'model_gluon.gluon_automl', 'model_gluon.fb_prophet', 'model_gluon.gluonts_model', 'model_sklearn.model_sklearn', 'model_sklearn.model_lightgbm', 'model_tch.nbeats', 'model_tch.transformer_classifier', 'model_tch.matchzoo_models', 'model_tch.torchhub', 'model_tch.03_nbeats_dataloader', 'model_tch.transformer_sentence', 'model_tch.pytorch_vae', 'model_tch.pplm', 'model_tch.textcnn', 'model_tch.mlp'] 





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//keras_gan.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//keras_gan.py", line 31, in <module>
    'AAE' : kg.aae.aae,
AttributeError: module 'mlmodels.model_keras.raw.keras_gan' has no attribute 'aae'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
Already up to date.
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
[master 29ba379] ml_store
 1 file changed, 60 insertions(+)
 create mode 100644 log_testall/log_testall_2020-05-13-16-11_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py
To github.com:arita37/mlmodels_store.git
   f52202c..29ba379  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn_dataloader.py 

  #### Module init   ############################################ 

  <module 'mlmodels.model_keras.textcnn_dataloader' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textcnn_dataloader.py'> 

  #### Loading params   ############################################## 
Using TensorFlow backend.
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn_dataloader.py", line 275, in <module>
    test_module(model_uri = MODEL_URI, param_pars= param_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 257, in test_module
    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textcnn_dataloader.py", line 182, in get_params
    cf = json.load(open(data_path, mode='r'))
FileNotFoundError: [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/textcnn_keras.json'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
Already up to date.
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
[master 45dff9c] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   29ba379..45dff9c  master -> master





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

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Already up to date.
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
[master 2192866] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   45dff9c..2192866  master -> master





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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         32          sequence_max[0][0]               
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
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         28          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-13 16:12:40.471014: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 16:12:40.485184: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 16:12:40.485453: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555c67dea640 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 16:12:40.485474: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 2ms/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2501 - val_binary_crossentropy: 0.6933

  #### metrics   #################################################### 
{'MSE': 0.2498324888962129}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         32          sequence_max[0][0]               
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_max[0][0]               
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
Total params: 428
Trainable params: 428
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2527 - binary_crossentropy: 0.6986500/500 [==============================] - 1s 2ms/sample - loss: 0.2504 - binary_crossentropy: 0.6936 - val_loss: 0.2497 - val_binary_crossentropy: 0.6925

  #### metrics   #################################################### 
{'MSE': 0.24996326535053004}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_max[0][0]               
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
Total params: 428
Trainable params: 428
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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         12          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_2[0][0]           
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
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 1s 3ms/sample - loss: 0.2501 - binary_crossentropy: 0.6933 - val_loss: 0.2503 - val_binary_crossentropy: 0.6937

  #### metrics   #################################################### 
{'MSE': 0.2500762349170346}

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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         12          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_2[0][0]           
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
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
weighted_sequence_layer_9 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
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
Total params: 403
Trainable params: 403
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2645 - binary_crossentropy: 0.8511500/500 [==============================] - 2s 3ms/sample - loss: 0.2704 - binary_crossentropy: 0.9956 - val_loss: 0.2662 - val_binary_crossentropy: 0.9359

  #### metrics   #################################################### 
{'MSE': 0.26583702442576773}

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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
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
Total params: 403
Trainable params: 403
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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_max[0][0]               
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
Total params: 163
Trainable params: 163
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 3s - loss: 0.2736 - binary_crossentropy: 0.7433500/500 [==============================] - 2s 4ms/sample - loss: 0.2609 - binary_crossentropy: 0.7165 - val_loss: 0.2571 - val_binary_crossentropy: 0.7077

  #### metrics   #################################################### 
{'MSE': 0.2575561128364009}

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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_max[0][0]               
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
Total params: 163
Trainable params: 163
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-13 16:14:16.145613: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 16:14:16.147848: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 16:14:16.154542: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-13 16:14:16.167349: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-13 16:14:16.169483: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 16:14:16.172029: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 16:14:16.173766: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2470 - val_binary_crossentropy: 0.6871
2020-05-13 16:14:17.746130: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 16:14:17.748131: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 16:14:17.752983: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-13 16:14:17.764145: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-13 16:14:17.766103: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 16:14:17.767767: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 16:14:17.769315: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24566510988050183}

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
2020-05-13 16:14:46.239612: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 16:14:46.241246: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 16:14:46.245716: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-13 16:14:46.253373: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-13 16:14:46.254621: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 16:14:46.255738: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 16:14:46.256871: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2490 - val_binary_crossentropy: 0.6911
2020-05-13 16:14:48.098771: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 16:14:48.100046: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 16:14:48.103022: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-13 16:14:48.109002: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-13 16:14:48.110049: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 16:14:48.110977: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 16:14:48.111824: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24856632631105904}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-13 16:15:27.358342: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 16:15:27.363626: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 16:15:27.379989: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-13 16:15:27.409156: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-13 16:15:27.414086: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 16:15:27.418838: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 16:15:27.423537: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 6s 6s/sample - loss: 0.1720 - binary_crossentropy: 0.5357 - val_loss: 0.3020 - val_binary_crossentropy: 0.8098
2020-05-13 16:15:30.043663: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 16:15:30.048634: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 16:15:30.062387: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-13 16:15:30.089862: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-13 16:15:30.094640: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 16:15:30.098799: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 16:15:30.102839: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2260217820694225}

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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
Total params: 690
Trainable params: 690
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2488 - binary_crossentropy: 0.6908500/500 [==============================] - 5s 10ms/sample - loss: 0.2512 - binary_crossentropy: 0.6956 - val_loss: 0.2565 - val_binary_crossentropy: 0.7327

  #### metrics   #################################################### 
{'MSE': 0.2536009167874031}

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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
Total params: 690
Trainable params: 690
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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 2)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 2)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 2)         18          sequence_max[0][0]               
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
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         8           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         8           sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         9           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_2[0][0]           
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
100/500 [=====>........................] - ETA: 8s - loss: 0.2883 - binary_crossentropy: 0.8150500/500 [==============================] - 5s 10ms/sample - loss: 0.3005 - binary_crossentropy: 0.8309 - val_loss: 0.3059 - val_binary_crossentropy: 0.8370

  #### metrics   #################################################### 
{'MSE': 0.3020432806199789}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 2)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 2)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 2)         18          sequence_max[0][0]               
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
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         8           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         8           sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         9           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_2[0][0]           
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_max[0][0]               
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
Total params: 1,904
Trainable params: 1,904
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.4200 - binary_crossentropy: 6.4767500/500 [==============================] - 5s 10ms/sample - loss: 0.5000 - binary_crossentropy: 7.7054 - val_loss: 0.4620 - val_binary_crossentropy: 7.1263

  #### metrics   #################################################### 
{'MSE': 0.488}

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
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_max[0][0]               
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
regionsequence_sum (InputLayer) [(None, 9)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 5)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 1)]          0                                            
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
region_10sparse_seq_emb_regions (None, 5, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 1, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         7           regionsequence_max[0][0]         
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
Total params: 232
Trainable params: 232
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 10s - loss: 0.3085 - binary_crossentropy: 1.9901500/500 [==============================] - 6s 13ms/sample - loss: 0.2699 - binary_crossentropy: 1.3519 - val_loss: 0.2994 - val_binary_crossentropy: 1.8364

  #### metrics   #################################################### 
{'MSE': 0.28412118443081746}

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
regionsequence_mean (InputLayer [(None, 5)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 1)]          0                                            
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
region_10sparse_seq_emb_regions (None, 5, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 1, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         7           regionsequence_max[0][0]         
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
Total params: 232
Trainable params: 232
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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_max[0][0]               
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
100/500 [=====>........................] - ETA: 7s - loss: 0.2773 - binary_crossentropy: 1.1412500/500 [==============================] - 6s 12ms/sample - loss: 0.2698 - binary_crossentropy: 1.0718 - val_loss: 0.2718 - val_binary_crossentropy: 1.1806

  #### metrics   #################################################### 
{'MSE': 0.26793396015837717}

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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_max[0][0]               
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         16          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         20          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 4, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 5, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         20          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         20          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 4, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         20          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 5, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 4, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 5, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 4, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 5, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           hash_10[0][0]                    
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
Total params: 2,984
Trainable params: 2,904
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 10s - loss: 0.5400 - binary_crossentropy: 8.3295500/500 [==============================] - 7s 14ms/sample - loss: 0.5340 - binary_crossentropy: 8.2369 - val_loss: 0.5060 - val_binary_crossentropy: 7.8050

  #### metrics   #################################################### 
{'MSE': 0.52}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         16          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         20          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 4, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 5, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         20          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         20          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 4, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         20          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 5, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 4, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 5, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 4, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 5, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           hash_10[0][0]                    
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
Total params: 2,984
Trainable params: 2,904
Non-trainable params: 80
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'PNN', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'PNN', 'sparse_feature_num': 1, 'dense_feature_num': 1} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_PNN.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Model: "model_14"
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
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_190 (Seq (None, 1, 4)         0           weighted_sequence_layer_43[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_191 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_192 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_193 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
no_mask_119 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_190[0][0] 
                                                                 sequence_pooling_layer_191[0][0] 
                                                                 sequence_pooling_layer_192[0][0] 
                                                                 sequence_pooling_layer_193[0][0] 
__________________________________________________________________________________________________
concatenate_83 (Concatenate)    (None, 1, 20)        0           no_mask_119[0][0]                
                                                                 no_mask_119[1][0]                
                                                                 no_mask_119[2][0]                
                                                                 no_mask_119[3][0]                
                                                                 no_mask_119[4][0]                
__________________________________________________________________________________________________
inner_product_layer (InnerProdu (None, 10, 1)        0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_190[0][0] 
                                                                 sequence_pooling_layer_191[0][0] 
                                                                 sequence_pooling_layer_192[0][0] 
                                                                 sequence_pooling_layer_193[0][0] 
__________________________________________________________________________________________________
reshape (Reshape)               (None, 20)           0           concatenate_83[0][0]             
__________________________________________________________________________________________________
flatten_31 (Flatten)            (None, 10)           0           inner_product_layer[0][0]        
__________________________________________________________________________________________________
outter_product_layer (OutterPro (None, 10)           160         sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_190[0][0] 
                                                                 sequence_pooling_layer_191[0][0] 
                                                                 sequence_pooling_layer_192[0][0] 
                                                                 sequence_pooling_layer_193[0][0] 
__________________________________________________________________________________________________
concatenate_84 (Concatenate)    (None, 40)           0           reshape[0][0]                    
                                                                 flatten_31[0][0]                 
                                                                 outter_product_layer[0][0]       
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_120 (NoMask)            (None, 40)           0           concatenate_84[0][0]             
__________________________________________________________________________________________________
no_mask_121 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
flatten_32 (Flatten)            (None, 40)           0           no_mask_120[0][0]                
__________________________________________________________________________________________________
flatten_33 (Flatten)            (None, 1)            0           no_mask_121[0][0]                
__________________________________________________________________________________________________
no_mask_122 (NoMask)            multiple             0           flatten_32[0][0]                 
                                                                 flatten_33[0][0]                 
__________________________________________________________________________________________________
concatenate_85 (Concatenate)    (None, 41)           0           no_mask_122[0][0]                
                                                                 no_mask_122[1][0]                
__________________________________________________________________________________________________
dnn_19 (DNN)                    (None, 4)            188         concatenate_85[0][0]             
__________________________________________________________________________________________________
dense_12 (Dense)                (None, 1)            4           dnn_19[0][0]                     
__________________________________________________________________________________________________
prediction_layer_17 (Prediction (None, 1)            1           dense_12[0][0]                   
==================================================================================================
Total params: 461
Trainable params: 461
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2518 - binary_crossentropy: 0.6968500/500 [==============================] - 7s 13ms/sample - loss: 0.2510 - binary_crossentropy: 0.6952 - val_loss: 0.2501 - val_binary_crossentropy: 0.6933

  #### metrics   #################################################### 
{'MSE': 0.2497415308498707}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_14"
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
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_190 (Seq (None, 1, 4)         0           weighted_sequence_layer_43[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_191 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_192 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_193 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
no_mask_119 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_190[0][0] 
                                                                 sequence_pooling_layer_191[0][0] 
                                                                 sequence_pooling_layer_192[0][0] 
                                                                 sequence_pooling_layer_193[0][0] 
__________________________________________________________________________________________________
concatenate_83 (Concatenate)    (None, 1, 20)        0           no_mask_119[0][0]                
                                                                 no_mask_119[1][0]                
                                                                 no_mask_119[2][0]                
                                                                 no_mask_119[3][0]                
                                                                 no_mask_119[4][0]                
__________________________________________________________________________________________________
inner_product_layer (InnerProdu (None, 10, 1)        0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_190[0][0] 
                                                                 sequence_pooling_layer_191[0][0] 
                                                                 sequence_pooling_layer_192[0][0] 
                                                                 sequence_pooling_layer_193[0][0] 
__________________________________________________________________________________________________
reshape (Reshape)               (None, 20)           0           concatenate_83[0][0]             
__________________________________________________________________________________________________
flatten_31 (Flatten)            (None, 10)           0           inner_product_layer[0][0]        
__________________________________________________________________________________________________
outter_product_layer (OutterPro (None, 10)           160         sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_190[0][0] 
                                                                 sequence_pooling_layer_191[0][0] 
                                                                 sequence_pooling_layer_192[0][0] 
                                                                 sequence_pooling_layer_193[0][0] 
__________________________________________________________________________________________________
concatenate_84 (Concatenate)    (None, 40)           0           reshape[0][0]                    
                                                                 flatten_31[0][0]                 
                                                                 outter_product_layer[0][0]       
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_120 (NoMask)            (None, 40)           0           concatenate_84[0][0]             
__________________________________________________________________________________________________
no_mask_121 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
flatten_32 (Flatten)            (None, 40)           0           no_mask_120[0][0]                
__________________________________________________________________________________________________
flatten_33 (Flatten)            (None, 1)            0           no_mask_121[0][0]                
__________________________________________________________________________________________________
no_mask_122 (NoMask)            multiple             0           flatten_32[0][0]                 
                                                                 flatten_33[0][0]                 
__________________________________________________________________________________________________
concatenate_85 (Concatenate)    (None, 41)           0           no_mask_122[0][0]                
                                                                 no_mask_122[1][0]                
__________________________________________________________________________________________________
dnn_19 (DNN)                    (None, 4)            188         concatenate_85[0][0]             
__________________________________________________________________________________________________
dense_12 (Dense)                (None, 1)            4           dnn_19[0][0]                     
__________________________________________________________________________________________________
prediction_layer_17 (Prediction (None, 1)            1           dense_12[0][0]                   
==================================================================================================
Total params: 461
Trainable params: 461
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'WDL', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'WDL', 'sparse_feature_num': 2, 'dense_feature_num': 0} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_WDL.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Model: "model_15"
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
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_44 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_194 (Seq (None, 1, 4)         0           weighted_sequence_layer_44[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_195 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_196 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_197 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
weighted_sequence_layer_45 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_125 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_194[0][0] 
                                                                 sequence_pooling_layer_195[0][0] 
                                                                 sequence_pooling_layer_196[0][0] 
                                                                 sequence_pooling_layer_197[0][0] 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_198 (Seq (None, 1, 1)         0           weighted_sequence_layer_45[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_199 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_200 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_201 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
concatenate_87 (Concatenate)    (None, 1, 24)        0           no_mask_125[0][0]                
                                                                 no_mask_125[1][0]                
                                                                 no_mask_125[2][0]                
                                                                 no_mask_125[3][0]                
                                                                 no_mask_125[4][0]                
                                                                 no_mask_125[5][0]                
__________________________________________________________________________________________________
no_mask_123 (NoMask)            (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_198[0][0] 
                                                                 sequence_pooling_layer_199[0][0] 
                                                                 sequence_pooling_layer_200[0][0] 
                                                                 sequence_pooling_layer_201[0][0] 
__________________________________________________________________________________________________
flatten_34 (Flatten)            (None, 24)           0           concatenate_87[0][0]             
__________________________________________________________________________________________________
concatenate_86 (Concatenate)    (None, 1, 6)         0           no_mask_123[0][0]                
                                                                 no_mask_123[1][0]                
                                                                 no_mask_123[2][0]                
                                                                 no_mask_123[3][0]                
                                                                 no_mask_123[4][0]                
                                                                 no_mask_123[5][0]                
__________________________________________________________________________________________________
dnn_20 (DNN)                    (None, 32)           1856        flatten_34[0][0]                 
__________________________________________________________________________________________________
linear_18 (Linear)              (None, 1, 1)         0           concatenate_86[0][0]             
__________________________________________________________________________________________________
dense_13 (Dense)                (None, 1)            32          dnn_20[0][0]                     
__________________________________________________________________________________________________
no_mask_124 (NoMask)            (None, 1, 1)         0           linear_18[0][0]                  
__________________________________________________________________________________________________
add_32 (Add)                    (None, 1, 1)         0           dense_13[0][0]                   
                                                                 no_mask_124[0][0]                
__________________________________________________________________________________________________
prediction_layer_18 (Prediction (None, 1)            1           add_32[0][0]                     
==================================================================================================
Total params: 2,039
Trainable params: 2,039
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 7s 14ms/sample - loss: 0.2501 - binary_crossentropy: 0.6933 - val_loss: 0.2523 - val_binary_crossentropy: 0.7242

  #### metrics   #################################################### 
{'MSE': 0.2509119782243603}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_15"
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
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_44 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_194 (Seq (None, 1, 4)         0           weighted_sequence_layer_44[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_195 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_196 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_197 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
weighted_sequence_layer_45 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_125 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_194[0][0] 
                                                                 sequence_pooling_layer_195[0][0] 
                                                                 sequence_pooling_layer_196[0][0] 
                                                                 sequence_pooling_layer_197[0][0] 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_198 (Seq (None, 1, 1)         0           weighted_sequence_layer_45[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_199 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_200 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_201 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
concatenate_87 (Concatenate)    (None, 1, 24)        0           no_mask_125[0][0]                
                                                                 no_mask_125[1][0]                
                                                                 no_mask_125[2][0]                
                                                                 no_mask_125[3][0]                
                                                                 no_mask_125[4][0]                
                                                                 no_mask_125[5][0]                
__________________________________________________________________________________________________
no_mask_123 (NoMask)            (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_198[0][0] 
                                                                 sequence_pooling_layer_199[0][0] 
                                                                 sequence_pooling_layer_200[0][0] 
                                                                 sequence_pooling_layer_201[0][0] 
__________________________________________________________________________________________________
flatten_34 (Flatten)            (None, 24)           0           concatenate_87[0][0]             
__________________________________________________________________________________________________
concatenate_86 (Concatenate)    (None, 1, 6)         0           no_mask_123[0][0]                
                                                                 no_mask_123[1][0]                
                                                                 no_mask_123[2][0]                
                                                                 no_mask_123[3][0]                
                                                                 no_mask_123[4][0]                
                                                                 no_mask_123[5][0]                
__________________________________________________________________________________________________
dnn_20 (DNN)                    (None, 32)           1856        flatten_34[0][0]                 
__________________________________________________________________________________________________
linear_18 (Linear)              (None, 1, 1)         0           concatenate_86[0][0]             
__________________________________________________________________________________________________
dense_13 (Dense)                (None, 1)            32          dnn_20[0][0]                     
__________________________________________________________________________________________________
no_mask_124 (NoMask)            (None, 1, 1)         0           linear_18[0][0]                  
__________________________________________________________________________________________________
add_32 (Add)                    (None, 1, 1)         0           dense_13[0][0]                   
                                                                 no_mask_124[0][0]                
__________________________________________________________________________________________________
prediction_layer_18 (Prediction (None, 1)            1           add_32[0][0]                     
==================================================================================================
Total params: 2,039
Trainable params: 2,039
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'xDeepFM', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'xDeepFM', 'sparse_feature_num': 1, 'dense_feature_num': 1} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_xDeepFM.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Model: "model_16"
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
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_47 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_206 (Seq (None, 1, 4)         0           weighted_sequence_layer_47[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_207 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_208 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_209 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_130 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_206[0][0] 
                                                                 sequence_pooling_layer_207[0][0] 
                                                                 sequence_pooling_layer_208[0][0] 
                                                                 sequence_pooling_layer_209[0][0] 
__________________________________________________________________________________________________
weighted_sequence_layer_48 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_90 (Concatenate)    (None, 1, 20)        0           no_mask_130[0][0]                
                                                                 no_mask_130[1][0]                
                                                                 no_mask_130[2][0]                
                                                                 no_mask_130[3][0]                
                                                                 no_mask_130[4][0]                
__________________________________________________________________________________________________
no_mask_131 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_210 (Seq (None, 1, 1)         0           weighted_sequence_layer_48[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_211 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_212 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_213 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
flatten_35 (Flatten)            (None, 20)           0           concatenate_90[0][0]             
__________________________________________________________________________________________________
flatten_36 (Flatten)            (None, 1)            0           no_mask_131[0][0]                
__________________________________________________________________________________________________
no_mask_126 (NoMask)            (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_210[0][0] 
                                                                 sequence_pooling_layer_211[0][0] 
                                                                 sequence_pooling_layer_212[0][0] 
                                                                 sequence_pooling_layer_213[0][0] 
__________________________________________________________________________________________________
no_mask_132 (NoMask)            multiple             0           flatten_35[0][0]                 
                                                                 flatten_36[0][0]                 
__________________________________________________________________________________________________
concatenate_88 (Concatenate)    (None, 1, 5)         0           no_mask_126[0][0]                
                                                                 no_mask_126[1][0]                
                                                                 no_mask_126[2][0]                
                                                                 no_mask_126[3][0]                
                                                                 no_mask_126[4][0]                
__________________________________________________________________________________________________
no_mask_127 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
concatenate_91 (Concatenate)    (None, 21)           0           no_mask_132[0][0]                
                                                                 no_mask_132[1][0]                
__________________________________________________________________________________________________
linear_19 (Linear)              (None, 1)            1           concatenate_88[0][0]             
                                                                 no_mask_127[0][0]                
__________________________________________________________________________________________________
dnn_21 (DNN)                    (None, 8)            176         concatenate_91[0][0]             
__________________________________________________________________________________________________
no_mask_128 (NoMask)            (None, 1)            0           linear_19[0][0]                  
__________________________________________________________________________________________________
dense_14 (Dense)                (None, 1)            8           dnn_21[0][0]                     
__________________________________________________________________________________________________
add_35 (Add)                    (None, 1)            0           no_mask_128[0][0]                
                                                                 dense_14[0][0]                   
__________________________________________________________________________________________________
prediction_layer_19 (Prediction (None, 1)            1           add_35[0][0]                     
==================================================================================================
Total params: 331
Trainable params: 331
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2652 - binary_crossentropy: 0.7264500/500 [==============================] - 7s 15ms/sample - loss: 0.2581 - binary_crossentropy: 0.7102 - val_loss: 0.2553 - val_binary_crossentropy: 0.7043

  #### metrics   #################################################### 
{'MSE': 0.2549827034644162}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_16"
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
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_47 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_206 (Seq (None, 1, 4)         0           weighted_sequence_layer_47[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_207 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_208 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_209 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_130 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_206[0][0] 
                                                                 sequence_pooling_layer_207[0][0] 
                                                                 sequence_pooling_layer_208[0][0] 
                                                                 sequence_pooling_layer_209[0][0] 
__________________________________________________________________________________________________
weighted_sequence_layer_48 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_90 (Concatenate)    (None, 1, 20)        0           no_mask_130[0][0]                
                                                                 no_mask_130[1][0]                
                                                                 no_mask_130[2][0]                
                                                                 no_mask_130[3][0]                
                                                                 no_mask_130[4][0]                
__________________________________________________________________________________________________
no_mask_131 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_210 (Seq (None, 1, 1)         0           weighted_sequence_layer_48[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_211 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_212 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_213 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
flatten_35 (Flatten)            (None, 20)           0           concatenate_90[0][0]             
__________________________________________________________________________________________________
flatten_36 (Flatten)            (None, 1)            0           no_mask_131[0][0]                
__________________________________________________________________________________________________
no_mask_126 (NoMask)            (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_210[0][0] 
                                                                 sequence_pooling_layer_211[0][0] 
                                                                 sequence_pooling_layer_212[0][0] 
                                                                 sequence_pooling_layer_213[0][0] 
__________________________________________________________________________________________________
no_mask_132 (NoMask)            multiple             0           flatten_35[0][0]                 
                                                                 flatten_36[0][0]                 
__________________________________________________________________________________________________
concatenate_88 (Concatenate)    (None, 1, 5)         0           no_mask_126[0][0]                
                                                                 no_mask_126[1][0]                
                                                                 no_mask_126[2][0]                
                                                                 no_mask_126[3][0]                
                                                                 no_mask_126[4][0]                
__________________________________________________________________________________________________
no_mask_127 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
concatenate_91 (Concatenate)    (None, 21)           0           no_mask_132[0][0]                
                                                                 no_mask_132[1][0]                
__________________________________________________________________________________________________
linear_19 (Linear)              (None, 1)            1           concatenate_88[0][0]             
                                                                 no_mask_127[0][0]                
__________________________________________________________________________________________________
dnn_21 (DNN)                    (None, 8)            176         concatenate_91[0][0]             
__________________________________________________________________________________________________
no_mask_128 (NoMask)            (None, 1)            0           linear_19[0][0]                  
__________________________________________________________________________________________________
dense_14 (Dense)                (None, 1)            8           dnn_21[0][0]                     
__________________________________________________________________________________________________
add_35 (Add)                    (None, 1)            0           no_mask_128[0][0]                
                                                                 dense_14[0][0]                   
__________________________________________________________________________________________________
prediction_layer_19 (Prediction (None, 1)            1           add_35[0][0]                     
==================================================================================================
Total params: 331
Trainable params: 331
Non-trainable params: 0
__________________________________________________________________________________________________

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   2192866..94b73d5  master     -> origin/master
Updating 2192866..94b73d5
Fast-forward
 error_list/20200513/list_log_benchmark_20200513.md |  438 +++--
 error_list/20200513/list_log_json_20200513.md      | 1146 +++++++-------
 error_list/20200513/list_log_jupyter_20200513.md   | 1668 ++++++++++----------
 .../20200513/list_log_pullrequest_20200513.md      |    2 +-
 error_list/20200513/list_log_test_cli_20200513.md  |  378 +++--
 error_list/20200513/list_log_testall_20200513.md   |  832 +---------
 ...-11_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py |  620 ++++++++
 7 files changed, 2443 insertions(+), 2641 deletions(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-13-16-11_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
[master 738d557] ml_store
 1 file changed, 5673 insertions(+)
To github.com:arita37/mlmodels_store.git
   94b73d5..738d557  master -> master





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

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Already up to date.
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
[master 29fb118] ml_store
 1 file changed, 50 insertions(+)
To github.com:arita37/mlmodels_store.git
   738d557..29fb118  master -> master





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

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Already up to date.
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
[master ccb26ee] ml_store
 1 file changed, 46 insertions(+)
To github.com:arita37/mlmodels_store.git
   29fb118..ccb26ee  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//Autokeras.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//Autokeras.py", line 12, in <module>
    import autokeras as ak
ModuleNotFoundError: No module named 'autokeras'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Already up to date.
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
[master 468eb0f] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   ccb26ee..468eb0f  master -> master





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

2020-05-13 16:29:36.214979: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 16:29:36.220373: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 16:29:36.220559: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559425d28450 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 16:29:36.220576: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 8s - loss: 1.3851
256/354 [====================>.........] - ETA: 3s - loss: 1.2568
354/354 [==============================] - 15s 43ms/step - loss: 1.5525 - val_loss: 2.1220

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

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Already up to date.
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
[master 525cb78] ml_store
 1 file changed, 149 insertions(+)
To github.com:arita37/mlmodels_store.git
   468eb0f..525cb78  master -> master





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

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Already up to date.
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
[master c6e4e16] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   525cb78..c6e4e16  master -> master





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

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Already up to date.
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
[master b0a816a] ml_store
 1 file changed, 44 insertions(+)
To github.com:arita37/mlmodels_store.git
   c6e4e16..b0a816a  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2285568/17464789 [==>...........................] - ETA: 0s
 8454144/17464789 [=============>................] - ETA: 0s
15613952/17464789 [=========================>....] - ETA: 0s
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
2020-05-13 16:30:41.060605: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 16:30:41.065435: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 16:30:41.065615: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557c10e91e30 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 16:30:41.065633: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 13s - loss: 7.8660 - accuracy: 0.4870
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7126 - accuracy: 0.4970
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7382 - accuracy: 0.4953 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.6705 - accuracy: 0.4997
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7341 - accuracy: 0.4956
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7688 - accuracy: 0.4933
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7126 - accuracy: 0.4970
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7375 - accuracy: 0.4954
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7433 - accuracy: 0.4950
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7464 - accuracy: 0.4948
11000/25000 [============>.................] - ETA: 4s - loss: 7.7377 - accuracy: 0.4954
12000/25000 [=============>................] - ETA: 4s - loss: 7.7011 - accuracy: 0.4978
13000/25000 [==============>...............] - ETA: 4s - loss: 7.7055 - accuracy: 0.4975
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6918 - accuracy: 0.4984
15000/25000 [=================>............] - ETA: 3s - loss: 7.7014 - accuracy: 0.4977
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7030 - accuracy: 0.4976
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7027 - accuracy: 0.4976
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7160 - accuracy: 0.4968
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7167 - accuracy: 0.4967
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7027 - accuracy: 0.4976
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6871 - accuracy: 0.4987
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6785 - accuracy: 0.4992
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6726 - accuracy: 0.4996
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6558 - accuracy: 0.5007
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
(<mlmodels.util.Model_empty object at 0x7f528e2e6d30>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f529b5b0fd0> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.9273 - accuracy: 0.4830
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7433 - accuracy: 0.4950
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6564 - accuracy: 0.5007 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6896 - accuracy: 0.4985
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6973 - accuracy: 0.4980
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7280 - accuracy: 0.4960
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7280 - accuracy: 0.4960
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7184 - accuracy: 0.4966
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7109 - accuracy: 0.4971
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7004 - accuracy: 0.4978
11000/25000 [============>.................] - ETA: 4s - loss: 7.6917 - accuracy: 0.4984
12000/25000 [=============>................] - ETA: 4s - loss: 7.6973 - accuracy: 0.4980
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7091 - accuracy: 0.4972
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7269 - accuracy: 0.4961
15000/25000 [=================>............] - ETA: 3s - loss: 7.6830 - accuracy: 0.4989
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6992 - accuracy: 0.4979
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6621 - accuracy: 0.5003
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6649 - accuracy: 0.5001
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6682 - accuracy: 0.4999
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6720 - accuracy: 0.4997
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6995 - accuracy: 0.4979
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6966 - accuracy: 0.4980
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6813 - accuracy: 0.4990
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6775 - accuracy: 0.4993
25000/25000 [==============================] - 10s 395us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 14s - loss: 7.7893 - accuracy: 0.4920
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6513 - accuracy: 0.5010
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6973 - accuracy: 0.4980 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.5516 - accuracy: 0.5075
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6298 - accuracy: 0.5024
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6360 - accuracy: 0.5020
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6754 - accuracy: 0.4994
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6436 - accuracy: 0.5015
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6530 - accuracy: 0.5009
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6636 - accuracy: 0.5002
11000/25000 [============>.................] - ETA: 4s - loss: 7.6206 - accuracy: 0.5030
12000/25000 [=============>................] - ETA: 4s - loss: 7.6360 - accuracy: 0.5020
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6478 - accuracy: 0.5012
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6228 - accuracy: 0.5029
15000/25000 [=================>............] - ETA: 3s - loss: 7.6319 - accuracy: 0.5023
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6484 - accuracy: 0.5012
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6630 - accuracy: 0.5002
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6743 - accuracy: 0.4995
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6908 - accuracy: 0.4984
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6950 - accuracy: 0.4981
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6863 - accuracy: 0.4987
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6757 - accuracy: 0.4994
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6866 - accuracy: 0.4987
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6743 - accuracy: 0.4995
25000/25000 [==============================] - 10s 393us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   b0a816a..54d0a8a  master     -> origin/master
Updating b0a816a..54d0a8a
Fast-forward
 error_list/20200513/list_log_benchmark_20200513.md |  438 ++++----
 .../20200513/list_log_dataloader_20200513.md       |    2 +-
 error_list/20200513/list_log_json_20200513.md      | 1146 ++++++++++----------
 .../20200513/list_log_pullrequest_20200513.md      |    2 +-
 error_list/20200513/list_log_test_cli_20200513.md  |  364 +++----
 error_list/20200513/list_log_testall_20200513.md   |   89 ++
 6 files changed, 1070 insertions(+), 971 deletions(-)
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
[master 2000f6b] ml_store
 1 file changed, 327 insertions(+)
To github.com:arita37/mlmodels_store.git
   54d0a8a..2000f6b  master -> master





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

13/13 [==============================] - 2s 148ms/step - loss: nan
Epoch 2/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 3/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 4/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 5/10

13/13 [==============================] - 0s 5ms/step - loss: nan
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

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Already up to date.
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
[master 91c2835] ml_store
 1 file changed, 125 insertions(+)
To github.com:arita37/mlmodels_store.git
   2000f6b..91c2835  master -> master





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
 2424832/11490434 [=====>........................] - ETA: 0s
 8847360/11490434 [======================>.......] - ETA: 0s
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

   32/60000 [..............................] - ETA: 8:06 - loss: 2.3337 - categorical_accuracy: 0.1562
   64/60000 [..............................] - ETA: 5:03 - loss: 2.2900 - categorical_accuracy: 0.1406
   96/60000 [..............................] - ETA: 3:57 - loss: 2.2460 - categorical_accuracy: 0.1562
  128/60000 [..............................] - ETA: 3:25 - loss: 2.2551 - categorical_accuracy: 0.1406
  160/60000 [..............................] - ETA: 3:06 - loss: 2.2494 - categorical_accuracy: 0.1312
  192/60000 [..............................] - ETA: 2:53 - loss: 2.2347 - categorical_accuracy: 0.1406
  224/60000 [..............................] - ETA: 2:45 - loss: 2.2116 - categorical_accuracy: 0.1741
  256/60000 [..............................] - ETA: 2:38 - loss: 2.1739 - categorical_accuracy: 0.1992
  288/60000 [..............................] - ETA: 2:32 - loss: 2.1439 - categorical_accuracy: 0.2083
  320/60000 [..............................] - ETA: 2:28 - loss: 2.1193 - categorical_accuracy: 0.2313
  352/60000 [..............................] - ETA: 2:24 - loss: 2.0699 - categorical_accuracy: 0.2614
  384/60000 [..............................] - ETA: 2:21 - loss: 2.0449 - categorical_accuracy: 0.2786
  416/60000 [..............................] - ETA: 2:19 - loss: 2.0154 - categorical_accuracy: 0.2957
  448/60000 [..............................] - ETA: 2:16 - loss: 1.9669 - categorical_accuracy: 0.3125
  480/60000 [..............................] - ETA: 2:14 - loss: 1.9513 - categorical_accuracy: 0.3271
  512/60000 [..............................] - ETA: 2:13 - loss: 1.9104 - categorical_accuracy: 0.3457
  544/60000 [..............................] - ETA: 2:11 - loss: 1.9045 - categorical_accuracy: 0.3456
  576/60000 [..............................] - ETA: 2:11 - loss: 1.8820 - categorical_accuracy: 0.3559
  608/60000 [..............................] - ETA: 2:09 - loss: 1.8473 - categorical_accuracy: 0.3717
  640/60000 [..............................] - ETA: 2:08 - loss: 1.8149 - categorical_accuracy: 0.3859
  672/60000 [..............................] - ETA: 2:07 - loss: 1.7957 - categorical_accuracy: 0.3943
  704/60000 [..............................] - ETA: 2:06 - loss: 1.7596 - categorical_accuracy: 0.4048
  736/60000 [..............................] - ETA: 2:05 - loss: 1.7366 - categorical_accuracy: 0.4171
  768/60000 [..............................] - ETA: 2:04 - loss: 1.7236 - categorical_accuracy: 0.4245
  800/60000 [..............................] - ETA: 2:03 - loss: 1.6939 - categorical_accuracy: 0.4387
  832/60000 [..............................] - ETA: 2:03 - loss: 1.6603 - categorical_accuracy: 0.4519
  864/60000 [..............................] - ETA: 2:02 - loss: 1.6315 - categorical_accuracy: 0.4595
  896/60000 [..............................] - ETA: 2:02 - loss: 1.6058 - categorical_accuracy: 0.4665
  928/60000 [..............................] - ETA: 2:01 - loss: 1.5858 - categorical_accuracy: 0.4731
  960/60000 [..............................] - ETA: 2:00 - loss: 1.5580 - categorical_accuracy: 0.4833
  992/60000 [..............................] - ETA: 2:00 - loss: 1.5383 - categorical_accuracy: 0.4879
 1024/60000 [..............................] - ETA: 1:59 - loss: 1.5227 - categorical_accuracy: 0.4902
 1056/60000 [..............................] - ETA: 1:59 - loss: 1.4956 - categorical_accuracy: 0.4981
 1088/60000 [..............................] - ETA: 1:58 - loss: 1.4796 - categorical_accuracy: 0.4982
 1120/60000 [..............................] - ETA: 1:58 - loss: 1.4620 - categorical_accuracy: 0.5045
 1152/60000 [..............................] - ETA: 1:58 - loss: 1.4419 - categorical_accuracy: 0.5130
 1184/60000 [..............................] - ETA: 1:57 - loss: 1.4374 - categorical_accuracy: 0.5144
 1216/60000 [..............................] - ETA: 1:57 - loss: 1.4210 - categorical_accuracy: 0.5214
 1248/60000 [..............................] - ETA: 1:56 - loss: 1.4029 - categorical_accuracy: 0.5264
 1280/60000 [..............................] - ETA: 1:56 - loss: 1.3908 - categorical_accuracy: 0.5320
 1312/60000 [..............................] - ETA: 1:56 - loss: 1.3834 - categorical_accuracy: 0.5343
 1344/60000 [..............................] - ETA: 1:55 - loss: 1.3605 - categorical_accuracy: 0.5424
 1376/60000 [..............................] - ETA: 1:56 - loss: 1.3466 - categorical_accuracy: 0.5458
 1408/60000 [..............................] - ETA: 1:56 - loss: 1.3404 - categorical_accuracy: 0.5476
 1440/60000 [..............................] - ETA: 1:55 - loss: 1.3212 - categorical_accuracy: 0.5549
 1472/60000 [..............................] - ETA: 1:55 - loss: 1.3049 - categorical_accuracy: 0.5625
 1504/60000 [..............................] - ETA: 1:55 - loss: 1.2932 - categorical_accuracy: 0.5665
 1536/60000 [..............................] - ETA: 1:54 - loss: 1.2813 - categorical_accuracy: 0.5736
 1568/60000 [..............................] - ETA: 1:54 - loss: 1.2641 - categorical_accuracy: 0.5810
 1600/60000 [..............................] - ETA: 1:54 - loss: 1.2458 - categorical_accuracy: 0.5881
 1632/60000 [..............................] - ETA: 1:54 - loss: 1.2375 - categorical_accuracy: 0.5907
 1664/60000 [..............................] - ETA: 1:53 - loss: 1.2241 - categorical_accuracy: 0.5962
 1696/60000 [..............................] - ETA: 1:53 - loss: 1.2223 - categorical_accuracy: 0.5991
 1728/60000 [..............................] - ETA: 1:53 - loss: 1.2096 - categorical_accuracy: 0.6024
 1760/60000 [..............................] - ETA: 1:53 - loss: 1.2017 - categorical_accuracy: 0.6068
 1792/60000 [..............................] - ETA: 1:53 - loss: 1.1946 - categorical_accuracy: 0.6088
 1824/60000 [..............................] - ETA: 1:52 - loss: 1.1857 - categorical_accuracy: 0.6113
 1856/60000 [..............................] - ETA: 1:52 - loss: 1.1752 - categorical_accuracy: 0.6148
 1888/60000 [..............................] - ETA: 1:52 - loss: 1.1644 - categorical_accuracy: 0.6181
 1920/60000 [..............................] - ETA: 1:52 - loss: 1.1534 - categorical_accuracy: 0.6219
 1952/60000 [..............................] - ETA: 1:52 - loss: 1.1437 - categorical_accuracy: 0.6245
 1984/60000 [..............................] - ETA: 1:51 - loss: 1.1333 - categorical_accuracy: 0.6280
 2016/60000 [>.............................] - ETA: 1:51 - loss: 1.1242 - categorical_accuracy: 0.6324
 2048/60000 [>.............................] - ETA: 1:51 - loss: 1.1149 - categorical_accuracy: 0.6362
 2080/60000 [>.............................] - ETA: 1:51 - loss: 1.1016 - categorical_accuracy: 0.6413
 2112/60000 [>.............................] - ETA: 1:51 - loss: 1.0922 - categorical_accuracy: 0.6439
 2144/60000 [>.............................] - ETA: 1:51 - loss: 1.0820 - categorical_accuracy: 0.6474
 2176/60000 [>.............................] - ETA: 1:51 - loss: 1.0738 - categorical_accuracy: 0.6507
 2208/60000 [>.............................] - ETA: 1:51 - loss: 1.0702 - categorical_accuracy: 0.6517
 2240/60000 [>.............................] - ETA: 1:50 - loss: 1.0626 - categorical_accuracy: 0.6545
 2272/60000 [>.............................] - ETA: 1:50 - loss: 1.0534 - categorical_accuracy: 0.6576
 2304/60000 [>.............................] - ETA: 1:50 - loss: 1.0494 - categorical_accuracy: 0.6593
 2336/60000 [>.............................] - ETA: 1:50 - loss: 1.0396 - categorical_accuracy: 0.6627
 2368/60000 [>.............................] - ETA: 1:50 - loss: 1.0293 - categorical_accuracy: 0.6664
 2400/60000 [>.............................] - ETA: 1:50 - loss: 1.0258 - categorical_accuracy: 0.6687
 2432/60000 [>.............................] - ETA: 1:50 - loss: 1.0199 - categorical_accuracy: 0.6702
 2464/60000 [>.............................] - ETA: 1:50 - loss: 1.0131 - categorical_accuracy: 0.6733
 2496/60000 [>.............................] - ETA: 1:49 - loss: 1.0079 - categorical_accuracy: 0.6747
 2528/60000 [>.............................] - ETA: 1:49 - loss: 1.0001 - categorical_accuracy: 0.6772
 2560/60000 [>.............................] - ETA: 1:49 - loss: 0.9920 - categorical_accuracy: 0.6805
 2592/60000 [>.............................] - ETA: 1:49 - loss: 0.9836 - categorical_accuracy: 0.6836
 2624/60000 [>.............................] - ETA: 1:49 - loss: 0.9758 - categorical_accuracy: 0.6860
 2656/60000 [>.............................] - ETA: 1:49 - loss: 0.9699 - categorical_accuracy: 0.6879
 2688/60000 [>.............................] - ETA: 1:49 - loss: 0.9646 - categorical_accuracy: 0.6897
 2720/60000 [>.............................] - ETA: 1:49 - loss: 0.9578 - categorical_accuracy: 0.6923
 2752/60000 [>.............................] - ETA: 1:49 - loss: 0.9497 - categorical_accuracy: 0.6948
 2784/60000 [>.............................] - ETA: 1:49 - loss: 0.9421 - categorical_accuracy: 0.6968
 2816/60000 [>.............................] - ETA: 1:49 - loss: 0.9338 - categorical_accuracy: 0.6999
 2848/60000 [>.............................] - ETA: 1:48 - loss: 0.9262 - categorical_accuracy: 0.7026
 2880/60000 [>.............................] - ETA: 1:48 - loss: 0.9195 - categorical_accuracy: 0.7052
 2912/60000 [>.............................] - ETA: 1:48 - loss: 0.9159 - categorical_accuracy: 0.7060
 2944/60000 [>.............................] - ETA: 1:48 - loss: 0.9077 - categorical_accuracy: 0.7089
 2976/60000 [>.............................] - ETA: 1:48 - loss: 0.9006 - categorical_accuracy: 0.7114
 3008/60000 [>.............................] - ETA: 1:48 - loss: 0.8946 - categorical_accuracy: 0.7131
 3040/60000 [>.............................] - ETA: 1:48 - loss: 0.8867 - categorical_accuracy: 0.7158
 3072/60000 [>.............................] - ETA: 1:48 - loss: 0.8880 - categorical_accuracy: 0.7165
 3104/60000 [>.............................] - ETA: 1:48 - loss: 0.8821 - categorical_accuracy: 0.7184
 3136/60000 [>.............................] - ETA: 1:48 - loss: 0.8772 - categorical_accuracy: 0.7200
 3168/60000 [>.............................] - ETA: 1:48 - loss: 0.8710 - categorical_accuracy: 0.7219
 3200/60000 [>.............................] - ETA: 1:48 - loss: 0.8667 - categorical_accuracy: 0.7234
 3232/60000 [>.............................] - ETA: 1:47 - loss: 0.8624 - categorical_accuracy: 0.7246
 3264/60000 [>.............................] - ETA: 1:47 - loss: 0.8557 - categorical_accuracy: 0.7270
 3296/60000 [>.............................] - ETA: 1:47 - loss: 0.8506 - categorical_accuracy: 0.7285
 3328/60000 [>.............................] - ETA: 1:47 - loss: 0.8469 - categorical_accuracy: 0.7302
 3360/60000 [>.............................] - ETA: 1:47 - loss: 0.8436 - categorical_accuracy: 0.7312
 3392/60000 [>.............................] - ETA: 1:47 - loss: 0.8393 - categorical_accuracy: 0.7323
 3424/60000 [>.............................] - ETA: 1:47 - loss: 0.8347 - categorical_accuracy: 0.7336
 3456/60000 [>.............................] - ETA: 1:47 - loss: 0.8319 - categorical_accuracy: 0.7344
 3488/60000 [>.............................] - ETA: 1:47 - loss: 0.8280 - categorical_accuracy: 0.7354
 3520/60000 [>.............................] - ETA: 1:46 - loss: 0.8258 - categorical_accuracy: 0.7364
 3552/60000 [>.............................] - ETA: 1:46 - loss: 0.8209 - categorical_accuracy: 0.7382
 3584/60000 [>.............................] - ETA: 1:46 - loss: 0.8186 - categorical_accuracy: 0.7383
 3616/60000 [>.............................] - ETA: 1:46 - loss: 0.8135 - categorical_accuracy: 0.7400
 3648/60000 [>.............................] - ETA: 1:46 - loss: 0.8082 - categorical_accuracy: 0.7418
 3680/60000 [>.............................] - ETA: 1:46 - loss: 0.8060 - categorical_accuracy: 0.7424
 3712/60000 [>.............................] - ETA: 1:46 - loss: 0.8021 - categorical_accuracy: 0.7430
 3744/60000 [>.............................] - ETA: 1:46 - loss: 0.7982 - categorical_accuracy: 0.7441
 3776/60000 [>.............................] - ETA: 1:46 - loss: 0.7946 - categorical_accuracy: 0.7452
 3808/60000 [>.............................] - ETA: 1:46 - loss: 0.7912 - categorical_accuracy: 0.7463
 3840/60000 [>.............................] - ETA: 1:45 - loss: 0.7889 - categorical_accuracy: 0.7471
 3872/60000 [>.............................] - ETA: 1:45 - loss: 0.7862 - categorical_accuracy: 0.7482
 3904/60000 [>.............................] - ETA: 1:45 - loss: 0.7830 - categorical_accuracy: 0.7492
 3936/60000 [>.............................] - ETA: 1:45 - loss: 0.7786 - categorical_accuracy: 0.7510
 3968/60000 [>.............................] - ETA: 1:45 - loss: 0.7762 - categorical_accuracy: 0.7515
 4000/60000 [=>............................] - ETA: 1:45 - loss: 0.7723 - categorical_accuracy: 0.7527
 4032/60000 [=>............................] - ETA: 1:45 - loss: 0.7683 - categorical_accuracy: 0.7540
 4064/60000 [=>............................] - ETA: 1:45 - loss: 0.7649 - categorical_accuracy: 0.7552
 4096/60000 [=>............................] - ETA: 1:45 - loss: 0.7601 - categorical_accuracy: 0.7568
 4128/60000 [=>............................] - ETA: 1:45 - loss: 0.7564 - categorical_accuracy: 0.7578
 4160/60000 [=>............................] - ETA: 1:45 - loss: 0.7539 - categorical_accuracy: 0.7582
 4192/60000 [=>............................] - ETA: 1:45 - loss: 0.7513 - categorical_accuracy: 0.7591
 4224/60000 [=>............................] - ETA: 1:44 - loss: 0.7487 - categorical_accuracy: 0.7602
 4256/60000 [=>............................] - ETA: 1:44 - loss: 0.7459 - categorical_accuracy: 0.7608
 4288/60000 [=>............................] - ETA: 1:44 - loss: 0.7428 - categorical_accuracy: 0.7614
 4320/60000 [=>............................] - ETA: 1:44 - loss: 0.7401 - categorical_accuracy: 0.7625
 4352/60000 [=>............................] - ETA: 1:44 - loss: 0.7361 - categorical_accuracy: 0.7638
 4384/60000 [=>............................] - ETA: 1:44 - loss: 0.7342 - categorical_accuracy: 0.7653
 4416/60000 [=>............................] - ETA: 1:44 - loss: 0.7316 - categorical_accuracy: 0.7659
 4448/60000 [=>............................] - ETA: 1:44 - loss: 0.7290 - categorical_accuracy: 0.7666
 4480/60000 [=>............................] - ETA: 1:44 - loss: 0.7258 - categorical_accuracy: 0.7674
 4512/60000 [=>............................] - ETA: 1:44 - loss: 0.7233 - categorical_accuracy: 0.7684
 4544/60000 [=>............................] - ETA: 1:44 - loss: 0.7208 - categorical_accuracy: 0.7694
 4576/60000 [=>............................] - ETA: 1:43 - loss: 0.7180 - categorical_accuracy: 0.7705
 4608/60000 [=>............................] - ETA: 1:43 - loss: 0.7161 - categorical_accuracy: 0.7708
 4640/60000 [=>............................] - ETA: 1:43 - loss: 0.7132 - categorical_accuracy: 0.7720
 4672/60000 [=>............................] - ETA: 1:43 - loss: 0.7108 - categorical_accuracy: 0.7729
 4704/60000 [=>............................] - ETA: 1:43 - loss: 0.7114 - categorical_accuracy: 0.7727
 4736/60000 [=>............................] - ETA: 1:43 - loss: 0.7087 - categorical_accuracy: 0.7734
 4768/60000 [=>............................] - ETA: 1:43 - loss: 0.7060 - categorical_accuracy: 0.7743
 4800/60000 [=>............................] - ETA: 1:43 - loss: 0.7035 - categorical_accuracy: 0.7750
 4832/60000 [=>............................] - ETA: 1:43 - loss: 0.6999 - categorical_accuracy: 0.7763
 4864/60000 [=>............................] - ETA: 1:43 - loss: 0.6972 - categorical_accuracy: 0.7775
 4896/60000 [=>............................] - ETA: 1:43 - loss: 0.6942 - categorical_accuracy: 0.7784
 4928/60000 [=>............................] - ETA: 1:43 - loss: 0.6903 - categorical_accuracy: 0.7798
 4960/60000 [=>............................] - ETA: 1:43 - loss: 0.6886 - categorical_accuracy: 0.7798
 4992/60000 [=>............................] - ETA: 1:43 - loss: 0.6857 - categorical_accuracy: 0.7808
 5024/60000 [=>............................] - ETA: 1:42 - loss: 0.6869 - categorical_accuracy: 0.7814
 5056/60000 [=>............................] - ETA: 1:42 - loss: 0.6857 - categorical_accuracy: 0.7820
 5088/60000 [=>............................] - ETA: 1:42 - loss: 0.6836 - categorical_accuracy: 0.7826
 5120/60000 [=>............................] - ETA: 1:42 - loss: 0.6810 - categorical_accuracy: 0.7836
 5152/60000 [=>............................] - ETA: 1:42 - loss: 0.6795 - categorical_accuracy: 0.7842
 5184/60000 [=>............................] - ETA: 1:42 - loss: 0.6773 - categorical_accuracy: 0.7847
 5216/60000 [=>............................] - ETA: 1:42 - loss: 0.6754 - categorical_accuracy: 0.7857
 5248/60000 [=>............................] - ETA: 1:42 - loss: 0.6743 - categorical_accuracy: 0.7856
 5280/60000 [=>............................] - ETA: 1:42 - loss: 0.6732 - categorical_accuracy: 0.7860
 5312/60000 [=>............................] - ETA: 1:42 - loss: 0.6716 - categorical_accuracy: 0.7865
 5344/60000 [=>............................] - ETA: 1:42 - loss: 0.6686 - categorical_accuracy: 0.7876
 5376/60000 [=>............................] - ETA: 1:42 - loss: 0.6658 - categorical_accuracy: 0.7885
 5408/60000 [=>............................] - ETA: 1:42 - loss: 0.6636 - categorical_accuracy: 0.7896
 5440/60000 [=>............................] - ETA: 1:41 - loss: 0.6622 - categorical_accuracy: 0.7904
 5472/60000 [=>............................] - ETA: 1:41 - loss: 0.6589 - categorical_accuracy: 0.7915
 5504/60000 [=>............................] - ETA: 1:41 - loss: 0.6561 - categorical_accuracy: 0.7923
 5536/60000 [=>............................] - ETA: 1:41 - loss: 0.6559 - categorical_accuracy: 0.7928
 5568/60000 [=>............................] - ETA: 1:41 - loss: 0.6535 - categorical_accuracy: 0.7936
 5600/60000 [=>............................] - ETA: 1:41 - loss: 0.6514 - categorical_accuracy: 0.7941
 5632/60000 [=>............................] - ETA: 1:41 - loss: 0.6488 - categorical_accuracy: 0.7949
 5664/60000 [=>............................] - ETA: 1:41 - loss: 0.6461 - categorical_accuracy: 0.7957
 5696/60000 [=>............................] - ETA: 1:41 - loss: 0.6432 - categorical_accuracy: 0.7969
 5728/60000 [=>............................] - ETA: 1:41 - loss: 0.6421 - categorical_accuracy: 0.7975
 5760/60000 [=>............................] - ETA: 1:41 - loss: 0.6402 - categorical_accuracy: 0.7981
 5792/60000 [=>............................] - ETA: 1:41 - loss: 0.6407 - categorical_accuracy: 0.7982
 5824/60000 [=>............................] - ETA: 1:40 - loss: 0.6382 - categorical_accuracy: 0.7989
 5856/60000 [=>............................] - ETA: 1:40 - loss: 0.6358 - categorical_accuracy: 0.7997
 5888/60000 [=>............................] - ETA: 1:40 - loss: 0.6336 - categorical_accuracy: 0.8006
 5920/60000 [=>............................] - ETA: 1:40 - loss: 0.6307 - categorical_accuracy: 0.8015
 5952/60000 [=>............................] - ETA: 1:40 - loss: 0.6277 - categorical_accuracy: 0.8026
 5984/60000 [=>............................] - ETA: 1:40 - loss: 0.6261 - categorical_accuracy: 0.8033
 6016/60000 [==>...........................] - ETA: 1:40 - loss: 0.6241 - categorical_accuracy: 0.8039
 6048/60000 [==>...........................] - ETA: 1:40 - loss: 0.6232 - categorical_accuracy: 0.8042
 6080/60000 [==>...........................] - ETA: 1:40 - loss: 0.6209 - categorical_accuracy: 0.8051
 6112/60000 [==>...........................] - ETA: 1:40 - loss: 0.6197 - categorical_accuracy: 0.8058
 6144/60000 [==>...........................] - ETA: 1:40 - loss: 0.6178 - categorical_accuracy: 0.8065
 6176/60000 [==>...........................] - ETA: 1:40 - loss: 0.6157 - categorical_accuracy: 0.8070
 6208/60000 [==>...........................] - ETA: 1:40 - loss: 0.6136 - categorical_accuracy: 0.8073
 6240/60000 [==>...........................] - ETA: 1:39 - loss: 0.6110 - categorical_accuracy: 0.8083
 6272/60000 [==>...........................] - ETA: 1:39 - loss: 0.6117 - categorical_accuracy: 0.8082
 6304/60000 [==>...........................] - ETA: 1:39 - loss: 0.6099 - categorical_accuracy: 0.8090
 6336/60000 [==>...........................] - ETA: 1:39 - loss: 0.6072 - categorical_accuracy: 0.8100
 6368/60000 [==>...........................] - ETA: 1:39 - loss: 0.6063 - categorical_accuracy: 0.8103
 6400/60000 [==>...........................] - ETA: 1:39 - loss: 0.6047 - categorical_accuracy: 0.8108
 6432/60000 [==>...........................] - ETA: 1:39 - loss: 0.6032 - categorical_accuracy: 0.8111
 6464/60000 [==>...........................] - ETA: 1:39 - loss: 0.6028 - categorical_accuracy: 0.8110
 6496/60000 [==>...........................] - ETA: 1:39 - loss: 0.6015 - categorical_accuracy: 0.8116
 6528/60000 [==>...........................] - ETA: 1:39 - loss: 0.6002 - categorical_accuracy: 0.8120
 6560/60000 [==>...........................] - ETA: 1:39 - loss: 0.5978 - categorical_accuracy: 0.8128
 6592/60000 [==>...........................] - ETA: 1:39 - loss: 0.5967 - categorical_accuracy: 0.8131
 6624/60000 [==>...........................] - ETA: 1:39 - loss: 0.5960 - categorical_accuracy: 0.8136
 6656/60000 [==>...........................] - ETA: 1:39 - loss: 0.5948 - categorical_accuracy: 0.8140
 6688/60000 [==>...........................] - ETA: 1:39 - loss: 0.5923 - categorical_accuracy: 0.8149
 6720/60000 [==>...........................] - ETA: 1:38 - loss: 0.5915 - categorical_accuracy: 0.8150
 6752/60000 [==>...........................] - ETA: 1:38 - loss: 0.5903 - categorical_accuracy: 0.8155
 6784/60000 [==>...........................] - ETA: 1:38 - loss: 0.5899 - categorical_accuracy: 0.8157
 6816/60000 [==>...........................] - ETA: 1:38 - loss: 0.5884 - categorical_accuracy: 0.8162
 6848/60000 [==>...........................] - ETA: 1:38 - loss: 0.5875 - categorical_accuracy: 0.8166
 6880/60000 [==>...........................] - ETA: 1:38 - loss: 0.5860 - categorical_accuracy: 0.8170
 6912/60000 [==>...........................] - ETA: 1:38 - loss: 0.5839 - categorical_accuracy: 0.8177
 6944/60000 [==>...........................] - ETA: 1:38 - loss: 0.5824 - categorical_accuracy: 0.8183
 6976/60000 [==>...........................] - ETA: 1:38 - loss: 0.5806 - categorical_accuracy: 0.8188
 7008/60000 [==>...........................] - ETA: 1:38 - loss: 0.5795 - categorical_accuracy: 0.8192
 7040/60000 [==>...........................] - ETA: 1:38 - loss: 0.5771 - categorical_accuracy: 0.8200
 7072/60000 [==>...........................] - ETA: 1:38 - loss: 0.5769 - categorical_accuracy: 0.8203
 7104/60000 [==>...........................] - ETA: 1:38 - loss: 0.5754 - categorical_accuracy: 0.8204
 7136/60000 [==>...........................] - ETA: 1:38 - loss: 0.5739 - categorical_accuracy: 0.8208
 7168/60000 [==>...........................] - ETA: 1:37 - loss: 0.5722 - categorical_accuracy: 0.8214
 7200/60000 [==>...........................] - ETA: 1:37 - loss: 0.5704 - categorical_accuracy: 0.8219
 7232/60000 [==>...........................] - ETA: 1:37 - loss: 0.5682 - categorical_accuracy: 0.8226
 7264/60000 [==>...........................] - ETA: 1:37 - loss: 0.5675 - categorical_accuracy: 0.8230
 7296/60000 [==>...........................] - ETA: 1:37 - loss: 0.5662 - categorical_accuracy: 0.8233
 7328/60000 [==>...........................] - ETA: 1:37 - loss: 0.5647 - categorical_accuracy: 0.8238
 7360/60000 [==>...........................] - ETA: 1:37 - loss: 0.5642 - categorical_accuracy: 0.8242
 7392/60000 [==>...........................] - ETA: 1:37 - loss: 0.5621 - categorical_accuracy: 0.8249
 7424/60000 [==>...........................] - ETA: 1:37 - loss: 0.5615 - categorical_accuracy: 0.8253
 7456/60000 [==>...........................] - ETA: 1:37 - loss: 0.5614 - categorical_accuracy: 0.8256
 7488/60000 [==>...........................] - ETA: 1:37 - loss: 0.5598 - categorical_accuracy: 0.8260
 7520/60000 [==>...........................] - ETA: 1:37 - loss: 0.5577 - categorical_accuracy: 0.8267
 7552/60000 [==>...........................] - ETA: 1:37 - loss: 0.5560 - categorical_accuracy: 0.8273
 7584/60000 [==>...........................] - ETA: 1:37 - loss: 0.5544 - categorical_accuracy: 0.8278
 7616/60000 [==>...........................] - ETA: 1:37 - loss: 0.5536 - categorical_accuracy: 0.8279
 7648/60000 [==>...........................] - ETA: 1:36 - loss: 0.5523 - categorical_accuracy: 0.8282
 7680/60000 [==>...........................] - ETA: 1:36 - loss: 0.5515 - categorical_accuracy: 0.8284
 7712/60000 [==>...........................] - ETA: 1:36 - loss: 0.5506 - categorical_accuracy: 0.8288
 7744/60000 [==>...........................] - ETA: 1:36 - loss: 0.5494 - categorical_accuracy: 0.8294
 7776/60000 [==>...........................] - ETA: 1:36 - loss: 0.5490 - categorical_accuracy: 0.8296
 7808/60000 [==>...........................] - ETA: 1:36 - loss: 0.5489 - categorical_accuracy: 0.8298
 7840/60000 [==>...........................] - ETA: 1:36 - loss: 0.5481 - categorical_accuracy: 0.8302
 7872/60000 [==>...........................] - ETA: 1:36 - loss: 0.5472 - categorical_accuracy: 0.8302
 7904/60000 [==>...........................] - ETA: 1:36 - loss: 0.5464 - categorical_accuracy: 0.8303
 7936/60000 [==>...........................] - ETA: 1:36 - loss: 0.5454 - categorical_accuracy: 0.8305
 7968/60000 [==>...........................] - ETA: 1:36 - loss: 0.5447 - categorical_accuracy: 0.8308
 8000/60000 [===>..........................] - ETA: 1:36 - loss: 0.5430 - categorical_accuracy: 0.8315
 8032/60000 [===>..........................] - ETA: 1:36 - loss: 0.5415 - categorical_accuracy: 0.8320
 8064/60000 [===>..........................] - ETA: 1:36 - loss: 0.5411 - categorical_accuracy: 0.8323
 8096/60000 [===>..........................] - ETA: 1:36 - loss: 0.5399 - categorical_accuracy: 0.8325
 8128/60000 [===>..........................] - ETA: 1:36 - loss: 0.5386 - categorical_accuracy: 0.8328
 8160/60000 [===>..........................] - ETA: 1:35 - loss: 0.5373 - categorical_accuracy: 0.8331
 8192/60000 [===>..........................] - ETA: 1:35 - loss: 0.5357 - categorical_accuracy: 0.8337
 8224/60000 [===>..........................] - ETA: 1:35 - loss: 0.5338 - categorical_accuracy: 0.8344
 8256/60000 [===>..........................] - ETA: 1:35 - loss: 0.5328 - categorical_accuracy: 0.8348
 8288/60000 [===>..........................] - ETA: 1:35 - loss: 0.5319 - categorical_accuracy: 0.8351
 8320/60000 [===>..........................] - ETA: 1:35 - loss: 0.5309 - categorical_accuracy: 0.8353
 8352/60000 [===>..........................] - ETA: 1:35 - loss: 0.5299 - categorical_accuracy: 0.8356
 8384/60000 [===>..........................] - ETA: 1:35 - loss: 0.5292 - categorical_accuracy: 0.8356
 8416/60000 [===>..........................] - ETA: 1:35 - loss: 0.5292 - categorical_accuracy: 0.8357
 8448/60000 [===>..........................] - ETA: 1:35 - loss: 0.5278 - categorical_accuracy: 0.8362
 8480/60000 [===>..........................] - ETA: 1:35 - loss: 0.5268 - categorical_accuracy: 0.8364
 8512/60000 [===>..........................] - ETA: 1:35 - loss: 0.5268 - categorical_accuracy: 0.8366
 8544/60000 [===>..........................] - ETA: 1:35 - loss: 0.5274 - categorical_accuracy: 0.8365
 8576/60000 [===>..........................] - ETA: 1:35 - loss: 0.5261 - categorical_accuracy: 0.8370
 8608/60000 [===>..........................] - ETA: 1:35 - loss: 0.5252 - categorical_accuracy: 0.8374
 8640/60000 [===>..........................] - ETA: 1:35 - loss: 0.5247 - categorical_accuracy: 0.8375
 8672/60000 [===>..........................] - ETA: 1:34 - loss: 0.5239 - categorical_accuracy: 0.8376
 8704/60000 [===>..........................] - ETA: 1:34 - loss: 0.5233 - categorical_accuracy: 0.8379
 8736/60000 [===>..........................] - ETA: 1:34 - loss: 0.5219 - categorical_accuracy: 0.8384
 8768/60000 [===>..........................] - ETA: 1:34 - loss: 0.5204 - categorical_accuracy: 0.8388
 8800/60000 [===>..........................] - ETA: 1:34 - loss: 0.5199 - categorical_accuracy: 0.8391
 8832/60000 [===>..........................] - ETA: 1:34 - loss: 0.5189 - categorical_accuracy: 0.8393
 8864/60000 [===>..........................] - ETA: 1:34 - loss: 0.5179 - categorical_accuracy: 0.8396
 8896/60000 [===>..........................] - ETA: 1:34 - loss: 0.5170 - categorical_accuracy: 0.8400
 8928/60000 [===>..........................] - ETA: 1:34 - loss: 0.5170 - categorical_accuracy: 0.8401
 8960/60000 [===>..........................] - ETA: 1:34 - loss: 0.5156 - categorical_accuracy: 0.8405
 8992/60000 [===>..........................] - ETA: 1:34 - loss: 0.5140 - categorical_accuracy: 0.8410
 9024/60000 [===>..........................] - ETA: 1:34 - loss: 0.5124 - categorical_accuracy: 0.8415
 9056/60000 [===>..........................] - ETA: 1:34 - loss: 0.5124 - categorical_accuracy: 0.8417
 9088/60000 [===>..........................] - ETA: 1:34 - loss: 0.5120 - categorical_accuracy: 0.8418
 9120/60000 [===>..........................] - ETA: 1:34 - loss: 0.5113 - categorical_accuracy: 0.8421
 9152/60000 [===>..........................] - ETA: 1:34 - loss: 0.5100 - categorical_accuracy: 0.8423
 9184/60000 [===>..........................] - ETA: 1:33 - loss: 0.5086 - categorical_accuracy: 0.8428
 9216/60000 [===>..........................] - ETA: 1:33 - loss: 0.5075 - categorical_accuracy: 0.8432
 9248/60000 [===>..........................] - ETA: 1:33 - loss: 0.5069 - categorical_accuracy: 0.8434
 9280/60000 [===>..........................] - ETA: 1:33 - loss: 0.5061 - categorical_accuracy: 0.8436
 9312/60000 [===>..........................] - ETA: 1:33 - loss: 0.5051 - categorical_accuracy: 0.8440
 9344/60000 [===>..........................] - ETA: 1:33 - loss: 0.5042 - categorical_accuracy: 0.8441
 9376/60000 [===>..........................] - ETA: 1:33 - loss: 0.5039 - categorical_accuracy: 0.8442
 9408/60000 [===>..........................] - ETA: 1:33 - loss: 0.5028 - categorical_accuracy: 0.8445
 9440/60000 [===>..........................] - ETA: 1:33 - loss: 0.5012 - categorical_accuracy: 0.8450
 9472/60000 [===>..........................] - ETA: 1:33 - loss: 0.4999 - categorical_accuracy: 0.8453
 9504/60000 [===>..........................] - ETA: 1:33 - loss: 0.4994 - categorical_accuracy: 0.8455
 9536/60000 [===>..........................] - ETA: 1:33 - loss: 0.4982 - categorical_accuracy: 0.8460
 9568/60000 [===>..........................] - ETA: 1:33 - loss: 0.4969 - categorical_accuracy: 0.8464
 9600/60000 [===>..........................] - ETA: 1:33 - loss: 0.4957 - categorical_accuracy: 0.8468
 9632/60000 [===>..........................] - ETA: 1:32 - loss: 0.4942 - categorical_accuracy: 0.8473
 9664/60000 [===>..........................] - ETA: 1:32 - loss: 0.4940 - categorical_accuracy: 0.8474
 9696/60000 [===>..........................] - ETA: 1:32 - loss: 0.4926 - categorical_accuracy: 0.8478
 9728/60000 [===>..........................] - ETA: 1:32 - loss: 0.4911 - categorical_accuracy: 0.8483
 9760/60000 [===>..........................] - ETA: 1:32 - loss: 0.4912 - categorical_accuracy: 0.8483
 9792/60000 [===>..........................] - ETA: 1:32 - loss: 0.4904 - categorical_accuracy: 0.8485
 9824/60000 [===>..........................] - ETA: 1:32 - loss: 0.4897 - categorical_accuracy: 0.8488
 9856/60000 [===>..........................] - ETA: 1:32 - loss: 0.4888 - categorical_accuracy: 0.8491
 9888/60000 [===>..........................] - ETA: 1:32 - loss: 0.4876 - categorical_accuracy: 0.8495
 9920/60000 [===>..........................] - ETA: 1:32 - loss: 0.4863 - categorical_accuracy: 0.8500
 9952/60000 [===>..........................] - ETA: 1:32 - loss: 0.4853 - categorical_accuracy: 0.8503
 9984/60000 [===>..........................] - ETA: 1:32 - loss: 0.4846 - categorical_accuracy: 0.8506
10016/60000 [====>.........................] - ETA: 1:32 - loss: 0.4837 - categorical_accuracy: 0.8508
10048/60000 [====>.........................] - ETA: 1:32 - loss: 0.4824 - categorical_accuracy: 0.8512
10080/60000 [====>.........................] - ETA: 1:32 - loss: 0.4815 - categorical_accuracy: 0.8515
10112/60000 [====>.........................] - ETA: 1:32 - loss: 0.4809 - categorical_accuracy: 0.8514
10144/60000 [====>.........................] - ETA: 1:32 - loss: 0.4803 - categorical_accuracy: 0.8515
10176/60000 [====>.........................] - ETA: 1:31 - loss: 0.4792 - categorical_accuracy: 0.8518
10208/60000 [====>.........................] - ETA: 1:31 - loss: 0.4781 - categorical_accuracy: 0.8522
10240/60000 [====>.........................] - ETA: 1:31 - loss: 0.4768 - categorical_accuracy: 0.8526
10272/60000 [====>.........................] - ETA: 1:31 - loss: 0.4754 - categorical_accuracy: 0.8531
10304/60000 [====>.........................] - ETA: 1:31 - loss: 0.4744 - categorical_accuracy: 0.8535
10336/60000 [====>.........................] - ETA: 1:31 - loss: 0.4730 - categorical_accuracy: 0.8539
10368/60000 [====>.........................] - ETA: 1:31 - loss: 0.4721 - categorical_accuracy: 0.8542
10400/60000 [====>.........................] - ETA: 1:31 - loss: 0.4707 - categorical_accuracy: 0.8546
10432/60000 [====>.........................] - ETA: 1:31 - loss: 0.4697 - categorical_accuracy: 0.8550
10464/60000 [====>.........................] - ETA: 1:31 - loss: 0.4686 - categorical_accuracy: 0.8552
10496/60000 [====>.........................] - ETA: 1:31 - loss: 0.4678 - categorical_accuracy: 0.8555
10528/60000 [====>.........................] - ETA: 1:31 - loss: 0.4671 - categorical_accuracy: 0.8557
10560/60000 [====>.........................] - ETA: 1:31 - loss: 0.4661 - categorical_accuracy: 0.8561
10592/60000 [====>.........................] - ETA: 1:31 - loss: 0.4649 - categorical_accuracy: 0.8564
10624/60000 [====>.........................] - ETA: 1:31 - loss: 0.4642 - categorical_accuracy: 0.8566
10656/60000 [====>.........................] - ETA: 1:31 - loss: 0.4633 - categorical_accuracy: 0.8568
10688/60000 [====>.........................] - ETA: 1:31 - loss: 0.4623 - categorical_accuracy: 0.8571
10720/60000 [====>.........................] - ETA: 1:30 - loss: 0.4621 - categorical_accuracy: 0.8573
10752/60000 [====>.........................] - ETA: 1:30 - loss: 0.4618 - categorical_accuracy: 0.8574
10784/60000 [====>.........................] - ETA: 1:30 - loss: 0.4606 - categorical_accuracy: 0.8578
10816/60000 [====>.........................] - ETA: 1:30 - loss: 0.4594 - categorical_accuracy: 0.8582
10848/60000 [====>.........................] - ETA: 1:30 - loss: 0.4585 - categorical_accuracy: 0.8584
10880/60000 [====>.........................] - ETA: 1:30 - loss: 0.4577 - categorical_accuracy: 0.8586
10912/60000 [====>.........................] - ETA: 1:30 - loss: 0.4568 - categorical_accuracy: 0.8590
10944/60000 [====>.........................] - ETA: 1:30 - loss: 0.4560 - categorical_accuracy: 0.8592
10976/60000 [====>.........................] - ETA: 1:30 - loss: 0.4548 - categorical_accuracy: 0.8595
11008/60000 [====>.........................] - ETA: 1:30 - loss: 0.4554 - categorical_accuracy: 0.8595
11040/60000 [====>.........................] - ETA: 1:30 - loss: 0.4546 - categorical_accuracy: 0.8597
11072/60000 [====>.........................] - ETA: 1:30 - loss: 0.4537 - categorical_accuracy: 0.8600
11104/60000 [====>.........................] - ETA: 1:30 - loss: 0.4533 - categorical_accuracy: 0.8602
11136/60000 [====>.........................] - ETA: 1:30 - loss: 0.4523 - categorical_accuracy: 0.8605
11168/60000 [====>.........................] - ETA: 1:30 - loss: 0.4515 - categorical_accuracy: 0.8609
11200/60000 [====>.........................] - ETA: 1:29 - loss: 0.4512 - categorical_accuracy: 0.8609
11232/60000 [====>.........................] - ETA: 1:29 - loss: 0.4502 - categorical_accuracy: 0.8612
11264/60000 [====>.........................] - ETA: 1:29 - loss: 0.4493 - categorical_accuracy: 0.8615
11296/60000 [====>.........................] - ETA: 1:29 - loss: 0.4491 - categorical_accuracy: 0.8618
11328/60000 [====>.........................] - ETA: 1:29 - loss: 0.4479 - categorical_accuracy: 0.8622
11360/60000 [====>.........................] - ETA: 1:29 - loss: 0.4470 - categorical_accuracy: 0.8625
11392/60000 [====>.........................] - ETA: 1:29 - loss: 0.4464 - categorical_accuracy: 0.8627
11424/60000 [====>.........................] - ETA: 1:29 - loss: 0.4455 - categorical_accuracy: 0.8630
11456/60000 [====>.........................] - ETA: 1:29 - loss: 0.4450 - categorical_accuracy: 0.8631
11488/60000 [====>.........................] - ETA: 1:29 - loss: 0.4441 - categorical_accuracy: 0.8633
11520/60000 [====>.........................] - ETA: 1:29 - loss: 0.4430 - categorical_accuracy: 0.8636
11552/60000 [====>.........................] - ETA: 1:29 - loss: 0.4421 - categorical_accuracy: 0.8640
11584/60000 [====>.........................] - ETA: 1:29 - loss: 0.4418 - categorical_accuracy: 0.8642
11616/60000 [====>.........................] - ETA: 1:29 - loss: 0.4409 - categorical_accuracy: 0.8644
11648/60000 [====>.........................] - ETA: 1:29 - loss: 0.4399 - categorical_accuracy: 0.8647
11680/60000 [====>.........................] - ETA: 1:29 - loss: 0.4389 - categorical_accuracy: 0.8651
11712/60000 [====>.........................] - ETA: 1:28 - loss: 0.4382 - categorical_accuracy: 0.8653
11744/60000 [====>.........................] - ETA: 1:28 - loss: 0.4374 - categorical_accuracy: 0.8654
11776/60000 [====>.........................] - ETA: 1:28 - loss: 0.4368 - categorical_accuracy: 0.8656
11808/60000 [====>.........................] - ETA: 1:28 - loss: 0.4360 - categorical_accuracy: 0.8659
11840/60000 [====>.........................] - ETA: 1:28 - loss: 0.4351 - categorical_accuracy: 0.8660
11872/60000 [====>.........................] - ETA: 1:28 - loss: 0.4342 - categorical_accuracy: 0.8663
11904/60000 [====>.........................] - ETA: 1:28 - loss: 0.4336 - categorical_accuracy: 0.8664
11936/60000 [====>.........................] - ETA: 1:28 - loss: 0.4328 - categorical_accuracy: 0.8666
11968/60000 [====>.........................] - ETA: 1:28 - loss: 0.4329 - categorical_accuracy: 0.8666
12000/60000 [=====>........................] - ETA: 1:28 - loss: 0.4322 - categorical_accuracy: 0.8668
12032/60000 [=====>........................] - ETA: 1:28 - loss: 0.4318 - categorical_accuracy: 0.8670
12064/60000 [=====>........................] - ETA: 1:28 - loss: 0.4310 - categorical_accuracy: 0.8673
12096/60000 [=====>........................] - ETA: 1:28 - loss: 0.4303 - categorical_accuracy: 0.8675
12128/60000 [=====>........................] - ETA: 1:28 - loss: 0.4295 - categorical_accuracy: 0.8677
12160/60000 [=====>........................] - ETA: 1:28 - loss: 0.4288 - categorical_accuracy: 0.8679
12192/60000 [=====>........................] - ETA: 1:28 - loss: 0.4279 - categorical_accuracy: 0.8682
12224/60000 [=====>........................] - ETA: 1:27 - loss: 0.4272 - categorical_accuracy: 0.8685
12256/60000 [=====>........................] - ETA: 1:27 - loss: 0.4265 - categorical_accuracy: 0.8686
12288/60000 [=====>........................] - ETA: 1:27 - loss: 0.4259 - categorical_accuracy: 0.8687
12320/60000 [=====>........................] - ETA: 1:27 - loss: 0.4251 - categorical_accuracy: 0.8689
12352/60000 [=====>........................] - ETA: 1:27 - loss: 0.4241 - categorical_accuracy: 0.8693
12384/60000 [=====>........................] - ETA: 1:27 - loss: 0.4233 - categorical_accuracy: 0.8695
12416/60000 [=====>........................] - ETA: 1:27 - loss: 0.4224 - categorical_accuracy: 0.8698
12448/60000 [=====>........................] - ETA: 1:27 - loss: 0.4214 - categorical_accuracy: 0.8701
12480/60000 [=====>........................] - ETA: 1:27 - loss: 0.4206 - categorical_accuracy: 0.8704
12512/60000 [=====>........................] - ETA: 1:27 - loss: 0.4196 - categorical_accuracy: 0.8707
12544/60000 [=====>........................] - ETA: 1:27 - loss: 0.4194 - categorical_accuracy: 0.8709
12576/60000 [=====>........................] - ETA: 1:27 - loss: 0.4184 - categorical_accuracy: 0.8713
12608/60000 [=====>........................] - ETA: 1:27 - loss: 0.4175 - categorical_accuracy: 0.8715
12640/60000 [=====>........................] - ETA: 1:27 - loss: 0.4167 - categorical_accuracy: 0.8718
12672/60000 [=====>........................] - ETA: 1:27 - loss: 0.4158 - categorical_accuracy: 0.8720
12704/60000 [=====>........................] - ETA: 1:27 - loss: 0.4152 - categorical_accuracy: 0.8722
12736/60000 [=====>........................] - ETA: 1:26 - loss: 0.4150 - categorical_accuracy: 0.8724
12768/60000 [=====>........................] - ETA: 1:26 - loss: 0.4150 - categorical_accuracy: 0.8724
12800/60000 [=====>........................] - ETA: 1:26 - loss: 0.4142 - categorical_accuracy: 0.8725
12832/60000 [=====>........................] - ETA: 1:26 - loss: 0.4133 - categorical_accuracy: 0.8728
12864/60000 [=====>........................] - ETA: 1:26 - loss: 0.4123 - categorical_accuracy: 0.8731
12896/60000 [=====>........................] - ETA: 1:26 - loss: 0.4116 - categorical_accuracy: 0.8734
12928/60000 [=====>........................] - ETA: 1:26 - loss: 0.4112 - categorical_accuracy: 0.8736
12960/60000 [=====>........................] - ETA: 1:26 - loss: 0.4107 - categorical_accuracy: 0.8738
12992/60000 [=====>........................] - ETA: 1:26 - loss: 0.4098 - categorical_accuracy: 0.8741
13024/60000 [=====>........................] - ETA: 1:26 - loss: 0.4106 - categorical_accuracy: 0.8741
13056/60000 [=====>........................] - ETA: 1:26 - loss: 0.4100 - categorical_accuracy: 0.8743
13088/60000 [=====>........................] - ETA: 1:26 - loss: 0.4099 - categorical_accuracy: 0.8744
13120/60000 [=====>........................] - ETA: 1:26 - loss: 0.4092 - categorical_accuracy: 0.8746
13152/60000 [=====>........................] - ETA: 1:26 - loss: 0.4085 - categorical_accuracy: 0.8748
13184/60000 [=====>........................] - ETA: 1:26 - loss: 0.4083 - categorical_accuracy: 0.8748
13216/60000 [=====>........................] - ETA: 1:26 - loss: 0.4076 - categorical_accuracy: 0.8751
13248/60000 [=====>........................] - ETA: 1:26 - loss: 0.4072 - categorical_accuracy: 0.8752
13280/60000 [=====>........................] - ETA: 1:26 - loss: 0.4066 - categorical_accuracy: 0.8752
13312/60000 [=====>........................] - ETA: 1:25 - loss: 0.4058 - categorical_accuracy: 0.8754
13344/60000 [=====>........................] - ETA: 1:25 - loss: 0.4058 - categorical_accuracy: 0.8752
13376/60000 [=====>........................] - ETA: 1:25 - loss: 0.4049 - categorical_accuracy: 0.8755
13408/60000 [=====>........................] - ETA: 1:25 - loss: 0.4040 - categorical_accuracy: 0.8758
13440/60000 [=====>........................] - ETA: 1:25 - loss: 0.4032 - categorical_accuracy: 0.8760
13472/60000 [=====>........................] - ETA: 1:25 - loss: 0.4027 - categorical_accuracy: 0.8762
13504/60000 [=====>........................] - ETA: 1:25 - loss: 0.4020 - categorical_accuracy: 0.8764
13536/60000 [=====>........................] - ETA: 1:25 - loss: 0.4011 - categorical_accuracy: 0.8766
13568/60000 [=====>........................] - ETA: 1:25 - loss: 0.4009 - categorical_accuracy: 0.8768
13600/60000 [=====>........................] - ETA: 1:25 - loss: 0.4004 - categorical_accuracy: 0.8769
13632/60000 [=====>........................] - ETA: 1:25 - loss: 0.4006 - categorical_accuracy: 0.8770
13664/60000 [=====>........................] - ETA: 1:25 - loss: 0.3998 - categorical_accuracy: 0.8773
13696/60000 [=====>........................] - ETA: 1:25 - loss: 0.3992 - categorical_accuracy: 0.8775
13728/60000 [=====>........................] - ETA: 1:25 - loss: 0.3989 - categorical_accuracy: 0.8777
13760/60000 [=====>........................] - ETA: 1:25 - loss: 0.3982 - categorical_accuracy: 0.8779
13792/60000 [=====>........................] - ETA: 1:25 - loss: 0.3978 - categorical_accuracy: 0.8779
13824/60000 [=====>........................] - ETA: 1:25 - loss: 0.3979 - categorical_accuracy: 0.8779
13856/60000 [=====>........................] - ETA: 1:24 - loss: 0.3971 - categorical_accuracy: 0.8782
13888/60000 [=====>........................] - ETA: 1:24 - loss: 0.3965 - categorical_accuracy: 0.8784
13920/60000 [=====>........................] - ETA: 1:24 - loss: 0.3961 - categorical_accuracy: 0.8785
13952/60000 [=====>........................] - ETA: 1:24 - loss: 0.3953 - categorical_accuracy: 0.8788
13984/60000 [=====>........................] - ETA: 1:24 - loss: 0.3946 - categorical_accuracy: 0.8790
14016/60000 [======>.......................] - ETA: 1:24 - loss: 0.3942 - categorical_accuracy: 0.8791
14048/60000 [======>.......................] - ETA: 1:24 - loss: 0.3939 - categorical_accuracy: 0.8791
14080/60000 [======>.......................] - ETA: 1:24 - loss: 0.3933 - categorical_accuracy: 0.8793
14112/60000 [======>.......................] - ETA: 1:24 - loss: 0.3925 - categorical_accuracy: 0.8796
14144/60000 [======>.......................] - ETA: 1:24 - loss: 0.3921 - categorical_accuracy: 0.8797
14176/60000 [======>.......................] - ETA: 1:24 - loss: 0.3916 - categorical_accuracy: 0.8798
14208/60000 [======>.......................] - ETA: 1:24 - loss: 0.3908 - categorical_accuracy: 0.8801
14240/60000 [======>.......................] - ETA: 1:24 - loss: 0.3901 - categorical_accuracy: 0.8803
14272/60000 [======>.......................] - ETA: 1:24 - loss: 0.3896 - categorical_accuracy: 0.8804
14304/60000 [======>.......................] - ETA: 1:24 - loss: 0.3889 - categorical_accuracy: 0.8806
14336/60000 [======>.......................] - ETA: 1:24 - loss: 0.3883 - categorical_accuracy: 0.8809
14368/60000 [======>.......................] - ETA: 1:23 - loss: 0.3885 - categorical_accuracy: 0.8808
14400/60000 [======>.......................] - ETA: 1:23 - loss: 0.3879 - categorical_accuracy: 0.8809
14432/60000 [======>.......................] - ETA: 1:23 - loss: 0.3874 - categorical_accuracy: 0.8810
14464/60000 [======>.......................] - ETA: 1:23 - loss: 0.3867 - categorical_accuracy: 0.8813
14496/60000 [======>.......................] - ETA: 1:23 - loss: 0.3859 - categorical_accuracy: 0.8816
14528/60000 [======>.......................] - ETA: 1:23 - loss: 0.3853 - categorical_accuracy: 0.8817
14560/60000 [======>.......................] - ETA: 1:23 - loss: 0.3849 - categorical_accuracy: 0.8819
14592/60000 [======>.......................] - ETA: 1:23 - loss: 0.3846 - categorical_accuracy: 0.8820
14624/60000 [======>.......................] - ETA: 1:23 - loss: 0.3841 - categorical_accuracy: 0.8822
14656/60000 [======>.......................] - ETA: 1:23 - loss: 0.3833 - categorical_accuracy: 0.8824
14688/60000 [======>.......................] - ETA: 1:23 - loss: 0.3830 - categorical_accuracy: 0.8825
14720/60000 [======>.......................] - ETA: 1:23 - loss: 0.3837 - categorical_accuracy: 0.8823
14752/60000 [======>.......................] - ETA: 1:23 - loss: 0.3840 - categorical_accuracy: 0.8824
14784/60000 [======>.......................] - ETA: 1:23 - loss: 0.3835 - categorical_accuracy: 0.8825
14816/60000 [======>.......................] - ETA: 1:23 - loss: 0.3830 - categorical_accuracy: 0.8826
14848/60000 [======>.......................] - ETA: 1:23 - loss: 0.3829 - categorical_accuracy: 0.8826
14880/60000 [======>.......................] - ETA: 1:23 - loss: 0.3823 - categorical_accuracy: 0.8828
14912/60000 [======>.......................] - ETA: 1:22 - loss: 0.3817 - categorical_accuracy: 0.8830
14944/60000 [======>.......................] - ETA: 1:22 - loss: 0.3815 - categorical_accuracy: 0.8830
14976/60000 [======>.......................] - ETA: 1:22 - loss: 0.3811 - categorical_accuracy: 0.8831
15008/60000 [======>.......................] - ETA: 1:22 - loss: 0.3810 - categorical_accuracy: 0.8832
15040/60000 [======>.......................] - ETA: 1:22 - loss: 0.3804 - categorical_accuracy: 0.8833
15072/60000 [======>.......................] - ETA: 1:22 - loss: 0.3800 - categorical_accuracy: 0.8835
15104/60000 [======>.......................] - ETA: 1:22 - loss: 0.3794 - categorical_accuracy: 0.8837
15136/60000 [======>.......................] - ETA: 1:22 - loss: 0.3788 - categorical_accuracy: 0.8839
15168/60000 [======>.......................] - ETA: 1:22 - loss: 0.3783 - categorical_accuracy: 0.8840
15200/60000 [======>.......................] - ETA: 1:22 - loss: 0.3777 - categorical_accuracy: 0.8842
15232/60000 [======>.......................] - ETA: 1:22 - loss: 0.3773 - categorical_accuracy: 0.8843
15264/60000 [======>.......................] - ETA: 1:22 - loss: 0.3768 - categorical_accuracy: 0.8844
15296/60000 [======>.......................] - ETA: 1:22 - loss: 0.3761 - categorical_accuracy: 0.8847
15328/60000 [======>.......................] - ETA: 1:22 - loss: 0.3754 - categorical_accuracy: 0.8849
15360/60000 [======>.......................] - ETA: 1:22 - loss: 0.3748 - categorical_accuracy: 0.8850
15392/60000 [======>.......................] - ETA: 1:22 - loss: 0.3742 - categorical_accuracy: 0.8851
15424/60000 [======>.......................] - ETA: 1:22 - loss: 0.3735 - categorical_accuracy: 0.8853
15456/60000 [======>.......................] - ETA: 1:22 - loss: 0.3732 - categorical_accuracy: 0.8853
15488/60000 [======>.......................] - ETA: 1:21 - loss: 0.3726 - categorical_accuracy: 0.8855
15520/60000 [======>.......................] - ETA: 1:21 - loss: 0.3720 - categorical_accuracy: 0.8857
15552/60000 [======>.......................] - ETA: 1:21 - loss: 0.3717 - categorical_accuracy: 0.8857
15584/60000 [======>.......................] - ETA: 1:21 - loss: 0.3712 - categorical_accuracy: 0.8858
15616/60000 [======>.......................] - ETA: 1:21 - loss: 0.3709 - categorical_accuracy: 0.8860
15648/60000 [======>.......................] - ETA: 1:21 - loss: 0.3703 - categorical_accuracy: 0.8862
15680/60000 [======>.......................] - ETA: 1:21 - loss: 0.3696 - categorical_accuracy: 0.8864
15712/60000 [======>.......................] - ETA: 1:21 - loss: 0.3691 - categorical_accuracy: 0.8866
15744/60000 [======>.......................] - ETA: 1:21 - loss: 0.3685 - categorical_accuracy: 0.8868
15776/60000 [======>.......................] - ETA: 1:21 - loss: 0.3683 - categorical_accuracy: 0.8869
15808/60000 [======>.......................] - ETA: 1:21 - loss: 0.3679 - categorical_accuracy: 0.8871
15840/60000 [======>.......................] - ETA: 1:21 - loss: 0.3675 - categorical_accuracy: 0.8872
15872/60000 [======>.......................] - ETA: 1:21 - loss: 0.3670 - categorical_accuracy: 0.8873
15904/60000 [======>.......................] - ETA: 1:21 - loss: 0.3668 - categorical_accuracy: 0.8874
15936/60000 [======>.......................] - ETA: 1:21 - loss: 0.3662 - categorical_accuracy: 0.8875
15968/60000 [======>.......................] - ETA: 1:21 - loss: 0.3658 - categorical_accuracy: 0.8877
16000/60000 [=======>......................] - ETA: 1:20 - loss: 0.3652 - categorical_accuracy: 0.8878
16032/60000 [=======>......................] - ETA: 1:20 - loss: 0.3646 - categorical_accuracy: 0.8880
16064/60000 [=======>......................] - ETA: 1:20 - loss: 0.3641 - categorical_accuracy: 0.8881
16096/60000 [=======>......................] - ETA: 1:20 - loss: 0.3639 - categorical_accuracy: 0.8880
16128/60000 [=======>......................] - ETA: 1:20 - loss: 0.3634 - categorical_accuracy: 0.8882
16160/60000 [=======>......................] - ETA: 1:20 - loss: 0.3629 - categorical_accuracy: 0.8884
16192/60000 [=======>......................] - ETA: 1:20 - loss: 0.3627 - categorical_accuracy: 0.8885
16224/60000 [=======>......................] - ETA: 1:20 - loss: 0.3620 - categorical_accuracy: 0.8887
16256/60000 [=======>......................] - ETA: 1:20 - loss: 0.3622 - categorical_accuracy: 0.8887
16288/60000 [=======>......................] - ETA: 1:20 - loss: 0.3615 - categorical_accuracy: 0.8889
16320/60000 [=======>......................] - ETA: 1:20 - loss: 0.3609 - categorical_accuracy: 0.8891
16352/60000 [=======>......................] - ETA: 1:20 - loss: 0.3604 - categorical_accuracy: 0.8893
16384/60000 [=======>......................] - ETA: 1:20 - loss: 0.3598 - categorical_accuracy: 0.8895
16416/60000 [=======>......................] - ETA: 1:20 - loss: 0.3596 - categorical_accuracy: 0.8896
16448/60000 [=======>......................] - ETA: 1:20 - loss: 0.3594 - categorical_accuracy: 0.8896
16480/60000 [=======>......................] - ETA: 1:20 - loss: 0.3588 - categorical_accuracy: 0.8898
16512/60000 [=======>......................] - ETA: 1:19 - loss: 0.3583 - categorical_accuracy: 0.8899
16544/60000 [=======>......................] - ETA: 1:19 - loss: 0.3576 - categorical_accuracy: 0.8901
16576/60000 [=======>......................] - ETA: 1:19 - loss: 0.3571 - categorical_accuracy: 0.8903
16608/60000 [=======>......................] - ETA: 1:19 - loss: 0.3568 - categorical_accuracy: 0.8904
16640/60000 [=======>......................] - ETA: 1:19 - loss: 0.3563 - categorical_accuracy: 0.8906
16672/60000 [=======>......................] - ETA: 1:19 - loss: 0.3563 - categorical_accuracy: 0.8907
16704/60000 [=======>......................] - ETA: 1:19 - loss: 0.3558 - categorical_accuracy: 0.8908
16736/60000 [=======>......................] - ETA: 1:19 - loss: 0.3556 - categorical_accuracy: 0.8908
16768/60000 [=======>......................] - ETA: 1:19 - loss: 0.3552 - categorical_accuracy: 0.8910
16800/60000 [=======>......................] - ETA: 1:19 - loss: 0.3546 - categorical_accuracy: 0.8912
16832/60000 [=======>......................] - ETA: 1:19 - loss: 0.3542 - categorical_accuracy: 0.8913
16864/60000 [=======>......................] - ETA: 1:19 - loss: 0.3541 - categorical_accuracy: 0.8913
16896/60000 [=======>......................] - ETA: 1:19 - loss: 0.3536 - categorical_accuracy: 0.8914
16928/60000 [=======>......................] - ETA: 1:19 - loss: 0.3531 - categorical_accuracy: 0.8915
16960/60000 [=======>......................] - ETA: 1:19 - loss: 0.3527 - categorical_accuracy: 0.8916
16992/60000 [=======>......................] - ETA: 1:19 - loss: 0.3522 - categorical_accuracy: 0.8918
17024/60000 [=======>......................] - ETA: 1:18 - loss: 0.3518 - categorical_accuracy: 0.8919
17056/60000 [=======>......................] - ETA: 1:18 - loss: 0.3512 - categorical_accuracy: 0.8921
17088/60000 [=======>......................] - ETA: 1:18 - loss: 0.3510 - categorical_accuracy: 0.8921
17120/60000 [=======>......................] - ETA: 1:18 - loss: 0.3509 - categorical_accuracy: 0.8922
17152/60000 [=======>......................] - ETA: 1:18 - loss: 0.3512 - categorical_accuracy: 0.8922
17184/60000 [=======>......................] - ETA: 1:18 - loss: 0.3510 - categorical_accuracy: 0.8922
17216/60000 [=======>......................] - ETA: 1:18 - loss: 0.3505 - categorical_accuracy: 0.8924
17248/60000 [=======>......................] - ETA: 1:18 - loss: 0.3501 - categorical_accuracy: 0.8926
17280/60000 [=======>......................] - ETA: 1:18 - loss: 0.3496 - categorical_accuracy: 0.8927
17312/60000 [=======>......................] - ETA: 1:18 - loss: 0.3493 - categorical_accuracy: 0.8927
17344/60000 [=======>......................] - ETA: 1:18 - loss: 0.3488 - categorical_accuracy: 0.8928
17376/60000 [=======>......................] - ETA: 1:18 - loss: 0.3483 - categorical_accuracy: 0.8930
17408/60000 [=======>......................] - ETA: 1:18 - loss: 0.3481 - categorical_accuracy: 0.8930
17440/60000 [=======>......................] - ETA: 1:18 - loss: 0.3476 - categorical_accuracy: 0.8932
17472/60000 [=======>......................] - ETA: 1:18 - loss: 0.3471 - categorical_accuracy: 0.8933
17504/60000 [=======>......................] - ETA: 1:18 - loss: 0.3466 - categorical_accuracy: 0.8935
17536/60000 [=======>......................] - ETA: 1:18 - loss: 0.3461 - categorical_accuracy: 0.8936
17568/60000 [=======>......................] - ETA: 1:17 - loss: 0.3457 - categorical_accuracy: 0.8938
17600/60000 [=======>......................] - ETA: 1:17 - loss: 0.3452 - categorical_accuracy: 0.8940
17632/60000 [=======>......................] - ETA: 1:17 - loss: 0.3449 - categorical_accuracy: 0.8941
17664/60000 [=======>......................] - ETA: 1:17 - loss: 0.3446 - categorical_accuracy: 0.8942
17696/60000 [=======>......................] - ETA: 1:17 - loss: 0.3448 - categorical_accuracy: 0.8942
17728/60000 [=======>......................] - ETA: 1:17 - loss: 0.3447 - categorical_accuracy: 0.8942
17760/60000 [=======>......................] - ETA: 1:17 - loss: 0.3442 - categorical_accuracy: 0.8944
17792/60000 [=======>......................] - ETA: 1:17 - loss: 0.3441 - categorical_accuracy: 0.8945
17824/60000 [=======>......................] - ETA: 1:17 - loss: 0.3439 - categorical_accuracy: 0.8944
17856/60000 [=======>......................] - ETA: 1:17 - loss: 0.3435 - categorical_accuracy: 0.8945
17888/60000 [=======>......................] - ETA: 1:17 - loss: 0.3431 - categorical_accuracy: 0.8947
17920/60000 [=======>......................] - ETA: 1:17 - loss: 0.3426 - categorical_accuracy: 0.8949
17952/60000 [=======>......................] - ETA: 1:17 - loss: 0.3428 - categorical_accuracy: 0.8949
17984/60000 [=======>......................] - ETA: 1:17 - loss: 0.3426 - categorical_accuracy: 0.8949
18016/60000 [========>.....................] - ETA: 1:17 - loss: 0.3421 - categorical_accuracy: 0.8951
18048/60000 [========>.....................] - ETA: 1:17 - loss: 0.3417 - categorical_accuracy: 0.8952
18080/60000 [========>.....................] - ETA: 1:17 - loss: 0.3413 - categorical_accuracy: 0.8953
18112/60000 [========>.....................] - ETA: 1:16 - loss: 0.3409 - categorical_accuracy: 0.8954
18144/60000 [========>.....................] - ETA: 1:16 - loss: 0.3404 - categorical_accuracy: 0.8956
18176/60000 [========>.....................] - ETA: 1:16 - loss: 0.3398 - categorical_accuracy: 0.8958
18208/60000 [========>.....................] - ETA: 1:16 - loss: 0.3394 - categorical_accuracy: 0.8959
18240/60000 [========>.....................] - ETA: 1:16 - loss: 0.3389 - categorical_accuracy: 0.8961
18272/60000 [========>.....................] - ETA: 1:16 - loss: 0.3384 - categorical_accuracy: 0.8962
18304/60000 [========>.....................] - ETA: 1:16 - loss: 0.3381 - categorical_accuracy: 0.8963
18336/60000 [========>.....................] - ETA: 1:16 - loss: 0.3387 - categorical_accuracy: 0.8963
18368/60000 [========>.....................] - ETA: 1:16 - loss: 0.3383 - categorical_accuracy: 0.8965
18400/60000 [========>.....................] - ETA: 1:16 - loss: 0.3386 - categorical_accuracy: 0.8965
18432/60000 [========>.....................] - ETA: 1:16 - loss: 0.3382 - categorical_accuracy: 0.8966
18464/60000 [========>.....................] - ETA: 1:16 - loss: 0.3378 - categorical_accuracy: 0.8967
18496/60000 [========>.....................] - ETA: 1:16 - loss: 0.3376 - categorical_accuracy: 0.8968
18528/60000 [========>.....................] - ETA: 1:16 - loss: 0.3374 - categorical_accuracy: 0.8969
18560/60000 [========>.....................] - ETA: 1:16 - loss: 0.3369 - categorical_accuracy: 0.8971
18592/60000 [========>.....................] - ETA: 1:16 - loss: 0.3365 - categorical_accuracy: 0.8972
18624/60000 [========>.....................] - ETA: 1:16 - loss: 0.3360 - categorical_accuracy: 0.8973
18656/60000 [========>.....................] - ETA: 1:15 - loss: 0.3356 - categorical_accuracy: 0.8974
18688/60000 [========>.....................] - ETA: 1:15 - loss: 0.3351 - categorical_accuracy: 0.8976
18720/60000 [========>.....................] - ETA: 1:15 - loss: 0.3350 - categorical_accuracy: 0.8975
18752/60000 [========>.....................] - ETA: 1:15 - loss: 0.3350 - categorical_accuracy: 0.8975
18784/60000 [========>.....................] - ETA: 1:15 - loss: 0.3347 - categorical_accuracy: 0.8975
18816/60000 [========>.....................] - ETA: 1:15 - loss: 0.3344 - categorical_accuracy: 0.8976
18848/60000 [========>.....................] - ETA: 1:15 - loss: 0.3340 - categorical_accuracy: 0.8977
18880/60000 [========>.....................] - ETA: 1:15 - loss: 0.3335 - categorical_accuracy: 0.8978
18912/60000 [========>.....................] - ETA: 1:15 - loss: 0.3331 - categorical_accuracy: 0.8979
18944/60000 [========>.....................] - ETA: 1:15 - loss: 0.3328 - categorical_accuracy: 0.8981
18976/60000 [========>.....................] - ETA: 1:15 - loss: 0.3329 - categorical_accuracy: 0.8981
19008/60000 [========>.....................] - ETA: 1:15 - loss: 0.3325 - categorical_accuracy: 0.8983
19040/60000 [========>.....................] - ETA: 1:15 - loss: 0.3328 - categorical_accuracy: 0.8983
19072/60000 [========>.....................] - ETA: 1:15 - loss: 0.3328 - categorical_accuracy: 0.8983
19104/60000 [========>.....................] - ETA: 1:15 - loss: 0.3325 - categorical_accuracy: 0.8984
19136/60000 [========>.....................] - ETA: 1:15 - loss: 0.3321 - categorical_accuracy: 0.8985
19168/60000 [========>.....................] - ETA: 1:14 - loss: 0.3318 - categorical_accuracy: 0.8985
19200/60000 [========>.....................] - ETA: 1:14 - loss: 0.3315 - categorical_accuracy: 0.8986
19232/60000 [========>.....................] - ETA: 1:14 - loss: 0.3311 - categorical_accuracy: 0.8987
19264/60000 [========>.....................] - ETA: 1:14 - loss: 0.3309 - categorical_accuracy: 0.8988
19296/60000 [========>.....................] - ETA: 1:14 - loss: 0.3317 - categorical_accuracy: 0.8987
19328/60000 [========>.....................] - ETA: 1:14 - loss: 0.3312 - categorical_accuracy: 0.8989
19360/60000 [========>.....................] - ETA: 1:14 - loss: 0.3308 - categorical_accuracy: 0.8991
19392/60000 [========>.....................] - ETA: 1:14 - loss: 0.3304 - categorical_accuracy: 0.8992
19424/60000 [========>.....................] - ETA: 1:14 - loss: 0.3302 - categorical_accuracy: 0.8992
19456/60000 [========>.....................] - ETA: 1:14 - loss: 0.3299 - categorical_accuracy: 0.8994
19488/60000 [========>.....................] - ETA: 1:14 - loss: 0.3295 - categorical_accuracy: 0.8995
19520/60000 [========>.....................] - ETA: 1:14 - loss: 0.3294 - categorical_accuracy: 0.8995
19552/60000 [========>.....................] - ETA: 1:14 - loss: 0.3291 - categorical_accuracy: 0.8995
19584/60000 [========>.....................] - ETA: 1:14 - loss: 0.3286 - categorical_accuracy: 0.8997
19616/60000 [========>.....................] - ETA: 1:14 - loss: 0.3284 - categorical_accuracy: 0.8997
19648/60000 [========>.....................] - ETA: 1:14 - loss: 0.3281 - categorical_accuracy: 0.8998
19680/60000 [========>.....................] - ETA: 1:14 - loss: 0.3277 - categorical_accuracy: 0.9000
19712/60000 [========>.....................] - ETA: 1:13 - loss: 0.3275 - categorical_accuracy: 0.9001
19744/60000 [========>.....................] - ETA: 1:13 - loss: 0.3270 - categorical_accuracy: 0.9002
19776/60000 [========>.....................] - ETA: 1:13 - loss: 0.3268 - categorical_accuracy: 0.9003
19808/60000 [========>.....................] - ETA: 1:13 - loss: 0.3263 - categorical_accuracy: 0.9004
19840/60000 [========>.....................] - ETA: 1:13 - loss: 0.3260 - categorical_accuracy: 0.9005
19872/60000 [========>.....................] - ETA: 1:13 - loss: 0.3257 - categorical_accuracy: 0.9006
19904/60000 [========>.....................] - ETA: 1:13 - loss: 0.3253 - categorical_accuracy: 0.9008
19936/60000 [========>.....................] - ETA: 1:13 - loss: 0.3250 - categorical_accuracy: 0.9008
19968/60000 [========>.....................] - ETA: 1:13 - loss: 0.3246 - categorical_accuracy: 0.9009
20000/60000 [=========>....................] - ETA: 1:13 - loss: 0.3243 - categorical_accuracy: 0.9010
20032/60000 [=========>....................] - ETA: 1:13 - loss: 0.3239 - categorical_accuracy: 0.9011
20064/60000 [=========>....................] - ETA: 1:13 - loss: 0.3236 - categorical_accuracy: 0.9011
20096/60000 [=========>....................] - ETA: 1:13 - loss: 0.3234 - categorical_accuracy: 0.9012
20128/60000 [=========>....................] - ETA: 1:13 - loss: 0.3231 - categorical_accuracy: 0.9013
20160/60000 [=========>....................] - ETA: 1:13 - loss: 0.3231 - categorical_accuracy: 0.9012
20192/60000 [=========>....................] - ETA: 1:13 - loss: 0.3230 - categorical_accuracy: 0.9012
20224/60000 [=========>....................] - ETA: 1:13 - loss: 0.3227 - categorical_accuracy: 0.9013
20256/60000 [=========>....................] - ETA: 1:13 - loss: 0.3223 - categorical_accuracy: 0.9015
20288/60000 [=========>....................] - ETA: 1:12 - loss: 0.3221 - categorical_accuracy: 0.9015
20320/60000 [=========>....................] - ETA: 1:12 - loss: 0.3219 - categorical_accuracy: 0.9015
20352/60000 [=========>....................] - ETA: 1:12 - loss: 0.3216 - categorical_accuracy: 0.9016
20384/60000 [=========>....................] - ETA: 1:12 - loss: 0.3215 - categorical_accuracy: 0.9016
20416/60000 [=========>....................] - ETA: 1:12 - loss: 0.3211 - categorical_accuracy: 0.9017
20448/60000 [=========>....................] - ETA: 1:12 - loss: 0.3208 - categorical_accuracy: 0.9018
20480/60000 [=========>....................] - ETA: 1:12 - loss: 0.3208 - categorical_accuracy: 0.9017
20512/60000 [=========>....................] - ETA: 1:12 - loss: 0.3206 - categorical_accuracy: 0.9018
20544/60000 [=========>....................] - ETA: 1:12 - loss: 0.3204 - categorical_accuracy: 0.9019
20576/60000 [=========>....................] - ETA: 1:12 - loss: 0.3200 - categorical_accuracy: 0.9020
20608/60000 [=========>....................] - ETA: 1:12 - loss: 0.3201 - categorical_accuracy: 0.9020
20640/60000 [=========>....................] - ETA: 1:12 - loss: 0.3199 - categorical_accuracy: 0.9020
20672/60000 [=========>....................] - ETA: 1:12 - loss: 0.3195 - categorical_accuracy: 0.9022
20704/60000 [=========>....................] - ETA: 1:12 - loss: 0.3192 - categorical_accuracy: 0.9023
20736/60000 [=========>....................] - ETA: 1:12 - loss: 0.3189 - categorical_accuracy: 0.9024
20768/60000 [=========>....................] - ETA: 1:12 - loss: 0.3189 - categorical_accuracy: 0.9025
20800/60000 [=========>....................] - ETA: 1:12 - loss: 0.3185 - categorical_accuracy: 0.9026
20832/60000 [=========>....................] - ETA: 1:11 - loss: 0.3182 - categorical_accuracy: 0.9027
20864/60000 [=========>....................] - ETA: 1:11 - loss: 0.3182 - categorical_accuracy: 0.9028
20896/60000 [=========>....................] - ETA: 1:11 - loss: 0.3181 - categorical_accuracy: 0.9028
20928/60000 [=========>....................] - ETA: 1:11 - loss: 0.3176 - categorical_accuracy: 0.9030
20960/60000 [=========>....................] - ETA: 1:11 - loss: 0.3173 - categorical_accuracy: 0.9030
20992/60000 [=========>....................] - ETA: 1:11 - loss: 0.3171 - categorical_accuracy: 0.9031
21024/60000 [=========>....................] - ETA: 1:11 - loss: 0.3169 - categorical_accuracy: 0.9031
21056/60000 [=========>....................] - ETA: 1:11 - loss: 0.3165 - categorical_accuracy: 0.9033
21088/60000 [=========>....................] - ETA: 1:11 - loss: 0.3162 - categorical_accuracy: 0.9034
21120/60000 [=========>....................] - ETA: 1:11 - loss: 0.3159 - categorical_accuracy: 0.9034
21152/60000 [=========>....................] - ETA: 1:11 - loss: 0.3157 - categorical_accuracy: 0.9034
21184/60000 [=========>....................] - ETA: 1:11 - loss: 0.3153 - categorical_accuracy: 0.9035
21216/60000 [=========>....................] - ETA: 1:11 - loss: 0.3154 - categorical_accuracy: 0.9035
21248/60000 [=========>....................] - ETA: 1:11 - loss: 0.3151 - categorical_accuracy: 0.9037
21280/60000 [=========>....................] - ETA: 1:11 - loss: 0.3148 - categorical_accuracy: 0.9038
21312/60000 [=========>....................] - ETA: 1:11 - loss: 0.3144 - categorical_accuracy: 0.9039
21344/60000 [=========>....................] - ETA: 1:11 - loss: 0.3144 - categorical_accuracy: 0.9039
21376/60000 [=========>....................] - ETA: 1:10 - loss: 0.3139 - categorical_accuracy: 0.9040
21408/60000 [=========>....................] - ETA: 1:10 - loss: 0.3135 - categorical_accuracy: 0.9041
21440/60000 [=========>....................] - ETA: 1:10 - loss: 0.3134 - categorical_accuracy: 0.9042
21472/60000 [=========>....................] - ETA: 1:10 - loss: 0.3131 - categorical_accuracy: 0.9042
21504/60000 [=========>....................] - ETA: 1:10 - loss: 0.3128 - categorical_accuracy: 0.9043
21536/60000 [=========>....................] - ETA: 1:10 - loss: 0.3126 - categorical_accuracy: 0.9043
21568/60000 [=========>....................] - ETA: 1:10 - loss: 0.3124 - categorical_accuracy: 0.9044
21600/60000 [=========>....................] - ETA: 1:10 - loss: 0.3123 - categorical_accuracy: 0.9044
21632/60000 [=========>....................] - ETA: 1:10 - loss: 0.3122 - categorical_accuracy: 0.9044
21664/60000 [=========>....................] - ETA: 1:10 - loss: 0.3119 - categorical_accuracy: 0.9045
21696/60000 [=========>....................] - ETA: 1:10 - loss: 0.3115 - categorical_accuracy: 0.9046
21728/60000 [=========>....................] - ETA: 1:10 - loss: 0.3113 - categorical_accuracy: 0.9047
21760/60000 [=========>....................] - ETA: 1:10 - loss: 0.3110 - categorical_accuracy: 0.9047
21792/60000 [=========>....................] - ETA: 1:10 - loss: 0.3109 - categorical_accuracy: 0.9048
21824/60000 [=========>....................] - ETA: 1:10 - loss: 0.3105 - categorical_accuracy: 0.9049
21856/60000 [=========>....................] - ETA: 1:10 - loss: 0.3103 - categorical_accuracy: 0.9049
21888/60000 [=========>....................] - ETA: 1:10 - loss: 0.3100 - categorical_accuracy: 0.9051
21920/60000 [=========>....................] - ETA: 1:10 - loss: 0.3096 - categorical_accuracy: 0.9052
21952/60000 [=========>....................] - ETA: 1:09 - loss: 0.3092 - categorical_accuracy: 0.9053
21984/60000 [=========>....................] - ETA: 1:09 - loss: 0.3090 - categorical_accuracy: 0.9053
22016/60000 [==========>...................] - ETA: 1:09 - loss: 0.3087 - categorical_accuracy: 0.9054
22048/60000 [==========>...................] - ETA: 1:09 - loss: 0.3083 - categorical_accuracy: 0.9055
22080/60000 [==========>...................] - ETA: 1:09 - loss: 0.3080 - categorical_accuracy: 0.9056
22112/60000 [==========>...................] - ETA: 1:09 - loss: 0.3076 - categorical_accuracy: 0.9057
22144/60000 [==========>...................] - ETA: 1:09 - loss: 0.3073 - categorical_accuracy: 0.9058
22176/60000 [==========>...................] - ETA: 1:09 - loss: 0.3072 - categorical_accuracy: 0.9059
22208/60000 [==========>...................] - ETA: 1:09 - loss: 0.3071 - categorical_accuracy: 0.9059
22240/60000 [==========>...................] - ETA: 1:09 - loss: 0.3069 - categorical_accuracy: 0.9060
22272/60000 [==========>...................] - ETA: 1:09 - loss: 0.3066 - categorical_accuracy: 0.9061
22304/60000 [==========>...................] - ETA: 1:09 - loss: 0.3062 - categorical_accuracy: 0.9062
22336/60000 [==========>...................] - ETA: 1:09 - loss: 0.3058 - categorical_accuracy: 0.9064
22368/60000 [==========>...................] - ETA: 1:09 - loss: 0.3055 - categorical_accuracy: 0.9065
22400/60000 [==========>...................] - ETA: 1:09 - loss: 0.3055 - categorical_accuracy: 0.9065
22432/60000 [==========>...................] - ETA: 1:09 - loss: 0.3061 - categorical_accuracy: 0.9063
22464/60000 [==========>...................] - ETA: 1:09 - loss: 0.3058 - categorical_accuracy: 0.9064
22496/60000 [==========>...................] - ETA: 1:09 - loss: 0.3054 - categorical_accuracy: 0.9066
22528/60000 [==========>...................] - ETA: 1:09 - loss: 0.3052 - categorical_accuracy: 0.9066
22560/60000 [==========>...................] - ETA: 1:08 - loss: 0.3050 - categorical_accuracy: 0.9067
22592/60000 [==========>...................] - ETA: 1:08 - loss: 0.3047 - categorical_accuracy: 0.9067
22624/60000 [==========>...................] - ETA: 1:08 - loss: 0.3044 - categorical_accuracy: 0.9069
22656/60000 [==========>...................] - ETA: 1:08 - loss: 0.3041 - categorical_accuracy: 0.9070
22688/60000 [==========>...................] - ETA: 1:08 - loss: 0.3037 - categorical_accuracy: 0.9071
22720/60000 [==========>...................] - ETA: 1:08 - loss: 0.3033 - categorical_accuracy: 0.9072
22752/60000 [==========>...................] - ETA: 1:08 - loss: 0.3032 - categorical_accuracy: 0.9073
22784/60000 [==========>...................] - ETA: 1:08 - loss: 0.3029 - categorical_accuracy: 0.9074
22816/60000 [==========>...................] - ETA: 1:08 - loss: 0.3028 - categorical_accuracy: 0.9074
22848/60000 [==========>...................] - ETA: 1:08 - loss: 0.3024 - categorical_accuracy: 0.9076
22880/60000 [==========>...................] - ETA: 1:08 - loss: 0.3023 - categorical_accuracy: 0.9076
22912/60000 [==========>...................] - ETA: 1:08 - loss: 0.3020 - categorical_accuracy: 0.9077
22944/60000 [==========>...................] - ETA: 1:08 - loss: 0.3016 - categorical_accuracy: 0.9078
22976/60000 [==========>...................] - ETA: 1:08 - loss: 0.3013 - categorical_accuracy: 0.9079
23008/60000 [==========>...................] - ETA: 1:08 - loss: 0.3013 - categorical_accuracy: 0.9079
23040/60000 [==========>...................] - ETA: 1:08 - loss: 0.3010 - categorical_accuracy: 0.9080
23072/60000 [==========>...................] - ETA: 1:08 - loss: 0.3010 - categorical_accuracy: 0.9080
23104/60000 [==========>...................] - ETA: 1:08 - loss: 0.3007 - categorical_accuracy: 0.9081
23136/60000 [==========>...................] - ETA: 1:07 - loss: 0.3006 - categorical_accuracy: 0.9082
23168/60000 [==========>...................] - ETA: 1:07 - loss: 0.3003 - categorical_accuracy: 0.9082
23200/60000 [==========>...................] - ETA: 1:07 - loss: 0.3000 - categorical_accuracy: 0.9083
23232/60000 [==========>...................] - ETA: 1:07 - loss: 0.2997 - categorical_accuracy: 0.9084
23264/60000 [==========>...................] - ETA: 1:07 - loss: 0.2993 - categorical_accuracy: 0.9085
23296/60000 [==========>...................] - ETA: 1:07 - loss: 0.2991 - categorical_accuracy: 0.9086
23328/60000 [==========>...................] - ETA: 1:07 - loss: 0.2989 - categorical_accuracy: 0.9087
23360/60000 [==========>...................] - ETA: 1:07 - loss: 0.2985 - categorical_accuracy: 0.9088
23392/60000 [==========>...................] - ETA: 1:07 - loss: 0.2982 - categorical_accuracy: 0.9089
23424/60000 [==========>...................] - ETA: 1:07 - loss: 0.2979 - categorical_accuracy: 0.9090
23456/60000 [==========>...................] - ETA: 1:07 - loss: 0.2977 - categorical_accuracy: 0.9091
23488/60000 [==========>...................] - ETA: 1:07 - loss: 0.2973 - categorical_accuracy: 0.9092
23520/60000 [==========>...................] - ETA: 1:07 - loss: 0.2971 - categorical_accuracy: 0.9092
23552/60000 [==========>...................] - ETA: 1:07 - loss: 0.2972 - categorical_accuracy: 0.9092
23584/60000 [==========>...................] - ETA: 1:07 - loss: 0.2971 - categorical_accuracy: 0.9093
23616/60000 [==========>...................] - ETA: 1:07 - loss: 0.2969 - categorical_accuracy: 0.9093
23648/60000 [==========>...................] - ETA: 1:06 - loss: 0.2965 - categorical_accuracy: 0.9095
23680/60000 [==========>...................] - ETA: 1:06 - loss: 0.2965 - categorical_accuracy: 0.9095
23712/60000 [==========>...................] - ETA: 1:06 - loss: 0.2962 - categorical_accuracy: 0.9096
23744/60000 [==========>...................] - ETA: 1:06 - loss: 0.2960 - categorical_accuracy: 0.9096
23776/60000 [==========>...................] - ETA: 1:06 - loss: 0.2957 - categorical_accuracy: 0.9097
23808/60000 [==========>...................] - ETA: 1:06 - loss: 0.2953 - categorical_accuracy: 0.9099
23840/60000 [==========>...................] - ETA: 1:06 - loss: 0.2950 - categorical_accuracy: 0.9099
23872/60000 [==========>...................] - ETA: 1:06 - loss: 0.2947 - categorical_accuracy: 0.9101
23904/60000 [==========>...................] - ETA: 1:06 - loss: 0.2944 - categorical_accuracy: 0.9101
23936/60000 [==========>...................] - ETA: 1:06 - loss: 0.2942 - categorical_accuracy: 0.9102
23968/60000 [==========>...................] - ETA: 1:06 - loss: 0.2939 - categorical_accuracy: 0.9103
24000/60000 [===========>..................] - ETA: 1:06 - loss: 0.2936 - categorical_accuracy: 0.9104
24032/60000 [===========>..................] - ETA: 1:06 - loss: 0.2933 - categorical_accuracy: 0.9105
24064/60000 [===========>..................] - ETA: 1:06 - loss: 0.2935 - categorical_accuracy: 0.9104
24096/60000 [===========>..................] - ETA: 1:06 - loss: 0.2934 - categorical_accuracy: 0.9105
24128/60000 [===========>..................] - ETA: 1:06 - loss: 0.2930 - categorical_accuracy: 0.9106
24160/60000 [===========>..................] - ETA: 1:06 - loss: 0.2927 - categorical_accuracy: 0.9107
24192/60000 [===========>..................] - ETA: 1:05 - loss: 0.2924 - categorical_accuracy: 0.9108
24224/60000 [===========>..................] - ETA: 1:05 - loss: 0.2921 - categorical_accuracy: 0.9109
24256/60000 [===========>..................] - ETA: 1:05 - loss: 0.2918 - categorical_accuracy: 0.9110
24288/60000 [===========>..................] - ETA: 1:05 - loss: 0.2916 - categorical_accuracy: 0.9110
24320/60000 [===========>..................] - ETA: 1:05 - loss: 0.2912 - categorical_accuracy: 0.9111
24352/60000 [===========>..................] - ETA: 1:05 - loss: 0.2908 - categorical_accuracy: 0.9113
24384/60000 [===========>..................] - ETA: 1:05 - loss: 0.2905 - categorical_accuracy: 0.9114
24416/60000 [===========>..................] - ETA: 1:05 - loss: 0.2901 - categorical_accuracy: 0.9115
24448/60000 [===========>..................] - ETA: 1:05 - loss: 0.2902 - categorical_accuracy: 0.9115
24480/60000 [===========>..................] - ETA: 1:05 - loss: 0.2899 - categorical_accuracy: 0.9116
24512/60000 [===========>..................] - ETA: 1:05 - loss: 0.2896 - categorical_accuracy: 0.9117
24544/60000 [===========>..................] - ETA: 1:05 - loss: 0.2893 - categorical_accuracy: 0.9118
24576/60000 [===========>..................] - ETA: 1:05 - loss: 0.2891 - categorical_accuracy: 0.9119
24608/60000 [===========>..................] - ETA: 1:05 - loss: 0.2887 - categorical_accuracy: 0.9120
24640/60000 [===========>..................] - ETA: 1:05 - loss: 0.2884 - categorical_accuracy: 0.9121
24672/60000 [===========>..................] - ETA: 1:05 - loss: 0.2881 - categorical_accuracy: 0.9122
24704/60000 [===========>..................] - ETA: 1:05 - loss: 0.2880 - categorical_accuracy: 0.9122
24736/60000 [===========>..................] - ETA: 1:05 - loss: 0.2876 - categorical_accuracy: 0.9124
24768/60000 [===========>..................] - ETA: 1:04 - loss: 0.2874 - categorical_accuracy: 0.9124
24800/60000 [===========>..................] - ETA: 1:04 - loss: 0.2872 - categorical_accuracy: 0.9125
24832/60000 [===========>..................] - ETA: 1:04 - loss: 0.2869 - categorical_accuracy: 0.9125
24864/60000 [===========>..................] - ETA: 1:04 - loss: 0.2871 - categorical_accuracy: 0.9124
24896/60000 [===========>..................] - ETA: 1:04 - loss: 0.2868 - categorical_accuracy: 0.9126
24928/60000 [===========>..................] - ETA: 1:04 - loss: 0.2866 - categorical_accuracy: 0.9126
24960/60000 [===========>..................] - ETA: 1:04 - loss: 0.2864 - categorical_accuracy: 0.9127
24992/60000 [===========>..................] - ETA: 1:04 - loss: 0.2863 - categorical_accuracy: 0.9127
25024/60000 [===========>..................] - ETA: 1:04 - loss: 0.2863 - categorical_accuracy: 0.9127
25056/60000 [===========>..................] - ETA: 1:04 - loss: 0.2860 - categorical_accuracy: 0.9128
25088/60000 [===========>..................] - ETA: 1:04 - loss: 0.2858 - categorical_accuracy: 0.9128
25120/60000 [===========>..................] - ETA: 1:04 - loss: 0.2854 - categorical_accuracy: 0.9129
25152/60000 [===========>..................] - ETA: 1:04 - loss: 0.2853 - categorical_accuracy: 0.9130
25184/60000 [===========>..................] - ETA: 1:04 - loss: 0.2851 - categorical_accuracy: 0.9130
25216/60000 [===========>..................] - ETA: 1:04 - loss: 0.2849 - categorical_accuracy: 0.9131
25248/60000 [===========>..................] - ETA: 1:04 - loss: 0.2848 - categorical_accuracy: 0.9131
25280/60000 [===========>..................] - ETA: 1:03 - loss: 0.2846 - categorical_accuracy: 0.9132
25312/60000 [===========>..................] - ETA: 1:03 - loss: 0.2847 - categorical_accuracy: 0.9132
25344/60000 [===========>..................] - ETA: 1:03 - loss: 0.2844 - categorical_accuracy: 0.9133
25376/60000 [===========>..................] - ETA: 1:03 - loss: 0.2841 - categorical_accuracy: 0.9134
25408/60000 [===========>..................] - ETA: 1:03 - loss: 0.2839 - categorical_accuracy: 0.9135
25440/60000 [===========>..................] - ETA: 1:03 - loss: 0.2837 - categorical_accuracy: 0.9136
25472/60000 [===========>..................] - ETA: 1:03 - loss: 0.2834 - categorical_accuracy: 0.9137
25504/60000 [===========>..................] - ETA: 1:03 - loss: 0.2831 - categorical_accuracy: 0.9138
25536/60000 [===========>..................] - ETA: 1:03 - loss: 0.2829 - categorical_accuracy: 0.9138
25568/60000 [===========>..................] - ETA: 1:03 - loss: 0.2828 - categorical_accuracy: 0.9138
25600/60000 [===========>..................] - ETA: 1:03 - loss: 0.2825 - categorical_accuracy: 0.9139
25632/60000 [===========>..................] - ETA: 1:03 - loss: 0.2822 - categorical_accuracy: 0.9141
25664/60000 [===========>..................] - ETA: 1:03 - loss: 0.2819 - categorical_accuracy: 0.9141
25696/60000 [===========>..................] - ETA: 1:03 - loss: 0.2817 - categorical_accuracy: 0.9141
25728/60000 [===========>..................] - ETA: 1:03 - loss: 0.2814 - categorical_accuracy: 0.9142
25760/60000 [===========>..................] - ETA: 1:03 - loss: 0.2811 - categorical_accuracy: 0.9143
25792/60000 [===========>..................] - ETA: 1:03 - loss: 0.2810 - categorical_accuracy: 0.9143
25824/60000 [===========>..................] - ETA: 1:02 - loss: 0.2809 - categorical_accuracy: 0.9143
25856/60000 [===========>..................] - ETA: 1:02 - loss: 0.2806 - categorical_accuracy: 0.9144
25888/60000 [===========>..................] - ETA: 1:02 - loss: 0.2803 - categorical_accuracy: 0.9145
25920/60000 [===========>..................] - ETA: 1:02 - loss: 0.2802 - categorical_accuracy: 0.9145
25952/60000 [===========>..................] - ETA: 1:02 - loss: 0.2799 - categorical_accuracy: 0.9146
25984/60000 [===========>..................] - ETA: 1:02 - loss: 0.2796 - categorical_accuracy: 0.9147
26016/60000 [============>.................] - ETA: 1:02 - loss: 0.2794 - categorical_accuracy: 0.9147
26048/60000 [============>.................] - ETA: 1:02 - loss: 0.2791 - categorical_accuracy: 0.9148
26080/60000 [============>.................] - ETA: 1:02 - loss: 0.2792 - categorical_accuracy: 0.9148
26112/60000 [============>.................] - ETA: 1:02 - loss: 0.2789 - categorical_accuracy: 0.9148
26144/60000 [============>.................] - ETA: 1:02 - loss: 0.2788 - categorical_accuracy: 0.9149
26176/60000 [============>.................] - ETA: 1:02 - loss: 0.2785 - categorical_accuracy: 0.9150
26208/60000 [============>.................] - ETA: 1:02 - loss: 0.2782 - categorical_accuracy: 0.9151
26240/60000 [============>.................] - ETA: 1:02 - loss: 0.2782 - categorical_accuracy: 0.9151
26272/60000 [============>.................] - ETA: 1:02 - loss: 0.2780 - categorical_accuracy: 0.9150
26304/60000 [============>.................] - ETA: 1:02 - loss: 0.2778 - categorical_accuracy: 0.9151
26336/60000 [============>.................] - ETA: 1:01 - loss: 0.2781 - categorical_accuracy: 0.9151
26368/60000 [============>.................] - ETA: 1:01 - loss: 0.2781 - categorical_accuracy: 0.9150
26400/60000 [============>.................] - ETA: 1:01 - loss: 0.2779 - categorical_accuracy: 0.9151
26432/60000 [============>.................] - ETA: 1:01 - loss: 0.2776 - categorical_accuracy: 0.9152
26464/60000 [============>.................] - ETA: 1:01 - loss: 0.2775 - categorical_accuracy: 0.9152
26496/60000 [============>.................] - ETA: 1:01 - loss: 0.2775 - categorical_accuracy: 0.9152
26528/60000 [============>.................] - ETA: 1:01 - loss: 0.2772 - categorical_accuracy: 0.9153
26560/60000 [============>.................] - ETA: 1:01 - loss: 0.2772 - categorical_accuracy: 0.9152
26592/60000 [============>.................] - ETA: 1:01 - loss: 0.2769 - categorical_accuracy: 0.9153
26624/60000 [============>.................] - ETA: 1:01 - loss: 0.2768 - categorical_accuracy: 0.9153
26656/60000 [============>.................] - ETA: 1:01 - loss: 0.2766 - categorical_accuracy: 0.9154
26688/60000 [============>.................] - ETA: 1:01 - loss: 0.2764 - categorical_accuracy: 0.9154
26720/60000 [============>.................] - ETA: 1:01 - loss: 0.2761 - categorical_accuracy: 0.9155
26752/60000 [============>.................] - ETA: 1:01 - loss: 0.2758 - categorical_accuracy: 0.9156
26784/60000 [============>.................] - ETA: 1:01 - loss: 0.2756 - categorical_accuracy: 0.9157
26816/60000 [============>.................] - ETA: 1:01 - loss: 0.2756 - categorical_accuracy: 0.9158
26848/60000 [============>.................] - ETA: 1:01 - loss: 0.2756 - categorical_accuracy: 0.9157
26880/60000 [============>.................] - ETA: 1:01 - loss: 0.2753 - categorical_accuracy: 0.9158
26912/60000 [============>.................] - ETA: 1:00 - loss: 0.2754 - categorical_accuracy: 0.9158
26944/60000 [============>.................] - ETA: 1:00 - loss: 0.2752 - categorical_accuracy: 0.9159
26976/60000 [============>.................] - ETA: 1:00 - loss: 0.2750 - categorical_accuracy: 0.9160
27008/60000 [============>.................] - ETA: 1:00 - loss: 0.2747 - categorical_accuracy: 0.9161
27040/60000 [============>.................] - ETA: 1:00 - loss: 0.2746 - categorical_accuracy: 0.9161
27072/60000 [============>.................] - ETA: 1:00 - loss: 0.2743 - categorical_accuracy: 0.9162
27104/60000 [============>.................] - ETA: 1:00 - loss: 0.2744 - categorical_accuracy: 0.9163
27136/60000 [============>.................] - ETA: 1:00 - loss: 0.2747 - categorical_accuracy: 0.9163
27168/60000 [============>.................] - ETA: 1:00 - loss: 0.2748 - categorical_accuracy: 0.9163
27200/60000 [============>.................] - ETA: 1:00 - loss: 0.2747 - categorical_accuracy: 0.9163
27232/60000 [============>.................] - ETA: 1:00 - loss: 0.2746 - categorical_accuracy: 0.9163
27264/60000 [============>.................] - ETA: 1:00 - loss: 0.2746 - categorical_accuracy: 0.9163
27296/60000 [============>.................] - ETA: 1:00 - loss: 0.2743 - categorical_accuracy: 0.9164
27328/60000 [============>.................] - ETA: 1:00 - loss: 0.2741 - categorical_accuracy: 0.9165
27360/60000 [============>.................] - ETA: 1:00 - loss: 0.2739 - categorical_accuracy: 0.9166
27392/60000 [============>.................] - ETA: 1:00 - loss: 0.2737 - categorical_accuracy: 0.9167
27424/60000 [============>.................] - ETA: 59s - loss: 0.2734 - categorical_accuracy: 0.9167 
27456/60000 [============>.................] - ETA: 59s - loss: 0.2732 - categorical_accuracy: 0.9168
27488/60000 [============>.................] - ETA: 59s - loss: 0.2731 - categorical_accuracy: 0.9168
27520/60000 [============>.................] - ETA: 59s - loss: 0.2730 - categorical_accuracy: 0.9169
27552/60000 [============>.................] - ETA: 59s - loss: 0.2727 - categorical_accuracy: 0.9170
27584/60000 [============>.................] - ETA: 59s - loss: 0.2726 - categorical_accuracy: 0.9170
27616/60000 [============>.................] - ETA: 59s - loss: 0.2725 - categorical_accuracy: 0.9170
27648/60000 [============>.................] - ETA: 59s - loss: 0.2723 - categorical_accuracy: 0.9171
27680/60000 [============>.................] - ETA: 59s - loss: 0.2723 - categorical_accuracy: 0.9171
27712/60000 [============>.................] - ETA: 59s - loss: 0.2720 - categorical_accuracy: 0.9172
27744/60000 [============>.................] - ETA: 59s - loss: 0.2718 - categorical_accuracy: 0.9173
27776/60000 [============>.................] - ETA: 59s - loss: 0.2716 - categorical_accuracy: 0.9173
27808/60000 [============>.................] - ETA: 59s - loss: 0.2715 - categorical_accuracy: 0.9174
27840/60000 [============>.................] - ETA: 59s - loss: 0.2714 - categorical_accuracy: 0.9174
27872/60000 [============>.................] - ETA: 59s - loss: 0.2717 - categorical_accuracy: 0.9174
27904/60000 [============>.................] - ETA: 59s - loss: 0.2715 - categorical_accuracy: 0.9175
27936/60000 [============>.................] - ETA: 59s - loss: 0.2713 - categorical_accuracy: 0.9175
27968/60000 [============>.................] - ETA: 58s - loss: 0.2712 - categorical_accuracy: 0.9175
28000/60000 [=============>................] - ETA: 58s - loss: 0.2714 - categorical_accuracy: 0.9175
28032/60000 [=============>................] - ETA: 58s - loss: 0.2711 - categorical_accuracy: 0.9176
28064/60000 [=============>................] - ETA: 58s - loss: 0.2709 - categorical_accuracy: 0.9177
28096/60000 [=============>................] - ETA: 58s - loss: 0.2706 - categorical_accuracy: 0.9177
28128/60000 [=============>................] - ETA: 58s - loss: 0.2705 - categorical_accuracy: 0.9178
28160/60000 [=============>................] - ETA: 58s - loss: 0.2705 - categorical_accuracy: 0.9177
28192/60000 [=============>................] - ETA: 58s - loss: 0.2705 - categorical_accuracy: 0.9176
28224/60000 [=============>................] - ETA: 58s - loss: 0.2705 - categorical_accuracy: 0.9176
28256/60000 [=============>................] - ETA: 58s - loss: 0.2704 - categorical_accuracy: 0.9176
28288/60000 [=============>................] - ETA: 58s - loss: 0.2705 - categorical_accuracy: 0.9176
28320/60000 [=============>................] - ETA: 58s - loss: 0.2704 - categorical_accuracy: 0.9177
28352/60000 [=============>................] - ETA: 58s - loss: 0.2702 - categorical_accuracy: 0.9177
28384/60000 [=============>................] - ETA: 58s - loss: 0.2699 - categorical_accuracy: 0.9178
28416/60000 [=============>................] - ETA: 58s - loss: 0.2697 - categorical_accuracy: 0.9179
28448/60000 [=============>................] - ETA: 58s - loss: 0.2695 - categorical_accuracy: 0.9180
28480/60000 [=============>................] - ETA: 58s - loss: 0.2694 - categorical_accuracy: 0.9180
28512/60000 [=============>................] - ETA: 57s - loss: 0.2691 - categorical_accuracy: 0.9181
28544/60000 [=============>................] - ETA: 57s - loss: 0.2692 - categorical_accuracy: 0.9181
28576/60000 [=============>................] - ETA: 57s - loss: 0.2691 - categorical_accuracy: 0.9181
28608/60000 [=============>................] - ETA: 57s - loss: 0.2691 - categorical_accuracy: 0.9181
28640/60000 [=============>................] - ETA: 57s - loss: 0.2689 - categorical_accuracy: 0.9182
28672/60000 [=============>................] - ETA: 57s - loss: 0.2686 - categorical_accuracy: 0.9182
28704/60000 [=============>................] - ETA: 57s - loss: 0.2684 - categorical_accuracy: 0.9183
28736/60000 [=============>................] - ETA: 57s - loss: 0.2682 - categorical_accuracy: 0.9184
28768/60000 [=============>................] - ETA: 57s - loss: 0.2682 - categorical_accuracy: 0.9184
28800/60000 [=============>................] - ETA: 57s - loss: 0.2679 - categorical_accuracy: 0.9185
28832/60000 [=============>................] - ETA: 57s - loss: 0.2677 - categorical_accuracy: 0.9185
28864/60000 [=============>................] - ETA: 57s - loss: 0.2676 - categorical_accuracy: 0.9185
28896/60000 [=============>................] - ETA: 57s - loss: 0.2674 - categorical_accuracy: 0.9186
28928/60000 [=============>................] - ETA: 57s - loss: 0.2674 - categorical_accuracy: 0.9186
28960/60000 [=============>................] - ETA: 57s - loss: 0.2672 - categorical_accuracy: 0.9187
28992/60000 [=============>................] - ETA: 57s - loss: 0.2673 - categorical_accuracy: 0.9187
29024/60000 [=============>................] - ETA: 57s - loss: 0.2671 - categorical_accuracy: 0.9187
29056/60000 [=============>................] - ETA: 56s - loss: 0.2670 - categorical_accuracy: 0.9188
29088/60000 [=============>................] - ETA: 56s - loss: 0.2667 - categorical_accuracy: 0.9188
29120/60000 [=============>................] - ETA: 56s - loss: 0.2666 - categorical_accuracy: 0.9189
29152/60000 [=============>................] - ETA: 56s - loss: 0.2664 - categorical_accuracy: 0.9190
29184/60000 [=============>................] - ETA: 56s - loss: 0.2665 - categorical_accuracy: 0.9190
29216/60000 [=============>................] - ETA: 56s - loss: 0.2664 - categorical_accuracy: 0.9190
29248/60000 [=============>................] - ETA: 56s - loss: 0.2663 - categorical_accuracy: 0.9190
29280/60000 [=============>................] - ETA: 56s - loss: 0.2661 - categorical_accuracy: 0.9191
29312/60000 [=============>................] - ETA: 56s - loss: 0.2658 - categorical_accuracy: 0.9192
29344/60000 [=============>................] - ETA: 56s - loss: 0.2658 - categorical_accuracy: 0.9193
29376/60000 [=============>................] - ETA: 56s - loss: 0.2655 - categorical_accuracy: 0.9194
29408/60000 [=============>................] - ETA: 56s - loss: 0.2653 - categorical_accuracy: 0.9194
29440/60000 [=============>................] - ETA: 56s - loss: 0.2652 - categorical_accuracy: 0.9194
29472/60000 [=============>................] - ETA: 56s - loss: 0.2649 - categorical_accuracy: 0.9195
29504/60000 [=============>................] - ETA: 56s - loss: 0.2647 - categorical_accuracy: 0.9195
29536/60000 [=============>................] - ETA: 56s - loss: 0.2645 - categorical_accuracy: 0.9196
29568/60000 [=============>................] - ETA: 55s - loss: 0.2643 - categorical_accuracy: 0.9196
29600/60000 [=============>................] - ETA: 55s - loss: 0.2642 - categorical_accuracy: 0.9197
29632/60000 [=============>................] - ETA: 55s - loss: 0.2640 - categorical_accuracy: 0.9197
29664/60000 [=============>................] - ETA: 55s - loss: 0.2642 - categorical_accuracy: 0.9196
29696/60000 [=============>................] - ETA: 55s - loss: 0.2639 - categorical_accuracy: 0.9197
29728/60000 [=============>................] - ETA: 55s - loss: 0.2637 - categorical_accuracy: 0.9197
29760/60000 [=============>................] - ETA: 55s - loss: 0.2635 - categorical_accuracy: 0.9198
29792/60000 [=============>................] - ETA: 55s - loss: 0.2632 - categorical_accuracy: 0.9199
29824/60000 [=============>................] - ETA: 55s - loss: 0.2629 - categorical_accuracy: 0.9200
29856/60000 [=============>................] - ETA: 55s - loss: 0.2628 - categorical_accuracy: 0.9200
29888/60000 [=============>................] - ETA: 55s - loss: 0.2629 - categorical_accuracy: 0.9200
29920/60000 [=============>................] - ETA: 55s - loss: 0.2626 - categorical_accuracy: 0.9201
29952/60000 [=============>................] - ETA: 55s - loss: 0.2624 - categorical_accuracy: 0.9201
29984/60000 [=============>................] - ETA: 55s - loss: 0.2621 - categorical_accuracy: 0.9202
30016/60000 [==============>...............] - ETA: 55s - loss: 0.2620 - categorical_accuracy: 0.9202
30048/60000 [==============>...............] - ETA: 55s - loss: 0.2620 - categorical_accuracy: 0.9202
30080/60000 [==============>...............] - ETA: 55s - loss: 0.2618 - categorical_accuracy: 0.9203
30112/60000 [==============>...............] - ETA: 54s - loss: 0.2616 - categorical_accuracy: 0.9203
30144/60000 [==============>...............] - ETA: 54s - loss: 0.2615 - categorical_accuracy: 0.9203
30176/60000 [==============>...............] - ETA: 54s - loss: 0.2613 - categorical_accuracy: 0.9204
30208/60000 [==============>...............] - ETA: 54s - loss: 0.2612 - categorical_accuracy: 0.9204
30240/60000 [==============>...............] - ETA: 54s - loss: 0.2611 - categorical_accuracy: 0.9205
30272/60000 [==============>...............] - ETA: 54s - loss: 0.2610 - categorical_accuracy: 0.9205
30304/60000 [==============>...............] - ETA: 54s - loss: 0.2611 - categorical_accuracy: 0.9205
30336/60000 [==============>...............] - ETA: 54s - loss: 0.2609 - categorical_accuracy: 0.9206
30368/60000 [==============>...............] - ETA: 54s - loss: 0.2607 - categorical_accuracy: 0.9207
30400/60000 [==============>...............] - ETA: 54s - loss: 0.2606 - categorical_accuracy: 0.9207
30432/60000 [==============>...............] - ETA: 54s - loss: 0.2604 - categorical_accuracy: 0.9208
30464/60000 [==============>...............] - ETA: 54s - loss: 0.2602 - categorical_accuracy: 0.9208
30496/60000 [==============>...............] - ETA: 54s - loss: 0.2600 - categorical_accuracy: 0.9209
30528/60000 [==============>...............] - ETA: 54s - loss: 0.2599 - categorical_accuracy: 0.9209
30560/60000 [==============>...............] - ETA: 54s - loss: 0.2598 - categorical_accuracy: 0.9209
30592/60000 [==============>...............] - ETA: 54s - loss: 0.2597 - categorical_accuracy: 0.9209
30624/60000 [==============>...............] - ETA: 54s - loss: 0.2597 - categorical_accuracy: 0.9209
30656/60000 [==============>...............] - ETA: 53s - loss: 0.2595 - categorical_accuracy: 0.9210
30688/60000 [==============>...............] - ETA: 53s - loss: 0.2595 - categorical_accuracy: 0.9210
30720/60000 [==============>...............] - ETA: 53s - loss: 0.2597 - categorical_accuracy: 0.9210
30752/60000 [==============>...............] - ETA: 53s - loss: 0.2595 - categorical_accuracy: 0.9211
30784/60000 [==============>...............] - ETA: 53s - loss: 0.2593 - categorical_accuracy: 0.9211
30816/60000 [==============>...............] - ETA: 53s - loss: 0.2591 - categorical_accuracy: 0.9212
30848/60000 [==============>...............] - ETA: 53s - loss: 0.2589 - categorical_accuracy: 0.9212
30880/60000 [==============>...............] - ETA: 53s - loss: 0.2589 - categorical_accuracy: 0.9213
30912/60000 [==============>...............] - ETA: 53s - loss: 0.2588 - categorical_accuracy: 0.9213
30944/60000 [==============>...............] - ETA: 53s - loss: 0.2586 - categorical_accuracy: 0.9213
30976/60000 [==============>...............] - ETA: 53s - loss: 0.2587 - categorical_accuracy: 0.9213
31008/60000 [==============>...............] - ETA: 53s - loss: 0.2587 - categorical_accuracy: 0.9213
31040/60000 [==============>...............] - ETA: 53s - loss: 0.2585 - categorical_accuracy: 0.9214
31072/60000 [==============>...............] - ETA: 53s - loss: 0.2583 - categorical_accuracy: 0.9215
31104/60000 [==============>...............] - ETA: 53s - loss: 0.2581 - categorical_accuracy: 0.9215
31136/60000 [==============>...............] - ETA: 53s - loss: 0.2580 - categorical_accuracy: 0.9216
31168/60000 [==============>...............] - ETA: 53s - loss: 0.2580 - categorical_accuracy: 0.9216
31200/60000 [==============>...............] - ETA: 52s - loss: 0.2582 - categorical_accuracy: 0.9216
31232/60000 [==============>...............] - ETA: 52s - loss: 0.2580 - categorical_accuracy: 0.9217
31264/60000 [==============>...............] - ETA: 52s - loss: 0.2578 - categorical_accuracy: 0.9217
31296/60000 [==============>...............] - ETA: 52s - loss: 0.2576 - categorical_accuracy: 0.9217
31328/60000 [==============>...............] - ETA: 52s - loss: 0.2576 - categorical_accuracy: 0.9218
31360/60000 [==============>...............] - ETA: 52s - loss: 0.2575 - categorical_accuracy: 0.9218
31392/60000 [==============>...............] - ETA: 52s - loss: 0.2572 - categorical_accuracy: 0.9218
31424/60000 [==============>...............] - ETA: 52s - loss: 0.2571 - categorical_accuracy: 0.9218
31456/60000 [==============>...............] - ETA: 52s - loss: 0.2570 - categorical_accuracy: 0.9219
31488/60000 [==============>...............] - ETA: 52s - loss: 0.2568 - categorical_accuracy: 0.9219
31520/60000 [==============>...............] - ETA: 52s - loss: 0.2568 - categorical_accuracy: 0.9219
31552/60000 [==============>...............] - ETA: 52s - loss: 0.2567 - categorical_accuracy: 0.9219
31584/60000 [==============>...............] - ETA: 52s - loss: 0.2567 - categorical_accuracy: 0.9219
31616/60000 [==============>...............] - ETA: 52s - loss: 0.2566 - categorical_accuracy: 0.9220
31648/60000 [==============>...............] - ETA: 52s - loss: 0.2563 - categorical_accuracy: 0.9220
31680/60000 [==============>...............] - ETA: 52s - loss: 0.2561 - categorical_accuracy: 0.9221
31712/60000 [==============>...............] - ETA: 52s - loss: 0.2560 - categorical_accuracy: 0.9221
31744/60000 [==============>...............] - ETA: 51s - loss: 0.2558 - categorical_accuracy: 0.9222
31776/60000 [==============>...............] - ETA: 51s - loss: 0.2557 - categorical_accuracy: 0.9223
31808/60000 [==============>...............] - ETA: 51s - loss: 0.2558 - categorical_accuracy: 0.9223
31840/60000 [==============>...............] - ETA: 51s - loss: 0.2557 - categorical_accuracy: 0.9223
31872/60000 [==============>...............] - ETA: 51s - loss: 0.2556 - categorical_accuracy: 0.9223
31904/60000 [==============>...............] - ETA: 51s - loss: 0.2556 - categorical_accuracy: 0.9224
31936/60000 [==============>...............] - ETA: 51s - loss: 0.2553 - categorical_accuracy: 0.9224
31968/60000 [==============>...............] - ETA: 51s - loss: 0.2551 - categorical_accuracy: 0.9225
32000/60000 [===============>..............] - ETA: 51s - loss: 0.2550 - categorical_accuracy: 0.9225
32032/60000 [===============>..............] - ETA: 51s - loss: 0.2548 - categorical_accuracy: 0.9226
32064/60000 [===============>..............] - ETA: 51s - loss: 0.2546 - categorical_accuracy: 0.9226
32096/60000 [===============>..............] - ETA: 51s - loss: 0.2544 - categorical_accuracy: 0.9227
32128/60000 [===============>..............] - ETA: 51s - loss: 0.2544 - categorical_accuracy: 0.9227
32160/60000 [===============>..............] - ETA: 51s - loss: 0.2542 - categorical_accuracy: 0.9228
32192/60000 [===============>..............] - ETA: 51s - loss: 0.2542 - categorical_accuracy: 0.9229
32224/60000 [===============>..............] - ETA: 51s - loss: 0.2542 - categorical_accuracy: 0.9229
32256/60000 [===============>..............] - ETA: 51s - loss: 0.2540 - categorical_accuracy: 0.9229
32288/60000 [===============>..............] - ETA: 50s - loss: 0.2541 - categorical_accuracy: 0.9229
32320/60000 [===============>..............] - ETA: 50s - loss: 0.2539 - categorical_accuracy: 0.9230
32352/60000 [===============>..............] - ETA: 50s - loss: 0.2538 - categorical_accuracy: 0.9230
32384/60000 [===============>..............] - ETA: 50s - loss: 0.2536 - categorical_accuracy: 0.9231
32416/60000 [===============>..............] - ETA: 50s - loss: 0.2535 - categorical_accuracy: 0.9231
32448/60000 [===============>..............] - ETA: 50s - loss: 0.2534 - categorical_accuracy: 0.9232
32480/60000 [===============>..............] - ETA: 50s - loss: 0.2534 - categorical_accuracy: 0.9232
32512/60000 [===============>..............] - ETA: 50s - loss: 0.2532 - categorical_accuracy: 0.9233
32544/60000 [===============>..............] - ETA: 50s - loss: 0.2532 - categorical_accuracy: 0.9233
32576/60000 [===============>..............] - ETA: 50s - loss: 0.2531 - categorical_accuracy: 0.9233
32608/60000 [===============>..............] - ETA: 50s - loss: 0.2530 - categorical_accuracy: 0.9233
32640/60000 [===============>..............] - ETA: 50s - loss: 0.2528 - categorical_accuracy: 0.9234
32672/60000 [===============>..............] - ETA: 50s - loss: 0.2526 - categorical_accuracy: 0.9235
32704/60000 [===============>..............] - ETA: 50s - loss: 0.2528 - categorical_accuracy: 0.9234
32736/60000 [===============>..............] - ETA: 50s - loss: 0.2525 - categorical_accuracy: 0.9235
32768/60000 [===============>..............] - ETA: 50s - loss: 0.2523 - categorical_accuracy: 0.9236
32800/60000 [===============>..............] - ETA: 50s - loss: 0.2523 - categorical_accuracy: 0.9236
32832/60000 [===============>..............] - ETA: 49s - loss: 0.2524 - categorical_accuracy: 0.9236
32864/60000 [===============>..............] - ETA: 49s - loss: 0.2522 - categorical_accuracy: 0.9237
32896/60000 [===============>..............] - ETA: 49s - loss: 0.2521 - categorical_accuracy: 0.9237
32928/60000 [===============>..............] - ETA: 49s - loss: 0.2522 - categorical_accuracy: 0.9237
32960/60000 [===============>..............] - ETA: 49s - loss: 0.2521 - categorical_accuracy: 0.9238
32992/60000 [===============>..............] - ETA: 49s - loss: 0.2519 - categorical_accuracy: 0.9238
33024/60000 [===============>..............] - ETA: 49s - loss: 0.2518 - categorical_accuracy: 0.9239
33056/60000 [===============>..............] - ETA: 49s - loss: 0.2516 - categorical_accuracy: 0.9239
33088/60000 [===============>..............] - ETA: 49s - loss: 0.2514 - categorical_accuracy: 0.9240
33120/60000 [===============>..............] - ETA: 49s - loss: 0.2513 - categorical_accuracy: 0.9240
33152/60000 [===============>..............] - ETA: 49s - loss: 0.2512 - categorical_accuracy: 0.9241
33184/60000 [===============>..............] - ETA: 49s - loss: 0.2510 - categorical_accuracy: 0.9241
33216/60000 [===============>..............] - ETA: 49s - loss: 0.2509 - categorical_accuracy: 0.9242
33248/60000 [===============>..............] - ETA: 49s - loss: 0.2508 - categorical_accuracy: 0.9242
33280/60000 [===============>..............] - ETA: 49s - loss: 0.2506 - categorical_accuracy: 0.9242
33312/60000 [===============>..............] - ETA: 49s - loss: 0.2504 - categorical_accuracy: 0.9242
33344/60000 [===============>..............] - ETA: 49s - loss: 0.2503 - categorical_accuracy: 0.9243
33376/60000 [===============>..............] - ETA: 48s - loss: 0.2501 - categorical_accuracy: 0.9243
33408/60000 [===============>..............] - ETA: 48s - loss: 0.2499 - categorical_accuracy: 0.9244
33440/60000 [===============>..............] - ETA: 48s - loss: 0.2499 - categorical_accuracy: 0.9244
33472/60000 [===============>..............] - ETA: 48s - loss: 0.2498 - categorical_accuracy: 0.9245
33504/60000 [===============>..............] - ETA: 48s - loss: 0.2496 - categorical_accuracy: 0.9245
33536/60000 [===============>..............] - ETA: 48s - loss: 0.2496 - categorical_accuracy: 0.9245
33568/60000 [===============>..............] - ETA: 48s - loss: 0.2495 - categorical_accuracy: 0.9246
33600/60000 [===============>..............] - ETA: 48s - loss: 0.2493 - categorical_accuracy: 0.9246
33632/60000 [===============>..............] - ETA: 48s - loss: 0.2491 - categorical_accuracy: 0.9247
33664/60000 [===============>..............] - ETA: 48s - loss: 0.2490 - categorical_accuracy: 0.9247
33696/60000 [===============>..............] - ETA: 48s - loss: 0.2488 - categorical_accuracy: 0.9247
33728/60000 [===============>..............] - ETA: 48s - loss: 0.2486 - categorical_accuracy: 0.9248
33760/60000 [===============>..............] - ETA: 48s - loss: 0.2486 - categorical_accuracy: 0.9248
33792/60000 [===============>..............] - ETA: 48s - loss: 0.2485 - categorical_accuracy: 0.9248
33824/60000 [===============>..............] - ETA: 48s - loss: 0.2484 - categorical_accuracy: 0.9248
33856/60000 [===============>..............] - ETA: 48s - loss: 0.2482 - categorical_accuracy: 0.9249
33888/60000 [===============>..............] - ETA: 48s - loss: 0.2480 - categorical_accuracy: 0.9249
33920/60000 [===============>..............] - ETA: 47s - loss: 0.2480 - categorical_accuracy: 0.9249
33952/60000 [===============>..............] - ETA: 47s - loss: 0.2479 - categorical_accuracy: 0.9250
33984/60000 [===============>..............] - ETA: 47s - loss: 0.2478 - categorical_accuracy: 0.9250
34016/60000 [================>.............] - ETA: 47s - loss: 0.2476 - categorical_accuracy: 0.9250
34048/60000 [================>.............] - ETA: 47s - loss: 0.2476 - categorical_accuracy: 0.9250
34080/60000 [================>.............] - ETA: 47s - loss: 0.2476 - categorical_accuracy: 0.9251
34112/60000 [================>.............] - ETA: 47s - loss: 0.2474 - categorical_accuracy: 0.9251
34144/60000 [================>.............] - ETA: 47s - loss: 0.2474 - categorical_accuracy: 0.9252
34176/60000 [================>.............] - ETA: 47s - loss: 0.2472 - categorical_accuracy: 0.9252
34208/60000 [================>.............] - ETA: 47s - loss: 0.2471 - categorical_accuracy: 0.9253
34240/60000 [================>.............] - ETA: 47s - loss: 0.2470 - categorical_accuracy: 0.9252
34272/60000 [================>.............] - ETA: 47s - loss: 0.2468 - categorical_accuracy: 0.9253
34304/60000 [================>.............] - ETA: 47s - loss: 0.2466 - categorical_accuracy: 0.9254
34336/60000 [================>.............] - ETA: 47s - loss: 0.2465 - categorical_accuracy: 0.9254
34368/60000 [================>.............] - ETA: 47s - loss: 0.2463 - categorical_accuracy: 0.9255
34400/60000 [================>.............] - ETA: 47s - loss: 0.2462 - categorical_accuracy: 0.9255
34432/60000 [================>.............] - ETA: 47s - loss: 0.2460 - categorical_accuracy: 0.9256
34464/60000 [================>.............] - ETA: 46s - loss: 0.2458 - categorical_accuracy: 0.9256
34496/60000 [================>.............] - ETA: 46s - loss: 0.2456 - categorical_accuracy: 0.9257
34528/60000 [================>.............] - ETA: 46s - loss: 0.2457 - categorical_accuracy: 0.9257
34560/60000 [================>.............] - ETA: 46s - loss: 0.2456 - categorical_accuracy: 0.9258
34592/60000 [================>.............] - ETA: 46s - loss: 0.2454 - categorical_accuracy: 0.9258
34624/60000 [================>.............] - ETA: 46s - loss: 0.2456 - categorical_accuracy: 0.9258
34656/60000 [================>.............] - ETA: 46s - loss: 0.2454 - categorical_accuracy: 0.9258
34688/60000 [================>.............] - ETA: 46s - loss: 0.2452 - categorical_accuracy: 0.9259
34720/60000 [================>.............] - ETA: 46s - loss: 0.2450 - categorical_accuracy: 0.9260
34752/60000 [================>.............] - ETA: 46s - loss: 0.2448 - categorical_accuracy: 0.9260
34784/60000 [================>.............] - ETA: 46s - loss: 0.2446 - categorical_accuracy: 0.9261
34816/60000 [================>.............] - ETA: 46s - loss: 0.2444 - categorical_accuracy: 0.9262
34848/60000 [================>.............] - ETA: 46s - loss: 0.2443 - categorical_accuracy: 0.9262
34880/60000 [================>.............] - ETA: 46s - loss: 0.2443 - categorical_accuracy: 0.9262
34912/60000 [================>.............] - ETA: 46s - loss: 0.2442 - categorical_accuracy: 0.9263
34944/60000 [================>.............] - ETA: 46s - loss: 0.2440 - categorical_accuracy: 0.9264
34976/60000 [================>.............] - ETA: 46s - loss: 0.2439 - categorical_accuracy: 0.9264
35008/60000 [================>.............] - ETA: 45s - loss: 0.2439 - categorical_accuracy: 0.9264
35040/60000 [================>.............] - ETA: 45s - loss: 0.2437 - categorical_accuracy: 0.9265
35072/60000 [================>.............] - ETA: 45s - loss: 0.2438 - categorical_accuracy: 0.9265
35104/60000 [================>.............] - ETA: 45s - loss: 0.2436 - categorical_accuracy: 0.9265
35136/60000 [================>.............] - ETA: 45s - loss: 0.2437 - categorical_accuracy: 0.9265
35168/60000 [================>.............] - ETA: 45s - loss: 0.2435 - categorical_accuracy: 0.9266
35200/60000 [================>.............] - ETA: 45s - loss: 0.2435 - categorical_accuracy: 0.9266
35232/60000 [================>.............] - ETA: 45s - loss: 0.2433 - categorical_accuracy: 0.9266
35264/60000 [================>.............] - ETA: 45s - loss: 0.2431 - categorical_accuracy: 0.9267
35296/60000 [================>.............] - ETA: 45s - loss: 0.2430 - categorical_accuracy: 0.9267
35328/60000 [================>.............] - ETA: 45s - loss: 0.2429 - categorical_accuracy: 0.9268
35360/60000 [================>.............] - ETA: 45s - loss: 0.2430 - categorical_accuracy: 0.9268
35392/60000 [================>.............] - ETA: 45s - loss: 0.2429 - categorical_accuracy: 0.9268
35424/60000 [================>.............] - ETA: 45s - loss: 0.2428 - categorical_accuracy: 0.9268
35456/60000 [================>.............] - ETA: 45s - loss: 0.2426 - categorical_accuracy: 0.9269
35488/60000 [================>.............] - ETA: 45s - loss: 0.2424 - categorical_accuracy: 0.9269
35520/60000 [================>.............] - ETA: 45s - loss: 0.2422 - categorical_accuracy: 0.9270
35552/60000 [================>.............] - ETA: 44s - loss: 0.2421 - categorical_accuracy: 0.9270
35584/60000 [================>.............] - ETA: 44s - loss: 0.2419 - categorical_accuracy: 0.9271
35616/60000 [================>.............] - ETA: 44s - loss: 0.2417 - categorical_accuracy: 0.9271
35648/60000 [================>.............] - ETA: 44s - loss: 0.2416 - categorical_accuracy: 0.9271
35680/60000 [================>.............] - ETA: 44s - loss: 0.2414 - categorical_accuracy: 0.9272
35712/60000 [================>.............] - ETA: 44s - loss: 0.2412 - categorical_accuracy: 0.9273
35744/60000 [================>.............] - ETA: 44s - loss: 0.2412 - categorical_accuracy: 0.9273
35776/60000 [================>.............] - ETA: 44s - loss: 0.2410 - categorical_accuracy: 0.9274
35808/60000 [================>.............] - ETA: 44s - loss: 0.2410 - categorical_accuracy: 0.9274
35840/60000 [================>.............] - ETA: 44s - loss: 0.2408 - categorical_accuracy: 0.9274
35872/60000 [================>.............] - ETA: 44s - loss: 0.2406 - categorical_accuracy: 0.9275
35904/60000 [================>.............] - ETA: 44s - loss: 0.2405 - categorical_accuracy: 0.9275
35936/60000 [================>.............] - ETA: 44s - loss: 0.2404 - categorical_accuracy: 0.9276
35968/60000 [================>.............] - ETA: 44s - loss: 0.2403 - categorical_accuracy: 0.9276
36000/60000 [=================>............] - ETA: 44s - loss: 0.2403 - categorical_accuracy: 0.9276
36032/60000 [=================>............] - ETA: 44s - loss: 0.2401 - categorical_accuracy: 0.9276
36064/60000 [=================>............] - ETA: 44s - loss: 0.2400 - categorical_accuracy: 0.9277
36096/60000 [=================>............] - ETA: 43s - loss: 0.2398 - categorical_accuracy: 0.9277
36128/60000 [=================>............] - ETA: 43s - loss: 0.2398 - categorical_accuracy: 0.9278
36160/60000 [=================>............] - ETA: 43s - loss: 0.2397 - categorical_accuracy: 0.9278
36192/60000 [=================>............] - ETA: 43s - loss: 0.2397 - categorical_accuracy: 0.9278
36224/60000 [=================>............] - ETA: 43s - loss: 0.2399 - categorical_accuracy: 0.9278
36256/60000 [=================>............] - ETA: 43s - loss: 0.2397 - categorical_accuracy: 0.9279
36288/60000 [=================>............] - ETA: 43s - loss: 0.2396 - categorical_accuracy: 0.9279
36320/60000 [=================>............] - ETA: 43s - loss: 0.2394 - categorical_accuracy: 0.9280
36352/60000 [=================>............] - ETA: 43s - loss: 0.2392 - categorical_accuracy: 0.9280
36384/60000 [=================>............] - ETA: 43s - loss: 0.2395 - categorical_accuracy: 0.9280
36416/60000 [=================>............] - ETA: 43s - loss: 0.2394 - categorical_accuracy: 0.9280
36448/60000 [=================>............] - ETA: 43s - loss: 0.2392 - categorical_accuracy: 0.9281
36480/60000 [=================>............] - ETA: 43s - loss: 0.2391 - categorical_accuracy: 0.9281
36512/60000 [=================>............] - ETA: 43s - loss: 0.2389 - categorical_accuracy: 0.9281
36544/60000 [=================>............] - ETA: 43s - loss: 0.2390 - categorical_accuracy: 0.9281
36576/60000 [=================>............] - ETA: 43s - loss: 0.2388 - categorical_accuracy: 0.9282
36608/60000 [=================>............] - ETA: 43s - loss: 0.2388 - categorical_accuracy: 0.9282
36640/60000 [=================>............] - ETA: 42s - loss: 0.2387 - categorical_accuracy: 0.9282
36672/60000 [=================>............] - ETA: 42s - loss: 0.2385 - categorical_accuracy: 0.9283
36704/60000 [=================>............] - ETA: 42s - loss: 0.2384 - categorical_accuracy: 0.9283
36736/60000 [=================>............] - ETA: 42s - loss: 0.2384 - categorical_accuracy: 0.9283
36768/60000 [=================>............] - ETA: 42s - loss: 0.2382 - categorical_accuracy: 0.9283
36800/60000 [=================>............] - ETA: 42s - loss: 0.2380 - categorical_accuracy: 0.9284
36832/60000 [=================>............] - ETA: 42s - loss: 0.2379 - categorical_accuracy: 0.9284
36864/60000 [=================>............] - ETA: 42s - loss: 0.2377 - categorical_accuracy: 0.9285
36896/60000 [=================>............] - ETA: 42s - loss: 0.2376 - categorical_accuracy: 0.9285
36928/60000 [=================>............] - ETA: 42s - loss: 0.2375 - categorical_accuracy: 0.9285
36960/60000 [=================>............] - ETA: 42s - loss: 0.2373 - categorical_accuracy: 0.9286
36992/60000 [=================>............] - ETA: 42s - loss: 0.2373 - categorical_accuracy: 0.9286
37024/60000 [=================>............] - ETA: 42s - loss: 0.2371 - categorical_accuracy: 0.9287
37056/60000 [=================>............] - ETA: 42s - loss: 0.2373 - categorical_accuracy: 0.9287
37088/60000 [=================>............] - ETA: 42s - loss: 0.2374 - categorical_accuracy: 0.9287
37120/60000 [=================>............] - ETA: 42s - loss: 0.2372 - categorical_accuracy: 0.9287
37152/60000 [=================>............] - ETA: 42s - loss: 0.2371 - categorical_accuracy: 0.9287
37184/60000 [=================>............] - ETA: 41s - loss: 0.2370 - categorical_accuracy: 0.9288
37216/60000 [=================>............] - ETA: 41s - loss: 0.2369 - categorical_accuracy: 0.9288
37248/60000 [=================>............] - ETA: 41s - loss: 0.2369 - categorical_accuracy: 0.9288
37280/60000 [=================>............] - ETA: 41s - loss: 0.2368 - categorical_accuracy: 0.9288
37312/60000 [=================>............] - ETA: 41s - loss: 0.2367 - categorical_accuracy: 0.9289
37344/60000 [=================>............] - ETA: 41s - loss: 0.2367 - categorical_accuracy: 0.9289
37376/60000 [=================>............] - ETA: 41s - loss: 0.2367 - categorical_accuracy: 0.9289
37408/60000 [=================>............] - ETA: 41s - loss: 0.2366 - categorical_accuracy: 0.9289
37440/60000 [=================>............] - ETA: 41s - loss: 0.2365 - categorical_accuracy: 0.9289
37472/60000 [=================>............] - ETA: 41s - loss: 0.2364 - categorical_accuracy: 0.9290
37504/60000 [=================>............] - ETA: 41s - loss: 0.2364 - categorical_accuracy: 0.9289
37536/60000 [=================>............] - ETA: 41s - loss: 0.2363 - categorical_accuracy: 0.9290
37568/60000 [=================>............] - ETA: 41s - loss: 0.2362 - categorical_accuracy: 0.9290
37600/60000 [=================>............] - ETA: 41s - loss: 0.2361 - categorical_accuracy: 0.9290
37632/60000 [=================>............] - ETA: 41s - loss: 0.2360 - categorical_accuracy: 0.9290
37664/60000 [=================>............] - ETA: 41s - loss: 0.2359 - categorical_accuracy: 0.9291
37696/60000 [=================>............] - ETA: 41s - loss: 0.2360 - categorical_accuracy: 0.9291
37728/60000 [=================>............] - ETA: 40s - loss: 0.2360 - categorical_accuracy: 0.9291
37760/60000 [=================>............] - ETA: 40s - loss: 0.2358 - categorical_accuracy: 0.9291
37792/60000 [=================>............] - ETA: 40s - loss: 0.2357 - categorical_accuracy: 0.9292
37824/60000 [=================>............] - ETA: 40s - loss: 0.2357 - categorical_accuracy: 0.9292
37856/60000 [=================>............] - ETA: 40s - loss: 0.2355 - categorical_accuracy: 0.9292
37888/60000 [=================>............] - ETA: 40s - loss: 0.2354 - categorical_accuracy: 0.9293
37920/60000 [=================>............] - ETA: 40s - loss: 0.2355 - categorical_accuracy: 0.9292
37952/60000 [=================>............] - ETA: 40s - loss: 0.2353 - categorical_accuracy: 0.9293
37984/60000 [=================>............] - ETA: 40s - loss: 0.2352 - categorical_accuracy: 0.9293
38016/60000 [==================>...........] - ETA: 40s - loss: 0.2352 - categorical_accuracy: 0.9293
38048/60000 [==================>...........] - ETA: 40s - loss: 0.2351 - categorical_accuracy: 0.9294
38080/60000 [==================>...........] - ETA: 40s - loss: 0.2351 - categorical_accuracy: 0.9294
38112/60000 [==================>...........] - ETA: 40s - loss: 0.2350 - categorical_accuracy: 0.9294
38144/60000 [==================>...........] - ETA: 40s - loss: 0.2348 - categorical_accuracy: 0.9295
38176/60000 [==================>...........] - ETA: 40s - loss: 0.2346 - categorical_accuracy: 0.9295
38208/60000 [==================>...........] - ETA: 40s - loss: 0.2346 - categorical_accuracy: 0.9295
38240/60000 [==================>...........] - ETA: 40s - loss: 0.2345 - categorical_accuracy: 0.9295
38272/60000 [==================>...........] - ETA: 39s - loss: 0.2345 - categorical_accuracy: 0.9295
38304/60000 [==================>...........] - ETA: 39s - loss: 0.2344 - categorical_accuracy: 0.9296
38336/60000 [==================>...........] - ETA: 39s - loss: 0.2344 - categorical_accuracy: 0.9296
38368/60000 [==================>...........] - ETA: 39s - loss: 0.2342 - categorical_accuracy: 0.9297
38400/60000 [==================>...........] - ETA: 39s - loss: 0.2342 - categorical_accuracy: 0.9297
38432/60000 [==================>...........] - ETA: 39s - loss: 0.2340 - categorical_accuracy: 0.9297
38464/60000 [==================>...........] - ETA: 39s - loss: 0.2339 - categorical_accuracy: 0.9297
38496/60000 [==================>...........] - ETA: 39s - loss: 0.2337 - categorical_accuracy: 0.9298
38528/60000 [==================>...........] - ETA: 39s - loss: 0.2336 - categorical_accuracy: 0.9298
38560/60000 [==================>...........] - ETA: 39s - loss: 0.2337 - categorical_accuracy: 0.9298
38592/60000 [==================>...........] - ETA: 39s - loss: 0.2336 - categorical_accuracy: 0.9298
38624/60000 [==================>...........] - ETA: 39s - loss: 0.2335 - categorical_accuracy: 0.9298
38656/60000 [==================>...........] - ETA: 39s - loss: 0.2333 - categorical_accuracy: 0.9299
38688/60000 [==================>...........] - ETA: 39s - loss: 0.2332 - categorical_accuracy: 0.9299
38720/60000 [==================>...........] - ETA: 39s - loss: 0.2333 - categorical_accuracy: 0.9299
38752/60000 [==================>...........] - ETA: 39s - loss: 0.2335 - categorical_accuracy: 0.9299
38784/60000 [==================>...........] - ETA: 39s - loss: 0.2334 - categorical_accuracy: 0.9299
38816/60000 [==================>...........] - ETA: 38s - loss: 0.2332 - categorical_accuracy: 0.9299
38848/60000 [==================>...........] - ETA: 38s - loss: 0.2331 - categorical_accuracy: 0.9300
38880/60000 [==================>...........] - ETA: 38s - loss: 0.2331 - categorical_accuracy: 0.9300
38912/60000 [==================>...........] - ETA: 38s - loss: 0.2329 - categorical_accuracy: 0.9300
38944/60000 [==================>...........] - ETA: 38s - loss: 0.2328 - categorical_accuracy: 0.9301
38976/60000 [==================>...........] - ETA: 38s - loss: 0.2327 - categorical_accuracy: 0.9301
39008/60000 [==================>...........] - ETA: 38s - loss: 0.2325 - categorical_accuracy: 0.9301
39040/60000 [==================>...........] - ETA: 38s - loss: 0.2324 - categorical_accuracy: 0.9302
39072/60000 [==================>...........] - ETA: 38s - loss: 0.2323 - categorical_accuracy: 0.9302
39104/60000 [==================>...........] - ETA: 38s - loss: 0.2322 - categorical_accuracy: 0.9302
39136/60000 [==================>...........] - ETA: 38s - loss: 0.2322 - categorical_accuracy: 0.9302
39168/60000 [==================>...........] - ETA: 38s - loss: 0.2321 - categorical_accuracy: 0.9303
39200/60000 [==================>...........] - ETA: 38s - loss: 0.2320 - categorical_accuracy: 0.9303
39232/60000 [==================>...........] - ETA: 38s - loss: 0.2319 - categorical_accuracy: 0.9303
39264/60000 [==================>...........] - ETA: 38s - loss: 0.2319 - categorical_accuracy: 0.9303
39296/60000 [==================>...........] - ETA: 38s - loss: 0.2318 - categorical_accuracy: 0.9303
39328/60000 [==================>...........] - ETA: 37s - loss: 0.2316 - categorical_accuracy: 0.9304
39360/60000 [==================>...........] - ETA: 37s - loss: 0.2315 - categorical_accuracy: 0.9304
39392/60000 [==================>...........] - ETA: 37s - loss: 0.2313 - categorical_accuracy: 0.9305
39424/60000 [==================>...........] - ETA: 37s - loss: 0.2312 - categorical_accuracy: 0.9305
39456/60000 [==================>...........] - ETA: 37s - loss: 0.2311 - categorical_accuracy: 0.9305
39488/60000 [==================>...........] - ETA: 37s - loss: 0.2310 - categorical_accuracy: 0.9306
39520/60000 [==================>...........] - ETA: 37s - loss: 0.2308 - categorical_accuracy: 0.9306
39552/60000 [==================>...........] - ETA: 37s - loss: 0.2307 - categorical_accuracy: 0.9306
39584/60000 [==================>...........] - ETA: 37s - loss: 0.2306 - categorical_accuracy: 0.9307
39616/60000 [==================>...........] - ETA: 37s - loss: 0.2304 - categorical_accuracy: 0.9307
39648/60000 [==================>...........] - ETA: 37s - loss: 0.2304 - categorical_accuracy: 0.9307
39680/60000 [==================>...........] - ETA: 37s - loss: 0.2303 - categorical_accuracy: 0.9307
39712/60000 [==================>...........] - ETA: 37s - loss: 0.2302 - categorical_accuracy: 0.9308
39744/60000 [==================>...........] - ETA: 37s - loss: 0.2301 - categorical_accuracy: 0.9308
39776/60000 [==================>...........] - ETA: 37s - loss: 0.2300 - categorical_accuracy: 0.9308
39808/60000 [==================>...........] - ETA: 37s - loss: 0.2299 - categorical_accuracy: 0.9308
39840/60000 [==================>...........] - ETA: 37s - loss: 0.2298 - categorical_accuracy: 0.9309
39872/60000 [==================>...........] - ETA: 36s - loss: 0.2298 - categorical_accuracy: 0.9309
39904/60000 [==================>...........] - ETA: 36s - loss: 0.2296 - categorical_accuracy: 0.9309
39936/60000 [==================>...........] - ETA: 36s - loss: 0.2295 - categorical_accuracy: 0.9310
39968/60000 [==================>...........] - ETA: 36s - loss: 0.2294 - categorical_accuracy: 0.9310
40000/60000 [===================>..........] - ETA: 36s - loss: 0.2294 - categorical_accuracy: 0.9310
40032/60000 [===================>..........] - ETA: 36s - loss: 0.2293 - categorical_accuracy: 0.9310
40064/60000 [===================>..........] - ETA: 36s - loss: 0.2292 - categorical_accuracy: 0.9311
40096/60000 [===================>..........] - ETA: 36s - loss: 0.2293 - categorical_accuracy: 0.9311
40128/60000 [===================>..........] - ETA: 36s - loss: 0.2293 - categorical_accuracy: 0.9310
40160/60000 [===================>..........] - ETA: 36s - loss: 0.2292 - categorical_accuracy: 0.9311
40192/60000 [===================>..........] - ETA: 36s - loss: 0.2292 - categorical_accuracy: 0.9311
40224/60000 [===================>..........] - ETA: 36s - loss: 0.2291 - categorical_accuracy: 0.9311
40256/60000 [===================>..........] - ETA: 36s - loss: 0.2289 - categorical_accuracy: 0.9312
40288/60000 [===================>..........] - ETA: 36s - loss: 0.2288 - categorical_accuracy: 0.9312
40320/60000 [===================>..........] - ETA: 36s - loss: 0.2286 - categorical_accuracy: 0.9313
40352/60000 [===================>..........] - ETA: 36s - loss: 0.2284 - categorical_accuracy: 0.9313
40384/60000 [===================>..........] - ETA: 36s - loss: 0.2283 - categorical_accuracy: 0.9313
40416/60000 [===================>..........] - ETA: 35s - loss: 0.2282 - categorical_accuracy: 0.9314
40448/60000 [===================>..........] - ETA: 35s - loss: 0.2280 - categorical_accuracy: 0.9314
40480/60000 [===================>..........] - ETA: 35s - loss: 0.2279 - categorical_accuracy: 0.9315
40512/60000 [===================>..........] - ETA: 35s - loss: 0.2279 - categorical_accuracy: 0.9315
40544/60000 [===================>..........] - ETA: 35s - loss: 0.2280 - categorical_accuracy: 0.9315
40576/60000 [===================>..........] - ETA: 35s - loss: 0.2280 - categorical_accuracy: 0.9314
40608/60000 [===================>..........] - ETA: 35s - loss: 0.2279 - categorical_accuracy: 0.9314
40640/60000 [===================>..........] - ETA: 35s - loss: 0.2278 - categorical_accuracy: 0.9314
40672/60000 [===================>..........] - ETA: 35s - loss: 0.2278 - categorical_accuracy: 0.9315
40704/60000 [===================>..........] - ETA: 35s - loss: 0.2276 - categorical_accuracy: 0.9315
40736/60000 [===================>..........] - ETA: 35s - loss: 0.2275 - categorical_accuracy: 0.9316
40768/60000 [===================>..........] - ETA: 35s - loss: 0.2274 - categorical_accuracy: 0.9316
40800/60000 [===================>..........] - ETA: 35s - loss: 0.2273 - categorical_accuracy: 0.9316
40832/60000 [===================>..........] - ETA: 35s - loss: 0.2272 - categorical_accuracy: 0.9316
40864/60000 [===================>..........] - ETA: 35s - loss: 0.2270 - categorical_accuracy: 0.9317
40896/60000 [===================>..........] - ETA: 35s - loss: 0.2270 - categorical_accuracy: 0.9317
40928/60000 [===================>..........] - ETA: 35s - loss: 0.2268 - categorical_accuracy: 0.9317
40960/60000 [===================>..........] - ETA: 34s - loss: 0.2267 - categorical_accuracy: 0.9317
40992/60000 [===================>..........] - ETA: 34s - loss: 0.2266 - categorical_accuracy: 0.9317
41024/60000 [===================>..........] - ETA: 34s - loss: 0.2265 - categorical_accuracy: 0.9318
41056/60000 [===================>..........] - ETA: 34s - loss: 0.2265 - categorical_accuracy: 0.9318
41088/60000 [===================>..........] - ETA: 34s - loss: 0.2265 - categorical_accuracy: 0.9318
41120/60000 [===================>..........] - ETA: 34s - loss: 0.2263 - categorical_accuracy: 0.9318
41152/60000 [===================>..........] - ETA: 34s - loss: 0.2264 - categorical_accuracy: 0.9318
41184/60000 [===================>..........] - ETA: 34s - loss: 0.2263 - categorical_accuracy: 0.9319
41216/60000 [===================>..........] - ETA: 34s - loss: 0.2261 - categorical_accuracy: 0.9319
41248/60000 [===================>..........] - ETA: 34s - loss: 0.2260 - categorical_accuracy: 0.9320
41280/60000 [===================>..........] - ETA: 34s - loss: 0.2259 - categorical_accuracy: 0.9320
41312/60000 [===================>..........] - ETA: 34s - loss: 0.2257 - categorical_accuracy: 0.9320
41344/60000 [===================>..........] - ETA: 34s - loss: 0.2256 - categorical_accuracy: 0.9321
41376/60000 [===================>..........] - ETA: 34s - loss: 0.2256 - categorical_accuracy: 0.9321
41408/60000 [===================>..........] - ETA: 34s - loss: 0.2254 - categorical_accuracy: 0.9321
41440/60000 [===================>..........] - ETA: 34s - loss: 0.2253 - categorical_accuracy: 0.9322
41472/60000 [===================>..........] - ETA: 34s - loss: 0.2253 - categorical_accuracy: 0.9321
41504/60000 [===================>..........] - ETA: 33s - loss: 0.2254 - categorical_accuracy: 0.9321
41536/60000 [===================>..........] - ETA: 33s - loss: 0.2254 - categorical_accuracy: 0.9321
41568/60000 [===================>..........] - ETA: 33s - loss: 0.2253 - categorical_accuracy: 0.9321
41600/60000 [===================>..........] - ETA: 33s - loss: 0.2252 - categorical_accuracy: 0.9322
41632/60000 [===================>..........] - ETA: 33s - loss: 0.2251 - categorical_accuracy: 0.9322
41664/60000 [===================>..........] - ETA: 33s - loss: 0.2250 - categorical_accuracy: 0.9322
41696/60000 [===================>..........] - ETA: 33s - loss: 0.2250 - categorical_accuracy: 0.9322
41728/60000 [===================>..........] - ETA: 33s - loss: 0.2249 - categorical_accuracy: 0.9323
41760/60000 [===================>..........] - ETA: 33s - loss: 0.2249 - categorical_accuracy: 0.9323
41792/60000 [===================>..........] - ETA: 33s - loss: 0.2249 - categorical_accuracy: 0.9323
41824/60000 [===================>..........] - ETA: 33s - loss: 0.2247 - categorical_accuracy: 0.9323
41856/60000 [===================>..........] - ETA: 33s - loss: 0.2246 - categorical_accuracy: 0.9324
41888/60000 [===================>..........] - ETA: 33s - loss: 0.2248 - categorical_accuracy: 0.9323
41920/60000 [===================>..........] - ETA: 33s - loss: 0.2248 - categorical_accuracy: 0.9323
41952/60000 [===================>..........] - ETA: 33s - loss: 0.2247 - categorical_accuracy: 0.9324
41984/60000 [===================>..........] - ETA: 33s - loss: 0.2247 - categorical_accuracy: 0.9323
42016/60000 [====================>.........] - ETA: 33s - loss: 0.2245 - categorical_accuracy: 0.9324
42048/60000 [====================>.........] - ETA: 32s - loss: 0.2244 - categorical_accuracy: 0.9324
42080/60000 [====================>.........] - ETA: 32s - loss: 0.2243 - categorical_accuracy: 0.9324
42112/60000 [====================>.........] - ETA: 32s - loss: 0.2242 - categorical_accuracy: 0.9324
42144/60000 [====================>.........] - ETA: 32s - loss: 0.2240 - categorical_accuracy: 0.9325
42176/60000 [====================>.........] - ETA: 32s - loss: 0.2239 - categorical_accuracy: 0.9325
42208/60000 [====================>.........] - ETA: 32s - loss: 0.2239 - categorical_accuracy: 0.9325
42240/60000 [====================>.........] - ETA: 32s - loss: 0.2238 - categorical_accuracy: 0.9325
42272/60000 [====================>.........] - ETA: 32s - loss: 0.2238 - categorical_accuracy: 0.9325
42304/60000 [====================>.........] - ETA: 32s - loss: 0.2236 - categorical_accuracy: 0.9325
42336/60000 [====================>.........] - ETA: 32s - loss: 0.2235 - categorical_accuracy: 0.9326
42368/60000 [====================>.........] - ETA: 32s - loss: 0.2234 - categorical_accuracy: 0.9326
42400/60000 [====================>.........] - ETA: 32s - loss: 0.2233 - categorical_accuracy: 0.9326
42432/60000 [====================>.........] - ETA: 32s - loss: 0.2234 - categorical_accuracy: 0.9326
42464/60000 [====================>.........] - ETA: 32s - loss: 0.2233 - categorical_accuracy: 0.9326
42496/60000 [====================>.........] - ETA: 32s - loss: 0.2232 - categorical_accuracy: 0.9327
42528/60000 [====================>.........] - ETA: 32s - loss: 0.2231 - categorical_accuracy: 0.9327
42560/60000 [====================>.........] - ETA: 32s - loss: 0.2230 - categorical_accuracy: 0.9327
42592/60000 [====================>.........] - ETA: 31s - loss: 0.2229 - categorical_accuracy: 0.9328
42624/60000 [====================>.........] - ETA: 31s - loss: 0.2230 - categorical_accuracy: 0.9327
42656/60000 [====================>.........] - ETA: 31s - loss: 0.2228 - categorical_accuracy: 0.9328
42688/60000 [====================>.........] - ETA: 31s - loss: 0.2227 - categorical_accuracy: 0.9328
42720/60000 [====================>.........] - ETA: 31s - loss: 0.2227 - categorical_accuracy: 0.9328
42752/60000 [====================>.........] - ETA: 31s - loss: 0.2226 - categorical_accuracy: 0.9328
42784/60000 [====================>.........] - ETA: 31s - loss: 0.2225 - categorical_accuracy: 0.9328
42816/60000 [====================>.........] - ETA: 31s - loss: 0.2224 - categorical_accuracy: 0.9329
42848/60000 [====================>.........] - ETA: 31s - loss: 0.2223 - categorical_accuracy: 0.9329
42880/60000 [====================>.........] - ETA: 31s - loss: 0.2224 - categorical_accuracy: 0.9329
42912/60000 [====================>.........] - ETA: 31s - loss: 0.2223 - categorical_accuracy: 0.9329
42944/60000 [====================>.........] - ETA: 31s - loss: 0.2221 - categorical_accuracy: 0.9330
42976/60000 [====================>.........] - ETA: 31s - loss: 0.2220 - categorical_accuracy: 0.9330
43008/60000 [====================>.........] - ETA: 31s - loss: 0.2220 - categorical_accuracy: 0.9330
43040/60000 [====================>.........] - ETA: 31s - loss: 0.2219 - categorical_accuracy: 0.9330
43072/60000 [====================>.........] - ETA: 31s - loss: 0.2218 - categorical_accuracy: 0.9330
43104/60000 [====================>.........] - ETA: 31s - loss: 0.2217 - categorical_accuracy: 0.9331
43136/60000 [====================>.........] - ETA: 30s - loss: 0.2216 - categorical_accuracy: 0.9331
43168/60000 [====================>.........] - ETA: 30s - loss: 0.2214 - categorical_accuracy: 0.9332
43200/60000 [====================>.........] - ETA: 30s - loss: 0.2214 - categorical_accuracy: 0.9332
43232/60000 [====================>.........] - ETA: 30s - loss: 0.2212 - categorical_accuracy: 0.9333
43264/60000 [====================>.........] - ETA: 30s - loss: 0.2212 - categorical_accuracy: 0.9333
43296/60000 [====================>.........] - ETA: 30s - loss: 0.2211 - categorical_accuracy: 0.9333
43328/60000 [====================>.........] - ETA: 30s - loss: 0.2210 - categorical_accuracy: 0.9333
43360/60000 [====================>.........] - ETA: 30s - loss: 0.2208 - categorical_accuracy: 0.9333
43392/60000 [====================>.........] - ETA: 30s - loss: 0.2209 - categorical_accuracy: 0.9333
43424/60000 [====================>.........] - ETA: 30s - loss: 0.2208 - categorical_accuracy: 0.9333
43456/60000 [====================>.........] - ETA: 30s - loss: 0.2206 - categorical_accuracy: 0.9334
43488/60000 [====================>.........] - ETA: 30s - loss: 0.2205 - categorical_accuracy: 0.9334
43520/60000 [====================>.........] - ETA: 30s - loss: 0.2203 - categorical_accuracy: 0.9335
43552/60000 [====================>.........] - ETA: 30s - loss: 0.2202 - categorical_accuracy: 0.9335
43584/60000 [====================>.........] - ETA: 30s - loss: 0.2201 - categorical_accuracy: 0.9336
43616/60000 [====================>.........] - ETA: 30s - loss: 0.2199 - categorical_accuracy: 0.9336
43648/60000 [====================>.........] - ETA: 30s - loss: 0.2199 - categorical_accuracy: 0.9336
43680/60000 [====================>.........] - ETA: 29s - loss: 0.2198 - categorical_accuracy: 0.9336
43712/60000 [====================>.........] - ETA: 29s - loss: 0.2198 - categorical_accuracy: 0.9336
43744/60000 [====================>.........] - ETA: 29s - loss: 0.2196 - categorical_accuracy: 0.9336
43776/60000 [====================>.........] - ETA: 29s - loss: 0.2195 - categorical_accuracy: 0.9337
43808/60000 [====================>.........] - ETA: 29s - loss: 0.2194 - categorical_accuracy: 0.9337
43840/60000 [====================>.........] - ETA: 29s - loss: 0.2193 - categorical_accuracy: 0.9337
43872/60000 [====================>.........] - ETA: 29s - loss: 0.2191 - categorical_accuracy: 0.9338
43904/60000 [====================>.........] - ETA: 29s - loss: 0.2190 - categorical_accuracy: 0.9338
43936/60000 [====================>.........] - ETA: 29s - loss: 0.2188 - categorical_accuracy: 0.9339
43968/60000 [====================>.........] - ETA: 29s - loss: 0.2188 - categorical_accuracy: 0.9339
44000/60000 [=====================>........] - ETA: 29s - loss: 0.2187 - categorical_accuracy: 0.9339
44032/60000 [=====================>........] - ETA: 29s - loss: 0.2185 - categorical_accuracy: 0.9340
44064/60000 [=====================>........] - ETA: 29s - loss: 0.2184 - categorical_accuracy: 0.9340
44096/60000 [=====================>........] - ETA: 29s - loss: 0.2184 - categorical_accuracy: 0.9340
44128/60000 [=====================>........] - ETA: 29s - loss: 0.2184 - categorical_accuracy: 0.9340
44160/60000 [=====================>........] - ETA: 29s - loss: 0.2183 - categorical_accuracy: 0.9341
44192/60000 [=====================>........] - ETA: 29s - loss: 0.2181 - categorical_accuracy: 0.9341
44224/60000 [=====================>........] - ETA: 28s - loss: 0.2180 - categorical_accuracy: 0.9342
44256/60000 [=====================>........] - ETA: 28s - loss: 0.2179 - categorical_accuracy: 0.9342
44288/60000 [=====================>........] - ETA: 28s - loss: 0.2178 - categorical_accuracy: 0.9342
44320/60000 [=====================>........] - ETA: 28s - loss: 0.2177 - categorical_accuracy: 0.9343
44352/60000 [=====================>........] - ETA: 28s - loss: 0.2176 - categorical_accuracy: 0.9343
44384/60000 [=====================>........] - ETA: 28s - loss: 0.2174 - categorical_accuracy: 0.9343
44416/60000 [=====================>........] - ETA: 28s - loss: 0.2173 - categorical_accuracy: 0.9343
44448/60000 [=====================>........] - ETA: 28s - loss: 0.2173 - categorical_accuracy: 0.9343
44480/60000 [=====================>........] - ETA: 28s - loss: 0.2172 - categorical_accuracy: 0.9344
44512/60000 [=====================>........] - ETA: 28s - loss: 0.2170 - categorical_accuracy: 0.9344
44544/60000 [=====================>........] - ETA: 28s - loss: 0.2170 - categorical_accuracy: 0.9344
44576/60000 [=====================>........] - ETA: 28s - loss: 0.2169 - categorical_accuracy: 0.9344
44608/60000 [=====================>........] - ETA: 28s - loss: 0.2167 - categorical_accuracy: 0.9345
44640/60000 [=====================>........] - ETA: 28s - loss: 0.2167 - categorical_accuracy: 0.9345
44672/60000 [=====================>........] - ETA: 28s - loss: 0.2165 - categorical_accuracy: 0.9345
44704/60000 [=====================>........] - ETA: 28s - loss: 0.2164 - categorical_accuracy: 0.9346
44736/60000 [=====================>........] - ETA: 27s - loss: 0.2163 - categorical_accuracy: 0.9346
44768/60000 [=====================>........] - ETA: 27s - loss: 0.2162 - categorical_accuracy: 0.9347
44800/60000 [=====================>........] - ETA: 27s - loss: 0.2161 - categorical_accuracy: 0.9347
44832/60000 [=====================>........] - ETA: 27s - loss: 0.2160 - categorical_accuracy: 0.9347
44864/60000 [=====================>........] - ETA: 27s - loss: 0.2159 - categorical_accuracy: 0.9348
44896/60000 [=====================>........] - ETA: 27s - loss: 0.2157 - categorical_accuracy: 0.9348
44928/60000 [=====================>........] - ETA: 27s - loss: 0.2156 - categorical_accuracy: 0.9349
44960/60000 [=====================>........] - ETA: 27s - loss: 0.2155 - categorical_accuracy: 0.9349
44992/60000 [=====================>........] - ETA: 27s - loss: 0.2154 - categorical_accuracy: 0.9349
45024/60000 [=====================>........] - ETA: 27s - loss: 0.2153 - categorical_accuracy: 0.9349
45056/60000 [=====================>........] - ETA: 27s - loss: 0.2152 - categorical_accuracy: 0.9350
45088/60000 [=====================>........] - ETA: 27s - loss: 0.2150 - categorical_accuracy: 0.9350
45120/60000 [=====================>........] - ETA: 27s - loss: 0.2149 - categorical_accuracy: 0.9351
45152/60000 [=====================>........] - ETA: 27s - loss: 0.2149 - categorical_accuracy: 0.9351
45184/60000 [=====================>........] - ETA: 27s - loss: 0.2147 - categorical_accuracy: 0.9351
45216/60000 [=====================>........] - ETA: 27s - loss: 0.2146 - categorical_accuracy: 0.9352
45248/60000 [=====================>........] - ETA: 27s - loss: 0.2144 - categorical_accuracy: 0.9352
45280/60000 [=====================>........] - ETA: 26s - loss: 0.2144 - categorical_accuracy: 0.9352
45312/60000 [=====================>........] - ETA: 26s - loss: 0.2143 - categorical_accuracy: 0.9353
45344/60000 [=====================>........] - ETA: 26s - loss: 0.2143 - categorical_accuracy: 0.9353
45376/60000 [=====================>........] - ETA: 26s - loss: 0.2142 - categorical_accuracy: 0.9353
45408/60000 [=====================>........] - ETA: 26s - loss: 0.2142 - categorical_accuracy: 0.9353
45440/60000 [=====================>........] - ETA: 26s - loss: 0.2141 - categorical_accuracy: 0.9353
45472/60000 [=====================>........] - ETA: 26s - loss: 0.2140 - categorical_accuracy: 0.9354
45504/60000 [=====================>........] - ETA: 26s - loss: 0.2138 - categorical_accuracy: 0.9354
45536/60000 [=====================>........] - ETA: 26s - loss: 0.2137 - categorical_accuracy: 0.9355
45568/60000 [=====================>........] - ETA: 26s - loss: 0.2136 - categorical_accuracy: 0.9355
45600/60000 [=====================>........] - ETA: 26s - loss: 0.2135 - categorical_accuracy: 0.9355
45632/60000 [=====================>........] - ETA: 26s - loss: 0.2134 - categorical_accuracy: 0.9355
45664/60000 [=====================>........] - ETA: 26s - loss: 0.2134 - categorical_accuracy: 0.9356
45696/60000 [=====================>........] - ETA: 26s - loss: 0.2133 - categorical_accuracy: 0.9356
45728/60000 [=====================>........] - ETA: 26s - loss: 0.2132 - categorical_accuracy: 0.9356
45760/60000 [=====================>........] - ETA: 26s - loss: 0.2131 - categorical_accuracy: 0.9357
45792/60000 [=====================>........] - ETA: 26s - loss: 0.2130 - categorical_accuracy: 0.9357
45824/60000 [=====================>........] - ETA: 25s - loss: 0.2130 - categorical_accuracy: 0.9357
45856/60000 [=====================>........] - ETA: 25s - loss: 0.2131 - categorical_accuracy: 0.9357
45888/60000 [=====================>........] - ETA: 25s - loss: 0.2132 - categorical_accuracy: 0.9357
45920/60000 [=====================>........] - ETA: 25s - loss: 0.2131 - categorical_accuracy: 0.9357
45952/60000 [=====================>........] - ETA: 25s - loss: 0.2130 - categorical_accuracy: 0.9357
45984/60000 [=====================>........] - ETA: 25s - loss: 0.2129 - categorical_accuracy: 0.9358
46016/60000 [======================>.......] - ETA: 25s - loss: 0.2128 - categorical_accuracy: 0.9358
46048/60000 [======================>.......] - ETA: 25s - loss: 0.2128 - categorical_accuracy: 0.9358
46080/60000 [======================>.......] - ETA: 25s - loss: 0.2128 - categorical_accuracy: 0.9358
46112/60000 [======================>.......] - ETA: 25s - loss: 0.2127 - categorical_accuracy: 0.9358
46144/60000 [======================>.......] - ETA: 25s - loss: 0.2126 - categorical_accuracy: 0.9359
46176/60000 [======================>.......] - ETA: 25s - loss: 0.2125 - categorical_accuracy: 0.9359
46208/60000 [======================>.......] - ETA: 25s - loss: 0.2124 - categorical_accuracy: 0.9359
46240/60000 [======================>.......] - ETA: 25s - loss: 0.2126 - categorical_accuracy: 0.9359
46272/60000 [======================>.......] - ETA: 25s - loss: 0.2126 - categorical_accuracy: 0.9359
46304/60000 [======================>.......] - ETA: 25s - loss: 0.2125 - categorical_accuracy: 0.9359
46336/60000 [======================>.......] - ETA: 25s - loss: 0.2124 - categorical_accuracy: 0.9359
46368/60000 [======================>.......] - ETA: 24s - loss: 0.2123 - categorical_accuracy: 0.9360
46400/60000 [======================>.......] - ETA: 24s - loss: 0.2122 - categorical_accuracy: 0.9360
46432/60000 [======================>.......] - ETA: 24s - loss: 0.2121 - categorical_accuracy: 0.9361
46464/60000 [======================>.......] - ETA: 24s - loss: 0.2119 - categorical_accuracy: 0.9361
46496/60000 [======================>.......] - ETA: 24s - loss: 0.2118 - categorical_accuracy: 0.9361
46528/60000 [======================>.......] - ETA: 24s - loss: 0.2117 - categorical_accuracy: 0.9362
46560/60000 [======================>.......] - ETA: 24s - loss: 0.2115 - categorical_accuracy: 0.9362
46592/60000 [======================>.......] - ETA: 24s - loss: 0.2117 - categorical_accuracy: 0.9362
46624/60000 [======================>.......] - ETA: 24s - loss: 0.2116 - categorical_accuracy: 0.9363
46656/60000 [======================>.......] - ETA: 24s - loss: 0.2116 - categorical_accuracy: 0.9363
46688/60000 [======================>.......] - ETA: 24s - loss: 0.2115 - categorical_accuracy: 0.9363
46720/60000 [======================>.......] - ETA: 24s - loss: 0.2114 - categorical_accuracy: 0.9364
46752/60000 [======================>.......] - ETA: 24s - loss: 0.2112 - categorical_accuracy: 0.9364
46784/60000 [======================>.......] - ETA: 24s - loss: 0.2111 - categorical_accuracy: 0.9364
46816/60000 [======================>.......] - ETA: 24s - loss: 0.2110 - categorical_accuracy: 0.9365
46848/60000 [======================>.......] - ETA: 24s - loss: 0.2109 - categorical_accuracy: 0.9365
46880/60000 [======================>.......] - ETA: 24s - loss: 0.2108 - categorical_accuracy: 0.9365
46912/60000 [======================>.......] - ETA: 23s - loss: 0.2106 - categorical_accuracy: 0.9366
46944/60000 [======================>.......] - ETA: 23s - loss: 0.2105 - categorical_accuracy: 0.9366
46976/60000 [======================>.......] - ETA: 23s - loss: 0.2106 - categorical_accuracy: 0.9366
47008/60000 [======================>.......] - ETA: 23s - loss: 0.2105 - categorical_accuracy: 0.9366
47040/60000 [======================>.......] - ETA: 23s - loss: 0.2106 - categorical_accuracy: 0.9366
47072/60000 [======================>.......] - ETA: 23s - loss: 0.2105 - categorical_accuracy: 0.9366
47104/60000 [======================>.......] - ETA: 23s - loss: 0.2105 - categorical_accuracy: 0.9367
47136/60000 [======================>.......] - ETA: 23s - loss: 0.2104 - categorical_accuracy: 0.9367
47168/60000 [======================>.......] - ETA: 23s - loss: 0.2103 - categorical_accuracy: 0.9367
47200/60000 [======================>.......] - ETA: 23s - loss: 0.2102 - categorical_accuracy: 0.9367
47232/60000 [======================>.......] - ETA: 23s - loss: 0.2101 - categorical_accuracy: 0.9368
47264/60000 [======================>.......] - ETA: 23s - loss: 0.2100 - categorical_accuracy: 0.9368
47296/60000 [======================>.......] - ETA: 23s - loss: 0.2099 - categorical_accuracy: 0.9368
47328/60000 [======================>.......] - ETA: 23s - loss: 0.2098 - categorical_accuracy: 0.9368
47360/60000 [======================>.......] - ETA: 23s - loss: 0.2100 - categorical_accuracy: 0.9369
47392/60000 [======================>.......] - ETA: 23s - loss: 0.2100 - categorical_accuracy: 0.9369
47424/60000 [======================>.......] - ETA: 23s - loss: 0.2098 - categorical_accuracy: 0.9369
47456/60000 [======================>.......] - ETA: 22s - loss: 0.2100 - categorical_accuracy: 0.9368
47488/60000 [======================>.......] - ETA: 22s - loss: 0.2099 - categorical_accuracy: 0.9368
47520/60000 [======================>.......] - ETA: 22s - loss: 0.2098 - categorical_accuracy: 0.9369
47552/60000 [======================>.......] - ETA: 22s - loss: 0.2097 - categorical_accuracy: 0.9369
47584/60000 [======================>.......] - ETA: 22s - loss: 0.2095 - categorical_accuracy: 0.9370
47616/60000 [======================>.......] - ETA: 22s - loss: 0.2094 - categorical_accuracy: 0.9370
47648/60000 [======================>.......] - ETA: 22s - loss: 0.2093 - categorical_accuracy: 0.9370
47680/60000 [======================>.......] - ETA: 22s - loss: 0.2092 - categorical_accuracy: 0.9371
47712/60000 [======================>.......] - ETA: 22s - loss: 0.2092 - categorical_accuracy: 0.9371
47744/60000 [======================>.......] - ETA: 22s - loss: 0.2090 - categorical_accuracy: 0.9371
47776/60000 [======================>.......] - ETA: 22s - loss: 0.2089 - categorical_accuracy: 0.9371
47808/60000 [======================>.......] - ETA: 22s - loss: 0.2088 - categorical_accuracy: 0.9372
47840/60000 [======================>.......] - ETA: 22s - loss: 0.2087 - categorical_accuracy: 0.9372
47872/60000 [======================>.......] - ETA: 22s - loss: 0.2086 - categorical_accuracy: 0.9372
47904/60000 [======================>.......] - ETA: 22s - loss: 0.2085 - categorical_accuracy: 0.9373
47936/60000 [======================>.......] - ETA: 22s - loss: 0.2084 - categorical_accuracy: 0.9373
47968/60000 [======================>.......] - ETA: 22s - loss: 0.2083 - categorical_accuracy: 0.9373
48000/60000 [=======================>......] - ETA: 21s - loss: 0.2084 - categorical_accuracy: 0.9373
48032/60000 [=======================>......] - ETA: 21s - loss: 0.2084 - categorical_accuracy: 0.9373
48064/60000 [=======================>......] - ETA: 21s - loss: 0.2082 - categorical_accuracy: 0.9374
48096/60000 [=======================>......] - ETA: 21s - loss: 0.2082 - categorical_accuracy: 0.9374
48128/60000 [=======================>......] - ETA: 21s - loss: 0.2082 - categorical_accuracy: 0.9374
48160/60000 [=======================>......] - ETA: 21s - loss: 0.2081 - categorical_accuracy: 0.9374
48192/60000 [=======================>......] - ETA: 21s - loss: 0.2081 - categorical_accuracy: 0.9374
48224/60000 [=======================>......] - ETA: 21s - loss: 0.2080 - categorical_accuracy: 0.9374
48256/60000 [=======================>......] - ETA: 21s - loss: 0.2079 - categorical_accuracy: 0.9375
48288/60000 [=======================>......] - ETA: 21s - loss: 0.2079 - categorical_accuracy: 0.9375
48320/60000 [=======================>......] - ETA: 21s - loss: 0.2079 - categorical_accuracy: 0.9375
48352/60000 [=======================>......] - ETA: 21s - loss: 0.2079 - categorical_accuracy: 0.9375
48384/60000 [=======================>......] - ETA: 21s - loss: 0.2079 - categorical_accuracy: 0.9375
48416/60000 [=======================>......] - ETA: 21s - loss: 0.2078 - categorical_accuracy: 0.9375
48448/60000 [=======================>......] - ETA: 21s - loss: 0.2077 - categorical_accuracy: 0.9376
48480/60000 [=======================>......] - ETA: 21s - loss: 0.2076 - categorical_accuracy: 0.9376
48512/60000 [=======================>......] - ETA: 21s - loss: 0.2075 - categorical_accuracy: 0.9376
48544/60000 [=======================>......] - ETA: 20s - loss: 0.2074 - categorical_accuracy: 0.9377
48576/60000 [=======================>......] - ETA: 20s - loss: 0.2073 - categorical_accuracy: 0.9377
48608/60000 [=======================>......] - ETA: 20s - loss: 0.2072 - categorical_accuracy: 0.9377
48640/60000 [=======================>......] - ETA: 20s - loss: 0.2071 - categorical_accuracy: 0.9378
48672/60000 [=======================>......] - ETA: 20s - loss: 0.2069 - categorical_accuracy: 0.9378
48704/60000 [=======================>......] - ETA: 20s - loss: 0.2069 - categorical_accuracy: 0.9378
48736/60000 [=======================>......] - ETA: 20s - loss: 0.2068 - categorical_accuracy: 0.9378
48768/60000 [=======================>......] - ETA: 20s - loss: 0.2067 - categorical_accuracy: 0.9379
48800/60000 [=======================>......] - ETA: 20s - loss: 0.2066 - categorical_accuracy: 0.9379
48832/60000 [=======================>......] - ETA: 20s - loss: 0.2065 - categorical_accuracy: 0.9379
48864/60000 [=======================>......] - ETA: 20s - loss: 0.2064 - categorical_accuracy: 0.9380
48896/60000 [=======================>......] - ETA: 20s - loss: 0.2063 - categorical_accuracy: 0.9380
48928/60000 [=======================>......] - ETA: 20s - loss: 0.2063 - categorical_accuracy: 0.9380
48960/60000 [=======================>......] - ETA: 20s - loss: 0.2061 - categorical_accuracy: 0.9380
48992/60000 [=======================>......] - ETA: 20s - loss: 0.2060 - categorical_accuracy: 0.9381
49024/60000 [=======================>......] - ETA: 20s - loss: 0.2061 - categorical_accuracy: 0.9380
49056/60000 [=======================>......] - ETA: 20s - loss: 0.2060 - categorical_accuracy: 0.9381
49088/60000 [=======================>......] - ETA: 19s - loss: 0.2058 - categorical_accuracy: 0.9381
49120/60000 [=======================>......] - ETA: 19s - loss: 0.2057 - categorical_accuracy: 0.9382
49152/60000 [=======================>......] - ETA: 19s - loss: 0.2058 - categorical_accuracy: 0.9381
49184/60000 [=======================>......] - ETA: 19s - loss: 0.2058 - categorical_accuracy: 0.9381
49216/60000 [=======================>......] - ETA: 19s - loss: 0.2057 - categorical_accuracy: 0.9382
49248/60000 [=======================>......] - ETA: 19s - loss: 0.2057 - categorical_accuracy: 0.9382
49280/60000 [=======================>......] - ETA: 19s - loss: 0.2056 - categorical_accuracy: 0.9382
49312/60000 [=======================>......] - ETA: 19s - loss: 0.2055 - categorical_accuracy: 0.9383
49344/60000 [=======================>......] - ETA: 19s - loss: 0.2054 - categorical_accuracy: 0.9383
49376/60000 [=======================>......] - ETA: 19s - loss: 0.2053 - categorical_accuracy: 0.9383
49408/60000 [=======================>......] - ETA: 19s - loss: 0.2053 - categorical_accuracy: 0.9383
49440/60000 [=======================>......] - ETA: 19s - loss: 0.2052 - categorical_accuracy: 0.9383
49472/60000 [=======================>......] - ETA: 19s - loss: 0.2052 - categorical_accuracy: 0.9383
49504/60000 [=======================>......] - ETA: 19s - loss: 0.2050 - categorical_accuracy: 0.9383
49536/60000 [=======================>......] - ETA: 19s - loss: 0.2050 - categorical_accuracy: 0.9384
49568/60000 [=======================>......] - ETA: 19s - loss: 0.2049 - categorical_accuracy: 0.9384
49600/60000 [=======================>......] - ETA: 19s - loss: 0.2048 - categorical_accuracy: 0.9384
49632/60000 [=======================>......] - ETA: 18s - loss: 0.2047 - categorical_accuracy: 0.9384
49664/60000 [=======================>......] - ETA: 18s - loss: 0.2047 - categorical_accuracy: 0.9384
49696/60000 [=======================>......] - ETA: 18s - loss: 0.2046 - categorical_accuracy: 0.9385
49728/60000 [=======================>......] - ETA: 18s - loss: 0.2045 - categorical_accuracy: 0.9385
49760/60000 [=======================>......] - ETA: 18s - loss: 0.2044 - categorical_accuracy: 0.9385
49792/60000 [=======================>......] - ETA: 18s - loss: 0.2044 - categorical_accuracy: 0.9385
49824/60000 [=======================>......] - ETA: 18s - loss: 0.2044 - categorical_accuracy: 0.9385
49856/60000 [=======================>......] - ETA: 18s - loss: 0.2043 - categorical_accuracy: 0.9385
49888/60000 [=======================>......] - ETA: 18s - loss: 0.2043 - categorical_accuracy: 0.9386
49920/60000 [=======================>......] - ETA: 18s - loss: 0.2042 - categorical_accuracy: 0.9386
49952/60000 [=======================>......] - ETA: 18s - loss: 0.2041 - categorical_accuracy: 0.9386
49984/60000 [=======================>......] - ETA: 18s - loss: 0.2040 - categorical_accuracy: 0.9387
50016/60000 [========================>.....] - ETA: 18s - loss: 0.2039 - categorical_accuracy: 0.9387
50048/60000 [========================>.....] - ETA: 18s - loss: 0.2038 - categorical_accuracy: 0.9387
50080/60000 [========================>.....] - ETA: 18s - loss: 0.2037 - categorical_accuracy: 0.9387
50112/60000 [========================>.....] - ETA: 18s - loss: 0.2037 - categorical_accuracy: 0.9388
50144/60000 [========================>.....] - ETA: 18s - loss: 0.2036 - categorical_accuracy: 0.9387
50176/60000 [========================>.....] - ETA: 17s - loss: 0.2036 - categorical_accuracy: 0.9387
50208/60000 [========================>.....] - ETA: 17s - loss: 0.2036 - categorical_accuracy: 0.9387
50240/60000 [========================>.....] - ETA: 17s - loss: 0.2035 - categorical_accuracy: 0.9387
50272/60000 [========================>.....] - ETA: 17s - loss: 0.2034 - categorical_accuracy: 0.9387
50304/60000 [========================>.....] - ETA: 17s - loss: 0.2033 - categorical_accuracy: 0.9388
50336/60000 [========================>.....] - ETA: 17s - loss: 0.2032 - categorical_accuracy: 0.9388
50368/60000 [========================>.....] - ETA: 17s - loss: 0.2032 - categorical_accuracy: 0.9388
50400/60000 [========================>.....] - ETA: 17s - loss: 0.2031 - categorical_accuracy: 0.9388
50432/60000 [========================>.....] - ETA: 17s - loss: 0.2030 - categorical_accuracy: 0.9389
50464/60000 [========================>.....] - ETA: 17s - loss: 0.2031 - categorical_accuracy: 0.9388
50496/60000 [========================>.....] - ETA: 17s - loss: 0.2030 - categorical_accuracy: 0.9389
50528/60000 [========================>.....] - ETA: 17s - loss: 0.2028 - categorical_accuracy: 0.9389
50560/60000 [========================>.....] - ETA: 17s - loss: 0.2028 - categorical_accuracy: 0.9389
50592/60000 [========================>.....] - ETA: 17s - loss: 0.2027 - categorical_accuracy: 0.9390
50624/60000 [========================>.....] - ETA: 17s - loss: 0.2026 - categorical_accuracy: 0.9390
50656/60000 [========================>.....] - ETA: 17s - loss: 0.2026 - categorical_accuracy: 0.9390
50688/60000 [========================>.....] - ETA: 17s - loss: 0.2027 - categorical_accuracy: 0.9390
50720/60000 [========================>.....] - ETA: 16s - loss: 0.2026 - categorical_accuracy: 0.9390
50752/60000 [========================>.....] - ETA: 16s - loss: 0.2026 - categorical_accuracy: 0.9389
50784/60000 [========================>.....] - ETA: 16s - loss: 0.2025 - categorical_accuracy: 0.9390
50816/60000 [========================>.....] - ETA: 16s - loss: 0.2024 - categorical_accuracy: 0.9390
50848/60000 [========================>.....] - ETA: 16s - loss: 0.2023 - categorical_accuracy: 0.9391
50880/60000 [========================>.....] - ETA: 16s - loss: 0.2022 - categorical_accuracy: 0.9391
50912/60000 [========================>.....] - ETA: 16s - loss: 0.2021 - categorical_accuracy: 0.9391
50944/60000 [========================>.....] - ETA: 16s - loss: 0.2020 - categorical_accuracy: 0.9391
50976/60000 [========================>.....] - ETA: 16s - loss: 0.2019 - categorical_accuracy: 0.9391
51008/60000 [========================>.....] - ETA: 16s - loss: 0.2018 - categorical_accuracy: 0.9392
51040/60000 [========================>.....] - ETA: 16s - loss: 0.2017 - categorical_accuracy: 0.9392
51072/60000 [========================>.....] - ETA: 16s - loss: 0.2016 - categorical_accuracy: 0.9392
51104/60000 [========================>.....] - ETA: 16s - loss: 0.2016 - categorical_accuracy: 0.9392
51136/60000 [========================>.....] - ETA: 16s - loss: 0.2015 - categorical_accuracy: 0.9392
51168/60000 [========================>.....] - ETA: 16s - loss: 0.2015 - categorical_accuracy: 0.9392
51200/60000 [========================>.....] - ETA: 16s - loss: 0.2014 - categorical_accuracy: 0.9392
51232/60000 [========================>.....] - ETA: 16s - loss: 0.2013 - categorical_accuracy: 0.9393
51264/60000 [========================>.....] - ETA: 15s - loss: 0.2013 - categorical_accuracy: 0.9393
51296/60000 [========================>.....] - ETA: 15s - loss: 0.2011 - categorical_accuracy: 0.9393
51328/60000 [========================>.....] - ETA: 15s - loss: 0.2010 - categorical_accuracy: 0.9393
51360/60000 [========================>.....] - ETA: 15s - loss: 0.2010 - categorical_accuracy: 0.9394
51392/60000 [========================>.....] - ETA: 15s - loss: 0.2008 - categorical_accuracy: 0.9394
51424/60000 [========================>.....] - ETA: 15s - loss: 0.2007 - categorical_accuracy: 0.9394
51456/60000 [========================>.....] - ETA: 15s - loss: 0.2006 - categorical_accuracy: 0.9395
51488/60000 [========================>.....] - ETA: 15s - loss: 0.2005 - categorical_accuracy: 0.9395
51520/60000 [========================>.....] - ETA: 15s - loss: 0.2005 - categorical_accuracy: 0.9395
51552/60000 [========================>.....] - ETA: 15s - loss: 0.2004 - categorical_accuracy: 0.9395
51584/60000 [========================>.....] - ETA: 15s - loss: 0.2003 - categorical_accuracy: 0.9395
51616/60000 [========================>.....] - ETA: 15s - loss: 0.2002 - categorical_accuracy: 0.9396
51648/60000 [========================>.....] - ETA: 15s - loss: 0.2001 - categorical_accuracy: 0.9396
51680/60000 [========================>.....] - ETA: 15s - loss: 0.2001 - categorical_accuracy: 0.9396
51712/60000 [========================>.....] - ETA: 15s - loss: 0.2000 - categorical_accuracy: 0.9396
51744/60000 [========================>.....] - ETA: 15s - loss: 0.2001 - categorical_accuracy: 0.9396
51776/60000 [========================>.....] - ETA: 15s - loss: 0.1999 - categorical_accuracy: 0.9397
51808/60000 [========================>.....] - ETA: 15s - loss: 0.1998 - categorical_accuracy: 0.9397
51840/60000 [========================>.....] - ETA: 14s - loss: 0.1999 - categorical_accuracy: 0.9396
51872/60000 [========================>.....] - ETA: 14s - loss: 0.2000 - categorical_accuracy: 0.9396
51904/60000 [========================>.....] - ETA: 14s - loss: 0.1999 - categorical_accuracy: 0.9397
51936/60000 [========================>.....] - ETA: 14s - loss: 0.1999 - categorical_accuracy: 0.9397
51968/60000 [========================>.....] - ETA: 14s - loss: 0.1998 - categorical_accuracy: 0.9397
52000/60000 [=========================>....] - ETA: 14s - loss: 0.1997 - categorical_accuracy: 0.9397
52032/60000 [=========================>....] - ETA: 14s - loss: 0.1996 - categorical_accuracy: 0.9397
52064/60000 [=========================>....] - ETA: 14s - loss: 0.1995 - categorical_accuracy: 0.9398
52096/60000 [=========================>....] - ETA: 14s - loss: 0.1994 - categorical_accuracy: 0.9398
52128/60000 [=========================>....] - ETA: 14s - loss: 0.1993 - categorical_accuracy: 0.9398
52160/60000 [=========================>....] - ETA: 14s - loss: 0.1993 - categorical_accuracy: 0.9398
52192/60000 [=========================>....] - ETA: 14s - loss: 0.1992 - categorical_accuracy: 0.9398
52224/60000 [=========================>....] - ETA: 14s - loss: 0.1992 - categorical_accuracy: 0.9398
52256/60000 [=========================>....] - ETA: 14s - loss: 0.1991 - categorical_accuracy: 0.9399
52288/60000 [=========================>....] - ETA: 14s - loss: 0.1990 - categorical_accuracy: 0.9399
52320/60000 [=========================>....] - ETA: 14s - loss: 0.1989 - categorical_accuracy: 0.9399
52352/60000 [=========================>....] - ETA: 14s - loss: 0.1988 - categorical_accuracy: 0.9400
52384/60000 [=========================>....] - ETA: 13s - loss: 0.1987 - categorical_accuracy: 0.9400
52416/60000 [=========================>....] - ETA: 13s - loss: 0.1989 - categorical_accuracy: 0.9400
52448/60000 [=========================>....] - ETA: 13s - loss: 0.1988 - categorical_accuracy: 0.9400
52480/60000 [=========================>....] - ETA: 13s - loss: 0.1988 - categorical_accuracy: 0.9400
52512/60000 [=========================>....] - ETA: 13s - loss: 0.1987 - categorical_accuracy: 0.9400
52544/60000 [=========================>....] - ETA: 13s - loss: 0.1987 - categorical_accuracy: 0.9400
52576/60000 [=========================>....] - ETA: 13s - loss: 0.1986 - categorical_accuracy: 0.9400
52608/60000 [=========================>....] - ETA: 13s - loss: 0.1985 - categorical_accuracy: 0.9401
52640/60000 [=========================>....] - ETA: 13s - loss: 0.1984 - categorical_accuracy: 0.9401
52672/60000 [=========================>....] - ETA: 13s - loss: 0.1984 - categorical_accuracy: 0.9401
52704/60000 [=========================>....] - ETA: 13s - loss: 0.1983 - categorical_accuracy: 0.9401
52736/60000 [=========================>....] - ETA: 13s - loss: 0.1982 - categorical_accuracy: 0.9402
52768/60000 [=========================>....] - ETA: 13s - loss: 0.1981 - categorical_accuracy: 0.9402
52800/60000 [=========================>....] - ETA: 13s - loss: 0.1981 - categorical_accuracy: 0.9402
52832/60000 [=========================>....] - ETA: 13s - loss: 0.1980 - categorical_accuracy: 0.9402
52864/60000 [=========================>....] - ETA: 13s - loss: 0.1979 - categorical_accuracy: 0.9402
52896/60000 [=========================>....] - ETA: 13s - loss: 0.1979 - categorical_accuracy: 0.9402
52928/60000 [=========================>....] - ETA: 12s - loss: 0.1978 - categorical_accuracy: 0.9403
52960/60000 [=========================>....] - ETA: 12s - loss: 0.1977 - categorical_accuracy: 0.9403
52992/60000 [=========================>....] - ETA: 12s - loss: 0.1976 - categorical_accuracy: 0.9403
53024/60000 [=========================>....] - ETA: 12s - loss: 0.1976 - categorical_accuracy: 0.9403
53056/60000 [=========================>....] - ETA: 12s - loss: 0.1975 - categorical_accuracy: 0.9403
53088/60000 [=========================>....] - ETA: 12s - loss: 0.1974 - categorical_accuracy: 0.9404
53120/60000 [=========================>....] - ETA: 12s - loss: 0.1973 - categorical_accuracy: 0.9404
53152/60000 [=========================>....] - ETA: 12s - loss: 0.1973 - categorical_accuracy: 0.9404
53184/60000 [=========================>....] - ETA: 12s - loss: 0.1974 - categorical_accuracy: 0.9404
53216/60000 [=========================>....] - ETA: 12s - loss: 0.1974 - categorical_accuracy: 0.9404
53248/60000 [=========================>....] - ETA: 12s - loss: 0.1974 - categorical_accuracy: 0.9404
53280/60000 [=========================>....] - ETA: 12s - loss: 0.1973 - categorical_accuracy: 0.9404
53312/60000 [=========================>....] - ETA: 12s - loss: 0.1972 - categorical_accuracy: 0.9404
53344/60000 [=========================>....] - ETA: 12s - loss: 0.1971 - categorical_accuracy: 0.9404
53376/60000 [=========================>....] - ETA: 12s - loss: 0.1970 - categorical_accuracy: 0.9405
53408/60000 [=========================>....] - ETA: 12s - loss: 0.1970 - categorical_accuracy: 0.9405
53440/60000 [=========================>....] - ETA: 12s - loss: 0.1971 - categorical_accuracy: 0.9405
53472/60000 [=========================>....] - ETA: 11s - loss: 0.1972 - categorical_accuracy: 0.9405
53504/60000 [=========================>....] - ETA: 11s - loss: 0.1971 - categorical_accuracy: 0.9405
53536/60000 [=========================>....] - ETA: 11s - loss: 0.1970 - categorical_accuracy: 0.9405
53568/60000 [=========================>....] - ETA: 11s - loss: 0.1970 - categorical_accuracy: 0.9405
53600/60000 [=========================>....] - ETA: 11s - loss: 0.1969 - categorical_accuracy: 0.9405
53632/60000 [=========================>....] - ETA: 11s - loss: 0.1969 - categorical_accuracy: 0.9405
53664/60000 [=========================>....] - ETA: 11s - loss: 0.1968 - categorical_accuracy: 0.9406
53696/60000 [=========================>....] - ETA: 11s - loss: 0.1967 - categorical_accuracy: 0.9406
53728/60000 [=========================>....] - ETA: 11s - loss: 0.1966 - categorical_accuracy: 0.9406
53760/60000 [=========================>....] - ETA: 11s - loss: 0.1965 - categorical_accuracy: 0.9407
53792/60000 [=========================>....] - ETA: 11s - loss: 0.1964 - categorical_accuracy: 0.9407
53824/60000 [=========================>....] - ETA: 11s - loss: 0.1963 - categorical_accuracy: 0.9407
53856/60000 [=========================>....] - ETA: 11s - loss: 0.1963 - categorical_accuracy: 0.9407
53888/60000 [=========================>....] - ETA: 11s - loss: 0.1963 - categorical_accuracy: 0.9407
53920/60000 [=========================>....] - ETA: 11s - loss: 0.1962 - categorical_accuracy: 0.9407
53952/60000 [=========================>....] - ETA: 11s - loss: 0.1962 - categorical_accuracy: 0.9407
53984/60000 [=========================>....] - ETA: 11s - loss: 0.1961 - categorical_accuracy: 0.9408
54016/60000 [==========================>...] - ETA: 10s - loss: 0.1961 - categorical_accuracy: 0.9408
54048/60000 [==========================>...] - ETA: 10s - loss: 0.1960 - categorical_accuracy: 0.9408
54080/60000 [==========================>...] - ETA: 10s - loss: 0.1960 - categorical_accuracy: 0.9408
54112/60000 [==========================>...] - ETA: 10s - loss: 0.1959 - categorical_accuracy: 0.9408
54144/60000 [==========================>...] - ETA: 10s - loss: 0.1959 - categorical_accuracy: 0.9408
54176/60000 [==========================>...] - ETA: 10s - loss: 0.1958 - categorical_accuracy: 0.9408
54208/60000 [==========================>...] - ETA: 10s - loss: 0.1958 - categorical_accuracy: 0.9409
54240/60000 [==========================>...] - ETA: 10s - loss: 0.1959 - categorical_accuracy: 0.9408
54272/60000 [==========================>...] - ETA: 10s - loss: 0.1959 - categorical_accuracy: 0.9409
54304/60000 [==========================>...] - ETA: 10s - loss: 0.1957 - categorical_accuracy: 0.9409
54336/60000 [==========================>...] - ETA: 10s - loss: 0.1957 - categorical_accuracy: 0.9409
54368/60000 [==========================>...] - ETA: 10s - loss: 0.1956 - categorical_accuracy: 0.9409
54400/60000 [==========================>...] - ETA: 10s - loss: 0.1956 - categorical_accuracy: 0.9409
54432/60000 [==========================>...] - ETA: 10s - loss: 0.1955 - categorical_accuracy: 0.9410
54464/60000 [==========================>...] - ETA: 10s - loss: 0.1954 - categorical_accuracy: 0.9410
54496/60000 [==========================>...] - ETA: 10s - loss: 0.1953 - categorical_accuracy: 0.9410
54528/60000 [==========================>...] - ETA: 10s - loss: 0.1952 - categorical_accuracy: 0.9411
54560/60000 [==========================>...] - ETA: 9s - loss: 0.1951 - categorical_accuracy: 0.9411 
54592/60000 [==========================>...] - ETA: 9s - loss: 0.1952 - categorical_accuracy: 0.9411
54624/60000 [==========================>...] - ETA: 9s - loss: 0.1951 - categorical_accuracy: 0.9411
54656/60000 [==========================>...] - ETA: 9s - loss: 0.1951 - categorical_accuracy: 0.9411
54688/60000 [==========================>...] - ETA: 9s - loss: 0.1950 - categorical_accuracy: 0.9411
54720/60000 [==========================>...] - ETA: 9s - loss: 0.1949 - categorical_accuracy: 0.9412
54752/60000 [==========================>...] - ETA: 9s - loss: 0.1948 - categorical_accuracy: 0.9412
54784/60000 [==========================>...] - ETA: 9s - loss: 0.1947 - categorical_accuracy: 0.9412
54816/60000 [==========================>...] - ETA: 9s - loss: 0.1947 - categorical_accuracy: 0.9412
54848/60000 [==========================>...] - ETA: 9s - loss: 0.1947 - categorical_accuracy: 0.9412
54880/60000 [==========================>...] - ETA: 9s - loss: 0.1947 - categorical_accuracy: 0.9412
54912/60000 [==========================>...] - ETA: 9s - loss: 0.1946 - categorical_accuracy: 0.9412
54944/60000 [==========================>...] - ETA: 9s - loss: 0.1945 - categorical_accuracy: 0.9412
54976/60000 [==========================>...] - ETA: 9s - loss: 0.1944 - categorical_accuracy: 0.9412
55008/60000 [==========================>...] - ETA: 9s - loss: 0.1943 - categorical_accuracy: 0.9412
55040/60000 [==========================>...] - ETA: 9s - loss: 0.1942 - categorical_accuracy: 0.9413
55072/60000 [==========================>...] - ETA: 9s - loss: 0.1942 - categorical_accuracy: 0.9413
55104/60000 [==========================>...] - ETA: 8s - loss: 0.1940 - categorical_accuracy: 0.9413
55136/60000 [==========================>...] - ETA: 8s - loss: 0.1939 - categorical_accuracy: 0.9414
55168/60000 [==========================>...] - ETA: 8s - loss: 0.1940 - categorical_accuracy: 0.9414
55200/60000 [==========================>...] - ETA: 8s - loss: 0.1939 - categorical_accuracy: 0.9414
55232/60000 [==========================>...] - ETA: 8s - loss: 0.1938 - categorical_accuracy: 0.9414
55264/60000 [==========================>...] - ETA: 8s - loss: 0.1938 - categorical_accuracy: 0.9414
55296/60000 [==========================>...] - ETA: 8s - loss: 0.1937 - categorical_accuracy: 0.9414
55328/60000 [==========================>...] - ETA: 8s - loss: 0.1937 - categorical_accuracy: 0.9414
55360/60000 [==========================>...] - ETA: 8s - loss: 0.1936 - categorical_accuracy: 0.9415
55392/60000 [==========================>...] - ETA: 8s - loss: 0.1936 - categorical_accuracy: 0.9415
55424/60000 [==========================>...] - ETA: 8s - loss: 0.1936 - categorical_accuracy: 0.9415
55456/60000 [==========================>...] - ETA: 8s - loss: 0.1936 - categorical_accuracy: 0.9415
55488/60000 [==========================>...] - ETA: 8s - loss: 0.1935 - categorical_accuracy: 0.9415
55520/60000 [==========================>...] - ETA: 8s - loss: 0.1935 - categorical_accuracy: 0.9415
55552/60000 [==========================>...] - ETA: 8s - loss: 0.1934 - categorical_accuracy: 0.9416
55584/60000 [==========================>...] - ETA: 8s - loss: 0.1934 - categorical_accuracy: 0.9415
55616/60000 [==========================>...] - ETA: 8s - loss: 0.1933 - categorical_accuracy: 0.9416
55648/60000 [==========================>...] - ETA: 7s - loss: 0.1933 - categorical_accuracy: 0.9416
55680/60000 [==========================>...] - ETA: 7s - loss: 0.1932 - categorical_accuracy: 0.9416
55712/60000 [==========================>...] - ETA: 7s - loss: 0.1931 - categorical_accuracy: 0.9416
55744/60000 [==========================>...] - ETA: 7s - loss: 0.1930 - categorical_accuracy: 0.9417
55776/60000 [==========================>...] - ETA: 7s - loss: 0.1929 - categorical_accuracy: 0.9417
55808/60000 [==========================>...] - ETA: 7s - loss: 0.1928 - categorical_accuracy: 0.9417
55840/60000 [==========================>...] - ETA: 7s - loss: 0.1927 - categorical_accuracy: 0.9417
55872/60000 [==========================>...] - ETA: 7s - loss: 0.1927 - categorical_accuracy: 0.9418
55904/60000 [==========================>...] - ETA: 7s - loss: 0.1926 - categorical_accuracy: 0.9418
55936/60000 [==========================>...] - ETA: 7s - loss: 0.1925 - categorical_accuracy: 0.9418
55968/60000 [==========================>...] - ETA: 7s - loss: 0.1924 - categorical_accuracy: 0.9418
56000/60000 [===========================>..] - ETA: 7s - loss: 0.1925 - categorical_accuracy: 0.9419
56032/60000 [===========================>..] - ETA: 7s - loss: 0.1924 - categorical_accuracy: 0.9418
56064/60000 [===========================>..] - ETA: 7s - loss: 0.1923 - categorical_accuracy: 0.9419
56096/60000 [===========================>..] - ETA: 7s - loss: 0.1922 - categorical_accuracy: 0.9419
56128/60000 [===========================>..] - ETA: 7s - loss: 0.1925 - categorical_accuracy: 0.9418
56160/60000 [===========================>..] - ETA: 7s - loss: 0.1924 - categorical_accuracy: 0.9419
56192/60000 [===========================>..] - ETA: 6s - loss: 0.1924 - categorical_accuracy: 0.9419
56224/60000 [===========================>..] - ETA: 6s - loss: 0.1923 - categorical_accuracy: 0.9419
56256/60000 [===========================>..] - ETA: 6s - loss: 0.1922 - categorical_accuracy: 0.9420
56288/60000 [===========================>..] - ETA: 6s - loss: 0.1922 - categorical_accuracy: 0.9420
56320/60000 [===========================>..] - ETA: 6s - loss: 0.1921 - categorical_accuracy: 0.9420
56352/60000 [===========================>..] - ETA: 6s - loss: 0.1920 - categorical_accuracy: 0.9420
56384/60000 [===========================>..] - ETA: 6s - loss: 0.1919 - categorical_accuracy: 0.9421
56416/60000 [===========================>..] - ETA: 6s - loss: 0.1919 - categorical_accuracy: 0.9421
56448/60000 [===========================>..] - ETA: 6s - loss: 0.1918 - categorical_accuracy: 0.9421
56480/60000 [===========================>..] - ETA: 6s - loss: 0.1917 - categorical_accuracy: 0.9422
56512/60000 [===========================>..] - ETA: 6s - loss: 0.1916 - categorical_accuracy: 0.9422
56544/60000 [===========================>..] - ETA: 6s - loss: 0.1915 - categorical_accuracy: 0.9422
56576/60000 [===========================>..] - ETA: 6s - loss: 0.1915 - categorical_accuracy: 0.9422
56608/60000 [===========================>..] - ETA: 6s - loss: 0.1914 - categorical_accuracy: 0.9423
56640/60000 [===========================>..] - ETA: 6s - loss: 0.1914 - categorical_accuracy: 0.9423
56672/60000 [===========================>..] - ETA: 6s - loss: 0.1914 - categorical_accuracy: 0.9423
56704/60000 [===========================>..] - ETA: 6s - loss: 0.1914 - categorical_accuracy: 0.9423
56736/60000 [===========================>..] - ETA: 5s - loss: 0.1913 - categorical_accuracy: 0.9423
56768/60000 [===========================>..] - ETA: 5s - loss: 0.1913 - categorical_accuracy: 0.9423
56800/60000 [===========================>..] - ETA: 5s - loss: 0.1912 - categorical_accuracy: 0.9423
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1912 - categorical_accuracy: 0.9424
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1911 - categorical_accuracy: 0.9424
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1910 - categorical_accuracy: 0.9424
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1909 - categorical_accuracy: 0.9424
56960/60000 [===========================>..] - ETA: 5s - loss: 0.1909 - categorical_accuracy: 0.9425
56992/60000 [===========================>..] - ETA: 5s - loss: 0.1908 - categorical_accuracy: 0.9425
57024/60000 [===========================>..] - ETA: 5s - loss: 0.1907 - categorical_accuracy: 0.9425
57056/60000 [===========================>..] - ETA: 5s - loss: 0.1906 - categorical_accuracy: 0.9425
57088/60000 [===========================>..] - ETA: 5s - loss: 0.1907 - categorical_accuracy: 0.9425
57120/60000 [===========================>..] - ETA: 5s - loss: 0.1906 - categorical_accuracy: 0.9425
57152/60000 [===========================>..] - ETA: 5s - loss: 0.1905 - categorical_accuracy: 0.9425
57184/60000 [===========================>..] - ETA: 5s - loss: 0.1905 - categorical_accuracy: 0.9425
57216/60000 [===========================>..] - ETA: 5s - loss: 0.1904 - categorical_accuracy: 0.9426
57248/60000 [===========================>..] - ETA: 5s - loss: 0.1904 - categorical_accuracy: 0.9426
57280/60000 [===========================>..] - ETA: 4s - loss: 0.1903 - categorical_accuracy: 0.9426
57312/60000 [===========================>..] - ETA: 4s - loss: 0.1902 - categorical_accuracy: 0.9426
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1902 - categorical_accuracy: 0.9426
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1901 - categorical_accuracy: 0.9426
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1901 - categorical_accuracy: 0.9427
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1900 - categorical_accuracy: 0.9427
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1899 - categorical_accuracy: 0.9427
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1898 - categorical_accuracy: 0.9427
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1898 - categorical_accuracy: 0.9427
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1898 - categorical_accuracy: 0.9428
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1897 - categorical_accuracy: 0.9428
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1896 - categorical_accuracy: 0.9428
57664/60000 [===========================>..] - ETA: 4s - loss: 0.1895 - categorical_accuracy: 0.9428
57696/60000 [===========================>..] - ETA: 4s - loss: 0.1895 - categorical_accuracy: 0.9429
57728/60000 [===========================>..] - ETA: 4s - loss: 0.1894 - categorical_accuracy: 0.9429
57760/60000 [===========================>..] - ETA: 4s - loss: 0.1893 - categorical_accuracy: 0.9429
57792/60000 [===========================>..] - ETA: 4s - loss: 0.1892 - categorical_accuracy: 0.9430
57824/60000 [===========================>..] - ETA: 3s - loss: 0.1894 - categorical_accuracy: 0.9429
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1893 - categorical_accuracy: 0.9429
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1892 - categorical_accuracy: 0.9429
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1893 - categorical_accuracy: 0.9430
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1892 - categorical_accuracy: 0.9430
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1892 - categorical_accuracy: 0.9430
58016/60000 [============================>.] - ETA: 3s - loss: 0.1891 - categorical_accuracy: 0.9430
58048/60000 [============================>.] - ETA: 3s - loss: 0.1890 - categorical_accuracy: 0.9430
58080/60000 [============================>.] - ETA: 3s - loss: 0.1889 - categorical_accuracy: 0.9431
58112/60000 [============================>.] - ETA: 3s - loss: 0.1888 - categorical_accuracy: 0.9431
58144/60000 [============================>.] - ETA: 3s - loss: 0.1887 - categorical_accuracy: 0.9431
58176/60000 [============================>.] - ETA: 3s - loss: 0.1886 - categorical_accuracy: 0.9432
58208/60000 [============================>.] - ETA: 3s - loss: 0.1886 - categorical_accuracy: 0.9432
58240/60000 [============================>.] - ETA: 3s - loss: 0.1885 - categorical_accuracy: 0.9432
58272/60000 [============================>.] - ETA: 3s - loss: 0.1884 - categorical_accuracy: 0.9432
58304/60000 [============================>.] - ETA: 3s - loss: 0.1883 - categorical_accuracy: 0.9433
58336/60000 [============================>.] - ETA: 3s - loss: 0.1884 - categorical_accuracy: 0.9433
58368/60000 [============================>.] - ETA: 2s - loss: 0.1883 - categorical_accuracy: 0.9433
58400/60000 [============================>.] - ETA: 2s - loss: 0.1882 - categorical_accuracy: 0.9433
58432/60000 [============================>.] - ETA: 2s - loss: 0.1882 - categorical_accuracy: 0.9433
58464/60000 [============================>.] - ETA: 2s - loss: 0.1882 - categorical_accuracy: 0.9433
58496/60000 [============================>.] - ETA: 2s - loss: 0.1881 - categorical_accuracy: 0.9433
58528/60000 [============================>.] - ETA: 2s - loss: 0.1880 - categorical_accuracy: 0.9434
58560/60000 [============================>.] - ETA: 2s - loss: 0.1880 - categorical_accuracy: 0.9434
58592/60000 [============================>.] - ETA: 2s - loss: 0.1880 - categorical_accuracy: 0.9434
58624/60000 [============================>.] - ETA: 2s - loss: 0.1879 - categorical_accuracy: 0.9434
58656/60000 [============================>.] - ETA: 2s - loss: 0.1878 - categorical_accuracy: 0.9434
58688/60000 [============================>.] - ETA: 2s - loss: 0.1878 - categorical_accuracy: 0.9434
58720/60000 [============================>.] - ETA: 2s - loss: 0.1878 - categorical_accuracy: 0.9435
58752/60000 [============================>.] - ETA: 2s - loss: 0.1877 - categorical_accuracy: 0.9435
58784/60000 [============================>.] - ETA: 2s - loss: 0.1877 - categorical_accuracy: 0.9435
58816/60000 [============================>.] - ETA: 2s - loss: 0.1876 - categorical_accuracy: 0.9435
58848/60000 [============================>.] - ETA: 2s - loss: 0.1875 - categorical_accuracy: 0.9435
58880/60000 [============================>.] - ETA: 2s - loss: 0.1876 - categorical_accuracy: 0.9435
58912/60000 [============================>.] - ETA: 1s - loss: 0.1875 - categorical_accuracy: 0.9435
58944/60000 [============================>.] - ETA: 1s - loss: 0.1875 - categorical_accuracy: 0.9435
58976/60000 [============================>.] - ETA: 1s - loss: 0.1874 - categorical_accuracy: 0.9435
59008/60000 [============================>.] - ETA: 1s - loss: 0.1874 - categorical_accuracy: 0.9435
59040/60000 [============================>.] - ETA: 1s - loss: 0.1873 - categorical_accuracy: 0.9436
59072/60000 [============================>.] - ETA: 1s - loss: 0.1872 - categorical_accuracy: 0.9436
59104/60000 [============================>.] - ETA: 1s - loss: 0.1873 - categorical_accuracy: 0.9436
59136/60000 [============================>.] - ETA: 1s - loss: 0.1872 - categorical_accuracy: 0.9436
59168/60000 [============================>.] - ETA: 1s - loss: 0.1872 - categorical_accuracy: 0.9436
59200/60000 [============================>.] - ETA: 1s - loss: 0.1871 - categorical_accuracy: 0.9436
59232/60000 [============================>.] - ETA: 1s - loss: 0.1870 - categorical_accuracy: 0.9437
59264/60000 [============================>.] - ETA: 1s - loss: 0.1869 - categorical_accuracy: 0.9437
59296/60000 [============================>.] - ETA: 1s - loss: 0.1868 - categorical_accuracy: 0.9437
59328/60000 [============================>.] - ETA: 1s - loss: 0.1868 - categorical_accuracy: 0.9437
59360/60000 [============================>.] - ETA: 1s - loss: 0.1867 - categorical_accuracy: 0.9438
59392/60000 [============================>.] - ETA: 1s - loss: 0.1866 - categorical_accuracy: 0.9438
59424/60000 [============================>.] - ETA: 1s - loss: 0.1865 - categorical_accuracy: 0.9438
59456/60000 [============================>.] - ETA: 0s - loss: 0.1864 - categorical_accuracy: 0.9438
59488/60000 [============================>.] - ETA: 0s - loss: 0.1864 - categorical_accuracy: 0.9439
59520/60000 [============================>.] - ETA: 0s - loss: 0.1865 - categorical_accuracy: 0.9438
59552/60000 [============================>.] - ETA: 0s - loss: 0.1865 - categorical_accuracy: 0.9438
59584/60000 [============================>.] - ETA: 0s - loss: 0.1865 - categorical_accuracy: 0.9438
59616/60000 [============================>.] - ETA: 0s - loss: 0.1864 - categorical_accuracy: 0.9439
59648/60000 [============================>.] - ETA: 0s - loss: 0.1863 - categorical_accuracy: 0.9439
59680/60000 [============================>.] - ETA: 0s - loss: 0.1862 - categorical_accuracy: 0.9439
59712/60000 [============================>.] - ETA: 0s - loss: 0.1861 - categorical_accuracy: 0.9439
59744/60000 [============================>.] - ETA: 0s - loss: 0.1861 - categorical_accuracy: 0.9440
59776/60000 [============================>.] - ETA: 0s - loss: 0.1860 - categorical_accuracy: 0.9440
59808/60000 [============================>.] - ETA: 0s - loss: 0.1859 - categorical_accuracy: 0.9440
59840/60000 [============================>.] - ETA: 0s - loss: 0.1859 - categorical_accuracy: 0.9440
59872/60000 [============================>.] - ETA: 0s - loss: 0.1858 - categorical_accuracy: 0.9440
59904/60000 [============================>.] - ETA: 0s - loss: 0.1857 - categorical_accuracy: 0.9440
59936/60000 [============================>.] - ETA: 0s - loss: 0.1856 - categorical_accuracy: 0.9441
59968/60000 [============================>.] - ETA: 0s - loss: 0.1856 - categorical_accuracy: 0.9441
60000/60000 [==============================] - 114s 2ms/step - loss: 0.1855 - categorical_accuracy: 0.9441 - val_loss: 0.0451 - val_categorical_accuracy: 0.9846

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 17s
  192/10000 [..............................] - ETA: 5s 
  352/10000 [>.............................] - ETA: 4s
  512/10000 [>.............................] - ETA: 4s
  672/10000 [=>............................] - ETA: 4s
  832/10000 [=>............................] - ETA: 3s
  992/10000 [=>............................] - ETA: 3s
 1152/10000 [==>...........................] - ETA: 3s
 1312/10000 [==>...........................] - ETA: 3s
 1472/10000 [===>..........................] - ETA: 3s
 1632/10000 [===>..........................] - ETA: 3s
 1792/10000 [====>.........................] - ETA: 3s
 1952/10000 [====>.........................] - ETA: 3s
 2112/10000 [=====>........................] - ETA: 3s
 2272/10000 [=====>........................] - ETA: 2s
 2432/10000 [======>.......................] - ETA: 2s
 2592/10000 [======>.......................] - ETA: 2s
 2752/10000 [=======>......................] - ETA: 2s
 2912/10000 [=======>......................] - ETA: 2s
 3040/10000 [========>.....................] - ETA: 2s
 3200/10000 [========>.....................] - ETA: 2s
 3360/10000 [=========>....................] - ETA: 2s
 3520/10000 [=========>....................] - ETA: 2s
 3648/10000 [=========>....................] - ETA: 2s
 3776/10000 [==========>...................] - ETA: 2s
 3936/10000 [==========>...................] - ETA: 2s
 4096/10000 [===========>..................] - ETA: 2s
 4256/10000 [===========>..................] - ETA: 2s
 4416/10000 [============>.................] - ETA: 2s
 4576/10000 [============>.................] - ETA: 2s
 4736/10000 [=============>................] - ETA: 1s
 4896/10000 [=============>................] - ETA: 1s
 5056/10000 [==============>...............] - ETA: 1s
 5216/10000 [==============>...............] - ETA: 1s
 5344/10000 [===============>..............] - ETA: 1s
 5504/10000 [===============>..............] - ETA: 1s
 5664/10000 [===============>..............] - ETA: 1s
 5824/10000 [================>.............] - ETA: 1s
 5984/10000 [================>.............] - ETA: 1s
 6144/10000 [=================>............] - ETA: 1s
 6304/10000 [=================>............] - ETA: 1s
 6464/10000 [==================>...........] - ETA: 1s
 6624/10000 [==================>...........] - ETA: 1s
 6784/10000 [===================>..........] - ETA: 1s
 6944/10000 [===================>..........] - ETA: 1s
 7072/10000 [====================>.........] - ETA: 1s
 7232/10000 [====================>.........] - ETA: 1s
 7392/10000 [=====================>........] - ETA: 0s
 7552/10000 [=====================>........] - ETA: 0s
 7712/10000 [======================>.......] - ETA: 0s
 7872/10000 [======================>.......] - ETA: 0s
 8032/10000 [=======================>......] - ETA: 0s
 8192/10000 [=======================>......] - ETA: 0s
 8352/10000 [========================>.....] - ETA: 0s
 8512/10000 [========================>.....] - ETA: 0s
 8672/10000 [=========================>....] - ETA: 0s
 8832/10000 [=========================>....] - ETA: 0s
 8992/10000 [=========================>....] - ETA: 0s
 9152/10000 [==========================>...] - ETA: 0s
 9312/10000 [==========================>...] - ETA: 0s
 9472/10000 [===========================>..] - ETA: 0s
 9632/10000 [===========================>..] - ETA: 0s
 9792/10000 [============================>.] - ETA: 0s
 9952/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 4s 375us/step
[[4.1978816e-09 1.3838577e-09 7.3054736e-07 ... 9.9999905e-01
  9.5278285e-10 9.8691736e-08]
 [3.1897360e-07 9.3331738e-07 9.9999809e-01 ... 4.8228515e-09
  1.2233450e-07 1.5024705e-10]
 [3.2969893e-07 9.9989235e-01 7.7363120e-06 ... 4.6816702e-05
  1.1085159e-05 5.6152078e-07]
 ...
 [1.4022752e-10 7.2192864e-07 1.8814483e-09 ... 2.5306035e-06
  1.9482709e-06 1.8808763e-05]
 [4.8436288e-08 6.8372952e-09 7.9991402e-10 ... 1.6069606e-09
  8.5671847e-05 6.7376931e-08]
 [1.4344859e-06 1.1638732e-06 1.5914295e-05 ... 1.5285064e-09
  8.8425008e-07 8.9266744e-10]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.04506867418292677, 'accuracy_test:': 0.9846000075340271}

  ('#### Save   #######################################################',) 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/charcnn/result'}

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   91c2835..c0b3ac0  master     -> origin/master
Updating 91c2835..c0b3ac0
Fast-forward
 .../20200513/list_log_dataloader_20200513.md       |    2 +-
 error_list/20200513/list_log_import_20200513.md    |    2 +-
 error_list/20200513/list_log_json_20200513.md      | 1146 ++++++++++----------
 .../20200513/list_log_pullrequest_20200513.md      |    2 +-
 error_list/20200513/list_log_test_cli_20200513.md  |  364 +++----
 error_list/20200513/list_log_testall_20200513.md   |  436 ++++++++
 6 files changed, 1194 insertions(+), 758 deletions(-)
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
[master 24f7df8] ml_store
 1 file changed, 2048 insertions(+)
To github.com:arita37/mlmodels_store.git
   c0b3ac0..24f7df8  master -> master





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
{'loss': 0.46718550473451614, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-13 16:34:35.404070: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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


   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Already up to date.
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
[master 0e82f3c] ml_store
 1 file changed, 233 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   24f7df8..0e82f3c  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//temporal_fusion_google.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//temporal_fusion_google.py", line 17, in <module>
    from mlmodels.mode_tf.raw  import temporal_fusion_google
ModuleNotFoundError: No module named 'mlmodels.mode_tf'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Already up to date.
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
[master 7384663] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   0e82f3c..7384663  master -> master





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
 40%|████      | 2/5 [00:21<00:32, 10.94s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.8487363278020583, 'learning_rate': 0.030898985611203175, 'min_data_in_leaf': 24, 'num_leaves': 49} and reward: 0.39
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeb(\xd9\x16Z\xc8\x82X\r\x00\x00\x00learning_rateq\x02G?\x9f\xa3\xfb\xd2\xb8\x01<X\x10\x00\x00\x00min_data_in_leafq\x03K\x18X\n\x00\x00\x00num_leavesq\x04K1u.' and reward: 0.39
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeb(\xd9\x16Z\xc8\x82X\r\x00\x00\x00learning_rateq\x02G?\x9f\xa3\xfb\xd2\xb8\x01<X\x10\x00\x00\x00min_data_in_leafq\x03K\x18X\n\x00\x00\x00num_leavesq\x04K1u.' and reward: 0.39
 60%|██████    | 3/5 [00:49<00:31, 15.86s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.7543886648433602, 'learning_rate': 0.039310779302901624, 'min_data_in_leaf': 25, 'num_leaves': 56} and reward: 0.3916
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8#\xf3\xb2\x7f5\xf6X\r\x00\x00\x00learning_rateq\x02G?\xa4 \x8a\xde\xf8\xf3\xf2X\x10\x00\x00\x00min_data_in_leafq\x03K\x19X\n\x00\x00\x00num_leavesq\x04K8u.' and reward: 0.3916
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8#\xf3\xb2\x7f5\xf6X\r\x00\x00\x00learning_rateq\x02G?\xa4 \x8a\xde\xf8\xf3\xf2X\x10\x00\x00\x00min_data_in_leafq\x03K\x19X\n\x00\x00\x00num_leavesq\x04K8u.' and reward: 0.3916
 80%|████████  | 4/5 [01:18<00:19, 19.96s/it] 80%|████████  | 4/5 [01:18<00:19, 19.69s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.9351997493939228, 'learning_rate': 0.06311065279707209, 'min_data_in_leaf': 9, 'num_leaves': 30} and reward: 0.392
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\xed(\x06[\xfa8X\r\x00\x00\x00learning_rateq\x02G?\xb0(\x05\r\xca\xea*X\x10\x00\x00\x00min_data_in_leafq\x03K\tX\n\x00\x00\x00num_leavesq\x04K\x1eu.' and reward: 0.392
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\xed(\x06[\xfa8X\r\x00\x00\x00learning_rateq\x02G?\xb0(\x05\r\xca\xea*X\x10\x00\x00\x00min_data_in_leafq\x03K\tX\n\x00\x00\x00num_leavesq\x04K\x1eu.' and reward: 0.392
Time for Gradient Boosting hyperparameter optimization: 98.61094617843628
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.9351997493939228, 'learning_rate': 0.06311065279707209, 'min_data_in_leaf': 9, 'num_leaves': 30}
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
 40%|████      | 2/5 [00:57<01:26, 28.74s/it] 40%|████      | 2/5 [00:57<01:26, 28.74s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.3025012026243939, 'embedding_size_factor': 1.136684130334273, 'layers.choice': 0, 'learning_rate': 0.0038085027407127737, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 0.011921263272579356} and reward: 0.3636
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd3\\.\x01\x11o\xadX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf2/\xdb\xb2\xda\xaf\xbcX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?o3\x02V\xfa\x93\x1fX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?\x88j,\xdf\x0f\xd6\xfcu.' and reward: 0.3636
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd3\\.\x01\x11o\xadX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf2/\xdb\xb2\xda\xaf\xbcX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?o3\x02V\xfa\x93\x1fX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?\x88j,\xdf\x0f\xd6\xfcu.' and reward: 0.3636
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 116.4551568031311
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.73s of the -98.87s of remaining time.
Ensemble size: 85
Ensemble weights: 
[0.21176471 0.03529412 0.15294118 0.15294118 0.27058824 0.17647059]
	0.3992	 = Validation accuracy score
	1.62s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 220.54s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f57c58a5240>

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   7384663..874c058  master     -> origin/master
Updating 7384663..874c058
Fast-forward
 error_list/20200513/list_log_benchmark_20200513.md |  170 +-
 error_list/20200513/list_log_import_20200513.md    |    2 +-
 error_list/20200513/list_log_json_20200513.md      | 1146 ++++-----
 error_list/20200513/list_log_jupyter_20200513.md   | 1668 +++++++------
 .../20200513/list_log_pullrequest_20200513.md      |    2 +-
 error_list/20200513/list_log_test_cli_20200513.md  |  364 +--
 error_list/20200513/list_log_testall_20200513.md   |  175 ++
 ...-12_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py | 2467 ++++++++++++++++++++
 8 files changed, 4313 insertions(+), 1681 deletions(-)
 create mode 100644 log_benchmark/log_benchmark_2020-05-13-16-12_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
[master 50bdee7] ml_store
 1 file changed, 214 insertions(+)
To github.com:arita37/mlmodels_store.git
   874c058..50bdee7  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon//fb_prophet.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon//fb_prophet.py", line 160, in <module>
    test(data_path = "model_fb/fbprophet.json", choice="json" )
TypeError: test() got an unexpected keyword argument 'choice'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Already up to date.
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
[master 2737038] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   50bdee7..2737038  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon//gluonts_model.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon//gluonts_model.py", line 15, in <module>
    from gluonts.model.deepar import DeepAREstimator
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/__init__.py", line 15, in <module>
    from ._estimator import DeepAREstimator
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/_estimator.py", line 24, in <module>
    from gluonts.distribution import DistributionOutput, StudentTOutput
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/__init__.py", line 15, in <module>
    from . import bijection
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 28, in <module>
    class Bijection:
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 36, in Bijection
    @validated()
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 398, in validator
    **init_fields,
  File "pydantic/main.py", line 778, in pydantic.main.create_model
TypeError: create_model() takes exactly 1 positional argument (0 given)

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Already up to date.
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
[master 81f4ffe] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   2737038..81f4ffe  master -> master





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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7fd856835fd0> 

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

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Already up to date.
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
[master 72482c4] ml_store
 1 file changed, 108 insertions(+)
To github.com:arita37/mlmodels_store.git
   81f4ffe..72482c4  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 9.80427414e-01  1.93752881e+00 -2.30839743e-01  3.66332015e-01
   1.10018476e+00 -1.04458938e+00 -3.44987210e-01  2.05117344e+00
   5.85662000e-01 -2.79308500e+00]
 [ 6.10942600e-01 -2.79099641e+00 -1.33520272e+00 -4.56117555e-01
  -9.44959948e-01 -9.79890252e-01 -1.56993672e-01  6.92574348e-01
  -4.78672356e-01 -1.06460122e-01]
 [ 7.73703613e-01  1.27852808e+00 -2.11416392e+00 -4.42229280e-01
   1.06821044e+00  3.23527354e-01 -2.50644065e+00 -1.09991490e-01
   8.54894544e-03 -4.11639163e-01]
 [ 1.16755486e+00  3.53600971e-02  7.14789597e-01 -1.53879325e+00
   1.10863359e+00 -4.47895185e-01 -1.75592564e+00  6.17985534e-01
  -1.84176326e-01  8.52704062e-01]
 [ 1.12062155e+00 -7.02920403e-01 -1.22957425e+00  7.25550518e-01
  -1.18013412e+00 -3.24204219e-01  1.10223673e+00  8.14343129e-01
   7.80469930e-01  1.10861676e+00]
 [ 5.58538729e-01 -5.16347909e-01 -5.18145552e-01  3.51116897e-01
   8.25506954e-01 -6.87704631e-02 -9.52062101e-01 -1.34776494e+00
   1.47073986e+00 -1.46140360e+00]
 [ 8.88838813e-01  1.03368687e+00 -4.97025792e-02  8.08844360e-01
   8.14051347e-01  1.78975468e+00  1.14690038e+00  4.51284016e-01
  -1.68405999e+00  4.66643267e-01]
 [ 9.29250600e-01 -1.10657307e+00 -1.95816909e+00 -3.59224096e-01
  -1.21258781e+00  5.05381903e-01  5.42645295e-01  1.21794090e+00
  -1.94068096e+00  6.77807571e-01]
 [ 6.23688521e-01  1.20660790e+00  9.03999174e-01 -2.82863552e-01
  -1.18913787e+00 -2.66326884e-01  1.42361443e+00  1.06897162e+00
   4.03714310e-02  1.57546791e+00]
 [ 1.13545112e+00  8.61623101e-01  4.90616924e-02 -2.08639057e+00
  -1.11469020e+00  3.61801641e-01 -8.06178212e-01  4.25920177e-01
   4.90803971e-02 -5.96086335e-01]
 [ 8.58774962e-01  2.29371761e+00 -1.47023709e+00 -8.30010986e-01
  -6.72049816e-01 -1.01951985e+00  5.99213235e-01 -2.14653842e-01
   1.02124813e+00  6.06403944e-01]
 [ 1.18947778e+00 -6.80678141e-01 -5.68244809e-02 -8.45080274e-02
   8.21783210e-01 -2.97361883e-01 -1.86578994e-01  4.17302005e-01
   7.84770651e-01  4.92336556e-01]
 [ 6.13636707e-01  3.16658895e-01  1.34710546e+00 -1.89526695e+00
  -7.60458095e-01  8.97291174e-02 -3.29051549e-01  4.10265745e-01
   8.59870972e-01 -1.04906775e+00]
 [ 5.63077902e-01 -1.17598267e+00 -1.74180344e-01  1.01012718e+00
   1.06796368e+00  9.20017933e-01 -1.68198840e-01 -1.95057341e-01
   8.05393424e-01  4.61164100e-01]
 [ 1.22867367e+00  1.34373116e-01 -1.82420406e-01 -2.68371304e-01
  -1.73963799e+00 -1.31675626e-01 -9.26871939e-01  1.01855247e+00
   1.23055820e+00 -4.91125138e-01]
 [ 1.98519313e+00  6.74711526e-01 -1.39662042e+00  6.18539131e-01
   1.22382712e+00 -4.43171931e-01 -1.89148284e-03  1.81053491e+00
  -1.30572692e+00 -8.61316361e-01]
 [ 8.61462558e-01  7.43205537e-02 -1.34501002e+00 -1.99560718e-01
  -1.47533915e+00 -6.54603169e-01 -3.14563862e-01  3.18014296e-01
  -8.90271552e-01 -1.29525789e+00]
 [ 1.03967316e+00 -7.31530982e-01  3.61847316e-01 -1.56573815e+00
   9.59288190e-01  1.01382247e+00 -1.78791289e+00 -2.22711263e+00
  -1.69933360e+00 -4.24492791e-01]
 [ 1.06523311e+00 -6.64867767e-01  1.00806543e+00 -1.94504696e+00
  -1.23017555e+00 -9.15424368e-01  3.37220938e-01  1.22515585e+00
  -1.05354607e+00  7.85226920e-01]
 [ 9.36211246e-01  2.04377395e-01 -1.49419377e+00  6.12232523e-01
  -9.84377246e-01  7.44884536e-01  4.94341651e-01 -3.62812886e-02
  -8.32395348e-01 -4.46699203e-01]
 [ 8.15836116e-01 -1.39169388e+00  2.50598029e+00  4.50217742e-01
  -8.82869820e-01  6.27437083e-01 -1.19586151e+00  7.51337235e-01
   1.40395436e-01  1.91979229e+00]
 [ 1.32720112e+00 -1.61198320e-01  6.02450901e-01 -2.86384915e-01
  -5.78962302e-01 -8.70887650e-01  1.37975819e+00  5.01429590e-01
  -4.78614074e-01 -8.92646674e-01]
 [ 1.09488485e+00 -6.96245395e-02 -1.16444148e-01  3.53870427e-01
  -1.44189096e+00 -1.86955017e-01  1.29118890e+00 -1.53236162e-01
  -2.43250851e+00 -2.27729800e+00]
 [ 8.76994650e-01  1.23225307e+00 -8.67787223e-01 -2.54179868e-01
   8.91891405e-01  1.39984394e+00 -8.77281519e-01 -7.81911683e-01
  -4.37508983e-01 -1.44087602e+00]
 [ 1.66752297e+00  1.22372221e+00 -4.59930104e-01 -5.93679025e-02
  -4.93856997e-01  1.44898940e+00 -1.18110317e+00 -4.77580855e-01
   2.59999942e-02 -7.90799954e-01]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7fae0cc88f98>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7fae26ffe7b8> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 8.58774962e-01  2.29371761e+00 -1.47023709e+00 -8.30010986e-01
  -6.72049816e-01 -1.01951985e+00  5.99213235e-01 -2.14653842e-01
   1.02124813e+00  6.06403944e-01]
 [ 6.23629500e-01  9.86352180e-01  1.45391758e+00 -4.66154857e-01
   9.36403332e-01  1.38499134e+00  3.49435894e-02 -1.07296428e+00
   4.95158611e-01  6.61681076e-01]
 [ 1.32857949e+00 -5.63236604e-01 -1.06179676e+00  2.39014596e+00
  -1.68450770e+00  2.45422849e-01 -5.69148654e-01  1.15259914e+00
  -2.24235772e-01  1.32247779e-01]
 [ 7.22978007e-01  1.85535621e-01  9.15499268e-01  3.94428030e-01
  -8.49830738e-01  7.25522558e-01 -1.50504326e-01  1.49588477e+00
   6.75453809e-01 -4.38200267e-01]
 [ 7.61706684e-01 -1.48515645e+00  1.30253554e+00 -5.92461285e-01
  -1.64162479e+00 -2.30490794e+00 -1.34869645e+00 -3.18171727e-02
   1.12487742e-01 -3.62612088e-01]
 [ 1.07258847e+00 -5.86523939e-01 -1.34267579e+00 -1.23685338e+00
   1.24328724e+00  8.75838928e-01 -3.26499498e-01  6.23362177e-01
  -4.34956683e-01  1.11438298e+00]
 [ 1.16755486e+00  3.53600971e-02  7.14789597e-01 -1.53879325e+00
   1.10863359e+00 -4.47895185e-01 -1.75592564e+00  6.17985534e-01
  -1.84176326e-01  8.52704062e-01]
 [ 1.25704434e+00 -1.82391985e+00 -6.12406973e-01  1.16707517e+00
  -6.23732812e-01 -3.96687001e-02  8.16043684e-01  8.85825799e-01
   1.89861649e-01  3.93109245e-01]
 [ 8.48069266e-01  4.51946037e-01  6.30195671e-01 -1.57915629e+00
   8.27987371e-01 -8.28627979e-01 -1.05344713e-01  5.28879746e-01
  -2.23708651e+00 -4.14846901e-01]
 [ 8.76994650e-01  1.23225307e+00 -8.67787223e-01 -2.54179868e-01
   8.91891405e-01  1.39984394e+00 -8.77281519e-01 -7.81911683e-01
  -4.37508983e-01 -1.44087602e+00]
 [ 1.21619061e+00 -1.90005215e-02  8.60891241e-01 -2.26760192e-01
  -1.36419132e+00 -1.56450785e+00  1.63169151e+00  9.31255679e-01
   9.49808815e-01 -8.80189065e-01]
 [ 1.34728643e+00 -3.64538050e-01  8.07509886e-02 -4.59717681e-01
  -8.89487596e-01  1.70548352e+00  9.49961101e-02  2.40505552e-01
  -9.99426501e-01 -7.67803746e-01]
 [ 8.88838813e-01  1.03368687e+00 -4.97025792e-02  8.08844360e-01
   8.14051347e-01  1.78975468e+00  1.14690038e+00  4.51284016e-01
  -1.68405999e+00  4.66643267e-01]
 [ 8.88389445e-01  2.82995534e-01  1.79558917e-02  1.08030817e-01
  -8.49671873e-01  2.94176190e-02 -5.03973949e-01 -1.34793129e-01
   1.04921829e+00 -1.27046078e+00]
 [ 9.29250600e-01 -1.10657307e+00 -1.95816909e+00 -3.59224096e-01
  -1.21258781e+00  5.05381903e-01  5.42645295e-01  1.21794090e+00
  -1.94068096e+00  6.77807571e-01]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 8.88611457e-01  8.49586845e-01 -3.09114176e-02 -1.22154015e-01
  -1.14722826e+00 -6.80851574e-01 -3.26061306e-01 -1.06787658e+00
  -7.66793627e-02  3.55717262e-01]
 [ 7.83440538e-01 -5.11884476e-02  8.24584625e-01 -7.25597119e-01
   9.31717197e-01 -8.67768678e-01  3.03085711e+00 -1.35977326e-01
  -7.97269785e-01  6.54580153e-01]
 [ 1.03967316e+00 -7.31530982e-01  3.61847316e-01 -1.56573815e+00
   9.59288190e-01  1.01382247e+00 -1.78791289e+00 -2.22711263e+00
  -1.69933360e+00 -4.24492791e-01]
 [ 6.67591795e-01 -4.52524973e-01 -6.05981321e-01  1.16128569e+00
  -1.44620987e+00  1.06996554e+00  1.92381543e+00 -1.04553425e+00
   3.55284507e-01  1.80358898e+00]
 [ 1.37661405e+00 -6.00225330e-01  7.25916853e-01 -3.79517516e-01
  -6.27546260e-01 -1.01480369e+00  9.66220863e-01  4.35986196e-01
  -6.87487393e-01  3.32107876e+00]
 [ 1.06040861e+00  5.10307597e-01  5.01725109e-01 -9.15791849e-01
  -9.07318361e-01 -4.07252043e-01 -1.79612295e-01  9.84951672e-01
   1.07125243e+00 -5.93343754e-01]
 [ 6.13636707e-01  3.16658895e-01  1.34710546e+00 -1.89526695e+00
  -7.60458095e-01  8.97291174e-02 -3.29051549e-01  4.10265745e-01
   8.59870972e-01 -1.04906775e+00]
 [ 1.06523311e+00 -6.64867767e-01  1.00806543e+00 -1.94504696e+00
  -1.23017555e+00 -9.15424368e-01  3.37220938e-01  1.22515585e+00
  -1.05354607e+00  7.85226920e-01]
 [ 9.26869810e-01  3.92334911e-01 -4.23478297e-01  4.48380651e-01
  -1.09230828e+00  1.12532350e+00 -9.48439656e-01  1.04053390e-01
   5.28003422e-01  1.00796648e+00]]
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
[[ 9.36211246e-01  2.04377395e-01 -1.49419377e+00  6.12232523e-01
  -9.84377246e-01  7.44884536e-01  4.94341651e-01 -3.62812886e-02
  -8.32395348e-01 -4.46699203e-01]
 [ 9.97855163e-01 -6.00138799e-01  4.57947076e-01  1.46765263e-01
  -9.33557290e-01  5.71804879e-01  5.72962726e-01 -3.68176565e-02
   1.12368489e-01 -1.78175491e-02]
 [ 4.73307772e-01 -9.73267585e-01 -2.28140691e-01  1.75167729e-01
  -1.01366961e+00 -5.34836927e-02  3.93787731e-01 -1.83061987e-01
  -2.21028902e-01  5.80330113e-01]
 [ 1.06523311e+00 -6.64867767e-01  1.00806543e+00 -1.94504696e+00
  -1.23017555e+00 -9.15424368e-01  3.37220938e-01  1.22515585e+00
  -1.05354607e+00  7.85226920e-01]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 9.83799588e-01 -4.07240024e-01  9.32721414e-01  1.60564992e-01
  -1.27861800e+00 -1.20149976e-01  1.99759555e-01  3.85602292e-01
   7.18290736e-01 -5.30119800e-01]
 [ 1.21619061e+00 -1.90005215e-02  8.60891241e-01 -2.26760192e-01
  -1.36419132e+00 -1.56450785e+00  1.63169151e+00  9.31255679e-01
   9.49808815e-01 -8.80189065e-01]
 [ 8.59823751e-01  1.71957132e-01 -3.48984191e-01  4.90561044e-01
  -1.15649503e+00 -1.39528303e+00  6.14726276e-01 -5.22356465e-01
  -3.69255902e-01 -9.77773002e-01]
 [ 1.46893146e+00 -1.47115693e+00  5.85910431e-01 -8.30171895e-01
   1.03345052e+00 -8.80577600e-01 -9.55425262e-01 -2.79097722e-01
   1.62284909e+00  2.06578332e+00]
 [ 8.78643802e-01  1.03703898e+00 -4.77124206e-01  6.72619748e-01
  -1.04948638e+00  2.42887697e+00  5.24750492e-01  1.00568668e+00
   3.53567216e-01 -3.59901817e-02]
 [ 9.47814113e-01 -1.13379204e+00  6.40985866e-01 -1.90548298e-01
  -1.23912256e+00  2.33339126e-01 -3.16901197e-01  4.34998324e-01
   9.10423603e-01  1.21987438e+00]
 [ 8.76994650e-01  1.23225307e+00 -8.67787223e-01 -2.54179868e-01
   8.91891405e-01  1.39984394e+00 -8.77281519e-01 -7.81911683e-01
  -4.37508983e-01 -1.44087602e+00]
 [ 8.78740711e-01 -1.92316341e-02  3.19656942e-01  1.50016279e-01
  -1.46662161e+00  4.63534322e-01 -8.98683193e-01  3.97880425e-01
  -9.96010889e-01  3.18154200e-01]
 [ 9.26869810e-01  3.92334911e-01 -4.23478297e-01  4.48380651e-01
  -1.09230828e+00  1.12532350e+00 -9.48439656e-01  1.04053390e-01
   5.28003422e-01  1.00796648e+00]
 [ 1.14377130e+00  7.27813500e-01  3.52494364e-01  5.15073614e-01
   1.17718111e+00 -2.78253447e+00 -1.94332341e+00  5.84646610e-01
   3.24274243e-01 -2.36436952e-01]
 [ 1.01195228e+00 -1.88141087e+00  1.70018815e+00  4.97269099e-01
  -9.17664624e-01  2.37332699e-01 -1.09033833e+00 -2.14444405e+00
  -3.69562425e-01  6.08783659e-01]
 [ 5.63077902e-01 -1.17598267e+00 -1.74180344e-01  1.01012718e+00
   1.06796368e+00  9.20017933e-01 -1.68198840e-01 -1.95057341e-01
   8.05393424e-01  4.61164100e-01]
 [ 1.32857949e+00 -5.63236604e-01 -1.06179676e+00  2.39014596e+00
  -1.68450770e+00  2.45422849e-01 -5.69148654e-01  1.15259914e+00
  -2.24235772e-01  1.32247779e-01]
 [ 1.03967316e+00 -7.31530982e-01  3.61847316e-01 -1.56573815e+00
   9.59288190e-01  1.01382247e+00 -1.78791289e+00 -2.22711263e+00
  -1.69933360e+00 -4.24492791e-01]
 [ 1.34740825e+00  7.33023232e-01  8.38634747e-01 -1.89881206e+00
  -5.42459922e-01 -1.11711069e+00 -1.09715436e+00 -5.08972278e-01
  -1.66485955e-01 -1.03918232e+00]
 [ 1.39198128e+00 -1.90221025e-01 -5.37223024e-01 -4.48738033e-01
   7.04557071e-01 -6.72448039e-01 -7.01344426e-01 -5.57494722e-01
   9.39168744e-01  1.56263850e-01]
 [ 1.12062155e+00 -7.02920403e-01 -1.22957425e+00  7.25550518e-01
  -1.18013412e+00 -3.24204219e-01  1.10223673e+00  8.14343129e-01
   7.80469930e-01  1.10861676e+00]
 [ 8.58774962e-01  2.29371761e+00 -1.47023709e+00 -8.30010986e-01
  -6.72049816e-01 -1.01951985e+00  5.99213235e-01 -2.14653842e-01
   1.02124813e+00  6.06403944e-01]
 [ 5.69983848e-01 -5.33020326e-01 -1.75458969e-01 -1.42655542e+00
   6.06604307e-01  1.76795995e+00 -1.15985185e-01 -4.75372875e-01
   4.77610182e-01 -9.33914656e-01]
 [ 1.22867367e+00  1.34373116e-01 -1.82420406e-01 -2.68371304e-01
  -1.73963799e+00 -1.31675626e-01 -9.26871939e-01  1.01855247e+00
   1.23055820e+00 -4.91125138e-01]]
None

  ############ Save/ Load ############################################ 

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Already up to date.
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
[master 70a951e] ml_store
 1 file changed, 321 insertions(+)
To github.com:arita37/mlmodels_store.git
   72482c4..70a951e  master -> master





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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @139645695713232
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @139645695713008
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @139645695711776
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @139645695711328
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @139645695710824
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @139645695710488

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
grad_step = 000000, loss = 0.795062
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.682007
grad_step = 000002, loss = 0.608163
grad_step = 000003, loss = 0.535012
grad_step = 000004, loss = 0.456378
grad_step = 000005, loss = 0.360820
grad_step = 000006, loss = 0.263606
grad_step = 000007, loss = 0.172645
grad_step = 000008, loss = 0.105345
grad_step = 000009, loss = 0.055868
grad_step = 000010, loss = 0.035567
grad_step = 000011, loss = 0.065717
grad_step = 000012, loss = 0.081805
grad_step = 000013, loss = 0.063778
grad_step = 000014, loss = 0.039292
grad_step = 000015, loss = 0.026650
grad_step = 000016, loss = 0.025457
grad_step = 000017, loss = 0.028379
grad_step = 000018, loss = 0.029563
grad_step = 000019, loss = 0.027621
grad_step = 000020, loss = 0.023687
grad_step = 000021, loss = 0.019752
grad_step = 000022, loss = 0.017343
grad_step = 000023, loss = 0.016527
grad_step = 000024, loss = 0.016122
grad_step = 000025, loss = 0.015597
grad_step = 000026, loss = 0.014249
grad_step = 000027, loss = 0.011922
grad_step = 000028, loss = 0.009294
grad_step = 000029, loss = 0.007453
grad_step = 000030, loss = 0.007089
grad_step = 000031, loss = 0.008067
grad_step = 000032, loss = 0.009587
grad_step = 000033, loss = 0.010708
grad_step = 000034, loss = 0.010849
grad_step = 000035, loss = 0.009966
grad_step = 000036, loss = 0.008472
grad_step = 000037, loss = 0.006995
grad_step = 000038, loss = 0.005992
grad_step = 000039, loss = 0.005694
grad_step = 000040, loss = 0.005962
grad_step = 000041, loss = 0.006438
grad_step = 000042, loss = 0.006796
grad_step = 000043, loss = 0.006874
grad_step = 000044, loss = 0.006706
grad_step = 000045, loss = 0.006425
grad_step = 000046, loss = 0.006159
grad_step = 000047, loss = 0.005979
grad_step = 000048, loss = 0.005877
grad_step = 000049, loss = 0.005805
grad_step = 000050, loss = 0.005697
grad_step = 000051, loss = 0.005554
grad_step = 000052, loss = 0.005423
grad_step = 000053, loss = 0.005365
grad_step = 000054, loss = 0.005393
grad_step = 000055, loss = 0.005459
grad_step = 000056, loss = 0.005495
grad_step = 000057, loss = 0.005460
grad_step = 000058, loss = 0.005355
grad_step = 000059, loss = 0.005216
grad_step = 000060, loss = 0.005088
grad_step = 000061, loss = 0.005008
grad_step = 000062, loss = 0.004988
grad_step = 000063, loss = 0.005002
grad_step = 000064, loss = 0.005013
grad_step = 000065, loss = 0.004993
grad_step = 000066, loss = 0.004949
grad_step = 000067, loss = 0.004886
grad_step = 000068, loss = 0.004814
grad_step = 000069, loss = 0.004747
grad_step = 000070, loss = 0.004692
grad_step = 000071, loss = 0.004657
grad_step = 000072, loss = 0.004627
grad_step = 000073, loss = 0.004598
grad_step = 000074, loss = 0.004562
grad_step = 000075, loss = 0.004514
grad_step = 000076, loss = 0.004461
grad_step = 000077, loss = 0.004410
grad_step = 000078, loss = 0.004363
grad_step = 000079, loss = 0.004323
grad_step = 000080, loss = 0.004284
grad_step = 000081, loss = 0.004235
grad_step = 000082, loss = 0.004178
grad_step = 000083, loss = 0.004119
grad_step = 000084, loss = 0.004066
grad_step = 000085, loss = 0.004023
grad_step = 000086, loss = 0.003974
grad_step = 000087, loss = 0.003913
grad_step = 000088, loss = 0.003845
grad_step = 000089, loss = 0.003779
grad_step = 000090, loss = 0.003718
grad_step = 000091, loss = 0.003652
grad_step = 000092, loss = 0.003580
grad_step = 000093, loss = 0.003502
grad_step = 000094, loss = 0.003424
grad_step = 000095, loss = 0.003346
grad_step = 000096, loss = 0.003260
grad_step = 000097, loss = 0.003168
grad_step = 000098, loss = 0.003080
grad_step = 000099, loss = 0.002991
grad_step = 000100, loss = 0.002894
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002798
grad_step = 000102, loss = 0.002707
grad_step = 000103, loss = 0.002614
grad_step = 000104, loss = 0.002526
grad_step = 000105, loss = 0.002445
grad_step = 000106, loss = 0.002368
grad_step = 000107, loss = 0.002299
grad_step = 000108, loss = 0.002242
grad_step = 000109, loss = 0.002189
grad_step = 000110, loss = 0.002144
grad_step = 000111, loss = 0.002102
grad_step = 000112, loss = 0.002068
grad_step = 000113, loss = 0.002036
grad_step = 000114, loss = 0.001991
grad_step = 000115, loss = 0.001933
grad_step = 000116, loss = 0.001884
grad_step = 000117, loss = 0.001845
grad_step = 000118, loss = 0.001798
grad_step = 000119, loss = 0.001742
grad_step = 000120, loss = 0.001701
grad_step = 000121, loss = 0.001660
grad_step = 000122, loss = 0.001611
grad_step = 000123, loss = 0.001572
grad_step = 000124, loss = 0.001536
grad_step = 000125, loss = 0.001495
grad_step = 000126, loss = 0.001453
grad_step = 000127, loss = 0.001416
grad_step = 000128, loss = 0.001377
grad_step = 000129, loss = 0.001343
grad_step = 000130, loss = 0.001311
grad_step = 000131, loss = 0.001278
grad_step = 000132, loss = 0.001243
grad_step = 000133, loss = 0.001212
grad_step = 000134, loss = 0.001180
grad_step = 000135, loss = 0.001145
grad_step = 000136, loss = 0.001111
grad_step = 000137, loss = 0.001079
grad_step = 000138, loss = 0.001046
grad_step = 000139, loss = 0.001016
grad_step = 000140, loss = 0.000987
grad_step = 000141, loss = 0.000960
grad_step = 000142, loss = 0.000936
grad_step = 000143, loss = 0.000913
grad_step = 000144, loss = 0.000889
grad_step = 000145, loss = 0.000866
grad_step = 000146, loss = 0.000846
grad_step = 000147, loss = 0.000831
grad_step = 000148, loss = 0.000821
grad_step = 000149, loss = 0.000815
grad_step = 000150, loss = 0.000802
grad_step = 000151, loss = 0.000776
grad_step = 000152, loss = 0.000747
grad_step = 000153, loss = 0.000733
grad_step = 000154, loss = 0.000734
grad_step = 000155, loss = 0.000728
grad_step = 000156, loss = 0.000710
grad_step = 000157, loss = 0.000680
grad_step = 000158, loss = 0.000677
grad_step = 000159, loss = 0.000684
grad_step = 000160, loss = 0.000668
grad_step = 000161, loss = 0.000649
grad_step = 000162, loss = 0.000646
grad_step = 000163, loss = 0.000643
grad_step = 000164, loss = 0.000631
grad_step = 000165, loss = 0.000620
grad_step = 000166, loss = 0.000620
grad_step = 000167, loss = 0.000618
grad_step = 000168, loss = 0.000603
grad_step = 000169, loss = 0.000594
grad_step = 000170, loss = 0.000592
grad_step = 000171, loss = 0.000591
grad_step = 000172, loss = 0.000589
grad_step = 000173, loss = 0.000575
grad_step = 000174, loss = 0.000568
grad_step = 000175, loss = 0.000563
grad_step = 000176, loss = 0.000561
grad_step = 000177, loss = 0.000557
grad_step = 000178, loss = 0.000552
grad_step = 000179, loss = 0.000546
grad_step = 000180, loss = 0.000543
grad_step = 000181, loss = 0.000543
grad_step = 000182, loss = 0.000548
grad_step = 000183, loss = 0.000554
grad_step = 000184, loss = 0.000558
grad_step = 000185, loss = 0.000549
grad_step = 000186, loss = 0.000528
grad_step = 000187, loss = 0.000511
grad_step = 000188, loss = 0.000511
grad_step = 000189, loss = 0.000518
grad_step = 000190, loss = 0.000518
grad_step = 000191, loss = 0.000506
grad_step = 000192, loss = 0.000494
grad_step = 000193, loss = 0.000490
grad_step = 000194, loss = 0.000491
grad_step = 000195, loss = 0.000493
grad_step = 000196, loss = 0.000489
grad_step = 000197, loss = 0.000481
grad_step = 000198, loss = 0.000477
grad_step = 000199, loss = 0.000480
grad_step = 000200, loss = 0.000488
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.000497
grad_step = 000202, loss = 0.000490
grad_step = 000203, loss = 0.000471
grad_step = 000204, loss = 0.000460
grad_step = 000205, loss = 0.000463
grad_step = 000206, loss = 0.000468
grad_step = 000207, loss = 0.000458
grad_step = 000208, loss = 0.000443
grad_step = 000209, loss = 0.000439
grad_step = 000210, loss = 0.000445
grad_step = 000211, loss = 0.000449
grad_step = 000212, loss = 0.000444
grad_step = 000213, loss = 0.000432
grad_step = 000214, loss = 0.000428
grad_step = 000215, loss = 0.000433
grad_step = 000216, loss = 0.000436
grad_step = 000217, loss = 0.000434
grad_step = 000218, loss = 0.000433
grad_step = 000219, loss = 0.000439
grad_step = 000220, loss = 0.000451
grad_step = 000221, loss = 0.000452
grad_step = 000222, loss = 0.000428
grad_step = 000223, loss = 0.000411
grad_step = 000224, loss = 0.000407
grad_step = 000225, loss = 0.000415
grad_step = 000226, loss = 0.000419
grad_step = 000227, loss = 0.000416
grad_step = 000228, loss = 0.000423
grad_step = 000229, loss = 0.000419
grad_step = 000230, loss = 0.000401
grad_step = 000231, loss = 0.000383
grad_step = 000232, loss = 0.000383
grad_step = 000233, loss = 0.000395
grad_step = 000234, loss = 0.000398
grad_step = 000235, loss = 0.000390
grad_step = 000236, loss = 0.000384
grad_step = 000237, loss = 0.000388
grad_step = 000238, loss = 0.000389
grad_step = 000239, loss = 0.000383
grad_step = 000240, loss = 0.000377
grad_step = 000241, loss = 0.000379
grad_step = 000242, loss = 0.000383
grad_step = 000243, loss = 0.000376
grad_step = 000244, loss = 0.000366
grad_step = 000245, loss = 0.000357
grad_step = 000246, loss = 0.000356
grad_step = 000247, loss = 0.000355
grad_step = 000248, loss = 0.000352
grad_step = 000249, loss = 0.000350
grad_step = 000250, loss = 0.000352
grad_step = 000251, loss = 0.000361
grad_step = 000252, loss = 0.000383
grad_step = 000253, loss = 0.000420
grad_step = 000254, loss = 0.000476
grad_step = 000255, loss = 0.000456
grad_step = 000256, loss = 0.000389
grad_step = 000257, loss = 0.000339
grad_step = 000258, loss = 0.000354
grad_step = 000259, loss = 0.000401
grad_step = 000260, loss = 0.000397
grad_step = 000261, loss = 0.000350
grad_step = 000262, loss = 0.000331
grad_step = 000263, loss = 0.000350
grad_step = 000264, loss = 0.000364
grad_step = 000265, loss = 0.000350
grad_step = 000266, loss = 0.000329
grad_step = 000267, loss = 0.000327
grad_step = 000268, loss = 0.000340
grad_step = 000269, loss = 0.000346
grad_step = 000270, loss = 0.000328
grad_step = 000271, loss = 0.000320
grad_step = 000272, loss = 0.000327
grad_step = 000273, loss = 0.000335
grad_step = 000274, loss = 0.000330
grad_step = 000275, loss = 0.000318
grad_step = 000276, loss = 0.000314
grad_step = 000277, loss = 0.000319
grad_step = 000278, loss = 0.000324
grad_step = 000279, loss = 0.000322
grad_step = 000280, loss = 0.000315
grad_step = 000281, loss = 0.000310
grad_step = 000282, loss = 0.000309
grad_step = 000283, loss = 0.000312
grad_step = 000284, loss = 0.000315
grad_step = 000285, loss = 0.000313
grad_step = 000286, loss = 0.000311
grad_step = 000287, loss = 0.000306
grad_step = 000288, loss = 0.000302
grad_step = 000289, loss = 0.000300
grad_step = 000290, loss = 0.000302
grad_step = 000291, loss = 0.000304
grad_step = 000292, loss = 0.000308
grad_step = 000293, loss = 0.000312
grad_step = 000294, loss = 0.000311
grad_step = 000295, loss = 0.000310
grad_step = 000296, loss = 0.000305
grad_step = 000297, loss = 0.000301
grad_step = 000298, loss = 0.000297
grad_step = 000299, loss = 0.000293
grad_step = 000300, loss = 0.000292
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000291
grad_step = 000302, loss = 0.000291
grad_step = 000303, loss = 0.000293
grad_step = 000304, loss = 0.000299
grad_step = 000305, loss = 0.000309
grad_step = 000306, loss = 0.000326
grad_step = 000307, loss = 0.000337
grad_step = 000308, loss = 0.000345
grad_step = 000309, loss = 0.000335
grad_step = 000310, loss = 0.000314
grad_step = 000311, loss = 0.000291
grad_step = 000312, loss = 0.000282
grad_step = 000313, loss = 0.000285
grad_step = 000314, loss = 0.000297
grad_step = 000315, loss = 0.000310
grad_step = 000316, loss = 0.000315
grad_step = 000317, loss = 0.000312
grad_step = 000318, loss = 0.000295
grad_step = 000319, loss = 0.000282
grad_step = 000320, loss = 0.000278
grad_step = 000321, loss = 0.000284
grad_step = 000322, loss = 0.000293
grad_step = 000323, loss = 0.000295
grad_step = 000324, loss = 0.000293
grad_step = 000325, loss = 0.000281
grad_step = 000326, loss = 0.000273
grad_step = 000327, loss = 0.000271
grad_step = 000328, loss = 0.000276
grad_step = 000329, loss = 0.000281
grad_step = 000330, loss = 0.000282
grad_step = 000331, loss = 0.000280
grad_step = 000332, loss = 0.000274
grad_step = 000333, loss = 0.000271
grad_step = 000334, loss = 0.000267
grad_step = 000335, loss = 0.000267
grad_step = 000336, loss = 0.000270
grad_step = 000337, loss = 0.000273
grad_step = 000338, loss = 0.000277
grad_step = 000339, loss = 0.000279
grad_step = 000340, loss = 0.000283
grad_step = 000341, loss = 0.000292
grad_step = 000342, loss = 0.000309
grad_step = 000343, loss = 0.000328
grad_step = 000344, loss = 0.000338
grad_step = 000345, loss = 0.000321
grad_step = 000346, loss = 0.000289
grad_step = 000347, loss = 0.000264
grad_step = 000348, loss = 0.000262
grad_step = 000349, loss = 0.000278
grad_step = 000350, loss = 0.000291
grad_step = 000351, loss = 0.000288
grad_step = 000352, loss = 0.000269
grad_step = 000353, loss = 0.000257
grad_step = 000354, loss = 0.000257
grad_step = 000355, loss = 0.000265
grad_step = 000356, loss = 0.000272
grad_step = 000357, loss = 0.000270
grad_step = 000358, loss = 0.000261
grad_step = 000359, loss = 0.000254
grad_step = 000360, loss = 0.000253
grad_step = 000361, loss = 0.000256
grad_step = 000362, loss = 0.000260
grad_step = 000363, loss = 0.000260
grad_step = 000364, loss = 0.000256
grad_step = 000365, loss = 0.000252
grad_step = 000366, loss = 0.000249
grad_step = 000367, loss = 0.000252
grad_step = 000368, loss = 0.000256
grad_step = 000369, loss = 0.000263
grad_step = 000370, loss = 0.000265
grad_step = 000371, loss = 0.000268
grad_step = 000372, loss = 0.000270
grad_step = 000373, loss = 0.000276
grad_step = 000374, loss = 0.000286
grad_step = 000375, loss = 0.000298
grad_step = 000376, loss = 0.000304
grad_step = 000377, loss = 0.000297
grad_step = 000378, loss = 0.000278
grad_step = 000379, loss = 0.000262
grad_step = 000380, loss = 0.000255
grad_step = 000381, loss = 0.000263
grad_step = 000382, loss = 0.000273
grad_step = 000383, loss = 0.000272
grad_step = 000384, loss = 0.000263
grad_step = 000385, loss = 0.000251
grad_step = 000386, loss = 0.000247
grad_step = 000387, loss = 0.000249
grad_step = 000388, loss = 0.000255
grad_step = 000389, loss = 0.000253
grad_step = 000390, loss = 0.000247
grad_step = 000391, loss = 0.000239
grad_step = 000392, loss = 0.000238
grad_step = 000393, loss = 0.000242
grad_step = 000394, loss = 0.000248
grad_step = 000395, loss = 0.000252
grad_step = 000396, loss = 0.000248
grad_step = 000397, loss = 0.000242
grad_step = 000398, loss = 0.000237
grad_step = 000399, loss = 0.000235
grad_step = 000400, loss = 0.000237
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000239
grad_step = 000402, loss = 0.000242
grad_step = 000403, loss = 0.000243
grad_step = 000404, loss = 0.000242
grad_step = 000405, loss = 0.000237
grad_step = 000406, loss = 0.000233
grad_step = 000407, loss = 0.000231
grad_step = 000408, loss = 0.000231
grad_step = 000409, loss = 0.000234
grad_step = 000410, loss = 0.000238
grad_step = 000411, loss = 0.000240
grad_step = 000412, loss = 0.000241
grad_step = 000413, loss = 0.000241
grad_step = 000414, loss = 0.000242
grad_step = 000415, loss = 0.000243
grad_step = 000416, loss = 0.000244
grad_step = 000417, loss = 0.000245
grad_step = 000418, loss = 0.000245
grad_step = 000419, loss = 0.000248
grad_step = 000420, loss = 0.000255
grad_step = 000421, loss = 0.000264
grad_step = 000422, loss = 0.000270
grad_step = 000423, loss = 0.000262
grad_step = 000424, loss = 0.000248
grad_step = 000425, loss = 0.000236
grad_step = 000426, loss = 0.000235
grad_step = 000427, loss = 0.000232
grad_step = 000428, loss = 0.000230
grad_step = 000429, loss = 0.000227
grad_step = 000430, loss = 0.000229
grad_step = 000431, loss = 0.000233
grad_step = 000432, loss = 0.000234
grad_step = 000433, loss = 0.000235
grad_step = 000434, loss = 0.000229
grad_step = 000435, loss = 0.000225
grad_step = 000436, loss = 0.000223
grad_step = 000437, loss = 0.000225
grad_step = 000438, loss = 0.000226
grad_step = 000439, loss = 0.000225
grad_step = 000440, loss = 0.000224
grad_step = 000441, loss = 0.000226
grad_step = 000442, loss = 0.000230
grad_step = 000443, loss = 0.000234
grad_step = 000444, loss = 0.000234
grad_step = 000445, loss = 0.000234
grad_step = 000446, loss = 0.000234
grad_step = 000447, loss = 0.000234
grad_step = 000448, loss = 0.000233
grad_step = 000449, loss = 0.000230
grad_step = 000450, loss = 0.000228
grad_step = 000451, loss = 0.000225
grad_step = 000452, loss = 0.000228
grad_step = 000453, loss = 0.000233
grad_step = 000454, loss = 0.000254
grad_step = 000455, loss = 0.000286
grad_step = 000456, loss = 0.000301
grad_step = 000457, loss = 0.000295
grad_step = 000458, loss = 0.000264
grad_step = 000459, loss = 0.000232
grad_step = 000460, loss = 0.000220
grad_step = 000461, loss = 0.000229
grad_step = 000462, loss = 0.000240
grad_step = 000463, loss = 0.000246
grad_step = 000464, loss = 0.000241
grad_step = 000465, loss = 0.000225
grad_step = 000466, loss = 0.000210
grad_step = 000467, loss = 0.000217
grad_step = 000468, loss = 0.000236
grad_step = 000469, loss = 0.000243
grad_step = 000470, loss = 0.000234
grad_step = 000471, loss = 0.000217
grad_step = 000472, loss = 0.000209
grad_step = 000473, loss = 0.000217
grad_step = 000474, loss = 0.000225
grad_step = 000475, loss = 0.000221
grad_step = 000476, loss = 0.000210
grad_step = 000477, loss = 0.000205
grad_step = 000478, loss = 0.000209
grad_step = 000479, loss = 0.000213
grad_step = 000480, loss = 0.000215
grad_step = 000481, loss = 0.000210
grad_step = 000482, loss = 0.000206
grad_step = 000483, loss = 0.000207
grad_step = 000484, loss = 0.000211
grad_step = 000485, loss = 0.000216
grad_step = 000486, loss = 0.000219
grad_step = 000487, loss = 0.000223
grad_step = 000488, loss = 0.000228
grad_step = 000489, loss = 0.000238
grad_step = 000490, loss = 0.000257
grad_step = 000491, loss = 0.000271
grad_step = 000492, loss = 0.000280
grad_step = 000493, loss = 0.000270
grad_step = 000494, loss = 0.000251
grad_step = 000495, loss = 0.000227
grad_step = 000496, loss = 0.000212
grad_step = 000497, loss = 0.000212
grad_step = 000498, loss = 0.000218
grad_step = 000499, loss = 0.000225
grad_step = 000500, loss = 0.000225
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
[[0.8783994  0.85242236 0.9366777  0.96012783 1.0013807 ]
 [0.84559816 0.9043288  0.9495226  1.0157156  0.9896745 ]
 [0.90994936 0.9235228  1.005233   0.98390484 0.9510527 ]
 [0.93166566 0.98922014 0.98893976 0.9407248  0.91296005]
 [1.0021981  0.9940831  0.9617467  0.91580886 0.8649434 ]
 [0.96976674 0.9535439  0.9251697  0.86212736 0.8620299 ]
 [0.93894064 0.91167164 0.86623454 0.85653436 0.81610847]
 [0.9006903  0.8451886  0.86069655 0.81253135 0.84541196]
 [0.82970977 0.8425892  0.8274094  0.83819443 0.8564567 ]
 [0.8453281  0.81458706 0.8518535  0.8525248  0.8224018 ]
 [0.80739594 0.8325295  0.8624402  0.81766886 0.9324019 ]
 [0.82705116 0.82018864 0.83514726 0.92895    0.9471929 ]
 [0.8707608  0.8466492  0.932601   0.959392   0.9963535 ]
 [0.84736866 0.90731394 0.95652246 1.0129137  0.975166  ]
 [0.91749424 0.9337445  1.0041339  0.9704855  0.93573886]
 [0.93953526 0.9965739  0.97970796 0.9229224  0.89096546]
 [1.0131412  0.99206114 0.94590175 0.89707136 0.8489319 ]
 [0.97415316 0.94180965 0.90476906 0.8414862  0.8485926 ]
 [0.93473375 0.9000516  0.84724057 0.8446515  0.8107434 ]
 [0.9047965  0.84226954 0.8481939  0.81301105 0.84441984]
 [0.8410727  0.84713715 0.8265316  0.8401455  0.8661753 ]
 [0.8638648  0.82517064 0.8514828  0.8573226  0.8279518 ]
 [0.8158414  0.8463297  0.8664104  0.8239256  0.9370978 ]
 [0.8335431  0.83310455 0.8387992  0.93190265 0.9510404 ]
 [0.8828052  0.85530484 0.9367665  0.966685   1.0063558 ]
 [0.85284054 0.90361506 0.94888973 1.0242084  1.0010985 ]
 [0.9204209  0.92618436 1.0090663  0.9990813  0.9624842 ]
 [0.9446216  0.9953438  0.99584657 0.9518478  0.9227361 ]
 [1.0170089  1.00459    0.96928304 0.92525893 0.8755543 ]
 [0.98381    0.96467614 0.93475485 0.8682172  0.8675601 ]
 [0.9475055  0.92010224 0.8727752  0.86119914 0.8233907 ]]

  #### Plot     ############################################### 
Saved image to ztest/model_tch/nbeats//n_beats_test.png.

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Already up to date.
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
