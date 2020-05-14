
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
Warning: Permanently added the RSA host key for IP address '192.30.255.112' to the list of known hosts.
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
[master 88bbe5b] ml_store
 1 file changed, 60 insertions(+)
 create mode 100644 log_testall/log_testall_2020-05-14-20-11_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py
To github.com:arita37/mlmodels_store.git
   fd63d85..88bbe5b  master -> master





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
[master 3bbccde] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   88bbe5b..3bbccde  master -> master





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
[master cd85f2d] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   3bbccde..cd85f2d  master -> master





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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         32          sequence_max[0][0]               
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
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         24          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-14 20:12:35.350004: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 20:12:35.354772: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-14 20:12:35.354932: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d9cc971930 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 20:12:35.354947: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 233
Trainable params: 233
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 1s 1ms/sample - loss: 0.2501 - binary_crossentropy: 0.6932 - val_loss: 0.2500 - val_binary_crossentropy: 0.6931

  #### metrics   #################################################### 
{'MSE': 0.24986265125653898}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         32          sequence_max[0][0]               
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
Total params: 233
Trainable params: 233
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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_max[0][0]               
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
Total params: 488
Trainable params: 488
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2512 - binary_crossentropy: 0.6956500/500 [==============================] - 1s 2ms/sample - loss: 0.2527 - binary_crossentropy: 0.7250 - val_loss: 0.2511 - val_binary_crossentropy: 0.6953

  #### metrics   #################################################### 
{'MSE': 0.25171023218936317}

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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_max[0][0]               
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
Total params: 488
Trainable params: 488
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         20          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_2[0][0]           
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
Total params: 627
Trainable params: 627
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 1s 2ms/sample - loss: 0.2502 - binary_crossentropy: 0.6935 - val_loss: 0.2499 - val_binary_crossentropy: 0.6930

  #### metrics   #################################################### 
{'MSE': 0.24993332307948385}

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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         20          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_2[0][0]           
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
Total params: 627
Trainable params: 627
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         28          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         16          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_2[0][0]           
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
Total params: 428
Trainable params: 428
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.4171 - binary_crossentropy: 4.5696500/500 [==============================] - 1s 3ms/sample - loss: 0.4044 - binary_crossentropy: 4.4825 - val_loss: 0.3920 - val_binary_crossentropy: 4.2625

  #### metrics   #################################################### 
{'MSE': 0.39778682337772486}

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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         28          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         16          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_2[0][0]           
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
Total params: 428
Trainable params: 428
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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
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
Total params: 173
Trainable params: 173
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.3281 - binary_crossentropy: 0.8854500/500 [==============================] - 2s 3ms/sample - loss: 0.3028 - binary_crossentropy: 0.8568 - val_loss: 0.3175 - val_binary_crossentropy: 0.8647

  #### metrics   #################################################### 
{'MSE': 0.30831416919037785}

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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
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
Total params: 173
Trainable params: 173
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-14 20:13:59.512710: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 20:13:59.514716: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 20:13:59.520188: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-14 20:13:59.530630: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-14 20:13:59.532380: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-14 20:13:59.533970: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 20:13:59.535402: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2437 - val_binary_crossentropy: 0.6805
2020-05-14 20:14:00.897830: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 20:14:00.899597: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 20:14:00.903937: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-14 20:14:00.912676: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-14 20:14:00.914190: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-14 20:14:00.915581: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 20:14:00.916821: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24041829135120096}

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
2020-05-14 20:14:25.450282: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 20:14:25.451626: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 20:14:25.455078: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-14 20:14:25.461204: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-14 20:14:25.462230: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-14 20:14:25.463175: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 20:14:25.464053: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2501 - val_binary_crossentropy: 0.6934
2020-05-14 20:14:27.061024: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 20:14:27.062176: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 20:14:27.064840: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-14 20:14:27.070312: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-14 20:14:27.071181: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-14 20:14:27.072106: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 20:14:27.072849: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.25009480180503135}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-14 20:15:02.096225: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 20:15:02.101431: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 20:15:02.117311: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-14 20:15:02.145573: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-14 20:15:02.150477: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-14 20:15:02.154986: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 20:15:02.159261: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.0506 - binary_crossentropy: 0.2548 - val_loss: 0.2513 - val_binary_crossentropy: 0.6957
2020-05-14 20:15:04.565199: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 20:15:04.570020: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 20:15:04.582668: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-14 20:15:04.608404: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-14 20:15:04.614543: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-14 20:15:04.618730: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 20:15:04.622831: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.23936093870229294}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 635
Trainable params: 635
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2595 - binary_crossentropy: 0.8447500/500 [==============================] - 4s 9ms/sample - loss: 0.2578 - binary_crossentropy: 0.8128 - val_loss: 0.2536 - val_binary_crossentropy: 0.7787

  #### metrics   #################################################### 
{'MSE': 0.2554725937114824}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 635
Trainable params: 635
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 2)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 2)         18          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 2)         6           sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         10          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         10          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         10          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_2[0][0]           
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
Total params: 242
Trainable params: 242
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.3958 - binary_crossentropy: 4.0017500/500 [==============================] - 5s 9ms/sample - loss: 0.3555 - binary_crossentropy: 3.2066 - val_loss: 0.3228 - val_binary_crossentropy: 2.7863

  #### metrics   #################################################### 
{'MSE': 0.33721971752673174}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 2)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 2)         18          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 2)         6           sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         10          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         10          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         10          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_2[0][0]           
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
Total params: 242
Trainable params: 242
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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_max[0][0]               
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
Total params: 1,894
Trainable params: 1,894
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2881 - binary_crossentropy: 0.9082500/500 [==============================] - 5s 10ms/sample - loss: 0.2724 - binary_crossentropy: 0.7953 - val_loss: 0.2575 - val_binary_crossentropy: 0.7098

  #### metrics   #################################################### 
{'MSE': 0.26412783849218663}

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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_max[0][0]               
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
regionsequence_sum (InputLayer) [(None, 1)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 6)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 2)]          0                                            
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
region_10sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 2, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 2, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 2, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 2, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 2, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 2, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 2, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 2, 1)         5           regionsequence_max[0][0]         
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
100/500 [=====>........................] - ETA: 9s - loss: 0.2566 - binary_crossentropy: 0.8365500/500 [==============================] - 6s 12ms/sample - loss: 0.2547 - binary_crossentropy: 0.8316 - val_loss: 0.2572 - val_binary_crossentropy: 0.9145

  #### metrics   #################################################### 
{'MSE': 0.2556754920631955}

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
regionsequence_sum (InputLayer) [(None, 1)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 6)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 2)]          0                                            
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
region_10sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 2, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 2, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 2, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 2, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 2, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 2, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 2, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 2, 1)         5           regionsequence_max[0][0]         
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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_max[0][0]               
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
Total params: 1,372
Trainable params: 1,372
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.3094 - binary_crossentropy: 0.8354500/500 [==============================] - 6s 12ms/sample - loss: 0.3067 - binary_crossentropy: 0.8857 - val_loss: 0.2927 - val_binary_crossentropy: 0.8213

  #### metrics   #################################################### 
{'MSE': 0.2967607341078326}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_max[0][0]               
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
Total params: 1,372
Trainable params: 1,372
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         16          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         16          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 6, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 7, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 6, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 7, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 6, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 7, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 6, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 7, 4)         12          sequence_mean[0][0]              
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           hash_10[0][0]                    
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
Total params: 2,916
Trainable params: 2,836
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2548 - binary_crossentropy: 0.7027500/500 [==============================] - 7s 13ms/sample - loss: 0.2631 - binary_crossentropy: 0.7202 - val_loss: 0.2546 - val_binary_crossentropy: 0.7024

  #### metrics   #################################################### 
{'MSE': 0.25700019955836206}

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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         16          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         16          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 6, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 7, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 6, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 7, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 6, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 7, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 6, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 7, 4)         12          sequence_mean[0][0]              
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           hash_10[0][0]                    
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
Total params: 2,916
Trainable params: 2,836
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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         36          sequence_max[0][0]               
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
Total params: 481
Trainable params: 481
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2506 - binary_crossentropy: 0.6944500/500 [==============================] - 7s 13ms/sample - loss: 0.2485 - binary_crossentropy: 0.6902 - val_loss: 0.2501 - val_binary_crossentropy: 0.6934

  #### metrics   #################################################### 
{'MSE': 0.25015505144872185}

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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         36          sequence_max[0][0]               
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
Total params: 481
Trainable params: 481
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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_44 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         28          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_125 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_194[0][0] 
                                                                 sequence_pooling_layer_195[0][0] 
                                                                 sequence_pooling_layer_196[0][0] 
                                                                 sequence_pooling_layer_197[0][0] 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
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
Total params: 2,014
Trainable params: 2,014
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 7s 14ms/sample - loss: 0.2500 - binary_crossentropy: 0.7184 - val_loss: 0.2528 - val_binary_crossentropy: 0.7512

  #### metrics   #################################################### 
{'MSE': 0.25126278565794435}

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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_44 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         28          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_125 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_194[0][0] 
                                                                 sequence_pooling_layer_195[0][0] 
                                                                 sequence_pooling_layer_196[0][0] 
                                                                 sequence_pooling_layer_197[0][0] 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
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
Total params: 2,014
Trainable params: 2,014
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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_47 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_90 (Concatenate)    (None, 1, 20)        0           no_mask_130[0][0]                
                                                                 no_mask_130[1][0]                
                                                                 no_mask_130[2][0]                
                                                                 no_mask_130[3][0]                
                                                                 no_mask_130[4][0]                
__________________________________________________________________________________________________
no_mask_131 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 286
Trainable params: 286
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2621 - binary_crossentropy: 0.9774500/500 [==============================] - 7s 14ms/sample - loss: 0.2677 - binary_crossentropy: 1.0952 - val_loss: 0.2892 - val_binary_crossentropy: 1.3225

  #### metrics   #################################################### 
{'MSE': 0.2776759571840125}

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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_47 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_90 (Concatenate)    (None, 1, 20)        0           no_mask_130[0][0]                
                                                                 no_mask_130[1][0]                
                                                                 no_mask_130[2][0]                
                                                                 no_mask_130[3][0]                
                                                                 no_mask_130[4][0]                
__________________________________________________________________________________________________
no_mask_131 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 286
Trainable params: 286
Non-trainable params: 0
__________________________________________________________________________________________________

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   cd85f2d..b96aa53  master     -> origin/master
Updating cd85f2d..b96aa53
Fast-forward
 error_list/20200514/list_log_json_20200514.md      | 1146 ++++++------
 error_list/20200514/list_log_jupyter_20200514.md   | 1773 +++++++++---------
 error_list/20200514/list_log_testall_20200514.md   |  799 +-------
 ...-14_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py | 1970 ++++++++++++++++++++
 ...-10_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py |  631 +++++++
 5 files changed, 4065 insertions(+), 2254 deletions(-)
 create mode 100644 log_jupyter/log_jupyter_2020-05-14-20-14_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py
 create mode 100644 log_pullrequest/log_pr_2020-05-14-20-10_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py
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
[master 6dfba91] ml_store
 1 file changed, 5672 insertions(+)
To github.com:arita37/mlmodels_store.git
   b96aa53..6dfba91  master -> master





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
[master 49fdeb0] ml_store
 1 file changed, 50 insertions(+)
To github.com:arita37/mlmodels_store.git
   6dfba91..49fdeb0  master -> master





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
[master d13e5eb] ml_store
 1 file changed, 46 insertions(+)
To github.com:arita37/mlmodels_store.git
   49fdeb0..d13e5eb  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//Autokeras.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//Autokeras.py", line 12, in <module>
    import autokeras as ak
ModuleNotFoundError: No module named 'autokeras'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Warning: Permanently added the RSA host key for IP address '192.30.255.113' to the list of known hosts.
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
[master 40d9bfc] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   d13e5eb..40d9bfc  master -> master





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

2020-05-14 20:28:37.292807: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 20:28:37.297779: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-14 20:28:37.298528: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5574089a0620 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 20:28:37.298546: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 5s - loss: 1.3862
256/354 [====================>.........] - ETA: 2s - loss: 1.2702
354/354 [==============================] - 10s 27ms/step - loss: 1.2527 - val_loss: 2.4196

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
[master 3a59f90] ml_store
 1 file changed, 149 insertions(+)
To github.com:arita37/mlmodels_store.git
   40d9bfc..3a59f90  master -> master





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
[master 2c95822] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   3a59f90..2c95822  master -> master





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
[master fe0a162] ml_store
 1 file changed, 44 insertions(+)
To github.com:arita37/mlmodels_store.git
   2c95822..fe0a162  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 46s
   57344/17464789 [..............................] - ETA: 40s
   90112/17464789 [..............................] - ETA: 38s
  122880/17464789 [..............................] - ETA: 37s
  139264/17464789 [..............................] - ETA: 41s
  212992/17464789 [..............................] - ETA: 32s
  303104/17464789 [..............................] - ETA: 26s
  368640/17464789 [..............................] - ETA: 24s
  458752/17464789 [..............................] - ETA: 22s
  540672/17464789 [..............................] - ETA: 20s
  630784/17464789 [>.............................] - ETA: 19s
  720896/17464789 [>.............................] - ETA: 18s
  802816/17464789 [>.............................] - ETA: 17s
  892928/17464789 [>.............................] - ETA: 17s
  974848/17464789 [>.............................] - ETA: 16s
 1081344/17464789 [>.............................] - ETA: 16s
 1171456/17464789 [=>............................] - ETA: 15s
 1269760/17464789 [=>............................] - ETA: 15s
 1359872/17464789 [=>............................] - ETA: 14s
 1466368/17464789 [=>............................] - ETA: 14s
 1556480/17464789 [=>............................] - ETA: 14s
 1654784/17464789 [=>............................] - ETA: 13s
 1761280/17464789 [==>...........................] - ETA: 13s
 1851392/17464789 [==>...........................] - ETA: 13s
 1949696/17464789 [==>...........................] - ETA: 13s
 2056192/17464789 [==>...........................] - ETA: 12s
 2162688/17464789 [==>...........................] - ETA: 12s
 2269184/17464789 [==>...........................] - ETA: 12s
 2367488/17464789 [===>..........................] - ETA: 12s
 2473984/17464789 [===>..........................] - ETA: 12s
 2580480/17464789 [===>..........................] - ETA: 11s
 2686976/17464789 [===>..........................] - ETA: 11s
 2785280/17464789 [===>..........................] - ETA: 11s
 2891776/17464789 [===>..........................] - ETA: 11s
 3014656/17464789 [====>.........................] - ETA: 11s
 3121152/17464789 [====>.........................] - ETA: 10s
 3227648/17464789 [====>.........................] - ETA: 10s
 3325952/17464789 [====>.........................] - ETA: 10s
 3448832/17464789 [====>.........................] - ETA: 10s
 3555328/17464789 [=====>........................] - ETA: 10s
 3661824/17464789 [=====>........................] - ETA: 10s
 3776512/17464789 [=====>........................] - ETA: 10s
 3883008/17464789 [=====>........................] - ETA: 9s 
 3989504/17464789 [=====>........................] - ETA: 9s
 4112384/17464789 [======>.......................] - ETA: 9s
 4218880/17464789 [======>.......................] - ETA: 9s
 4317184/17464789 [======>.......................] - ETA: 9s
 4440064/17464789 [======>.......................] - ETA: 9s
 4546560/17464789 [======>.......................] - ETA: 9s
 4669440/17464789 [=======>......................] - ETA: 9s
 4775936/17464789 [=======>......................] - ETA: 8s
 4874240/17464789 [=======>......................] - ETA: 8s
 4997120/17464789 [=======>......................] - ETA: 8s
 5103616/17464789 [=======>......................] - ETA: 8s
 5210112/17464789 [=======>......................] - ETA: 8s
 5332992/17464789 [========>.....................] - ETA: 8s
 5431296/17464789 [========>.....................] - ETA: 8s
 5554176/17464789 [========>.....................] - ETA: 8s
 5660672/17464789 [========>.....................] - ETA: 8s
 5783552/17464789 [========>.....................] - ETA: 7s
 5890048/17464789 [=========>....................] - ETA: 7s
 6012928/17464789 [=========>....................] - ETA: 7s
 6127616/17464789 [=========>....................] - ETA: 7s
 6234112/17464789 [=========>....................] - ETA: 7s
 6356992/17464789 [=========>....................] - ETA: 7s
 6479872/17464789 [==========>...................] - ETA: 7s
 6602752/17464789 [==========>...................] - ETA: 7s
 6709248/17464789 [==========>...................] - ETA: 7s
 6823936/17464789 [==========>...................] - ETA: 7s
 6946816/17464789 [==========>...................] - ETA: 6s
 7069696/17464789 [===========>..................] - ETA: 6s
 7192576/17464789 [===========>..................] - ETA: 6s
 7315456/17464789 [===========>..................] - ETA: 6s
 7438336/17464789 [===========>..................] - ETA: 6s
 7561216/17464789 [===========>..................] - ETA: 6s
 7684096/17464789 [============>.................] - ETA: 6s
 7798784/17464789 [============>.................] - ETA: 6s
 7921664/17464789 [============>.................] - ETA: 6s
 8044544/17464789 [============>.................] - ETA: 6s
 8183808/17464789 [=============>................] - ETA: 5s
 8306688/17464789 [=============>................] - ETA: 5s
 8429568/17464789 [=============>................] - ETA: 5s
 8552448/17464789 [=============>................] - ETA: 5s
 8691712/17464789 [=============>................] - ETA: 5s
 8814592/17464789 [==============>...............] - ETA: 5s
 8937472/17464789 [==============>...............] - ETA: 5s
 9076736/17464789 [==============>...............] - ETA: 5s
 9191424/17464789 [==============>...............] - ETA: 5s
 9330688/17464789 [===============>..............] - ETA: 5s
 9453568/17464789 [===============>..............] - ETA: 5s
 9592832/17464789 [===============>..............] - ETA: 4s
 9715712/17464789 [===============>..............] - ETA: 4s
 9854976/17464789 [===============>..............] - ETA: 4s
 9977856/17464789 [================>.............] - ETA: 4s
10117120/17464789 [================>.............] - ETA: 4s
10256384/17464789 [================>.............] - ETA: 4s
10395648/17464789 [================>.............] - ETA: 4s
10518528/17464789 [=================>............] - ETA: 4s
10657792/17464789 [=================>............] - ETA: 4s
10797056/17464789 [=================>............] - ETA: 4s
10936320/17464789 [=================>............] - ETA: 3s
11075584/17464789 [==================>...........] - ETA: 3s
11214848/17464789 [==================>...........] - ETA: 3s
11354112/17464789 [==================>...........] - ETA: 3s
11493376/17464789 [==================>...........] - ETA: 3s
11649024/17464789 [===================>..........] - ETA: 3s
11788288/17464789 [===================>..........] - ETA: 3s
11927552/17464789 [===================>..........] - ETA: 3s
12066816/17464789 [===================>..........] - ETA: 3s
12222464/17464789 [===================>..........] - ETA: 3s
12361728/17464789 [====================>.........] - ETA: 3s
12517376/17464789 [====================>.........] - ETA: 2s
12656640/17464789 [====================>.........] - ETA: 2s
12812288/17464789 [=====================>........] - ETA: 2s
12951552/17464789 [=====================>........] - ETA: 2s
13115392/17464789 [=====================>........] - ETA: 2s
13254656/17464789 [=====================>........] - ETA: 2s
13410304/17464789 [======================>.......] - ETA: 2s
13565952/17464789 [======================>.......] - ETA: 2s
13705216/17464789 [======================>.......] - ETA: 2s
13860864/17464789 [======================>.......] - ETA: 2s
14016512/17464789 [=======================>......] - ETA: 1s
14172160/17464789 [=======================>......] - ETA: 1s
14327808/17464789 [=======================>......] - ETA: 1s
14483456/17464789 [=======================>......] - ETA: 1s
14647296/17464789 [========================>.....] - ETA: 1s
14802944/17464789 [========================>.....] - ETA: 1s
14958592/17464789 [========================>.....] - ETA: 1s
15114240/17464789 [========================>.....] - ETA: 1s
15269888/17464789 [=========================>....] - ETA: 1s
15425536/17464789 [=========================>....] - ETA: 1s
15597568/17464789 [=========================>....] - ETA: 1s
15761408/17464789 [==========================>...] - ETA: 0s
15917056/17464789 [==========================>...] - ETA: 0s
16089088/17464789 [==========================>...] - ETA: 0s
16261120/17464789 [==========================>...] - ETA: 0s
16433152/17464789 [===========================>..] - ETA: 0s
16613376/17464789 [===========================>..] - ETA: 0s
16785408/17464789 [===========================>..] - ETA: 0s
16957440/17464789 [============================>.] - ETA: 0s
17154048/17464789 [============================>.] - ETA: 0s
17326080/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 9s 1us/step
Pad sequences (samples x time)...

  #### Model init, fit   ############################################# 
Using TensorFlow backend.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 20:29:45.691427: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 20:29:45.695975: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-14 20:29:45.696136: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555632094ed0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 20:29:45.696151: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 11s - loss: 7.7586 - accuracy: 0.4940
 2000/25000 [=>............................] - ETA: 8s - loss: 7.9426 - accuracy: 0.4820 
 3000/25000 [==>...........................] - ETA: 6s - loss: 8.0295 - accuracy: 0.4763
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.9695 - accuracy: 0.4803
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.8537 - accuracy: 0.4878
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.8174 - accuracy: 0.4902
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7718 - accuracy: 0.4931
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7337 - accuracy: 0.4956
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6990 - accuracy: 0.4979
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6620 - accuracy: 0.5003
11000/25000 [============>.................] - ETA: 3s - loss: 7.6457 - accuracy: 0.5014
12000/25000 [=============>................] - ETA: 3s - loss: 7.6232 - accuracy: 0.5028
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6053 - accuracy: 0.5040
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6206 - accuracy: 0.5030
15000/25000 [=================>............] - ETA: 2s - loss: 7.6349 - accuracy: 0.5021
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6618 - accuracy: 0.5003
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6811 - accuracy: 0.4991
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7015 - accuracy: 0.4977
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6876 - accuracy: 0.4986
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6781 - accuracy: 0.4992
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6761 - accuracy: 0.4994
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6827 - accuracy: 0.4990
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6693 - accuracy: 0.4998
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6564 - accuracy: 0.5007
25000/25000 [==============================] - 7s 276us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7f148d470c50>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f1438bb82e8> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 7.8046 - accuracy: 0.4910
 2000/25000 [=>............................] - ETA: 7s - loss: 7.6743 - accuracy: 0.4995 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7126 - accuracy: 0.4970
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.8276 - accuracy: 0.4895
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7678 - accuracy: 0.4934
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.7280 - accuracy: 0.4960
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6951 - accuracy: 0.4981
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7088 - accuracy: 0.4972
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7058 - accuracy: 0.4974
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7280 - accuracy: 0.4960
11000/25000 [============>.................] - ETA: 3s - loss: 7.7377 - accuracy: 0.4954
12000/25000 [=============>................] - ETA: 3s - loss: 7.7305 - accuracy: 0.4958
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7197 - accuracy: 0.4965
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7017 - accuracy: 0.4977
15000/25000 [=================>............] - ETA: 2s - loss: 7.6942 - accuracy: 0.4982
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6781 - accuracy: 0.4992
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7018 - accuracy: 0.4977
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7237 - accuracy: 0.4963
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7029 - accuracy: 0.4976
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6981 - accuracy: 0.4979
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6827 - accuracy: 0.4990
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6820 - accuracy: 0.4990
25000/25000 [==============================] - 7s 275us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 11s - loss: 7.7740 - accuracy: 0.4930
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7663 - accuracy: 0.4935 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6257 - accuracy: 0.5027
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6666 - accuracy: 0.5000
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6820 - accuracy: 0.4990
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6615 - accuracy: 0.5003
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6820 - accuracy: 0.4990
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7203 - accuracy: 0.4965
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7365 - accuracy: 0.4954
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6927 - accuracy: 0.4983
11000/25000 [============>.................] - ETA: 3s - loss: 7.6861 - accuracy: 0.4987
12000/25000 [=============>................] - ETA: 3s - loss: 7.6768 - accuracy: 0.4993
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6678 - accuracy: 0.4999
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6754 - accuracy: 0.4994
15000/25000 [=================>............] - ETA: 2s - loss: 7.6472 - accuracy: 0.5013
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6407 - accuracy: 0.5017
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6585 - accuracy: 0.5005
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6700 - accuracy: 0.4998
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6884 - accuracy: 0.4986
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6689 - accuracy: 0.4999
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6820 - accuracy: 0.4990
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6743 - accuracy: 0.4995
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6640 - accuracy: 0.5002
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6634 - accuracy: 0.5002
25000/25000 [==============================] - 7s 279us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
   fe0a162..99598cf  master     -> origin/master
Updating fe0a162..99598cf
Fast-forward
 error_list/20200514/list_log_benchmark_20200514.md |  166 +-
 error_list/20200514/list_log_json_20200514.md      | 1146 ++++++-------
 error_list/20200514/list_log_jupyter_20200514.md   | 1754 ++++++++++----------
 error_list/20200514/list_log_testall_20200514.md   |   89 +
 4 files changed, 1617 insertions(+), 1538 deletions(-)
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
[master 78b0ff6] ml_store
 1 file changed, 464 insertions(+)
To github.com:arita37/mlmodels_store.git
   99598cf..78b0ff6  master -> master





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
[master ea14fa0] ml_store
 1 file changed, 125 insertions(+)
To github.com:arita37/mlmodels_store.git
   78b0ff6..ea14fa0  master -> master





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
   24576/11490434 [..............................] - ETA: 30s
   57344/11490434 [..............................] - ETA: 26s
   90112/11490434 [..............................] - ETA: 25s
  163840/11490434 [..............................] - ETA: 18s
  352256/11490434 [..............................] - ETA: 10s
  720896/11490434 [>.............................] - ETA: 5s 
 1433600/11490434 [==>...........................] - ETA: 3s
 2883584/11490434 [======>.......................] - ETA: 1s
 5783552/11490434 [==============>...............] - ETA: 0s
 8814592/11490434 [======================>.......] - ETA: 0s
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

   32/60000 [..............................] - ETA: 8:07 - loss: 2.3045 - categorical_accuracy: 0.1250
   64/60000 [..............................] - ETA: 4:51 - loss: 2.2701 - categorical_accuracy: 0.1406
  128/60000 [..............................] - ETA: 3:11 - loss: 2.2322 - categorical_accuracy: 0.1641
  192/60000 [..............................] - ETA: 2:35 - loss: 2.1408 - categorical_accuracy: 0.2500
  256/60000 [..............................] - ETA: 2:17 - loss: 2.0787 - categorical_accuracy: 0.2773
  320/60000 [..............................] - ETA: 2:08 - loss: 2.0325 - categorical_accuracy: 0.3156
  384/60000 [..............................] - ETA: 2:02 - loss: 1.9821 - categorical_accuracy: 0.3333
  448/60000 [..............................] - ETA: 1:57 - loss: 1.9145 - categorical_accuracy: 0.3616
  512/60000 [..............................] - ETA: 1:52 - loss: 1.9153 - categorical_accuracy: 0.3574
  576/60000 [..............................] - ETA: 1:49 - loss: 1.8689 - categorical_accuracy: 0.3785
  640/60000 [..............................] - ETA: 1:46 - loss: 1.8031 - categorical_accuracy: 0.4047
  704/60000 [..............................] - ETA: 1:44 - loss: 1.7507 - categorical_accuracy: 0.4247
  768/60000 [..............................] - ETA: 1:42 - loss: 1.7132 - categorical_accuracy: 0.4414
  832/60000 [..............................] - ETA: 1:40 - loss: 1.6486 - categorical_accuracy: 0.4615
  896/60000 [..............................] - ETA: 1:39 - loss: 1.6107 - categorical_accuracy: 0.4721
  960/60000 [..............................] - ETA: 1:37 - loss: 1.5731 - categorical_accuracy: 0.4896
 1024/60000 [..............................] - ETA: 1:37 - loss: 1.5213 - categorical_accuracy: 0.5049
 1088/60000 [..............................] - ETA: 1:36 - loss: 1.4690 - categorical_accuracy: 0.5230
 1152/60000 [..............................] - ETA: 1:35 - loss: 1.4286 - categorical_accuracy: 0.5347
 1216/60000 [..............................] - ETA: 1:34 - loss: 1.3951 - categorical_accuracy: 0.5444
 1280/60000 [..............................] - ETA: 1:33 - loss: 1.3621 - categorical_accuracy: 0.5547
 1344/60000 [..............................] - ETA: 1:32 - loss: 1.3287 - categorical_accuracy: 0.5647
 1408/60000 [..............................] - ETA: 1:31 - loss: 1.3069 - categorical_accuracy: 0.5717
 1472/60000 [..............................] - ETA: 1:31 - loss: 1.2872 - categorical_accuracy: 0.5788
 1536/60000 [..............................] - ETA: 1:30 - loss: 1.2564 - categorical_accuracy: 0.5898
 1600/60000 [..............................] - ETA: 1:30 - loss: 1.2370 - categorical_accuracy: 0.5981
 1664/60000 [..............................] - ETA: 1:30 - loss: 1.2126 - categorical_accuracy: 0.6052
 1728/60000 [..............................] - ETA: 1:29 - loss: 1.2023 - categorical_accuracy: 0.6065
 1792/60000 [..............................] - ETA: 1:29 - loss: 1.1794 - categorical_accuracy: 0.6155
 1856/60000 [..............................] - ETA: 1:29 - loss: 1.1746 - categorical_accuracy: 0.6196
 1920/60000 [..............................] - ETA: 1:28 - loss: 1.1518 - categorical_accuracy: 0.6286
 1984/60000 [..............................] - ETA: 1:28 - loss: 1.1305 - categorical_accuracy: 0.6341
 2048/60000 [>.............................] - ETA: 1:27 - loss: 1.1092 - categorical_accuracy: 0.6406
 2112/60000 [>.............................] - ETA: 1:27 - loss: 1.0939 - categorical_accuracy: 0.6468
 2176/60000 [>.............................] - ETA: 1:27 - loss: 1.0818 - categorical_accuracy: 0.6507
 2240/60000 [>.............................] - ETA: 1:26 - loss: 1.0647 - categorical_accuracy: 0.6580
 2304/60000 [>.............................] - ETA: 1:26 - loss: 1.0469 - categorical_accuracy: 0.6645
 2368/60000 [>.............................] - ETA: 1:26 - loss: 1.0377 - categorical_accuracy: 0.6668
 2432/60000 [>.............................] - ETA: 1:26 - loss: 1.0192 - categorical_accuracy: 0.6743
 2496/60000 [>.............................] - ETA: 1:25 - loss: 1.0070 - categorical_accuracy: 0.6771
 2560/60000 [>.............................] - ETA: 1:25 - loss: 0.9928 - categorical_accuracy: 0.6816
 2624/60000 [>.............................] - ETA: 1:25 - loss: 0.9789 - categorical_accuracy: 0.6867
 2688/60000 [>.............................] - ETA: 1:25 - loss: 0.9681 - categorical_accuracy: 0.6908
 2752/60000 [>.............................] - ETA: 1:25 - loss: 0.9542 - categorical_accuracy: 0.6959
 2816/60000 [>.............................] - ETA: 1:24 - loss: 0.9403 - categorical_accuracy: 0.7003
 2880/60000 [>.............................] - ETA: 1:24 - loss: 0.9259 - categorical_accuracy: 0.7052
 2944/60000 [>.............................] - ETA: 1:24 - loss: 0.9190 - categorical_accuracy: 0.7089
 3008/60000 [>.............................] - ETA: 1:24 - loss: 0.9106 - categorical_accuracy: 0.7124
 3072/60000 [>.............................] - ETA: 1:23 - loss: 0.9002 - categorical_accuracy: 0.7155
 3136/60000 [>.............................] - ETA: 1:23 - loss: 0.8906 - categorical_accuracy: 0.7178
 3200/60000 [>.............................] - ETA: 1:23 - loss: 0.8793 - categorical_accuracy: 0.7209
 3264/60000 [>.............................] - ETA: 1:23 - loss: 0.8690 - categorical_accuracy: 0.7243
 3328/60000 [>.............................] - ETA: 1:23 - loss: 0.8585 - categorical_accuracy: 0.7281
 3392/60000 [>.............................] - ETA: 1:23 - loss: 0.8516 - categorical_accuracy: 0.7305
 3456/60000 [>.............................] - ETA: 1:23 - loss: 0.8422 - categorical_accuracy: 0.7344
 3520/60000 [>.............................] - ETA: 1:22 - loss: 0.8365 - categorical_accuracy: 0.7361
 3584/60000 [>.............................] - ETA: 1:22 - loss: 0.8295 - categorical_accuracy: 0.7380
 3648/60000 [>.............................] - ETA: 1:22 - loss: 0.8234 - categorical_accuracy: 0.7399
 3712/60000 [>.............................] - ETA: 1:22 - loss: 0.8159 - categorical_accuracy: 0.7419
 3776/60000 [>.............................] - ETA: 1:22 - loss: 0.8042 - categorical_accuracy: 0.7455
 3840/60000 [>.............................] - ETA: 1:22 - loss: 0.7960 - categorical_accuracy: 0.7479
 3904/60000 [>.............................] - ETA: 1:22 - loss: 0.7884 - categorical_accuracy: 0.7503
 3968/60000 [>.............................] - ETA: 1:22 - loss: 0.7796 - categorical_accuracy: 0.7520
 4032/60000 [=>............................] - ETA: 1:21 - loss: 0.7732 - categorical_accuracy: 0.7535
 4096/60000 [=>............................] - ETA: 1:21 - loss: 0.7703 - categorical_accuracy: 0.7546
 4160/60000 [=>............................] - ETA: 1:21 - loss: 0.7617 - categorical_accuracy: 0.7575
 4224/60000 [=>............................] - ETA: 1:21 - loss: 0.7563 - categorical_accuracy: 0.7597
 4288/60000 [=>............................] - ETA: 1:21 - loss: 0.7502 - categorical_accuracy: 0.7614
 4352/60000 [=>............................] - ETA: 1:21 - loss: 0.7480 - categorical_accuracy: 0.7622
 4416/60000 [=>............................] - ETA: 1:20 - loss: 0.7408 - categorical_accuracy: 0.7643
 4480/60000 [=>............................] - ETA: 1:20 - loss: 0.7358 - categorical_accuracy: 0.7661
 4544/60000 [=>............................] - ETA: 1:20 - loss: 0.7298 - categorical_accuracy: 0.7685
 4608/60000 [=>............................] - ETA: 1:20 - loss: 0.7220 - categorical_accuracy: 0.7711
 4672/60000 [=>............................] - ETA: 1:20 - loss: 0.7179 - categorical_accuracy: 0.7723
 4736/60000 [=>............................] - ETA: 1:20 - loss: 0.7115 - categorical_accuracy: 0.7743
 4800/60000 [=>............................] - ETA: 1:20 - loss: 0.7074 - categorical_accuracy: 0.7763
 4864/60000 [=>............................] - ETA: 1:20 - loss: 0.7016 - categorical_accuracy: 0.7780
 4928/60000 [=>............................] - ETA: 1:20 - loss: 0.6958 - categorical_accuracy: 0.7802
 4960/60000 [=>............................] - ETA: 1:20 - loss: 0.6930 - categorical_accuracy: 0.7808
 5024/60000 [=>............................] - ETA: 1:20 - loss: 0.6880 - categorical_accuracy: 0.7822
 5056/60000 [=>............................] - ETA: 1:20 - loss: 0.6854 - categorical_accuracy: 0.7832
 5120/60000 [=>............................] - ETA: 1:20 - loss: 0.6823 - categorical_accuracy: 0.7842
 5184/60000 [=>............................] - ETA: 1:19 - loss: 0.6770 - categorical_accuracy: 0.7861
 5248/60000 [=>............................] - ETA: 1:19 - loss: 0.6738 - categorical_accuracy: 0.7866
 5312/60000 [=>............................] - ETA: 1:19 - loss: 0.6690 - categorical_accuracy: 0.7884
 5376/60000 [=>............................] - ETA: 1:19 - loss: 0.6642 - categorical_accuracy: 0.7896
 5440/60000 [=>............................] - ETA: 1:19 - loss: 0.6598 - categorical_accuracy: 0.7919
 5504/60000 [=>............................] - ETA: 1:19 - loss: 0.6544 - categorical_accuracy: 0.7938
 5568/60000 [=>............................] - ETA: 1:18 - loss: 0.6494 - categorical_accuracy: 0.7951
 5632/60000 [=>............................] - ETA: 1:18 - loss: 0.6459 - categorical_accuracy: 0.7963
 5696/60000 [=>............................] - ETA: 1:18 - loss: 0.6413 - categorical_accuracy: 0.7978
 5760/60000 [=>............................] - ETA: 1:18 - loss: 0.6354 - categorical_accuracy: 0.7997
 5824/60000 [=>............................] - ETA: 1:18 - loss: 0.6315 - categorical_accuracy: 0.8003
 5888/60000 [=>............................] - ETA: 1:18 - loss: 0.6276 - categorical_accuracy: 0.8018
 5952/60000 [=>............................] - ETA: 1:18 - loss: 0.6244 - categorical_accuracy: 0.8024
 5984/60000 [=>............................] - ETA: 1:18 - loss: 0.6218 - categorical_accuracy: 0.8030
 6048/60000 [==>...........................] - ETA: 1:18 - loss: 0.6177 - categorical_accuracy: 0.8046
 6112/60000 [==>...........................] - ETA: 1:18 - loss: 0.6146 - categorical_accuracy: 0.8055
 6176/60000 [==>...........................] - ETA: 1:17 - loss: 0.6096 - categorical_accuracy: 0.8068
 6240/60000 [==>...........................] - ETA: 1:17 - loss: 0.6059 - categorical_accuracy: 0.8082
 6304/60000 [==>...........................] - ETA: 1:17 - loss: 0.6027 - categorical_accuracy: 0.8092
 6368/60000 [==>...........................] - ETA: 1:17 - loss: 0.5988 - categorical_accuracy: 0.8101
 6432/60000 [==>...........................] - ETA: 1:17 - loss: 0.5947 - categorical_accuracy: 0.8113
 6496/60000 [==>...........................] - ETA: 1:17 - loss: 0.5904 - categorical_accuracy: 0.8127
 6560/60000 [==>...........................] - ETA: 1:17 - loss: 0.5878 - categorical_accuracy: 0.8139
 6624/60000 [==>...........................] - ETA: 1:17 - loss: 0.5852 - categorical_accuracy: 0.8143
 6688/60000 [==>...........................] - ETA: 1:17 - loss: 0.5808 - categorical_accuracy: 0.8156
 6752/60000 [==>...........................] - ETA: 1:17 - loss: 0.5774 - categorical_accuracy: 0.8165
 6816/60000 [==>...........................] - ETA: 1:16 - loss: 0.5738 - categorical_accuracy: 0.8178
 6880/60000 [==>...........................] - ETA: 1:16 - loss: 0.5720 - categorical_accuracy: 0.8183
 6912/60000 [==>...........................] - ETA: 1:16 - loss: 0.5702 - categorical_accuracy: 0.8187
 6976/60000 [==>...........................] - ETA: 1:16 - loss: 0.5659 - categorical_accuracy: 0.8201
 7040/60000 [==>...........................] - ETA: 1:16 - loss: 0.5625 - categorical_accuracy: 0.8213
 7104/60000 [==>...........................] - ETA: 1:16 - loss: 0.5593 - categorical_accuracy: 0.8222
 7168/60000 [==>...........................] - ETA: 1:16 - loss: 0.5558 - categorical_accuracy: 0.8234
 7232/60000 [==>...........................] - ETA: 1:16 - loss: 0.5539 - categorical_accuracy: 0.8245
 7296/60000 [==>...........................] - ETA: 1:16 - loss: 0.5527 - categorical_accuracy: 0.8247
 7360/60000 [==>...........................] - ETA: 1:15 - loss: 0.5499 - categorical_accuracy: 0.8258
 7424/60000 [==>...........................] - ETA: 1:15 - loss: 0.5461 - categorical_accuracy: 0.8270
 7488/60000 [==>...........................] - ETA: 1:15 - loss: 0.5427 - categorical_accuracy: 0.8280
 7552/60000 [==>...........................] - ETA: 1:15 - loss: 0.5408 - categorical_accuracy: 0.8287
 7616/60000 [==>...........................] - ETA: 1:15 - loss: 0.5397 - categorical_accuracy: 0.8290
 7680/60000 [==>...........................] - ETA: 1:15 - loss: 0.5391 - categorical_accuracy: 0.8294
 7744/60000 [==>...........................] - ETA: 1:15 - loss: 0.5364 - categorical_accuracy: 0.8304
 7808/60000 [==>...........................] - ETA: 1:14 - loss: 0.5353 - categorical_accuracy: 0.8308
 7872/60000 [==>...........................] - ETA: 1:14 - loss: 0.5326 - categorical_accuracy: 0.8314
 7936/60000 [==>...........................] - ETA: 1:14 - loss: 0.5293 - categorical_accuracy: 0.8324
 8000/60000 [===>..........................] - ETA: 1:14 - loss: 0.5271 - categorical_accuracy: 0.8332
 8064/60000 [===>..........................] - ETA: 1:14 - loss: 0.5256 - categorical_accuracy: 0.8341
 8128/60000 [===>..........................] - ETA: 1:14 - loss: 0.5221 - categorical_accuracy: 0.8353
 8192/60000 [===>..........................] - ETA: 1:14 - loss: 0.5188 - categorical_accuracy: 0.8363
 8256/60000 [===>..........................] - ETA: 1:14 - loss: 0.5176 - categorical_accuracy: 0.8367
 8320/60000 [===>..........................] - ETA: 1:14 - loss: 0.5147 - categorical_accuracy: 0.8376
 8384/60000 [===>..........................] - ETA: 1:14 - loss: 0.5120 - categorical_accuracy: 0.8385
 8448/60000 [===>..........................] - ETA: 1:13 - loss: 0.5097 - categorical_accuracy: 0.8389
 8512/60000 [===>..........................] - ETA: 1:13 - loss: 0.5083 - categorical_accuracy: 0.8394
 8576/60000 [===>..........................] - ETA: 1:13 - loss: 0.5064 - categorical_accuracy: 0.8399
 8640/60000 [===>..........................] - ETA: 1:13 - loss: 0.5048 - categorical_accuracy: 0.8403
 8704/60000 [===>..........................] - ETA: 1:13 - loss: 0.5025 - categorical_accuracy: 0.8411
 8768/60000 [===>..........................] - ETA: 1:13 - loss: 0.5000 - categorical_accuracy: 0.8419
 8800/60000 [===>..........................] - ETA: 1:13 - loss: 0.4992 - categorical_accuracy: 0.8423
 8832/60000 [===>..........................] - ETA: 1:13 - loss: 0.4977 - categorical_accuracy: 0.8428
 8896/60000 [===>..........................] - ETA: 1:13 - loss: 0.4966 - categorical_accuracy: 0.8434
 8960/60000 [===>..........................] - ETA: 1:13 - loss: 0.4949 - categorical_accuracy: 0.8438
 9024/60000 [===>..........................] - ETA: 1:13 - loss: 0.4932 - categorical_accuracy: 0.8443
 9088/60000 [===>..........................] - ETA: 1:12 - loss: 0.4921 - categorical_accuracy: 0.8446
 9152/60000 [===>..........................] - ETA: 1:12 - loss: 0.4901 - categorical_accuracy: 0.8454
 9184/60000 [===>..........................] - ETA: 1:12 - loss: 0.4891 - categorical_accuracy: 0.8457
 9248/60000 [===>..........................] - ETA: 1:12 - loss: 0.4870 - categorical_accuracy: 0.8466
 9312/60000 [===>..........................] - ETA: 1:12 - loss: 0.4845 - categorical_accuracy: 0.8472
 9376/60000 [===>..........................] - ETA: 1:12 - loss: 0.4834 - categorical_accuracy: 0.8476
 9440/60000 [===>..........................] - ETA: 1:12 - loss: 0.4827 - categorical_accuracy: 0.8480
 9504/60000 [===>..........................] - ETA: 1:12 - loss: 0.4816 - categorical_accuracy: 0.8483
 9568/60000 [===>..........................] - ETA: 1:12 - loss: 0.4794 - categorical_accuracy: 0.8492
 9632/60000 [===>..........................] - ETA: 1:12 - loss: 0.4774 - categorical_accuracy: 0.8498
 9696/60000 [===>..........................] - ETA: 1:12 - loss: 0.4767 - categorical_accuracy: 0.8501
 9760/60000 [===>..........................] - ETA: 1:11 - loss: 0.4752 - categorical_accuracy: 0.8505
 9824/60000 [===>..........................] - ETA: 1:11 - loss: 0.4729 - categorical_accuracy: 0.8513
 9888/60000 [===>..........................] - ETA: 1:11 - loss: 0.4723 - categorical_accuracy: 0.8511
 9952/60000 [===>..........................] - ETA: 1:11 - loss: 0.4704 - categorical_accuracy: 0.8518
10016/60000 [====>.........................] - ETA: 1:11 - loss: 0.4685 - categorical_accuracy: 0.8523
10080/60000 [====>.........................] - ETA: 1:11 - loss: 0.4677 - categorical_accuracy: 0.8526
10144/60000 [====>.........................] - ETA: 1:11 - loss: 0.4652 - categorical_accuracy: 0.8533
10208/60000 [====>.........................] - ETA: 1:11 - loss: 0.4629 - categorical_accuracy: 0.8540
10272/60000 [====>.........................] - ETA: 1:11 - loss: 0.4615 - categorical_accuracy: 0.8546
10336/60000 [====>.........................] - ETA: 1:11 - loss: 0.4598 - categorical_accuracy: 0.8551
10400/60000 [====>.........................] - ETA: 1:10 - loss: 0.4577 - categorical_accuracy: 0.8559
10464/60000 [====>.........................] - ETA: 1:10 - loss: 0.4561 - categorical_accuracy: 0.8565
10528/60000 [====>.........................] - ETA: 1:10 - loss: 0.4545 - categorical_accuracy: 0.8570
10592/60000 [====>.........................] - ETA: 1:10 - loss: 0.4528 - categorical_accuracy: 0.8576
10656/60000 [====>.........................] - ETA: 1:10 - loss: 0.4512 - categorical_accuracy: 0.8580
10720/60000 [====>.........................] - ETA: 1:10 - loss: 0.4489 - categorical_accuracy: 0.8589
10784/60000 [====>.........................] - ETA: 1:10 - loss: 0.4483 - categorical_accuracy: 0.8590
10848/60000 [====>.........................] - ETA: 1:10 - loss: 0.4471 - categorical_accuracy: 0.8595
10912/60000 [====>.........................] - ETA: 1:10 - loss: 0.4453 - categorical_accuracy: 0.8601
10976/60000 [====>.........................] - ETA: 1:09 - loss: 0.4437 - categorical_accuracy: 0.8605
11008/60000 [====>.........................] - ETA: 1:09 - loss: 0.4427 - categorical_accuracy: 0.8609
11072/60000 [====>.........................] - ETA: 1:09 - loss: 0.4404 - categorical_accuracy: 0.8616
11136/60000 [====>.........................] - ETA: 1:09 - loss: 0.4387 - categorical_accuracy: 0.8620
11200/60000 [====>.........................] - ETA: 1:09 - loss: 0.4376 - categorical_accuracy: 0.8625
11264/60000 [====>.........................] - ETA: 1:09 - loss: 0.4365 - categorical_accuracy: 0.8627
11328/60000 [====>.........................] - ETA: 1:09 - loss: 0.4348 - categorical_accuracy: 0.8631
11392/60000 [====>.........................] - ETA: 1:09 - loss: 0.4335 - categorical_accuracy: 0.8635
11456/60000 [====>.........................] - ETA: 1:09 - loss: 0.4318 - categorical_accuracy: 0.8640
11520/60000 [====>.........................] - ETA: 1:09 - loss: 0.4301 - categorical_accuracy: 0.8646
11584/60000 [====>.........................] - ETA: 1:09 - loss: 0.4304 - categorical_accuracy: 0.8649
11648/60000 [====>.........................] - ETA: 1:08 - loss: 0.4291 - categorical_accuracy: 0.8654
11712/60000 [====>.........................] - ETA: 1:08 - loss: 0.4281 - categorical_accuracy: 0.8656
11776/60000 [====>.........................] - ETA: 1:08 - loss: 0.4269 - categorical_accuracy: 0.8659
11840/60000 [====>.........................] - ETA: 1:08 - loss: 0.4252 - categorical_accuracy: 0.8666
11904/60000 [====>.........................] - ETA: 1:08 - loss: 0.4234 - categorical_accuracy: 0.8672
11968/60000 [====>.........................] - ETA: 1:08 - loss: 0.4218 - categorical_accuracy: 0.8676
12032/60000 [=====>........................] - ETA: 1:08 - loss: 0.4218 - categorical_accuracy: 0.8679
12096/60000 [=====>........................] - ETA: 1:08 - loss: 0.4203 - categorical_accuracy: 0.8684
12160/60000 [=====>........................] - ETA: 1:08 - loss: 0.4192 - categorical_accuracy: 0.8687
12224/60000 [=====>........................] - ETA: 1:08 - loss: 0.4177 - categorical_accuracy: 0.8692
12256/60000 [=====>........................] - ETA: 1:08 - loss: 0.4178 - categorical_accuracy: 0.8692
12320/60000 [=====>........................] - ETA: 1:07 - loss: 0.4163 - categorical_accuracy: 0.8697
12384/60000 [=====>........................] - ETA: 1:07 - loss: 0.4148 - categorical_accuracy: 0.8702
12448/60000 [=====>........................] - ETA: 1:07 - loss: 0.4142 - categorical_accuracy: 0.8705
12512/60000 [=====>........................] - ETA: 1:07 - loss: 0.4125 - categorical_accuracy: 0.8712
12576/60000 [=====>........................] - ETA: 1:07 - loss: 0.4119 - categorical_accuracy: 0.8715
12608/60000 [=====>........................] - ETA: 1:07 - loss: 0.4111 - categorical_accuracy: 0.8717
12672/60000 [=====>........................] - ETA: 1:07 - loss: 0.4100 - categorical_accuracy: 0.8722
12736/60000 [=====>........................] - ETA: 1:07 - loss: 0.4083 - categorical_accuracy: 0.8727
12800/60000 [=====>........................] - ETA: 1:07 - loss: 0.4072 - categorical_accuracy: 0.8730
12864/60000 [=====>........................] - ETA: 1:07 - loss: 0.4062 - categorical_accuracy: 0.8731
12928/60000 [=====>........................] - ETA: 1:07 - loss: 0.4049 - categorical_accuracy: 0.8736
12992/60000 [=====>........................] - ETA: 1:06 - loss: 0.4036 - categorical_accuracy: 0.8738
13056/60000 [=====>........................] - ETA: 1:06 - loss: 0.4019 - categorical_accuracy: 0.8745
13120/60000 [=====>........................] - ETA: 1:06 - loss: 0.4004 - categorical_accuracy: 0.8748
13184/60000 [=====>........................] - ETA: 1:06 - loss: 0.3990 - categorical_accuracy: 0.8753
13248/60000 [=====>........................] - ETA: 1:06 - loss: 0.3977 - categorical_accuracy: 0.8758
13312/60000 [=====>........................] - ETA: 1:06 - loss: 0.3963 - categorical_accuracy: 0.8761
13376/60000 [=====>........................] - ETA: 1:06 - loss: 0.3963 - categorical_accuracy: 0.8762
13440/60000 [=====>........................] - ETA: 1:06 - loss: 0.3960 - categorical_accuracy: 0.8763
13504/60000 [=====>........................] - ETA: 1:06 - loss: 0.3950 - categorical_accuracy: 0.8767
13568/60000 [=====>........................] - ETA: 1:06 - loss: 0.3941 - categorical_accuracy: 0.8770
13632/60000 [=====>........................] - ETA: 1:05 - loss: 0.3929 - categorical_accuracy: 0.8773
13696/60000 [=====>........................] - ETA: 1:05 - loss: 0.3917 - categorical_accuracy: 0.8776
13760/60000 [=====>........................] - ETA: 1:05 - loss: 0.3906 - categorical_accuracy: 0.8779
13824/60000 [=====>........................] - ETA: 1:05 - loss: 0.3894 - categorical_accuracy: 0.8783
13888/60000 [=====>........................] - ETA: 1:05 - loss: 0.3883 - categorical_accuracy: 0.8787
13952/60000 [=====>........................] - ETA: 1:05 - loss: 0.3883 - categorical_accuracy: 0.8789
14016/60000 [======>.......................] - ETA: 1:05 - loss: 0.3872 - categorical_accuracy: 0.8793
14080/60000 [======>.......................] - ETA: 1:05 - loss: 0.3861 - categorical_accuracy: 0.8795
14144/60000 [======>.......................] - ETA: 1:05 - loss: 0.3855 - categorical_accuracy: 0.8799
14208/60000 [======>.......................] - ETA: 1:05 - loss: 0.3846 - categorical_accuracy: 0.8801
14272/60000 [======>.......................] - ETA: 1:04 - loss: 0.3835 - categorical_accuracy: 0.8805
14336/60000 [======>.......................] - ETA: 1:04 - loss: 0.3825 - categorical_accuracy: 0.8808
14400/60000 [======>.......................] - ETA: 1:04 - loss: 0.3824 - categorical_accuracy: 0.8811
14464/60000 [======>.......................] - ETA: 1:04 - loss: 0.3814 - categorical_accuracy: 0.8816
14528/60000 [======>.......................] - ETA: 1:04 - loss: 0.3806 - categorical_accuracy: 0.8817
14592/60000 [======>.......................] - ETA: 1:04 - loss: 0.3803 - categorical_accuracy: 0.8817
14656/60000 [======>.......................] - ETA: 1:04 - loss: 0.3796 - categorical_accuracy: 0.8820
14720/60000 [======>.......................] - ETA: 1:04 - loss: 0.3787 - categorical_accuracy: 0.8823
14784/60000 [======>.......................] - ETA: 1:04 - loss: 0.3788 - categorical_accuracy: 0.8824
14848/60000 [======>.......................] - ETA: 1:04 - loss: 0.3783 - categorical_accuracy: 0.8825
14912/60000 [======>.......................] - ETA: 1:03 - loss: 0.3771 - categorical_accuracy: 0.8830
14976/60000 [======>.......................] - ETA: 1:03 - loss: 0.3759 - categorical_accuracy: 0.8835
15040/60000 [======>.......................] - ETA: 1:03 - loss: 0.3747 - categorical_accuracy: 0.8838
15104/60000 [======>.......................] - ETA: 1:03 - loss: 0.3741 - categorical_accuracy: 0.8839
15168/60000 [======>.......................] - ETA: 1:03 - loss: 0.3731 - categorical_accuracy: 0.8841
15232/60000 [======>.......................] - ETA: 1:03 - loss: 0.3723 - categorical_accuracy: 0.8842
15296/60000 [======>.......................] - ETA: 1:03 - loss: 0.3715 - categorical_accuracy: 0.8845
15360/60000 [======>.......................] - ETA: 1:03 - loss: 0.3709 - categorical_accuracy: 0.8848
15392/60000 [======>.......................] - ETA: 1:03 - loss: 0.3709 - categorical_accuracy: 0.8848
15456/60000 [======>.......................] - ETA: 1:03 - loss: 0.3698 - categorical_accuracy: 0.8852
15520/60000 [======>.......................] - ETA: 1:03 - loss: 0.3687 - categorical_accuracy: 0.8856
15584/60000 [======>.......................] - ETA: 1:03 - loss: 0.3677 - categorical_accuracy: 0.8858
15648/60000 [======>.......................] - ETA: 1:02 - loss: 0.3666 - categorical_accuracy: 0.8862
15712/60000 [======>.......................] - ETA: 1:02 - loss: 0.3659 - categorical_accuracy: 0.8865
15776/60000 [======>.......................] - ETA: 1:02 - loss: 0.3652 - categorical_accuracy: 0.8867
15840/60000 [======>.......................] - ETA: 1:02 - loss: 0.3642 - categorical_accuracy: 0.8869
15904/60000 [======>.......................] - ETA: 1:02 - loss: 0.3632 - categorical_accuracy: 0.8873
15968/60000 [======>.......................] - ETA: 1:02 - loss: 0.3619 - categorical_accuracy: 0.8878
16032/60000 [=======>......................] - ETA: 1:02 - loss: 0.3608 - categorical_accuracy: 0.8881
16096/60000 [=======>......................] - ETA: 1:02 - loss: 0.3599 - categorical_accuracy: 0.8884
16160/60000 [=======>......................] - ETA: 1:02 - loss: 0.3596 - categorical_accuracy: 0.8886
16224/60000 [=======>......................] - ETA: 1:02 - loss: 0.3594 - categorical_accuracy: 0.8886
16288/60000 [=======>......................] - ETA: 1:01 - loss: 0.3587 - categorical_accuracy: 0.8888
16352/60000 [=======>......................] - ETA: 1:01 - loss: 0.3578 - categorical_accuracy: 0.8891
16416/60000 [=======>......................] - ETA: 1:01 - loss: 0.3577 - categorical_accuracy: 0.8891
16480/60000 [=======>......................] - ETA: 1:01 - loss: 0.3568 - categorical_accuracy: 0.8894
16544/60000 [=======>......................] - ETA: 1:01 - loss: 0.3561 - categorical_accuracy: 0.8896
16608/60000 [=======>......................] - ETA: 1:01 - loss: 0.3549 - categorical_accuracy: 0.8901
16672/60000 [=======>......................] - ETA: 1:01 - loss: 0.3538 - categorical_accuracy: 0.8904
16736/60000 [=======>......................] - ETA: 1:01 - loss: 0.3539 - categorical_accuracy: 0.8904
16800/60000 [=======>......................] - ETA: 1:01 - loss: 0.3533 - categorical_accuracy: 0.8904
16864/60000 [=======>......................] - ETA: 1:01 - loss: 0.3524 - categorical_accuracy: 0.8908
16928/60000 [=======>......................] - ETA: 1:01 - loss: 0.3518 - categorical_accuracy: 0.8910
16992/60000 [=======>......................] - ETA: 1:00 - loss: 0.3515 - categorical_accuracy: 0.8911
17056/60000 [=======>......................] - ETA: 1:00 - loss: 0.3513 - categorical_accuracy: 0.8912
17120/60000 [=======>......................] - ETA: 1:00 - loss: 0.3501 - categorical_accuracy: 0.8916
17184/60000 [=======>......................] - ETA: 1:00 - loss: 0.3492 - categorical_accuracy: 0.8919
17248/60000 [=======>......................] - ETA: 1:00 - loss: 0.3492 - categorical_accuracy: 0.8919
17312/60000 [=======>......................] - ETA: 1:00 - loss: 0.3492 - categorical_accuracy: 0.8920
17376/60000 [=======>......................] - ETA: 1:00 - loss: 0.3482 - categorical_accuracy: 0.8923
17440/60000 [=======>......................] - ETA: 1:00 - loss: 0.3481 - categorical_accuracy: 0.8923
17504/60000 [=======>......................] - ETA: 1:00 - loss: 0.3482 - categorical_accuracy: 0.8924
17568/60000 [=======>......................] - ETA: 1:00 - loss: 0.3477 - categorical_accuracy: 0.8925
17632/60000 [=======>......................] - ETA: 1:00 - loss: 0.3469 - categorical_accuracy: 0.8927
17696/60000 [=======>......................] - ETA: 59s - loss: 0.3460 - categorical_accuracy: 0.8930 
17760/60000 [=======>......................] - ETA: 59s - loss: 0.3455 - categorical_accuracy: 0.8931
17824/60000 [=======>......................] - ETA: 59s - loss: 0.3445 - categorical_accuracy: 0.8934
17888/60000 [=======>......................] - ETA: 59s - loss: 0.3438 - categorical_accuracy: 0.8936
17952/60000 [=======>......................] - ETA: 59s - loss: 0.3444 - categorical_accuracy: 0.8935
18016/60000 [========>.....................] - ETA: 59s - loss: 0.3434 - categorical_accuracy: 0.8939
18080/60000 [========>.....................] - ETA: 59s - loss: 0.3425 - categorical_accuracy: 0.8941
18144/60000 [========>.....................] - ETA: 59s - loss: 0.3417 - categorical_accuracy: 0.8945
18208/60000 [========>.....................] - ETA: 59s - loss: 0.3415 - categorical_accuracy: 0.8947
18272/60000 [========>.....................] - ETA: 59s - loss: 0.3414 - categorical_accuracy: 0.8947
18336/60000 [========>.....................] - ETA: 58s - loss: 0.3405 - categorical_accuracy: 0.8950
18400/60000 [========>.....................] - ETA: 58s - loss: 0.3397 - categorical_accuracy: 0.8951
18464/60000 [========>.....................] - ETA: 58s - loss: 0.3397 - categorical_accuracy: 0.8953
18528/60000 [========>.....................] - ETA: 58s - loss: 0.3392 - categorical_accuracy: 0.8953
18560/60000 [========>.....................] - ETA: 58s - loss: 0.3388 - categorical_accuracy: 0.8955
18624/60000 [========>.....................] - ETA: 58s - loss: 0.3379 - categorical_accuracy: 0.8957
18688/60000 [========>.....................] - ETA: 58s - loss: 0.3374 - categorical_accuracy: 0.8959
18752/60000 [========>.....................] - ETA: 58s - loss: 0.3366 - categorical_accuracy: 0.8962
18816/60000 [========>.....................] - ETA: 58s - loss: 0.3357 - categorical_accuracy: 0.8964
18880/60000 [========>.....................] - ETA: 58s - loss: 0.3354 - categorical_accuracy: 0.8965
18944/60000 [========>.....................] - ETA: 58s - loss: 0.3351 - categorical_accuracy: 0.8965
19008/60000 [========>.....................] - ETA: 57s - loss: 0.3343 - categorical_accuracy: 0.8967
19072/60000 [========>.....................] - ETA: 57s - loss: 0.3334 - categorical_accuracy: 0.8969
19136/60000 [========>.....................] - ETA: 57s - loss: 0.3325 - categorical_accuracy: 0.8972
19200/60000 [========>.....................] - ETA: 57s - loss: 0.3316 - categorical_accuracy: 0.8975
19264/60000 [========>.....................] - ETA: 57s - loss: 0.3310 - categorical_accuracy: 0.8977
19328/60000 [========>.....................] - ETA: 57s - loss: 0.3300 - categorical_accuracy: 0.8980
19392/60000 [========>.....................] - ETA: 57s - loss: 0.3299 - categorical_accuracy: 0.8982
19456/60000 [========>.....................] - ETA: 57s - loss: 0.3292 - categorical_accuracy: 0.8983
19520/60000 [========>.....................] - ETA: 57s - loss: 0.3290 - categorical_accuracy: 0.8984
19584/60000 [========>.....................] - ETA: 57s - loss: 0.3285 - categorical_accuracy: 0.8985
19648/60000 [========>.....................] - ETA: 57s - loss: 0.3280 - categorical_accuracy: 0.8987
19712/60000 [========>.....................] - ETA: 56s - loss: 0.3277 - categorical_accuracy: 0.8987
19744/60000 [========>.....................] - ETA: 56s - loss: 0.3273 - categorical_accuracy: 0.8988
19808/60000 [========>.....................] - ETA: 56s - loss: 0.3270 - categorical_accuracy: 0.8988
19872/60000 [========>.....................] - ETA: 56s - loss: 0.3270 - categorical_accuracy: 0.8989
19936/60000 [========>.....................] - ETA: 56s - loss: 0.3262 - categorical_accuracy: 0.8992
20000/60000 [=========>....................] - ETA: 56s - loss: 0.3259 - categorical_accuracy: 0.8993
20064/60000 [=========>....................] - ETA: 56s - loss: 0.3252 - categorical_accuracy: 0.8994
20128/60000 [=========>....................] - ETA: 56s - loss: 0.3246 - categorical_accuracy: 0.8995
20192/60000 [=========>....................] - ETA: 56s - loss: 0.3238 - categorical_accuracy: 0.8998
20256/60000 [=========>....................] - ETA: 56s - loss: 0.3234 - categorical_accuracy: 0.8998
20320/60000 [=========>....................] - ETA: 56s - loss: 0.3233 - categorical_accuracy: 0.9000
20384/60000 [=========>....................] - ETA: 55s - loss: 0.3237 - categorical_accuracy: 0.8999
20448/60000 [=========>....................] - ETA: 55s - loss: 0.3231 - categorical_accuracy: 0.9001
20512/60000 [=========>....................] - ETA: 55s - loss: 0.3225 - categorical_accuracy: 0.9003
20576/60000 [=========>....................] - ETA: 55s - loss: 0.3221 - categorical_accuracy: 0.9004
20640/60000 [=========>....................] - ETA: 55s - loss: 0.3215 - categorical_accuracy: 0.9005
20704/60000 [=========>....................] - ETA: 55s - loss: 0.3211 - categorical_accuracy: 0.9007
20768/60000 [=========>....................] - ETA: 55s - loss: 0.3208 - categorical_accuracy: 0.9008
20832/60000 [=========>....................] - ETA: 55s - loss: 0.3208 - categorical_accuracy: 0.9009
20896/60000 [=========>....................] - ETA: 55s - loss: 0.3201 - categorical_accuracy: 0.9010
20960/60000 [=========>....................] - ETA: 55s - loss: 0.3198 - categorical_accuracy: 0.9011
21024/60000 [=========>....................] - ETA: 55s - loss: 0.3190 - categorical_accuracy: 0.9014
21088/60000 [=========>....................] - ETA: 54s - loss: 0.3184 - categorical_accuracy: 0.9017
21152/60000 [=========>....................] - ETA: 54s - loss: 0.3186 - categorical_accuracy: 0.9017
21216/60000 [=========>....................] - ETA: 54s - loss: 0.3185 - categorical_accuracy: 0.9017
21280/60000 [=========>....................] - ETA: 54s - loss: 0.3179 - categorical_accuracy: 0.9018
21344/60000 [=========>....................] - ETA: 54s - loss: 0.3178 - categorical_accuracy: 0.9019
21408/60000 [=========>....................] - ETA: 54s - loss: 0.3170 - categorical_accuracy: 0.9021
21472/60000 [=========>....................] - ETA: 54s - loss: 0.3166 - categorical_accuracy: 0.9023
21536/60000 [=========>....................] - ETA: 54s - loss: 0.3160 - categorical_accuracy: 0.9025
21600/60000 [=========>....................] - ETA: 54s - loss: 0.3153 - categorical_accuracy: 0.9027
21664/60000 [=========>....................] - ETA: 54s - loss: 0.3147 - categorical_accuracy: 0.9029
21728/60000 [=========>....................] - ETA: 53s - loss: 0.3142 - categorical_accuracy: 0.9030
21792/60000 [=========>....................] - ETA: 53s - loss: 0.3135 - categorical_accuracy: 0.9032
21856/60000 [=========>....................] - ETA: 53s - loss: 0.3130 - categorical_accuracy: 0.9033
21920/60000 [=========>....................] - ETA: 53s - loss: 0.3125 - categorical_accuracy: 0.9034
21984/60000 [=========>....................] - ETA: 53s - loss: 0.3122 - categorical_accuracy: 0.9036
22048/60000 [==========>...................] - ETA: 53s - loss: 0.3115 - categorical_accuracy: 0.9038
22112/60000 [==========>...................] - ETA: 53s - loss: 0.3107 - categorical_accuracy: 0.9040
22176/60000 [==========>...................] - ETA: 53s - loss: 0.3100 - categorical_accuracy: 0.9042
22240/60000 [==========>...................] - ETA: 53s - loss: 0.3094 - categorical_accuracy: 0.9044
22304/60000 [==========>...................] - ETA: 53s - loss: 0.3091 - categorical_accuracy: 0.9044
22368/60000 [==========>...................] - ETA: 53s - loss: 0.3083 - categorical_accuracy: 0.9047
22432/60000 [==========>...................] - ETA: 52s - loss: 0.3078 - categorical_accuracy: 0.9049
22496/60000 [==========>...................] - ETA: 52s - loss: 0.3074 - categorical_accuracy: 0.9050
22560/60000 [==========>...................] - ETA: 52s - loss: 0.3069 - categorical_accuracy: 0.9052
22624/60000 [==========>...................] - ETA: 52s - loss: 0.3064 - categorical_accuracy: 0.9053
22688/60000 [==========>...................] - ETA: 52s - loss: 0.3061 - categorical_accuracy: 0.9055
22752/60000 [==========>...................] - ETA: 52s - loss: 0.3058 - categorical_accuracy: 0.9056
22816/60000 [==========>...................] - ETA: 52s - loss: 0.3051 - categorical_accuracy: 0.9058
22880/60000 [==========>...................] - ETA: 52s - loss: 0.3044 - categorical_accuracy: 0.9060
22912/60000 [==========>...................] - ETA: 52s - loss: 0.3041 - categorical_accuracy: 0.9061
22976/60000 [==========>...................] - ETA: 52s - loss: 0.3037 - categorical_accuracy: 0.9062
23040/60000 [==========>...................] - ETA: 52s - loss: 0.3030 - categorical_accuracy: 0.9064
23104/60000 [==========>...................] - ETA: 52s - loss: 0.3024 - categorical_accuracy: 0.9066
23168/60000 [==========>...................] - ETA: 51s - loss: 0.3020 - categorical_accuracy: 0.9067
23232/60000 [==========>...................] - ETA: 51s - loss: 0.3014 - categorical_accuracy: 0.9069
23296/60000 [==========>...................] - ETA: 51s - loss: 0.3010 - categorical_accuracy: 0.9070
23360/60000 [==========>...................] - ETA: 51s - loss: 0.3008 - categorical_accuracy: 0.9072
23424/60000 [==========>...................] - ETA: 51s - loss: 0.3002 - categorical_accuracy: 0.9073
23488/60000 [==========>...................] - ETA: 51s - loss: 0.3001 - categorical_accuracy: 0.9074
23552/60000 [==========>...................] - ETA: 51s - loss: 0.2998 - categorical_accuracy: 0.9074
23616/60000 [==========>...................] - ETA: 51s - loss: 0.2992 - categorical_accuracy: 0.9076
23680/60000 [==========>...................] - ETA: 51s - loss: 0.2984 - categorical_accuracy: 0.9078
23744/60000 [==========>...................] - ETA: 51s - loss: 0.2978 - categorical_accuracy: 0.9080
23808/60000 [==========>...................] - ETA: 51s - loss: 0.2974 - categorical_accuracy: 0.9081
23872/60000 [==========>...................] - ETA: 50s - loss: 0.2970 - categorical_accuracy: 0.9082
23936/60000 [==========>...................] - ETA: 50s - loss: 0.2968 - categorical_accuracy: 0.9083
24000/60000 [===========>..................] - ETA: 50s - loss: 0.2963 - categorical_accuracy: 0.9084
24064/60000 [===========>..................] - ETA: 50s - loss: 0.2957 - categorical_accuracy: 0.9086
24128/60000 [===========>..................] - ETA: 50s - loss: 0.2954 - categorical_accuracy: 0.9087
24192/60000 [===========>..................] - ETA: 50s - loss: 0.2948 - categorical_accuracy: 0.9089
24224/60000 [===========>..................] - ETA: 50s - loss: 0.2947 - categorical_accuracy: 0.9089
24288/60000 [===========>..................] - ETA: 50s - loss: 0.2944 - categorical_accuracy: 0.9090
24352/60000 [===========>..................] - ETA: 50s - loss: 0.2941 - categorical_accuracy: 0.9090
24416/60000 [===========>..................] - ETA: 50s - loss: 0.2940 - categorical_accuracy: 0.9091
24480/60000 [===========>..................] - ETA: 50s - loss: 0.2938 - categorical_accuracy: 0.9092
24544/60000 [===========>..................] - ETA: 49s - loss: 0.2931 - categorical_accuracy: 0.9094
24608/60000 [===========>..................] - ETA: 49s - loss: 0.2927 - categorical_accuracy: 0.9095
24672/60000 [===========>..................] - ETA: 49s - loss: 0.2921 - categorical_accuracy: 0.9097
24736/60000 [===========>..................] - ETA: 49s - loss: 0.2917 - categorical_accuracy: 0.9098
24768/60000 [===========>..................] - ETA: 49s - loss: 0.2914 - categorical_accuracy: 0.9100
24832/60000 [===========>..................] - ETA: 49s - loss: 0.2911 - categorical_accuracy: 0.9101
24896/60000 [===========>..................] - ETA: 49s - loss: 0.2904 - categorical_accuracy: 0.9103
24960/60000 [===========>..................] - ETA: 49s - loss: 0.2899 - categorical_accuracy: 0.9105
24992/60000 [===========>..................] - ETA: 49s - loss: 0.2896 - categorical_accuracy: 0.9106
25056/60000 [===========>..................] - ETA: 49s - loss: 0.2891 - categorical_accuracy: 0.9107
25120/60000 [===========>..................] - ETA: 49s - loss: 0.2885 - categorical_accuracy: 0.9109
25184/60000 [===========>..................] - ETA: 49s - loss: 0.2881 - categorical_accuracy: 0.9110
25248/60000 [===========>..................] - ETA: 49s - loss: 0.2878 - categorical_accuracy: 0.9111
25312/60000 [===========>..................] - ETA: 48s - loss: 0.2872 - categorical_accuracy: 0.9113
25376/60000 [===========>..................] - ETA: 48s - loss: 0.2866 - categorical_accuracy: 0.9115
25440/60000 [===========>..................] - ETA: 48s - loss: 0.2862 - categorical_accuracy: 0.9117
25504/60000 [===========>..................] - ETA: 48s - loss: 0.2859 - categorical_accuracy: 0.9118
25568/60000 [===========>..................] - ETA: 48s - loss: 0.2853 - categorical_accuracy: 0.9120
25632/60000 [===========>..................] - ETA: 48s - loss: 0.2851 - categorical_accuracy: 0.9120
25696/60000 [===========>..................] - ETA: 48s - loss: 0.2849 - categorical_accuracy: 0.9121
25760/60000 [===========>..................] - ETA: 48s - loss: 0.2845 - categorical_accuracy: 0.9122
25824/60000 [===========>..................] - ETA: 48s - loss: 0.2839 - categorical_accuracy: 0.9124
25888/60000 [===========>..................] - ETA: 48s - loss: 0.2833 - categorical_accuracy: 0.9126
25952/60000 [===========>..................] - ETA: 47s - loss: 0.2829 - categorical_accuracy: 0.9127
26016/60000 [============>.................] - ETA: 47s - loss: 0.2824 - categorical_accuracy: 0.9129
26080/60000 [============>.................] - ETA: 47s - loss: 0.2822 - categorical_accuracy: 0.9130
26144/60000 [============>.................] - ETA: 47s - loss: 0.2816 - categorical_accuracy: 0.9131
26208/60000 [============>.................] - ETA: 47s - loss: 0.2810 - categorical_accuracy: 0.9133
26272/60000 [============>.................] - ETA: 47s - loss: 0.2808 - categorical_accuracy: 0.9134
26336/60000 [============>.................] - ETA: 47s - loss: 0.2804 - categorical_accuracy: 0.9135
26400/60000 [============>.................] - ETA: 47s - loss: 0.2799 - categorical_accuracy: 0.9137
26464/60000 [============>.................] - ETA: 47s - loss: 0.2793 - categorical_accuracy: 0.9139
26528/60000 [============>.................] - ETA: 47s - loss: 0.2788 - categorical_accuracy: 0.9140
26560/60000 [============>.................] - ETA: 47s - loss: 0.2785 - categorical_accuracy: 0.9141
26624/60000 [============>.................] - ETA: 47s - loss: 0.2781 - categorical_accuracy: 0.9142
26688/60000 [============>.................] - ETA: 46s - loss: 0.2777 - categorical_accuracy: 0.9143
26752/60000 [============>.................] - ETA: 46s - loss: 0.2773 - categorical_accuracy: 0.9144
26816/60000 [============>.................] - ETA: 46s - loss: 0.2767 - categorical_accuracy: 0.9145
26880/60000 [============>.................] - ETA: 46s - loss: 0.2763 - categorical_accuracy: 0.9146
26912/60000 [============>.................] - ETA: 46s - loss: 0.2760 - categorical_accuracy: 0.9147
26976/60000 [============>.................] - ETA: 46s - loss: 0.2757 - categorical_accuracy: 0.9149
27040/60000 [============>.................] - ETA: 46s - loss: 0.2756 - categorical_accuracy: 0.9149
27104/60000 [============>.................] - ETA: 46s - loss: 0.2752 - categorical_accuracy: 0.9150
27168/60000 [============>.................] - ETA: 46s - loss: 0.2747 - categorical_accuracy: 0.9152
27232/60000 [============>.................] - ETA: 46s - loss: 0.2743 - categorical_accuracy: 0.9153
27296/60000 [============>.................] - ETA: 46s - loss: 0.2739 - categorical_accuracy: 0.9154
27360/60000 [============>.................] - ETA: 46s - loss: 0.2736 - categorical_accuracy: 0.9155
27424/60000 [============>.................] - ETA: 45s - loss: 0.2731 - categorical_accuracy: 0.9157
27488/60000 [============>.................] - ETA: 45s - loss: 0.2727 - categorical_accuracy: 0.9159
27552/60000 [============>.................] - ETA: 45s - loss: 0.2725 - categorical_accuracy: 0.9159
27616/60000 [============>.................] - ETA: 45s - loss: 0.2724 - categorical_accuracy: 0.9160
27680/60000 [============>.................] - ETA: 45s - loss: 0.2722 - categorical_accuracy: 0.9160
27744/60000 [============>.................] - ETA: 45s - loss: 0.2716 - categorical_accuracy: 0.9162
27808/60000 [============>.................] - ETA: 45s - loss: 0.2711 - categorical_accuracy: 0.9164
27872/60000 [============>.................] - ETA: 45s - loss: 0.2705 - categorical_accuracy: 0.9166
27936/60000 [============>.................] - ETA: 45s - loss: 0.2702 - categorical_accuracy: 0.9166
28000/60000 [=============>................] - ETA: 45s - loss: 0.2701 - categorical_accuracy: 0.9167
28064/60000 [=============>................] - ETA: 44s - loss: 0.2698 - categorical_accuracy: 0.9168
28128/60000 [=============>................] - ETA: 44s - loss: 0.2695 - categorical_accuracy: 0.9169
28192/60000 [=============>................] - ETA: 44s - loss: 0.2693 - categorical_accuracy: 0.9169
28256/60000 [=============>................] - ETA: 44s - loss: 0.2690 - categorical_accuracy: 0.9170
28320/60000 [=============>................] - ETA: 44s - loss: 0.2686 - categorical_accuracy: 0.9171
28384/60000 [=============>................] - ETA: 44s - loss: 0.2681 - categorical_accuracy: 0.9172
28448/60000 [=============>................] - ETA: 44s - loss: 0.2677 - categorical_accuracy: 0.9174
28512/60000 [=============>................] - ETA: 44s - loss: 0.2674 - categorical_accuracy: 0.9175
28544/60000 [=============>................] - ETA: 44s - loss: 0.2671 - categorical_accuracy: 0.9176
28608/60000 [=============>................] - ETA: 44s - loss: 0.2668 - categorical_accuracy: 0.9176
28672/60000 [=============>................] - ETA: 44s - loss: 0.2665 - categorical_accuracy: 0.9177
28736/60000 [=============>................] - ETA: 44s - loss: 0.2662 - categorical_accuracy: 0.9178
28800/60000 [=============>................] - ETA: 43s - loss: 0.2660 - categorical_accuracy: 0.9179
28864/60000 [=============>................] - ETA: 43s - loss: 0.2654 - categorical_accuracy: 0.9181
28928/60000 [=============>................] - ETA: 43s - loss: 0.2651 - categorical_accuracy: 0.9182
28992/60000 [=============>................] - ETA: 43s - loss: 0.2647 - categorical_accuracy: 0.9183
29056/60000 [=============>................] - ETA: 43s - loss: 0.2644 - categorical_accuracy: 0.9183
29120/60000 [=============>................] - ETA: 43s - loss: 0.2640 - categorical_accuracy: 0.9184
29184/60000 [=============>................] - ETA: 43s - loss: 0.2636 - categorical_accuracy: 0.9186
29248/60000 [=============>................] - ETA: 43s - loss: 0.2634 - categorical_accuracy: 0.9186
29312/60000 [=============>................] - ETA: 43s - loss: 0.2633 - categorical_accuracy: 0.9185
29376/60000 [=============>................] - ETA: 43s - loss: 0.2632 - categorical_accuracy: 0.9186
29440/60000 [=============>................] - ETA: 43s - loss: 0.2629 - categorical_accuracy: 0.9187
29504/60000 [=============>................] - ETA: 42s - loss: 0.2625 - categorical_accuracy: 0.9188
29568/60000 [=============>................] - ETA: 42s - loss: 0.2622 - categorical_accuracy: 0.9190
29632/60000 [=============>................] - ETA: 42s - loss: 0.2618 - categorical_accuracy: 0.9191
29696/60000 [=============>................] - ETA: 42s - loss: 0.2614 - categorical_accuracy: 0.9192
29760/60000 [=============>................] - ETA: 42s - loss: 0.2609 - categorical_accuracy: 0.9194
29824/60000 [=============>................] - ETA: 42s - loss: 0.2610 - categorical_accuracy: 0.9194
29888/60000 [=============>................] - ETA: 42s - loss: 0.2610 - categorical_accuracy: 0.9195
29952/60000 [=============>................] - ETA: 42s - loss: 0.2607 - categorical_accuracy: 0.9196
30016/60000 [==============>...............] - ETA: 42s - loss: 0.2605 - categorical_accuracy: 0.9197
30080/60000 [==============>...............] - ETA: 42s - loss: 0.2603 - categorical_accuracy: 0.9197
30144/60000 [==============>...............] - ETA: 42s - loss: 0.2600 - categorical_accuracy: 0.9198
30176/60000 [==============>...............] - ETA: 41s - loss: 0.2597 - categorical_accuracy: 0.9198
30240/60000 [==============>...............] - ETA: 41s - loss: 0.2596 - categorical_accuracy: 0.9199
30304/60000 [==============>...............] - ETA: 41s - loss: 0.2594 - categorical_accuracy: 0.9201
30368/60000 [==============>...............] - ETA: 41s - loss: 0.2591 - categorical_accuracy: 0.9201
30432/60000 [==============>...............] - ETA: 41s - loss: 0.2587 - categorical_accuracy: 0.9202
30496/60000 [==============>...............] - ETA: 41s - loss: 0.2585 - categorical_accuracy: 0.9203
30560/60000 [==============>...............] - ETA: 41s - loss: 0.2581 - categorical_accuracy: 0.9205
30624/60000 [==============>...............] - ETA: 41s - loss: 0.2577 - categorical_accuracy: 0.9206
30688/60000 [==============>...............] - ETA: 41s - loss: 0.2575 - categorical_accuracy: 0.9207
30752/60000 [==============>...............] - ETA: 41s - loss: 0.2571 - categorical_accuracy: 0.9208
30816/60000 [==============>...............] - ETA: 41s - loss: 0.2569 - categorical_accuracy: 0.9209
30848/60000 [==============>...............] - ETA: 41s - loss: 0.2567 - categorical_accuracy: 0.9210
30912/60000 [==============>...............] - ETA: 40s - loss: 0.2569 - categorical_accuracy: 0.9210
30976/60000 [==============>...............] - ETA: 40s - loss: 0.2565 - categorical_accuracy: 0.9211
31040/60000 [==============>...............] - ETA: 40s - loss: 0.2560 - categorical_accuracy: 0.9212
31104/60000 [==============>...............] - ETA: 40s - loss: 0.2558 - categorical_accuracy: 0.9213
31168/60000 [==============>...............] - ETA: 40s - loss: 0.2553 - categorical_accuracy: 0.9214
31232/60000 [==============>...............] - ETA: 40s - loss: 0.2551 - categorical_accuracy: 0.9214
31296/60000 [==============>...............] - ETA: 40s - loss: 0.2547 - categorical_accuracy: 0.9216
31360/60000 [==============>...............] - ETA: 40s - loss: 0.2543 - categorical_accuracy: 0.9217
31424/60000 [==============>...............] - ETA: 40s - loss: 0.2538 - categorical_accuracy: 0.9218
31488/60000 [==============>...............] - ETA: 40s - loss: 0.2538 - categorical_accuracy: 0.9219
31552/60000 [==============>...............] - ETA: 40s - loss: 0.2536 - categorical_accuracy: 0.9220
31616/60000 [==============>...............] - ETA: 39s - loss: 0.2532 - categorical_accuracy: 0.9221
31680/60000 [==============>...............] - ETA: 39s - loss: 0.2529 - categorical_accuracy: 0.9222
31744/60000 [==============>...............] - ETA: 39s - loss: 0.2529 - categorical_accuracy: 0.9221
31808/60000 [==============>...............] - ETA: 39s - loss: 0.2528 - categorical_accuracy: 0.9222
31872/60000 [==============>...............] - ETA: 39s - loss: 0.2524 - categorical_accuracy: 0.9223
31904/60000 [==============>...............] - ETA: 39s - loss: 0.2524 - categorical_accuracy: 0.9223
31968/60000 [==============>...............] - ETA: 39s - loss: 0.2520 - categorical_accuracy: 0.9225
32000/60000 [===============>..............] - ETA: 39s - loss: 0.2518 - categorical_accuracy: 0.9225
32064/60000 [===============>..............] - ETA: 39s - loss: 0.2515 - categorical_accuracy: 0.9226
32128/60000 [===============>..............] - ETA: 39s - loss: 0.2511 - categorical_accuracy: 0.9227
32192/60000 [===============>..............] - ETA: 39s - loss: 0.2506 - categorical_accuracy: 0.9229
32256/60000 [===============>..............] - ETA: 39s - loss: 0.2504 - categorical_accuracy: 0.9230
32320/60000 [===============>..............] - ETA: 38s - loss: 0.2505 - categorical_accuracy: 0.9230
32384/60000 [===============>..............] - ETA: 38s - loss: 0.2507 - categorical_accuracy: 0.9230
32416/60000 [===============>..............] - ETA: 38s - loss: 0.2509 - categorical_accuracy: 0.9230
32448/60000 [===============>..............] - ETA: 38s - loss: 0.2508 - categorical_accuracy: 0.9230
32512/60000 [===============>..............] - ETA: 38s - loss: 0.2506 - categorical_accuracy: 0.9231
32576/60000 [===============>..............] - ETA: 38s - loss: 0.2503 - categorical_accuracy: 0.9232
32640/60000 [===============>..............] - ETA: 38s - loss: 0.2501 - categorical_accuracy: 0.9232
32704/60000 [===============>..............] - ETA: 38s - loss: 0.2498 - categorical_accuracy: 0.9233
32768/60000 [===============>..............] - ETA: 38s - loss: 0.2494 - categorical_accuracy: 0.9234
32832/60000 [===============>..............] - ETA: 38s - loss: 0.2491 - categorical_accuracy: 0.9236
32896/60000 [===============>..............] - ETA: 38s - loss: 0.2489 - categorical_accuracy: 0.9235
32960/60000 [===============>..............] - ETA: 38s - loss: 0.2485 - categorical_accuracy: 0.9237
32992/60000 [===============>..............] - ETA: 38s - loss: 0.2483 - categorical_accuracy: 0.9237
33056/60000 [===============>..............] - ETA: 37s - loss: 0.2479 - categorical_accuracy: 0.9239
33120/60000 [===============>..............] - ETA: 37s - loss: 0.2479 - categorical_accuracy: 0.9239
33184/60000 [===============>..............] - ETA: 37s - loss: 0.2479 - categorical_accuracy: 0.9239
33248/60000 [===============>..............] - ETA: 37s - loss: 0.2475 - categorical_accuracy: 0.9241
33312/60000 [===============>..............] - ETA: 37s - loss: 0.2472 - categorical_accuracy: 0.9242
33376/60000 [===============>..............] - ETA: 37s - loss: 0.2468 - categorical_accuracy: 0.9243
33440/60000 [===============>..............] - ETA: 37s - loss: 0.2468 - categorical_accuracy: 0.9244
33504/60000 [===============>..............] - ETA: 37s - loss: 0.2464 - categorical_accuracy: 0.9244
33568/60000 [===============>..............] - ETA: 37s - loss: 0.2461 - categorical_accuracy: 0.9245
33632/60000 [===============>..............] - ETA: 37s - loss: 0.2459 - categorical_accuracy: 0.9246
33696/60000 [===============>..............] - ETA: 37s - loss: 0.2455 - categorical_accuracy: 0.9247
33760/60000 [===============>..............] - ETA: 36s - loss: 0.2452 - categorical_accuracy: 0.9248
33824/60000 [===============>..............] - ETA: 36s - loss: 0.2448 - categorical_accuracy: 0.9249
33888/60000 [===============>..............] - ETA: 36s - loss: 0.2444 - categorical_accuracy: 0.9250
33952/60000 [===============>..............] - ETA: 36s - loss: 0.2443 - categorical_accuracy: 0.9251
34016/60000 [================>.............] - ETA: 36s - loss: 0.2441 - categorical_accuracy: 0.9251
34080/60000 [================>.............] - ETA: 36s - loss: 0.2442 - categorical_accuracy: 0.9252
34144/60000 [================>.............] - ETA: 36s - loss: 0.2443 - categorical_accuracy: 0.9252
34176/60000 [================>.............] - ETA: 36s - loss: 0.2444 - categorical_accuracy: 0.9252
34240/60000 [================>.............] - ETA: 36s - loss: 0.2440 - categorical_accuracy: 0.9253
34304/60000 [================>.............] - ETA: 36s - loss: 0.2437 - categorical_accuracy: 0.9254
34368/60000 [================>.............] - ETA: 36s - loss: 0.2435 - categorical_accuracy: 0.9255
34432/60000 [================>.............] - ETA: 35s - loss: 0.2431 - categorical_accuracy: 0.9256
34496/60000 [================>.............] - ETA: 35s - loss: 0.2427 - categorical_accuracy: 0.9257
34560/60000 [================>.............] - ETA: 35s - loss: 0.2425 - categorical_accuracy: 0.9258
34624/60000 [================>.............] - ETA: 35s - loss: 0.2424 - categorical_accuracy: 0.9258
34688/60000 [================>.............] - ETA: 35s - loss: 0.2425 - categorical_accuracy: 0.9259
34752/60000 [================>.............] - ETA: 35s - loss: 0.2421 - categorical_accuracy: 0.9260
34816/60000 [================>.............] - ETA: 35s - loss: 0.2422 - categorical_accuracy: 0.9260
34880/60000 [================>.............] - ETA: 35s - loss: 0.2417 - categorical_accuracy: 0.9261
34944/60000 [================>.............] - ETA: 35s - loss: 0.2415 - categorical_accuracy: 0.9262
35008/60000 [================>.............] - ETA: 35s - loss: 0.2411 - categorical_accuracy: 0.9263
35072/60000 [================>.............] - ETA: 35s - loss: 0.2410 - categorical_accuracy: 0.9264
35136/60000 [================>.............] - ETA: 34s - loss: 0.2407 - categorical_accuracy: 0.9265
35168/60000 [================>.............] - ETA: 34s - loss: 0.2405 - categorical_accuracy: 0.9266
35200/60000 [================>.............] - ETA: 34s - loss: 0.2404 - categorical_accuracy: 0.9266
35264/60000 [================>.............] - ETA: 34s - loss: 0.2401 - categorical_accuracy: 0.9267
35328/60000 [================>.............] - ETA: 34s - loss: 0.2398 - categorical_accuracy: 0.9268
35392/60000 [================>.............] - ETA: 34s - loss: 0.2396 - categorical_accuracy: 0.9268
35456/60000 [================>.............] - ETA: 34s - loss: 0.2394 - categorical_accuracy: 0.9270
35520/60000 [================>.............] - ETA: 34s - loss: 0.2391 - categorical_accuracy: 0.9271
35584/60000 [================>.............] - ETA: 34s - loss: 0.2387 - categorical_accuracy: 0.9272
35648/60000 [================>.............] - ETA: 34s - loss: 0.2383 - categorical_accuracy: 0.9273
35712/60000 [================>.............] - ETA: 34s - loss: 0.2380 - categorical_accuracy: 0.9274
35776/60000 [================>.............] - ETA: 34s - loss: 0.2378 - categorical_accuracy: 0.9274
35840/60000 [================>.............] - ETA: 33s - loss: 0.2378 - categorical_accuracy: 0.9274
35904/60000 [================>.............] - ETA: 33s - loss: 0.2381 - categorical_accuracy: 0.9274
35968/60000 [================>.............] - ETA: 33s - loss: 0.2378 - categorical_accuracy: 0.9275
36032/60000 [=================>............] - ETA: 33s - loss: 0.2376 - categorical_accuracy: 0.9276
36064/60000 [=================>............] - ETA: 33s - loss: 0.2375 - categorical_accuracy: 0.9276
36128/60000 [=================>............] - ETA: 33s - loss: 0.2371 - categorical_accuracy: 0.9277
36192/60000 [=================>............] - ETA: 33s - loss: 0.2370 - categorical_accuracy: 0.9277
36256/60000 [=================>............] - ETA: 33s - loss: 0.2368 - categorical_accuracy: 0.9278
36320/60000 [=================>............] - ETA: 33s - loss: 0.2366 - categorical_accuracy: 0.9278
36384/60000 [=================>............] - ETA: 33s - loss: 0.2364 - categorical_accuracy: 0.9279
36448/60000 [=================>............] - ETA: 33s - loss: 0.2361 - categorical_accuracy: 0.9280
36512/60000 [=================>............] - ETA: 33s - loss: 0.2360 - categorical_accuracy: 0.9281
36576/60000 [=================>............] - ETA: 32s - loss: 0.2358 - categorical_accuracy: 0.9281
36640/60000 [=================>............] - ETA: 32s - loss: 0.2356 - categorical_accuracy: 0.9281
36704/60000 [=================>............] - ETA: 32s - loss: 0.2353 - categorical_accuracy: 0.9282
36768/60000 [=================>............] - ETA: 32s - loss: 0.2351 - categorical_accuracy: 0.9283
36832/60000 [=================>............] - ETA: 32s - loss: 0.2349 - categorical_accuracy: 0.9284
36896/60000 [=================>............] - ETA: 32s - loss: 0.2347 - categorical_accuracy: 0.9284
36928/60000 [=================>............] - ETA: 32s - loss: 0.2348 - categorical_accuracy: 0.9284
36992/60000 [=================>............] - ETA: 32s - loss: 0.2345 - categorical_accuracy: 0.9286
37056/60000 [=================>............] - ETA: 32s - loss: 0.2343 - categorical_accuracy: 0.9286
37120/60000 [=================>............] - ETA: 32s - loss: 0.2339 - categorical_accuracy: 0.9288
37184/60000 [=================>............] - ETA: 32s - loss: 0.2337 - categorical_accuracy: 0.9288
37248/60000 [=================>............] - ETA: 31s - loss: 0.2336 - categorical_accuracy: 0.9289
37312/60000 [=================>............] - ETA: 31s - loss: 0.2339 - categorical_accuracy: 0.9288
37344/60000 [=================>............] - ETA: 31s - loss: 0.2337 - categorical_accuracy: 0.9289
37376/60000 [=================>............] - ETA: 31s - loss: 0.2336 - categorical_accuracy: 0.9289
37440/60000 [=================>............] - ETA: 31s - loss: 0.2334 - categorical_accuracy: 0.9290
37504/60000 [=================>............] - ETA: 31s - loss: 0.2331 - categorical_accuracy: 0.9291
37568/60000 [=================>............] - ETA: 31s - loss: 0.2329 - categorical_accuracy: 0.9291
37632/60000 [=================>............] - ETA: 31s - loss: 0.2326 - categorical_accuracy: 0.9292
37696/60000 [=================>............] - ETA: 31s - loss: 0.2322 - categorical_accuracy: 0.9294
37760/60000 [=================>............] - ETA: 31s - loss: 0.2320 - categorical_accuracy: 0.9294
37824/60000 [=================>............] - ETA: 31s - loss: 0.2317 - categorical_accuracy: 0.9295
37888/60000 [=================>............] - ETA: 31s - loss: 0.2315 - categorical_accuracy: 0.9295
37952/60000 [=================>............] - ETA: 31s - loss: 0.2313 - categorical_accuracy: 0.9296
38016/60000 [==================>...........] - ETA: 30s - loss: 0.2310 - categorical_accuracy: 0.9297
38080/60000 [==================>...........] - ETA: 30s - loss: 0.2310 - categorical_accuracy: 0.9297
38144/60000 [==================>...........] - ETA: 30s - loss: 0.2306 - categorical_accuracy: 0.9298
38208/60000 [==================>...........] - ETA: 30s - loss: 0.2305 - categorical_accuracy: 0.9299
38272/60000 [==================>...........] - ETA: 30s - loss: 0.2302 - categorical_accuracy: 0.9299
38336/60000 [==================>...........] - ETA: 30s - loss: 0.2300 - categorical_accuracy: 0.9299
38400/60000 [==================>...........] - ETA: 30s - loss: 0.2300 - categorical_accuracy: 0.9299
38432/60000 [==================>...........] - ETA: 30s - loss: 0.2298 - categorical_accuracy: 0.9300
38496/60000 [==================>...........] - ETA: 30s - loss: 0.2297 - categorical_accuracy: 0.9300
38560/60000 [==================>...........] - ETA: 30s - loss: 0.2293 - categorical_accuracy: 0.9302
38624/60000 [==================>...........] - ETA: 30s - loss: 0.2292 - categorical_accuracy: 0.9302
38688/60000 [==================>...........] - ETA: 29s - loss: 0.2290 - categorical_accuracy: 0.9302
38752/60000 [==================>...........] - ETA: 29s - loss: 0.2287 - categorical_accuracy: 0.9303
38816/60000 [==================>...........] - ETA: 29s - loss: 0.2284 - categorical_accuracy: 0.9304
38880/60000 [==================>...........] - ETA: 29s - loss: 0.2282 - categorical_accuracy: 0.9304
38944/60000 [==================>...........] - ETA: 29s - loss: 0.2279 - categorical_accuracy: 0.9305
39008/60000 [==================>...........] - ETA: 29s - loss: 0.2277 - categorical_accuracy: 0.9306
39072/60000 [==================>...........] - ETA: 29s - loss: 0.2276 - categorical_accuracy: 0.9306
39136/60000 [==================>...........] - ETA: 29s - loss: 0.2273 - categorical_accuracy: 0.9306
39200/60000 [==================>...........] - ETA: 29s - loss: 0.2273 - categorical_accuracy: 0.9307
39264/60000 [==================>...........] - ETA: 29s - loss: 0.2270 - categorical_accuracy: 0.9308
39328/60000 [==================>...........] - ETA: 29s - loss: 0.2267 - categorical_accuracy: 0.9308
39392/60000 [==================>...........] - ETA: 28s - loss: 0.2265 - categorical_accuracy: 0.9309
39456/60000 [==================>...........] - ETA: 28s - loss: 0.2262 - categorical_accuracy: 0.9310
39520/60000 [==================>...........] - ETA: 28s - loss: 0.2261 - categorical_accuracy: 0.9310
39584/60000 [==================>...........] - ETA: 28s - loss: 0.2261 - categorical_accuracy: 0.9310
39648/60000 [==================>...........] - ETA: 28s - loss: 0.2259 - categorical_accuracy: 0.9311
39712/60000 [==================>...........] - ETA: 28s - loss: 0.2257 - categorical_accuracy: 0.9312
39776/60000 [==================>...........] - ETA: 28s - loss: 0.2254 - categorical_accuracy: 0.9312
39840/60000 [==================>...........] - ETA: 28s - loss: 0.2252 - categorical_accuracy: 0.9313
39904/60000 [==================>...........] - ETA: 28s - loss: 0.2250 - categorical_accuracy: 0.9313
39968/60000 [==================>...........] - ETA: 28s - loss: 0.2250 - categorical_accuracy: 0.9313
40032/60000 [===================>..........] - ETA: 28s - loss: 0.2248 - categorical_accuracy: 0.9314
40096/60000 [===================>..........] - ETA: 27s - loss: 0.2246 - categorical_accuracy: 0.9315
40160/60000 [===================>..........] - ETA: 27s - loss: 0.2243 - categorical_accuracy: 0.9316
40224/60000 [===================>..........] - ETA: 27s - loss: 0.2242 - categorical_accuracy: 0.9316
40288/60000 [===================>..........] - ETA: 27s - loss: 0.2239 - categorical_accuracy: 0.9316
40352/60000 [===================>..........] - ETA: 27s - loss: 0.2238 - categorical_accuracy: 0.9317
40416/60000 [===================>..........] - ETA: 27s - loss: 0.2237 - categorical_accuracy: 0.9317
40480/60000 [===================>..........] - ETA: 27s - loss: 0.2236 - categorical_accuracy: 0.9318
40544/60000 [===================>..........] - ETA: 27s - loss: 0.2233 - categorical_accuracy: 0.9319
40608/60000 [===================>..........] - ETA: 27s - loss: 0.2231 - categorical_accuracy: 0.9319
40672/60000 [===================>..........] - ETA: 27s - loss: 0.2230 - categorical_accuracy: 0.9319
40736/60000 [===================>..........] - ETA: 27s - loss: 0.2228 - categorical_accuracy: 0.9320
40800/60000 [===================>..........] - ETA: 26s - loss: 0.2225 - categorical_accuracy: 0.9321
40864/60000 [===================>..........] - ETA: 26s - loss: 0.2224 - categorical_accuracy: 0.9321
40928/60000 [===================>..........] - ETA: 26s - loss: 0.2220 - categorical_accuracy: 0.9322
40992/60000 [===================>..........] - ETA: 26s - loss: 0.2220 - categorical_accuracy: 0.9323
41056/60000 [===================>..........] - ETA: 26s - loss: 0.2218 - categorical_accuracy: 0.9323
41120/60000 [===================>..........] - ETA: 26s - loss: 0.2217 - categorical_accuracy: 0.9323
41152/60000 [===================>..........] - ETA: 26s - loss: 0.2216 - categorical_accuracy: 0.9324
41216/60000 [===================>..........] - ETA: 26s - loss: 0.2214 - categorical_accuracy: 0.9325
41280/60000 [===================>..........] - ETA: 26s - loss: 0.2213 - categorical_accuracy: 0.9325
41344/60000 [===================>..........] - ETA: 26s - loss: 0.2210 - categorical_accuracy: 0.9326
41408/60000 [===================>..........] - ETA: 26s - loss: 0.2210 - categorical_accuracy: 0.9325
41472/60000 [===================>..........] - ETA: 26s - loss: 0.2207 - categorical_accuracy: 0.9327
41536/60000 [===================>..........] - ETA: 25s - loss: 0.2206 - categorical_accuracy: 0.9327
41600/60000 [===================>..........] - ETA: 25s - loss: 0.2206 - categorical_accuracy: 0.9328
41664/60000 [===================>..........] - ETA: 25s - loss: 0.2205 - categorical_accuracy: 0.9328
41728/60000 [===================>..........] - ETA: 25s - loss: 0.2202 - categorical_accuracy: 0.9329
41792/60000 [===================>..........] - ETA: 25s - loss: 0.2201 - categorical_accuracy: 0.9329
41856/60000 [===================>..........] - ETA: 25s - loss: 0.2198 - categorical_accuracy: 0.9330
41920/60000 [===================>..........] - ETA: 25s - loss: 0.2197 - categorical_accuracy: 0.9330
41984/60000 [===================>..........] - ETA: 25s - loss: 0.2194 - categorical_accuracy: 0.9331
42048/60000 [====================>.........] - ETA: 25s - loss: 0.2193 - categorical_accuracy: 0.9331
42112/60000 [====================>.........] - ETA: 25s - loss: 0.2191 - categorical_accuracy: 0.9332
42176/60000 [====================>.........] - ETA: 25s - loss: 0.2188 - categorical_accuracy: 0.9333
42240/60000 [====================>.........] - ETA: 24s - loss: 0.2188 - categorical_accuracy: 0.9333
42304/60000 [====================>.........] - ETA: 24s - loss: 0.2188 - categorical_accuracy: 0.9333
42368/60000 [====================>.........] - ETA: 24s - loss: 0.2185 - categorical_accuracy: 0.9334
42432/60000 [====================>.........] - ETA: 24s - loss: 0.2182 - categorical_accuracy: 0.9335
42496/60000 [====================>.........] - ETA: 24s - loss: 0.2180 - categorical_accuracy: 0.9335
42560/60000 [====================>.........] - ETA: 24s - loss: 0.2178 - categorical_accuracy: 0.9336
42624/60000 [====================>.........] - ETA: 24s - loss: 0.2177 - categorical_accuracy: 0.9336
42688/60000 [====================>.........] - ETA: 24s - loss: 0.2174 - categorical_accuracy: 0.9337
42752/60000 [====================>.........] - ETA: 24s - loss: 0.2175 - categorical_accuracy: 0.9337
42816/60000 [====================>.........] - ETA: 24s - loss: 0.2172 - categorical_accuracy: 0.9338
42880/60000 [====================>.........] - ETA: 24s - loss: 0.2169 - categorical_accuracy: 0.9339
42944/60000 [====================>.........] - ETA: 23s - loss: 0.2166 - categorical_accuracy: 0.9340
43008/60000 [====================>.........] - ETA: 23s - loss: 0.2165 - categorical_accuracy: 0.9340
43072/60000 [====================>.........] - ETA: 23s - loss: 0.2162 - categorical_accuracy: 0.9341
43136/60000 [====================>.........] - ETA: 23s - loss: 0.2161 - categorical_accuracy: 0.9341
43200/60000 [====================>.........] - ETA: 23s - loss: 0.2160 - categorical_accuracy: 0.9341
43264/60000 [====================>.........] - ETA: 23s - loss: 0.2158 - categorical_accuracy: 0.9342
43328/60000 [====================>.........] - ETA: 23s - loss: 0.2158 - categorical_accuracy: 0.9342
43392/60000 [====================>.........] - ETA: 23s - loss: 0.2157 - categorical_accuracy: 0.9343
43456/60000 [====================>.........] - ETA: 23s - loss: 0.2155 - categorical_accuracy: 0.9343
43520/60000 [====================>.........] - ETA: 23s - loss: 0.2155 - categorical_accuracy: 0.9343
43584/60000 [====================>.........] - ETA: 23s - loss: 0.2153 - categorical_accuracy: 0.9344
43648/60000 [====================>.........] - ETA: 22s - loss: 0.2150 - categorical_accuracy: 0.9345
43712/60000 [====================>.........] - ETA: 22s - loss: 0.2149 - categorical_accuracy: 0.9345
43776/60000 [====================>.........] - ETA: 22s - loss: 0.2148 - categorical_accuracy: 0.9346
43840/60000 [====================>.........] - ETA: 22s - loss: 0.2148 - categorical_accuracy: 0.9346
43904/60000 [====================>.........] - ETA: 22s - loss: 0.2146 - categorical_accuracy: 0.9346
43968/60000 [====================>.........] - ETA: 22s - loss: 0.2144 - categorical_accuracy: 0.9347
44032/60000 [=====================>........] - ETA: 22s - loss: 0.2142 - categorical_accuracy: 0.9348
44096/60000 [=====================>........] - ETA: 22s - loss: 0.2140 - categorical_accuracy: 0.9348
44160/60000 [=====================>........] - ETA: 22s - loss: 0.2139 - categorical_accuracy: 0.9349
44224/60000 [=====================>........] - ETA: 22s - loss: 0.2137 - categorical_accuracy: 0.9349
44288/60000 [=====================>........] - ETA: 22s - loss: 0.2134 - categorical_accuracy: 0.9350
44352/60000 [=====================>........] - ETA: 21s - loss: 0.2133 - categorical_accuracy: 0.9350
44416/60000 [=====================>........] - ETA: 21s - loss: 0.2131 - categorical_accuracy: 0.9351
44480/60000 [=====================>........] - ETA: 21s - loss: 0.2129 - categorical_accuracy: 0.9351
44544/60000 [=====================>........] - ETA: 21s - loss: 0.2127 - categorical_accuracy: 0.9352
44576/60000 [=====================>........] - ETA: 21s - loss: 0.2125 - categorical_accuracy: 0.9352
44640/60000 [=====================>........] - ETA: 21s - loss: 0.2126 - categorical_accuracy: 0.9352
44672/60000 [=====================>........] - ETA: 21s - loss: 0.2125 - categorical_accuracy: 0.9353
44736/60000 [=====================>........] - ETA: 21s - loss: 0.2125 - categorical_accuracy: 0.9353
44768/60000 [=====================>........] - ETA: 21s - loss: 0.2124 - categorical_accuracy: 0.9353
44832/60000 [=====================>........] - ETA: 21s - loss: 0.2124 - categorical_accuracy: 0.9353
44896/60000 [=====================>........] - ETA: 21s - loss: 0.2122 - categorical_accuracy: 0.9354
44960/60000 [=====================>........] - ETA: 21s - loss: 0.2122 - categorical_accuracy: 0.9353
45024/60000 [=====================>........] - ETA: 21s - loss: 0.2120 - categorical_accuracy: 0.9354
45088/60000 [=====================>........] - ETA: 20s - loss: 0.2119 - categorical_accuracy: 0.9354
45152/60000 [=====================>........] - ETA: 20s - loss: 0.2117 - categorical_accuracy: 0.9355
45216/60000 [=====================>........] - ETA: 20s - loss: 0.2115 - categorical_accuracy: 0.9356
45280/60000 [=====================>........] - ETA: 20s - loss: 0.2115 - categorical_accuracy: 0.9356
45344/60000 [=====================>........] - ETA: 20s - loss: 0.2113 - categorical_accuracy: 0.9357
45376/60000 [=====================>........] - ETA: 20s - loss: 0.2113 - categorical_accuracy: 0.9357
45408/60000 [=====================>........] - ETA: 20s - loss: 0.2112 - categorical_accuracy: 0.9357
45472/60000 [=====================>........] - ETA: 20s - loss: 0.2111 - categorical_accuracy: 0.9357
45536/60000 [=====================>........] - ETA: 20s - loss: 0.2110 - categorical_accuracy: 0.9358
45600/60000 [=====================>........] - ETA: 20s - loss: 0.2108 - categorical_accuracy: 0.9359
45664/60000 [=====================>........] - ETA: 20s - loss: 0.2107 - categorical_accuracy: 0.9359
45728/60000 [=====================>........] - ETA: 20s - loss: 0.2107 - categorical_accuracy: 0.9359
45792/60000 [=====================>........] - ETA: 19s - loss: 0.2108 - categorical_accuracy: 0.9359
45856/60000 [=====================>........] - ETA: 19s - loss: 0.2106 - categorical_accuracy: 0.9360
45920/60000 [=====================>........] - ETA: 19s - loss: 0.2103 - categorical_accuracy: 0.9361
45984/60000 [=====================>........] - ETA: 19s - loss: 0.2102 - categorical_accuracy: 0.9361
46048/60000 [======================>.......] - ETA: 19s - loss: 0.2100 - categorical_accuracy: 0.9362
46112/60000 [======================>.......] - ETA: 19s - loss: 0.2099 - categorical_accuracy: 0.9362
46176/60000 [======================>.......] - ETA: 19s - loss: 0.2098 - categorical_accuracy: 0.9363
46240/60000 [======================>.......] - ETA: 19s - loss: 0.2095 - categorical_accuracy: 0.9364
46272/60000 [======================>.......] - ETA: 19s - loss: 0.2094 - categorical_accuracy: 0.9364
46304/60000 [======================>.......] - ETA: 19s - loss: 0.2093 - categorical_accuracy: 0.9364
46368/60000 [======================>.......] - ETA: 19s - loss: 0.2091 - categorical_accuracy: 0.9365
46432/60000 [======================>.......] - ETA: 19s - loss: 0.2089 - categorical_accuracy: 0.9365
46496/60000 [======================>.......] - ETA: 18s - loss: 0.2087 - categorical_accuracy: 0.9366
46560/60000 [======================>.......] - ETA: 18s - loss: 0.2085 - categorical_accuracy: 0.9366
46624/60000 [======================>.......] - ETA: 18s - loss: 0.2085 - categorical_accuracy: 0.9367
46688/60000 [======================>.......] - ETA: 18s - loss: 0.2083 - categorical_accuracy: 0.9367
46752/60000 [======================>.......] - ETA: 18s - loss: 0.2081 - categorical_accuracy: 0.9368
46816/60000 [======================>.......] - ETA: 18s - loss: 0.2079 - categorical_accuracy: 0.9368
46880/60000 [======================>.......] - ETA: 18s - loss: 0.2078 - categorical_accuracy: 0.9368
46944/60000 [======================>.......] - ETA: 18s - loss: 0.2076 - categorical_accuracy: 0.9369
47008/60000 [======================>.......] - ETA: 18s - loss: 0.2076 - categorical_accuracy: 0.9369
47072/60000 [======================>.......] - ETA: 18s - loss: 0.2074 - categorical_accuracy: 0.9369
47136/60000 [======================>.......] - ETA: 18s - loss: 0.2073 - categorical_accuracy: 0.9370
47200/60000 [======================>.......] - ETA: 17s - loss: 0.2071 - categorical_accuracy: 0.9371
47264/60000 [======================>.......] - ETA: 17s - loss: 0.2069 - categorical_accuracy: 0.9371
47328/60000 [======================>.......] - ETA: 17s - loss: 0.2068 - categorical_accuracy: 0.9372
47392/60000 [======================>.......] - ETA: 17s - loss: 0.2068 - categorical_accuracy: 0.9372
47456/60000 [======================>.......] - ETA: 17s - loss: 0.2066 - categorical_accuracy: 0.9373
47488/60000 [======================>.......] - ETA: 17s - loss: 0.2065 - categorical_accuracy: 0.9373
47520/60000 [======================>.......] - ETA: 17s - loss: 0.2066 - categorical_accuracy: 0.9373
47584/60000 [======================>.......] - ETA: 17s - loss: 0.2064 - categorical_accuracy: 0.9374
47648/60000 [======================>.......] - ETA: 17s - loss: 0.2062 - categorical_accuracy: 0.9374
47680/60000 [======================>.......] - ETA: 17s - loss: 0.2061 - categorical_accuracy: 0.9375
47744/60000 [======================>.......] - ETA: 17s - loss: 0.2061 - categorical_accuracy: 0.9375
47808/60000 [======================>.......] - ETA: 17s - loss: 0.2058 - categorical_accuracy: 0.9376
47872/60000 [======================>.......] - ETA: 17s - loss: 0.2057 - categorical_accuracy: 0.9376
47936/60000 [======================>.......] - ETA: 16s - loss: 0.2057 - categorical_accuracy: 0.9376
47968/60000 [======================>.......] - ETA: 16s - loss: 0.2057 - categorical_accuracy: 0.9376
48032/60000 [=======================>......] - ETA: 16s - loss: 0.2056 - categorical_accuracy: 0.9376
48096/60000 [=======================>......] - ETA: 16s - loss: 0.2055 - categorical_accuracy: 0.9377
48160/60000 [=======================>......] - ETA: 16s - loss: 0.2053 - categorical_accuracy: 0.9377
48224/60000 [=======================>......] - ETA: 16s - loss: 0.2051 - categorical_accuracy: 0.9378
48288/60000 [=======================>......] - ETA: 16s - loss: 0.2050 - categorical_accuracy: 0.9378
48352/60000 [=======================>......] - ETA: 16s - loss: 0.2049 - categorical_accuracy: 0.9379
48416/60000 [=======================>......] - ETA: 16s - loss: 0.2047 - categorical_accuracy: 0.9379
48480/60000 [=======================>......] - ETA: 16s - loss: 0.2044 - categorical_accuracy: 0.9380
48544/60000 [=======================>......] - ETA: 16s - loss: 0.2043 - categorical_accuracy: 0.9380
48608/60000 [=======================>......] - ETA: 15s - loss: 0.2041 - categorical_accuracy: 0.9381
48672/60000 [=======================>......] - ETA: 15s - loss: 0.2039 - categorical_accuracy: 0.9382
48736/60000 [=======================>......] - ETA: 15s - loss: 0.2037 - categorical_accuracy: 0.9382
48800/60000 [=======================>......] - ETA: 15s - loss: 0.2035 - categorical_accuracy: 0.9382
48864/60000 [=======================>......] - ETA: 15s - loss: 0.2035 - categorical_accuracy: 0.9383
48928/60000 [=======================>......] - ETA: 15s - loss: 0.2034 - categorical_accuracy: 0.9383
48992/60000 [=======================>......] - ETA: 15s - loss: 0.2031 - categorical_accuracy: 0.9384
49056/60000 [=======================>......] - ETA: 15s - loss: 0.2031 - categorical_accuracy: 0.9384
49120/60000 [=======================>......] - ETA: 15s - loss: 0.2029 - categorical_accuracy: 0.9385
49184/60000 [=======================>......] - ETA: 15s - loss: 0.2027 - categorical_accuracy: 0.9385
49248/60000 [=======================>......] - ETA: 15s - loss: 0.2026 - categorical_accuracy: 0.9386
49312/60000 [=======================>......] - ETA: 15s - loss: 0.2026 - categorical_accuracy: 0.9386
49376/60000 [=======================>......] - ETA: 14s - loss: 0.2024 - categorical_accuracy: 0.9386
49440/60000 [=======================>......] - ETA: 14s - loss: 0.2022 - categorical_accuracy: 0.9387
49504/60000 [=======================>......] - ETA: 14s - loss: 0.2020 - categorical_accuracy: 0.9388
49568/60000 [=======================>......] - ETA: 14s - loss: 0.2019 - categorical_accuracy: 0.9388
49632/60000 [=======================>......] - ETA: 14s - loss: 0.2018 - categorical_accuracy: 0.9388
49696/60000 [=======================>......] - ETA: 14s - loss: 0.2018 - categorical_accuracy: 0.9389
49760/60000 [=======================>......] - ETA: 14s - loss: 0.2017 - categorical_accuracy: 0.9389
49824/60000 [=======================>......] - ETA: 14s - loss: 0.2015 - categorical_accuracy: 0.9390
49888/60000 [=======================>......] - ETA: 14s - loss: 0.2013 - categorical_accuracy: 0.9390
49952/60000 [=======================>......] - ETA: 14s - loss: 0.2013 - categorical_accuracy: 0.9391
50016/60000 [========================>.....] - ETA: 14s - loss: 0.2011 - categorical_accuracy: 0.9391
50080/60000 [========================>.....] - ETA: 13s - loss: 0.2011 - categorical_accuracy: 0.9391
50144/60000 [========================>.....] - ETA: 13s - loss: 0.2009 - categorical_accuracy: 0.9392
50208/60000 [========================>.....] - ETA: 13s - loss: 0.2007 - categorical_accuracy: 0.9392
50272/60000 [========================>.....] - ETA: 13s - loss: 0.2006 - categorical_accuracy: 0.9393
50336/60000 [========================>.....] - ETA: 13s - loss: 0.2006 - categorical_accuracy: 0.9393
50400/60000 [========================>.....] - ETA: 13s - loss: 0.2005 - categorical_accuracy: 0.9393
50464/60000 [========================>.....] - ETA: 13s - loss: 0.2003 - categorical_accuracy: 0.9394
50528/60000 [========================>.....] - ETA: 13s - loss: 0.2001 - categorical_accuracy: 0.9394
50592/60000 [========================>.....] - ETA: 13s - loss: 0.2001 - categorical_accuracy: 0.9395
50656/60000 [========================>.....] - ETA: 13s - loss: 0.1999 - categorical_accuracy: 0.9395
50720/60000 [========================>.....] - ETA: 13s - loss: 0.2000 - categorical_accuracy: 0.9395
50784/60000 [========================>.....] - ETA: 12s - loss: 0.1999 - categorical_accuracy: 0.9396
50848/60000 [========================>.....] - ETA: 12s - loss: 0.2000 - categorical_accuracy: 0.9395
50912/60000 [========================>.....] - ETA: 12s - loss: 0.1998 - categorical_accuracy: 0.9396
50976/60000 [========================>.....] - ETA: 12s - loss: 0.1996 - categorical_accuracy: 0.9396
51040/60000 [========================>.....] - ETA: 12s - loss: 0.1994 - categorical_accuracy: 0.9397
51104/60000 [========================>.....] - ETA: 12s - loss: 0.1992 - categorical_accuracy: 0.9398
51168/60000 [========================>.....] - ETA: 12s - loss: 0.1990 - categorical_accuracy: 0.9398
51232/60000 [========================>.....] - ETA: 12s - loss: 0.1989 - categorical_accuracy: 0.9398
51296/60000 [========================>.....] - ETA: 12s - loss: 0.1986 - categorical_accuracy: 0.9399
51360/60000 [========================>.....] - ETA: 12s - loss: 0.1984 - categorical_accuracy: 0.9400
51424/60000 [========================>.....] - ETA: 12s - loss: 0.1982 - categorical_accuracy: 0.9400
51488/60000 [========================>.....] - ETA: 11s - loss: 0.1980 - categorical_accuracy: 0.9401
51552/60000 [========================>.....] - ETA: 11s - loss: 0.1979 - categorical_accuracy: 0.9401
51616/60000 [========================>.....] - ETA: 11s - loss: 0.1977 - categorical_accuracy: 0.9402
51648/60000 [========================>.....] - ETA: 11s - loss: 0.1976 - categorical_accuracy: 0.9402
51712/60000 [========================>.....] - ETA: 11s - loss: 0.1974 - categorical_accuracy: 0.9403
51776/60000 [========================>.....] - ETA: 11s - loss: 0.1973 - categorical_accuracy: 0.9403
51840/60000 [========================>.....] - ETA: 11s - loss: 0.1971 - categorical_accuracy: 0.9404
51904/60000 [========================>.....] - ETA: 11s - loss: 0.1969 - categorical_accuracy: 0.9404
51968/60000 [========================>.....] - ETA: 11s - loss: 0.1969 - categorical_accuracy: 0.9404
52032/60000 [=========================>....] - ETA: 11s - loss: 0.1968 - categorical_accuracy: 0.9405
52096/60000 [=========================>....] - ETA: 11s - loss: 0.1966 - categorical_accuracy: 0.9406
52160/60000 [=========================>....] - ETA: 11s - loss: 0.1965 - categorical_accuracy: 0.9406
52224/60000 [=========================>....] - ETA: 10s - loss: 0.1965 - categorical_accuracy: 0.9406
52288/60000 [=========================>....] - ETA: 10s - loss: 0.1965 - categorical_accuracy: 0.9407
52352/60000 [=========================>....] - ETA: 10s - loss: 0.1963 - categorical_accuracy: 0.9407
52416/60000 [=========================>....] - ETA: 10s - loss: 0.1961 - categorical_accuracy: 0.9408
52480/60000 [=========================>....] - ETA: 10s - loss: 0.1960 - categorical_accuracy: 0.9408
52544/60000 [=========================>....] - ETA: 10s - loss: 0.1960 - categorical_accuracy: 0.9408
52608/60000 [=========================>....] - ETA: 10s - loss: 0.1960 - categorical_accuracy: 0.9408
52672/60000 [=========================>....] - ETA: 10s - loss: 0.1958 - categorical_accuracy: 0.9408
52704/60000 [=========================>....] - ETA: 10s - loss: 0.1958 - categorical_accuracy: 0.9408
52736/60000 [=========================>....] - ETA: 10s - loss: 0.1957 - categorical_accuracy: 0.9409
52800/60000 [=========================>....] - ETA: 10s - loss: 0.1955 - categorical_accuracy: 0.9409
52864/60000 [=========================>....] - ETA: 10s - loss: 0.1953 - categorical_accuracy: 0.9410
52928/60000 [=========================>....] - ETA: 9s - loss: 0.1952 - categorical_accuracy: 0.9411 
52992/60000 [=========================>....] - ETA: 9s - loss: 0.1953 - categorical_accuracy: 0.9411
53056/60000 [=========================>....] - ETA: 9s - loss: 0.1951 - categorical_accuracy: 0.9411
53120/60000 [=========================>....] - ETA: 9s - loss: 0.1951 - categorical_accuracy: 0.9411
53152/60000 [=========================>....] - ETA: 9s - loss: 0.1950 - categorical_accuracy: 0.9411
53216/60000 [=========================>....] - ETA: 9s - loss: 0.1948 - categorical_accuracy: 0.9412
53280/60000 [=========================>....] - ETA: 9s - loss: 0.1946 - categorical_accuracy: 0.9413
53344/60000 [=========================>....] - ETA: 9s - loss: 0.1945 - categorical_accuracy: 0.9413
53408/60000 [=========================>....] - ETA: 9s - loss: 0.1943 - categorical_accuracy: 0.9413
53472/60000 [=========================>....] - ETA: 9s - loss: 0.1946 - categorical_accuracy: 0.9413
53536/60000 [=========================>....] - ETA: 9s - loss: 0.1944 - categorical_accuracy: 0.9413
53600/60000 [=========================>....] - ETA: 8s - loss: 0.1942 - categorical_accuracy: 0.9414
53664/60000 [=========================>....] - ETA: 8s - loss: 0.1941 - categorical_accuracy: 0.9414
53728/60000 [=========================>....] - ETA: 8s - loss: 0.1940 - categorical_accuracy: 0.9415
53792/60000 [=========================>....] - ETA: 8s - loss: 0.1938 - categorical_accuracy: 0.9415
53856/60000 [=========================>....] - ETA: 8s - loss: 0.1937 - categorical_accuracy: 0.9415
53920/60000 [=========================>....] - ETA: 8s - loss: 0.1936 - categorical_accuracy: 0.9415
53984/60000 [=========================>....] - ETA: 8s - loss: 0.1935 - categorical_accuracy: 0.9416
54048/60000 [==========================>...] - ETA: 8s - loss: 0.1933 - categorical_accuracy: 0.9416
54112/60000 [==========================>...] - ETA: 8s - loss: 0.1933 - categorical_accuracy: 0.9416
54176/60000 [==========================>...] - ETA: 8s - loss: 0.1932 - categorical_accuracy: 0.9417
54240/60000 [==========================>...] - ETA: 8s - loss: 0.1930 - categorical_accuracy: 0.9417
54304/60000 [==========================>...] - ETA: 7s - loss: 0.1929 - categorical_accuracy: 0.9418
54368/60000 [==========================>...] - ETA: 7s - loss: 0.1927 - categorical_accuracy: 0.9418
54432/60000 [==========================>...] - ETA: 7s - loss: 0.1926 - categorical_accuracy: 0.9419
54496/60000 [==========================>...] - ETA: 7s - loss: 0.1924 - categorical_accuracy: 0.9419
54560/60000 [==========================>...] - ETA: 7s - loss: 0.1924 - categorical_accuracy: 0.9419
54624/60000 [==========================>...] - ETA: 7s - loss: 0.1923 - categorical_accuracy: 0.9419
54656/60000 [==========================>...] - ETA: 7s - loss: 0.1923 - categorical_accuracy: 0.9420
54720/60000 [==========================>...] - ETA: 7s - loss: 0.1923 - categorical_accuracy: 0.9420
54784/60000 [==========================>...] - ETA: 7s - loss: 0.1923 - categorical_accuracy: 0.9420
54848/60000 [==========================>...] - ETA: 7s - loss: 0.1921 - categorical_accuracy: 0.9420
54912/60000 [==========================>...] - ETA: 7s - loss: 0.1920 - categorical_accuracy: 0.9421
54976/60000 [==========================>...] - ETA: 7s - loss: 0.1919 - categorical_accuracy: 0.9421
55040/60000 [==========================>...] - ETA: 6s - loss: 0.1917 - categorical_accuracy: 0.9422
55104/60000 [==========================>...] - ETA: 6s - loss: 0.1917 - categorical_accuracy: 0.9422
55168/60000 [==========================>...] - ETA: 6s - loss: 0.1915 - categorical_accuracy: 0.9422
55232/60000 [==========================>...] - ETA: 6s - loss: 0.1914 - categorical_accuracy: 0.9423
55296/60000 [==========================>...] - ETA: 6s - loss: 0.1914 - categorical_accuracy: 0.9423
55360/60000 [==========================>...] - ETA: 6s - loss: 0.1913 - categorical_accuracy: 0.9423
55424/60000 [==========================>...] - ETA: 6s - loss: 0.1911 - categorical_accuracy: 0.9423
55488/60000 [==========================>...] - ETA: 6s - loss: 0.1911 - categorical_accuracy: 0.9423
55552/60000 [==========================>...] - ETA: 6s - loss: 0.1910 - categorical_accuracy: 0.9423
55616/60000 [==========================>...] - ETA: 6s - loss: 0.1909 - categorical_accuracy: 0.9423
55680/60000 [==========================>...] - ETA: 6s - loss: 0.1910 - categorical_accuracy: 0.9423
55744/60000 [==========================>...] - ETA: 5s - loss: 0.1909 - categorical_accuracy: 0.9424
55808/60000 [==========================>...] - ETA: 5s - loss: 0.1909 - categorical_accuracy: 0.9424
55872/60000 [==========================>...] - ETA: 5s - loss: 0.1908 - categorical_accuracy: 0.9424
55936/60000 [==========================>...] - ETA: 5s - loss: 0.1907 - categorical_accuracy: 0.9424
56000/60000 [===========================>..] - ETA: 5s - loss: 0.1905 - categorical_accuracy: 0.9425
56064/60000 [===========================>..] - ETA: 5s - loss: 0.1903 - categorical_accuracy: 0.9426
56128/60000 [===========================>..] - ETA: 5s - loss: 0.1902 - categorical_accuracy: 0.9426
56192/60000 [===========================>..] - ETA: 5s - loss: 0.1903 - categorical_accuracy: 0.9426
56256/60000 [===========================>..] - ETA: 5s - loss: 0.1902 - categorical_accuracy: 0.9426
56320/60000 [===========================>..] - ETA: 5s - loss: 0.1902 - categorical_accuracy: 0.9426
56384/60000 [===========================>..] - ETA: 5s - loss: 0.1902 - categorical_accuracy: 0.9426
56448/60000 [===========================>..] - ETA: 4s - loss: 0.1901 - categorical_accuracy: 0.9426
56512/60000 [===========================>..] - ETA: 4s - loss: 0.1900 - categorical_accuracy: 0.9427
56576/60000 [===========================>..] - ETA: 4s - loss: 0.1898 - categorical_accuracy: 0.9427
56640/60000 [===========================>..] - ETA: 4s - loss: 0.1898 - categorical_accuracy: 0.9428
56704/60000 [===========================>..] - ETA: 4s - loss: 0.1898 - categorical_accuracy: 0.9428
56768/60000 [===========================>..] - ETA: 4s - loss: 0.1897 - categorical_accuracy: 0.9428
56832/60000 [===========================>..] - ETA: 4s - loss: 0.1895 - categorical_accuracy: 0.9429
56896/60000 [===========================>..] - ETA: 4s - loss: 0.1895 - categorical_accuracy: 0.9429
56960/60000 [===========================>..] - ETA: 4s - loss: 0.1893 - categorical_accuracy: 0.9429
57024/60000 [===========================>..] - ETA: 4s - loss: 0.1891 - categorical_accuracy: 0.9430
57088/60000 [===========================>..] - ETA: 4s - loss: 0.1890 - categorical_accuracy: 0.9430
57152/60000 [===========================>..] - ETA: 3s - loss: 0.1889 - categorical_accuracy: 0.9431
57216/60000 [===========================>..] - ETA: 3s - loss: 0.1888 - categorical_accuracy: 0.9431
57280/60000 [===========================>..] - ETA: 3s - loss: 0.1887 - categorical_accuracy: 0.9431
57344/60000 [===========================>..] - ETA: 3s - loss: 0.1886 - categorical_accuracy: 0.9431
57408/60000 [===========================>..] - ETA: 3s - loss: 0.1887 - categorical_accuracy: 0.9431
57472/60000 [===========================>..] - ETA: 3s - loss: 0.1886 - categorical_accuracy: 0.9431
57536/60000 [===========================>..] - ETA: 3s - loss: 0.1884 - categorical_accuracy: 0.9432
57600/60000 [===========================>..] - ETA: 3s - loss: 0.1883 - categorical_accuracy: 0.9432
57664/60000 [===========================>..] - ETA: 3s - loss: 0.1882 - categorical_accuracy: 0.9432
57696/60000 [===========================>..] - ETA: 3s - loss: 0.1881 - categorical_accuracy: 0.9433
57760/60000 [===========================>..] - ETA: 3s - loss: 0.1879 - categorical_accuracy: 0.9433
57824/60000 [===========================>..] - ETA: 3s - loss: 0.1878 - categorical_accuracy: 0.9434
57888/60000 [===========================>..] - ETA: 2s - loss: 0.1877 - categorical_accuracy: 0.9434
57952/60000 [===========================>..] - ETA: 2s - loss: 0.1875 - categorical_accuracy: 0.9434
58016/60000 [============================>.] - ETA: 2s - loss: 0.1874 - categorical_accuracy: 0.9435
58080/60000 [============================>.] - ETA: 2s - loss: 0.1875 - categorical_accuracy: 0.9435
58144/60000 [============================>.] - ETA: 2s - loss: 0.1874 - categorical_accuracy: 0.9435
58208/60000 [============================>.] - ETA: 2s - loss: 0.1873 - categorical_accuracy: 0.9435
58240/60000 [============================>.] - ETA: 2s - loss: 0.1872 - categorical_accuracy: 0.9435
58304/60000 [============================>.] - ETA: 2s - loss: 0.1870 - categorical_accuracy: 0.9436
58368/60000 [============================>.] - ETA: 2s - loss: 0.1871 - categorical_accuracy: 0.9436
58432/60000 [============================>.] - ETA: 2s - loss: 0.1869 - categorical_accuracy: 0.9436
58496/60000 [============================>.] - ETA: 2s - loss: 0.1869 - categorical_accuracy: 0.9436
58560/60000 [============================>.] - ETA: 2s - loss: 0.1868 - categorical_accuracy: 0.9436
58624/60000 [============================>.] - ETA: 1s - loss: 0.1866 - categorical_accuracy: 0.9437
58688/60000 [============================>.] - ETA: 1s - loss: 0.1868 - categorical_accuracy: 0.9437
58752/60000 [============================>.] - ETA: 1s - loss: 0.1866 - categorical_accuracy: 0.9437
58816/60000 [============================>.] - ETA: 1s - loss: 0.1865 - categorical_accuracy: 0.9438
58880/60000 [============================>.] - ETA: 1s - loss: 0.1864 - categorical_accuracy: 0.9438
58944/60000 [============================>.] - ETA: 1s - loss: 0.1864 - categorical_accuracy: 0.9438
59008/60000 [============================>.] - ETA: 1s - loss: 0.1862 - categorical_accuracy: 0.9438
59072/60000 [============================>.] - ETA: 1s - loss: 0.1863 - categorical_accuracy: 0.9438
59136/60000 [============================>.] - ETA: 1s - loss: 0.1865 - categorical_accuracy: 0.9438
59200/60000 [============================>.] - ETA: 1s - loss: 0.1865 - categorical_accuracy: 0.9438
59264/60000 [============================>.] - ETA: 1s - loss: 0.1865 - categorical_accuracy: 0.9438
59328/60000 [============================>.] - ETA: 0s - loss: 0.1863 - categorical_accuracy: 0.9438
59392/60000 [============================>.] - ETA: 0s - loss: 0.1863 - categorical_accuracy: 0.9438
59456/60000 [============================>.] - ETA: 0s - loss: 0.1862 - categorical_accuracy: 0.9439
59520/60000 [============================>.] - ETA: 0s - loss: 0.1860 - categorical_accuracy: 0.9439
59584/60000 [============================>.] - ETA: 0s - loss: 0.1859 - categorical_accuracy: 0.9440
59648/60000 [============================>.] - ETA: 0s - loss: 0.1858 - categorical_accuracy: 0.9440
59712/60000 [============================>.] - ETA: 0s - loss: 0.1857 - categorical_accuracy: 0.9441
59776/60000 [============================>.] - ETA: 0s - loss: 0.1856 - categorical_accuracy: 0.9441
59840/60000 [============================>.] - ETA: 0s - loss: 0.1855 - categorical_accuracy: 0.9442
59904/60000 [============================>.] - ETA: 0s - loss: 0.1853 - categorical_accuracy: 0.9442
59968/60000 [============================>.] - ETA: 0s - loss: 0.1852 - categorical_accuracy: 0.9442
60000/60000 [==============================] - 87s 1ms/step - loss: 0.1852 - categorical_accuracy: 0.9443 - val_loss: 0.0457 - val_categorical_accuracy: 0.9848

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
 3072/10000 [========>.....................] - ETA: 2s
 3264/10000 [========>.....................] - ETA: 1s
 3456/10000 [=========>....................] - ETA: 1s
 3648/10000 [=========>....................] - ETA: 1s
 3840/10000 [==========>...................] - ETA: 1s
 4032/10000 [===========>..................] - ETA: 1s
 4224/10000 [===========>..................] - ETA: 1s
 4416/10000 [============>.................] - ETA: 1s
 4608/10000 [============>.................] - ETA: 1s
 4800/10000 [=============>................] - ETA: 1s
 4992/10000 [=============>................] - ETA: 1s
 5184/10000 [==============>...............] - ETA: 1s
 5376/10000 [===============>..............] - ETA: 1s
 5536/10000 [===============>..............] - ETA: 1s
 5728/10000 [================>.............] - ETA: 1s
 5920/10000 [================>.............] - ETA: 1s
 6112/10000 [=================>............] - ETA: 1s
 6304/10000 [=================>............] - ETA: 1s
 6496/10000 [==================>...........] - ETA: 1s
 6688/10000 [===================>..........] - ETA: 0s
 6880/10000 [===================>..........] - ETA: 0s
 7072/10000 [====================>.........] - ETA: 0s
 7264/10000 [====================>.........] - ETA: 0s
 7456/10000 [=====================>........] - ETA: 0s
 7648/10000 [=====================>........] - ETA: 0s
 7840/10000 [======================>.......] - ETA: 0s
 8032/10000 [=======================>......] - ETA: 0s
 8224/10000 [=======================>......] - ETA: 0s
 8416/10000 [========================>.....] - ETA: 0s
 8608/10000 [========================>.....] - ETA: 0s
 8800/10000 [=========================>....] - ETA: 0s
 8992/10000 [=========================>....] - ETA: 0s
 9184/10000 [==========================>...] - ETA: 0s
 9376/10000 [===========================>..] - ETA: 0s
 9568/10000 [===========================>..] - ETA: 0s
 9760/10000 [============================>.] - ETA: 0s
 9952/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 3s 285us/step
[[1.3190526e-08 3.0718846e-07 1.0664202e-06 ... 9.9998605e-01
  5.5730279e-08 1.1049097e-05]
 [3.4514651e-05 4.5123776e-05 9.9988306e-01 ... 3.0056819e-09
  1.1899381e-05 1.1945115e-08]
 [2.1709221e-07 9.9995184e-01 8.1328408e-06 ... 2.4110239e-05
  6.1863489e-06 1.1932622e-07]
 ...
 [1.0604913e-08 1.2995258e-06 1.3067532e-07 ... 3.0772703e-06
  2.1277008e-05 1.1340581e-04]
 [1.3665998e-05 1.8111335e-06 2.3359118e-07 ... 5.4299716e-07
  7.4097137e-03 3.2228620e-06]
 [6.6189864e-06 4.0490618e-06 1.0668946e-05 ... 7.5672872e-09
  2.3305042e-06 2.8227218e-08]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.04567991894130828, 'accuracy_test:': 0.9847999811172485}

  ('#### Save   #######################################################',) 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/charcnn/result'}

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   ea14fa0..a0e9947  master     -> origin/master
Updating ea14fa0..a0e9947
Fast-forward
 error_list/20200514/list_log_json_20200514.md    | 1146 +++++++++++-----------
 error_list/20200514/list_log_testall_20200514.md |  573 +++++++++++
 2 files changed, 1146 insertions(+), 573 deletions(-)
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
[master 375e151] ml_store
 1 file changed, 1132 insertions(+)
To github.com:arita37/mlmodels_store.git
   a0e9947..375e151  master -> master





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
{'loss': 0.41947210766375065, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-14 20:32:58.778904: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
[master 795ed49] ml_store
 1 file changed, 233 insertions(+)
To github.com:arita37/mlmodels_store.git
   375e151..795ed49  master -> master





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
[master df0f30b] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   795ed49..df0f30b  master -> master





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
 40%|████      | 2/5 [00:20<00:31, 10.46s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.8538239187242113, 'learning_rate': 0.012175534417642962, 'min_data_in_leaf': 7, 'num_leaves': 33} and reward: 0.3888
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xebR\x86\x89\xee\xd1MX\r\x00\x00\x00learning_rateq\x02G?\x88\xef|\x91\x14\xae\xbfX\x10\x00\x00\x00min_data_in_leafq\x03K\x07X\n\x00\x00\x00num_leavesq\x04K!u.' and reward: 0.3888
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xebR\x86\x89\xee\xd1MX\r\x00\x00\x00learning_rateq\x02G?\x88\xef|\x91\x14\xae\xbfX\x10\x00\x00\x00min_data_in_leafq\x03K\x07X\n\x00\x00\x00num_leavesq\x04K!u.' and reward: 0.3888
 60%|██████    | 3/5 [00:41<00:26, 13.41s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.8564624404347312, 'learning_rate': 0.029287203048185287, 'min_data_in_leaf': 11, 'num_leaves': 55} and reward: 0.392
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xebh#\xeb}l\xb2X\r\x00\x00\x00learning_rateq\x02G?\x9d\xfdv\xed"\x10\xd4X\x10\x00\x00\x00min_data_in_leafq\x03K\x0bX\n\x00\x00\x00num_leavesq\x04K7u.' and reward: 0.392
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xebh#\xeb}l\xb2X\r\x00\x00\x00learning_rateq\x02G?\x9d\xfdv\xed"\x10\xd4X\x10\x00\x00\x00min_data_in_leafq\x03K\x0bX\n\x00\x00\x00num_leavesq\x04K7u.' and reward: 0.392
 80%|████████  | 4/5 [01:09<00:17, 17.80s/it] 80%|████████  | 4/5 [01:09<00:17, 17.32s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.8017539679749515, 'learning_rate': 0.02255266909348274, 'min_data_in_leaf': 29, 'num_leaves': 34} and reward: 0.3914
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9\xa7\xf7\xef\xfc\x800X\r\x00\x00\x00learning_rateq\x02G?\x97\x18\x0c\x00\xc6\xac\xbeX\x10\x00\x00\x00min_data_in_leafq\x03K\x1dX\n\x00\x00\x00num_leavesq\x04K"u.' and reward: 0.3914
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9\xa7\xf7\xef\xfc\x800X\r\x00\x00\x00learning_rateq\x02G?\x97\x18\x0c\x00\xc6\xac\xbeX\x10\x00\x00\x00min_data_in_leafq\x03K\x1dX\n\x00\x00\x00num_leavesq\x04K"u.' and reward: 0.3914
Time for Gradient Boosting hyperparameter optimization: 89.79005455970764
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.8564624404347312, 'learning_rate': 0.029287203048185287, 'min_data_in_leaf': 11, 'num_leaves': 55}
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
Saving dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3894
 40%|████      | 2/5 [00:50<01:15, 25.24s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.40544211633999405, 'embedding_size_factor': 0.6583476802320501, 'layers.choice': 3, 'learning_rate': 0.00606241724794919, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 3.154687703498482e-07} and reward: 0.3696
Finished Task with config: b"\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd9\xf2\xc3}\x86{\xbaX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe5\x11/'\x7f\xcf\xd7X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?x\xd4\xe7\xbd\x08\xccqX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\x95+\xb6S&\x0e\x02u." and reward: 0.3696
Finished Task with config: b"\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd9\xf2\xc3}\x86{\xbaX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe5\x11/'\x7f\xcf\xd7X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?x\xd4\xe7\xbd\x08\xccqX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\x95+\xb6S&\x0e\x02u." and reward: 0.3696
 60%|██████    | 3/5 [01:42<01:06, 33.37s/it] 60%|██████    | 3/5 [01:42<01:08, 34.27s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.08109751819219609, 'embedding_size_factor': 1.3607190429159113, 'layers.choice': 0, 'learning_rate': 0.003920392261180147, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 6.269850727005202e-12} and reward: 0.3798
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xb4\xc2\xce\x94l\x18eX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5\xc5\x81T\xc5\xe4dX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?p\x0e\xd4H\xc9\xc5\x16X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\x9b\x939n\xf78\x92u.' and reward: 0.3798
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xb4\xc2\xce\x94l\x18eX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5\xc5\x81T\xc5\xe4dX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?p\x0e\xd4H\xc9\xc5\x16X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\x9b\x939n\xf78\x92u.' and reward: 0.3798
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 154.75170373916626
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
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.76s of the -128.66s of remaining time.
Ensemble size: 12
Ensemble weights: 
[0.66666667 0.         0.         0.08333333 0.25       0.
 0.        ]
	0.3926	 = Validation accuracy score
	1.57s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 250.28s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7fc0b0259cf8>
Task exception was never retrieved
future: <Task finished coro=<InProcConnector.connect() done, defined at /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/distributed/comm/inproc.py:285> exception=OSError("no endpoint for inproc address '10.1.0.4/7328/1'",)>
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/distributed/comm/inproc.py", line 288, in connect
    raise IOError("no endpoint for inproc address %r" % (address,))
OSError: no endpoint for inproc address '10.1.0.4/7328/1'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   df0f30b..0bad116  master     -> origin/master
Updating df0f30b..0bad116
Fast-forward
 error_list/20200514/list_log_benchmark_20200514.md |  166 +--
 error_list/20200514/list_log_json_20200514.md      | 1146 ++++++++++----------
 error_list/20200514/list_log_testall_20200514.md   |  174 +++
 3 files changed, 830 insertions(+), 656 deletions(-)
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
[master 55ba88e] ml_store
 1 file changed, 274 insertions(+)
To github.com:arita37/mlmodels_store.git
   0bad116..55ba88e  master -> master





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
[master 3b14a8b] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   55ba88e..3b14a8b  master -> master





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
[master 3c5ce53] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   3b14a8b..3c5ce53  master -> master





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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7fdcab1dda20> 

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
[master f85b70e] ml_store
 1 file changed, 108 insertions(+)
To github.com:arita37/mlmodels_store.git
   3c5ce53..f85b70e  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 8.59823751e-01  1.71957132e-01 -3.48984191e-01  4.90561044e-01
  -1.15649503e+00 -1.39528303e+00  6.14726276e-01 -5.22356465e-01
  -3.69255902e-01 -9.77773002e-01]
 [ 8.57719529e-01  9.81122462e-02 -2.60466059e-01  1.06032751e+00
  -1.39003042e+00 -1.71116766e+00  2.65642403e-01  1.65712464e+00
   1.41767401e+00  4.45096710e-01]
 [ 1.13545112e+00  8.61623101e-01  4.90616924e-02 -2.08639057e+00
  -1.11469020e+00  3.61801641e-01 -8.06178212e-01  4.25920177e-01
   4.90803971e-02 -5.96086335e-01]
 [ 6.23688521e-01  1.20660790e+00  9.03999174e-01 -2.82863552e-01
  -1.18913787e+00 -2.66326884e-01  1.42361443e+00  1.06897162e+00
   4.03714310e-02  1.57546791e+00]
 [ 8.78740711e-01 -1.92316341e-02  3.19656942e-01  1.50016279e-01
  -1.46662161e+00  4.63534322e-01 -8.98683193e-01  3.97880425e-01
  -9.96010889e-01  3.18154200e-01]
 [ 1.06702918e+00 -4.29142278e-01  3.50167159e-01  1.20845633e+00
   7.51480619e-01  1.11570180e+00 -4.79157099e-01  8.40861558e-01
  -1.02887218e-01  1.71647264e-02]
 [ 3.54133613e-01  2.11124755e-01  9.21450069e-01  1.65275673e-02
   9.03945451e-01  1.77187720e-01  9.54250872e-02 -1.11647002e+00
   8.09271010e-02  6.07501958e-02]
 [ 1.66752297e+00  1.22372221e+00 -4.59930104e-01 -5.93679025e-02
  -4.93856997e-01  1.44898940e+00 -1.18110317e+00 -4.77580855e-01
   2.59999942e-02 -7.90799954e-01]
 [ 6.23629500e-01  9.86352180e-01  1.45391758e+00 -4.66154857e-01
   9.36403332e-01  1.38499134e+00  3.49435894e-02 -1.07296428e+00
   4.95158611e-01  6.61681076e-01]
 [ 1.17867274e+00 -5.99804531e-01 -6.94693595e-01  1.12341216e+00
   1.17899425e+00  3.05267040e-01  1.33526763e-02  1.38877940e+00
  -6.61344243e-01  6.21803504e-01]
 [ 7.90323893e-01  1.61336137e+00 -2.09424782e+00 -3.74804687e-01
   9.15884042e-01 -7.49969617e-01  3.10272288e-01  2.05462410e+00
   5.34095368e-02 -2.28765829e-01]
 [ 1.32857949e+00 -5.63236604e-01 -1.06179676e+00  2.39014596e+00
  -1.68450770e+00  2.45422849e-01 -5.69148654e-01  1.15259914e+00
  -2.24235772e-01  1.32247779e-01]
 [ 6.67591795e-01 -4.52524973e-01 -6.05981321e-01  1.16128569e+00
  -1.44620987e+00  1.06996554e+00  1.92381543e+00 -1.04553425e+00
   3.55284507e-01  1.80358898e+00]
 [ 1.12641981e+00 -6.29441604e-01  1.10100020e+00 -1.11343610e+00
   9.44595066e-01 -6.74100249e-02 -1.83400197e-01  1.16143998e+00
  -2.75293863e-02  7.80027135e-01]
 [ 8.88389445e-01  2.82995534e-01  1.79558917e-02  1.08030817e-01
  -8.49671873e-01  2.94176190e-02 -5.03973949e-01 -1.34793129e-01
   1.04921829e+00 -1.27046078e+00]
 [ 8.88611457e-01  8.49586845e-01 -3.09114176e-02 -1.22154015e-01
  -1.14722826e+00 -6.80851574e-01 -3.26061306e-01 -1.06787658e+00
  -7.66793627e-02  3.55717262e-01]
 [ 9.26869810e-01  3.92334911e-01 -4.23478297e-01  4.48380651e-01
  -1.09230828e+00  1.12532350e+00 -9.48439656e-01  1.04053390e-01
   5.28003422e-01  1.00796648e+00]
 [ 1.83829400e+00  5.02740882e-01  1.29101580e-01  1.55880554e+00
   1.32551412e+00  1.09402696e-01  1.40754000e+00 -1.21974440e+00
   2.44936865e+00  1.61694960e+00]
 [ 9.80427414e-01  1.93752881e+00 -2.30839743e-01  3.66332015e-01
   1.10018476e+00 -1.04458938e+00 -3.44987210e-01  2.05117344e+00
   5.85662000e-01 -2.79308500e+00]
 [ 8.95623122e-01 -2.29820588e+00 -1.95225583e-02  1.45652739e+00
  -1.85064099e+00  3.16637236e-01  1.11337266e-01 -2.66412594e+00
  -4.26428618e-01 -8.39988915e-01]
 [ 1.98519313e+00  6.74711526e-01 -1.39662042e+00  6.18539131e-01
   1.22382712e+00 -4.43171931e-01 -1.89148284e-03  1.81053491e+00
  -1.30572692e+00 -8.61316361e-01]
 [ 1.22867367e+00  1.34373116e-01 -1.82420406e-01 -2.68371304e-01
  -1.73963799e+00 -1.31675626e-01 -9.26871939e-01  1.01855247e+00
   1.23055820e+00 -4.91125138e-01]
 [ 5.58538729e-01 -5.16347909e-01 -5.18145552e-01  3.51116897e-01
   8.25506954e-01 -6.87704631e-02 -9.52062101e-01 -1.34776494e+00
   1.47073986e+00 -1.46140360e+00]
 [ 1.14377130e+00  7.27813500e-01  3.52494364e-01  5.15073614e-01
   1.17718111e+00 -2.78253447e+00 -1.94332341e+00  5.84646610e-01
   3.24274243e-01 -2.36436952e-01]
 [ 8.78643802e-01  1.03703898e+00 -4.77124206e-01  6.72619748e-01
  -1.04948638e+00  2.42887697e+00  5.24750492e-01  1.00568668e+00
   3.53567216e-01 -3.59901817e-02]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7f842fec1f28>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7f844a231710> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 8.78740711e-01 -1.92316341e-02  3.19656942e-01  1.50016279e-01
  -1.46662161e+00  4.63534322e-01 -8.98683193e-01  3.97880425e-01
  -9.96010889e-01  3.18154200e-01]
 [ 9.36211246e-01  2.04377395e-01 -1.49419377e+00  6.12232523e-01
  -9.84377246e-01  7.44884536e-01  4.94341651e-01 -3.62812886e-02
  -8.32395348e-01 -4.46699203e-01]
 [ 1.37661405e+00 -6.00225330e-01  7.25916853e-01 -3.79517516e-01
  -6.27546260e-01 -1.01480369e+00  9.66220863e-01  4.35986196e-01
  -6.87487393e-01  3.32107876e+00]
 [ 6.67591795e-01 -4.52524973e-01 -6.05981321e-01  1.16128569e+00
  -1.44620987e+00  1.06996554e+00  1.92381543e+00 -1.04553425e+00
   3.55284507e-01  1.80358898e+00]
 [ 7.90323893e-01  1.61336137e+00 -2.09424782e+00 -3.74804687e-01
   9.15884042e-01 -7.49969617e-01  3.10272288e-01  2.05462410e+00
   5.34095368e-02 -2.28765829e-01]
 [ 8.88611457e-01  8.49586845e-01 -3.09114176e-02 -1.22154015e-01
  -1.14722826e+00 -6.80851574e-01 -3.26061306e-01 -1.06787658e+00
  -7.66793627e-02  3.55717262e-01]
 [ 9.67037267e-01  3.82715174e-01 -8.06184817e-01 -2.88997343e-01
   9.08526041e-01 -3.91816240e-01  1.62091229e+00  6.84001328e-01
  -3.53409983e-01 -2.51674208e-01]
 [ 7.75285326e-01  1.47016034e+00  1.03298378e+00 -8.70008223e-01
   7.86556511e-01  3.69190470e-01 -1.43195745e-01  8.53282186e-01
  -1.39711730e-01 -2.22414029e-01]
 [ 9.26869810e-01  3.92334911e-01 -4.23478297e-01  4.48380651e-01
  -1.09230828e+00  1.12532350e+00 -9.48439656e-01  1.04053390e-01
   5.28003422e-01  1.00796648e+00]
 [ 4.41189807e-01  4.79852371e-01 -1.92003697e-01 -1.55269878e+00
  -1.88873982e+00  5.78464420e-01  3.98598388e-01 -9.61263599e-01
  -1.45832446e+00 -3.05376438e+00]
 [ 6.92114488e-01 -6.06524918e-02  2.05635552e+00 -2.41350300e+00
   1.17456965e+00 -1.77756638e+00 -2.81736269e-01 -7.77858827e-01
   1.11584111e+00  1.76024923e+00]
 [ 8.53355545e-01 -7.04350332e-01 -6.79383783e-01 -4.58666861e-02
  -1.29936179e+00 -2.18733459e-01  5.90039464e-01  1.53920701e+00
  -1.14870423e+00 -9.50909251e-01]
 [ 6.81889336e-01 -1.15498263e+00  1.22895559e+00 -1.77632196e-01
   9.98545187e-01 -1.51045638e+00 -2.75846063e-01  1.01120706e+00
  -1.47656266e+00  1.30970591e+00]
 [ 1.25704434e+00 -1.82391985e+00 -6.12406973e-01  1.16707517e+00
  -6.23732812e-01 -3.96687001e-02  8.16043684e-01  8.85825799e-01
   1.89861649e-01  3.93109245e-01]
 [ 1.01177337e+00  9.57467711e-02  7.31402517e-01  1.03345080e+00
  -1.42203164e+00 -1.46273275e-01 -1.74549518e-02 -8.57496825e-01
  -9.34181843e-01  9.54495667e-01]
 [ 1.12641981e+00 -6.29441604e-01  1.10100020e+00 -1.11343610e+00
   9.44595066e-01 -6.74100249e-02 -1.83400197e-01  1.16143998e+00
  -2.75293863e-02  7.80027135e-01]
 [ 9.47814113e-01 -1.13379204e+00  6.40985866e-01 -1.90548298e-01
  -1.23912256e+00  2.33339126e-01 -3.16901197e-01  4.34998324e-01
   9.10423603e-01  1.21987438e+00]
 [ 1.18947778e+00 -6.80678141e-01 -5.68244809e-02 -8.45080274e-02
   8.21783210e-01 -2.97361883e-01 -1.86578994e-01  4.17302005e-01
   7.84770651e-01  4.92336556e-01]
 [ 6.23629500e-01  9.86352180e-01  1.45391758e+00 -4.66154857e-01
   9.36403332e-01  1.38499134e+00  3.49435894e-02 -1.07296428e+00
   4.95158611e-01  6.61681076e-01]
 [ 1.27991386e+00 -8.71422066e-01 -3.24032329e-01 -8.64829941e-01
  -9.68539694e-01  6.08749082e-01  5.07984337e-01  5.61638097e-01
   1.51475038e+00 -1.51107661e+00]
 [ 1.17867274e+00 -5.99804531e-01 -6.94693595e-01  1.12341216e+00
   1.17899425e+00  3.05267040e-01  1.33526763e-02  1.38877940e+00
  -6.61344243e-01  6.21803504e-01]
 [ 1.01195228e+00 -1.88141087e+00  1.70018815e+00  4.97269099e-01
  -9.17664624e-01  2.37332699e-01 -1.09033833e+00 -2.14444405e+00
  -3.69562425e-01  6.08783659e-01]
 [ 1.98519313e+00  6.74711526e-01 -1.39662042e+00  6.18539131e-01
   1.22382712e+00 -4.43171931e-01 -1.89148284e-03  1.81053491e+00
  -1.30572692e+00 -8.61316361e-01]
 [ 6.10942600e-01 -2.79099641e+00 -1.33520272e+00 -4.56117555e-01
  -9.44959948e-01 -9.79890252e-01 -1.56993672e-01  6.92574348e-01
  -4.78672356e-01 -1.06460122e-01]
 [ 1.06040861e+00  5.10307597e-01  5.01725109e-01 -9.15791849e-01
  -9.07318361e-01 -4.07252043e-01 -1.79612295e-01  9.84951672e-01
   1.07125243e+00 -5.93343754e-01]]
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
[[ 1.06702918 -0.42914228  0.35016716  1.20845633  0.75148062  1.1157018
  -0.4791571   0.84086156 -0.10288722  0.01716473]
 [ 1.27991386 -0.87142207 -0.32403233 -0.86482994 -0.96853969  0.60874908
   0.50798434  0.5616381   1.51475038 -1.51107661]
 [ 0.68188934 -1.15498263  1.22895559 -0.1776322   0.99854519 -1.51045638
  -0.27584606  1.01120706 -1.47656266  1.30970591]
 [ 0.35413361  0.21112476  0.92145007  0.01652757  0.90394545  0.17718772
   0.09542509 -1.11647002  0.0809271   0.0607502 ]
 [ 0.62153099 -1.50957268 -0.10193204 -1.08071069 -1.13742855  0.725474
   0.7980638  -0.03917826 -0.22875417  0.74335654]
 [ 1.39198128 -0.19022103 -0.53722302 -0.44873803  0.70455707 -0.67244804
  -0.70134443 -0.55749472  0.93916874  0.15626385]
 [ 1.46893146 -1.47115693  0.58591043 -0.8301719   1.03345052 -0.8805776
  -0.95542526 -0.27909772  1.62284909  2.06578332]
 [ 1.12062155 -0.7029204  -1.22957425  0.72555052 -1.18013412 -0.32420422
   1.10223673  0.81434313  0.78046993  1.10861676]
 [ 0.77370361  1.27852808 -2.11416392 -0.44222928  1.06821044  0.32352735
  -2.50644065 -0.10999149  0.00854895 -0.41163916]
 [ 0.88838944  0.28299553  0.01795589  0.10803082 -0.84967187  0.02941762
  -0.50397395 -0.13479313  1.04921829 -1.27046078]
 [ 0.87699465  1.23225307 -0.86778722 -0.25417987  0.89189141  1.39984394
  -0.87728152 -0.78191168 -0.43750898 -1.44087602]
 [ 0.85982375  0.17195713 -0.34898419  0.49056104 -1.15649503 -1.39528303
   0.61472628 -0.52235647 -0.3692559  -0.977773  ]
 [ 1.01177337  0.09574677  0.73140252  1.0334508  -1.42203164 -0.14627327
  -0.01745495 -0.85749682 -0.93418184  0.95449567]
 [ 0.69174373  1.00978733 -1.21333813 -1.55694156 -1.20257258 -0.61244213
  -2.69836174 -0.13935181 -0.72853749  0.0722519 ]
 [ 0.47330777 -0.97326759 -0.22814069  0.17516773 -1.01366961 -0.05348369
   0.39378773 -0.18306199 -0.2210289   0.58033011]
 [ 1.18468624 -1.00016919 -0.59384307  1.04499441  0.96548233  0.6085147
  -0.625342   -0.0693287  -0.10839207 -0.34390071]
 [ 1.24549398 -0.72239191  1.1181334   1.09899633  1.00277655 -0.90163449
  -0.53223402 -0.82246719  0.72171129  0.6743961 ]
 [ 0.88861146  0.84958685 -0.03091142 -0.12215402 -1.14722826 -0.68085157
  -0.32606131 -1.06787658 -0.07667936  0.35571726]
 [ 0.99785516 -0.6001388   0.45794708  0.14676526 -0.93355729  0.57180488
   0.57296273 -0.03681766  0.11236849 -0.01781755]
 [ 1.16755486  0.0353601   0.7147896  -1.53879325  1.10863359 -0.44789518
  -1.75592564  0.61798553 -0.18417633  0.85270406]
 [ 1.36586461  3.9586027   0.54812958  0.64864364  0.84917607  0.10734329
   1.38631426 -1.39881282  0.08176782 -1.63744959]
 [ 1.13545112  0.8616231   0.04906169 -2.08639057 -1.1146902   0.36180164
  -0.80617821  0.42592018  0.0490804  -0.59608633]
 [ 1.32720112 -0.16119832  0.6024509  -0.28638492 -0.5789623  -0.87088765
   1.37975819  0.50142959 -0.47861407 -0.89264667]
 [ 0.6675918  -0.45252497 -0.60598132  1.16128569 -1.44620987  1.06996554
   1.92381543 -1.04553425  0.35528451  1.80358898]
 [ 0.8786438   1.03703898 -0.47712421  0.67261975 -1.04948638  2.42887697
   0.52475049  1.00568668  0.35356722 -0.03599018]]
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
[master 0899444] ml_store
 1 file changed, 296 insertions(+)
To github.com:arita37/mlmodels_store.git
   f85b70e..0899444  master -> master





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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140241689006032
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140241689005808
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140241689004576
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140241689004128
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140241689003624
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140241689003288

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
grad_step = 000000, loss = 1.901460
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 1.617026
grad_step = 000002, loss = 1.440428
grad_step = 000003, loss = 1.227016
grad_step = 000004, loss = 0.971975
grad_step = 000005, loss = 0.691603
grad_step = 000006, loss = 0.427314
grad_step = 000007, loss = 0.265549
grad_step = 000008, loss = 0.245048
grad_step = 000009, loss = 0.262770
grad_step = 000010, loss = 0.220666
grad_step = 000011, loss = 0.131125
grad_step = 000012, loss = 0.063608
grad_step = 000013, loss = 0.034498
grad_step = 000014, loss = 0.040603
grad_step = 000015, loss = 0.061691
grad_step = 000016, loss = 0.076793
grad_step = 000017, loss = 0.076848
grad_step = 000018, loss = 0.064176
grad_step = 000019, loss = 0.044828
grad_step = 000020, loss = 0.027154
grad_step = 000021, loss = 0.018252
grad_step = 000022, loss = 0.019811
grad_step = 000023, loss = 0.025723
grad_step = 000024, loss = 0.027339
grad_step = 000025, loss = 0.022021
grad_step = 000026, loss = 0.014174
grad_step = 000027, loss = 0.009454
grad_step = 000028, loss = 0.009832
grad_step = 000029, loss = 0.013339
grad_step = 000030, loss = 0.016567
grad_step = 000031, loss = 0.017120
grad_step = 000032, loss = 0.014557
grad_step = 000033, loss = 0.010251
grad_step = 000034, loss = 0.006463
grad_step = 000035, loss = 0.005041
grad_step = 000036, loss = 0.006183
grad_step = 000037, loss = 0.008216
grad_step = 000038, loss = 0.009050
grad_step = 000039, loss = 0.008068
grad_step = 000040, loss = 0.006458
grad_step = 000041, loss = 0.005719
grad_step = 000042, loss = 0.006243
grad_step = 000043, loss = 0.007207
grad_step = 000044, loss = 0.007573
grad_step = 000045, loss = 0.006899
grad_step = 000046, loss = 0.005578
grad_step = 000047, loss = 0.004414
grad_step = 000048, loss = 0.004044
grad_step = 000049, loss = 0.004452
grad_step = 000050, loss = 0.005023
grad_step = 000051, loss = 0.005135
grad_step = 000052, loss = 0.004709
grad_step = 000053, loss = 0.004168
grad_step = 000054, loss = 0.003943
grad_step = 000055, loss = 0.004081
grad_step = 000056, loss = 0.004289
grad_step = 000057, loss = 0.004259
grad_step = 000058, loss = 0.003947
grad_step = 000059, loss = 0.003575
grad_step = 000060, loss = 0.003411
grad_step = 000061, loss = 0.003524
grad_step = 000062, loss = 0.003732
grad_step = 000063, loss = 0.003801
grad_step = 000064, loss = 0.003671
grad_step = 000065, loss = 0.003484
grad_step = 000066, loss = 0.003401
grad_step = 000067, loss = 0.003438
grad_step = 000068, loss = 0.003486
grad_step = 000069, loss = 0.003442
grad_step = 000070, loss = 0.003314
grad_step = 000071, loss = 0.003198
grad_step = 000072, loss = 0.003168
grad_step = 000073, loss = 0.003206
grad_step = 000074, loss = 0.003231
grad_step = 000075, loss = 0.003189
grad_step = 000076, loss = 0.003107
grad_step = 000077, loss = 0.003046
grad_step = 000078, loss = 0.003033
grad_step = 000079, loss = 0.003039
grad_step = 000080, loss = 0.003024
grad_step = 000081, loss = 0.002980
grad_step = 000082, loss = 0.002935
grad_step = 000083, loss = 0.002918
grad_step = 000084, loss = 0.002921
grad_step = 000085, loss = 0.002918
grad_step = 000086, loss = 0.002890
grad_step = 000087, loss = 0.002850
grad_step = 000088, loss = 0.002821
grad_step = 000089, loss = 0.002809
grad_step = 000090, loss = 0.002800
grad_step = 000091, loss = 0.002780
grad_step = 000092, loss = 0.002752
grad_step = 000093, loss = 0.002729
grad_step = 000094, loss = 0.002717
grad_step = 000095, loss = 0.002708
grad_step = 000096, loss = 0.002690
grad_step = 000097, loss = 0.002665
grad_step = 000098, loss = 0.002643
grad_step = 000099, loss = 0.002630
grad_step = 000100, loss = 0.002619
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002604
grad_step = 000102, loss = 0.002585
grad_step = 000103, loss = 0.002569
grad_step = 000104, loss = 0.002556
grad_step = 000105, loss = 0.002545
grad_step = 000106, loss = 0.002529
grad_step = 000107, loss = 0.002511
grad_step = 000108, loss = 0.002496
grad_step = 000109, loss = 0.002484
grad_step = 000110, loss = 0.002471
grad_step = 000111, loss = 0.002456
grad_step = 000112, loss = 0.002441
grad_step = 000113, loss = 0.002428
grad_step = 000114, loss = 0.002417
grad_step = 000115, loss = 0.002404
grad_step = 000116, loss = 0.002390
grad_step = 000117, loss = 0.002376
grad_step = 000118, loss = 0.002364
grad_step = 000119, loss = 0.002352
grad_step = 000120, loss = 0.002339
grad_step = 000121, loss = 0.002325
grad_step = 000122, loss = 0.002311
grad_step = 000123, loss = 0.002298
grad_step = 000124, loss = 0.002283
grad_step = 000125, loss = 0.002268
grad_step = 000126, loss = 0.002252
grad_step = 000127, loss = 0.002236
grad_step = 000128, loss = 0.002218
grad_step = 000129, loss = 0.002203
grad_step = 000130, loss = 0.002185
grad_step = 000131, loss = 0.002164
grad_step = 000132, loss = 0.002146
grad_step = 000133, loss = 0.002126
grad_step = 000134, loss = 0.002105
grad_step = 000135, loss = 0.002085
grad_step = 000136, loss = 0.002064
grad_step = 000137, loss = 0.002042
grad_step = 000138, loss = 0.002020
grad_step = 000139, loss = 0.001999
grad_step = 000140, loss = 0.001977
grad_step = 000141, loss = 0.001955
grad_step = 000142, loss = 0.001933
grad_step = 000143, loss = 0.001912
grad_step = 000144, loss = 0.001891
grad_step = 000145, loss = 0.001871
grad_step = 000146, loss = 0.001853
grad_step = 000147, loss = 0.001834
grad_step = 000148, loss = 0.001817
grad_step = 000149, loss = 0.001801
grad_step = 000150, loss = 0.001786
grad_step = 000151, loss = 0.001772
grad_step = 000152, loss = 0.001758
grad_step = 000153, loss = 0.001746
grad_step = 000154, loss = 0.001737
grad_step = 000155, loss = 0.001727
grad_step = 000156, loss = 0.001719
grad_step = 000157, loss = 0.001712
grad_step = 000158, loss = 0.001704
grad_step = 000159, loss = 0.001698
grad_step = 000160, loss = 0.001693
grad_step = 000161, loss = 0.001688
grad_step = 000162, loss = 0.001682
grad_step = 000163, loss = 0.001677
grad_step = 000164, loss = 0.001672
grad_step = 000165, loss = 0.001667
grad_step = 000166, loss = 0.001663
grad_step = 000167, loss = 0.001658
grad_step = 000168, loss = 0.001652
grad_step = 000169, loss = 0.001647
grad_step = 000170, loss = 0.001642
grad_step = 000171, loss = 0.001637
grad_step = 000172, loss = 0.001632
grad_step = 000173, loss = 0.001627
grad_step = 000174, loss = 0.001622
grad_step = 000175, loss = 0.001617
grad_step = 000176, loss = 0.001613
grad_step = 000177, loss = 0.001608
grad_step = 000178, loss = 0.001604
grad_step = 000179, loss = 0.001600
grad_step = 000180, loss = 0.001596
grad_step = 000181, loss = 0.001591
grad_step = 000182, loss = 0.001587
grad_step = 000183, loss = 0.001584
grad_step = 000184, loss = 0.001580
grad_step = 000185, loss = 0.001576
grad_step = 000186, loss = 0.001572
grad_step = 000187, loss = 0.001568
grad_step = 000188, loss = 0.001564
grad_step = 000189, loss = 0.001560
grad_step = 000190, loss = 0.001556
grad_step = 000191, loss = 0.001552
grad_step = 000192, loss = 0.001548
grad_step = 000193, loss = 0.001544
grad_step = 000194, loss = 0.001540
grad_step = 000195, loss = 0.001537
grad_step = 000196, loss = 0.001535
grad_step = 000197, loss = 0.001531
grad_step = 000198, loss = 0.001524
grad_step = 000199, loss = 0.001520
grad_step = 000200, loss = 0.001518
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001514
grad_step = 000202, loss = 0.001508
grad_step = 000203, loss = 0.001505
grad_step = 000204, loss = 0.001502
grad_step = 000205, loss = 0.001497
grad_step = 000206, loss = 0.001493
grad_step = 000207, loss = 0.001490
grad_step = 000208, loss = 0.001486
grad_step = 000209, loss = 0.001480
grad_step = 000210, loss = 0.001477
grad_step = 000211, loss = 0.001473
grad_step = 000212, loss = 0.001469
grad_step = 000213, loss = 0.001464
grad_step = 000214, loss = 0.001459
grad_step = 000215, loss = 0.001455
grad_step = 000216, loss = 0.001451
grad_step = 000217, loss = 0.001446
grad_step = 000218, loss = 0.001441
grad_step = 000219, loss = 0.001437
grad_step = 000220, loss = 0.001432
grad_step = 000221, loss = 0.001427
grad_step = 000222, loss = 0.001422
grad_step = 000223, loss = 0.001417
grad_step = 000224, loss = 0.001412
grad_step = 000225, loss = 0.001407
grad_step = 000226, loss = 0.001402
grad_step = 000227, loss = 0.001396
grad_step = 000228, loss = 0.001391
grad_step = 000229, loss = 0.001386
grad_step = 000230, loss = 0.001381
grad_step = 000231, loss = 0.001375
grad_step = 000232, loss = 0.001370
grad_step = 000233, loss = 0.001365
grad_step = 000234, loss = 0.001360
grad_step = 000235, loss = 0.001354
grad_step = 000236, loss = 0.001350
grad_step = 000237, loss = 0.001346
grad_step = 000238, loss = 0.001346
grad_step = 000239, loss = 0.001344
grad_step = 000240, loss = 0.001335
grad_step = 000241, loss = 0.001324
grad_step = 000242, loss = 0.001323
grad_step = 000243, loss = 0.001321
grad_step = 000244, loss = 0.001311
grad_step = 000245, loss = 0.001305
grad_step = 000246, loss = 0.001304
grad_step = 000247, loss = 0.001298
grad_step = 000248, loss = 0.001290
grad_step = 000249, loss = 0.001287
grad_step = 000250, loss = 0.001284
grad_step = 000251, loss = 0.001276
grad_step = 000252, loss = 0.001271
grad_step = 000253, loss = 0.001268
grad_step = 000254, loss = 0.001263
grad_step = 000255, loss = 0.001256
grad_step = 000256, loss = 0.001252
grad_step = 000257, loss = 0.001248
grad_step = 000258, loss = 0.001243
grad_step = 000259, loss = 0.001237
grad_step = 000260, loss = 0.001232
grad_step = 000261, loss = 0.001228
grad_step = 000262, loss = 0.001223
grad_step = 000263, loss = 0.001217
grad_step = 000264, loss = 0.001212
grad_step = 000265, loss = 0.001208
grad_step = 000266, loss = 0.001203
grad_step = 000267, loss = 0.001198
grad_step = 000268, loss = 0.001193
grad_step = 000269, loss = 0.001188
grad_step = 000270, loss = 0.001183
grad_step = 000271, loss = 0.001178
grad_step = 000272, loss = 0.001173
grad_step = 000273, loss = 0.001168
grad_step = 000274, loss = 0.001163
grad_step = 000275, loss = 0.001159
grad_step = 000276, loss = 0.001154
grad_step = 000277, loss = 0.001150
grad_step = 000278, loss = 0.001145
grad_step = 000279, loss = 0.001141
grad_step = 000280, loss = 0.001135
grad_step = 000281, loss = 0.001130
grad_step = 000282, loss = 0.001124
grad_step = 000283, loss = 0.001118
grad_step = 000284, loss = 0.001113
grad_step = 000285, loss = 0.001109
grad_step = 000286, loss = 0.001106
grad_step = 000287, loss = 0.001104
grad_step = 000288, loss = 0.001100
grad_step = 000289, loss = 0.001098
grad_step = 000290, loss = 0.001095
grad_step = 000291, loss = 0.001089
grad_step = 000292, loss = 0.001080
grad_step = 000293, loss = 0.001071
grad_step = 000294, loss = 0.001065
grad_step = 000295, loss = 0.001063
grad_step = 000296, loss = 0.001062
grad_step = 000297, loss = 0.001059
grad_step = 000298, loss = 0.001051
grad_step = 000299, loss = 0.001043
grad_step = 000300, loss = 0.001036
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001032
grad_step = 000302, loss = 0.001030
grad_step = 000303, loss = 0.001028
grad_step = 000304, loss = 0.001025
grad_step = 000305, loss = 0.001019
grad_step = 000306, loss = 0.001011
grad_step = 000307, loss = 0.001004
grad_step = 000308, loss = 0.000999
grad_step = 000309, loss = 0.000995
grad_step = 000310, loss = 0.000993
grad_step = 000311, loss = 0.000990
grad_step = 000312, loss = 0.000988
grad_step = 000313, loss = 0.000985
grad_step = 000314, loss = 0.000982
grad_step = 000315, loss = 0.000978
grad_step = 000316, loss = 0.000972
grad_step = 000317, loss = 0.000965
grad_step = 000318, loss = 0.000957
grad_step = 000319, loss = 0.000950
grad_step = 000320, loss = 0.000944
grad_step = 000321, loss = 0.000940
grad_step = 000322, loss = 0.000937
grad_step = 000323, loss = 0.000936
grad_step = 000324, loss = 0.000938
grad_step = 000325, loss = 0.000943
grad_step = 000326, loss = 0.000950
grad_step = 000327, loss = 0.000947
grad_step = 000328, loss = 0.000933
grad_step = 000329, loss = 0.000911
grad_step = 000330, loss = 0.000900
grad_step = 000331, loss = 0.000902
grad_step = 000332, loss = 0.000908
grad_step = 000333, loss = 0.000909
grad_step = 000334, loss = 0.000899
grad_step = 000335, loss = 0.000883
grad_step = 000336, loss = 0.000875
grad_step = 000337, loss = 0.000875
grad_step = 000338, loss = 0.000877
grad_step = 000339, loss = 0.000875
grad_step = 000340, loss = 0.000868
grad_step = 000341, loss = 0.000858
grad_step = 000342, loss = 0.000850
grad_step = 000343, loss = 0.000846
grad_step = 000344, loss = 0.000845
grad_step = 000345, loss = 0.000845
grad_step = 000346, loss = 0.000844
grad_step = 000347, loss = 0.000840
grad_step = 000348, loss = 0.000836
grad_step = 000349, loss = 0.000829
grad_step = 000350, loss = 0.000823
grad_step = 000351, loss = 0.000816
grad_step = 000352, loss = 0.000810
grad_step = 000353, loss = 0.000805
grad_step = 000354, loss = 0.000801
grad_step = 000355, loss = 0.000798
grad_step = 000356, loss = 0.000795
grad_step = 000357, loss = 0.000796
grad_step = 000358, loss = 0.000803
grad_step = 000359, loss = 0.000821
grad_step = 000360, loss = 0.000851
grad_step = 000361, loss = 0.000865
grad_step = 000362, loss = 0.000839
grad_step = 000363, loss = 0.000780
grad_step = 000364, loss = 0.000756
grad_step = 000365, loss = 0.000782
grad_step = 000366, loss = 0.000803
grad_step = 000367, loss = 0.000783
grad_step = 000368, loss = 0.000744
grad_step = 000369, loss = 0.000738
grad_step = 000370, loss = 0.000759
grad_step = 000371, loss = 0.000763
grad_step = 000372, loss = 0.000740
grad_step = 000373, loss = 0.000719
grad_step = 000374, loss = 0.000722
grad_step = 000375, loss = 0.000733
grad_step = 000376, loss = 0.000731
grad_step = 000377, loss = 0.000713
grad_step = 000378, loss = 0.000700
grad_step = 000379, loss = 0.000702
grad_step = 000380, loss = 0.000708
grad_step = 000381, loss = 0.000706
grad_step = 000382, loss = 0.000696
grad_step = 000383, loss = 0.000684
grad_step = 000384, loss = 0.000679
grad_step = 000385, loss = 0.000681
grad_step = 000386, loss = 0.000684
grad_step = 000387, loss = 0.000682
grad_step = 000388, loss = 0.000677
grad_step = 000389, loss = 0.000669
grad_step = 000390, loss = 0.000661
grad_step = 000391, loss = 0.000656
grad_step = 000392, loss = 0.000654
grad_step = 000393, loss = 0.000654
grad_step = 000394, loss = 0.000655
grad_step = 000395, loss = 0.000657
grad_step = 000396, loss = 0.000663
grad_step = 000397, loss = 0.000673
grad_step = 000398, loss = 0.000689
grad_step = 000399, loss = 0.000704
grad_step = 000400, loss = 0.000711
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000692
grad_step = 000402, loss = 0.000657
grad_step = 000403, loss = 0.000629
grad_step = 000404, loss = 0.000626
grad_step = 000405, loss = 0.000641
grad_step = 000406, loss = 0.000655
grad_step = 000407, loss = 0.000652
grad_step = 000408, loss = 0.000633
grad_step = 000409, loss = 0.000616
grad_step = 000410, loss = 0.000611
grad_step = 000411, loss = 0.000617
grad_step = 000412, loss = 0.000626
grad_step = 000413, loss = 0.000628
grad_step = 000414, loss = 0.000623
grad_step = 000415, loss = 0.000612
grad_step = 000416, loss = 0.000602
grad_step = 000417, loss = 0.000596
grad_step = 000418, loss = 0.000595
grad_step = 000419, loss = 0.000598
grad_step = 000420, loss = 0.000601
grad_step = 000421, loss = 0.000604
grad_step = 000422, loss = 0.000607
grad_step = 000423, loss = 0.000612
grad_step = 000424, loss = 0.000614
grad_step = 000425, loss = 0.000617
grad_step = 000426, loss = 0.000614
grad_step = 000427, loss = 0.000609
grad_step = 000428, loss = 0.000598
grad_step = 000429, loss = 0.000587
grad_step = 000430, loss = 0.000578
grad_step = 000431, loss = 0.000572
grad_step = 000432, loss = 0.000570
grad_step = 000433, loss = 0.000570
grad_step = 000434, loss = 0.000574
grad_step = 000435, loss = 0.000579
grad_step = 000436, loss = 0.000587
grad_step = 000437, loss = 0.000599
grad_step = 000438, loss = 0.000614
grad_step = 000439, loss = 0.000628
grad_step = 000440, loss = 0.000638
grad_step = 000441, loss = 0.000629
grad_step = 000442, loss = 0.000606
grad_step = 000443, loss = 0.000574
grad_step = 000444, loss = 0.000555
grad_step = 000445, loss = 0.000554
grad_step = 000446, loss = 0.000567
grad_step = 000447, loss = 0.000582
grad_step = 000448, loss = 0.000588
grad_step = 000449, loss = 0.000582
grad_step = 000450, loss = 0.000566
grad_step = 000451, loss = 0.000550
grad_step = 000452, loss = 0.000541
grad_step = 000453, loss = 0.000542
grad_step = 000454, loss = 0.000549
grad_step = 000455, loss = 0.000556
grad_step = 000456, loss = 0.000558
grad_step = 000457, loss = 0.000555
grad_step = 000458, loss = 0.000548
grad_step = 000459, loss = 0.000540
grad_step = 000460, loss = 0.000533
grad_step = 000461, loss = 0.000529
grad_step = 000462, loss = 0.000529
grad_step = 000463, loss = 0.000532
grad_step = 000464, loss = 0.000535
grad_step = 000465, loss = 0.000538
grad_step = 000466, loss = 0.000541
grad_step = 000467, loss = 0.000545
grad_step = 000468, loss = 0.000548
grad_step = 000469, loss = 0.000552
grad_step = 000470, loss = 0.000552
grad_step = 000471, loss = 0.000552
grad_step = 000472, loss = 0.000547
grad_step = 000473, loss = 0.000541
grad_step = 000474, loss = 0.000533
grad_step = 000475, loss = 0.000525
grad_step = 000476, loss = 0.000517
grad_step = 000477, loss = 0.000512
grad_step = 000478, loss = 0.000510
grad_step = 000479, loss = 0.000511
grad_step = 000480, loss = 0.000515
grad_step = 000481, loss = 0.000519
grad_step = 000482, loss = 0.000525
grad_step = 000483, loss = 0.000532
grad_step = 000484, loss = 0.000545
grad_step = 000485, loss = 0.000561
grad_step = 000486, loss = 0.000580
grad_step = 000487, loss = 0.000591
grad_step = 000488, loss = 0.000587
grad_step = 000489, loss = 0.000562
grad_step = 000490, loss = 0.000528
grad_step = 000491, loss = 0.000502
grad_step = 000492, loss = 0.000501
grad_step = 000493, loss = 0.000517
grad_step = 000494, loss = 0.000533
grad_step = 000495, loss = 0.000536
grad_step = 000496, loss = 0.000523
grad_step = 000497, loss = 0.000506
grad_step = 000498, loss = 0.000493
grad_step = 000499, loss = 0.000491
grad_step = 000500, loss = 0.000499
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000509
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
[[0.8450835  0.8846826  0.9232012  0.946696   0.9856477 ]
 [0.8330442  0.92801124 0.9382948  1.0056189  0.97724974]
 [0.8965895  0.9266937  0.97182643 1.0002158  0.9526104 ]
 [0.92425936 0.98837733 0.98630106 0.95899147 0.9290691 ]
 [0.96481895 0.9791519  0.9861304  0.91055954 0.8829238 ]
 [0.9630811  0.9451434  0.94773173 0.85943884 0.85084224]
 [0.8860369  0.8914902  0.8748689  0.8487371  0.8089759 ]
 [0.88571024 0.83895767 0.85500443 0.81092125 0.8177336 ]
 [0.805133   0.83430076 0.82209915 0.8427628  0.85282946]
 [0.81758523 0.7970342  0.8307145  0.8425478  0.86350447]
 [0.7986227  0.81160396 0.83775556 0.84370524 0.9252443 ]
 [0.79503596 0.8316608  0.8630274  0.9069369  0.9513569 ]
 [0.8376389  0.8803457  0.9174037  0.94581074 0.97957873]
 [0.8314368  0.9326212  0.9382366  1.0097868  0.96809727]
 [0.910572   0.9365185  0.9771868  0.989138   0.9383346 ]
 [0.9366195  0.99199784 0.9852653  0.9386767  0.91008466]
 [0.96856236 0.9740516  0.9773812  0.88926125 0.8620331 ]
 [0.9514029  0.9268301  0.9263838  0.83686185 0.8317441 ]
 [0.8838695  0.8885104  0.86658204 0.83417636 0.80219406]
 [0.893162   0.8441996  0.8583816  0.8104683  0.8167827 ]
 [0.8182064  0.8452144  0.8310654  0.84391403 0.8550848 ]
 [0.83056366 0.8124254  0.8415849  0.84969    0.8667989 ]
 [0.80927306 0.81820154 0.8445584  0.8495328  0.92818123]
 [0.80663025 0.8382983  0.8720106  0.9130818  0.95876527]
 [0.8496739  0.88966054 0.9287075  0.94896734 0.9942826 ]
 [0.83849007 0.9351661  0.94583535 1.012267   0.9874491 ]
 [0.9024495  0.9365569  0.98161113 1.0122318  0.96310127]
 [0.9385836  1.0029873  1.0021479  0.9708425  0.9393188 ]
 [0.9771955  0.9930601  1.0005733  0.92123747 0.8918406 ]
 [0.97492766 0.9557136  0.9605253  0.8662418  0.8594784 ]
 [0.89502203 0.89940524 0.883438   0.8518905  0.81627077]]

  #### Plot     ############################################### 
Saved image to ztest/model_tch/nbeats//n_beats_test.png.

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   0899444..18aa1b1  master     -> origin/master
Updating 0899444..18aa1b1
Fast-forward
 ...-14_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py | 2497 ++++++++++++++++++++
 1 file changed, 2497 insertions(+)
 create mode 100644 log_benchmark/log_benchmark_2020-05-14-20-14_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py
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
[master 938f3ff] ml_store
 1 file changed, 1128 insertions(+)
To github.com:arita37/mlmodels_store.git
   18aa1b1..938f3ff  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_classifier.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_classifier.py", line 522, in <module>
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_classifier.py", line 418, in get_params
    cf = json.load(open(data_path, mode='r'))
FileNotFoundError: [Errno 2] No such file or directory: 'model_tch/transformer_classifier.json'

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
[master 7587eb8] ml_store
 1 file changed, 37 insertions(+)
To github.com:arita37/mlmodels_store.git
   938f3ff..7587eb8  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//matchzoo_models.py 

  #### Loading params   ############################################## 

  {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'dataset_pars': {'data_pack': '', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'resample': True, 'sort': False, 'callbacks': 'PADDING'}, 'dataloader': '', 'dataloader_pars': {'device': 'cpu', 'dataset': 'None', 'stage': 'train', 'callback': 'PADDING'}, 'preprocess': {'train': {'transform': True, 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'stage': 'train', 'resample': True, 'sort': False, 'dataloader_callback': 'PADDING'}, 'test': {'transform': True, 'batch_size': 20, 'stage': 'dev', 'dataloader_callback': 'PADDING'}}} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Loading dataset   ############################################# 

  #### Model init   ################################################## 
  0%|          | 0/231508 [00:00<?, ?B/s]  8%|▊         | 17408/231508 [00:00<00:01, 118836.29B/s] 38%|███▊      | 87040/231508 [00:00<00:00, 152961.21B/s]100%|██████████| 231508/231508 [00:00<00:00, 616364.79B/s]
  0%|          | 0/433 [00:00<?, ?B/s]100%|██████████| 433/433 [00:00<00:00, 259188.47B/s]
  0%|          | 0/440473133 [00:00<?, ?B/s]  0%|          | 52224/440473133 [00:00<18:52, 389054.86B/s]  0%|          | 174080/440473133 [00:00<15:39, 468616.21B/s]  0%|          | 382976/440473133 [00:00<12:24, 591330.33B/s]  0%|          | 504832/440473133 [00:00<11:07, 659103.34B/s]  0%|          | 696320/440473133 [00:00<09:21, 783805.75B/s]  0%|          | 905216/440473133 [00:00<07:58, 918125.04B/s]  0%|          | 1113088/440473133 [00:00<07:00, 1043866.35B/s]  0%|          | 1323008/440473133 [00:01<06:20, 1155146.42B/s]  0%|          | 1531904/440473133 [00:01<05:51, 1248948.25B/s]  0%|          | 1758208/440473133 [00:01<05:24, 1351453.86B/s]  0%|          | 1984512/440473133 [00:01<05:05, 1433742.43B/s]  1%|          | 2228224/440473133 [00:01<04:48, 1521148.30B/s]  1%|          | 2454528/440473133 [00:01<04:39, 1568210.30B/s]  1%|          | 2698240/440473133 [00:01<04:29, 1626619.62B/s]  1%|          | 2941952/440473133 [00:02<04:21, 1672924.45B/s]  1%|          | 3185664/440473133 [00:02<04:16, 1707626.97B/s]  1%|          | 3446784/440473133 [00:02<04:07, 1763574.73B/s]  1%|          | 3690496/440473133 [00:02<04:06, 1775149.06B/s]  1%|          | 3951616/440473133 [00:02<04:00, 1818523.87B/s]  1%|          | 4212736/440473133 [00:02<03:56, 1845941.80B/s]  1%|          | 4456448/440473133 [00:02<03:57, 1832826.50B/s]  1%|          | 4717568/440473133 [00:02<03:54, 1859735.93B/s]  1%|          | 4978688/440473133 [00:03<03:51, 1877853.88B/s]  1%|          | 5239808/440473133 [00:03<03:50, 1891946.30B/s]  1%|          | 5500928/440473133 [00:03<03:50, 1890295.23B/s]  1%|▏         | 5762048/440473133 [00:03<03:50, 1888681.84B/s]  1%|▏         | 6023168/440473133 [00:03<03:40, 1968871.12B/s]  1%|▏         | 6284288/440473133 [00:03<03:43, 1940082.33B/s]  1%|▏         | 6562816/440473133 [00:03<03:49, 1893122.16B/s]  2%|▏         | 6823936/440473133 [00:04<03:50, 1878017.52B/s]  2%|▏         | 7085056/440473133 [00:04<03:53, 1857765.75B/s]  2%|▏         | 7346176/440473133 [00:04<03:53, 1852373.27B/s]  2%|▏         | 7607296/440473133 [00:04<03:56, 1832382.29B/s]  2%|▏         | 7868416/440473133 [00:04<03:57, 1820346.43B/s]  2%|▏         | 8129536/440473133 [00:04<03:58, 1809008.83B/s]  2%|▏         | 8390656/440473133 [00:04<03:54, 1842760.36B/s]  2%|▏         | 8669184/440473133 [00:05<03:59, 1806448.42B/s]  2%|▏         | 8930304/440473133 [00:05<03:56, 1828129.48B/s]  2%|▏         | 9208832/440473133 [00:05<03:48, 1886479.85B/s]  2%|▏         | 9487360/440473133 [00:05<03:44, 1918705.97B/s]  2%|▏         | 9765888/440473133 [00:05<03:40, 1956708.39B/s]  2%|▏         | 9974784/440473133 [00:05<03:36, 1987325.46B/s]  2%|▏         | 10183680/440473133 [00:05<03:35, 1999871.78B/s]  2%|▏         | 10409984/440473133 [00:05<03:29, 2049350.72B/s]  2%|▏         | 10628096/440473133 [00:06<03:25, 2087148.05B/s]  2%|▏         | 10845184/440473133 [00:06<03:26, 2083477.45B/s]  3%|▎         | 11071488/440473133 [00:06<03:30, 2035841.11B/s]  3%|▎         | 11350016/440473133 [00:06<03:21, 2125239.68B/s]  3%|▎         | 11576320/440473133 [00:06<03:21, 2132996.86B/s]  3%|▎         | 11820032/440473133 [00:06<03:22, 2119802.36B/s]  3%|▎         | 12098560/440473133 [00:06<03:15, 2189094.76B/s]  3%|▎         | 12342272/440473133 [00:06<03:12, 2222430.38B/s]  3%|▎         | 12585984/440473133 [00:06<03:17, 2170034.96B/s]  3%|▎         | 12864512/440473133 [00:07<03:11, 2234363.97B/s]  3%|▎         | 13108224/440473133 [00:07<03:09, 2259382.64B/s]  3%|▎         | 13351936/440473133 [00:07<03:06, 2288127.45B/s]  3%|▎         | 13595648/440473133 [00:07<03:04, 2319750.08B/s]  3%|▎         | 13829120/440473133 [00:07<03:03, 2321672.41B/s]  3%|▎         | 14065664/440473133 [00:07<03:03, 2327860.17B/s]  3%|▎         | 14299136/440473133 [00:07<03:03, 2327245.08B/s]  3%|▎         | 14553088/440473133 [00:07<03:00, 2362888.98B/s]  3%|▎         | 14790656/440473133 [00:07<02:59, 2365089.57B/s]  3%|▎         | 15040512/440473133 [00:08<02:58, 2381460.45B/s]  3%|▎         | 15279104/440473133 [00:08<02:59, 2374298.55B/s]  4%|▎         | 15545344/440473133 [00:08<02:54, 2438716.79B/s]  4%|▎         | 15790080/440473133 [00:08<02:55, 2420281.52B/s]  4%|▎         | 16050176/440473133 [00:08<02:52, 2464429.92B/s]  4%|▎         | 16297984/440473133 [00:08<02:56, 2409194.94B/s]  4%|▍         | 16579584/440473133 [00:08<02:48, 2518296.50B/s]  4%|▍         | 16833536/440473133 [00:08<02:59, 2362525.01B/s]  4%|▍         | 17146880/440473133 [00:08<02:50, 2478766.09B/s]  4%|▍         | 17430528/440473133 [00:08<02:44, 2576055.85B/s]  4%|▍         | 17692672/440473133 [00:09<02:50, 2484458.99B/s]  4%|▍         | 18034688/440473133 [00:09<02:45, 2553684.77B/s]  4%|▍         | 18358272/440473133 [00:09<02:34, 2726065.78B/s]  4%|▍         | 18636800/440473133 [00:09<02:41, 2619436.01B/s]  4%|▍         | 19009536/440473133 [00:09<02:35, 2716401.48B/s]  4%|▍         | 19375104/440473133 [00:09<02:24, 2913001.95B/s]  4%|▍         | 19674112/440473133 [00:09<02:31, 2776641.20B/s]  5%|▍         | 20054016/440473133 [00:09<02:24, 2904379.14B/s]  5%|▍         | 20419584/440473133 [00:09<02:16, 3069886.14B/s]  5%|▍         | 20733952/440473133 [00:10<02:16, 3073453.23B/s]  5%|▍         | 21115904/440473133 [00:10<02:09, 3244140.89B/s]  5%|▍         | 21446656/440473133 [00:10<02:13, 3127812.69B/s]  5%|▍         | 21899264/440473133 [00:10<02:08, 3260561.37B/s]  5%|▌         | 22333440/440473133 [00:10<01:58, 3523386.07B/s]  5%|▌         | 22695936/440473133 [00:10<02:03, 3380581.47B/s]  5%|▌         | 23187456/440473133 [00:10<01:58, 3533936.98B/s]  5%|▌         | 23716864/440473133 [00:10<01:46, 3925468.65B/s]  5%|▌         | 24128512/440473133 [00:11<01:53, 3670152.41B/s]  6%|▌         | 24580096/440473133 [00:11<01:49, 3797255.03B/s]  6%|▌         | 25137152/440473133 [00:11<01:39, 4180663.33B/s]  6%|▌         | 25576448/440473133 [00:11<01:45, 3922558.87B/s]  6%|▌         | 26094592/440473133 [00:11<01:40, 4131007.84B/s]  6%|▌         | 26679296/440473133 [00:11<01:31, 4529769.03B/s]  6%|▌         | 27155456/440473133 [00:11<01:35, 4314379.92B/s]  6%|▋         | 27764736/440473133 [00:11<01:31, 4518160.23B/s]  6%|▋         | 28310528/440473133 [00:11<01:26, 4764210.72B/s]  7%|▋         | 28827648/440473133 [00:12<01:27, 4722769.04B/s]  7%|▋         | 29471744/440473133 [00:12<01:20, 5132768.21B/s]  7%|▋         | 30003200/440473133 [00:12<01:20, 5082727.18B/s]  7%|▋         | 30655488/440473133 [00:12<01:15, 5422712.65B/s]  7%|▋         | 31213568/440473133 [00:12<01:16, 5369397.73B/s]  7%|▋         | 31978496/440473133 [00:12<01:14, 5512683.21B/s]  7%|▋         | 32692224/440473133 [00:12<01:09, 5866345.22B/s]  8%|▊         | 33291264/440473133 [00:12<01:09, 5889227.14B/s]  8%|▊         | 34032640/440473133 [00:12<01:05, 6176852.15B/s]  8%|▊         | 34711552/440473133 [00:12<01:04, 6316692.92B/s]  8%|▊         | 35442688/440473133 [00:13<01:01, 6565563.18B/s]  8%|▊         | 36121600/440473133 [00:13<01:01, 6559519.42B/s]  8%|▊         | 36943872/440473133 [00:13<00:57, 6983113.62B/s]  9%|▊         | 37654528/440473133 [00:13<00:58, 6905498.17B/s]  9%|▊         | 38540288/440473133 [00:13<00:54, 7386370.31B/s]  9%|▉         | 39293952/440473133 [00:13<00:54, 7351991.84B/s]  9%|▉         | 40200192/440473133 [00:13<00:51, 7792891.55B/s]  9%|▉         | 40994816/440473133 [00:13<00:51, 7761092.64B/s] 10%|▉         | 41935872/440473133 [00:13<00:48, 8189591.90B/s] 10%|▉         | 42768384/440473133 [00:13<00:48, 8168547.36B/s] 10%|▉         | 43781120/440473133 [00:14<00:45, 8657646.55B/s] 10%|█         | 44661760/440473133 [00:14<00:45, 8632479.44B/s] 10%|█         | 45713408/440473133 [00:14<00:43, 9087026.54B/s] 11%|█         | 46636032/440473133 [00:14<00:43, 9077901.34B/s] 11%|█         | 47748096/440473133 [00:14<00:40, 9607248.67B/s] 11%|█         | 48723968/440473133 [00:14<00:41, 9496877.35B/s] 11%|█▏        | 49893376/440473133 [00:14<00:38, 10056002.34B/s] 12%|█▏        | 50916352/440473133 [00:14<00:38, 10008822.63B/s] 12%|█▏        | 52109312/440473133 [00:14<00:36, 10516326.28B/s] 12%|█▏        | 53176320/440473133 [00:15<00:37, 10417752.91B/s] 12%|█▏        | 54497280/440473133 [00:15<00:34, 11089987.35B/s] 13%|█▎        | 55625728/440473133 [00:15<00:34, 11029986.49B/s] 13%|█▎        | 56889344/440473133 [00:15<00:33, 11296285.75B/s] 13%|█▎        | 58200064/440473133 [00:15<00:33, 11404284.33B/s] 14%|█▎        | 59740160/440473133 [00:15<00:30, 12353309.87B/s] 14%|█▍        | 61004800/440473133 [00:15<00:31, 12036809.98B/s] 14%|█▍        | 62492672/440473133 [00:15<00:30, 12429663.70B/s] 15%|█▍        | 64098304/440473133 [00:15<00:28, 13064331.86B/s] 15%|█▍        | 65582080/440473133 [00:15<00:27, 13549824.61B/s] 15%|█▌        | 67047424/440473133 [00:16<00:27, 13706339.24B/s] 16%|█▌        | 68562944/440473133 [00:16<00:26, 14110706.83B/s] 16%|█▌        | 70127616/440473133 [00:16<00:25, 14393805.52B/s] 16%|█▋        | 71577600/440473133 [00:16<00:25, 14411817.29B/s] 17%|█▋        | 73355264/440473133 [00:16<00:24, 15160278.35B/s] 17%|█▋        | 74887168/440473133 [00:16<00:24, 15184060.99B/s] 17%|█▋        | 76637184/440473133 [00:16<00:23, 15811412.51B/s] 18%|█▊        | 78237696/440473133 [00:16<00:22, 15777553.60B/s] 18%|█▊        | 80072704/440473133 [00:16<00:21, 16469757.94B/s] 19%|█▊        | 81825792/440473133 [00:16<00:21, 16623322.87B/s] 19%|█▉        | 83644416/440473133 [00:17<00:21, 16937828.51B/s] 19%|█▉        | 85594112/440473133 [00:17<00:20, 17589894.41B/s] 20%|█▉        | 87478272/440473133 [00:17<00:19, 17827540.14B/s] 20%|██        | 89526272/440473133 [00:17<00:19, 18469368.33B/s] 21%|██        | 91508736/440473133 [00:17<00:18, 18717838.41B/s] 21%|██▏       | 93638656/440473133 [00:17<00:17, 19372946.76B/s] 22%|██▏       | 95686656/440473133 [00:17<00:17, 19523793.71B/s] 22%|██▏       | 97914880/440473133 [00:17<00:17, 20088117.35B/s] 23%|██▎       | 100061184/440473133 [00:17<00:16, 20442908.75B/s] 23%|██▎       | 102387712/440473133 [00:17<00:16, 21017827.91B/s] 24%|██▍       | 104640512/440473133 [00:18<00:15, 21447729.14B/s] 24%|██▍       | 107024384/440473133 [00:18<00:15, 21884550.25B/s] 25%|██▍       | 109378560/440473133 [00:18<00:14, 22356312.82B/s] 25%|██▌       | 111857664/440473133 [00:18<00:14, 22817647.14B/s] 26%|██▌       | 114315264/440473133 [00:18<00:14, 23291242.03B/s] 27%|██▋       | 116920320/440473133 [00:18<00:13, 23835269.20B/s] 27%|██▋       | 119436288/440473133 [00:18<00:13, 24217473.47B/s] 28%|██▊       | 122179584/440473133 [00:18<00:12, 24797661.62B/s] 28%|██▊       | 124778496/440473133 [00:18<00:12, 25142118.52B/s] 29%|██▉       | 127668224/440473133 [00:19<00:12, 25840167.18B/s] 30%|██▉       | 130514944/440473133 [00:19<00:11, 26575303.18B/s] 30%|███       | 133353472/440473133 [00:19<00:11, 26627239.97B/s] 31%|███       | 136161280/440473133 [00:19<00:11, 27045837.60B/s] 32%|███▏      | 139284480/440473133 [00:19<00:10, 27864300.03B/s] 32%|███▏      | 142397440/440473133 [00:19<00:10, 28640025.50B/s] 33%|███▎      | 145444864/440473133 [00:19<00:10, 28898294.34B/s] 34%|███▍      | 148688896/440473133 [00:19<00:09, 29789854.35B/s] 34%|███▍      | 151851008/440473133 [00:19<00:09, 29911653.03B/s] 35%|███▌      | 155242496/440473133 [00:19<00:09, 30840875.25B/s] 36%|███▌      | 158486528/440473133 [00:20<00:09, 31029434.27B/s] 37%|███▋      | 162025472/440473133 [00:20<00:08, 32158638.65B/s] 38%|███▊      | 165384192/440473133 [00:20<00:08, 32311842.91B/s] 38%|███▊      | 169054208/440473133 [00:20<00:08, 33384574.56B/s] 39%|███▉      | 172544000/440473133 [00:20<00:07, 33529810.59B/s] 40%|████      | 176345088/440473133 [00:20<00:07, 34741526.08B/s] 41%|████      | 179965952/440473133 [00:20<00:07, 34785611.29B/s] 42%|████▏     | 183947264/440473133 [00:20<00:07, 36096101.76B/s] 43%|████▎     | 187666432/440473133 [00:20<00:07, 35758922.02B/s] 44%|████▎     | 191844352/440473133 [00:20<00:06, 37354793.03B/s] 44%|████▍     | 195645440/440473133 [00:21<00:06, 37022812.09B/s] 45%|████▌     | 200015872/440473133 [00:21<00:06, 38802405.73B/s] 46%|████▋     | 203934720/440473133 [00:21<00:06, 38524974.49B/s] 47%|████▋     | 208293888/440473133 [00:21<00:05, 39915289.16B/s] 48%|████▊     | 212340736/440473133 [00:21<00:05, 39802642.04B/s] 49%|████▉     | 216949760/440473133 [00:21<00:05, 41500156.06B/s] 50%|█████     | 221134848/440473133 [00:21<00:05, 41001567.05B/s] 51%|█████     | 225599488/440473133 [00:21<00:05, 42030378.34B/s] 52%|█████▏    | 229827584/440473133 [00:21<00:05, 41566170.15B/s] 53%|█████▎    | 234630144/440473133 [00:21<00:04, 43312635.53B/s] 54%|█████▍    | 238993408/440473133 [00:22<00:04, 42132166.65B/s] 55%|█████▌    | 243650560/440473133 [00:22<00:04, 43371692.58B/s] 56%|█████▋    | 248017920/440473133 [00:22<00:04, 42166510.98B/s] 57%|█████▋    | 252311552/440473133 [00:22<00:04, 42393413.87B/s] 58%|█████▊    | 256571392/440473133 [00:22<00:04, 41789371.08B/s] 59%|█████▉    | 261373952/440473133 [00:22<00:04, 43482401.31B/s] 60%|██████    | 265751552/440473133 [00:22<00:04, 41506160.25B/s] 61%|██████▏   | 270651392/440473133 [00:22<00:04, 41562023.93B/s] 62%|██████▏   | 275141632/440473133 [00:22<00:03, 42510081.82B/s] 63%|██████▎   | 279417856/440473133 [00:23<00:03, 41303207.77B/s] 65%|██████▍   | 284124160/440473133 [00:23<00:03, 42877297.58B/s] 65%|██████▌   | 288445440/440473133 [00:23<00:03, 41508109.61B/s] 67%|██████▋   | 293212160/440473133 [00:23<00:03, 42793734.05B/s] 68%|██████▊   | 297524224/440473133 [00:23<00:03, 41319790.06B/s] 69%|██████▊   | 301969408/440473133 [00:23<00:03, 42211183.68B/s] 70%|██████▉   | 306219008/440473133 [00:23<00:03, 41515233.67B/s] 71%|███████   | 310620160/440473133 [00:23<00:03, 42233825.30B/s] 71%|███████▏  | 314862592/440473133 [00:23<00:03, 41431810.45B/s] 72%|███████▏  | 319246336/440473133 [00:24<00:03, 40283246.73B/s] 74%|███████▎  | 323934208/440473133 [00:24<00:02, 42058365.52B/s] 75%|███████▍  | 328173568/440473133 [00:24<00:02, 40822253.11B/s] 75%|███████▌  | 332288000/440473133 [00:24<00:02, 40900926.87B/s] 76%|███████▋  | 336400384/440473133 [00:24<00:02, 40510363.13B/s] 77%|███████▋  | 341069824/440473133 [00:24<00:02, 41328869.17B/s] 78%|███████▊  | 345219072/440473133 [00:24<00:02, 40830222.15B/s] 79%|███████▉  | 349401088/440473133 [00:24<00:02, 41120379.07B/s] 80%|████████  | 353522688/440473133 [00:24<00:02, 40445156.39B/s] 81%|████████▏ | 357926912/440473133 [00:24<00:01, 41459875.38B/s] 82%|████████▏ | 362085376/440473133 [00:25<00:01, 41033148.07B/s] 83%|████████▎ | 366440448/440473133 [00:25<00:01, 41755868.86B/s] 84%|████████▍ | 370626560/440473133 [00:25<00:01, 40895195.52B/s] 85%|████████▌ | 375066624/440473133 [00:25<00:01, 41885502.65B/s] 86%|████████▌ | 379269120/440473133 [00:25<00:01, 41452258.36B/s] 87%|████████▋ | 383510528/440473133 [00:25<00:01, 41735588.02B/s] 88%|████████▊ | 387692544/440473133 [00:25<00:01, 41144133.92B/s] 89%|████████▉ | 392363008/440473133 [00:25<00:01, 42667399.17B/s] 90%|█████████ | 396649472/440473133 [00:25<00:01, 41115004.19B/s] 91%|█████████ | 401068032/440473133 [00:25<00:00, 40614785.64B/s] 92%|█████████▏| 405702656/440473133 [00:26<00:00, 42179268.57B/s] 93%|█████████▎| 409950208/440473133 [00:26<00:00, 41170457.52B/s] 94%|█████████▍| 414571520/440473133 [00:26<00:00, 42563721.73B/s] 95%|█████████▌| 418857984/440473133 [00:26<00:00, 40937310.65B/s] 96%|█████████▌| 423449600/440473133 [00:26<00:00, 42313261.10B/s] 97%|█████████▋| 427717632/440473133 [00:26<00:00, 41223440.29B/s] 98%|█████████▊| 432115712/440473133 [00:26<00:00, 40090390.22B/s] 99%|█████████▉| 436496384/440473133 [00:26<00:00, 41136755.85B/s]100%|██████████| 440473133/440473133 [00:26<00:00, 16348771.19B/s]Downloading data from https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip

   8192/7094233 [..............................] - ETA: 0s
5578752/7094233 [======================>.......] - ETA: 0s
7094272/7094233 [==============================] - 0s 0us/step

Processing text_left with encode:   0%|          | 0/2118 [00:00<?, ?it/s]Processing text_left with encode:   0%|          | 2/2118 [00:00<03:09, 11.16it/s]Processing text_left with encode:  21%|██        | 440/2118 [00:00<01:45, 15.93it/s]Processing text_left with encode:  43%|████▎     | 902/2118 [00:00<00:53, 22.72it/s]Processing text_left with encode:  62%|██████▏   | 1317/2118 [00:00<00:24, 32.38it/s]Processing text_left with encode:  84%|████████▍ | 1788/2118 [00:00<00:07, 46.12it/s]Processing text_left with encode: 100%|██████████| 2118/2118 [00:00<00:00, 3250.46it/s]
Processing text_right with encode:   0%|          | 0/18841 [00:00<?, ?it/s]Processing text_right with encode:   1%|          | 160/18841 [00:00<00:11, 1599.00it/s]Processing text_right with encode:   2%|▏         | 336/18841 [00:00<00:11, 1641.72it/s]Processing text_right with encode:   3%|▎         | 510/18841 [00:00<00:10, 1667.08it/s]Processing text_right with encode:   4%|▎         | 681/18841 [00:00<00:10, 1679.02it/s]Processing text_right with encode:   4%|▍         | 845/18841 [00:00<00:10, 1663.83it/s]Processing text_right with encode:   5%|▌         | 1009/18841 [00:00<00:10, 1655.61it/s]Processing text_right with encode:   6%|▌         | 1163/18841 [00:00<00:10, 1617.21it/s]Processing text_right with encode:   7%|▋         | 1332/18841 [00:00<00:10, 1637.83it/s]Processing text_right with encode:   8%|▊         | 1495/18841 [00:00<00:10, 1634.27it/s]Processing text_right with encode:   9%|▉         | 1673/18841 [00:01<00:10, 1674.49it/s]Processing text_right with encode:  10%|▉         | 1837/18841 [00:01<00:10, 1654.37it/s]Processing text_right with encode:  11%|█         | 2001/18841 [00:01<00:10, 1648.34it/s]Processing text_right with encode:  12%|█▏        | 2177/18841 [00:01<00:09, 1676.13it/s]Processing text_right with encode:  12%|█▏        | 2345/18841 [00:01<00:09, 1676.50it/s]Processing text_right with encode:  13%|█▎        | 2518/18841 [00:01<00:09, 1690.21it/s]Processing text_right with encode:  14%|█▍        | 2687/18841 [00:01<00:09, 1682.64it/s]Processing text_right with encode:  15%|█▌        | 2880/18841 [00:01<00:09, 1746.51it/s]Processing text_right with encode:  16%|█▌        | 3055/18841 [00:01<00:09, 1697.44it/s]Processing text_right with encode:  17%|█▋        | 3226/18841 [00:01<00:09, 1687.08it/s]Processing text_right with encode:  18%|█▊        | 3396/18841 [00:02<00:09, 1659.49it/s]Processing text_right with encode:  19%|█▉        | 3563/18841 [00:02<00:09, 1621.24it/s]Processing text_right with encode:  20%|█▉        | 3727/18841 [00:02<00:09, 1626.71it/s]Processing text_right with encode:  21%|██        | 3898/18841 [00:02<00:09, 1650.13it/s]Processing text_right with encode:  22%|██▏       | 4076/18841 [00:02<00:08, 1686.46it/s]Processing text_right with encode:  23%|██▎       | 4246/18841 [00:02<00:08, 1643.11it/s]Processing text_right with encode:  23%|██▎       | 4413/18841 [00:02<00:08, 1651.05it/s]Processing text_right with encode:  24%|██▍       | 4579/18841 [00:02<00:08, 1638.87it/s]Processing text_right with encode:  25%|██▌       | 4749/18841 [00:02<00:08, 1653.46it/s]Processing text_right with encode:  26%|██▌       | 4931/18841 [00:02<00:08, 1694.42it/s]Processing text_right with encode:  27%|██▋       | 5101/18841 [00:03<00:08, 1670.41it/s]Processing text_right with encode:  28%|██▊       | 5289/18841 [00:03<00:07, 1728.06it/s]Processing text_right with encode:  29%|██▉       | 5463/18841 [00:03<00:07, 1711.86it/s]Processing text_right with encode:  30%|██▉       | 5635/18841 [00:03<00:07, 1676.59it/s]Processing text_right with encode:  31%|███       | 5804/18841 [00:03<00:07, 1671.73it/s]Processing text_right with encode:  32%|███▏      | 5973/18841 [00:03<00:07, 1676.85it/s]Processing text_right with encode:  33%|███▎      | 6141/18841 [00:03<00:07, 1670.39it/s]Processing text_right with encode:  33%|███▎      | 6309/18841 [00:03<00:07, 1659.00it/s]Processing text_right with encode:  34%|███▍      | 6485/18841 [00:03<00:07, 1685.25it/s]Processing text_right with encode:  35%|███▌      | 6665/18841 [00:03<00:07, 1717.50it/s]Processing text_right with encode:  36%|███▋      | 6838/18841 [00:04<00:07, 1694.52it/s]Processing text_right with encode:  37%|███▋      | 7008/18841 [00:04<00:07, 1680.66it/s]Processing text_right with encode:  38%|███▊      | 7177/18841 [00:04<00:06, 1673.19it/s]Processing text_right with encode:  39%|███▉      | 7352/18841 [00:04<00:06, 1695.02it/s]Processing text_right with encode:  40%|███▉      | 7527/18841 [00:04<00:06, 1707.57it/s]Processing text_right with encode:  41%|████      | 7698/18841 [00:04<00:06, 1690.71it/s]Processing text_right with encode:  42%|████▏     | 7870/18841 [00:04<00:06, 1698.50it/s]Processing text_right with encode:  43%|████▎     | 8040/18841 [00:04<00:06, 1661.07it/s]Processing text_right with encode:  44%|████▎     | 8214/18841 [00:04<00:06, 1681.01it/s]Processing text_right with encode:  45%|████▍     | 8389/18841 [00:04<00:06, 1697.72it/s]Processing text_right with encode:  45%|████▌     | 8559/18841 [00:05<00:06, 1660.18it/s]Processing text_right with encode:  46%|████▋     | 8732/18841 [00:05<00:06, 1676.64it/s]Processing text_right with encode:  47%|████▋     | 8909/18841 [00:05<00:05, 1702.41it/s]Processing text_right with encode:  48%|████▊     | 9080/18841 [00:05<00:06, 1622.55it/s]Processing text_right with encode:  49%|████▉     | 9258/18841 [00:05<00:05, 1665.58it/s]Processing text_right with encode:  50%|█████     | 9426/18841 [00:05<00:05, 1651.78it/s]Processing text_right with encode:  51%|█████     | 9608/18841 [00:05<00:05, 1698.77it/s]Processing text_right with encode:  52%|█████▏    | 9779/18841 [00:05<00:05, 1646.24it/s]Processing text_right with encode:  53%|█████▎    | 9956/18841 [00:05<00:05, 1678.76it/s]Processing text_right with encode:  54%|█████▎    | 10126/18841 [00:06<00:05, 1683.62it/s]Processing text_right with encode:  55%|█████▍    | 10295/18841 [00:06<00:05, 1658.33it/s]Processing text_right with encode:  56%|█████▌    | 10499/18841 [00:06<00:04, 1754.88it/s]Processing text_right with encode:  57%|█████▋    | 10677/18841 [00:06<00:04, 1724.93it/s]Processing text_right with encode:  58%|█████▊    | 10851/18841 [00:06<00:04, 1689.26it/s]Processing text_right with encode:  59%|█████▊    | 11022/18841 [00:06<00:04, 1668.92it/s]Processing text_right with encode:  59%|█████▉    | 11192/18841 [00:06<00:04, 1675.49it/s]Processing text_right with encode:  60%|██████    | 11361/18841 [00:06<00:04, 1669.26it/s]Processing text_right with encode:  61%|██████    | 11530/18841 [00:06<00:04, 1674.36it/s]Processing text_right with encode:  62%|██████▏   | 11698/18841 [00:06<00:04, 1669.43it/s]Processing text_right with encode:  63%|██████▎   | 11873/18841 [00:07<00:04, 1688.66it/s]Processing text_right with encode:  64%|██████▍   | 12046/18841 [00:07<00:04, 1698.45it/s]Processing text_right with encode:  65%|██████▍   | 12217/18841 [00:07<00:03, 1689.71it/s]Processing text_right with encode:  66%|██████▌   | 12387/18841 [00:07<00:03, 1688.98it/s]Processing text_right with encode:  67%|██████▋   | 12556/18841 [00:07<00:03, 1678.30it/s]Processing text_right with encode:  68%|██████▊   | 12728/18841 [00:07<00:03, 1689.58it/s]Processing text_right with encode:  68%|██████▊   | 12898/18841 [00:07<00:03, 1657.25it/s]Processing text_right with encode:  69%|██████▉   | 13075/18841 [00:07<00:03, 1688.08it/s]Processing text_right with encode:  70%|███████   | 13245/18841 [00:07<00:03, 1666.36it/s]Processing text_right with encode:  71%|███████   | 13423/18841 [00:07<00:03, 1698.29it/s]Processing text_right with encode:  72%|███████▏  | 13594/18841 [00:08<00:03, 1682.68it/s]Processing text_right with encode:  73%|███████▎  | 13771/18841 [00:08<00:02, 1706.15it/s]Processing text_right with encode:  74%|███████▍  | 13942/18841 [00:08<00:02, 1703.38it/s]Processing text_right with encode:  75%|███████▍  | 14113/18841 [00:08<00:02, 1698.63it/s]Processing text_right with encode:  76%|███████▌  | 14283/18841 [00:08<00:02, 1679.77it/s]Processing text_right with encode:  77%|███████▋  | 14452/18841 [00:08<00:02, 1678.96it/s]Processing text_right with encode:  78%|███████▊  | 14630/18841 [00:08<00:02, 1706.39it/s]Processing text_right with encode:  79%|███████▊  | 14810/18841 [00:08<00:02, 1732.96it/s]Processing text_right with encode:  80%|███████▉  | 14984/18841 [00:08<00:02, 1707.73it/s]Processing text_right with encode:  80%|████████  | 15157/18841 [00:09<00:02, 1712.70it/s]Processing text_right with encode:  81%|████████▏ | 15329/18841 [00:09<00:02, 1698.45it/s]Processing text_right with encode:  82%|████████▏ | 15499/18841 [00:09<00:01, 1685.70it/s]Processing text_right with encode:  83%|████████▎ | 15678/18841 [00:09<00:01, 1714.91it/s]Processing text_right with encode:  84%|████████▍ | 15855/18841 [00:09<00:01, 1728.16it/s]Processing text_right with encode:  85%|████████▌ | 16028/18841 [00:09<00:01, 1717.83it/s]Processing text_right with encode:  86%|████████▌ | 16200/18841 [00:09<00:01, 1698.13it/s]Processing text_right with encode:  87%|████████▋ | 16370/18841 [00:09<00:01, 1658.45it/s]Processing text_right with encode:  88%|████████▊ | 16537/18841 [00:09<00:01, 1631.32it/s]Processing text_right with encode:  89%|████████▊ | 16701/18841 [00:09<00:01, 1632.16it/s]Processing text_right with encode:  90%|████████▉ | 16871/18841 [00:10<00:01, 1645.97it/s]Processing text_right with encode:  90%|█████████ | 17036/18841 [00:10<00:01, 1640.42it/s]Processing text_right with encode:  91%|█████████▏| 17205/18841 [00:10<00:00, 1653.62it/s]Processing text_right with encode:  92%|█████████▏| 17378/18841 [00:10<00:00, 1667.11it/s]Processing text_right with encode:  93%|█████████▎| 17545/18841 [00:10<00:00, 1654.11it/s]Processing text_right with encode:  94%|█████████▍| 17725/18841 [00:10<00:00, 1692.63it/s]Processing text_right with encode:  95%|█████████▍| 17895/18841 [00:10<00:00, 1677.33it/s]Processing text_right with encode:  96%|█████████▌| 18083/18841 [00:10<00:00, 1731.24it/s]Processing text_right with encode:  97%|█████████▋| 18257/18841 [00:10<00:00, 1659.88it/s]Processing text_right with encode:  98%|█████████▊| 18444/18841 [00:10<00:00, 1715.89it/s]Processing text_right with encode:  99%|█████████▉| 18619/18841 [00:11<00:00, 1724.70it/s]Processing text_right with encode: 100%|█████████▉| 18793/18841 [00:11<00:00, 1704.85it/s]Processing text_right with encode: 100%|██████████| 18841/18841 [00:11<00:00, 1682.60it/s]
Processing length_left with len:   0%|          | 0/2118 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 2118/2118 [00:00<00:00, 522714.67it/s]
Processing length_right with len:   0%|          | 0/18841 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 18841/18841 [00:00<00:00, 767492.66it/s]
Processing text_left with encode:   0%|          | 0/633 [00:00<?, ?it/s]Processing text_left with encode:  74%|███████▍  | 468/633 [00:00<00:00, 4671.80it/s]Processing text_left with encode: 100%|██████████| 633/633 [00:00<00:00, 4601.90it/s]
Processing text_right with encode:   0%|          | 0/5961 [00:00<?, ?it/s]Processing text_right with encode:   3%|▎         | 176/5961 [00:00<00:03, 1752.53it/s]Processing text_right with encode:   6%|▌         | 349/5961 [00:00<00:03, 1745.23it/s]Processing text_right with encode:   9%|▊         | 513/5961 [00:00<00:03, 1712.01it/s]Processing text_right with encode:  12%|█▏        | 687/5961 [00:00<00:03, 1719.60it/s]Processing text_right with encode:  15%|█▍        | 869/5961 [00:00<00:02, 1746.74it/s]Processing text_right with encode:  17%|█▋        | 1041/5961 [00:00<00:02, 1738.30it/s]Processing text_right with encode:  20%|██        | 1218/5961 [00:00<00:02, 1747.63it/s]Processing text_right with encode:  23%|██▎       | 1398/5961 [00:00<00:02, 1761.45it/s]Processing text_right with encode:  26%|██▋       | 1574/5961 [00:00<00:02, 1760.61it/s]Processing text_right with encode:  29%|██▉       | 1744/5961 [00:01<00:02, 1704.78it/s]Processing text_right with encode:  32%|███▏      | 1913/5961 [00:01<00:02, 1698.79it/s]Processing text_right with encode:  35%|███▍      | 2080/5961 [00:01<00:02, 1679.62it/s]Processing text_right with encode:  38%|███▊      | 2257/5961 [00:01<00:02, 1696.68it/s]Processing text_right with encode:  41%|████      | 2426/5961 [00:01<00:02, 1603.81it/s]Processing text_right with encode:  44%|████▎     | 2607/5961 [00:01<00:02, 1659.00it/s]Processing text_right with encode:  47%|████▋     | 2797/5961 [00:01<00:01, 1724.42it/s]Processing text_right with encode:  50%|████▉     | 2972/5961 [00:01<00:01, 1729.67it/s]Processing text_right with encode:  53%|█████▎    | 3159/5961 [00:01<00:01, 1767.66it/s]Processing text_right with encode:  56%|█████▌    | 3337/5961 [00:01<00:01, 1653.37it/s]Processing text_right with encode:  59%|█████▉    | 3505/5961 [00:02<00:01, 1621.80it/s]Processing text_right with encode:  62%|██████▏   | 3679/5961 [00:02<00:01, 1655.41it/s]Processing text_right with encode:  65%|██████▍   | 3846/5961 [00:02<00:01, 1645.30it/s]Processing text_right with encode:  68%|██████▊   | 4036/5961 [00:02<00:01, 1712.90it/s]Processing text_right with encode:  71%|███████   | 4209/5961 [00:02<00:01, 1676.19it/s]Processing text_right with encode:  73%|███████▎  | 4379/5961 [00:02<00:00, 1683.03it/s]Processing text_right with encode:  76%|███████▋  | 4549/5961 [00:02<00:00, 1665.50it/s]Processing text_right with encode:  79%|███████▉  | 4721/5961 [00:02<00:00, 1680.58it/s]Processing text_right with encode:  82%|████████▏ | 4890/5961 [00:02<00:00, 1671.63it/s]Processing text_right with encode:  85%|████████▍ | 5058/5961 [00:02<00:00, 1650.26it/s]Processing text_right with encode:  88%|████████▊ | 5238/5961 [00:03<00:00, 1691.68it/s]Processing text_right with encode:  91%|█████████ | 5408/5961 [00:03<00:00, 1644.14it/s]Processing text_right with encode:  94%|█████████▎| 5574/5961 [00:03<00:00, 1564.85it/s]Processing text_right with encode:  96%|█████████▌| 5732/5961 [00:03<00:00, 1556.84it/s]Processing text_right with encode:  99%|█████████▉| 5918/5961 [00:03<00:00, 1634.25it/s]Processing text_right with encode: 100%|██████████| 5961/5961 [00:03<00:00, 1683.68it/s]
Processing length_left with len:   0%|          | 0/633 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 633/633 [00:00<00:00, 486708.42it/s]
Processing length_right with len:   0%|          | 0/5961 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 5961/5961 [00:00<00:00, 670643.12it/s]
  #### Model  fit   ############################################# 

  0%|          | 0/102 [00:00<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:17<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:17<?, ?it/s, loss=0.864]Epoch 1/1:   1%|          | 1/102 [00:17<29:17, 17.40s/it, loss=0.864]Epoch 1/1:   1%|          | 1/102 [01:39<29:17, 17.40s/it, loss=0.864]Epoch 1/1:   1%|          | 1/102 [01:39<29:17, 17.40s/it, loss=0.964]Epoch 1/1:   2%|▏         | 2/102 [01:39<1:01:26, 36.87s/it, loss=0.964]Epoch 1/1:   2%|▏         | 2/102 [02:37<1:01:26, 36.87s/it, loss=0.964]Epoch 1/1:   2%|▏         | 2/102 [02:37<1:01:26, 36.87s/it, loss=0.836]Epoch 1/1:   3%|▎         | 3/102 [02:37<1:10:57, 43.01s/it, loss=0.836]Killed

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   7587eb8..77f42b7  master     -> origin/master
Updating 7587eb8..77f42b7
Fast-forward
 .../20200514/list_log_dataloader_20200514.md       |    2 +-
 error_list/20200514/list_log_json_20200514.md      | 1146 ++++++-------
 error_list/20200514/list_log_jupyter_20200514.md   | 1686 ++++++++++----------
 error_list/20200514/list_log_testall_20200514.md   |   43 +
 4 files changed, 1471 insertions(+), 1406 deletions(-)
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
