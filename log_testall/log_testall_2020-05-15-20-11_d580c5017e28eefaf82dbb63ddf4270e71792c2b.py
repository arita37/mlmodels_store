
  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_all 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'd580c5017e28eefaf82dbb63ddf4270e71792c2b', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/d580c5017e28eefaf82dbb63ddf4270e71792c2b

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
[master 942a4c3] ml_store
 1 file changed, 60 insertions(+)
 create mode 100644 log_testall/log_testall_2020-05-15-20-11_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py
To github.com:arita37/mlmodels_store.git
   fd553e7..942a4c3  master -> master





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
[master 980a5e6] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   942a4c3..980a5e6  master -> master





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
[master 7fcb3b3] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   980a5e6..7fcb3b3  master -> master





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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         20          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         20          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-15 20:11:59.032246: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 20:11:59.036988: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-15 20:11:59.037156: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558d7bcc6220 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 20:11:59.037170: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 203
Trainable params: 203
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2475 - binary_crossentropy: 0.6862500/500 [==============================] - 1s 1ms/sample - loss: 0.2481 - binary_crossentropy: 0.6878 - val_loss: 0.2520 - val_binary_crossentropy: 0.7746

  #### metrics   #################################################### 
{'MSE': 0.24993527809017269}

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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         20          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
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
Total params: 203
Trainable params: 203
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
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 448
Trainable params: 448
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2640 - binary_crossentropy: 0.7219500/500 [==============================] - 1s 1ms/sample - loss: 0.2553 - binary_crossentropy: 0.7038 - val_loss: 0.2548 - val_binary_crossentropy: 0.7296

  #### metrics   #################################################### 
{'MSE': 0.2546564885864895}

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
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 448
Trainable params: 448
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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
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
Total params: 597
Trainable params: 597
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 2ms/sample - loss: 0.2499 - binary_crossentropy: 0.6930 - val_loss: 0.2497 - val_binary_crossentropy: 0.6925

  #### metrics   #################################################### 
{'MSE': 0.24961773683475094}

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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
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
Total params: 597
Trainable params: 597
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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         36          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
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
Total params: 473
Trainable params: 473
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2791 - binary_crossentropy: 0.7570500/500 [==============================] - 1s 2ms/sample - loss: 0.2789 - binary_crossentropy: 0.7610 - val_loss: 0.2737 - val_binary_crossentropy: 0.7457

  #### metrics   #################################################### 
{'MSE': 0.27503829538936836}

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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         36          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
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
Total params: 473
Trainable params: 473
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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_max[0][0]               
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
Total params: 138
Trainable params: 138
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2516 - binary_crossentropy: 0.6991500/500 [==============================] - 1s 3ms/sample - loss: 0.2579 - binary_crossentropy: 0.7111 - val_loss: 0.2613 - val_binary_crossentropy: 0.7174

  #### metrics   #################################################### 
{'MSE': 0.25845634238118576}

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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_max[0][0]               
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-15 20:13:11.583601: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 20:13:11.585552: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 20:13:11.592222: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-15 20:13:11.601757: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-15 20:13:11.603623: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-15 20:13:11.605338: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 20:13:11.606792: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2545 - val_binary_crossentropy: 0.7022
2020-05-15 20:13:12.755213: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 20:13:12.756885: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 20:13:12.761232: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-15 20:13:12.769979: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-15 20:13:12.772986: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-15 20:13:12.774465: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 20:13:12.775931: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2556657130133185}

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
2020-05-15 20:13:34.600128: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 20:13:34.601617: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 20:13:34.605746: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-15 20:13:34.613189: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-15 20:13:34.614381: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-15 20:13:34.615515: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 20:13:34.616658: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2505 - val_binary_crossentropy: 0.6942
2020-05-15 20:13:36.071704: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 20:13:36.073154: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 20:13:36.076028: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-15 20:13:36.081424: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-15 20:13:36.082418: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-15 20:13:36.083337: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 20:13:36.084139: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2506008120720393}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-15 20:14:07.610792: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 20:14:07.615508: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 20:14:07.629001: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-15 20:14:07.651365: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-15 20:14:07.655147: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-15 20:14:07.658581: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 20:14:07.662460: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.2142 - binary_crossentropy: 0.6214 - val_loss: 0.2571 - val_binary_crossentropy: 0.7075
2020-05-15 20:14:09.749389: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 20:14:09.753757: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 20:14:09.766252: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-15 20:14:09.788082: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-15 20:14:09.792072: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-15 20:14:09.795424: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 20:14:09.798663: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.22899016331058064}

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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         28          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
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
Total params: 725
Trainable params: 725
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2839 - binary_crossentropy: 1.1508500/500 [==============================] - 4s 8ms/sample - loss: 0.2852 - binary_crossentropy: 1.4907 - val_loss: 0.3084 - val_binary_crossentropy: 1.7807

  #### metrics   #################################################### 
{'MSE': 0.29621889689467845}

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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         28          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
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
Total params: 725
Trainable params: 725
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
sequence_max (InputLayer)       [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_mean (E (None, 9, 2)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 2)         4           sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         10          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         4           sparse_feature_4[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_4[0][0]           
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
Total params: 206
Trainable params: 206
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2626 - binary_crossentropy: 0.7286500/500 [==============================] - 4s 8ms/sample - loss: 0.2615 - binary_crossentropy: 0.7231 - val_loss: 0.2554 - val_binary_crossentropy: 0.7323

  #### metrics   #################################################### 
{'MSE': 0.2568267125514522}

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
sequence_max (InputLayer)       [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_mean (E (None, 9, 2)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 2)         4           sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         10          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         4           sparse_feature_4[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_4[0][0]           
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
Total params: 206
Trainable params: 206
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
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
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
100/500 [=====>........................] - ETA: 6s - loss: 0.4600 - binary_crossentropy: 7.0937500/500 [==============================] - 5s 9ms/sample - loss: 0.4720 - binary_crossentropy: 7.2795 - val_loss: 0.5020 - val_binary_crossentropy: 7.7433

  #### metrics   #################################################### 
{'MSE': 0.49}

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
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
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
regionsequence_sum (InputLayer) [(None, 8)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 2)]          0                                            
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
region_10sparse_seq_emb_regions (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 2, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 2, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 2, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 2, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 2, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 2, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 2, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 2, 1)         1           regionsequence_mean[0][0]        
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
Total params: 136
Trainable params: 136
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2501 - binary_crossentropy: 0.6933500/500 [==============================] - 5s 11ms/sample - loss: 0.2498 - binary_crossentropy: 0.6927 - val_loss: 0.2505 - val_binary_crossentropy: 0.6942

  #### metrics   #################################################### 
{'MSE': 0.24997634118932716}

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
regionsequence_mean (InputLayer [(None, 2)]          0                                            
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
region_10sparse_seq_emb_regions (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 2, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 2, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 2, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 2, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 2, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 2, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 2, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 2, 1)         1           regionsequence_mean[0][0]        
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_max[0][0]               
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
Total params: 1,407
Trainable params: 1,407
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2731 - binary_crossentropy: 0.7436500/500 [==============================] - 5s 11ms/sample - loss: 0.2616 - binary_crossentropy: 0.7181 - val_loss: 0.2573 - val_binary_crossentropy: 0.7082

  #### metrics   #################################################### 
{'MSE': 0.25781926043974385}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_max[0][0]               
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
Total params: 1,407
Trainable params: 1,407
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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         16          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         4           hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           hash_10[0][0]                    
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
Total params: 2,933
Trainable params: 2,853
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.3068 - binary_crossentropy: 0.9050500/500 [==============================] - 6s 12ms/sample - loss: 0.3607 - binary_crossentropy: 1.0315 - val_loss: 0.3165 - val_binary_crossentropy: 0.9080

  #### metrics   #################################################### 
{'MSE': 0.33614058516446754}

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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         16          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         4           hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           hash_10[0][0]                    
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
Total params: 2,933
Trainable params: 2,853
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
Total params: 449
Trainable params: 449
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2550 - binary_crossentropy: 0.7034500/500 [==============================] - 6s 12ms/sample - loss: 0.2522 - binary_crossentropy: 0.6978 - val_loss: 0.2522 - val_binary_crossentropy: 0.6975

  #### metrics   #################################################### 
{'MSE': 0.2512943800578989}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
Total params: 449
Trainable params: 449
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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_44 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_125 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_194[0][0] 
                                                                 sequence_pooling_layer_195[0][0] 
                                                                 sequence_pooling_layer_196[0][0] 
                                                                 sequence_pooling_layer_197[0][0] 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
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
Total params: 2,029
Trainable params: 2,029
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2950 - binary_crossentropy: 2.3362500/500 [==============================] - 6s 13ms/sample - loss: 0.3030 - binary_crossentropy: 2.2824 - val_loss: 0.3061 - val_binary_crossentropy: 2.2151

  #### metrics   #################################################### 
{'MSE': 0.3063703734317288}

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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_44 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_125 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_194[0][0] 
                                                                 sequence_pooling_layer_195[0][0] 
                                                                 sequence_pooling_layer_196[0][0] 
                                                                 sequence_pooling_layer_197[0][0] 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
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
Total params: 2,029
Trainable params: 2,029
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_47 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_90 (Concatenate)    (None, 1, 20)        0           no_mask_130[0][0]                
                                                                 no_mask_130[1][0]                
                                                                 no_mask_130[2][0]                
                                                                 no_mask_130[3][0]                
                                                                 no_mask_130[4][0]                
__________________________________________________________________________________________________
no_mask_131 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
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
Total params: 306
Trainable params: 306
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.5200 - binary_crossentropy: 8.0210500/500 [==============================] - 7s 13ms/sample - loss: 0.4980 - binary_crossentropy: 7.6816 - val_loss: 0.5140 - val_binary_crossentropy: 7.9284

  #### metrics   #################################################### 
{'MSE': 0.506}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_47 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_90 (Concatenate)    (None, 1, 20)        0           no_mask_130[0][0]                
                                                                 no_mask_130[1][0]                
                                                                 no_mask_130[2][0]                
                                                                 no_mask_130[3][0]                
                                                                 no_mask_130[4][0]                
__________________________________________________________________________________________________
no_mask_131 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
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
Total params: 306
Trainable params: 306
Non-trainable params: 0
__________________________________________________________________________________________________

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   7fcb3b3..e90ec66  master     -> origin/master
Updating 7fcb3b3..e90ec66
Fast-forward
 .../20200515/list_log_dataloader_20200515.md       |    2 +-
 error_list/20200515/list_log_import_20200515.md    |    2 +-
 error_list/20200515/list_log_json_20200515.md      | 1146 +++++------
 error_list/20200515/list_log_jupyter_20200515.md   | 1749 +++++++++--------
 .../20200515/list_log_pullrequest_20200515.md      |    2 +-
 error_list/20200515/list_log_test_cli_20200515.md  |  378 ++--
 ...-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py | 2023 ++++++++++++++++++++
 ...-10_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py |  612 ++++++
 8 files changed, 4269 insertions(+), 1645 deletions(-)
 create mode 100644 log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py
 create mode 100644 log_pullrequest/log_pr_2020-05-15-20-10_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py
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
[master d91b101] ml_store
 1 file changed, 5675 insertions(+)
To github.com:arita37/mlmodels_store.git
   e90ec66..d91b101  master -> master





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
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
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
[master ff0095f] ml_store
 1 file changed, 51 insertions(+)
To github.com:arita37/mlmodels_store.git
   d91b101..ff0095f  master -> master





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
[master 470275a] ml_store
 1 file changed, 46 insertions(+)
To github.com:arita37/mlmodels_store.git
   ff0095f..470275a  master -> master





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
[master 76b2dbd] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   470275a..76b2dbd  master -> master





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

2020-05-15 20:26:29.635934: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 20:26:29.640969: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-15 20:26:29.641122: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c1bd478810 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 20:26:29.641137: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 8s - loss: 1.3830
256/354 [====================>.........] - ETA: 3s - loss: 1.1847
354/354 [==============================] - 15s 42ms/step - loss: 1.6602 - val_loss: 2.1371

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
[master 40cf995] ml_store
 1 file changed, 149 insertions(+)
To github.com:arita37/mlmodels_store.git
   76b2dbd..40cf995  master -> master





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
Warning: Permanently added the RSA host key for IP address '140.82.113.4' to the list of known hosts.
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
[master bb30c3e] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   40cf995..bb30c3e  master -> master





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
[master 6c9467d] ml_store
 1 file changed, 44 insertions(+)
To github.com:arita37/mlmodels_store.git
   bb30c3e..6c9467d  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1875968/17464789 [==>...........................] - ETA: 0s
 8708096/17464789 [=============>................] - ETA: 0s
16244736/17464789 [==========================>...] - ETA: 0s
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
2020-05-15 20:27:29.855332: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 20:27:29.859658: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-15 20:27:29.859778: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b5889f5b40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 20:27:29.859793: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.9273 - accuracy: 0.4830
 2000/25000 [=>............................] - ETA: 9s - loss: 7.7433 - accuracy: 0.4950 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7433 - accuracy: 0.4950
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6015 - accuracy: 0.5042
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6176 - accuracy: 0.5032
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6308 - accuracy: 0.5023
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5615 - accuracy: 0.5069
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5689 - accuracy: 0.5064
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6121 - accuracy: 0.5036
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6160 - accuracy: 0.5033
11000/25000 [============>.................] - ETA: 4s - loss: 7.6178 - accuracy: 0.5032
12000/25000 [=============>................] - ETA: 4s - loss: 7.6066 - accuracy: 0.5039
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6265 - accuracy: 0.5026
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6611 - accuracy: 0.5004
15000/25000 [=================>............] - ETA: 3s - loss: 7.6789 - accuracy: 0.4992
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6810 - accuracy: 0.4991
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6639 - accuracy: 0.5002
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6820 - accuracy: 0.4990
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6513 - accuracy: 0.5010
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6597 - accuracy: 0.5005
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6491 - accuracy: 0.5011
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6534 - accuracy: 0.5009
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6573 - accuracy: 0.5006
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6609 - accuracy: 0.5004
25000/25000 [==============================] - 9s 369us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7fc550cddd68>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7fc5473cda20> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.6513 - accuracy: 0.5010
 2000/25000 [=>............................] - ETA: 9s - loss: 7.5823 - accuracy: 0.5055 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.5951 - accuracy: 0.5047
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5401 - accuracy: 0.5082
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5501 - accuracy: 0.5076
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5644 - accuracy: 0.5067
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6316 - accuracy: 0.5023
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5880 - accuracy: 0.5051
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6274 - accuracy: 0.5026
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6421 - accuracy: 0.5016
11000/25000 [============>.................] - ETA: 4s - loss: 7.6624 - accuracy: 0.5003
12000/25000 [=============>................] - ETA: 4s - loss: 7.6270 - accuracy: 0.5026
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6466 - accuracy: 0.5013
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6283 - accuracy: 0.5025
15000/25000 [=================>............] - ETA: 3s - loss: 7.6298 - accuracy: 0.5024
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6331 - accuracy: 0.5022
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6179 - accuracy: 0.5032
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6172 - accuracy: 0.5032
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6481 - accuracy: 0.5012
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6436 - accuracy: 0.5015
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6608 - accuracy: 0.5004
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6645 - accuracy: 0.5001
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6760 - accuracy: 0.4994
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6743 - accuracy: 0.4995
25000/25000 [==============================] - 9s 363us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.5286 - accuracy: 0.5090
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6973 - accuracy: 0.4980 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5593 - accuracy: 0.5070
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5861 - accuracy: 0.5052
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6574 - accuracy: 0.5006
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6334 - accuracy: 0.5022
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6403 - accuracy: 0.5017
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6340 - accuracy: 0.5021
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5917 - accuracy: 0.5049
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5961 - accuracy: 0.5046
11000/25000 [============>.................] - ETA: 4s - loss: 7.5886 - accuracy: 0.5051
12000/25000 [=============>................] - ETA: 3s - loss: 7.6117 - accuracy: 0.5036
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6053 - accuracy: 0.5040
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6414 - accuracy: 0.5016
15000/25000 [=================>............] - ETA: 2s - loss: 7.6390 - accuracy: 0.5018
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6647 - accuracy: 0.5001
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6747 - accuracy: 0.4995
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6641 - accuracy: 0.5002
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6586 - accuracy: 0.5005
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6682 - accuracy: 0.4999
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6754 - accuracy: 0.4994
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6840 - accuracy: 0.4989
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6786 - accuracy: 0.4992
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6743 - accuracy: 0.4995
25000/25000 [==============================] - 9s 361us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
   6c9467d..500451a  master     -> origin/master
Updating 6c9467d..500451a
Fast-forward
 error_list/20200515/list_log_import_20200515.md    |    2 +-
 error_list/20200515/list_log_json_20200515.md      | 1146 ++++++++++----------
 .../20200515/list_log_pullrequest_20200515.md      |    2 +-
 error_list/20200515/list_log_test_cli_20200515.md  |  378 +++----
 4 files changed, 769 insertions(+), 759 deletions(-)
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
[master 4e55411] ml_store
 1 file changed, 325 insertions(+)
To github.com:arita37/mlmodels_store.git
   500451a..4e55411  master -> master





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

13/13 [==============================] - 1s 109ms/step - loss: nan
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

13/13 [==============================] - 0s 5ms/step - loss: nan
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
[master 9c54071] ml_store
 1 file changed, 125 insertions(+)
To github.com:arita37/mlmodels_store.git
   4e55411..9c54071  master -> master





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
 1728512/11490434 [===>..........................] - ETA: 0s
 8134656/11490434 [====================>.........] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:47 - loss: 2.3274 - categorical_accuracy: 0.0312
   64/60000 [..............................] - ETA: 4:52 - loss: 2.2515 - categorical_accuracy: 0.1406
   96/60000 [..............................] - ETA: 3:49 - loss: 2.2677 - categorical_accuracy: 0.1354
  128/60000 [..............................] - ETA: 3:20 - loss: 2.2279 - categorical_accuracy: 0.1562
  160/60000 [..............................] - ETA: 3:03 - loss: 2.2034 - categorical_accuracy: 0.1750
  192/60000 [..............................] - ETA: 2:51 - loss: 2.1774 - categorical_accuracy: 0.1875
  224/60000 [..............................] - ETA: 2:42 - loss: 2.1354 - categorical_accuracy: 0.2098
  256/60000 [..............................] - ETA: 2:35 - loss: 2.1056 - categorical_accuracy: 0.2227
  288/60000 [..............................] - ETA: 2:29 - loss: 2.1034 - categorical_accuracy: 0.2222
  320/60000 [..............................] - ETA: 2:24 - loss: 2.0740 - categorical_accuracy: 0.2406
  352/60000 [..............................] - ETA: 2:20 - loss: 2.0240 - categorical_accuracy: 0.2699
  384/60000 [..............................] - ETA: 2:16 - loss: 1.9740 - categorical_accuracy: 0.2891
  416/60000 [..............................] - ETA: 2:14 - loss: 1.9179 - categorical_accuracy: 0.3221
  448/60000 [..............................] - ETA: 2:11 - loss: 1.9063 - categorical_accuracy: 0.3304
  480/60000 [..............................] - ETA: 2:09 - loss: 1.8881 - categorical_accuracy: 0.3396
  512/60000 [..............................] - ETA: 2:07 - loss: 1.8791 - categorical_accuracy: 0.3457
  544/60000 [..............................] - ETA: 2:06 - loss: 1.8523 - categorical_accuracy: 0.3658
  576/60000 [..............................] - ETA: 2:04 - loss: 1.8244 - categorical_accuracy: 0.3750
  608/60000 [..............................] - ETA: 2:03 - loss: 1.8003 - categorical_accuracy: 0.3849
  640/60000 [..............................] - ETA: 2:02 - loss: 1.7564 - categorical_accuracy: 0.4031
  672/60000 [..............................] - ETA: 2:01 - loss: 1.7316 - categorical_accuracy: 0.4107
  704/60000 [..............................] - ETA: 2:00 - loss: 1.7041 - categorical_accuracy: 0.4219
  736/60000 [..............................] - ETA: 1:59 - loss: 1.6997 - categorical_accuracy: 0.4226
  768/60000 [..............................] - ETA: 1:58 - loss: 1.6826 - categorical_accuracy: 0.4284
  800/60000 [..............................] - ETA: 1:58 - loss: 1.6498 - categorical_accuracy: 0.4412
  832/60000 [..............................] - ETA: 1:58 - loss: 1.6192 - categorical_accuracy: 0.4495
  864/60000 [..............................] - ETA: 1:57 - loss: 1.5906 - categorical_accuracy: 0.4595
  896/60000 [..............................] - ETA: 1:57 - loss: 1.5862 - categorical_accuracy: 0.4643
  928/60000 [..............................] - ETA: 1:56 - loss: 1.5763 - categorical_accuracy: 0.4666
  960/60000 [..............................] - ETA: 1:56 - loss: 1.5581 - categorical_accuracy: 0.4740
  992/60000 [..............................] - ETA: 1:55 - loss: 1.5381 - categorical_accuracy: 0.4829
 1024/60000 [..............................] - ETA: 1:55 - loss: 1.5210 - categorical_accuracy: 0.4873
 1056/60000 [..............................] - ETA: 1:55 - loss: 1.5064 - categorical_accuracy: 0.4924
 1088/60000 [..............................] - ETA: 1:54 - loss: 1.4873 - categorical_accuracy: 0.5000
 1120/60000 [..............................] - ETA: 1:54 - loss: 1.4750 - categorical_accuracy: 0.5045
 1152/60000 [..............................] - ETA: 1:54 - loss: 1.4566 - categorical_accuracy: 0.5104
 1184/60000 [..............................] - ETA: 1:54 - loss: 1.4412 - categorical_accuracy: 0.5135
 1216/60000 [..............................] - ETA: 1:54 - loss: 1.4251 - categorical_accuracy: 0.5197
 1248/60000 [..............................] - ETA: 1:54 - loss: 1.4182 - categorical_accuracy: 0.5224
 1280/60000 [..............................] - ETA: 1:53 - loss: 1.4038 - categorical_accuracy: 0.5289
 1312/60000 [..............................] - ETA: 1:53 - loss: 1.3858 - categorical_accuracy: 0.5366
 1344/60000 [..............................] - ETA: 1:53 - loss: 1.3834 - categorical_accuracy: 0.5387
 1376/60000 [..............................] - ETA: 1:52 - loss: 1.3766 - categorical_accuracy: 0.5429
 1408/60000 [..............................] - ETA: 1:52 - loss: 1.3631 - categorical_accuracy: 0.5504
 1440/60000 [..............................] - ETA: 1:52 - loss: 1.3482 - categorical_accuracy: 0.5556
 1472/60000 [..............................] - ETA: 1:52 - loss: 1.3308 - categorical_accuracy: 0.5598
 1504/60000 [..............................] - ETA: 1:52 - loss: 1.3209 - categorical_accuracy: 0.5638
 1536/60000 [..............................] - ETA: 1:51 - loss: 1.3116 - categorical_accuracy: 0.5658
 1568/60000 [..............................] - ETA: 1:51 - loss: 1.2994 - categorical_accuracy: 0.5702
 1600/60000 [..............................] - ETA: 1:51 - loss: 1.2855 - categorical_accuracy: 0.5756
 1632/60000 [..............................] - ETA: 1:51 - loss: 1.2727 - categorical_accuracy: 0.5797
 1664/60000 [..............................] - ETA: 1:50 - loss: 1.2632 - categorical_accuracy: 0.5835
 1696/60000 [..............................] - ETA: 1:50 - loss: 1.2493 - categorical_accuracy: 0.5879
 1728/60000 [..............................] - ETA: 1:50 - loss: 1.2399 - categorical_accuracy: 0.5909
 1760/60000 [..............................] - ETA: 1:50 - loss: 1.2323 - categorical_accuracy: 0.5920
 1792/60000 [..............................] - ETA: 1:50 - loss: 1.2189 - categorical_accuracy: 0.5960
 1824/60000 [..............................] - ETA: 1:50 - loss: 1.2164 - categorical_accuracy: 0.5959
 1856/60000 [..............................] - ETA: 1:50 - loss: 1.2042 - categorical_accuracy: 0.6008
 1888/60000 [..............................] - ETA: 1:49 - loss: 1.1959 - categorical_accuracy: 0.6033
 1920/60000 [..............................] - ETA: 1:49 - loss: 1.1911 - categorical_accuracy: 0.6047
 1952/60000 [..............................] - ETA: 1:49 - loss: 1.1858 - categorical_accuracy: 0.6071
 1984/60000 [..............................] - ETA: 1:49 - loss: 1.1755 - categorical_accuracy: 0.6109
 2016/60000 [>.............................] - ETA: 1:49 - loss: 1.1636 - categorical_accuracy: 0.6161
 2048/60000 [>.............................] - ETA: 1:49 - loss: 1.1549 - categorical_accuracy: 0.6191
 2080/60000 [>.............................] - ETA: 1:49 - loss: 1.1468 - categorical_accuracy: 0.6212
 2112/60000 [>.............................] - ETA: 1:49 - loss: 1.1426 - categorical_accuracy: 0.6236
 2144/60000 [>.............................] - ETA: 1:48 - loss: 1.1369 - categorical_accuracy: 0.6255
 2176/60000 [>.............................] - ETA: 1:48 - loss: 1.1268 - categorical_accuracy: 0.6287
 2208/60000 [>.............................] - ETA: 1:48 - loss: 1.1219 - categorical_accuracy: 0.6304
 2240/60000 [>.............................] - ETA: 1:48 - loss: 1.1191 - categorical_accuracy: 0.6321
 2272/60000 [>.............................] - ETA: 1:48 - loss: 1.1098 - categorical_accuracy: 0.6360
 2304/60000 [>.............................] - ETA: 1:48 - loss: 1.1029 - categorical_accuracy: 0.6380
 2336/60000 [>.............................] - ETA: 1:48 - loss: 1.0943 - categorical_accuracy: 0.6417
 2368/60000 [>.............................] - ETA: 1:48 - loss: 1.0872 - categorical_accuracy: 0.6444
 2400/60000 [>.............................] - ETA: 1:47 - loss: 1.0794 - categorical_accuracy: 0.6467
 2432/60000 [>.............................] - ETA: 1:47 - loss: 1.0733 - categorical_accuracy: 0.6488
 2464/60000 [>.............................] - ETA: 1:47 - loss: 1.0629 - categorical_accuracy: 0.6522
 2496/60000 [>.............................] - ETA: 1:47 - loss: 1.0538 - categorical_accuracy: 0.6550
 2528/60000 [>.............................] - ETA: 1:47 - loss: 1.0456 - categorical_accuracy: 0.6582
 2560/60000 [>.............................] - ETA: 1:47 - loss: 1.0386 - categorical_accuracy: 0.6598
 2592/60000 [>.............................] - ETA: 1:47 - loss: 1.0391 - categorical_accuracy: 0.6601
 2624/60000 [>.............................] - ETA: 1:47 - loss: 1.0306 - categorical_accuracy: 0.6627
 2656/60000 [>.............................] - ETA: 1:47 - loss: 1.0235 - categorical_accuracy: 0.6653
 2688/60000 [>.............................] - ETA: 1:47 - loss: 1.0163 - categorical_accuracy: 0.6674
 2720/60000 [>.............................] - ETA: 1:47 - loss: 1.0105 - categorical_accuracy: 0.6691
 2752/60000 [>.............................] - ETA: 1:47 - loss: 1.0040 - categorical_accuracy: 0.6715
 2784/60000 [>.............................] - ETA: 1:47 - loss: 0.9961 - categorical_accuracy: 0.6742
 2816/60000 [>.............................] - ETA: 1:47 - loss: 0.9899 - categorical_accuracy: 0.6761
 2848/60000 [>.............................] - ETA: 1:47 - loss: 0.9847 - categorical_accuracy: 0.6777
 2880/60000 [>.............................] - ETA: 1:47 - loss: 0.9782 - categorical_accuracy: 0.6799
 2912/60000 [>.............................] - ETA: 1:47 - loss: 0.9723 - categorical_accuracy: 0.6820
 2944/60000 [>.............................] - ETA: 1:47 - loss: 0.9648 - categorical_accuracy: 0.6848
 2976/60000 [>.............................] - ETA: 1:47 - loss: 0.9570 - categorical_accuracy: 0.6875
 3008/60000 [>.............................] - ETA: 1:47 - loss: 0.9541 - categorical_accuracy: 0.6885
 3040/60000 [>.............................] - ETA: 1:47 - loss: 0.9490 - categorical_accuracy: 0.6905
 3072/60000 [>.............................] - ETA: 1:47 - loss: 0.9426 - categorical_accuracy: 0.6930
 3104/60000 [>.............................] - ETA: 1:47 - loss: 0.9347 - categorical_accuracy: 0.6959
 3136/60000 [>.............................] - ETA: 1:47 - loss: 0.9289 - categorical_accuracy: 0.6974
 3168/60000 [>.............................] - ETA: 1:47 - loss: 0.9239 - categorical_accuracy: 0.6995
 3200/60000 [>.............................] - ETA: 1:47 - loss: 0.9167 - categorical_accuracy: 0.7019
 3232/60000 [>.............................] - ETA: 1:47 - loss: 0.9114 - categorical_accuracy: 0.7036
 3264/60000 [>.............................] - ETA: 1:46 - loss: 0.9060 - categorical_accuracy: 0.7053
 3296/60000 [>.............................] - ETA: 1:46 - loss: 0.9034 - categorical_accuracy: 0.7072
 3328/60000 [>.............................] - ETA: 1:46 - loss: 0.9006 - categorical_accuracy: 0.7076
 3360/60000 [>.............................] - ETA: 1:46 - loss: 0.9018 - categorical_accuracy: 0.7071
 3392/60000 [>.............................] - ETA: 1:46 - loss: 0.8972 - categorical_accuracy: 0.7093
 3424/60000 [>.............................] - ETA: 1:46 - loss: 0.8931 - categorical_accuracy: 0.7112
 3456/60000 [>.............................] - ETA: 1:46 - loss: 0.8899 - categorical_accuracy: 0.7130
 3488/60000 [>.............................] - ETA: 1:46 - loss: 0.8847 - categorical_accuracy: 0.7147
 3520/60000 [>.............................] - ETA: 1:45 - loss: 0.8813 - categorical_accuracy: 0.7159
 3552/60000 [>.............................] - ETA: 1:45 - loss: 0.8756 - categorical_accuracy: 0.7179
 3584/60000 [>.............................] - ETA: 1:45 - loss: 0.8707 - categorical_accuracy: 0.7199
 3616/60000 [>.............................] - ETA: 1:45 - loss: 0.8653 - categorical_accuracy: 0.7218
 3648/60000 [>.............................] - ETA: 1:45 - loss: 0.8633 - categorical_accuracy: 0.7226
 3680/60000 [>.............................] - ETA: 1:45 - loss: 0.8637 - categorical_accuracy: 0.7228
 3712/60000 [>.............................] - ETA: 1:45 - loss: 0.8624 - categorical_accuracy: 0.7236
 3744/60000 [>.............................] - ETA: 1:45 - loss: 0.8606 - categorical_accuracy: 0.7233
 3776/60000 [>.............................] - ETA: 1:44 - loss: 0.8569 - categorical_accuracy: 0.7240
 3808/60000 [>.............................] - ETA: 1:44 - loss: 0.8541 - categorical_accuracy: 0.7251
 3840/60000 [>.............................] - ETA: 1:44 - loss: 0.8501 - categorical_accuracy: 0.7263
 3872/60000 [>.............................] - ETA: 1:44 - loss: 0.8445 - categorical_accuracy: 0.7283
 3904/60000 [>.............................] - ETA: 1:44 - loss: 0.8434 - categorical_accuracy: 0.7287
 3936/60000 [>.............................] - ETA: 1:44 - loss: 0.8386 - categorical_accuracy: 0.7304
 3968/60000 [>.............................] - ETA: 1:44 - loss: 0.8337 - categorical_accuracy: 0.7321
 4000/60000 [=>............................] - ETA: 1:44 - loss: 0.8314 - categorical_accuracy: 0.7327
 4032/60000 [=>............................] - ETA: 1:43 - loss: 0.8280 - categorical_accuracy: 0.7341
 4064/60000 [=>............................] - ETA: 1:43 - loss: 0.8247 - categorical_accuracy: 0.7352
 4096/60000 [=>............................] - ETA: 1:43 - loss: 0.8219 - categorical_accuracy: 0.7354
 4128/60000 [=>............................] - ETA: 1:43 - loss: 0.8171 - categorical_accuracy: 0.7372
 4160/60000 [=>............................] - ETA: 1:43 - loss: 0.8123 - categorical_accuracy: 0.7389
 4192/60000 [=>............................] - ETA: 1:43 - loss: 0.8078 - categorical_accuracy: 0.7407
 4224/60000 [=>............................] - ETA: 1:43 - loss: 0.8037 - categorical_accuracy: 0.7420
 4256/60000 [=>............................] - ETA: 1:43 - loss: 0.7995 - categorical_accuracy: 0.7432
 4288/60000 [=>............................] - ETA: 1:42 - loss: 0.7957 - categorical_accuracy: 0.7442
 4320/60000 [=>............................] - ETA: 1:42 - loss: 0.7917 - categorical_accuracy: 0.7454
 4352/60000 [=>............................] - ETA: 1:42 - loss: 0.7871 - categorical_accuracy: 0.7468
 4384/60000 [=>............................] - ETA: 1:42 - loss: 0.7848 - categorical_accuracy: 0.7477
 4416/60000 [=>............................] - ETA: 1:42 - loss: 0.7808 - categorical_accuracy: 0.7489
 4448/60000 [=>............................] - ETA: 1:42 - loss: 0.7787 - categorical_accuracy: 0.7498
 4480/60000 [=>............................] - ETA: 1:42 - loss: 0.7763 - categorical_accuracy: 0.7504
 4512/60000 [=>............................] - ETA: 1:42 - loss: 0.7727 - categorical_accuracy: 0.7516
 4544/60000 [=>............................] - ETA: 1:42 - loss: 0.7697 - categorical_accuracy: 0.7520
 4576/60000 [=>............................] - ETA: 1:42 - loss: 0.7669 - categorical_accuracy: 0.7531
 4608/60000 [=>............................] - ETA: 1:42 - loss: 0.7648 - categorical_accuracy: 0.7533
 4640/60000 [=>............................] - ETA: 1:42 - loss: 0.7617 - categorical_accuracy: 0.7545
 4672/60000 [=>............................] - ETA: 1:42 - loss: 0.7582 - categorical_accuracy: 0.7556
 4704/60000 [=>............................] - ETA: 1:41 - loss: 0.7546 - categorical_accuracy: 0.7568
 4736/60000 [=>............................] - ETA: 1:41 - loss: 0.7536 - categorical_accuracy: 0.7576
 4768/60000 [=>............................] - ETA: 1:41 - loss: 0.7502 - categorical_accuracy: 0.7586
 4800/60000 [=>............................] - ETA: 1:41 - loss: 0.7485 - categorical_accuracy: 0.7596
 4832/60000 [=>............................] - ETA: 1:41 - loss: 0.7446 - categorical_accuracy: 0.7608
 4864/60000 [=>............................] - ETA: 1:41 - loss: 0.7412 - categorical_accuracy: 0.7617
 4896/60000 [=>............................] - ETA: 1:41 - loss: 0.7393 - categorical_accuracy: 0.7627
 4928/60000 [=>............................] - ETA: 1:41 - loss: 0.7368 - categorical_accuracy: 0.7634
 4960/60000 [=>............................] - ETA: 1:41 - loss: 0.7360 - categorical_accuracy: 0.7637
 4992/60000 [=>............................] - ETA: 1:41 - loss: 0.7331 - categorical_accuracy: 0.7646
 5024/60000 [=>............................] - ETA: 1:41 - loss: 0.7308 - categorical_accuracy: 0.7653
 5056/60000 [=>............................] - ETA: 1:41 - loss: 0.7289 - categorical_accuracy: 0.7662
 5088/60000 [=>............................] - ETA: 1:40 - loss: 0.7269 - categorical_accuracy: 0.7669
 5120/60000 [=>............................] - ETA: 1:40 - loss: 0.7245 - categorical_accuracy: 0.7676
 5152/60000 [=>............................] - ETA: 1:40 - loss: 0.7241 - categorical_accuracy: 0.7681
 5184/60000 [=>............................] - ETA: 1:40 - loss: 0.7229 - categorical_accuracy: 0.7685
 5216/60000 [=>............................] - ETA: 1:40 - loss: 0.7208 - categorical_accuracy: 0.7694
 5248/60000 [=>............................] - ETA: 1:40 - loss: 0.7187 - categorical_accuracy: 0.7700
 5280/60000 [=>............................] - ETA: 1:40 - loss: 0.7177 - categorical_accuracy: 0.7699
 5312/60000 [=>............................] - ETA: 1:40 - loss: 0.7156 - categorical_accuracy: 0.7705
 5344/60000 [=>............................] - ETA: 1:40 - loss: 0.7120 - categorical_accuracy: 0.7717
 5376/60000 [=>............................] - ETA: 1:40 - loss: 0.7117 - categorical_accuracy: 0.7723
 5408/60000 [=>............................] - ETA: 1:39 - loss: 0.7096 - categorical_accuracy: 0.7727
 5440/60000 [=>............................] - ETA: 1:39 - loss: 0.7074 - categorical_accuracy: 0.7732
 5472/60000 [=>............................] - ETA: 1:39 - loss: 0.7055 - categorical_accuracy: 0.7739
 5504/60000 [=>............................] - ETA: 1:39 - loss: 0.7029 - categorical_accuracy: 0.7749
 5536/60000 [=>............................] - ETA: 1:39 - loss: 0.7011 - categorical_accuracy: 0.7755
 5568/60000 [=>............................] - ETA: 1:39 - loss: 0.6980 - categorical_accuracy: 0.7766
 5600/60000 [=>............................] - ETA: 1:39 - loss: 0.6946 - categorical_accuracy: 0.7779
 5632/60000 [=>............................] - ETA: 1:39 - loss: 0.6933 - categorical_accuracy: 0.7786
 5664/60000 [=>............................] - ETA: 1:39 - loss: 0.6917 - categorical_accuracy: 0.7795
 5696/60000 [=>............................] - ETA: 1:39 - loss: 0.6923 - categorical_accuracy: 0.7795
 5728/60000 [=>............................] - ETA: 1:39 - loss: 0.6905 - categorical_accuracy: 0.7802
 5760/60000 [=>............................] - ETA: 1:39 - loss: 0.6876 - categorical_accuracy: 0.7811
 5792/60000 [=>............................] - ETA: 1:39 - loss: 0.6852 - categorical_accuracy: 0.7819
 5824/60000 [=>............................] - ETA: 1:39 - loss: 0.6839 - categorical_accuracy: 0.7826
 5856/60000 [=>............................] - ETA: 1:38 - loss: 0.6813 - categorical_accuracy: 0.7833
 5888/60000 [=>............................] - ETA: 1:38 - loss: 0.6793 - categorical_accuracy: 0.7836
 5920/60000 [=>............................] - ETA: 1:38 - loss: 0.6769 - categorical_accuracy: 0.7846
 5952/60000 [=>............................] - ETA: 1:38 - loss: 0.6747 - categorical_accuracy: 0.7853
 5984/60000 [=>............................] - ETA: 1:38 - loss: 0.6724 - categorical_accuracy: 0.7861
 6016/60000 [==>...........................] - ETA: 1:38 - loss: 0.6719 - categorical_accuracy: 0.7864
 6048/60000 [==>...........................] - ETA: 1:38 - loss: 0.6705 - categorical_accuracy: 0.7869
 6080/60000 [==>...........................] - ETA: 1:38 - loss: 0.6694 - categorical_accuracy: 0.7872
 6112/60000 [==>...........................] - ETA: 1:38 - loss: 0.6671 - categorical_accuracy: 0.7880
 6144/60000 [==>...........................] - ETA: 1:38 - loss: 0.6642 - categorical_accuracy: 0.7889
 6176/60000 [==>...........................] - ETA: 1:37 - loss: 0.6620 - categorical_accuracy: 0.7897
 6208/60000 [==>...........................] - ETA: 1:37 - loss: 0.6595 - categorical_accuracy: 0.7904
 6240/60000 [==>...........................] - ETA: 1:37 - loss: 0.6576 - categorical_accuracy: 0.7909
 6272/60000 [==>...........................] - ETA: 1:37 - loss: 0.6547 - categorical_accuracy: 0.7919
 6304/60000 [==>...........................] - ETA: 1:37 - loss: 0.6530 - categorical_accuracy: 0.7925
 6336/60000 [==>...........................] - ETA: 1:37 - loss: 0.6508 - categorical_accuracy: 0.7932
 6368/60000 [==>...........................] - ETA: 1:37 - loss: 0.6483 - categorical_accuracy: 0.7940
 6400/60000 [==>...........................] - ETA: 1:37 - loss: 0.6464 - categorical_accuracy: 0.7945
 6432/60000 [==>...........................] - ETA: 1:37 - loss: 0.6447 - categorical_accuracy: 0.7952
 6464/60000 [==>...........................] - ETA: 1:37 - loss: 0.6425 - categorical_accuracy: 0.7959
 6496/60000 [==>...........................] - ETA: 1:37 - loss: 0.6406 - categorical_accuracy: 0.7966
 6528/60000 [==>...........................] - ETA: 1:37 - loss: 0.6379 - categorical_accuracy: 0.7976
 6560/60000 [==>...........................] - ETA: 1:36 - loss: 0.6362 - categorical_accuracy: 0.7983
 6592/60000 [==>...........................] - ETA: 1:36 - loss: 0.6344 - categorical_accuracy: 0.7990
 6624/60000 [==>...........................] - ETA: 1:36 - loss: 0.6333 - categorical_accuracy: 0.7994
 6656/60000 [==>...........................] - ETA: 1:36 - loss: 0.6315 - categorical_accuracy: 0.7999
 6688/60000 [==>...........................] - ETA: 1:36 - loss: 0.6306 - categorical_accuracy: 0.7999
 6720/60000 [==>...........................] - ETA: 1:36 - loss: 0.6287 - categorical_accuracy: 0.8006
 6752/60000 [==>...........................] - ETA: 1:36 - loss: 0.6268 - categorical_accuracy: 0.8011
 6784/60000 [==>...........................] - ETA: 1:36 - loss: 0.6261 - categorical_accuracy: 0.8013
 6816/60000 [==>...........................] - ETA: 1:36 - loss: 0.6239 - categorical_accuracy: 0.8021
 6848/60000 [==>...........................] - ETA: 1:36 - loss: 0.6224 - categorical_accuracy: 0.8024
 6880/60000 [==>...........................] - ETA: 1:36 - loss: 0.6226 - categorical_accuracy: 0.8026
 6912/60000 [==>...........................] - ETA: 1:36 - loss: 0.6205 - categorical_accuracy: 0.8032
 6944/60000 [==>...........................] - ETA: 1:36 - loss: 0.6177 - categorical_accuracy: 0.8041
 6976/60000 [==>...........................] - ETA: 1:36 - loss: 0.6156 - categorical_accuracy: 0.8048
 7008/60000 [==>...........................] - ETA: 1:35 - loss: 0.6136 - categorical_accuracy: 0.8054
 7040/60000 [==>...........................] - ETA: 1:35 - loss: 0.6117 - categorical_accuracy: 0.8061
 7072/60000 [==>...........................] - ETA: 1:35 - loss: 0.6094 - categorical_accuracy: 0.8070
 7104/60000 [==>...........................] - ETA: 1:35 - loss: 0.6075 - categorical_accuracy: 0.8076
 7136/60000 [==>...........................] - ETA: 1:35 - loss: 0.6070 - categorical_accuracy: 0.8079
 7168/60000 [==>...........................] - ETA: 1:35 - loss: 0.6052 - categorical_accuracy: 0.8086
 7200/60000 [==>...........................] - ETA: 1:35 - loss: 0.6037 - categorical_accuracy: 0.8092
 7232/60000 [==>...........................] - ETA: 1:35 - loss: 0.6020 - categorical_accuracy: 0.8097
 7264/60000 [==>...........................] - ETA: 1:35 - loss: 0.6007 - categorical_accuracy: 0.8100
 7296/60000 [==>...........................] - ETA: 1:35 - loss: 0.5985 - categorical_accuracy: 0.8107
 7328/60000 [==>...........................] - ETA: 1:35 - loss: 0.5964 - categorical_accuracy: 0.8114
 7360/60000 [==>...........................] - ETA: 1:35 - loss: 0.5946 - categorical_accuracy: 0.8120
 7392/60000 [==>...........................] - ETA: 1:35 - loss: 0.5931 - categorical_accuracy: 0.8125
 7424/60000 [==>...........................] - ETA: 1:35 - loss: 0.5926 - categorical_accuracy: 0.8128
 7456/60000 [==>...........................] - ETA: 1:35 - loss: 0.5910 - categorical_accuracy: 0.8133
 7488/60000 [==>...........................] - ETA: 1:34 - loss: 0.5888 - categorical_accuracy: 0.8141
 7520/60000 [==>...........................] - ETA: 1:34 - loss: 0.5878 - categorical_accuracy: 0.8146
 7552/60000 [==>...........................] - ETA: 1:34 - loss: 0.5866 - categorical_accuracy: 0.8148
 7584/60000 [==>...........................] - ETA: 1:34 - loss: 0.5851 - categorical_accuracy: 0.8153
 7616/60000 [==>...........................] - ETA: 1:34 - loss: 0.5838 - categorical_accuracy: 0.8157
 7648/60000 [==>...........................] - ETA: 1:34 - loss: 0.5828 - categorical_accuracy: 0.8159
 7680/60000 [==>...........................] - ETA: 1:34 - loss: 0.5811 - categorical_accuracy: 0.8163
 7712/60000 [==>...........................] - ETA: 1:34 - loss: 0.5797 - categorical_accuracy: 0.8169
 7744/60000 [==>...........................] - ETA: 1:34 - loss: 0.5777 - categorical_accuracy: 0.8175
 7776/60000 [==>...........................] - ETA: 1:34 - loss: 0.5774 - categorical_accuracy: 0.8178
 7808/60000 [==>...........................] - ETA: 1:34 - loss: 0.5766 - categorical_accuracy: 0.8180
 7840/60000 [==>...........................] - ETA: 1:34 - loss: 0.5760 - categorical_accuracy: 0.8182
 7872/60000 [==>...........................] - ETA: 1:34 - loss: 0.5744 - categorical_accuracy: 0.8187
 7904/60000 [==>...........................] - ETA: 1:34 - loss: 0.5740 - categorical_accuracy: 0.8190
 7936/60000 [==>...........................] - ETA: 1:33 - loss: 0.5720 - categorical_accuracy: 0.8196
 7968/60000 [==>...........................] - ETA: 1:33 - loss: 0.5709 - categorical_accuracy: 0.8199
 8000/60000 [===>..........................] - ETA: 1:33 - loss: 0.5707 - categorical_accuracy: 0.8201
 8032/60000 [===>..........................] - ETA: 1:33 - loss: 0.5690 - categorical_accuracy: 0.8207
 8064/60000 [===>..........................] - ETA: 1:33 - loss: 0.5674 - categorical_accuracy: 0.8212
 8096/60000 [===>..........................] - ETA: 1:33 - loss: 0.5658 - categorical_accuracy: 0.8216
 8128/60000 [===>..........................] - ETA: 1:33 - loss: 0.5645 - categorical_accuracy: 0.8220
 8160/60000 [===>..........................] - ETA: 1:33 - loss: 0.5626 - categorical_accuracy: 0.8225
 8192/60000 [===>..........................] - ETA: 1:33 - loss: 0.5617 - categorical_accuracy: 0.8228
 8224/60000 [===>..........................] - ETA: 1:33 - loss: 0.5600 - categorical_accuracy: 0.8234
 8256/60000 [===>..........................] - ETA: 1:33 - loss: 0.5583 - categorical_accuracy: 0.8241
 8288/60000 [===>..........................] - ETA: 1:33 - loss: 0.5569 - categorical_accuracy: 0.8246
 8320/60000 [===>..........................] - ETA: 1:33 - loss: 0.5556 - categorical_accuracy: 0.8250
 8352/60000 [===>..........................] - ETA: 1:33 - loss: 0.5545 - categorical_accuracy: 0.8254
 8384/60000 [===>..........................] - ETA: 1:33 - loss: 0.5543 - categorical_accuracy: 0.8255
 8416/60000 [===>..........................] - ETA: 1:33 - loss: 0.5530 - categorical_accuracy: 0.8259
 8448/60000 [===>..........................] - ETA: 1:32 - loss: 0.5524 - categorical_accuracy: 0.8262
 8480/60000 [===>..........................] - ETA: 1:32 - loss: 0.5515 - categorical_accuracy: 0.8264
 8512/60000 [===>..........................] - ETA: 1:32 - loss: 0.5515 - categorical_accuracy: 0.8265
 8544/60000 [===>..........................] - ETA: 1:32 - loss: 0.5506 - categorical_accuracy: 0.8268
 8576/60000 [===>..........................] - ETA: 1:32 - loss: 0.5496 - categorical_accuracy: 0.8271
 8608/60000 [===>..........................] - ETA: 1:32 - loss: 0.5477 - categorical_accuracy: 0.8277
 8640/60000 [===>..........................] - ETA: 1:32 - loss: 0.5464 - categorical_accuracy: 0.8281
 8672/60000 [===>..........................] - ETA: 1:32 - loss: 0.5449 - categorical_accuracy: 0.8285
 8704/60000 [===>..........................] - ETA: 1:32 - loss: 0.5436 - categorical_accuracy: 0.8289
 8736/60000 [===>..........................] - ETA: 1:32 - loss: 0.5421 - categorical_accuracy: 0.8293
 8768/60000 [===>..........................] - ETA: 1:32 - loss: 0.5404 - categorical_accuracy: 0.8299
 8800/60000 [===>..........................] - ETA: 1:32 - loss: 0.5409 - categorical_accuracy: 0.8300
 8832/60000 [===>..........................] - ETA: 1:32 - loss: 0.5400 - categorical_accuracy: 0.8303
 8864/60000 [===>..........................] - ETA: 1:32 - loss: 0.5391 - categorical_accuracy: 0.8306
 8896/60000 [===>..........................] - ETA: 1:32 - loss: 0.5380 - categorical_accuracy: 0.8308
 8928/60000 [===>..........................] - ETA: 1:31 - loss: 0.5366 - categorical_accuracy: 0.8312
 8960/60000 [===>..........................] - ETA: 1:31 - loss: 0.5358 - categorical_accuracy: 0.8315
 8992/60000 [===>..........................] - ETA: 1:31 - loss: 0.5353 - categorical_accuracy: 0.8317
 9024/60000 [===>..........................] - ETA: 1:31 - loss: 0.5340 - categorical_accuracy: 0.8320
 9056/60000 [===>..........................] - ETA: 1:31 - loss: 0.5328 - categorical_accuracy: 0.8324
 9088/60000 [===>..........................] - ETA: 1:31 - loss: 0.5312 - categorical_accuracy: 0.8330
 9120/60000 [===>..........................] - ETA: 1:31 - loss: 0.5305 - categorical_accuracy: 0.8332
 9152/60000 [===>..........................] - ETA: 1:31 - loss: 0.5291 - categorical_accuracy: 0.8338
 9184/60000 [===>..........................] - ETA: 1:31 - loss: 0.5277 - categorical_accuracy: 0.8342
 9216/60000 [===>..........................] - ETA: 1:31 - loss: 0.5265 - categorical_accuracy: 0.8345
 9248/60000 [===>..........................] - ETA: 1:31 - loss: 0.5254 - categorical_accuracy: 0.8347
 9280/60000 [===>..........................] - ETA: 1:31 - loss: 0.5241 - categorical_accuracy: 0.8351
 9312/60000 [===>..........................] - ETA: 1:31 - loss: 0.5228 - categorical_accuracy: 0.8355
 9344/60000 [===>..........................] - ETA: 1:31 - loss: 0.5217 - categorical_accuracy: 0.8358
 9376/60000 [===>..........................] - ETA: 1:31 - loss: 0.5208 - categorical_accuracy: 0.8361
 9408/60000 [===>..........................] - ETA: 1:30 - loss: 0.5201 - categorical_accuracy: 0.8364
 9440/60000 [===>..........................] - ETA: 1:30 - loss: 0.5186 - categorical_accuracy: 0.8370
 9472/60000 [===>..........................] - ETA: 1:30 - loss: 0.5172 - categorical_accuracy: 0.8374
 9504/60000 [===>..........................] - ETA: 1:30 - loss: 0.5160 - categorical_accuracy: 0.8376
 9536/60000 [===>..........................] - ETA: 1:30 - loss: 0.5147 - categorical_accuracy: 0.8380
 9568/60000 [===>..........................] - ETA: 1:30 - loss: 0.5132 - categorical_accuracy: 0.8385
 9600/60000 [===>..........................] - ETA: 1:30 - loss: 0.5119 - categorical_accuracy: 0.8389
 9632/60000 [===>..........................] - ETA: 1:30 - loss: 0.5106 - categorical_accuracy: 0.8393
 9664/60000 [===>..........................] - ETA: 1:30 - loss: 0.5091 - categorical_accuracy: 0.8396
 9696/60000 [===>..........................] - ETA: 1:30 - loss: 0.5079 - categorical_accuracy: 0.8398
 9728/60000 [===>..........................] - ETA: 1:30 - loss: 0.5066 - categorical_accuracy: 0.8402
 9760/60000 [===>..........................] - ETA: 1:30 - loss: 0.5052 - categorical_accuracy: 0.8406
 9792/60000 [===>..........................] - ETA: 1:30 - loss: 0.5040 - categorical_accuracy: 0.8409
 9824/60000 [===>..........................] - ETA: 1:30 - loss: 0.5026 - categorical_accuracy: 0.8413
 9856/60000 [===>..........................] - ETA: 1:30 - loss: 0.5031 - categorical_accuracy: 0.8416
 9888/60000 [===>..........................] - ETA: 1:29 - loss: 0.5025 - categorical_accuracy: 0.8419
 9920/60000 [===>..........................] - ETA: 1:29 - loss: 0.5015 - categorical_accuracy: 0.8423
 9952/60000 [===>..........................] - ETA: 1:29 - loss: 0.5010 - categorical_accuracy: 0.8425
 9984/60000 [===>..........................] - ETA: 1:29 - loss: 0.5003 - categorical_accuracy: 0.8426
10016/60000 [====>.........................] - ETA: 1:29 - loss: 0.4995 - categorical_accuracy: 0.8430
10048/60000 [====>.........................] - ETA: 1:29 - loss: 0.4982 - categorical_accuracy: 0.8434
10080/60000 [====>.........................] - ETA: 1:29 - loss: 0.4973 - categorical_accuracy: 0.8435
10112/60000 [====>.........................] - ETA: 1:29 - loss: 0.4962 - categorical_accuracy: 0.8438
10144/60000 [====>.........................] - ETA: 1:29 - loss: 0.4966 - categorical_accuracy: 0.8439
10176/60000 [====>.........................] - ETA: 1:29 - loss: 0.4954 - categorical_accuracy: 0.8443
10208/60000 [====>.........................] - ETA: 1:29 - loss: 0.4950 - categorical_accuracy: 0.8443
10240/60000 [====>.........................] - ETA: 1:29 - loss: 0.4938 - categorical_accuracy: 0.8446
10272/60000 [====>.........................] - ETA: 1:29 - loss: 0.4931 - categorical_accuracy: 0.8447
10304/60000 [====>.........................] - ETA: 1:29 - loss: 0.4920 - categorical_accuracy: 0.8451
10336/60000 [====>.........................] - ETA: 1:29 - loss: 0.4913 - categorical_accuracy: 0.8454
10368/60000 [====>.........................] - ETA: 1:28 - loss: 0.4906 - categorical_accuracy: 0.8456
10400/60000 [====>.........................] - ETA: 1:28 - loss: 0.4892 - categorical_accuracy: 0.8461
10432/60000 [====>.........................] - ETA: 1:28 - loss: 0.4883 - categorical_accuracy: 0.8464
10464/60000 [====>.........................] - ETA: 1:28 - loss: 0.4873 - categorical_accuracy: 0.8467
10496/60000 [====>.........................] - ETA: 1:28 - loss: 0.4859 - categorical_accuracy: 0.8472
10528/60000 [====>.........................] - ETA: 1:28 - loss: 0.4858 - categorical_accuracy: 0.8473
10560/60000 [====>.........................] - ETA: 1:28 - loss: 0.4846 - categorical_accuracy: 0.8476
10592/60000 [====>.........................] - ETA: 1:28 - loss: 0.4838 - categorical_accuracy: 0.8480
10624/60000 [====>.........................] - ETA: 1:28 - loss: 0.4828 - categorical_accuracy: 0.8483
10656/60000 [====>.........................] - ETA: 1:28 - loss: 0.4816 - categorical_accuracy: 0.8486
10688/60000 [====>.........................] - ETA: 1:28 - loss: 0.4806 - categorical_accuracy: 0.8489
10720/60000 [====>.........................] - ETA: 1:28 - loss: 0.4803 - categorical_accuracy: 0.8492
10752/60000 [====>.........................] - ETA: 1:28 - loss: 0.4791 - categorical_accuracy: 0.8495
10784/60000 [====>.........................] - ETA: 1:28 - loss: 0.4779 - categorical_accuracy: 0.8499
10816/60000 [====>.........................] - ETA: 1:28 - loss: 0.4787 - categorical_accuracy: 0.8500
10848/60000 [====>.........................] - ETA: 1:28 - loss: 0.4781 - categorical_accuracy: 0.8503
10880/60000 [====>.........................] - ETA: 1:27 - loss: 0.4774 - categorical_accuracy: 0.8506
10912/60000 [====>.........................] - ETA: 1:27 - loss: 0.4767 - categorical_accuracy: 0.8508
10944/60000 [====>.........................] - ETA: 1:27 - loss: 0.4764 - categorical_accuracy: 0.8509
10976/60000 [====>.........................] - ETA: 1:27 - loss: 0.4755 - categorical_accuracy: 0.8511
11008/60000 [====>.........................] - ETA: 1:27 - loss: 0.4747 - categorical_accuracy: 0.8512
11040/60000 [====>.........................] - ETA: 1:27 - loss: 0.4735 - categorical_accuracy: 0.8516
11072/60000 [====>.........................] - ETA: 1:27 - loss: 0.4728 - categorical_accuracy: 0.8519
11104/60000 [====>.........................] - ETA: 1:27 - loss: 0.4723 - categorical_accuracy: 0.8520
11136/60000 [====>.........................] - ETA: 1:27 - loss: 0.4710 - categorical_accuracy: 0.8525
11168/60000 [====>.........................] - ETA: 1:27 - loss: 0.4703 - categorical_accuracy: 0.8527
11200/60000 [====>.........................] - ETA: 1:27 - loss: 0.4693 - categorical_accuracy: 0.8530
11232/60000 [====>.........................] - ETA: 1:27 - loss: 0.4681 - categorical_accuracy: 0.8534
11264/60000 [====>.........................] - ETA: 1:27 - loss: 0.4678 - categorical_accuracy: 0.8536
11296/60000 [====>.........................] - ETA: 1:27 - loss: 0.4667 - categorical_accuracy: 0.8540
11328/60000 [====>.........................] - ETA: 1:27 - loss: 0.4664 - categorical_accuracy: 0.8542
11360/60000 [====>.........................] - ETA: 1:26 - loss: 0.4657 - categorical_accuracy: 0.8545
11392/60000 [====>.........................] - ETA: 1:26 - loss: 0.4660 - categorical_accuracy: 0.8546
11424/60000 [====>.........................] - ETA: 1:26 - loss: 0.4650 - categorical_accuracy: 0.8549
11456/60000 [====>.........................] - ETA: 1:26 - loss: 0.4646 - categorical_accuracy: 0.8550
11488/60000 [====>.........................] - ETA: 1:26 - loss: 0.4645 - categorical_accuracy: 0.8552
11520/60000 [====>.........................] - ETA: 1:26 - loss: 0.4638 - categorical_accuracy: 0.8554
11552/60000 [====>.........................] - ETA: 1:26 - loss: 0.4628 - categorical_accuracy: 0.8556
11584/60000 [====>.........................] - ETA: 1:26 - loss: 0.4619 - categorical_accuracy: 0.8559
11616/60000 [====>.........................] - ETA: 1:26 - loss: 0.4616 - categorical_accuracy: 0.8560
11648/60000 [====>.........................] - ETA: 1:26 - loss: 0.4613 - categorical_accuracy: 0.8562
11680/60000 [====>.........................] - ETA: 1:26 - loss: 0.4603 - categorical_accuracy: 0.8566
11712/60000 [====>.........................] - ETA: 1:26 - loss: 0.4595 - categorical_accuracy: 0.8569
11744/60000 [====>.........................] - ETA: 1:26 - loss: 0.4587 - categorical_accuracy: 0.8571
11776/60000 [====>.........................] - ETA: 1:26 - loss: 0.4581 - categorical_accuracy: 0.8574
11808/60000 [====>.........................] - ETA: 1:26 - loss: 0.4571 - categorical_accuracy: 0.8576
11840/60000 [====>.........................] - ETA: 1:26 - loss: 0.4570 - categorical_accuracy: 0.8579
11872/60000 [====>.........................] - ETA: 1:25 - loss: 0.4560 - categorical_accuracy: 0.8582
11904/60000 [====>.........................] - ETA: 1:25 - loss: 0.4552 - categorical_accuracy: 0.8585
11936/60000 [====>.........................] - ETA: 1:25 - loss: 0.4545 - categorical_accuracy: 0.8587
11968/60000 [====>.........................] - ETA: 1:25 - loss: 0.4535 - categorical_accuracy: 0.8590
12000/60000 [=====>........................] - ETA: 1:25 - loss: 0.4530 - categorical_accuracy: 0.8592
12032/60000 [=====>........................] - ETA: 1:25 - loss: 0.4520 - categorical_accuracy: 0.8595
12064/60000 [=====>........................] - ETA: 1:25 - loss: 0.4510 - categorical_accuracy: 0.8597
12096/60000 [=====>........................] - ETA: 1:25 - loss: 0.4503 - categorical_accuracy: 0.8600
12128/60000 [=====>........................] - ETA: 1:25 - loss: 0.4497 - categorical_accuracy: 0.8603
12160/60000 [=====>........................] - ETA: 1:25 - loss: 0.4494 - categorical_accuracy: 0.8604
12192/60000 [=====>........................] - ETA: 1:25 - loss: 0.4488 - categorical_accuracy: 0.8606
12224/60000 [=====>........................] - ETA: 1:25 - loss: 0.4479 - categorical_accuracy: 0.8609
12256/60000 [=====>........................] - ETA: 1:25 - loss: 0.4469 - categorical_accuracy: 0.8613
12288/60000 [=====>........................] - ETA: 1:25 - loss: 0.4466 - categorical_accuracy: 0.8614
12320/60000 [=====>........................] - ETA: 1:25 - loss: 0.4457 - categorical_accuracy: 0.8617
12352/60000 [=====>........................] - ETA: 1:24 - loss: 0.4448 - categorical_accuracy: 0.8620
12384/60000 [=====>........................] - ETA: 1:24 - loss: 0.4440 - categorical_accuracy: 0.8622
12416/60000 [=====>........................] - ETA: 1:24 - loss: 0.4437 - categorical_accuracy: 0.8622
12448/60000 [=====>........................] - ETA: 1:24 - loss: 0.4427 - categorical_accuracy: 0.8625
12480/60000 [=====>........................] - ETA: 1:24 - loss: 0.4422 - categorical_accuracy: 0.8627
12512/60000 [=====>........................] - ETA: 1:24 - loss: 0.4415 - categorical_accuracy: 0.8629
12544/60000 [=====>........................] - ETA: 1:24 - loss: 0.4408 - categorical_accuracy: 0.8630
12576/60000 [=====>........................] - ETA: 1:24 - loss: 0.4406 - categorical_accuracy: 0.8630
12608/60000 [=====>........................] - ETA: 1:24 - loss: 0.4404 - categorical_accuracy: 0.8630
12640/60000 [=====>........................] - ETA: 1:24 - loss: 0.4396 - categorical_accuracy: 0.8633
12672/60000 [=====>........................] - ETA: 1:24 - loss: 0.4388 - categorical_accuracy: 0.8636
12704/60000 [=====>........................] - ETA: 1:24 - loss: 0.4381 - categorical_accuracy: 0.8638
12736/60000 [=====>........................] - ETA: 1:24 - loss: 0.4372 - categorical_accuracy: 0.8642
12768/60000 [=====>........................] - ETA: 1:24 - loss: 0.4367 - categorical_accuracy: 0.8643
12800/60000 [=====>........................] - ETA: 1:24 - loss: 0.4370 - categorical_accuracy: 0.8644
12832/60000 [=====>........................] - ETA: 1:23 - loss: 0.4360 - categorical_accuracy: 0.8647
12864/60000 [=====>........................] - ETA: 1:23 - loss: 0.4352 - categorical_accuracy: 0.8650
12896/60000 [=====>........................] - ETA: 1:23 - loss: 0.4346 - categorical_accuracy: 0.8652
12928/60000 [=====>........................] - ETA: 1:23 - loss: 0.4340 - categorical_accuracy: 0.8652
12960/60000 [=====>........................] - ETA: 1:23 - loss: 0.4331 - categorical_accuracy: 0.8655
12992/60000 [=====>........................] - ETA: 1:23 - loss: 0.4323 - categorical_accuracy: 0.8658
13056/60000 [=====>........................] - ETA: 1:23 - loss: 0.4311 - categorical_accuracy: 0.8662
13088/60000 [=====>........................] - ETA: 1:23 - loss: 0.4301 - categorical_accuracy: 0.8665
13120/60000 [=====>........................] - ETA: 1:23 - loss: 0.4293 - categorical_accuracy: 0.8668
13152/60000 [=====>........................] - ETA: 1:23 - loss: 0.4285 - categorical_accuracy: 0.8671
13184/60000 [=====>........................] - ETA: 1:23 - loss: 0.4287 - categorical_accuracy: 0.8672
13216/60000 [=====>........................] - ETA: 1:23 - loss: 0.4280 - categorical_accuracy: 0.8674
13248/60000 [=====>........................] - ETA: 1:23 - loss: 0.4276 - categorical_accuracy: 0.8677
13280/60000 [=====>........................] - ETA: 1:23 - loss: 0.4269 - categorical_accuracy: 0.8679
13312/60000 [=====>........................] - ETA: 1:22 - loss: 0.4262 - categorical_accuracy: 0.8682
13344/60000 [=====>........................] - ETA: 1:22 - loss: 0.4255 - categorical_accuracy: 0.8683
13376/60000 [=====>........................] - ETA: 1:22 - loss: 0.4250 - categorical_accuracy: 0.8686
13408/60000 [=====>........................] - ETA: 1:22 - loss: 0.4242 - categorical_accuracy: 0.8688
13440/60000 [=====>........................] - ETA: 1:22 - loss: 0.4241 - categorical_accuracy: 0.8690
13472/60000 [=====>........................] - ETA: 1:22 - loss: 0.4237 - categorical_accuracy: 0.8691
13504/60000 [=====>........................] - ETA: 1:22 - loss: 0.4228 - categorical_accuracy: 0.8694
13536/60000 [=====>........................] - ETA: 1:22 - loss: 0.4222 - categorical_accuracy: 0.8696
13568/60000 [=====>........................] - ETA: 1:22 - loss: 0.4217 - categorical_accuracy: 0.8697
13600/60000 [=====>........................] - ETA: 1:22 - loss: 0.4210 - categorical_accuracy: 0.8699
13632/60000 [=====>........................] - ETA: 1:22 - loss: 0.4204 - categorical_accuracy: 0.8701
13664/60000 [=====>........................] - ETA: 1:22 - loss: 0.4198 - categorical_accuracy: 0.8703
13696/60000 [=====>........................] - ETA: 1:22 - loss: 0.4190 - categorical_accuracy: 0.8705
13728/60000 [=====>........................] - ETA: 1:22 - loss: 0.4182 - categorical_accuracy: 0.8708
13760/60000 [=====>........................] - ETA: 1:22 - loss: 0.4174 - categorical_accuracy: 0.8711
13792/60000 [=====>........................] - ETA: 1:22 - loss: 0.4179 - categorical_accuracy: 0.8713
13824/60000 [=====>........................] - ETA: 1:22 - loss: 0.4172 - categorical_accuracy: 0.8715
13856/60000 [=====>........................] - ETA: 1:21 - loss: 0.4164 - categorical_accuracy: 0.8718
13888/60000 [=====>........................] - ETA: 1:21 - loss: 0.4163 - categorical_accuracy: 0.8717
13920/60000 [=====>........................] - ETA: 1:21 - loss: 0.4156 - categorical_accuracy: 0.8719
13952/60000 [=====>........................] - ETA: 1:21 - loss: 0.4147 - categorical_accuracy: 0.8722
13984/60000 [=====>........................] - ETA: 1:21 - loss: 0.4145 - categorical_accuracy: 0.8724
14016/60000 [======>.......................] - ETA: 1:21 - loss: 0.4138 - categorical_accuracy: 0.8726
14048/60000 [======>.......................] - ETA: 1:21 - loss: 0.4130 - categorical_accuracy: 0.8729
14080/60000 [======>.......................] - ETA: 1:21 - loss: 0.4123 - categorical_accuracy: 0.8731
14112/60000 [======>.......................] - ETA: 1:21 - loss: 0.4115 - categorical_accuracy: 0.8734
14144/60000 [======>.......................] - ETA: 1:21 - loss: 0.4108 - categorical_accuracy: 0.8736
14176/60000 [======>.......................] - ETA: 1:21 - loss: 0.4102 - categorical_accuracy: 0.8738
14208/60000 [======>.......................] - ETA: 1:21 - loss: 0.4102 - categorical_accuracy: 0.8739
14240/60000 [======>.......................] - ETA: 1:21 - loss: 0.4097 - categorical_accuracy: 0.8740
14272/60000 [======>.......................] - ETA: 1:21 - loss: 0.4096 - categorical_accuracy: 0.8739
14304/60000 [======>.......................] - ETA: 1:21 - loss: 0.4087 - categorical_accuracy: 0.8742
14336/60000 [======>.......................] - ETA: 1:21 - loss: 0.4081 - categorical_accuracy: 0.8744
14368/60000 [======>.......................] - ETA: 1:20 - loss: 0.4074 - categorical_accuracy: 0.8747
14400/60000 [======>.......................] - ETA: 1:20 - loss: 0.4072 - categorical_accuracy: 0.8748
14432/60000 [======>.......................] - ETA: 1:20 - loss: 0.4067 - categorical_accuracy: 0.8750
14464/60000 [======>.......................] - ETA: 1:20 - loss: 0.4064 - categorical_accuracy: 0.8751
14496/60000 [======>.......................] - ETA: 1:20 - loss: 0.4057 - categorical_accuracy: 0.8753
14528/60000 [======>.......................] - ETA: 1:20 - loss: 0.4056 - categorical_accuracy: 0.8752
14560/60000 [======>.......................] - ETA: 1:20 - loss: 0.4054 - categorical_accuracy: 0.8753
14592/60000 [======>.......................] - ETA: 1:20 - loss: 0.4046 - categorical_accuracy: 0.8756
14624/60000 [======>.......................] - ETA: 1:20 - loss: 0.4046 - categorical_accuracy: 0.8755
14656/60000 [======>.......................] - ETA: 1:20 - loss: 0.4043 - categorical_accuracy: 0.8756
14688/60000 [======>.......................] - ETA: 1:20 - loss: 0.4041 - categorical_accuracy: 0.8755
14720/60000 [======>.......................] - ETA: 1:20 - loss: 0.4037 - categorical_accuracy: 0.8757
14752/60000 [======>.......................] - ETA: 1:20 - loss: 0.4032 - categorical_accuracy: 0.8758
14784/60000 [======>.......................] - ETA: 1:20 - loss: 0.4025 - categorical_accuracy: 0.8761
14816/60000 [======>.......................] - ETA: 1:20 - loss: 0.4019 - categorical_accuracy: 0.8763
14848/60000 [======>.......................] - ETA: 1:20 - loss: 0.4013 - categorical_accuracy: 0.8764
14880/60000 [======>.......................] - ETA: 1:20 - loss: 0.4011 - categorical_accuracy: 0.8765
14912/60000 [======>.......................] - ETA: 1:19 - loss: 0.4007 - categorical_accuracy: 0.8766
14944/60000 [======>.......................] - ETA: 1:19 - loss: 0.4003 - categorical_accuracy: 0.8768
14976/60000 [======>.......................] - ETA: 1:19 - loss: 0.3998 - categorical_accuracy: 0.8769
15008/60000 [======>.......................] - ETA: 1:19 - loss: 0.3991 - categorical_accuracy: 0.8771
15040/60000 [======>.......................] - ETA: 1:19 - loss: 0.3986 - categorical_accuracy: 0.8773
15072/60000 [======>.......................] - ETA: 1:19 - loss: 0.3978 - categorical_accuracy: 0.8776
15104/60000 [======>.......................] - ETA: 1:19 - loss: 0.3970 - categorical_accuracy: 0.8778
15136/60000 [======>.......................] - ETA: 1:19 - loss: 0.3971 - categorical_accuracy: 0.8778
15168/60000 [======>.......................] - ETA: 1:19 - loss: 0.3965 - categorical_accuracy: 0.8780
15200/60000 [======>.......................] - ETA: 1:19 - loss: 0.3960 - categorical_accuracy: 0.8782
15232/60000 [======>.......................] - ETA: 1:19 - loss: 0.3954 - categorical_accuracy: 0.8783
15264/60000 [======>.......................] - ETA: 1:19 - loss: 0.3953 - categorical_accuracy: 0.8784
15296/60000 [======>.......................] - ETA: 1:19 - loss: 0.3948 - categorical_accuracy: 0.8785
15328/60000 [======>.......................] - ETA: 1:19 - loss: 0.3943 - categorical_accuracy: 0.8786
15360/60000 [======>.......................] - ETA: 1:19 - loss: 0.3936 - categorical_accuracy: 0.8788
15392/60000 [======>.......................] - ETA: 1:19 - loss: 0.3932 - categorical_accuracy: 0.8790
15424/60000 [======>.......................] - ETA: 1:18 - loss: 0.3924 - categorical_accuracy: 0.8792
15456/60000 [======>.......................] - ETA: 1:18 - loss: 0.3919 - categorical_accuracy: 0.8793
15488/60000 [======>.......................] - ETA: 1:18 - loss: 0.3913 - categorical_accuracy: 0.8795
15520/60000 [======>.......................] - ETA: 1:18 - loss: 0.3907 - categorical_accuracy: 0.8796
15552/60000 [======>.......................] - ETA: 1:18 - loss: 0.3899 - categorical_accuracy: 0.8799
15584/60000 [======>.......................] - ETA: 1:18 - loss: 0.3895 - categorical_accuracy: 0.8800
15616/60000 [======>.......................] - ETA: 1:18 - loss: 0.3894 - categorical_accuracy: 0.8801
15648/60000 [======>.......................] - ETA: 1:18 - loss: 0.3887 - categorical_accuracy: 0.8803
15680/60000 [======>.......................] - ETA: 1:18 - loss: 0.3882 - categorical_accuracy: 0.8805
15712/60000 [======>.......................] - ETA: 1:18 - loss: 0.3878 - categorical_accuracy: 0.8806
15744/60000 [======>.......................] - ETA: 1:18 - loss: 0.3872 - categorical_accuracy: 0.8808
15776/60000 [======>.......................] - ETA: 1:18 - loss: 0.3867 - categorical_accuracy: 0.8809
15808/60000 [======>.......................] - ETA: 1:18 - loss: 0.3860 - categorical_accuracy: 0.8811
15840/60000 [======>.......................] - ETA: 1:18 - loss: 0.3856 - categorical_accuracy: 0.8813
15872/60000 [======>.......................] - ETA: 1:18 - loss: 0.3849 - categorical_accuracy: 0.8815
15904/60000 [======>.......................] - ETA: 1:18 - loss: 0.3842 - categorical_accuracy: 0.8817
15936/60000 [======>.......................] - ETA: 1:18 - loss: 0.3835 - categorical_accuracy: 0.8820
15968/60000 [======>.......................] - ETA: 1:17 - loss: 0.3833 - categorical_accuracy: 0.8821
16000/60000 [=======>......................] - ETA: 1:17 - loss: 0.3826 - categorical_accuracy: 0.8823
16032/60000 [=======>......................] - ETA: 1:17 - loss: 0.3821 - categorical_accuracy: 0.8824
16064/60000 [=======>......................] - ETA: 1:17 - loss: 0.3819 - categorical_accuracy: 0.8824
16096/60000 [=======>......................] - ETA: 1:17 - loss: 0.3813 - categorical_accuracy: 0.8826
16128/60000 [=======>......................] - ETA: 1:17 - loss: 0.3806 - categorical_accuracy: 0.8829
16160/60000 [=======>......................] - ETA: 1:17 - loss: 0.3803 - categorical_accuracy: 0.8830
16192/60000 [=======>......................] - ETA: 1:17 - loss: 0.3796 - categorical_accuracy: 0.8832
16224/60000 [=======>......................] - ETA: 1:17 - loss: 0.3791 - categorical_accuracy: 0.8833
16256/60000 [=======>......................] - ETA: 1:17 - loss: 0.3785 - categorical_accuracy: 0.8836
16288/60000 [=======>......................] - ETA: 1:17 - loss: 0.3778 - categorical_accuracy: 0.8838
16320/60000 [=======>......................] - ETA: 1:17 - loss: 0.3774 - categorical_accuracy: 0.8839
16352/60000 [=======>......................] - ETA: 1:17 - loss: 0.3775 - categorical_accuracy: 0.8840
16384/60000 [=======>......................] - ETA: 1:17 - loss: 0.3769 - categorical_accuracy: 0.8842
16416/60000 [=======>......................] - ETA: 1:17 - loss: 0.3766 - categorical_accuracy: 0.8842
16448/60000 [=======>......................] - ETA: 1:17 - loss: 0.3761 - categorical_accuracy: 0.8843
16480/60000 [=======>......................] - ETA: 1:16 - loss: 0.3762 - categorical_accuracy: 0.8843
16512/60000 [=======>......................] - ETA: 1:16 - loss: 0.3756 - categorical_accuracy: 0.8845
16544/60000 [=======>......................] - ETA: 1:16 - loss: 0.3752 - categorical_accuracy: 0.8846
16576/60000 [=======>......................] - ETA: 1:16 - loss: 0.3746 - categorical_accuracy: 0.8848
16608/60000 [=======>......................] - ETA: 1:16 - loss: 0.3741 - categorical_accuracy: 0.8849
16640/60000 [=======>......................] - ETA: 1:16 - loss: 0.3737 - categorical_accuracy: 0.8850
16672/60000 [=======>......................] - ETA: 1:16 - loss: 0.3735 - categorical_accuracy: 0.8851
16704/60000 [=======>......................] - ETA: 1:16 - loss: 0.3730 - categorical_accuracy: 0.8852
16736/60000 [=======>......................] - ETA: 1:16 - loss: 0.3726 - categorical_accuracy: 0.8853
16768/60000 [=======>......................] - ETA: 1:16 - loss: 0.3722 - categorical_accuracy: 0.8854
16800/60000 [=======>......................] - ETA: 1:16 - loss: 0.3716 - categorical_accuracy: 0.8856
16832/60000 [=======>......................] - ETA: 1:16 - loss: 0.3710 - categorical_accuracy: 0.8858
16864/60000 [=======>......................] - ETA: 1:16 - loss: 0.3706 - categorical_accuracy: 0.8859
16896/60000 [=======>......................] - ETA: 1:16 - loss: 0.3702 - categorical_accuracy: 0.8859
16928/60000 [=======>......................] - ETA: 1:16 - loss: 0.3697 - categorical_accuracy: 0.8861
16960/60000 [=======>......................] - ETA: 1:16 - loss: 0.3691 - categorical_accuracy: 0.8863
16992/60000 [=======>......................] - ETA: 1:16 - loss: 0.3686 - categorical_accuracy: 0.8865
17024/60000 [=======>......................] - ETA: 1:16 - loss: 0.3681 - categorical_accuracy: 0.8866
17056/60000 [=======>......................] - ETA: 1:16 - loss: 0.3674 - categorical_accuracy: 0.8868
17088/60000 [=======>......................] - ETA: 1:15 - loss: 0.3674 - categorical_accuracy: 0.8869
17120/60000 [=======>......................] - ETA: 1:15 - loss: 0.3669 - categorical_accuracy: 0.8870
17152/60000 [=======>......................] - ETA: 1:15 - loss: 0.3668 - categorical_accuracy: 0.8871
17184/60000 [=======>......................] - ETA: 1:15 - loss: 0.3664 - categorical_accuracy: 0.8872
17216/60000 [=======>......................] - ETA: 1:15 - loss: 0.3662 - categorical_accuracy: 0.8873
17248/60000 [=======>......................] - ETA: 1:15 - loss: 0.3657 - categorical_accuracy: 0.8873
17280/60000 [=======>......................] - ETA: 1:15 - loss: 0.3655 - categorical_accuracy: 0.8875
17312/60000 [=======>......................] - ETA: 1:15 - loss: 0.3650 - categorical_accuracy: 0.8877
17344/60000 [=======>......................] - ETA: 1:15 - loss: 0.3644 - categorical_accuracy: 0.8878
17376/60000 [=======>......................] - ETA: 1:15 - loss: 0.3640 - categorical_accuracy: 0.8879
17408/60000 [=======>......................] - ETA: 1:15 - loss: 0.3635 - categorical_accuracy: 0.8880
17440/60000 [=======>......................] - ETA: 1:15 - loss: 0.3629 - categorical_accuracy: 0.8882
17472/60000 [=======>......................] - ETA: 1:15 - loss: 0.3626 - categorical_accuracy: 0.8883
17504/60000 [=======>......................] - ETA: 1:15 - loss: 0.3622 - categorical_accuracy: 0.8883
17536/60000 [=======>......................] - ETA: 1:15 - loss: 0.3621 - categorical_accuracy: 0.8883
17568/60000 [=======>......................] - ETA: 1:15 - loss: 0.3617 - categorical_accuracy: 0.8884
17600/60000 [=======>......................] - ETA: 1:15 - loss: 0.3615 - categorical_accuracy: 0.8885
17632/60000 [=======>......................] - ETA: 1:14 - loss: 0.3613 - categorical_accuracy: 0.8886
17664/60000 [=======>......................] - ETA: 1:14 - loss: 0.3618 - categorical_accuracy: 0.8884
17696/60000 [=======>......................] - ETA: 1:14 - loss: 0.3618 - categorical_accuracy: 0.8884
17728/60000 [=======>......................] - ETA: 1:14 - loss: 0.3613 - categorical_accuracy: 0.8885
17760/60000 [=======>......................] - ETA: 1:14 - loss: 0.3613 - categorical_accuracy: 0.8885
17792/60000 [=======>......................] - ETA: 1:14 - loss: 0.3608 - categorical_accuracy: 0.8887
17824/60000 [=======>......................] - ETA: 1:14 - loss: 0.3606 - categorical_accuracy: 0.8886
17856/60000 [=======>......................] - ETA: 1:14 - loss: 0.3606 - categorical_accuracy: 0.8887
17888/60000 [=======>......................] - ETA: 1:14 - loss: 0.3601 - categorical_accuracy: 0.8889
17920/60000 [=======>......................] - ETA: 1:14 - loss: 0.3598 - categorical_accuracy: 0.8890
17952/60000 [=======>......................] - ETA: 1:14 - loss: 0.3597 - categorical_accuracy: 0.8890
17984/60000 [=======>......................] - ETA: 1:14 - loss: 0.3597 - categorical_accuracy: 0.8890
18016/60000 [========>.....................] - ETA: 1:14 - loss: 0.3592 - categorical_accuracy: 0.8892
18048/60000 [========>.....................] - ETA: 1:14 - loss: 0.3589 - categorical_accuracy: 0.8893
18080/60000 [========>.....................] - ETA: 1:14 - loss: 0.3584 - categorical_accuracy: 0.8894
18112/60000 [========>.....................] - ETA: 1:14 - loss: 0.3578 - categorical_accuracy: 0.8895
18144/60000 [========>.....................] - ETA: 1:14 - loss: 0.3576 - categorical_accuracy: 0.8897
18176/60000 [========>.....................] - ETA: 1:13 - loss: 0.3571 - categorical_accuracy: 0.8898
18208/60000 [========>.....................] - ETA: 1:13 - loss: 0.3568 - categorical_accuracy: 0.8899
18240/60000 [========>.....................] - ETA: 1:13 - loss: 0.3564 - categorical_accuracy: 0.8901
18272/60000 [========>.....................] - ETA: 1:13 - loss: 0.3558 - categorical_accuracy: 0.8903
18304/60000 [========>.....................] - ETA: 1:13 - loss: 0.3555 - categorical_accuracy: 0.8904
18336/60000 [========>.....................] - ETA: 1:13 - loss: 0.3553 - categorical_accuracy: 0.8904
18368/60000 [========>.....................] - ETA: 1:13 - loss: 0.3549 - categorical_accuracy: 0.8906
18400/60000 [========>.....................] - ETA: 1:13 - loss: 0.3544 - categorical_accuracy: 0.8907
18432/60000 [========>.....................] - ETA: 1:13 - loss: 0.3540 - categorical_accuracy: 0.8908
18464/60000 [========>.....................] - ETA: 1:13 - loss: 0.3540 - categorical_accuracy: 0.8908
18496/60000 [========>.....................] - ETA: 1:13 - loss: 0.3537 - categorical_accuracy: 0.8909
18528/60000 [========>.....................] - ETA: 1:13 - loss: 0.3533 - categorical_accuracy: 0.8910
18560/60000 [========>.....................] - ETA: 1:13 - loss: 0.3527 - categorical_accuracy: 0.8912
18592/60000 [========>.....................] - ETA: 1:13 - loss: 0.3524 - categorical_accuracy: 0.8913
18624/60000 [========>.....................] - ETA: 1:13 - loss: 0.3527 - categorical_accuracy: 0.8914
18656/60000 [========>.....................] - ETA: 1:13 - loss: 0.3524 - categorical_accuracy: 0.8915
18688/60000 [========>.....................] - ETA: 1:12 - loss: 0.3518 - categorical_accuracy: 0.8917
18720/60000 [========>.....................] - ETA: 1:12 - loss: 0.3515 - categorical_accuracy: 0.8917
18752/60000 [========>.....................] - ETA: 1:12 - loss: 0.3513 - categorical_accuracy: 0.8918
18784/60000 [========>.....................] - ETA: 1:12 - loss: 0.3508 - categorical_accuracy: 0.8920
18816/60000 [========>.....................] - ETA: 1:12 - loss: 0.3507 - categorical_accuracy: 0.8920
18848/60000 [========>.....................] - ETA: 1:12 - loss: 0.3505 - categorical_accuracy: 0.8920
18880/60000 [========>.....................] - ETA: 1:12 - loss: 0.3500 - categorical_accuracy: 0.8922
18912/60000 [========>.....................] - ETA: 1:12 - loss: 0.3497 - categorical_accuracy: 0.8923
18944/60000 [========>.....................] - ETA: 1:12 - loss: 0.3493 - categorical_accuracy: 0.8924
18976/60000 [========>.....................] - ETA: 1:12 - loss: 0.3487 - categorical_accuracy: 0.8926
19008/60000 [========>.....................] - ETA: 1:12 - loss: 0.3486 - categorical_accuracy: 0.8927
19040/60000 [========>.....................] - ETA: 1:12 - loss: 0.3480 - categorical_accuracy: 0.8929
19072/60000 [========>.....................] - ETA: 1:12 - loss: 0.3476 - categorical_accuracy: 0.8930
19104/60000 [========>.....................] - ETA: 1:12 - loss: 0.3473 - categorical_accuracy: 0.8931
19136/60000 [========>.....................] - ETA: 1:12 - loss: 0.3470 - categorical_accuracy: 0.8931
19168/60000 [========>.....................] - ETA: 1:12 - loss: 0.3471 - categorical_accuracy: 0.8932
19200/60000 [========>.....................] - ETA: 1:12 - loss: 0.3466 - categorical_accuracy: 0.8933
19232/60000 [========>.....................] - ETA: 1:11 - loss: 0.3466 - categorical_accuracy: 0.8934
19264/60000 [========>.....................] - ETA: 1:11 - loss: 0.3462 - categorical_accuracy: 0.8935
19296/60000 [========>.....................] - ETA: 1:11 - loss: 0.3457 - categorical_accuracy: 0.8937
19328/60000 [========>.....................] - ETA: 1:11 - loss: 0.3460 - categorical_accuracy: 0.8937
19360/60000 [========>.....................] - ETA: 1:11 - loss: 0.3456 - categorical_accuracy: 0.8938
19392/60000 [========>.....................] - ETA: 1:11 - loss: 0.3453 - categorical_accuracy: 0.8939
19424/60000 [========>.....................] - ETA: 1:11 - loss: 0.3449 - categorical_accuracy: 0.8940
19456/60000 [========>.....................] - ETA: 1:11 - loss: 0.3445 - categorical_accuracy: 0.8942
19488/60000 [========>.....................] - ETA: 1:11 - loss: 0.3441 - categorical_accuracy: 0.8943
19520/60000 [========>.....................] - ETA: 1:11 - loss: 0.3438 - categorical_accuracy: 0.8944
19552/60000 [========>.....................] - ETA: 1:11 - loss: 0.3434 - categorical_accuracy: 0.8945
19584/60000 [========>.....................] - ETA: 1:11 - loss: 0.3431 - categorical_accuracy: 0.8946
19616/60000 [========>.....................] - ETA: 1:11 - loss: 0.3427 - categorical_accuracy: 0.8947
19648/60000 [========>.....................] - ETA: 1:11 - loss: 0.3422 - categorical_accuracy: 0.8949
19680/60000 [========>.....................] - ETA: 1:11 - loss: 0.3418 - categorical_accuracy: 0.8950
19712/60000 [========>.....................] - ETA: 1:11 - loss: 0.3413 - categorical_accuracy: 0.8952
19744/60000 [========>.....................] - ETA: 1:11 - loss: 0.3412 - categorical_accuracy: 0.8952
19776/60000 [========>.....................] - ETA: 1:10 - loss: 0.3408 - categorical_accuracy: 0.8954
19808/60000 [========>.....................] - ETA: 1:10 - loss: 0.3404 - categorical_accuracy: 0.8955
19840/60000 [========>.....................] - ETA: 1:10 - loss: 0.3399 - categorical_accuracy: 0.8957
19872/60000 [========>.....................] - ETA: 1:10 - loss: 0.3399 - categorical_accuracy: 0.8957
19904/60000 [========>.....................] - ETA: 1:10 - loss: 0.3396 - categorical_accuracy: 0.8957
19936/60000 [========>.....................] - ETA: 1:10 - loss: 0.3395 - categorical_accuracy: 0.8958
19968/60000 [========>.....................] - ETA: 1:10 - loss: 0.3391 - categorical_accuracy: 0.8959
20000/60000 [=========>....................] - ETA: 1:10 - loss: 0.3387 - categorical_accuracy: 0.8960
20032/60000 [=========>....................] - ETA: 1:10 - loss: 0.3382 - categorical_accuracy: 0.8962
20064/60000 [=========>....................] - ETA: 1:10 - loss: 0.3379 - categorical_accuracy: 0.8963
20096/60000 [=========>....................] - ETA: 1:10 - loss: 0.3381 - categorical_accuracy: 0.8963
20128/60000 [=========>....................] - ETA: 1:10 - loss: 0.3376 - categorical_accuracy: 0.8964
20160/60000 [=========>....................] - ETA: 1:10 - loss: 0.3372 - categorical_accuracy: 0.8965
20192/60000 [=========>....................] - ETA: 1:10 - loss: 0.3369 - categorical_accuracy: 0.8966
20224/60000 [=========>....................] - ETA: 1:10 - loss: 0.3365 - categorical_accuracy: 0.8967
20256/60000 [=========>....................] - ETA: 1:10 - loss: 0.3364 - categorical_accuracy: 0.8968
20288/60000 [=========>....................] - ETA: 1:10 - loss: 0.3359 - categorical_accuracy: 0.8970
20320/60000 [=========>....................] - ETA: 1:09 - loss: 0.3356 - categorical_accuracy: 0.8971
20352/60000 [=========>....................] - ETA: 1:09 - loss: 0.3353 - categorical_accuracy: 0.8971
20384/60000 [=========>....................] - ETA: 1:09 - loss: 0.3351 - categorical_accuracy: 0.8972
20416/60000 [=========>....................] - ETA: 1:09 - loss: 0.3349 - categorical_accuracy: 0.8971
20448/60000 [=========>....................] - ETA: 1:09 - loss: 0.3347 - categorical_accuracy: 0.8972
20480/60000 [=========>....................] - ETA: 1:09 - loss: 0.3343 - categorical_accuracy: 0.8973
20512/60000 [=========>....................] - ETA: 1:09 - loss: 0.3338 - categorical_accuracy: 0.8975
20544/60000 [=========>....................] - ETA: 1:09 - loss: 0.3336 - categorical_accuracy: 0.8975
20576/60000 [=========>....................] - ETA: 1:09 - loss: 0.3335 - categorical_accuracy: 0.8975
20608/60000 [=========>....................] - ETA: 1:09 - loss: 0.3332 - categorical_accuracy: 0.8976
20640/60000 [=========>....................] - ETA: 1:09 - loss: 0.3328 - categorical_accuracy: 0.8977
20672/60000 [=========>....................] - ETA: 1:09 - loss: 0.3324 - categorical_accuracy: 0.8979
20704/60000 [=========>....................] - ETA: 1:09 - loss: 0.3320 - categorical_accuracy: 0.8980
20736/60000 [=========>....................] - ETA: 1:09 - loss: 0.3316 - categorical_accuracy: 0.8981
20768/60000 [=========>....................] - ETA: 1:09 - loss: 0.3313 - categorical_accuracy: 0.8982
20800/60000 [=========>....................] - ETA: 1:09 - loss: 0.3309 - categorical_accuracy: 0.8983
20832/60000 [=========>....................] - ETA: 1:09 - loss: 0.3305 - categorical_accuracy: 0.8984
20864/60000 [=========>....................] - ETA: 1:09 - loss: 0.3301 - categorical_accuracy: 0.8985
20896/60000 [=========>....................] - ETA: 1:08 - loss: 0.3298 - categorical_accuracy: 0.8986
20928/60000 [=========>....................] - ETA: 1:08 - loss: 0.3293 - categorical_accuracy: 0.8987
20960/60000 [=========>....................] - ETA: 1:08 - loss: 0.3291 - categorical_accuracy: 0.8988
20992/60000 [=========>....................] - ETA: 1:08 - loss: 0.3288 - categorical_accuracy: 0.8989
21024/60000 [=========>....................] - ETA: 1:08 - loss: 0.3286 - categorical_accuracy: 0.8990
21056/60000 [=========>....................] - ETA: 1:08 - loss: 0.3284 - categorical_accuracy: 0.8990
21088/60000 [=========>....................] - ETA: 1:08 - loss: 0.3280 - categorical_accuracy: 0.8991
21120/60000 [=========>....................] - ETA: 1:08 - loss: 0.3278 - categorical_accuracy: 0.8991
21152/60000 [=========>....................] - ETA: 1:08 - loss: 0.3277 - categorical_accuracy: 0.8991
21184/60000 [=========>....................] - ETA: 1:08 - loss: 0.3274 - categorical_accuracy: 0.8992
21216/60000 [=========>....................] - ETA: 1:08 - loss: 0.3272 - categorical_accuracy: 0.8993
21248/60000 [=========>....................] - ETA: 1:08 - loss: 0.3267 - categorical_accuracy: 0.8994
21280/60000 [=========>....................] - ETA: 1:08 - loss: 0.3263 - categorical_accuracy: 0.8995
21344/60000 [=========>....................] - ETA: 1:08 - loss: 0.3256 - categorical_accuracy: 0.8998
21376/60000 [=========>....................] - ETA: 1:08 - loss: 0.3256 - categorical_accuracy: 0.8997
21408/60000 [=========>....................] - ETA: 1:08 - loss: 0.3256 - categorical_accuracy: 0.8999
21440/60000 [=========>....................] - ETA: 1:07 - loss: 0.3251 - categorical_accuracy: 0.9000
21472/60000 [=========>....................] - ETA: 1:07 - loss: 0.3248 - categorical_accuracy: 0.9001
21504/60000 [=========>....................] - ETA: 1:07 - loss: 0.3245 - categorical_accuracy: 0.9002
21536/60000 [=========>....................] - ETA: 1:07 - loss: 0.3242 - categorical_accuracy: 0.9003
21568/60000 [=========>....................] - ETA: 1:07 - loss: 0.3241 - categorical_accuracy: 0.9003
21600/60000 [=========>....................] - ETA: 1:07 - loss: 0.3244 - categorical_accuracy: 0.9003
21632/60000 [=========>....................] - ETA: 1:07 - loss: 0.3239 - categorical_accuracy: 0.9004
21664/60000 [=========>....................] - ETA: 1:07 - loss: 0.3235 - categorical_accuracy: 0.9006
21696/60000 [=========>....................] - ETA: 1:07 - loss: 0.3232 - categorical_accuracy: 0.9006
21728/60000 [=========>....................] - ETA: 1:07 - loss: 0.3230 - categorical_accuracy: 0.9006
21760/60000 [=========>....................] - ETA: 1:07 - loss: 0.3227 - categorical_accuracy: 0.9007
21792/60000 [=========>....................] - ETA: 1:07 - loss: 0.3227 - categorical_accuracy: 0.9008
21824/60000 [=========>....................] - ETA: 1:07 - loss: 0.3224 - categorical_accuracy: 0.9009
21856/60000 [=========>....................] - ETA: 1:07 - loss: 0.3222 - categorical_accuracy: 0.9009
21888/60000 [=========>....................] - ETA: 1:07 - loss: 0.3220 - categorical_accuracy: 0.9010
21920/60000 [=========>....................] - ETA: 1:07 - loss: 0.3216 - categorical_accuracy: 0.9011
21952/60000 [=========>....................] - ETA: 1:07 - loss: 0.3214 - categorical_accuracy: 0.9011
21984/60000 [=========>....................] - ETA: 1:06 - loss: 0.3210 - categorical_accuracy: 0.9012
22016/60000 [==========>...................] - ETA: 1:06 - loss: 0.3212 - categorical_accuracy: 0.9012
22048/60000 [==========>...................] - ETA: 1:06 - loss: 0.3208 - categorical_accuracy: 0.9013
22080/60000 [==========>...................] - ETA: 1:06 - loss: 0.3208 - categorical_accuracy: 0.9013
22112/60000 [==========>...................] - ETA: 1:06 - loss: 0.3205 - categorical_accuracy: 0.9015
22144/60000 [==========>...................] - ETA: 1:06 - loss: 0.3203 - categorical_accuracy: 0.9015
22176/60000 [==========>...................] - ETA: 1:06 - loss: 0.3199 - categorical_accuracy: 0.9016
22208/60000 [==========>...................] - ETA: 1:06 - loss: 0.3196 - categorical_accuracy: 0.9017
22240/60000 [==========>...................] - ETA: 1:06 - loss: 0.3193 - categorical_accuracy: 0.9018
22272/60000 [==========>...................] - ETA: 1:06 - loss: 0.3189 - categorical_accuracy: 0.9019
22304/60000 [==========>...................] - ETA: 1:06 - loss: 0.3190 - categorical_accuracy: 0.9019
22336/60000 [==========>...................] - ETA: 1:06 - loss: 0.3187 - categorical_accuracy: 0.9020
22368/60000 [==========>...................] - ETA: 1:06 - loss: 0.3184 - categorical_accuracy: 0.9021
22400/60000 [==========>...................] - ETA: 1:06 - loss: 0.3181 - categorical_accuracy: 0.9022
22432/60000 [==========>...................] - ETA: 1:06 - loss: 0.3177 - categorical_accuracy: 0.9023
22464/60000 [==========>...................] - ETA: 1:06 - loss: 0.3173 - categorical_accuracy: 0.9024
22496/60000 [==========>...................] - ETA: 1:06 - loss: 0.3174 - categorical_accuracy: 0.9025
22528/60000 [==========>...................] - ETA: 1:05 - loss: 0.3173 - categorical_accuracy: 0.9025
22560/60000 [==========>...................] - ETA: 1:05 - loss: 0.3169 - categorical_accuracy: 0.9026
22592/60000 [==========>...................] - ETA: 1:05 - loss: 0.3169 - categorical_accuracy: 0.9027
22624/60000 [==========>...................] - ETA: 1:05 - loss: 0.3167 - categorical_accuracy: 0.9028
22656/60000 [==========>...................] - ETA: 1:05 - loss: 0.3164 - categorical_accuracy: 0.9029
22688/60000 [==========>...................] - ETA: 1:05 - loss: 0.3161 - categorical_accuracy: 0.9030
22720/60000 [==========>...................] - ETA: 1:05 - loss: 0.3160 - categorical_accuracy: 0.9030
22752/60000 [==========>...................] - ETA: 1:05 - loss: 0.3161 - categorical_accuracy: 0.9031
22784/60000 [==========>...................] - ETA: 1:05 - loss: 0.3163 - categorical_accuracy: 0.9030
22816/60000 [==========>...................] - ETA: 1:05 - loss: 0.3161 - categorical_accuracy: 0.9031
22848/60000 [==========>...................] - ETA: 1:05 - loss: 0.3157 - categorical_accuracy: 0.9032
22880/60000 [==========>...................] - ETA: 1:05 - loss: 0.3155 - categorical_accuracy: 0.9032
22912/60000 [==========>...................] - ETA: 1:05 - loss: 0.3157 - categorical_accuracy: 0.9033
22944/60000 [==========>...................] - ETA: 1:05 - loss: 0.3154 - categorical_accuracy: 0.9034
22976/60000 [==========>...................] - ETA: 1:05 - loss: 0.3151 - categorical_accuracy: 0.9035
23008/60000 [==========>...................] - ETA: 1:05 - loss: 0.3149 - categorical_accuracy: 0.9034
23040/60000 [==========>...................] - ETA: 1:05 - loss: 0.3148 - categorical_accuracy: 0.9035
23072/60000 [==========>...................] - ETA: 1:05 - loss: 0.3144 - categorical_accuracy: 0.9036
23104/60000 [==========>...................] - ETA: 1:05 - loss: 0.3140 - categorical_accuracy: 0.9037
23136/60000 [==========>...................] - ETA: 1:04 - loss: 0.3136 - categorical_accuracy: 0.9038
23168/60000 [==========>...................] - ETA: 1:04 - loss: 0.3136 - categorical_accuracy: 0.9039
23200/60000 [==========>...................] - ETA: 1:04 - loss: 0.3133 - categorical_accuracy: 0.9040
23232/60000 [==========>...................] - ETA: 1:04 - loss: 0.3131 - categorical_accuracy: 0.9041
23264/60000 [==========>...................] - ETA: 1:04 - loss: 0.3133 - categorical_accuracy: 0.9040
23296/60000 [==========>...................] - ETA: 1:04 - loss: 0.3131 - categorical_accuracy: 0.9041
23328/60000 [==========>...................] - ETA: 1:04 - loss: 0.3129 - categorical_accuracy: 0.9041
23360/60000 [==========>...................] - ETA: 1:04 - loss: 0.3126 - categorical_accuracy: 0.9042
23392/60000 [==========>...................] - ETA: 1:04 - loss: 0.3126 - categorical_accuracy: 0.9042
23424/60000 [==========>...................] - ETA: 1:04 - loss: 0.3124 - categorical_accuracy: 0.9042
23456/60000 [==========>...................] - ETA: 1:04 - loss: 0.3121 - categorical_accuracy: 0.9043
23488/60000 [==========>...................] - ETA: 1:04 - loss: 0.3118 - categorical_accuracy: 0.9044
23520/60000 [==========>...................] - ETA: 1:04 - loss: 0.3115 - categorical_accuracy: 0.9045
23552/60000 [==========>...................] - ETA: 1:04 - loss: 0.3114 - categorical_accuracy: 0.9046
23584/60000 [==========>...................] - ETA: 1:04 - loss: 0.3111 - categorical_accuracy: 0.9046
23616/60000 [==========>...................] - ETA: 1:04 - loss: 0.3111 - categorical_accuracy: 0.9046
23648/60000 [==========>...................] - ETA: 1:03 - loss: 0.3107 - categorical_accuracy: 0.9048
23680/60000 [==========>...................] - ETA: 1:03 - loss: 0.3105 - categorical_accuracy: 0.9048
23712/60000 [==========>...................] - ETA: 1:03 - loss: 0.3102 - categorical_accuracy: 0.9049
23744/60000 [==========>...................] - ETA: 1:03 - loss: 0.3099 - categorical_accuracy: 0.9050
23776/60000 [==========>...................] - ETA: 1:03 - loss: 0.3096 - categorical_accuracy: 0.9051
23808/60000 [==========>...................] - ETA: 1:03 - loss: 0.3096 - categorical_accuracy: 0.9051
23840/60000 [==========>...................] - ETA: 1:03 - loss: 0.3092 - categorical_accuracy: 0.9052
23872/60000 [==========>...................] - ETA: 1:03 - loss: 0.3090 - categorical_accuracy: 0.9052
23904/60000 [==========>...................] - ETA: 1:03 - loss: 0.3093 - categorical_accuracy: 0.9052
23936/60000 [==========>...................] - ETA: 1:03 - loss: 0.3090 - categorical_accuracy: 0.9052
23968/60000 [==========>...................] - ETA: 1:03 - loss: 0.3089 - categorical_accuracy: 0.9053
24000/60000 [===========>..................] - ETA: 1:03 - loss: 0.3087 - categorical_accuracy: 0.9054
24032/60000 [===========>..................] - ETA: 1:03 - loss: 0.3083 - categorical_accuracy: 0.9055
24064/60000 [===========>..................] - ETA: 1:03 - loss: 0.3080 - categorical_accuracy: 0.9056
24096/60000 [===========>..................] - ETA: 1:03 - loss: 0.3077 - categorical_accuracy: 0.9057
24128/60000 [===========>..................] - ETA: 1:03 - loss: 0.3075 - categorical_accuracy: 0.9058
24160/60000 [===========>..................] - ETA: 1:03 - loss: 0.3072 - categorical_accuracy: 0.9058
24192/60000 [===========>..................] - ETA: 1:03 - loss: 0.3071 - categorical_accuracy: 0.9058
24224/60000 [===========>..................] - ETA: 1:02 - loss: 0.3069 - categorical_accuracy: 0.9058
24256/60000 [===========>..................] - ETA: 1:02 - loss: 0.3066 - categorical_accuracy: 0.9059
24288/60000 [===========>..................] - ETA: 1:02 - loss: 0.3065 - categorical_accuracy: 0.9060
24320/60000 [===========>..................] - ETA: 1:02 - loss: 0.3065 - categorical_accuracy: 0.9060
24352/60000 [===========>..................] - ETA: 1:02 - loss: 0.3063 - categorical_accuracy: 0.9060
24384/60000 [===========>..................] - ETA: 1:02 - loss: 0.3061 - categorical_accuracy: 0.9060
24416/60000 [===========>..................] - ETA: 1:02 - loss: 0.3059 - categorical_accuracy: 0.9061
24448/60000 [===========>..................] - ETA: 1:02 - loss: 0.3055 - categorical_accuracy: 0.9062
24480/60000 [===========>..................] - ETA: 1:02 - loss: 0.3052 - categorical_accuracy: 0.9064
24512/60000 [===========>..................] - ETA: 1:02 - loss: 0.3050 - categorical_accuracy: 0.9064
24544/60000 [===========>..................] - ETA: 1:02 - loss: 0.3048 - categorical_accuracy: 0.9065
24608/60000 [===========>..................] - ETA: 1:02 - loss: 0.3042 - categorical_accuracy: 0.9067
24640/60000 [===========>..................] - ETA: 1:02 - loss: 0.3040 - categorical_accuracy: 0.9067
24672/60000 [===========>..................] - ETA: 1:02 - loss: 0.3042 - categorical_accuracy: 0.9068
24704/60000 [===========>..................] - ETA: 1:02 - loss: 0.3043 - categorical_accuracy: 0.9068
24736/60000 [===========>..................] - ETA: 1:01 - loss: 0.3039 - categorical_accuracy: 0.9069
24768/60000 [===========>..................] - ETA: 1:01 - loss: 0.3038 - categorical_accuracy: 0.9069
24800/60000 [===========>..................] - ETA: 1:01 - loss: 0.3034 - categorical_accuracy: 0.9071
24832/60000 [===========>..................] - ETA: 1:01 - loss: 0.3033 - categorical_accuracy: 0.9071
24864/60000 [===========>..................] - ETA: 1:01 - loss: 0.3031 - categorical_accuracy: 0.9072
24896/60000 [===========>..................] - ETA: 1:01 - loss: 0.3029 - categorical_accuracy: 0.9072
24928/60000 [===========>..................] - ETA: 1:01 - loss: 0.3026 - categorical_accuracy: 0.9073
24960/60000 [===========>..................] - ETA: 1:01 - loss: 0.3024 - categorical_accuracy: 0.9074
24992/60000 [===========>..................] - ETA: 1:01 - loss: 0.3021 - categorical_accuracy: 0.9075
25024/60000 [===========>..................] - ETA: 1:01 - loss: 0.3019 - categorical_accuracy: 0.9076
25056/60000 [===========>..................] - ETA: 1:01 - loss: 0.3018 - categorical_accuracy: 0.9076
25088/60000 [===========>..................] - ETA: 1:01 - loss: 0.3015 - categorical_accuracy: 0.9077
25120/60000 [===========>..................] - ETA: 1:01 - loss: 0.3014 - categorical_accuracy: 0.9077
25152/60000 [===========>..................] - ETA: 1:01 - loss: 0.3011 - categorical_accuracy: 0.9078
25184/60000 [===========>..................] - ETA: 1:01 - loss: 0.3009 - categorical_accuracy: 0.9079
25216/60000 [===========>..................] - ETA: 1:01 - loss: 0.3006 - categorical_accuracy: 0.9080
25248/60000 [===========>..................] - ETA: 1:01 - loss: 0.3005 - categorical_accuracy: 0.9080
25280/60000 [===========>..................] - ETA: 1:01 - loss: 0.3003 - categorical_accuracy: 0.9080
25312/60000 [===========>..................] - ETA: 1:00 - loss: 0.3002 - categorical_accuracy: 0.9081
25344/60000 [===========>..................] - ETA: 1:00 - loss: 0.3000 - categorical_accuracy: 0.9081
25376/60000 [===========>..................] - ETA: 1:00 - loss: 0.2997 - categorical_accuracy: 0.9082
25408/60000 [===========>..................] - ETA: 1:00 - loss: 0.2996 - categorical_accuracy: 0.9082
25440/60000 [===========>..................] - ETA: 1:00 - loss: 0.2995 - categorical_accuracy: 0.9083
25472/60000 [===========>..................] - ETA: 1:00 - loss: 0.2992 - categorical_accuracy: 0.9083
25504/60000 [===========>..................] - ETA: 1:00 - loss: 0.2992 - categorical_accuracy: 0.9084
25536/60000 [===========>..................] - ETA: 1:00 - loss: 0.2989 - categorical_accuracy: 0.9085
25568/60000 [===========>..................] - ETA: 1:00 - loss: 0.2987 - categorical_accuracy: 0.9085
25600/60000 [===========>..................] - ETA: 1:00 - loss: 0.2985 - categorical_accuracy: 0.9086
25632/60000 [===========>..................] - ETA: 1:00 - loss: 0.2982 - categorical_accuracy: 0.9086
25664/60000 [===========>..................] - ETA: 1:00 - loss: 0.2980 - categorical_accuracy: 0.9087
25696/60000 [===========>..................] - ETA: 1:00 - loss: 0.2981 - categorical_accuracy: 0.9086
25728/60000 [===========>..................] - ETA: 1:00 - loss: 0.2981 - categorical_accuracy: 0.9086
25760/60000 [===========>..................] - ETA: 1:00 - loss: 0.2978 - categorical_accuracy: 0.9087
25792/60000 [===========>..................] - ETA: 1:00 - loss: 0.2975 - categorical_accuracy: 0.9088
25824/60000 [===========>..................] - ETA: 1:00 - loss: 0.2973 - categorical_accuracy: 0.9089
25856/60000 [===========>..................] - ETA: 1:00 - loss: 0.2970 - categorical_accuracy: 0.9090
25888/60000 [===========>..................] - ETA: 1:00 - loss: 0.2968 - categorical_accuracy: 0.9090
25920/60000 [===========>..................] - ETA: 59s - loss: 0.2966 - categorical_accuracy: 0.9091 
25952/60000 [===========>..................] - ETA: 59s - loss: 0.2967 - categorical_accuracy: 0.9091
25984/60000 [===========>..................] - ETA: 59s - loss: 0.2965 - categorical_accuracy: 0.9092
26016/60000 [============>.................] - ETA: 59s - loss: 0.2962 - categorical_accuracy: 0.9092
26048/60000 [============>.................] - ETA: 59s - loss: 0.2959 - categorical_accuracy: 0.9093
26080/60000 [============>.................] - ETA: 59s - loss: 0.2957 - categorical_accuracy: 0.9093
26112/60000 [============>.................] - ETA: 59s - loss: 0.2955 - categorical_accuracy: 0.9094
26144/60000 [============>.................] - ETA: 59s - loss: 0.2951 - categorical_accuracy: 0.9095
26176/60000 [============>.................] - ETA: 59s - loss: 0.2948 - categorical_accuracy: 0.9097
26208/60000 [============>.................] - ETA: 59s - loss: 0.2944 - categorical_accuracy: 0.9098
26240/60000 [============>.................] - ETA: 59s - loss: 0.2944 - categorical_accuracy: 0.9098
26272/60000 [============>.................] - ETA: 59s - loss: 0.2941 - categorical_accuracy: 0.9099
26304/60000 [============>.................] - ETA: 59s - loss: 0.2941 - categorical_accuracy: 0.9099
26336/60000 [============>.................] - ETA: 59s - loss: 0.2939 - categorical_accuracy: 0.9099
26368/60000 [============>.................] - ETA: 59s - loss: 0.2938 - categorical_accuracy: 0.9100
26400/60000 [============>.................] - ETA: 59s - loss: 0.2935 - categorical_accuracy: 0.9100
26432/60000 [============>.................] - ETA: 59s - loss: 0.2933 - categorical_accuracy: 0.9101
26464/60000 [============>.................] - ETA: 59s - loss: 0.2933 - categorical_accuracy: 0.9101
26496/60000 [============>.................] - ETA: 58s - loss: 0.2931 - categorical_accuracy: 0.9101
26528/60000 [============>.................] - ETA: 58s - loss: 0.2928 - categorical_accuracy: 0.9102
26560/60000 [============>.................] - ETA: 58s - loss: 0.2926 - categorical_accuracy: 0.9103
26592/60000 [============>.................] - ETA: 58s - loss: 0.2926 - categorical_accuracy: 0.9103
26624/60000 [============>.................] - ETA: 58s - loss: 0.2926 - categorical_accuracy: 0.9103
26656/60000 [============>.................] - ETA: 58s - loss: 0.2923 - categorical_accuracy: 0.9104
26688/60000 [============>.................] - ETA: 58s - loss: 0.2921 - categorical_accuracy: 0.9104
26720/60000 [============>.................] - ETA: 58s - loss: 0.2917 - categorical_accuracy: 0.9106
26752/60000 [============>.................] - ETA: 58s - loss: 0.2915 - categorical_accuracy: 0.9106
26784/60000 [============>.................] - ETA: 58s - loss: 0.2915 - categorical_accuracy: 0.9106
26816/60000 [============>.................] - ETA: 58s - loss: 0.2912 - categorical_accuracy: 0.9107
26848/60000 [============>.................] - ETA: 58s - loss: 0.2911 - categorical_accuracy: 0.9107
26880/60000 [============>.................] - ETA: 58s - loss: 0.2908 - categorical_accuracy: 0.9108
26912/60000 [============>.................] - ETA: 58s - loss: 0.2905 - categorical_accuracy: 0.9109
26944/60000 [============>.................] - ETA: 58s - loss: 0.2901 - categorical_accuracy: 0.9110
26976/60000 [============>.................] - ETA: 58s - loss: 0.2900 - categorical_accuracy: 0.9111
27008/60000 [============>.................] - ETA: 58s - loss: 0.2897 - categorical_accuracy: 0.9111
27040/60000 [============>.................] - ETA: 57s - loss: 0.2896 - categorical_accuracy: 0.9112
27072/60000 [============>.................] - ETA: 57s - loss: 0.2895 - categorical_accuracy: 0.9112
27104/60000 [============>.................] - ETA: 57s - loss: 0.2896 - categorical_accuracy: 0.9112
27136/60000 [============>.................] - ETA: 57s - loss: 0.2896 - categorical_accuracy: 0.9111
27168/60000 [============>.................] - ETA: 57s - loss: 0.2895 - categorical_accuracy: 0.9111
27200/60000 [============>.................] - ETA: 57s - loss: 0.2892 - categorical_accuracy: 0.9112
27232/60000 [============>.................] - ETA: 57s - loss: 0.2890 - categorical_accuracy: 0.9113
27264/60000 [============>.................] - ETA: 57s - loss: 0.2889 - categorical_accuracy: 0.9114
27296/60000 [============>.................] - ETA: 57s - loss: 0.2886 - categorical_accuracy: 0.9115
27328/60000 [============>.................] - ETA: 57s - loss: 0.2885 - categorical_accuracy: 0.9115
27360/60000 [============>.................] - ETA: 57s - loss: 0.2882 - categorical_accuracy: 0.9116
27392/60000 [============>.................] - ETA: 57s - loss: 0.2880 - categorical_accuracy: 0.9117
27424/60000 [============>.................] - ETA: 57s - loss: 0.2876 - categorical_accuracy: 0.9118
27456/60000 [============>.................] - ETA: 57s - loss: 0.2874 - categorical_accuracy: 0.9119
27488/60000 [============>.................] - ETA: 57s - loss: 0.2872 - categorical_accuracy: 0.9120
27520/60000 [============>.................] - ETA: 57s - loss: 0.2870 - categorical_accuracy: 0.9120
27552/60000 [============>.................] - ETA: 57s - loss: 0.2869 - categorical_accuracy: 0.9121
27584/60000 [============>.................] - ETA: 57s - loss: 0.2866 - categorical_accuracy: 0.9122
27616/60000 [============>.................] - ETA: 56s - loss: 0.2868 - categorical_accuracy: 0.9122
27648/60000 [============>.................] - ETA: 56s - loss: 0.2867 - categorical_accuracy: 0.9122
27680/60000 [============>.................] - ETA: 56s - loss: 0.2865 - categorical_accuracy: 0.9122
27712/60000 [============>.................] - ETA: 56s - loss: 0.2864 - categorical_accuracy: 0.9122
27744/60000 [============>.................] - ETA: 56s - loss: 0.2863 - categorical_accuracy: 0.9123
27776/60000 [============>.................] - ETA: 56s - loss: 0.2860 - categorical_accuracy: 0.9123
27808/60000 [============>.................] - ETA: 56s - loss: 0.2858 - categorical_accuracy: 0.9124
27840/60000 [============>.................] - ETA: 56s - loss: 0.2857 - categorical_accuracy: 0.9124
27872/60000 [============>.................] - ETA: 56s - loss: 0.2855 - categorical_accuracy: 0.9125
27904/60000 [============>.................] - ETA: 56s - loss: 0.2853 - categorical_accuracy: 0.9126
27936/60000 [============>.................] - ETA: 56s - loss: 0.2850 - categorical_accuracy: 0.9126
27968/60000 [============>.................] - ETA: 56s - loss: 0.2847 - categorical_accuracy: 0.9127
28000/60000 [=============>................] - ETA: 56s - loss: 0.2846 - categorical_accuracy: 0.9128
28032/60000 [=============>................] - ETA: 56s - loss: 0.2845 - categorical_accuracy: 0.9128
28064/60000 [=============>................] - ETA: 56s - loss: 0.2842 - categorical_accuracy: 0.9129
28096/60000 [=============>................] - ETA: 56s - loss: 0.2839 - categorical_accuracy: 0.9130
28128/60000 [=============>................] - ETA: 56s - loss: 0.2836 - categorical_accuracy: 0.9131
28160/60000 [=============>................] - ETA: 55s - loss: 0.2833 - categorical_accuracy: 0.9132
28192/60000 [=============>................] - ETA: 55s - loss: 0.2831 - categorical_accuracy: 0.9133
28224/60000 [=============>................] - ETA: 55s - loss: 0.2828 - categorical_accuracy: 0.9134
28256/60000 [=============>................] - ETA: 55s - loss: 0.2826 - categorical_accuracy: 0.9134
28288/60000 [=============>................] - ETA: 55s - loss: 0.2823 - categorical_accuracy: 0.9135
28320/60000 [=============>................] - ETA: 55s - loss: 0.2821 - categorical_accuracy: 0.9136
28352/60000 [=============>................] - ETA: 55s - loss: 0.2820 - categorical_accuracy: 0.9136
28384/60000 [=============>................] - ETA: 55s - loss: 0.2820 - categorical_accuracy: 0.9136
28416/60000 [=============>................] - ETA: 55s - loss: 0.2818 - categorical_accuracy: 0.9137
28448/60000 [=============>................] - ETA: 55s - loss: 0.2816 - categorical_accuracy: 0.9137
28480/60000 [=============>................] - ETA: 55s - loss: 0.2816 - categorical_accuracy: 0.9137
28512/60000 [=============>................] - ETA: 55s - loss: 0.2813 - categorical_accuracy: 0.9138
28544/60000 [=============>................] - ETA: 55s - loss: 0.2813 - categorical_accuracy: 0.9138
28576/60000 [=============>................] - ETA: 55s - loss: 0.2814 - categorical_accuracy: 0.9138
28608/60000 [=============>................] - ETA: 55s - loss: 0.2813 - categorical_accuracy: 0.9139
28640/60000 [=============>................] - ETA: 55s - loss: 0.2810 - categorical_accuracy: 0.9140
28672/60000 [=============>................] - ETA: 55s - loss: 0.2808 - categorical_accuracy: 0.9140
28704/60000 [=============>................] - ETA: 54s - loss: 0.2805 - categorical_accuracy: 0.9141
28736/60000 [=============>................] - ETA: 54s - loss: 0.2806 - categorical_accuracy: 0.9140
28768/60000 [=============>................] - ETA: 54s - loss: 0.2804 - categorical_accuracy: 0.9141
28800/60000 [=============>................] - ETA: 54s - loss: 0.2801 - categorical_accuracy: 0.9142
28832/60000 [=============>................] - ETA: 54s - loss: 0.2798 - categorical_accuracy: 0.9143
28864/60000 [=============>................] - ETA: 54s - loss: 0.2797 - categorical_accuracy: 0.9143
28896/60000 [=============>................] - ETA: 54s - loss: 0.2795 - categorical_accuracy: 0.9143
28928/60000 [=============>................] - ETA: 54s - loss: 0.2798 - categorical_accuracy: 0.9143
28960/60000 [=============>................] - ETA: 54s - loss: 0.2795 - categorical_accuracy: 0.9143
28992/60000 [=============>................] - ETA: 54s - loss: 0.2795 - categorical_accuracy: 0.9143
29024/60000 [=============>................] - ETA: 54s - loss: 0.2792 - categorical_accuracy: 0.9144
29056/60000 [=============>................] - ETA: 54s - loss: 0.2790 - categorical_accuracy: 0.9145
29088/60000 [=============>................] - ETA: 54s - loss: 0.2789 - categorical_accuracy: 0.9145
29120/60000 [=============>................] - ETA: 54s - loss: 0.2786 - categorical_accuracy: 0.9146
29152/60000 [=============>................] - ETA: 54s - loss: 0.2784 - categorical_accuracy: 0.9147
29184/60000 [=============>................] - ETA: 54s - loss: 0.2782 - categorical_accuracy: 0.9147
29216/60000 [=============>................] - ETA: 54s - loss: 0.2779 - categorical_accuracy: 0.9148
29248/60000 [=============>................] - ETA: 54s - loss: 0.2777 - categorical_accuracy: 0.9149
29280/60000 [=============>................] - ETA: 53s - loss: 0.2775 - categorical_accuracy: 0.9149
29312/60000 [=============>................] - ETA: 53s - loss: 0.2774 - categorical_accuracy: 0.9149
29344/60000 [=============>................] - ETA: 53s - loss: 0.2772 - categorical_accuracy: 0.9150
29376/60000 [=============>................] - ETA: 53s - loss: 0.2772 - categorical_accuracy: 0.9150
29408/60000 [=============>................] - ETA: 53s - loss: 0.2771 - categorical_accuracy: 0.9151
29440/60000 [=============>................] - ETA: 53s - loss: 0.2770 - categorical_accuracy: 0.9151
29472/60000 [=============>................] - ETA: 53s - loss: 0.2769 - categorical_accuracy: 0.9151
29504/60000 [=============>................] - ETA: 53s - loss: 0.2766 - categorical_accuracy: 0.9152
29536/60000 [=============>................] - ETA: 53s - loss: 0.2765 - categorical_accuracy: 0.9152
29600/60000 [=============>................] - ETA: 53s - loss: 0.2761 - categorical_accuracy: 0.9153
29632/60000 [=============>................] - ETA: 53s - loss: 0.2758 - categorical_accuracy: 0.9154
29664/60000 [=============>................] - ETA: 53s - loss: 0.2756 - categorical_accuracy: 0.9155
29728/60000 [=============>................] - ETA: 53s - loss: 0.2754 - categorical_accuracy: 0.9155
29760/60000 [=============>................] - ETA: 53s - loss: 0.2752 - categorical_accuracy: 0.9156
29792/60000 [=============>................] - ETA: 53s - loss: 0.2750 - categorical_accuracy: 0.9156
29824/60000 [=============>................] - ETA: 52s - loss: 0.2751 - categorical_accuracy: 0.9156
29856/60000 [=============>................] - ETA: 52s - loss: 0.2753 - categorical_accuracy: 0.9156
29888/60000 [=============>................] - ETA: 52s - loss: 0.2751 - categorical_accuracy: 0.9156
29920/60000 [=============>................] - ETA: 52s - loss: 0.2752 - categorical_accuracy: 0.9156
29952/60000 [=============>................] - ETA: 52s - loss: 0.2749 - categorical_accuracy: 0.9157
29984/60000 [=============>................] - ETA: 52s - loss: 0.2748 - categorical_accuracy: 0.9157
30016/60000 [==============>...............] - ETA: 52s - loss: 0.2747 - categorical_accuracy: 0.9157
30048/60000 [==============>...............] - ETA: 52s - loss: 0.2745 - categorical_accuracy: 0.9158
30080/60000 [==============>...............] - ETA: 52s - loss: 0.2742 - categorical_accuracy: 0.9159
30112/60000 [==============>...............] - ETA: 52s - loss: 0.2740 - categorical_accuracy: 0.9159
30144/60000 [==============>...............] - ETA: 52s - loss: 0.2738 - categorical_accuracy: 0.9160
30176/60000 [==============>...............] - ETA: 52s - loss: 0.2737 - categorical_accuracy: 0.9161
30208/60000 [==============>...............] - ETA: 52s - loss: 0.2735 - categorical_accuracy: 0.9161
30240/60000 [==============>...............] - ETA: 52s - loss: 0.2735 - categorical_accuracy: 0.9161
30272/60000 [==============>...............] - ETA: 52s - loss: 0.2732 - categorical_accuracy: 0.9162
30304/60000 [==============>...............] - ETA: 52s - loss: 0.2732 - categorical_accuracy: 0.9162
30336/60000 [==============>...............] - ETA: 52s - loss: 0.2732 - categorical_accuracy: 0.9163
30368/60000 [==============>...............] - ETA: 52s - loss: 0.2731 - categorical_accuracy: 0.9163
30400/60000 [==============>...............] - ETA: 51s - loss: 0.2728 - categorical_accuracy: 0.9164
30432/60000 [==============>...............] - ETA: 51s - loss: 0.2727 - categorical_accuracy: 0.9164
30464/60000 [==============>...............] - ETA: 51s - loss: 0.2725 - categorical_accuracy: 0.9165
30496/60000 [==============>...............] - ETA: 51s - loss: 0.2724 - categorical_accuracy: 0.9165
30528/60000 [==============>...............] - ETA: 51s - loss: 0.2723 - categorical_accuracy: 0.9165
30560/60000 [==============>...............] - ETA: 51s - loss: 0.2722 - categorical_accuracy: 0.9166
30592/60000 [==============>...............] - ETA: 51s - loss: 0.2720 - categorical_accuracy: 0.9166
30624/60000 [==============>...............] - ETA: 51s - loss: 0.2718 - categorical_accuracy: 0.9167
30688/60000 [==============>...............] - ETA: 51s - loss: 0.2715 - categorical_accuracy: 0.9168
30720/60000 [==============>...............] - ETA: 51s - loss: 0.2713 - categorical_accuracy: 0.9169
30752/60000 [==============>...............] - ETA: 51s - loss: 0.2710 - categorical_accuracy: 0.9170
30784/60000 [==============>...............] - ETA: 51s - loss: 0.2710 - categorical_accuracy: 0.9170
30816/60000 [==============>...............] - ETA: 51s - loss: 0.2708 - categorical_accuracy: 0.9171
30848/60000 [==============>...............] - ETA: 51s - loss: 0.2708 - categorical_accuracy: 0.9171
30880/60000 [==============>...............] - ETA: 51s - loss: 0.2707 - categorical_accuracy: 0.9171
30912/60000 [==============>...............] - ETA: 51s - loss: 0.2705 - categorical_accuracy: 0.9172
30944/60000 [==============>...............] - ETA: 50s - loss: 0.2703 - categorical_accuracy: 0.9173
30976/60000 [==============>...............] - ETA: 50s - loss: 0.2700 - categorical_accuracy: 0.9174
31008/60000 [==============>...............] - ETA: 50s - loss: 0.2698 - categorical_accuracy: 0.9175
31040/60000 [==============>...............] - ETA: 50s - loss: 0.2697 - categorical_accuracy: 0.9175
31072/60000 [==============>...............] - ETA: 50s - loss: 0.2695 - categorical_accuracy: 0.9175
31104/60000 [==============>...............] - ETA: 50s - loss: 0.2693 - categorical_accuracy: 0.9176
31136/60000 [==============>...............] - ETA: 50s - loss: 0.2692 - categorical_accuracy: 0.9176
31168/60000 [==============>...............] - ETA: 50s - loss: 0.2690 - categorical_accuracy: 0.9177
31200/60000 [==============>...............] - ETA: 50s - loss: 0.2688 - categorical_accuracy: 0.9177
31232/60000 [==============>...............] - ETA: 50s - loss: 0.2688 - categorical_accuracy: 0.9177
31264/60000 [==============>...............] - ETA: 50s - loss: 0.2686 - categorical_accuracy: 0.9178
31296/60000 [==============>...............] - ETA: 50s - loss: 0.2684 - categorical_accuracy: 0.9178
31328/60000 [==============>...............] - ETA: 50s - loss: 0.2682 - categorical_accuracy: 0.9179
31360/60000 [==============>...............] - ETA: 50s - loss: 0.2680 - categorical_accuracy: 0.9179
31392/60000 [==============>...............] - ETA: 50s - loss: 0.2678 - categorical_accuracy: 0.9179
31424/60000 [==============>...............] - ETA: 50s - loss: 0.2676 - categorical_accuracy: 0.9180
31456/60000 [==============>...............] - ETA: 50s - loss: 0.2673 - categorical_accuracy: 0.9181
31488/60000 [==============>...............] - ETA: 50s - loss: 0.2671 - categorical_accuracy: 0.9182
31520/60000 [==============>...............] - ETA: 49s - loss: 0.2671 - categorical_accuracy: 0.9182
31552/60000 [==============>...............] - ETA: 49s - loss: 0.2668 - categorical_accuracy: 0.9183
31584/60000 [==============>...............] - ETA: 49s - loss: 0.2666 - categorical_accuracy: 0.9184
31616/60000 [==============>...............] - ETA: 49s - loss: 0.2666 - categorical_accuracy: 0.9184
31648/60000 [==============>...............] - ETA: 49s - loss: 0.2664 - categorical_accuracy: 0.9184
31680/60000 [==============>...............] - ETA: 49s - loss: 0.2661 - categorical_accuracy: 0.9185
31712/60000 [==============>...............] - ETA: 49s - loss: 0.2660 - categorical_accuracy: 0.9185
31744/60000 [==============>...............] - ETA: 49s - loss: 0.2657 - categorical_accuracy: 0.9186
31776/60000 [==============>...............] - ETA: 49s - loss: 0.2657 - categorical_accuracy: 0.9186
31808/60000 [==============>...............] - ETA: 49s - loss: 0.2655 - categorical_accuracy: 0.9187
31840/60000 [==============>...............] - ETA: 49s - loss: 0.2652 - categorical_accuracy: 0.9188
31872/60000 [==============>...............] - ETA: 49s - loss: 0.2650 - categorical_accuracy: 0.9189
31904/60000 [==============>...............] - ETA: 49s - loss: 0.2649 - categorical_accuracy: 0.9189
31936/60000 [==============>...............] - ETA: 49s - loss: 0.2648 - categorical_accuracy: 0.9189
31968/60000 [==============>...............] - ETA: 49s - loss: 0.2646 - categorical_accuracy: 0.9190
32000/60000 [===============>..............] - ETA: 49s - loss: 0.2647 - categorical_accuracy: 0.9190
32032/60000 [===============>..............] - ETA: 49s - loss: 0.2646 - categorical_accuracy: 0.9190
32064/60000 [===============>..............] - ETA: 48s - loss: 0.2644 - categorical_accuracy: 0.9191
32096/60000 [===============>..............] - ETA: 48s - loss: 0.2642 - categorical_accuracy: 0.9191
32128/60000 [===============>..............] - ETA: 48s - loss: 0.2640 - categorical_accuracy: 0.9191
32160/60000 [===============>..............] - ETA: 48s - loss: 0.2638 - categorical_accuracy: 0.9192
32192/60000 [===============>..............] - ETA: 48s - loss: 0.2638 - categorical_accuracy: 0.9192
32224/60000 [===============>..............] - ETA: 48s - loss: 0.2639 - categorical_accuracy: 0.9192
32256/60000 [===============>..............] - ETA: 48s - loss: 0.2641 - categorical_accuracy: 0.9192
32288/60000 [===============>..............] - ETA: 48s - loss: 0.2642 - categorical_accuracy: 0.9192
32320/60000 [===============>..............] - ETA: 48s - loss: 0.2640 - categorical_accuracy: 0.9192
32352/60000 [===============>..............] - ETA: 48s - loss: 0.2639 - categorical_accuracy: 0.9193
32384/60000 [===============>..............] - ETA: 48s - loss: 0.2638 - categorical_accuracy: 0.9193
32416/60000 [===============>..............] - ETA: 48s - loss: 0.2636 - categorical_accuracy: 0.9193
32448/60000 [===============>..............] - ETA: 48s - loss: 0.2634 - categorical_accuracy: 0.9194
32480/60000 [===============>..............] - ETA: 48s - loss: 0.2632 - categorical_accuracy: 0.9195
32512/60000 [===============>..............] - ETA: 48s - loss: 0.2632 - categorical_accuracy: 0.9195
32544/60000 [===============>..............] - ETA: 48s - loss: 0.2630 - categorical_accuracy: 0.9195
32576/60000 [===============>..............] - ETA: 48s - loss: 0.2629 - categorical_accuracy: 0.9195
32608/60000 [===============>..............] - ETA: 48s - loss: 0.2627 - categorical_accuracy: 0.9196
32640/60000 [===============>..............] - ETA: 47s - loss: 0.2625 - categorical_accuracy: 0.9196
32672/60000 [===============>..............] - ETA: 47s - loss: 0.2623 - categorical_accuracy: 0.9197
32704/60000 [===============>..............] - ETA: 47s - loss: 0.2623 - categorical_accuracy: 0.9197
32736/60000 [===============>..............] - ETA: 47s - loss: 0.2623 - categorical_accuracy: 0.9198
32768/60000 [===============>..............] - ETA: 47s - loss: 0.2621 - categorical_accuracy: 0.9198
32800/60000 [===============>..............] - ETA: 47s - loss: 0.2619 - categorical_accuracy: 0.9199
32832/60000 [===============>..............] - ETA: 47s - loss: 0.2618 - categorical_accuracy: 0.9200
32864/60000 [===============>..............] - ETA: 47s - loss: 0.2617 - categorical_accuracy: 0.9200
32896/60000 [===============>..............] - ETA: 47s - loss: 0.2614 - categorical_accuracy: 0.9201
32928/60000 [===============>..............] - ETA: 47s - loss: 0.2613 - categorical_accuracy: 0.9201
32960/60000 [===============>..............] - ETA: 47s - loss: 0.2611 - categorical_accuracy: 0.9201
32992/60000 [===============>..............] - ETA: 47s - loss: 0.2609 - categorical_accuracy: 0.9202
33024/60000 [===============>..............] - ETA: 47s - loss: 0.2608 - categorical_accuracy: 0.9202
33056/60000 [===============>..............] - ETA: 47s - loss: 0.2607 - categorical_accuracy: 0.9202
33088/60000 [===============>..............] - ETA: 47s - loss: 0.2605 - categorical_accuracy: 0.9203
33120/60000 [===============>..............] - ETA: 47s - loss: 0.2604 - categorical_accuracy: 0.9204
33152/60000 [===============>..............] - ETA: 47s - loss: 0.2603 - categorical_accuracy: 0.9204
33184/60000 [===============>..............] - ETA: 47s - loss: 0.2605 - categorical_accuracy: 0.9203
33216/60000 [===============>..............] - ETA: 46s - loss: 0.2602 - categorical_accuracy: 0.9204
33248/60000 [===============>..............] - ETA: 46s - loss: 0.2600 - categorical_accuracy: 0.9204
33280/60000 [===============>..............] - ETA: 46s - loss: 0.2600 - categorical_accuracy: 0.9204
33312/60000 [===============>..............] - ETA: 46s - loss: 0.2598 - categorical_accuracy: 0.9205
33344/60000 [===============>..............] - ETA: 46s - loss: 0.2596 - categorical_accuracy: 0.9206
33376/60000 [===============>..............] - ETA: 46s - loss: 0.2595 - categorical_accuracy: 0.9206
33408/60000 [===============>..............] - ETA: 46s - loss: 0.2593 - categorical_accuracy: 0.9206
33440/60000 [===============>..............] - ETA: 46s - loss: 0.2593 - categorical_accuracy: 0.9206
33472/60000 [===============>..............] - ETA: 46s - loss: 0.2591 - categorical_accuracy: 0.9207
33504/60000 [===============>..............] - ETA: 46s - loss: 0.2590 - categorical_accuracy: 0.9208
33536/60000 [===============>..............] - ETA: 46s - loss: 0.2590 - categorical_accuracy: 0.9208
33568/60000 [===============>..............] - ETA: 46s - loss: 0.2589 - categorical_accuracy: 0.9208
33600/60000 [===============>..............] - ETA: 46s - loss: 0.2588 - categorical_accuracy: 0.9208
33632/60000 [===============>..............] - ETA: 46s - loss: 0.2587 - categorical_accuracy: 0.9209
33664/60000 [===============>..............] - ETA: 46s - loss: 0.2586 - categorical_accuracy: 0.9209
33696/60000 [===============>..............] - ETA: 46s - loss: 0.2583 - categorical_accuracy: 0.9210
33728/60000 [===============>..............] - ETA: 46s - loss: 0.2581 - categorical_accuracy: 0.9211
33760/60000 [===============>..............] - ETA: 46s - loss: 0.2580 - categorical_accuracy: 0.9211
33792/60000 [===============>..............] - ETA: 45s - loss: 0.2579 - categorical_accuracy: 0.9211
33824/60000 [===============>..............] - ETA: 45s - loss: 0.2577 - categorical_accuracy: 0.9212
33856/60000 [===============>..............] - ETA: 45s - loss: 0.2576 - categorical_accuracy: 0.9212
33888/60000 [===============>..............] - ETA: 45s - loss: 0.2574 - categorical_accuracy: 0.9212
33920/60000 [===============>..............] - ETA: 45s - loss: 0.2573 - categorical_accuracy: 0.9213
33952/60000 [===============>..............] - ETA: 45s - loss: 0.2571 - categorical_accuracy: 0.9213
33984/60000 [===============>..............] - ETA: 45s - loss: 0.2569 - categorical_accuracy: 0.9213
34016/60000 [================>.............] - ETA: 45s - loss: 0.2570 - categorical_accuracy: 0.9214
34048/60000 [================>.............] - ETA: 45s - loss: 0.2569 - categorical_accuracy: 0.9214
34080/60000 [================>.............] - ETA: 45s - loss: 0.2567 - categorical_accuracy: 0.9215
34112/60000 [================>.............] - ETA: 45s - loss: 0.2565 - categorical_accuracy: 0.9216
34144/60000 [================>.............] - ETA: 45s - loss: 0.2564 - categorical_accuracy: 0.9216
34176/60000 [================>.............] - ETA: 45s - loss: 0.2562 - categorical_accuracy: 0.9217
34208/60000 [================>.............] - ETA: 45s - loss: 0.2559 - categorical_accuracy: 0.9218
34240/60000 [================>.............] - ETA: 45s - loss: 0.2557 - categorical_accuracy: 0.9218
34272/60000 [================>.............] - ETA: 45s - loss: 0.2556 - categorical_accuracy: 0.9219
34304/60000 [================>.............] - ETA: 45s - loss: 0.2555 - categorical_accuracy: 0.9219
34336/60000 [================>.............] - ETA: 44s - loss: 0.2553 - categorical_accuracy: 0.9219
34368/60000 [================>.............] - ETA: 44s - loss: 0.2552 - categorical_accuracy: 0.9220
34400/60000 [================>.............] - ETA: 44s - loss: 0.2550 - categorical_accuracy: 0.9221
34432/60000 [================>.............] - ETA: 44s - loss: 0.2548 - categorical_accuracy: 0.9221
34464/60000 [================>.............] - ETA: 44s - loss: 0.2546 - categorical_accuracy: 0.9222
34496/60000 [================>.............] - ETA: 44s - loss: 0.2545 - categorical_accuracy: 0.9222
34528/60000 [================>.............] - ETA: 44s - loss: 0.2543 - categorical_accuracy: 0.9223
34560/60000 [================>.............] - ETA: 44s - loss: 0.2541 - categorical_accuracy: 0.9224
34592/60000 [================>.............] - ETA: 44s - loss: 0.2538 - categorical_accuracy: 0.9224
34624/60000 [================>.............] - ETA: 44s - loss: 0.2538 - categorical_accuracy: 0.9225
34656/60000 [================>.............] - ETA: 44s - loss: 0.2537 - categorical_accuracy: 0.9225
34688/60000 [================>.............] - ETA: 44s - loss: 0.2536 - categorical_accuracy: 0.9225
34720/60000 [================>.............] - ETA: 44s - loss: 0.2534 - categorical_accuracy: 0.9226
34752/60000 [================>.............] - ETA: 44s - loss: 0.2533 - categorical_accuracy: 0.9227
34784/60000 [================>.............] - ETA: 44s - loss: 0.2531 - categorical_accuracy: 0.9227
34816/60000 [================>.............] - ETA: 44s - loss: 0.2529 - categorical_accuracy: 0.9228
34848/60000 [================>.............] - ETA: 44s - loss: 0.2530 - categorical_accuracy: 0.9228
34880/60000 [================>.............] - ETA: 44s - loss: 0.2529 - categorical_accuracy: 0.9228
34912/60000 [================>.............] - ETA: 44s - loss: 0.2527 - categorical_accuracy: 0.9229
34944/60000 [================>.............] - ETA: 43s - loss: 0.2527 - categorical_accuracy: 0.9229
34976/60000 [================>.............] - ETA: 43s - loss: 0.2525 - categorical_accuracy: 0.9229
35008/60000 [================>.............] - ETA: 43s - loss: 0.2523 - categorical_accuracy: 0.9230
35040/60000 [================>.............] - ETA: 43s - loss: 0.2522 - categorical_accuracy: 0.9230
35072/60000 [================>.............] - ETA: 43s - loss: 0.2521 - categorical_accuracy: 0.9230
35104/60000 [================>.............] - ETA: 43s - loss: 0.2520 - categorical_accuracy: 0.9230
35136/60000 [================>.............] - ETA: 43s - loss: 0.2518 - categorical_accuracy: 0.9231
35168/60000 [================>.............] - ETA: 43s - loss: 0.2516 - categorical_accuracy: 0.9232
35200/60000 [================>.............] - ETA: 43s - loss: 0.2514 - categorical_accuracy: 0.9232
35232/60000 [================>.............] - ETA: 43s - loss: 0.2512 - categorical_accuracy: 0.9233
35264/60000 [================>.............] - ETA: 43s - loss: 0.2511 - categorical_accuracy: 0.9233
35296/60000 [================>.............] - ETA: 43s - loss: 0.2511 - categorical_accuracy: 0.9234
35328/60000 [================>.............] - ETA: 43s - loss: 0.2510 - categorical_accuracy: 0.9234
35360/60000 [================>.............] - ETA: 43s - loss: 0.2510 - categorical_accuracy: 0.9234
35392/60000 [================>.............] - ETA: 43s - loss: 0.2511 - categorical_accuracy: 0.9233
35424/60000 [================>.............] - ETA: 43s - loss: 0.2511 - categorical_accuracy: 0.9233
35456/60000 [================>.............] - ETA: 43s - loss: 0.2509 - categorical_accuracy: 0.9233
35488/60000 [================>.............] - ETA: 42s - loss: 0.2508 - categorical_accuracy: 0.9234
35520/60000 [================>.............] - ETA: 42s - loss: 0.2506 - categorical_accuracy: 0.9235
35552/60000 [================>.............] - ETA: 42s - loss: 0.2504 - categorical_accuracy: 0.9235
35584/60000 [================>.............] - ETA: 42s - loss: 0.2503 - categorical_accuracy: 0.9236
35616/60000 [================>.............] - ETA: 42s - loss: 0.2501 - categorical_accuracy: 0.9236
35648/60000 [================>.............] - ETA: 42s - loss: 0.2499 - categorical_accuracy: 0.9237
35680/60000 [================>.............] - ETA: 42s - loss: 0.2498 - categorical_accuracy: 0.9237
35712/60000 [================>.............] - ETA: 42s - loss: 0.2497 - categorical_accuracy: 0.9237
35744/60000 [================>.............] - ETA: 42s - loss: 0.2495 - categorical_accuracy: 0.9238
35776/60000 [================>.............] - ETA: 42s - loss: 0.2496 - categorical_accuracy: 0.9237
35808/60000 [================>.............] - ETA: 42s - loss: 0.2494 - categorical_accuracy: 0.9238
35840/60000 [================>.............] - ETA: 42s - loss: 0.2493 - categorical_accuracy: 0.9239
35872/60000 [================>.............] - ETA: 42s - loss: 0.2492 - categorical_accuracy: 0.9239
35904/60000 [================>.............] - ETA: 42s - loss: 0.2491 - categorical_accuracy: 0.9239
35936/60000 [================>.............] - ETA: 42s - loss: 0.2489 - categorical_accuracy: 0.9240
35968/60000 [================>.............] - ETA: 42s - loss: 0.2489 - categorical_accuracy: 0.9240
36000/60000 [=================>............] - ETA: 42s - loss: 0.2491 - categorical_accuracy: 0.9239
36032/60000 [=================>............] - ETA: 42s - loss: 0.2489 - categorical_accuracy: 0.9240
36064/60000 [=================>............] - ETA: 41s - loss: 0.2489 - categorical_accuracy: 0.9239
36096/60000 [=================>............] - ETA: 41s - loss: 0.2487 - categorical_accuracy: 0.9240
36128/60000 [=================>............] - ETA: 41s - loss: 0.2485 - categorical_accuracy: 0.9240
36192/60000 [=================>............] - ETA: 41s - loss: 0.2482 - categorical_accuracy: 0.9241
36224/60000 [=================>............] - ETA: 41s - loss: 0.2483 - categorical_accuracy: 0.9241
36256/60000 [=================>............] - ETA: 41s - loss: 0.2482 - categorical_accuracy: 0.9241
36288/60000 [=================>............] - ETA: 41s - loss: 0.2481 - categorical_accuracy: 0.9241
36320/60000 [=================>............] - ETA: 41s - loss: 0.2480 - categorical_accuracy: 0.9242
36352/60000 [=================>............] - ETA: 41s - loss: 0.2478 - categorical_accuracy: 0.9242
36384/60000 [=================>............] - ETA: 41s - loss: 0.2477 - categorical_accuracy: 0.9243
36416/60000 [=================>............] - ETA: 41s - loss: 0.2475 - categorical_accuracy: 0.9243
36448/60000 [=================>............] - ETA: 41s - loss: 0.2478 - categorical_accuracy: 0.9242
36480/60000 [=================>............] - ETA: 41s - loss: 0.2477 - categorical_accuracy: 0.9243
36512/60000 [=================>............] - ETA: 41s - loss: 0.2478 - categorical_accuracy: 0.9242
36544/60000 [=================>............] - ETA: 41s - loss: 0.2479 - categorical_accuracy: 0.9242
36576/60000 [=================>............] - ETA: 41s - loss: 0.2478 - categorical_accuracy: 0.9242
36608/60000 [=================>............] - ETA: 41s - loss: 0.2478 - categorical_accuracy: 0.9243
36640/60000 [=================>............] - ETA: 40s - loss: 0.2477 - categorical_accuracy: 0.9243
36672/60000 [=================>............] - ETA: 40s - loss: 0.2476 - categorical_accuracy: 0.9243
36704/60000 [=================>............] - ETA: 40s - loss: 0.2474 - categorical_accuracy: 0.9243
36736/60000 [=================>............] - ETA: 40s - loss: 0.2473 - categorical_accuracy: 0.9244
36768/60000 [=================>............] - ETA: 40s - loss: 0.2472 - categorical_accuracy: 0.9244
36800/60000 [=================>............] - ETA: 40s - loss: 0.2472 - categorical_accuracy: 0.9244
36832/60000 [=================>............] - ETA: 40s - loss: 0.2471 - categorical_accuracy: 0.9244
36864/60000 [=================>............] - ETA: 40s - loss: 0.2471 - categorical_accuracy: 0.9245
36896/60000 [=================>............] - ETA: 40s - loss: 0.2469 - categorical_accuracy: 0.9245
36928/60000 [=================>............] - ETA: 40s - loss: 0.2468 - categorical_accuracy: 0.9246
36960/60000 [=================>............] - ETA: 40s - loss: 0.2467 - categorical_accuracy: 0.9246
36992/60000 [=================>............] - ETA: 40s - loss: 0.2466 - categorical_accuracy: 0.9247
37024/60000 [=================>............] - ETA: 40s - loss: 0.2464 - categorical_accuracy: 0.9247
37056/60000 [=================>............] - ETA: 40s - loss: 0.2462 - categorical_accuracy: 0.9248
37088/60000 [=================>............] - ETA: 40s - loss: 0.2461 - categorical_accuracy: 0.9248
37120/60000 [=================>............] - ETA: 40s - loss: 0.2461 - categorical_accuracy: 0.9248
37152/60000 [=================>............] - ETA: 40s - loss: 0.2459 - categorical_accuracy: 0.9249
37184/60000 [=================>............] - ETA: 40s - loss: 0.2458 - categorical_accuracy: 0.9249
37216/60000 [=================>............] - ETA: 39s - loss: 0.2456 - categorical_accuracy: 0.9250
37248/60000 [=================>............] - ETA: 39s - loss: 0.2455 - categorical_accuracy: 0.9250
37280/60000 [=================>............] - ETA: 39s - loss: 0.2453 - categorical_accuracy: 0.9251
37312/60000 [=================>............] - ETA: 39s - loss: 0.2452 - categorical_accuracy: 0.9251
37344/60000 [=================>............] - ETA: 39s - loss: 0.2451 - categorical_accuracy: 0.9251
37376/60000 [=================>............] - ETA: 39s - loss: 0.2449 - categorical_accuracy: 0.9251
37408/60000 [=================>............] - ETA: 39s - loss: 0.2448 - categorical_accuracy: 0.9252
37440/60000 [=================>............] - ETA: 39s - loss: 0.2446 - categorical_accuracy: 0.9252
37472/60000 [=================>............] - ETA: 39s - loss: 0.2446 - categorical_accuracy: 0.9253
37504/60000 [=================>............] - ETA: 39s - loss: 0.2444 - categorical_accuracy: 0.9253
37536/60000 [=================>............] - ETA: 39s - loss: 0.2442 - categorical_accuracy: 0.9254
37568/60000 [=================>............] - ETA: 39s - loss: 0.2443 - categorical_accuracy: 0.9254
37600/60000 [=================>............] - ETA: 39s - loss: 0.2441 - categorical_accuracy: 0.9255
37632/60000 [=================>............] - ETA: 39s - loss: 0.2439 - categorical_accuracy: 0.9255
37664/60000 [=================>............] - ETA: 39s - loss: 0.2438 - categorical_accuracy: 0.9255
37696/60000 [=================>............] - ETA: 39s - loss: 0.2437 - categorical_accuracy: 0.9255
37728/60000 [=================>............] - ETA: 39s - loss: 0.2436 - categorical_accuracy: 0.9256
37760/60000 [=================>............] - ETA: 39s - loss: 0.2435 - categorical_accuracy: 0.9256
37792/60000 [=================>............] - ETA: 38s - loss: 0.2434 - categorical_accuracy: 0.9256
37824/60000 [=================>............] - ETA: 38s - loss: 0.2433 - categorical_accuracy: 0.9257
37856/60000 [=================>............] - ETA: 38s - loss: 0.2431 - categorical_accuracy: 0.9257
37888/60000 [=================>............] - ETA: 38s - loss: 0.2429 - categorical_accuracy: 0.9258
37920/60000 [=================>............] - ETA: 38s - loss: 0.2427 - categorical_accuracy: 0.9258
37952/60000 [=================>............] - ETA: 38s - loss: 0.2427 - categorical_accuracy: 0.9259
37984/60000 [=================>............] - ETA: 38s - loss: 0.2425 - categorical_accuracy: 0.9259
38016/60000 [==================>...........] - ETA: 38s - loss: 0.2425 - categorical_accuracy: 0.9259
38048/60000 [==================>...........] - ETA: 38s - loss: 0.2424 - categorical_accuracy: 0.9259
38080/60000 [==================>...........] - ETA: 38s - loss: 0.2425 - categorical_accuracy: 0.9259
38112/60000 [==================>...........] - ETA: 38s - loss: 0.2424 - categorical_accuracy: 0.9259
38144/60000 [==================>...........] - ETA: 38s - loss: 0.2423 - categorical_accuracy: 0.9259
38176/60000 [==================>...........] - ETA: 38s - loss: 0.2421 - categorical_accuracy: 0.9260
38208/60000 [==================>...........] - ETA: 38s - loss: 0.2420 - categorical_accuracy: 0.9260
38240/60000 [==================>...........] - ETA: 38s - loss: 0.2419 - categorical_accuracy: 0.9260
38272/60000 [==================>...........] - ETA: 38s - loss: 0.2417 - categorical_accuracy: 0.9261
38304/60000 [==================>...........] - ETA: 38s - loss: 0.2417 - categorical_accuracy: 0.9261
38336/60000 [==================>...........] - ETA: 38s - loss: 0.2417 - categorical_accuracy: 0.9261
38368/60000 [==================>...........] - ETA: 37s - loss: 0.2416 - categorical_accuracy: 0.9261
38400/60000 [==================>...........] - ETA: 37s - loss: 0.2414 - categorical_accuracy: 0.9262
38432/60000 [==================>...........] - ETA: 37s - loss: 0.2413 - categorical_accuracy: 0.9262
38464/60000 [==================>...........] - ETA: 37s - loss: 0.2412 - categorical_accuracy: 0.9262
38496/60000 [==================>...........] - ETA: 37s - loss: 0.2410 - categorical_accuracy: 0.9263
38528/60000 [==================>...........] - ETA: 37s - loss: 0.2409 - categorical_accuracy: 0.9263
38560/60000 [==================>...........] - ETA: 37s - loss: 0.2408 - categorical_accuracy: 0.9263
38592/60000 [==================>...........] - ETA: 37s - loss: 0.2410 - categorical_accuracy: 0.9263
38624/60000 [==================>...........] - ETA: 37s - loss: 0.2408 - categorical_accuracy: 0.9263
38656/60000 [==================>...........] - ETA: 37s - loss: 0.2407 - categorical_accuracy: 0.9264
38688/60000 [==================>...........] - ETA: 37s - loss: 0.2405 - categorical_accuracy: 0.9265
38720/60000 [==================>...........] - ETA: 37s - loss: 0.2405 - categorical_accuracy: 0.9265
38752/60000 [==================>...........] - ETA: 37s - loss: 0.2404 - categorical_accuracy: 0.9265
38784/60000 [==================>...........] - ETA: 37s - loss: 0.2402 - categorical_accuracy: 0.9266
38816/60000 [==================>...........] - ETA: 37s - loss: 0.2402 - categorical_accuracy: 0.9266
38848/60000 [==================>...........] - ETA: 37s - loss: 0.2400 - categorical_accuracy: 0.9266
38880/60000 [==================>...........] - ETA: 37s - loss: 0.2399 - categorical_accuracy: 0.9266
38912/60000 [==================>...........] - ETA: 36s - loss: 0.2398 - categorical_accuracy: 0.9266
38944/60000 [==================>...........] - ETA: 36s - loss: 0.2398 - categorical_accuracy: 0.9266
38976/60000 [==================>...........] - ETA: 36s - loss: 0.2396 - categorical_accuracy: 0.9267
39008/60000 [==================>...........] - ETA: 36s - loss: 0.2395 - categorical_accuracy: 0.9267
39040/60000 [==================>...........] - ETA: 36s - loss: 0.2396 - categorical_accuracy: 0.9268
39072/60000 [==================>...........] - ETA: 36s - loss: 0.2394 - categorical_accuracy: 0.9268
39104/60000 [==================>...........] - ETA: 36s - loss: 0.2396 - categorical_accuracy: 0.9268
39136/60000 [==================>...........] - ETA: 36s - loss: 0.2394 - categorical_accuracy: 0.9268
39168/60000 [==================>...........] - ETA: 36s - loss: 0.2395 - categorical_accuracy: 0.9269
39200/60000 [==================>...........] - ETA: 36s - loss: 0.2395 - categorical_accuracy: 0.9269
39232/60000 [==================>...........] - ETA: 36s - loss: 0.2393 - categorical_accuracy: 0.9269
39264/60000 [==================>...........] - ETA: 36s - loss: 0.2391 - categorical_accuracy: 0.9270
39296/60000 [==================>...........] - ETA: 36s - loss: 0.2390 - categorical_accuracy: 0.9270
39328/60000 [==================>...........] - ETA: 36s - loss: 0.2389 - categorical_accuracy: 0.9271
39360/60000 [==================>...........] - ETA: 36s - loss: 0.2388 - categorical_accuracy: 0.9271
39392/60000 [==================>...........] - ETA: 36s - loss: 0.2387 - categorical_accuracy: 0.9272
39424/60000 [==================>...........] - ETA: 36s - loss: 0.2385 - categorical_accuracy: 0.9272
39456/60000 [==================>...........] - ETA: 36s - loss: 0.2384 - categorical_accuracy: 0.9273
39488/60000 [==================>...........] - ETA: 35s - loss: 0.2383 - categorical_accuracy: 0.9273
39520/60000 [==================>...........] - ETA: 35s - loss: 0.2382 - categorical_accuracy: 0.9273
39552/60000 [==================>...........] - ETA: 35s - loss: 0.2385 - categorical_accuracy: 0.9273
39584/60000 [==================>...........] - ETA: 35s - loss: 0.2383 - categorical_accuracy: 0.9273
39616/60000 [==================>...........] - ETA: 35s - loss: 0.2382 - categorical_accuracy: 0.9274
39648/60000 [==================>...........] - ETA: 35s - loss: 0.2381 - categorical_accuracy: 0.9274
39680/60000 [==================>...........] - ETA: 35s - loss: 0.2379 - categorical_accuracy: 0.9275
39712/60000 [==================>...........] - ETA: 35s - loss: 0.2378 - categorical_accuracy: 0.9275
39744/60000 [==================>...........] - ETA: 35s - loss: 0.2376 - categorical_accuracy: 0.9276
39776/60000 [==================>...........] - ETA: 35s - loss: 0.2375 - categorical_accuracy: 0.9276
39808/60000 [==================>...........] - ETA: 35s - loss: 0.2373 - categorical_accuracy: 0.9277
39840/60000 [==================>...........] - ETA: 35s - loss: 0.2371 - categorical_accuracy: 0.9277
39872/60000 [==================>...........] - ETA: 35s - loss: 0.2370 - categorical_accuracy: 0.9278
39904/60000 [==================>...........] - ETA: 35s - loss: 0.2370 - categorical_accuracy: 0.9278
39936/60000 [==================>...........] - ETA: 35s - loss: 0.2369 - categorical_accuracy: 0.9278
39968/60000 [==================>...........] - ETA: 35s - loss: 0.2367 - categorical_accuracy: 0.9278
40000/60000 [===================>..........] - ETA: 35s - loss: 0.2365 - categorical_accuracy: 0.9279
40032/60000 [===================>..........] - ETA: 35s - loss: 0.2364 - categorical_accuracy: 0.9279
40064/60000 [===================>..........] - ETA: 34s - loss: 0.2362 - categorical_accuracy: 0.9280
40096/60000 [===================>..........] - ETA: 34s - loss: 0.2365 - categorical_accuracy: 0.9280
40128/60000 [===================>..........] - ETA: 34s - loss: 0.2363 - categorical_accuracy: 0.9280
40160/60000 [===================>..........] - ETA: 34s - loss: 0.2363 - categorical_accuracy: 0.9280
40192/60000 [===================>..........] - ETA: 34s - loss: 0.2362 - categorical_accuracy: 0.9281
40224/60000 [===================>..........] - ETA: 34s - loss: 0.2361 - categorical_accuracy: 0.9281
40256/60000 [===================>..........] - ETA: 34s - loss: 0.2360 - categorical_accuracy: 0.9282
40288/60000 [===================>..........] - ETA: 34s - loss: 0.2359 - categorical_accuracy: 0.9282
40320/60000 [===================>..........] - ETA: 34s - loss: 0.2359 - categorical_accuracy: 0.9282
40352/60000 [===================>..........] - ETA: 34s - loss: 0.2357 - categorical_accuracy: 0.9282
40384/60000 [===================>..........] - ETA: 34s - loss: 0.2356 - categorical_accuracy: 0.9282
40416/60000 [===================>..........] - ETA: 34s - loss: 0.2356 - categorical_accuracy: 0.9282
40448/60000 [===================>..........] - ETA: 34s - loss: 0.2354 - categorical_accuracy: 0.9283
40480/60000 [===================>..........] - ETA: 34s - loss: 0.2352 - categorical_accuracy: 0.9283
40512/60000 [===================>..........] - ETA: 34s - loss: 0.2351 - categorical_accuracy: 0.9284
40544/60000 [===================>..........] - ETA: 34s - loss: 0.2350 - categorical_accuracy: 0.9284
40576/60000 [===================>..........] - ETA: 34s - loss: 0.2349 - categorical_accuracy: 0.9284
40608/60000 [===================>..........] - ETA: 33s - loss: 0.2348 - categorical_accuracy: 0.9285
40640/60000 [===================>..........] - ETA: 33s - loss: 0.2347 - categorical_accuracy: 0.9285
40672/60000 [===================>..........] - ETA: 33s - loss: 0.2345 - categorical_accuracy: 0.9285
40704/60000 [===================>..........] - ETA: 33s - loss: 0.2343 - categorical_accuracy: 0.9286
40736/60000 [===================>..........] - ETA: 33s - loss: 0.2342 - categorical_accuracy: 0.9286
40768/60000 [===================>..........] - ETA: 33s - loss: 0.2341 - categorical_accuracy: 0.9287
40800/60000 [===================>..........] - ETA: 33s - loss: 0.2340 - categorical_accuracy: 0.9287
40832/60000 [===================>..........] - ETA: 33s - loss: 0.2338 - categorical_accuracy: 0.9287
40864/60000 [===================>..........] - ETA: 33s - loss: 0.2338 - categorical_accuracy: 0.9288
40896/60000 [===================>..........] - ETA: 33s - loss: 0.2339 - categorical_accuracy: 0.9288
40928/60000 [===================>..........] - ETA: 33s - loss: 0.2338 - categorical_accuracy: 0.9288
40960/60000 [===================>..........] - ETA: 33s - loss: 0.2339 - categorical_accuracy: 0.9289
40992/60000 [===================>..........] - ETA: 33s - loss: 0.2339 - categorical_accuracy: 0.9289
41024/60000 [===================>..........] - ETA: 33s - loss: 0.2337 - categorical_accuracy: 0.9289
41056/60000 [===================>..........] - ETA: 33s - loss: 0.2336 - categorical_accuracy: 0.9290
41088/60000 [===================>..........] - ETA: 33s - loss: 0.2334 - categorical_accuracy: 0.9290
41120/60000 [===================>..........] - ETA: 33s - loss: 0.2333 - categorical_accuracy: 0.9291
41152/60000 [===================>..........] - ETA: 33s - loss: 0.2332 - categorical_accuracy: 0.9291
41184/60000 [===================>..........] - ETA: 32s - loss: 0.2330 - categorical_accuracy: 0.9291
41216/60000 [===================>..........] - ETA: 32s - loss: 0.2329 - categorical_accuracy: 0.9292
41248/60000 [===================>..........] - ETA: 32s - loss: 0.2329 - categorical_accuracy: 0.9292
41280/60000 [===================>..........] - ETA: 32s - loss: 0.2328 - categorical_accuracy: 0.9292
41312/60000 [===================>..........] - ETA: 32s - loss: 0.2327 - categorical_accuracy: 0.9292
41344/60000 [===================>..........] - ETA: 32s - loss: 0.2329 - categorical_accuracy: 0.9292
41376/60000 [===================>..........] - ETA: 32s - loss: 0.2330 - categorical_accuracy: 0.9292
41408/60000 [===================>..........] - ETA: 32s - loss: 0.2328 - categorical_accuracy: 0.9292
41440/60000 [===================>..........] - ETA: 32s - loss: 0.2327 - categorical_accuracy: 0.9293
41472/60000 [===================>..........] - ETA: 32s - loss: 0.2325 - categorical_accuracy: 0.9293
41504/60000 [===================>..........] - ETA: 32s - loss: 0.2325 - categorical_accuracy: 0.9294
41536/60000 [===================>..........] - ETA: 32s - loss: 0.2324 - categorical_accuracy: 0.9294
41568/60000 [===================>..........] - ETA: 32s - loss: 0.2323 - categorical_accuracy: 0.9294
41600/60000 [===================>..........] - ETA: 32s - loss: 0.2322 - categorical_accuracy: 0.9294
41632/60000 [===================>..........] - ETA: 32s - loss: 0.2321 - categorical_accuracy: 0.9294
41664/60000 [===================>..........] - ETA: 32s - loss: 0.2320 - categorical_accuracy: 0.9295
41696/60000 [===================>..........] - ETA: 32s - loss: 0.2318 - categorical_accuracy: 0.9295
41728/60000 [===================>..........] - ETA: 32s - loss: 0.2317 - categorical_accuracy: 0.9295
41760/60000 [===================>..........] - ETA: 31s - loss: 0.2316 - categorical_accuracy: 0.9295
41792/60000 [===================>..........] - ETA: 31s - loss: 0.2315 - categorical_accuracy: 0.9296
41824/60000 [===================>..........] - ETA: 31s - loss: 0.2314 - categorical_accuracy: 0.9296
41856/60000 [===================>..........] - ETA: 31s - loss: 0.2312 - categorical_accuracy: 0.9297
41888/60000 [===================>..........] - ETA: 31s - loss: 0.2313 - categorical_accuracy: 0.9297
41920/60000 [===================>..........] - ETA: 31s - loss: 0.2312 - categorical_accuracy: 0.9297
41952/60000 [===================>..........] - ETA: 31s - loss: 0.2311 - categorical_accuracy: 0.9298
41984/60000 [===================>..........] - ETA: 31s - loss: 0.2310 - categorical_accuracy: 0.9298
42016/60000 [====================>.........] - ETA: 31s - loss: 0.2308 - categorical_accuracy: 0.9298
42048/60000 [====================>.........] - ETA: 31s - loss: 0.2307 - categorical_accuracy: 0.9299
42080/60000 [====================>.........] - ETA: 31s - loss: 0.2306 - categorical_accuracy: 0.9299
42112/60000 [====================>.........] - ETA: 31s - loss: 0.2305 - categorical_accuracy: 0.9299
42144/60000 [====================>.........] - ETA: 31s - loss: 0.2303 - categorical_accuracy: 0.9300
42176/60000 [====================>.........] - ETA: 31s - loss: 0.2301 - categorical_accuracy: 0.9301
42208/60000 [====================>.........] - ETA: 31s - loss: 0.2300 - categorical_accuracy: 0.9301
42240/60000 [====================>.........] - ETA: 31s - loss: 0.2298 - categorical_accuracy: 0.9301
42272/60000 [====================>.........] - ETA: 31s - loss: 0.2298 - categorical_accuracy: 0.9302
42304/60000 [====================>.........] - ETA: 30s - loss: 0.2298 - categorical_accuracy: 0.9302
42336/60000 [====================>.........] - ETA: 30s - loss: 0.2297 - categorical_accuracy: 0.9302
42368/60000 [====================>.........] - ETA: 30s - loss: 0.2296 - categorical_accuracy: 0.9303
42400/60000 [====================>.........] - ETA: 30s - loss: 0.2295 - categorical_accuracy: 0.9303
42432/60000 [====================>.........] - ETA: 30s - loss: 0.2294 - categorical_accuracy: 0.9303
42464/60000 [====================>.........] - ETA: 30s - loss: 0.2292 - categorical_accuracy: 0.9304
42496/60000 [====================>.........] - ETA: 30s - loss: 0.2291 - categorical_accuracy: 0.9304
42528/60000 [====================>.........] - ETA: 30s - loss: 0.2293 - categorical_accuracy: 0.9304
42560/60000 [====================>.........] - ETA: 30s - loss: 0.2292 - categorical_accuracy: 0.9305
42592/60000 [====================>.........] - ETA: 30s - loss: 0.2290 - categorical_accuracy: 0.9305
42624/60000 [====================>.........] - ETA: 30s - loss: 0.2289 - categorical_accuracy: 0.9306
42656/60000 [====================>.........] - ETA: 30s - loss: 0.2288 - categorical_accuracy: 0.9306
42688/60000 [====================>.........] - ETA: 30s - loss: 0.2287 - categorical_accuracy: 0.9306
42720/60000 [====================>.........] - ETA: 30s - loss: 0.2287 - categorical_accuracy: 0.9306
42752/60000 [====================>.........] - ETA: 30s - loss: 0.2285 - categorical_accuracy: 0.9307
42784/60000 [====================>.........] - ETA: 30s - loss: 0.2284 - categorical_accuracy: 0.9307
42816/60000 [====================>.........] - ETA: 30s - loss: 0.2282 - categorical_accuracy: 0.9308
42848/60000 [====================>.........] - ETA: 30s - loss: 0.2281 - categorical_accuracy: 0.9308
42880/60000 [====================>.........] - ETA: 29s - loss: 0.2280 - categorical_accuracy: 0.9309
42912/60000 [====================>.........] - ETA: 29s - loss: 0.2279 - categorical_accuracy: 0.9309
42944/60000 [====================>.........] - ETA: 29s - loss: 0.2278 - categorical_accuracy: 0.9309
42976/60000 [====================>.........] - ETA: 29s - loss: 0.2277 - categorical_accuracy: 0.9309
43008/60000 [====================>.........] - ETA: 29s - loss: 0.2277 - categorical_accuracy: 0.9310
43040/60000 [====================>.........] - ETA: 29s - loss: 0.2276 - categorical_accuracy: 0.9310
43072/60000 [====================>.........] - ETA: 29s - loss: 0.2274 - categorical_accuracy: 0.9310
43104/60000 [====================>.........] - ETA: 29s - loss: 0.2273 - categorical_accuracy: 0.9311
43136/60000 [====================>.........] - ETA: 29s - loss: 0.2271 - categorical_accuracy: 0.9311
43168/60000 [====================>.........] - ETA: 29s - loss: 0.2273 - categorical_accuracy: 0.9311
43200/60000 [====================>.........] - ETA: 29s - loss: 0.2271 - categorical_accuracy: 0.9312
43232/60000 [====================>.........] - ETA: 29s - loss: 0.2270 - categorical_accuracy: 0.9312
43264/60000 [====================>.........] - ETA: 29s - loss: 0.2269 - categorical_accuracy: 0.9312
43296/60000 [====================>.........] - ETA: 29s - loss: 0.2271 - categorical_accuracy: 0.9312
43328/60000 [====================>.........] - ETA: 29s - loss: 0.2270 - categorical_accuracy: 0.9312
43360/60000 [====================>.........] - ETA: 29s - loss: 0.2269 - categorical_accuracy: 0.9313
43392/60000 [====================>.........] - ETA: 29s - loss: 0.2268 - categorical_accuracy: 0.9313
43424/60000 [====================>.........] - ETA: 29s - loss: 0.2267 - categorical_accuracy: 0.9313
43456/60000 [====================>.........] - ETA: 28s - loss: 0.2266 - categorical_accuracy: 0.9313
43488/60000 [====================>.........] - ETA: 28s - loss: 0.2265 - categorical_accuracy: 0.9313
43520/60000 [====================>.........] - ETA: 28s - loss: 0.2263 - categorical_accuracy: 0.9314
43552/60000 [====================>.........] - ETA: 28s - loss: 0.2262 - categorical_accuracy: 0.9314
43584/60000 [====================>.........] - ETA: 28s - loss: 0.2261 - categorical_accuracy: 0.9315
43616/60000 [====================>.........] - ETA: 28s - loss: 0.2260 - categorical_accuracy: 0.9315
43648/60000 [====================>.........] - ETA: 28s - loss: 0.2259 - categorical_accuracy: 0.9315
43680/60000 [====================>.........] - ETA: 28s - loss: 0.2259 - categorical_accuracy: 0.9315
43712/60000 [====================>.........] - ETA: 28s - loss: 0.2259 - categorical_accuracy: 0.9315
43744/60000 [====================>.........] - ETA: 28s - loss: 0.2257 - categorical_accuracy: 0.9316
43776/60000 [====================>.........] - ETA: 28s - loss: 0.2256 - categorical_accuracy: 0.9316
43808/60000 [====================>.........] - ETA: 28s - loss: 0.2256 - categorical_accuracy: 0.9316
43840/60000 [====================>.........] - ETA: 28s - loss: 0.2256 - categorical_accuracy: 0.9316
43872/60000 [====================>.........] - ETA: 28s - loss: 0.2256 - categorical_accuracy: 0.9316
43904/60000 [====================>.........] - ETA: 28s - loss: 0.2255 - categorical_accuracy: 0.9317
43936/60000 [====================>.........] - ETA: 28s - loss: 0.2253 - categorical_accuracy: 0.9317
43968/60000 [====================>.........] - ETA: 28s - loss: 0.2252 - categorical_accuracy: 0.9318
44000/60000 [=====================>........] - ETA: 28s - loss: 0.2251 - categorical_accuracy: 0.9318
44032/60000 [=====================>........] - ETA: 27s - loss: 0.2249 - categorical_accuracy: 0.9319
44064/60000 [=====================>........] - ETA: 27s - loss: 0.2249 - categorical_accuracy: 0.9319
44096/60000 [=====================>........] - ETA: 27s - loss: 0.2248 - categorical_accuracy: 0.9319
44128/60000 [=====================>........] - ETA: 27s - loss: 0.2247 - categorical_accuracy: 0.9319
44160/60000 [=====================>........] - ETA: 27s - loss: 0.2248 - categorical_accuracy: 0.9319
44192/60000 [=====================>........] - ETA: 27s - loss: 0.2247 - categorical_accuracy: 0.9319
44224/60000 [=====================>........] - ETA: 27s - loss: 0.2247 - categorical_accuracy: 0.9320
44256/60000 [=====================>........] - ETA: 27s - loss: 0.2246 - categorical_accuracy: 0.9320
44288/60000 [=====================>........] - ETA: 27s - loss: 0.2245 - categorical_accuracy: 0.9320
44320/60000 [=====================>........] - ETA: 27s - loss: 0.2245 - categorical_accuracy: 0.9320
44352/60000 [=====================>........] - ETA: 27s - loss: 0.2243 - categorical_accuracy: 0.9321
44384/60000 [=====================>........] - ETA: 27s - loss: 0.2242 - categorical_accuracy: 0.9321
44416/60000 [=====================>........] - ETA: 27s - loss: 0.2241 - categorical_accuracy: 0.9321
44448/60000 [=====================>........] - ETA: 27s - loss: 0.2240 - categorical_accuracy: 0.9322
44480/60000 [=====================>........] - ETA: 27s - loss: 0.2240 - categorical_accuracy: 0.9322
44512/60000 [=====================>........] - ETA: 27s - loss: 0.2240 - categorical_accuracy: 0.9322
44544/60000 [=====================>........] - ETA: 27s - loss: 0.2240 - categorical_accuracy: 0.9322
44576/60000 [=====================>........] - ETA: 27s - loss: 0.2238 - categorical_accuracy: 0.9322
44608/60000 [=====================>........] - ETA: 26s - loss: 0.2239 - categorical_accuracy: 0.9322
44640/60000 [=====================>........] - ETA: 26s - loss: 0.2239 - categorical_accuracy: 0.9322
44672/60000 [=====================>........] - ETA: 26s - loss: 0.2238 - categorical_accuracy: 0.9323
44704/60000 [=====================>........] - ETA: 26s - loss: 0.2240 - categorical_accuracy: 0.9323
44736/60000 [=====================>........] - ETA: 26s - loss: 0.2239 - categorical_accuracy: 0.9323
44768/60000 [=====================>........] - ETA: 26s - loss: 0.2239 - categorical_accuracy: 0.9323
44800/60000 [=====================>........] - ETA: 26s - loss: 0.2237 - categorical_accuracy: 0.9324
44832/60000 [=====================>........] - ETA: 26s - loss: 0.2236 - categorical_accuracy: 0.9324
44864/60000 [=====================>........] - ETA: 26s - loss: 0.2236 - categorical_accuracy: 0.9324
44896/60000 [=====================>........] - ETA: 26s - loss: 0.2235 - categorical_accuracy: 0.9325
44928/60000 [=====================>........] - ETA: 26s - loss: 0.2235 - categorical_accuracy: 0.9325
44960/60000 [=====================>........] - ETA: 26s - loss: 0.2234 - categorical_accuracy: 0.9325
44992/60000 [=====================>........] - ETA: 26s - loss: 0.2233 - categorical_accuracy: 0.9325
45024/60000 [=====================>........] - ETA: 26s - loss: 0.2233 - categorical_accuracy: 0.9325
45056/60000 [=====================>........] - ETA: 26s - loss: 0.2232 - categorical_accuracy: 0.9326
45088/60000 [=====================>........] - ETA: 26s - loss: 0.2230 - categorical_accuracy: 0.9326
45120/60000 [=====================>........] - ETA: 26s - loss: 0.2229 - categorical_accuracy: 0.9326
45152/60000 [=====================>........] - ETA: 25s - loss: 0.2228 - categorical_accuracy: 0.9327
45184/60000 [=====================>........] - ETA: 25s - loss: 0.2228 - categorical_accuracy: 0.9327
45216/60000 [=====================>........] - ETA: 25s - loss: 0.2227 - categorical_accuracy: 0.9327
45248/60000 [=====================>........] - ETA: 25s - loss: 0.2226 - categorical_accuracy: 0.9328
45280/60000 [=====================>........] - ETA: 25s - loss: 0.2224 - categorical_accuracy: 0.9328
45312/60000 [=====================>........] - ETA: 25s - loss: 0.2223 - categorical_accuracy: 0.9328
45344/60000 [=====================>........] - ETA: 25s - loss: 0.2222 - categorical_accuracy: 0.9329
45376/60000 [=====================>........] - ETA: 25s - loss: 0.2221 - categorical_accuracy: 0.9329
45408/60000 [=====================>........] - ETA: 25s - loss: 0.2219 - categorical_accuracy: 0.9330
45440/60000 [=====================>........] - ETA: 25s - loss: 0.2218 - categorical_accuracy: 0.9330
45472/60000 [=====================>........] - ETA: 25s - loss: 0.2218 - categorical_accuracy: 0.9330
45504/60000 [=====================>........] - ETA: 25s - loss: 0.2218 - categorical_accuracy: 0.9329
45536/60000 [=====================>........] - ETA: 25s - loss: 0.2217 - categorical_accuracy: 0.9330
45568/60000 [=====================>........] - ETA: 25s - loss: 0.2216 - categorical_accuracy: 0.9330
45600/60000 [=====================>........] - ETA: 25s - loss: 0.2216 - categorical_accuracy: 0.9330
45632/60000 [=====================>........] - ETA: 25s - loss: 0.2215 - categorical_accuracy: 0.9330
45664/60000 [=====================>........] - ETA: 25s - loss: 0.2214 - categorical_accuracy: 0.9331
45696/60000 [=====================>........] - ETA: 25s - loss: 0.2213 - categorical_accuracy: 0.9331
45728/60000 [=====================>........] - ETA: 24s - loss: 0.2213 - categorical_accuracy: 0.9331
45760/60000 [=====================>........] - ETA: 24s - loss: 0.2211 - categorical_accuracy: 0.9331
45792/60000 [=====================>........] - ETA: 24s - loss: 0.2211 - categorical_accuracy: 0.9331
45824/60000 [=====================>........] - ETA: 24s - loss: 0.2210 - categorical_accuracy: 0.9332
45856/60000 [=====================>........] - ETA: 24s - loss: 0.2208 - categorical_accuracy: 0.9332
45888/60000 [=====================>........] - ETA: 24s - loss: 0.2207 - categorical_accuracy: 0.9332
45920/60000 [=====================>........] - ETA: 24s - loss: 0.2206 - categorical_accuracy: 0.9333
45984/60000 [=====================>........] - ETA: 24s - loss: 0.2205 - categorical_accuracy: 0.9333
46016/60000 [======================>.......] - ETA: 24s - loss: 0.2204 - categorical_accuracy: 0.9333
46048/60000 [======================>.......] - ETA: 24s - loss: 0.2203 - categorical_accuracy: 0.9333
46080/60000 [======================>.......] - ETA: 24s - loss: 0.2203 - categorical_accuracy: 0.9333
46112/60000 [======================>.......] - ETA: 24s - loss: 0.2202 - categorical_accuracy: 0.9333
46144/60000 [======================>.......] - ETA: 24s - loss: 0.2202 - categorical_accuracy: 0.9333
46176/60000 [======================>.......] - ETA: 24s - loss: 0.2201 - categorical_accuracy: 0.9333
46208/60000 [======================>.......] - ETA: 24s - loss: 0.2201 - categorical_accuracy: 0.9333
46240/60000 [======================>.......] - ETA: 24s - loss: 0.2200 - categorical_accuracy: 0.9333
46272/60000 [======================>.......] - ETA: 24s - loss: 0.2198 - categorical_accuracy: 0.9334
46304/60000 [======================>.......] - ETA: 23s - loss: 0.2198 - categorical_accuracy: 0.9334
46336/60000 [======================>.......] - ETA: 23s - loss: 0.2197 - categorical_accuracy: 0.9334
46368/60000 [======================>.......] - ETA: 23s - loss: 0.2196 - categorical_accuracy: 0.9334
46400/60000 [======================>.......] - ETA: 23s - loss: 0.2196 - categorical_accuracy: 0.9334
46432/60000 [======================>.......] - ETA: 23s - loss: 0.2195 - categorical_accuracy: 0.9335
46464/60000 [======================>.......] - ETA: 23s - loss: 0.2194 - categorical_accuracy: 0.9335
46496/60000 [======================>.......] - ETA: 23s - loss: 0.2193 - categorical_accuracy: 0.9335
46528/60000 [======================>.......] - ETA: 23s - loss: 0.2193 - categorical_accuracy: 0.9335
46560/60000 [======================>.......] - ETA: 23s - loss: 0.2192 - categorical_accuracy: 0.9335
46592/60000 [======================>.......] - ETA: 23s - loss: 0.2191 - categorical_accuracy: 0.9335
46624/60000 [======================>.......] - ETA: 23s - loss: 0.2192 - categorical_accuracy: 0.9335
46656/60000 [======================>.......] - ETA: 23s - loss: 0.2191 - categorical_accuracy: 0.9335
46688/60000 [======================>.......] - ETA: 23s - loss: 0.2191 - categorical_accuracy: 0.9335
46720/60000 [======================>.......] - ETA: 23s - loss: 0.2190 - categorical_accuracy: 0.9335
46752/60000 [======================>.......] - ETA: 23s - loss: 0.2188 - categorical_accuracy: 0.9336
46784/60000 [======================>.......] - ETA: 23s - loss: 0.2187 - categorical_accuracy: 0.9336
46816/60000 [======================>.......] - ETA: 23s - loss: 0.2186 - categorical_accuracy: 0.9337
46848/60000 [======================>.......] - ETA: 23s - loss: 0.2185 - categorical_accuracy: 0.9337
46880/60000 [======================>.......] - ETA: 22s - loss: 0.2185 - categorical_accuracy: 0.9336
46912/60000 [======================>.......] - ETA: 22s - loss: 0.2185 - categorical_accuracy: 0.9336
46944/60000 [======================>.......] - ETA: 22s - loss: 0.2184 - categorical_accuracy: 0.9336
46976/60000 [======================>.......] - ETA: 22s - loss: 0.2183 - categorical_accuracy: 0.9337
47008/60000 [======================>.......] - ETA: 22s - loss: 0.2183 - categorical_accuracy: 0.9337
47040/60000 [======================>.......] - ETA: 22s - loss: 0.2183 - categorical_accuracy: 0.9337
47072/60000 [======================>.......] - ETA: 22s - loss: 0.2181 - categorical_accuracy: 0.9337
47104/60000 [======================>.......] - ETA: 22s - loss: 0.2181 - categorical_accuracy: 0.9337
47136/60000 [======================>.......] - ETA: 22s - loss: 0.2180 - categorical_accuracy: 0.9337
47168/60000 [======================>.......] - ETA: 22s - loss: 0.2179 - categorical_accuracy: 0.9338
47200/60000 [======================>.......] - ETA: 22s - loss: 0.2178 - categorical_accuracy: 0.9338
47232/60000 [======================>.......] - ETA: 22s - loss: 0.2177 - categorical_accuracy: 0.9338
47264/60000 [======================>.......] - ETA: 22s - loss: 0.2176 - categorical_accuracy: 0.9339
47296/60000 [======================>.......] - ETA: 22s - loss: 0.2175 - categorical_accuracy: 0.9339
47328/60000 [======================>.......] - ETA: 22s - loss: 0.2175 - categorical_accuracy: 0.9339
47360/60000 [======================>.......] - ETA: 22s - loss: 0.2174 - categorical_accuracy: 0.9340
47392/60000 [======================>.......] - ETA: 22s - loss: 0.2172 - categorical_accuracy: 0.9340
47424/60000 [======================>.......] - ETA: 21s - loss: 0.2171 - categorical_accuracy: 0.9341
47456/60000 [======================>.......] - ETA: 21s - loss: 0.2172 - categorical_accuracy: 0.9340
47488/60000 [======================>.......] - ETA: 21s - loss: 0.2171 - categorical_accuracy: 0.9341
47520/60000 [======================>.......] - ETA: 21s - loss: 0.2170 - categorical_accuracy: 0.9341
47552/60000 [======================>.......] - ETA: 21s - loss: 0.2170 - categorical_accuracy: 0.9341
47584/60000 [======================>.......] - ETA: 21s - loss: 0.2169 - categorical_accuracy: 0.9342
47616/60000 [======================>.......] - ETA: 21s - loss: 0.2168 - categorical_accuracy: 0.9342
47648/60000 [======================>.......] - ETA: 21s - loss: 0.2167 - categorical_accuracy: 0.9342
47680/60000 [======================>.......] - ETA: 21s - loss: 0.2166 - categorical_accuracy: 0.9342
47712/60000 [======================>.......] - ETA: 21s - loss: 0.2165 - categorical_accuracy: 0.9343
47744/60000 [======================>.......] - ETA: 21s - loss: 0.2164 - categorical_accuracy: 0.9343
47776/60000 [======================>.......] - ETA: 21s - loss: 0.2164 - categorical_accuracy: 0.9343
47808/60000 [======================>.......] - ETA: 21s - loss: 0.2163 - categorical_accuracy: 0.9343
47840/60000 [======================>.......] - ETA: 21s - loss: 0.2164 - categorical_accuracy: 0.9343
47872/60000 [======================>.......] - ETA: 21s - loss: 0.2163 - categorical_accuracy: 0.9344
47904/60000 [======================>.......] - ETA: 21s - loss: 0.2162 - categorical_accuracy: 0.9344
47936/60000 [======================>.......] - ETA: 21s - loss: 0.2162 - categorical_accuracy: 0.9344
47968/60000 [======================>.......] - ETA: 21s - loss: 0.2160 - categorical_accuracy: 0.9345
48000/60000 [=======================>......] - ETA: 20s - loss: 0.2160 - categorical_accuracy: 0.9344
48032/60000 [=======================>......] - ETA: 20s - loss: 0.2159 - categorical_accuracy: 0.9345
48064/60000 [=======================>......] - ETA: 20s - loss: 0.2158 - categorical_accuracy: 0.9345
48096/60000 [=======================>......] - ETA: 20s - loss: 0.2157 - categorical_accuracy: 0.9345
48128/60000 [=======================>......] - ETA: 20s - loss: 0.2156 - categorical_accuracy: 0.9346
48160/60000 [=======================>......] - ETA: 20s - loss: 0.2156 - categorical_accuracy: 0.9346
48192/60000 [=======================>......] - ETA: 20s - loss: 0.2155 - categorical_accuracy: 0.9346
48224/60000 [=======================>......] - ETA: 20s - loss: 0.2155 - categorical_accuracy: 0.9346
48256/60000 [=======================>......] - ETA: 20s - loss: 0.2153 - categorical_accuracy: 0.9347
48288/60000 [=======================>......] - ETA: 20s - loss: 0.2152 - categorical_accuracy: 0.9347
48320/60000 [=======================>......] - ETA: 20s - loss: 0.2151 - categorical_accuracy: 0.9347
48352/60000 [=======================>......] - ETA: 20s - loss: 0.2150 - categorical_accuracy: 0.9348
48384/60000 [=======================>......] - ETA: 20s - loss: 0.2150 - categorical_accuracy: 0.9348
48416/60000 [=======================>......] - ETA: 20s - loss: 0.2149 - categorical_accuracy: 0.9348
48448/60000 [=======================>......] - ETA: 20s - loss: 0.2147 - categorical_accuracy: 0.9349
48480/60000 [=======================>......] - ETA: 20s - loss: 0.2148 - categorical_accuracy: 0.9349
48512/60000 [=======================>......] - ETA: 20s - loss: 0.2146 - categorical_accuracy: 0.9349
48544/60000 [=======================>......] - ETA: 20s - loss: 0.2147 - categorical_accuracy: 0.9349
48576/60000 [=======================>......] - ETA: 19s - loss: 0.2145 - categorical_accuracy: 0.9350
48608/60000 [=======================>......] - ETA: 19s - loss: 0.2144 - categorical_accuracy: 0.9350
48640/60000 [=======================>......] - ETA: 19s - loss: 0.2143 - categorical_accuracy: 0.9351
48672/60000 [=======================>......] - ETA: 19s - loss: 0.2143 - categorical_accuracy: 0.9351
48704/60000 [=======================>......] - ETA: 19s - loss: 0.2142 - categorical_accuracy: 0.9351
48736/60000 [=======================>......] - ETA: 19s - loss: 0.2142 - categorical_accuracy: 0.9351
48768/60000 [=======================>......] - ETA: 19s - loss: 0.2142 - categorical_accuracy: 0.9351
48800/60000 [=======================>......] - ETA: 19s - loss: 0.2140 - categorical_accuracy: 0.9352
48832/60000 [=======================>......] - ETA: 19s - loss: 0.2140 - categorical_accuracy: 0.9352
48864/60000 [=======================>......] - ETA: 19s - loss: 0.2139 - categorical_accuracy: 0.9352
48896/60000 [=======================>......] - ETA: 19s - loss: 0.2138 - categorical_accuracy: 0.9352
48928/60000 [=======================>......] - ETA: 19s - loss: 0.2137 - categorical_accuracy: 0.9353
48960/60000 [=======================>......] - ETA: 19s - loss: 0.2136 - categorical_accuracy: 0.9353
48992/60000 [=======================>......] - ETA: 19s - loss: 0.2136 - categorical_accuracy: 0.9353
49024/60000 [=======================>......] - ETA: 19s - loss: 0.2136 - categorical_accuracy: 0.9353
49056/60000 [=======================>......] - ETA: 19s - loss: 0.2135 - categorical_accuracy: 0.9354
49088/60000 [=======================>......] - ETA: 19s - loss: 0.2133 - categorical_accuracy: 0.9354
49120/60000 [=======================>......] - ETA: 19s - loss: 0.2132 - categorical_accuracy: 0.9354
49152/60000 [=======================>......] - ETA: 18s - loss: 0.2133 - categorical_accuracy: 0.9354
49184/60000 [=======================>......] - ETA: 18s - loss: 0.2132 - categorical_accuracy: 0.9355
49216/60000 [=======================>......] - ETA: 18s - loss: 0.2131 - categorical_accuracy: 0.9355
49248/60000 [=======================>......] - ETA: 18s - loss: 0.2130 - categorical_accuracy: 0.9355
49280/60000 [=======================>......] - ETA: 18s - loss: 0.2129 - categorical_accuracy: 0.9356
49312/60000 [=======================>......] - ETA: 18s - loss: 0.2128 - categorical_accuracy: 0.9356
49344/60000 [=======================>......] - ETA: 18s - loss: 0.2127 - categorical_accuracy: 0.9356
49376/60000 [=======================>......] - ETA: 18s - loss: 0.2127 - categorical_accuracy: 0.9356
49408/60000 [=======================>......] - ETA: 18s - loss: 0.2127 - categorical_accuracy: 0.9356
49440/60000 [=======================>......] - ETA: 18s - loss: 0.2126 - categorical_accuracy: 0.9356
49472/60000 [=======================>......] - ETA: 18s - loss: 0.2126 - categorical_accuracy: 0.9356
49504/60000 [=======================>......] - ETA: 18s - loss: 0.2125 - categorical_accuracy: 0.9356
49536/60000 [=======================>......] - ETA: 18s - loss: 0.2124 - categorical_accuracy: 0.9357
49568/60000 [=======================>......] - ETA: 18s - loss: 0.2124 - categorical_accuracy: 0.9357
49600/60000 [=======================>......] - ETA: 18s - loss: 0.2125 - categorical_accuracy: 0.9357
49632/60000 [=======================>......] - ETA: 18s - loss: 0.2124 - categorical_accuracy: 0.9357
49664/60000 [=======================>......] - ETA: 18s - loss: 0.2123 - categorical_accuracy: 0.9358
49696/60000 [=======================>......] - ETA: 18s - loss: 0.2123 - categorical_accuracy: 0.9358
49728/60000 [=======================>......] - ETA: 17s - loss: 0.2122 - categorical_accuracy: 0.9358
49760/60000 [=======================>......] - ETA: 17s - loss: 0.2121 - categorical_accuracy: 0.9358
49792/60000 [=======================>......] - ETA: 17s - loss: 0.2121 - categorical_accuracy: 0.9359
49824/60000 [=======================>......] - ETA: 17s - loss: 0.2120 - categorical_accuracy: 0.9359
49888/60000 [=======================>......] - ETA: 17s - loss: 0.2119 - categorical_accuracy: 0.9359
49920/60000 [=======================>......] - ETA: 17s - loss: 0.2122 - categorical_accuracy: 0.9359
49952/60000 [=======================>......] - ETA: 17s - loss: 0.2121 - categorical_accuracy: 0.9359
49984/60000 [=======================>......] - ETA: 17s - loss: 0.2120 - categorical_accuracy: 0.9359
50016/60000 [========================>.....] - ETA: 17s - loss: 0.2119 - categorical_accuracy: 0.9359
50048/60000 [========================>.....] - ETA: 17s - loss: 0.2118 - categorical_accuracy: 0.9360
50080/60000 [========================>.....] - ETA: 17s - loss: 0.2118 - categorical_accuracy: 0.9360
50112/60000 [========================>.....] - ETA: 17s - loss: 0.2117 - categorical_accuracy: 0.9360
50144/60000 [========================>.....] - ETA: 17s - loss: 0.2115 - categorical_accuracy: 0.9360
50176/60000 [========================>.....] - ETA: 17s - loss: 0.2114 - categorical_accuracy: 0.9361
50208/60000 [========================>.....] - ETA: 17s - loss: 0.2114 - categorical_accuracy: 0.9361
50240/60000 [========================>.....] - ETA: 17s - loss: 0.2115 - categorical_accuracy: 0.9360
50272/60000 [========================>.....] - ETA: 17s - loss: 0.2114 - categorical_accuracy: 0.9360
50304/60000 [========================>.....] - ETA: 16s - loss: 0.2114 - categorical_accuracy: 0.9360
50336/60000 [========================>.....] - ETA: 16s - loss: 0.2113 - categorical_accuracy: 0.9361
50368/60000 [========================>.....] - ETA: 16s - loss: 0.2112 - categorical_accuracy: 0.9361
50400/60000 [========================>.....] - ETA: 16s - loss: 0.2110 - categorical_accuracy: 0.9362
50432/60000 [========================>.....] - ETA: 16s - loss: 0.2110 - categorical_accuracy: 0.9362
50464/60000 [========================>.....] - ETA: 16s - loss: 0.2110 - categorical_accuracy: 0.9362
50496/60000 [========================>.....] - ETA: 16s - loss: 0.2109 - categorical_accuracy: 0.9362
50528/60000 [========================>.....] - ETA: 16s - loss: 0.2108 - categorical_accuracy: 0.9362
50560/60000 [========================>.....] - ETA: 16s - loss: 0.2107 - categorical_accuracy: 0.9363
50592/60000 [========================>.....] - ETA: 16s - loss: 0.2107 - categorical_accuracy: 0.9363
50624/60000 [========================>.....] - ETA: 16s - loss: 0.2105 - categorical_accuracy: 0.9363
50656/60000 [========================>.....] - ETA: 16s - loss: 0.2104 - categorical_accuracy: 0.9363
50688/60000 [========================>.....] - ETA: 16s - loss: 0.2103 - categorical_accuracy: 0.9364
50720/60000 [========================>.....] - ETA: 16s - loss: 0.2103 - categorical_accuracy: 0.9364
50752/60000 [========================>.....] - ETA: 16s - loss: 0.2102 - categorical_accuracy: 0.9364
50784/60000 [========================>.....] - ETA: 16s - loss: 0.2101 - categorical_accuracy: 0.9364
50816/60000 [========================>.....] - ETA: 16s - loss: 0.2101 - categorical_accuracy: 0.9364
50848/60000 [========================>.....] - ETA: 15s - loss: 0.2101 - categorical_accuracy: 0.9365
50880/60000 [========================>.....] - ETA: 15s - loss: 0.2103 - categorical_accuracy: 0.9364
50912/60000 [========================>.....] - ETA: 15s - loss: 0.2102 - categorical_accuracy: 0.9365
50944/60000 [========================>.....] - ETA: 15s - loss: 0.2102 - categorical_accuracy: 0.9365
50976/60000 [========================>.....] - ETA: 15s - loss: 0.2102 - categorical_accuracy: 0.9365
51008/60000 [========================>.....] - ETA: 15s - loss: 0.2101 - categorical_accuracy: 0.9365
51040/60000 [========================>.....] - ETA: 15s - loss: 0.2101 - categorical_accuracy: 0.9365
51072/60000 [========================>.....] - ETA: 15s - loss: 0.2100 - categorical_accuracy: 0.9365
51104/60000 [========================>.....] - ETA: 15s - loss: 0.2099 - categorical_accuracy: 0.9365
51136/60000 [========================>.....] - ETA: 15s - loss: 0.2098 - categorical_accuracy: 0.9366
51168/60000 [========================>.....] - ETA: 15s - loss: 0.2098 - categorical_accuracy: 0.9366
51200/60000 [========================>.....] - ETA: 15s - loss: 0.2097 - categorical_accuracy: 0.9366
51232/60000 [========================>.....] - ETA: 15s - loss: 0.2096 - categorical_accuracy: 0.9366
51264/60000 [========================>.....] - ETA: 15s - loss: 0.2095 - categorical_accuracy: 0.9367
51296/60000 [========================>.....] - ETA: 15s - loss: 0.2094 - categorical_accuracy: 0.9367
51328/60000 [========================>.....] - ETA: 15s - loss: 0.2093 - categorical_accuracy: 0.9368
51360/60000 [========================>.....] - ETA: 15s - loss: 0.2092 - categorical_accuracy: 0.9368
51392/60000 [========================>.....] - ETA: 15s - loss: 0.2092 - categorical_accuracy: 0.9368
51424/60000 [========================>.....] - ETA: 14s - loss: 0.2092 - categorical_accuracy: 0.9368
51456/60000 [========================>.....] - ETA: 14s - loss: 0.2091 - categorical_accuracy: 0.9368
51488/60000 [========================>.....] - ETA: 14s - loss: 0.2090 - categorical_accuracy: 0.9368
51520/60000 [========================>.....] - ETA: 14s - loss: 0.2090 - categorical_accuracy: 0.9368
51552/60000 [========================>.....] - ETA: 14s - loss: 0.2089 - categorical_accuracy: 0.9368
51584/60000 [========================>.....] - ETA: 14s - loss: 0.2088 - categorical_accuracy: 0.9369
51616/60000 [========================>.....] - ETA: 14s - loss: 0.2088 - categorical_accuracy: 0.9369
51648/60000 [========================>.....] - ETA: 14s - loss: 0.2087 - categorical_accuracy: 0.9369
51680/60000 [========================>.....] - ETA: 14s - loss: 0.2086 - categorical_accuracy: 0.9369
51712/60000 [========================>.....] - ETA: 14s - loss: 0.2086 - categorical_accuracy: 0.9369
51744/60000 [========================>.....] - ETA: 14s - loss: 0.2084 - categorical_accuracy: 0.9370
51776/60000 [========================>.....] - ETA: 14s - loss: 0.2084 - categorical_accuracy: 0.9370
51808/60000 [========================>.....] - ETA: 14s - loss: 0.2083 - categorical_accuracy: 0.9370
51840/60000 [========================>.....] - ETA: 14s - loss: 0.2082 - categorical_accuracy: 0.9370
51872/60000 [========================>.....] - ETA: 14s - loss: 0.2081 - categorical_accuracy: 0.9370
51904/60000 [========================>.....] - ETA: 14s - loss: 0.2080 - categorical_accuracy: 0.9371
51936/60000 [========================>.....] - ETA: 14s - loss: 0.2079 - categorical_accuracy: 0.9371
51968/60000 [========================>.....] - ETA: 14s - loss: 0.2078 - categorical_accuracy: 0.9372
52000/60000 [=========================>....] - ETA: 13s - loss: 0.2077 - categorical_accuracy: 0.9372
52032/60000 [=========================>....] - ETA: 13s - loss: 0.2076 - categorical_accuracy: 0.9372
52064/60000 [=========================>....] - ETA: 13s - loss: 0.2076 - categorical_accuracy: 0.9372
52096/60000 [=========================>....] - ETA: 13s - loss: 0.2075 - categorical_accuracy: 0.9372
52128/60000 [=========================>....] - ETA: 13s - loss: 0.2073 - categorical_accuracy: 0.9373
52160/60000 [=========================>....] - ETA: 13s - loss: 0.2074 - categorical_accuracy: 0.9373
52192/60000 [=========================>....] - ETA: 13s - loss: 0.2074 - categorical_accuracy: 0.9373
52224/60000 [=========================>....] - ETA: 13s - loss: 0.2073 - categorical_accuracy: 0.9373
52256/60000 [=========================>....] - ETA: 13s - loss: 0.2071 - categorical_accuracy: 0.9373
52288/60000 [=========================>....] - ETA: 13s - loss: 0.2071 - categorical_accuracy: 0.9373
52320/60000 [=========================>....] - ETA: 13s - loss: 0.2071 - categorical_accuracy: 0.9373
52352/60000 [=========================>....] - ETA: 13s - loss: 0.2070 - categorical_accuracy: 0.9374
52384/60000 [=========================>....] - ETA: 13s - loss: 0.2071 - categorical_accuracy: 0.9374
52416/60000 [=========================>....] - ETA: 13s - loss: 0.2070 - categorical_accuracy: 0.9374
52448/60000 [=========================>....] - ETA: 13s - loss: 0.2069 - categorical_accuracy: 0.9374
52480/60000 [=========================>....] - ETA: 13s - loss: 0.2069 - categorical_accuracy: 0.9374
52512/60000 [=========================>....] - ETA: 13s - loss: 0.2067 - categorical_accuracy: 0.9375
52544/60000 [=========================>....] - ETA: 13s - loss: 0.2067 - categorical_accuracy: 0.9375
52576/60000 [=========================>....] - ETA: 12s - loss: 0.2066 - categorical_accuracy: 0.9375
52608/60000 [=========================>....] - ETA: 12s - loss: 0.2065 - categorical_accuracy: 0.9376
52640/60000 [=========================>....] - ETA: 12s - loss: 0.2064 - categorical_accuracy: 0.9376
52672/60000 [=========================>....] - ETA: 12s - loss: 0.2064 - categorical_accuracy: 0.9376
52704/60000 [=========================>....] - ETA: 12s - loss: 0.2063 - categorical_accuracy: 0.9376
52736/60000 [=========================>....] - ETA: 12s - loss: 0.2062 - categorical_accuracy: 0.9377
52768/60000 [=========================>....] - ETA: 12s - loss: 0.2061 - categorical_accuracy: 0.9377
52800/60000 [=========================>....] - ETA: 12s - loss: 0.2061 - categorical_accuracy: 0.9377
52832/60000 [=========================>....] - ETA: 12s - loss: 0.2060 - categorical_accuracy: 0.9377
52864/60000 [=========================>....] - ETA: 12s - loss: 0.2061 - categorical_accuracy: 0.9377
52896/60000 [=========================>....] - ETA: 12s - loss: 0.2061 - categorical_accuracy: 0.9377
52928/60000 [=========================>....] - ETA: 12s - loss: 0.2060 - categorical_accuracy: 0.9377
52960/60000 [=========================>....] - ETA: 12s - loss: 0.2059 - categorical_accuracy: 0.9377
52992/60000 [=========================>....] - ETA: 12s - loss: 0.2058 - categorical_accuracy: 0.9377
53024/60000 [=========================>....] - ETA: 12s - loss: 0.2058 - categorical_accuracy: 0.9377
53056/60000 [=========================>....] - ETA: 12s - loss: 0.2057 - categorical_accuracy: 0.9378
53088/60000 [=========================>....] - ETA: 12s - loss: 0.2057 - categorical_accuracy: 0.9378
53120/60000 [=========================>....] - ETA: 12s - loss: 0.2055 - categorical_accuracy: 0.9378
53152/60000 [=========================>....] - ETA: 11s - loss: 0.2056 - categorical_accuracy: 0.9378
53184/60000 [=========================>....] - ETA: 11s - loss: 0.2055 - categorical_accuracy: 0.9379
53216/60000 [=========================>....] - ETA: 11s - loss: 0.2054 - categorical_accuracy: 0.9379
53248/60000 [=========================>....] - ETA: 11s - loss: 0.2053 - categorical_accuracy: 0.9379
53280/60000 [=========================>....] - ETA: 11s - loss: 0.2052 - categorical_accuracy: 0.9379
53312/60000 [=========================>....] - ETA: 11s - loss: 0.2053 - categorical_accuracy: 0.9379
53344/60000 [=========================>....] - ETA: 11s - loss: 0.2052 - categorical_accuracy: 0.9379
53376/60000 [=========================>....] - ETA: 11s - loss: 0.2051 - categorical_accuracy: 0.9380
53408/60000 [=========================>....] - ETA: 11s - loss: 0.2050 - categorical_accuracy: 0.9380
53440/60000 [=========================>....] - ETA: 11s - loss: 0.2049 - categorical_accuracy: 0.9380
53472/60000 [=========================>....] - ETA: 11s - loss: 0.2048 - categorical_accuracy: 0.9380
53504/60000 [=========================>....] - ETA: 11s - loss: 0.2048 - categorical_accuracy: 0.9380
53536/60000 [=========================>....] - ETA: 11s - loss: 0.2047 - categorical_accuracy: 0.9380
53568/60000 [=========================>....] - ETA: 11s - loss: 0.2046 - categorical_accuracy: 0.9381
53632/60000 [=========================>....] - ETA: 11s - loss: 0.2044 - categorical_accuracy: 0.9382
53664/60000 [=========================>....] - ETA: 11s - loss: 0.2044 - categorical_accuracy: 0.9382
53696/60000 [=========================>....] - ETA: 11s - loss: 0.2043 - categorical_accuracy: 0.9382
53728/60000 [=========================>....] - ETA: 10s - loss: 0.2042 - categorical_accuracy: 0.9382
53760/60000 [=========================>....] - ETA: 10s - loss: 0.2043 - categorical_accuracy: 0.9382
53792/60000 [=========================>....] - ETA: 10s - loss: 0.2042 - categorical_accuracy: 0.9383
53824/60000 [=========================>....] - ETA: 10s - loss: 0.2041 - categorical_accuracy: 0.9383
53856/60000 [=========================>....] - ETA: 10s - loss: 0.2040 - categorical_accuracy: 0.9383
53888/60000 [=========================>....] - ETA: 10s - loss: 0.2040 - categorical_accuracy: 0.9383
53920/60000 [=========================>....] - ETA: 10s - loss: 0.2039 - categorical_accuracy: 0.9383
53952/60000 [=========================>....] - ETA: 10s - loss: 0.2038 - categorical_accuracy: 0.9383
53984/60000 [=========================>....] - ETA: 10s - loss: 0.2037 - categorical_accuracy: 0.9384
54016/60000 [==========================>...] - ETA: 10s - loss: 0.2036 - categorical_accuracy: 0.9384
54048/60000 [==========================>...] - ETA: 10s - loss: 0.2036 - categorical_accuracy: 0.9384
54080/60000 [==========================>...] - ETA: 10s - loss: 0.2036 - categorical_accuracy: 0.9384
54112/60000 [==========================>...] - ETA: 10s - loss: 0.2035 - categorical_accuracy: 0.9384
54144/60000 [==========================>...] - ETA: 10s - loss: 0.2034 - categorical_accuracy: 0.9384
54176/60000 [==========================>...] - ETA: 10s - loss: 0.2034 - categorical_accuracy: 0.9384
54208/60000 [==========================>...] - ETA: 10s - loss: 0.2033 - categorical_accuracy: 0.9384
54240/60000 [==========================>...] - ETA: 10s - loss: 0.2032 - categorical_accuracy: 0.9385
54272/60000 [==========================>...] - ETA: 10s - loss: 0.2031 - categorical_accuracy: 0.9385
54304/60000 [==========================>...] - ETA: 9s - loss: 0.2030 - categorical_accuracy: 0.9385 
54336/60000 [==========================>...] - ETA: 9s - loss: 0.2029 - categorical_accuracy: 0.9385
54368/60000 [==========================>...] - ETA: 9s - loss: 0.2029 - categorical_accuracy: 0.9385
54400/60000 [==========================>...] - ETA: 9s - loss: 0.2029 - categorical_accuracy: 0.9385
54432/60000 [==========================>...] - ETA: 9s - loss: 0.2028 - categorical_accuracy: 0.9385
54464/60000 [==========================>...] - ETA: 9s - loss: 0.2029 - categorical_accuracy: 0.9386
54496/60000 [==========================>...] - ETA: 9s - loss: 0.2028 - categorical_accuracy: 0.9386
54528/60000 [==========================>...] - ETA: 9s - loss: 0.2027 - categorical_accuracy: 0.9386
54560/60000 [==========================>...] - ETA: 9s - loss: 0.2026 - categorical_accuracy: 0.9387
54592/60000 [==========================>...] - ETA: 9s - loss: 0.2025 - categorical_accuracy: 0.9387
54624/60000 [==========================>...] - ETA: 9s - loss: 0.2025 - categorical_accuracy: 0.9387
54656/60000 [==========================>...] - ETA: 9s - loss: 0.2024 - categorical_accuracy: 0.9387
54688/60000 [==========================>...] - ETA: 9s - loss: 0.2023 - categorical_accuracy: 0.9387
54720/60000 [==========================>...] - ETA: 9s - loss: 0.2022 - categorical_accuracy: 0.9388
54752/60000 [==========================>...] - ETA: 9s - loss: 0.2022 - categorical_accuracy: 0.9388
54784/60000 [==========================>...] - ETA: 9s - loss: 0.2023 - categorical_accuracy: 0.9388
54816/60000 [==========================>...] - ETA: 9s - loss: 0.2022 - categorical_accuracy: 0.9388
54848/60000 [==========================>...] - ETA: 9s - loss: 0.2023 - categorical_accuracy: 0.9388
54880/60000 [==========================>...] - ETA: 8s - loss: 0.2023 - categorical_accuracy: 0.9388
54912/60000 [==========================>...] - ETA: 8s - loss: 0.2022 - categorical_accuracy: 0.9388
54944/60000 [==========================>...] - ETA: 8s - loss: 0.2022 - categorical_accuracy: 0.9388
54976/60000 [==========================>...] - ETA: 8s - loss: 0.2021 - categorical_accuracy: 0.9388
55008/60000 [==========================>...] - ETA: 8s - loss: 0.2020 - categorical_accuracy: 0.9389
55040/60000 [==========================>...] - ETA: 8s - loss: 0.2020 - categorical_accuracy: 0.9389
55072/60000 [==========================>...] - ETA: 8s - loss: 0.2020 - categorical_accuracy: 0.9389
55136/60000 [==========================>...] - ETA: 8s - loss: 0.2018 - categorical_accuracy: 0.9390
55168/60000 [==========================>...] - ETA: 8s - loss: 0.2017 - categorical_accuracy: 0.9390
55200/60000 [==========================>...] - ETA: 8s - loss: 0.2016 - categorical_accuracy: 0.9390
55232/60000 [==========================>...] - ETA: 8s - loss: 0.2016 - categorical_accuracy: 0.9390
55264/60000 [==========================>...] - ETA: 8s - loss: 0.2014 - categorical_accuracy: 0.9391
55296/60000 [==========================>...] - ETA: 8s - loss: 0.2014 - categorical_accuracy: 0.9391
55328/60000 [==========================>...] - ETA: 8s - loss: 0.2013 - categorical_accuracy: 0.9391
55360/60000 [==========================>...] - ETA: 8s - loss: 0.2013 - categorical_accuracy: 0.9391
55392/60000 [==========================>...] - ETA: 8s - loss: 0.2012 - categorical_accuracy: 0.9391
55424/60000 [==========================>...] - ETA: 8s - loss: 0.2012 - categorical_accuracy: 0.9391
55456/60000 [==========================>...] - ETA: 7s - loss: 0.2011 - categorical_accuracy: 0.9392
55488/60000 [==========================>...] - ETA: 7s - loss: 0.2010 - categorical_accuracy: 0.9392
55520/60000 [==========================>...] - ETA: 7s - loss: 0.2009 - categorical_accuracy: 0.9392
55552/60000 [==========================>...] - ETA: 7s - loss: 0.2010 - categorical_accuracy: 0.9392
55584/60000 [==========================>...] - ETA: 7s - loss: 0.2011 - categorical_accuracy: 0.9392
55616/60000 [==========================>...] - ETA: 7s - loss: 0.2010 - categorical_accuracy: 0.9392
55648/60000 [==========================>...] - ETA: 7s - loss: 0.2010 - categorical_accuracy: 0.9392
55680/60000 [==========================>...] - ETA: 7s - loss: 0.2010 - categorical_accuracy: 0.9392
55712/60000 [==========================>...] - ETA: 7s - loss: 0.2009 - categorical_accuracy: 0.9392
55744/60000 [==========================>...] - ETA: 7s - loss: 0.2009 - categorical_accuracy: 0.9392
55776/60000 [==========================>...] - ETA: 7s - loss: 0.2008 - categorical_accuracy: 0.9392
55808/60000 [==========================>...] - ETA: 7s - loss: 0.2008 - categorical_accuracy: 0.9393
55840/60000 [==========================>...] - ETA: 7s - loss: 0.2007 - categorical_accuracy: 0.9393
55872/60000 [==========================>...] - ETA: 7s - loss: 0.2007 - categorical_accuracy: 0.9393
55904/60000 [==========================>...] - ETA: 7s - loss: 0.2006 - categorical_accuracy: 0.9393
55936/60000 [==========================>...] - ETA: 7s - loss: 0.2005 - categorical_accuracy: 0.9394
55968/60000 [==========================>...] - ETA: 7s - loss: 0.2005 - categorical_accuracy: 0.9394
56000/60000 [===========================>..] - ETA: 6s - loss: 0.2005 - categorical_accuracy: 0.9394
56032/60000 [===========================>..] - ETA: 6s - loss: 0.2003 - categorical_accuracy: 0.9394
56064/60000 [===========================>..] - ETA: 6s - loss: 0.2002 - categorical_accuracy: 0.9394
56096/60000 [===========================>..] - ETA: 6s - loss: 0.2002 - categorical_accuracy: 0.9395
56128/60000 [===========================>..] - ETA: 6s - loss: 0.2001 - categorical_accuracy: 0.9395
56160/60000 [===========================>..] - ETA: 6s - loss: 0.2000 - categorical_accuracy: 0.9395
56192/60000 [===========================>..] - ETA: 6s - loss: 0.2000 - categorical_accuracy: 0.9396
56224/60000 [===========================>..] - ETA: 6s - loss: 0.1999 - categorical_accuracy: 0.9396
56256/60000 [===========================>..] - ETA: 6s - loss: 0.1999 - categorical_accuracy: 0.9396
56288/60000 [===========================>..] - ETA: 6s - loss: 0.1999 - categorical_accuracy: 0.9396
56320/60000 [===========================>..] - ETA: 6s - loss: 0.1999 - categorical_accuracy: 0.9396
56352/60000 [===========================>..] - ETA: 6s - loss: 0.1999 - categorical_accuracy: 0.9396
56384/60000 [===========================>..] - ETA: 6s - loss: 0.2000 - categorical_accuracy: 0.9396
56416/60000 [===========================>..] - ETA: 6s - loss: 0.1999 - categorical_accuracy: 0.9396
56448/60000 [===========================>..] - ETA: 6s - loss: 0.1998 - categorical_accuracy: 0.9396
56480/60000 [===========================>..] - ETA: 6s - loss: 0.1998 - categorical_accuracy: 0.9396
56512/60000 [===========================>..] - ETA: 6s - loss: 0.1997 - categorical_accuracy: 0.9396
56544/60000 [===========================>..] - ETA: 6s - loss: 0.1996 - categorical_accuracy: 0.9397
56576/60000 [===========================>..] - ETA: 5s - loss: 0.1995 - categorical_accuracy: 0.9397
56608/60000 [===========================>..] - ETA: 5s - loss: 0.1994 - categorical_accuracy: 0.9397
56640/60000 [===========================>..] - ETA: 5s - loss: 0.1993 - categorical_accuracy: 0.9398
56672/60000 [===========================>..] - ETA: 5s - loss: 0.1994 - categorical_accuracy: 0.9398
56704/60000 [===========================>..] - ETA: 5s - loss: 0.1994 - categorical_accuracy: 0.9398
56736/60000 [===========================>..] - ETA: 5s - loss: 0.1995 - categorical_accuracy: 0.9398
56768/60000 [===========================>..] - ETA: 5s - loss: 0.1994 - categorical_accuracy: 0.9398
56800/60000 [===========================>..] - ETA: 5s - loss: 0.1994 - categorical_accuracy: 0.9398
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1993 - categorical_accuracy: 0.9398
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1993 - categorical_accuracy: 0.9399
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1993 - categorical_accuracy: 0.9399
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1992 - categorical_accuracy: 0.9399
56960/60000 [===========================>..] - ETA: 5s - loss: 0.1991 - categorical_accuracy: 0.9399
56992/60000 [===========================>..] - ETA: 5s - loss: 0.1990 - categorical_accuracy: 0.9399
57024/60000 [===========================>..] - ETA: 5s - loss: 0.1989 - categorical_accuracy: 0.9400
57056/60000 [===========================>..] - ETA: 5s - loss: 0.1988 - categorical_accuracy: 0.9400
57088/60000 [===========================>..] - ETA: 5s - loss: 0.1988 - categorical_accuracy: 0.9400
57120/60000 [===========================>..] - ETA: 5s - loss: 0.1988 - categorical_accuracy: 0.9400
57152/60000 [===========================>..] - ETA: 4s - loss: 0.1987 - categorical_accuracy: 0.9400
57184/60000 [===========================>..] - ETA: 4s - loss: 0.1987 - categorical_accuracy: 0.9400
57216/60000 [===========================>..] - ETA: 4s - loss: 0.1986 - categorical_accuracy: 0.9401
57248/60000 [===========================>..] - ETA: 4s - loss: 0.1985 - categorical_accuracy: 0.9401
57280/60000 [===========================>..] - ETA: 4s - loss: 0.1984 - categorical_accuracy: 0.9401
57312/60000 [===========================>..] - ETA: 4s - loss: 0.1983 - categorical_accuracy: 0.9401
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1982 - categorical_accuracy: 0.9402
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1982 - categorical_accuracy: 0.9402
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1981 - categorical_accuracy: 0.9402
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1980 - categorical_accuracy: 0.9402
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1980 - categorical_accuracy: 0.9402
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1979 - categorical_accuracy: 0.9403
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1978 - categorical_accuracy: 0.9403
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1977 - categorical_accuracy: 0.9403
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1976 - categorical_accuracy: 0.9403
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1976 - categorical_accuracy: 0.9403
57664/60000 [===========================>..] - ETA: 4s - loss: 0.1976 - categorical_accuracy: 0.9404
57696/60000 [===========================>..] - ETA: 4s - loss: 0.1976 - categorical_accuracy: 0.9404
57728/60000 [===========================>..] - ETA: 3s - loss: 0.1975 - categorical_accuracy: 0.9404
57760/60000 [===========================>..] - ETA: 3s - loss: 0.1974 - categorical_accuracy: 0.9404
57792/60000 [===========================>..] - ETA: 3s - loss: 0.1974 - categorical_accuracy: 0.9404
57824/60000 [===========================>..] - ETA: 3s - loss: 0.1974 - categorical_accuracy: 0.9404
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1975 - categorical_accuracy: 0.9404
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1974 - categorical_accuracy: 0.9405
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1973 - categorical_accuracy: 0.9405
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1971 - categorical_accuracy: 0.9405
58016/60000 [============================>.] - ETA: 3s - loss: 0.1970 - categorical_accuracy: 0.9406
58048/60000 [============================>.] - ETA: 3s - loss: 0.1969 - categorical_accuracy: 0.9406
58080/60000 [============================>.] - ETA: 3s - loss: 0.1969 - categorical_accuracy: 0.9406
58112/60000 [============================>.] - ETA: 3s - loss: 0.1968 - categorical_accuracy: 0.9406
58144/60000 [============================>.] - ETA: 3s - loss: 0.1967 - categorical_accuracy: 0.9407
58176/60000 [============================>.] - ETA: 3s - loss: 0.1966 - categorical_accuracy: 0.9407
58208/60000 [============================>.] - ETA: 3s - loss: 0.1965 - categorical_accuracy: 0.9407
58240/60000 [============================>.] - ETA: 3s - loss: 0.1965 - categorical_accuracy: 0.9407
58272/60000 [============================>.] - ETA: 3s - loss: 0.1965 - categorical_accuracy: 0.9407
58304/60000 [============================>.] - ETA: 2s - loss: 0.1964 - categorical_accuracy: 0.9408
58336/60000 [============================>.] - ETA: 2s - loss: 0.1964 - categorical_accuracy: 0.9408
58368/60000 [============================>.] - ETA: 2s - loss: 0.1963 - categorical_accuracy: 0.9408
58400/60000 [============================>.] - ETA: 2s - loss: 0.1963 - categorical_accuracy: 0.9408
58432/60000 [============================>.] - ETA: 2s - loss: 0.1962 - categorical_accuracy: 0.9408
58464/60000 [============================>.] - ETA: 2s - loss: 0.1962 - categorical_accuracy: 0.9408
58496/60000 [============================>.] - ETA: 2s - loss: 0.1962 - categorical_accuracy: 0.9408
58528/60000 [============================>.] - ETA: 2s - loss: 0.1961 - categorical_accuracy: 0.9408
58560/60000 [============================>.] - ETA: 2s - loss: 0.1961 - categorical_accuracy: 0.9408
58592/60000 [============================>.] - ETA: 2s - loss: 0.1960 - categorical_accuracy: 0.9409
58624/60000 [============================>.] - ETA: 2s - loss: 0.1959 - categorical_accuracy: 0.9409
58656/60000 [============================>.] - ETA: 2s - loss: 0.1958 - categorical_accuracy: 0.9409
58688/60000 [============================>.] - ETA: 2s - loss: 0.1958 - categorical_accuracy: 0.9409
58720/60000 [============================>.] - ETA: 2s - loss: 0.1957 - categorical_accuracy: 0.9409
58752/60000 [============================>.] - ETA: 2s - loss: 0.1956 - categorical_accuracy: 0.9410
58784/60000 [============================>.] - ETA: 2s - loss: 0.1956 - categorical_accuracy: 0.9410
58816/60000 [============================>.] - ETA: 2s - loss: 0.1955 - categorical_accuracy: 0.9410
58848/60000 [============================>.] - ETA: 2s - loss: 0.1954 - categorical_accuracy: 0.9410
58880/60000 [============================>.] - ETA: 1s - loss: 0.1953 - categorical_accuracy: 0.9410
58912/60000 [============================>.] - ETA: 1s - loss: 0.1953 - categorical_accuracy: 0.9410
58944/60000 [============================>.] - ETA: 1s - loss: 0.1952 - categorical_accuracy: 0.9410
58976/60000 [============================>.] - ETA: 1s - loss: 0.1951 - categorical_accuracy: 0.9411
59008/60000 [============================>.] - ETA: 1s - loss: 0.1950 - categorical_accuracy: 0.9411
59040/60000 [============================>.] - ETA: 1s - loss: 0.1950 - categorical_accuracy: 0.9411
59072/60000 [============================>.] - ETA: 1s - loss: 0.1950 - categorical_accuracy: 0.9411
59104/60000 [============================>.] - ETA: 1s - loss: 0.1949 - categorical_accuracy: 0.9412
59136/60000 [============================>.] - ETA: 1s - loss: 0.1948 - categorical_accuracy: 0.9412
59168/60000 [============================>.] - ETA: 1s - loss: 0.1947 - categorical_accuracy: 0.9412
59200/60000 [============================>.] - ETA: 1s - loss: 0.1946 - categorical_accuracy: 0.9413
59232/60000 [============================>.] - ETA: 1s - loss: 0.1945 - categorical_accuracy: 0.9413
59264/60000 [============================>.] - ETA: 1s - loss: 0.1945 - categorical_accuracy: 0.9413
59296/60000 [============================>.] - ETA: 1s - loss: 0.1944 - categorical_accuracy: 0.9413
59328/60000 [============================>.] - ETA: 1s - loss: 0.1943 - categorical_accuracy: 0.9414
59360/60000 [============================>.] - ETA: 1s - loss: 0.1943 - categorical_accuracy: 0.9414
59392/60000 [============================>.] - ETA: 1s - loss: 0.1942 - categorical_accuracy: 0.9414
59424/60000 [============================>.] - ETA: 1s - loss: 0.1942 - categorical_accuracy: 0.9414
59456/60000 [============================>.] - ETA: 0s - loss: 0.1944 - categorical_accuracy: 0.9414
59488/60000 [============================>.] - ETA: 0s - loss: 0.1943 - categorical_accuracy: 0.9414
59520/60000 [============================>.] - ETA: 0s - loss: 0.1942 - categorical_accuracy: 0.9414
59552/60000 [============================>.] - ETA: 0s - loss: 0.1941 - categorical_accuracy: 0.9415
59584/60000 [============================>.] - ETA: 0s - loss: 0.1940 - categorical_accuracy: 0.9415
59616/60000 [============================>.] - ETA: 0s - loss: 0.1940 - categorical_accuracy: 0.9415
59648/60000 [============================>.] - ETA: 0s - loss: 0.1939 - categorical_accuracy: 0.9415
59680/60000 [============================>.] - ETA: 0s - loss: 0.1938 - categorical_accuracy: 0.9416
59712/60000 [============================>.] - ETA: 0s - loss: 0.1938 - categorical_accuracy: 0.9416
59744/60000 [============================>.] - ETA: 0s - loss: 0.1937 - categorical_accuracy: 0.9416
59776/60000 [============================>.] - ETA: 0s - loss: 0.1936 - categorical_accuracy: 0.9416
59808/60000 [============================>.] - ETA: 0s - loss: 0.1937 - categorical_accuracy: 0.9416
59840/60000 [============================>.] - ETA: 0s - loss: 0.1937 - categorical_accuracy: 0.9416
59872/60000 [============================>.] - ETA: 0s - loss: 0.1936 - categorical_accuracy: 0.9416
59904/60000 [============================>.] - ETA: 0s - loss: 0.1935 - categorical_accuracy: 0.9417
59936/60000 [============================>.] - ETA: 0s - loss: 0.1935 - categorical_accuracy: 0.9417
59968/60000 [============================>.] - ETA: 0s - loss: 0.1935 - categorical_accuracy: 0.9417
60000/60000 [==============================] - 109s 2ms/step - loss: 0.1934 - categorical_accuracy: 0.9417 - val_loss: 0.0518 - val_categorical_accuracy: 0.9831

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 16s
  160/10000 [..............................] - ETA: 6s 
  320/10000 [..............................] - ETA: 4s
  480/10000 [>.............................] - ETA: 4s
  640/10000 [>.............................] - ETA: 4s
  800/10000 [=>............................] - ETA: 3s
  960/10000 [=>............................] - ETA: 3s
 1120/10000 [==>...........................] - ETA: 3s
 1248/10000 [==>...........................] - ETA: 3s
 1376/10000 [===>..........................] - ETA: 3s
 1504/10000 [===>..........................] - ETA: 3s
 1632/10000 [===>..........................] - ETA: 3s
 1792/10000 [====>.........................] - ETA: 3s
 1952/10000 [====>.........................] - ETA: 3s
 2080/10000 [=====>........................] - ETA: 3s
 2240/10000 [=====>........................] - ETA: 3s
 2400/10000 [======>.......................] - ETA: 3s
 2528/10000 [======>.......................] - ETA: 3s
 2688/10000 [=======>......................] - ETA: 2s
 2848/10000 [=======>......................] - ETA: 2s
 2976/10000 [=======>......................] - ETA: 2s
 3104/10000 [========>.....................] - ETA: 2s
 3264/10000 [========>.....................] - ETA: 2s
 3392/10000 [=========>....................] - ETA: 2s
 3520/10000 [=========>....................] - ETA: 2s
 3680/10000 [==========>...................] - ETA: 2s
 3840/10000 [==========>...................] - ETA: 2s
 4000/10000 [===========>..................] - ETA: 2s
 4160/10000 [===========>..................] - ETA: 2s
 4320/10000 [===========>..................] - ETA: 2s
 4480/10000 [============>.................] - ETA: 2s
 4640/10000 [============>.................] - ETA: 2s
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
 8960/10000 [=========================>....] - ETA: 0s
 9120/10000 [==========================>...] - ETA: 0s
 9280/10000 [==========================>...] - ETA: 0s
 9440/10000 [===========================>..] - ETA: 0s
 9600/10000 [===========================>..] - ETA: 0s
 9760/10000 [============================>.] - ETA: 0s
 9888/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 4s 364us/step
[[2.55169645e-08 5.07109341e-08 1.15731018e-06 ... 9.99996781e-01
  1.10574156e-07 3.96537558e-07]
 [5.51020594e-06 1.52073131e-04 9.99829412e-01 ... 2.79052230e-08
  2.63742550e-07 2.60034577e-10]
 [1.07533685e-06 9.99843955e-01 3.84137966e-05 ... 7.51751359e-05
  1.36831004e-05 8.74040836e-07]
 ...
 [3.00886249e-09 1.85479325e-06 5.16118526e-09 ... 4.99739087e-07
  1.54219504e-06 7.08613834e-06]
 [2.20803258e-06 2.31967897e-06 3.69252540e-08 ... 3.13813651e-08
  4.07164171e-03 1.15493331e-05]
 [8.94416462e-06 2.76439749e-07 8.61281387e-06 ... 5.85502313e-10
  1.12743157e-06 3.14066710e-08]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.051819462867290715, 'accuracy_test:': 0.9830999970436096}

  ('#### Save   #######################################################',) 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/charcnn/result'}

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
[master e6d31e9] ml_store
 1 file changed, 2028 insertions(+)
To github.com:arita37/mlmodels_store.git
   9c54071..e6d31e9  master -> master





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
{'loss': 0.43521930649876595, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-15 20:31:31.196443: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
[master f35081a] ml_store
 1 file changed, 233 insertions(+)
To github.com:arita37/mlmodels_store.git
   e6d31e9..f35081a  master -> master





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
[master 6560cee] ml_store
 1 file changed, 35 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (fetch first)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the remote contains work that you do
hint: not have locally. This is usually caused by another repository pushing
hint: to the same ref. You may want to first integrate the remote changes
hint: (e.g., 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





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
 40%|████      | 2/5 [00:19<00:29,  9.74s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.9311090751070144, 'learning_rate': 0.0800945743170797, 'min_data_in_leaf': 6, 'num_leaves': 43} and reward: 0.392
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\xcb\xa5BR\xfdjX\r\x00\x00\x00learning_rateq\x02G?\xb4\x81\x13\xf9Ge\xeaX\x10\x00\x00\x00min_data_in_leafq\x03K\x06X\n\x00\x00\x00num_leavesq\x04K+u.' and reward: 0.392
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\xcb\xa5BR\xfdjX\r\x00\x00\x00learning_rateq\x02G?\xb4\x81\x13\xf9Ge\xeaX\x10\x00\x00\x00min_data_in_leafq\x03K\x06X\n\x00\x00\x00num_leavesq\x04K+u.' and reward: 0.392
 60%|██████    | 3/5 [00:41<00:26, 13.39s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.7950648549944922, 'learning_rate': 0.1829497078804086, 'min_data_in_leaf': 3, 'num_leaves': 54} and reward: 0.388
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9q+\xd9\xcc\xcfyX\r\x00\x00\x00learning_rateq\x02G?\xc7j\xe5b\x14]\xaaX\x10\x00\x00\x00min_data_in_leafq\x03K\x03X\n\x00\x00\x00num_leavesq\x04K6u.' and reward: 0.388
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9q+\xd9\xcc\xcfyX\r\x00\x00\x00learning_rateq\x02G?\xc7j\xe5b\x14]\xaaX\x10\x00\x00\x00min_data_in_leafq\x03K\x03X\n\x00\x00\x00num_leavesq\x04K6u.' and reward: 0.388
 80%|████████  | 4/5 [01:06<00:16, 16.83s/it] 80%|████████  | 4/5 [01:06<00:16, 16.56s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.8997956271080572, 'learning_rate': 0.02590948734377072, 'min_data_in_leaf': 10, 'num_leaves': 28} and reward: 0.3902
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xec\xcb 2\xf0i\xd0X\r\x00\x00\x00learning_rateq\x02G?\x9a\x88\x04C0\xc3\x99X\x10\x00\x00\x00min_data_in_leafq\x03K\nX\n\x00\x00\x00num_leavesq\x04K\x1cu.' and reward: 0.3902
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xec\xcb 2\xf0i\xd0X\r\x00\x00\x00learning_rateq\x02G?\x9a\x88\x04C0\xc3\x99X\x10\x00\x00\x00min_data_in_leafq\x03K\nX\n\x00\x00\x00num_leavesq\x04K\x1cu.' and reward: 0.3902
Time for Gradient Boosting hyperparameter optimization: 84.50966596603394
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.9311090751070144, 'learning_rate': 0.0800945743170797, 'min_data_in_leaf': 6, 'num_leaves': 43}
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
 40%|████      | 2/5 [00:46<01:10, 23.40s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.23048700032645036, 'embedding_size_factor': 0.5735788651366686, 'layers.choice': 3, 'learning_rate': 0.00011444021186953055, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 8.591950943622668e-10} and reward: 0.3122
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xcd\x80\x99\x18G\x12JX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe2Z\xc2\x10n\n\x89X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?\x1d\xff\xf3\xde\x8a\xf4aX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\r\x85\x8fZx\xb3 u.' and reward: 0.3122
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xcd\x80\x99\x18G\x12JX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe2Z\xc2\x10n\n\x89X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?\x1d\xff\xf3\xde\x8a\xf4aX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\r\x85\x8fZx\xb3 u.' and reward: 0.3122
 60%|██████    | 3/5 [01:39<01:04, 32.12s/it] 60%|██████    | 3/5 [01:39<01:06, 33.09s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.31982576163894777, 'embedding_size_factor': 0.6520123880905919, 'layers.choice': 1, 'learning_rate': 0.0010767617394252633, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 1.9977619432457603e-07} and reward: 0.3638
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd4x\x06x\xaa\x15\xa9X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe4\xddI\x15m\xf3RX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?Q\xa4D\x1d5\xe9\xb1X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\x8a\xd0A\xfd(\xa8\xf7u.' and reward: 0.3638
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd4x\x06x\xaa\x15\xa9X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe4\xddI\x15m\xf3RX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?Q\xa4D\x1d5\xe9\xb1X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\x8a\xd0A\xfd(\xa8\xf7u.' and reward: 0.3638
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 191.2140760421753
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
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.79s of the -159.61s of remaining time.
Ensemble size: 90
Ensemble weights: 
[0.07777778 0.26666667 0.2        0.27777778 0.02222222 0.05555556
 0.1       ]
	0.4006	 = Validation accuracy score
	1.64s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 281.3s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f64b513c470>

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   f35081a..7625d68  master     -> origin/master
Merge made by the 'recursive' strategy.
 error_list/20200515/list_log_import_20200515.md   |   2 +-
 error_list/20200515/list_log_test_cli_20200515.md | 378 +++++++++++-----------
 error_list/20200515/list_log_testall_20200515.md  | 364 +++++++++------------
 3 files changed, 340 insertions(+), 404 deletions(-)
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
[master d87391e] ml_store
 1 file changed, 221 insertions(+)
To github.com:arita37/mlmodels_store.git
   7625d68..d87391e  master -> master





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
[master c42a2b6] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   d87391e..c42a2b6  master -> master





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
[master f77ca86] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   c42a2b6..f77ca86  master -> master





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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7f292189f780> 

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
[master c1cdc20] ml_store
 1 file changed, 108 insertions(+)
To github.com:arita37/mlmodels_store.git
   f77ca86..c1cdc20  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 8.78740711e-01 -1.92316341e-02  3.19656942e-01  1.50016279e-01
  -1.46662161e+00  4.63534322e-01 -8.98683193e-01  3.97880425e-01
  -9.96010889e-01  3.18154200e-01]
 [ 1.36586461e+00  3.95860270e+00  5.48129585e-01  6.48643644e-01
   8.49176066e-01  1.07343294e-01  1.38631426e+00 -1.39881282e+00
   8.17678188e-02 -1.63744959e+00]
 [ 5.58538729e-01 -5.16347909e-01 -5.18145552e-01  3.51116897e-01
   8.25506954e-01 -6.87704631e-02 -9.52062101e-01 -1.34776494e+00
   1.47073986e+00 -1.46140360e+00]
 [ 8.71225789e-01 -2.09752935e-01 -4.56987858e-01  9.35147780e-01
  -8.73535822e-01  1.81252782e+00  9.25501215e-01  1.40109881e-01
  -1.41914878e+00  1.06898597e+00]
 [ 4.67397905e-01 -2.37875265e-01 -1.54491194e-01 -7.55662765e-01
  -5.47062239e-01  1.85143789e+00 -1.46405357e+00  2.09096677e-01
   1.55501599e+00 -9.24323185e-02]
 [ 8.61462558e-01  7.43205537e-02 -1.34501002e+00 -1.99560718e-01
  -1.47533915e+00 -6.54603169e-01 -3.14563862e-01  3.18014296e-01
  -8.90271552e-01 -1.29525789e+00]
 [ 1.05936450e-01 -7.37289628e-01  6.50323214e-01  1.64665066e-01
  -1.53556118e+00  7.78174179e-01  5.03170861e-02  3.09816759e-01
   1.05132077e+00  6.06548400e-01]
 [ 1.02817479e+00 -5.08457134e-01  1.76533510e+00  7.77419205e-01
   6.17714185e-01 -1.18771172e-01  4.50155513e-01 -1.98998184e-01
   1.86647138e+00  8.70969803e-01]
 [ 6.92114488e-01 -6.06524918e-02  2.05635552e+00 -2.41350300e+00
   1.17456965e+00 -1.77756638e+00 -2.81736269e-01 -7.77858827e-01
   1.11584111e+00  1.76024923e+00]
 [ 9.80427414e-01  1.93752881e+00 -2.30839743e-01  3.66332015e-01
   1.10018476e+00 -1.04458938e+00 -3.44987210e-01  2.05117344e+00
   5.85662000e-01 -2.79308500e+00]
 [ 7.22978007e-01  1.85535621e-01  9.15499268e-01  3.94428030e-01
  -8.49830738e-01  7.25522558e-01 -1.50504326e-01  1.49588477e+00
   6.75453809e-01 -4.38200267e-01]
 [ 4.46895161e-01  3.86539145e-01  1.35010682e+00 -8.51455657e-01
   8.50637963e-01  1.00088142e+00 -1.16017010e+00 -3.84832249e-01
   1.45810824e+00 -3.31283170e-01]
 [ 2.07582971e+00 -1.40232915e+00 -4.79184915e-01  4.51122939e-01
   1.03436581e+00 -6.94920901e-01 -4.18937898e-01  5.15413802e-01
  -1.11487105e+00 -1.95210529e+00]
 [ 1.03967316e+00 -7.31530982e-01  3.61847316e-01 -1.56573815e+00
   9.59288190e-01  1.01382247e+00 -1.78791289e+00 -2.22711263e+00
  -1.69933360e+00 -4.24492791e-01]
 [ 8.59823751e-01  1.71957132e-01 -3.48984191e-01  4.90561044e-01
  -1.15649503e+00 -1.39528303e+00  6.14726276e-01 -5.22356465e-01
  -3.69255902e-01 -9.77773002e-01]
 [ 1.37661405e+00 -6.00225330e-01  7.25916853e-01 -3.79517516e-01
  -6.27546260e-01 -1.01480369e+00  9.66220863e-01  4.35986196e-01
  -6.87487393e-01  3.32107876e+00]
 [ 1.16755486e+00  3.53600971e-02  7.14789597e-01 -1.53879325e+00
   1.10863359e+00 -4.47895185e-01 -1.75592564e+00  6.17985534e-01
  -1.84176326e-01  8.52704062e-01]
 [ 1.32857949e+00 -5.63236604e-01 -1.06179676e+00  2.39014596e+00
  -1.68450770e+00  2.45422849e-01 -5.69148654e-01  1.15259914e+00
  -2.24235772e-01  1.32247779e-01]
 [ 1.18559003e+00  8.64644065e-02  1.23289919e+00 -2.14246673e+00
   1.03334100e+00 -8.30168864e-01  3.67231814e-01  4.51615951e-01
   1.10417433e+00 -4.22856961e-01]
 [ 1.06702918e+00 -4.29142278e-01  3.50167159e-01  1.20845633e+00
   7.51480619e-01  1.11570180e+00 -4.79157099e-01  8.40861558e-01
  -1.02887218e-01  1.71647264e-02]
 [ 1.98519313e+00  6.74711526e-01 -1.39662042e+00  6.18539131e-01
   1.22382712e+00 -4.43171931e-01 -1.89148284e-03  1.81053491e+00
  -1.30572692e+00 -8.61316361e-01]
 [ 5.69983848e-01 -5.33020326e-01 -1.75458969e-01 -1.42655542e+00
   6.06604307e-01  1.76795995e+00 -1.15985185e-01 -4.75372875e-01
   4.77610182e-01 -9.33914656e-01]
 [ 9.64572049e-01 -1.06793987e-01  1.12232832e+00  1.45142926e+00
   1.21828168e+00 -6.18036848e-01  4.38166347e-01 -2.03720123e+00
  -1.94258918e+00 -9.97019796e-01]
 [ 7.61706684e-01 -1.48515645e+00  1.30253554e+00 -5.92461285e-01
  -1.64162479e+00 -2.30490794e+00 -1.34869645e+00 -3.18171727e-02
   1.12487742e-01 -3.62612088e-01]
 [ 1.01195228e+00 -1.88141087e+00  1.70018815e+00  4.97269099e-01
  -9.17664624e-01  2.37332699e-01 -1.09033833e+00 -2.14444405e+00
  -3.69562425e-01  6.08783659e-01]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7f7bc03cdf28>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7f7bda736710> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 6.21530991e-01 -1.50957268e+00 -1.01932039e-01 -1.08071069e+00
  -1.13742855e+00  7.25474004e-01  7.98063795e-01 -3.91782562e-02
  -2.28754171e-01  7.43356544e-01]
 [ 8.15836116e-01 -1.39169388e+00  2.50598029e+00  4.50217742e-01
  -8.82869820e-01  6.27437083e-01 -1.19586151e+00  7.51337235e-01
   1.40395436e-01  1.91979229e+00]
 [ 1.25704434e+00 -1.82391985e+00 -6.12406973e-01  1.16707517e+00
  -6.23732812e-01 -3.96687001e-02  8.16043684e-01  8.85825799e-01
   1.89861649e-01  3.93109245e-01]
 [ 1.03967316e+00 -7.31530982e-01  3.61847316e-01 -1.56573815e+00
   9.59288190e-01  1.01382247e+00 -1.78791289e+00 -2.22711263e+00
  -1.69933360e+00 -4.24492791e-01]
 [ 1.58463774e+00  5.71209961e-02 -1.77183179e-02 -7.99547491e-01
   1.32970299e+00 -2.91594596e-01 -1.10771250e+00 -2.58982853e-01
   1.89293198e-01 -1.71939447e+00]
 [ 3.45715997e-01 -4.13029310e-01 -4.68673816e-01  1.83471763e+00
   7.71514409e-01  5.64382855e-01  2.18628366e-02  2.13782807e+00
  -7.85533997e-01  8.53281222e-01]
 [ 1.18468624e+00 -1.00016919e+00 -5.93843067e-01  1.04499441e+00
   9.65482331e-01  6.08514698e-01 -6.25342001e-01 -6.93286967e-02
  -1.08392067e-01 -3.43900709e-01]
 [ 1.18559003e+00  8.64644065e-02  1.23289919e+00 -2.14246673e+00
   1.03334100e+00 -8.30168864e-01  3.67231814e-01  4.51615951e-01
   1.10417433e+00 -4.22856961e-01]
 [ 9.71395338e-01  7.13049050e-01  1.76041518e+00  1.30620607e+00
   1.05765490e+00 -6.04602969e-01  1.28376990e-01  6.36583409e-01
   1.40925339e+00  9.66539250e-01]
 [ 1.77547698e+00 -2.03394449e-01 -1.98837863e-01  2.42669441e-01
   9.64350564e-01  2.01830179e-01 -5.45774168e-01  6.61020288e-01
   1.79215821e+00 -7.00398505e-01]
 [ 8.76994650e-01  1.23225307e+00 -8.67787223e-01 -2.54179868e-01
   8.91891405e-01  1.39984394e+00 -8.77281519e-01 -7.81911683e-01
  -4.37508983e-01 -1.44087602e+00]
 [ 6.91743730e-01  1.00978733e+00 -1.21333813e+00 -1.55694156e+00
  -1.20257258e+00 -6.12442128e-01 -2.69836174e+00 -1.39351805e-01
  -7.28537489e-01  7.22518992e-02]
 [ 9.29250600e-01 -1.10657307e+00 -1.95816909e+00 -3.59224096e-01
  -1.21258781e+00  5.05381903e-01  5.42645295e-01  1.21794090e+00
  -1.94068096e+00  6.77807571e-01]
 [ 1.34740825e+00  7.33023232e-01  8.38634747e-01 -1.89881206e+00
  -5.42459922e-01 -1.11711069e+00 -1.09715436e+00 -5.08972278e-01
  -1.66485955e-01 -1.03918232e+00]
 [ 8.78740711e-01 -1.92316341e-02  3.19656942e-01  1.50016279e-01
  -1.46662161e+00  4.63534322e-01 -8.98683193e-01  3.97880425e-01
  -9.96010889e-01  3.18154200e-01]
 [ 1.64661853e+00 -1.52568032e+00 -6.06998398e-01  7.95026094e-01
   1.08480038e+00 -3.74438319e-01  4.29526140e-01  1.34048197e-01
   1.20205486e+00  1.06222724e-01]
 [ 8.78643802e-01  1.03703898e+00 -4.77124206e-01  6.72619748e-01
  -1.04948638e+00  2.42887697e+00  5.24750492e-01  1.00568668e+00
   3.53567216e-01 -3.59901817e-02]
 [ 8.71225789e-01 -2.09752935e-01 -4.56987858e-01  9.35147780e-01
  -8.73535822e-01  1.81252782e+00  9.25501215e-01  1.40109881e-01
  -1.41914878e+00  1.06898597e+00]
 [ 1.46893146e+00 -1.47115693e+00  5.85910431e-01 -8.30171895e-01
   1.03345052e+00 -8.80577600e-01 -9.55425262e-01 -2.79097722e-01
   1.62284909e+00  2.06578332e+00]
 [ 1.01177337e+00  9.57467711e-02  7.31402517e-01  1.03345080e+00
  -1.42203164e+00 -1.46273275e-01 -1.74549518e-02 -8.57496825e-01
  -9.34181843e-01  9.54495667e-01]
 [ 8.48069266e-01  4.51946037e-01  6.30195671e-01 -1.57915629e+00
   8.27987371e-01 -8.28627979e-01 -1.05344713e-01  5.28879746e-01
  -2.23708651e+00 -4.14846901e-01]
 [ 1.21619061e+00 -1.90005215e-02  8.60891241e-01 -2.26760192e-01
  -1.36419132e+00 -1.56450785e+00  1.63169151e+00  9.31255679e-01
   9.49808815e-01 -8.80189065e-01]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 7.00175710e-01  5.56073510e-01  8.96864073e-02  1.69380911e+00
   8.82393314e-01  1.96869779e-01 -5.63788735e-01  1.69869255e-01
  -1.16400797e+00 -6.01156801e-01]
 [ 1.01195228e+00 -1.88141087e+00  1.70018815e+00  4.97269099e-01
  -9.17664624e-01  2.37332699e-01 -1.09033833e+00 -2.14444405e+00
  -3.69562425e-01  6.08783659e-01]]
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
[[ 5.58538729e-01 -5.16347909e-01 -5.18145552e-01  3.51116897e-01
   8.25506954e-01 -6.87704631e-02 -9.52062101e-01 -1.34776494e+00
   1.47073986e+00 -1.46140360e+00]
 [ 8.95623122e-01 -2.29820588e+00 -1.95225583e-02  1.45652739e+00
  -1.85064099e+00  3.16637236e-01  1.11337266e-01 -2.66412594e+00
  -4.26428618e-01 -8.39988915e-01]
 [ 7.61706684e-01 -1.48515645e+00  1.30253554e+00 -5.92461285e-01
  -1.64162479e+00 -2.30490794e+00 -1.34869645e+00 -3.18171727e-02
   1.12487742e-01 -3.62612088e-01]
 [ 1.16777676e+00 -6.65754518e-01 -1.23312074e+00 -1.67419581e+00
   1.01313574e+00  8.25029824e-01 -1.20464572e-01 -4.98213564e-01
  -3.10984978e-01 -1.18231813e+00]
 [ 9.83799588e-01 -4.07240024e-01  9.32721414e-01  1.60564992e-01
  -1.27861800e+00 -1.20149976e-01  1.99759555e-01  3.85602292e-01
   7.18290736e-01 -5.30119800e-01]
 [ 1.37661405e+00 -6.00225330e-01  7.25916853e-01 -3.79517516e-01
  -6.27546260e-01 -1.01480369e+00  9.66220863e-01  4.35986196e-01
  -6.87487393e-01  3.32107876e+00]
 [ 3.45715997e-01 -4.13029310e-01 -4.68673816e-01  1.83471763e+00
   7.71514409e-01  5.64382855e-01  2.18628366e-02  2.13782807e+00
  -7.85533997e-01  8.53281222e-01]
 [ 6.92114488e-01 -6.06524918e-02  2.05635552e+00 -2.41350300e+00
   1.17456965e+00 -1.77756638e+00 -2.81736269e-01 -7.77858827e-01
   1.11584111e+00  1.76024923e+00]
 [ 6.81889336e-01 -1.15498263e+00  1.22895559e+00 -1.77632196e-01
   9.98545187e-01 -1.51045638e+00 -2.75846063e-01  1.01120706e+00
  -1.47656266e+00  1.30970591e+00]
 [ 1.01177337e+00  9.57467711e-02  7.31402517e-01  1.03345080e+00
  -1.42203164e+00 -1.46273275e-01 -1.74549518e-02 -8.57496825e-01
  -9.34181843e-01  9.54495667e-01]
 [ 6.13636707e-01  3.16658895e-01  1.34710546e+00 -1.89526695e+00
  -7.60458095e-01  8.97291174e-02 -3.29051549e-01  4.10265745e-01
   8.59870972e-01 -1.04906775e+00]
 [ 3.54133613e-01  2.11124755e-01  9.21450069e-01  1.65275673e-02
   9.03945451e-01  1.77187720e-01  9.54250872e-02 -1.11647002e+00
   8.09271010e-02  6.07501958e-02]
 [ 1.77547698e+00 -2.03394449e-01 -1.98837863e-01  2.42669441e-01
   9.64350564e-01  2.01830179e-01 -5.45774168e-01  6.61020288e-01
   1.79215821e+00 -7.00398505e-01]
 [ 1.32720112e+00 -1.61198320e-01  6.02450901e-01 -2.86384915e-01
  -5.78962302e-01 -8.70887650e-01  1.37975819e+00  5.01429590e-01
  -4.78614074e-01 -8.92646674e-01]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 1.09488485e+00 -6.96245395e-02 -1.16444148e-01  3.53870427e-01
  -1.44189096e+00 -1.86955017e-01  1.29118890e+00 -1.53236162e-01
  -2.43250851e+00 -2.27729800e+00]
 [ 4.73307772e-01 -9.73267585e-01 -2.28140691e-01  1.75167729e-01
  -1.01366961e+00 -5.34836927e-02  3.93787731e-01 -1.83061987e-01
  -2.21028902e-01  5.80330113e-01]
 [ 1.64661853e+00 -1.52568032e+00 -6.06998398e-01  7.95026094e-01
   1.08480038e+00 -3.74438319e-01  4.29526140e-01  1.34048197e-01
   1.20205486e+00  1.06222724e-01]
 [ 8.95510508e-01  9.20615118e-01  7.94528240e-01 -3.53679249e-02
   8.78099103e-01  2.11060505e+00 -1.02188594e+00 -1.30653407e+00
   7.63804802e-02 -1.87316098e+00]
 [ 1.14377130e+00  7.27813500e-01  3.52494364e-01  5.15073614e-01
   1.17718111e+00 -2.78253447e+00 -1.94332341e+00  5.84646610e-01
   3.24274243e-01 -2.36436952e-01]
 [ 7.90323893e-01  1.61336137e+00 -2.09424782e+00 -3.74804687e-01
   9.15884042e-01 -7.49969617e-01  3.10272288e-01  2.05462410e+00
   5.34095368e-02 -2.28765829e-01]
 [ 4.67397905e-01 -2.37875265e-01 -1.54491194e-01 -7.55662765e-01
  -5.47062239e-01  1.85143789e+00 -1.46405357e+00  2.09096677e-01
   1.55501599e+00 -9.24323185e-02]
 [ 1.18947778e+00 -6.80678141e-01 -5.68244809e-02 -8.45080274e-02
   8.21783210e-01 -2.97361883e-01 -1.86578994e-01  4.17302005e-01
   7.84770651e-01  4.92336556e-01]
 [ 9.71395338e-01  7.13049050e-01  1.76041518e+00  1.30620607e+00
   1.05765490e+00 -6.04602969e-01  1.28376990e-01  6.36583409e-01
   1.40925339e+00  9.66539250e-01]
 [ 9.67037267e-01  3.82715174e-01 -8.06184817e-01 -2.88997343e-01
   9.08526041e-01 -3.91816240e-01  1.62091229e+00  6.84001328e-01
  -3.53409983e-01 -2.51674208e-01]]
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
[master bcc60b3] ml_store
 1 file changed, 321 insertions(+)
To github.com:arita37/mlmodels_store.git
   c1cdc20..bcc60b3  master -> master





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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140031973109712
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140031973109488
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140031973108256
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140031973107808
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140031973107304
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140031973106968

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
grad_step = 000000, loss = 0.711732
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.586860
grad_step = 000002, loss = 0.495913
grad_step = 000003, loss = 0.403848
grad_step = 000004, loss = 0.300166
grad_step = 000005, loss = 0.198233
grad_step = 000006, loss = 0.122662
grad_step = 000007, loss = 0.099182
grad_step = 000008, loss = 0.095972
grad_step = 000009, loss = 0.073468
grad_step = 000010, loss = 0.046105
grad_step = 000011, loss = 0.032666
grad_step = 000012, loss = 0.031871
grad_step = 000013, loss = 0.036682
grad_step = 000014, loss = 0.041075
grad_step = 000015, loss = 0.041884
grad_step = 000016, loss = 0.038435
grad_step = 000017, loss = 0.031814
grad_step = 000018, loss = 0.024042
grad_step = 000019, loss = 0.017679
grad_step = 000020, loss = 0.014961
grad_step = 000021, loss = 0.016205
grad_step = 000022, loss = 0.019053
grad_step = 000023, loss = 0.020005
grad_step = 000024, loss = 0.017526
grad_step = 000025, loss = 0.013039
grad_step = 000026, loss = 0.009130
grad_step = 000027, loss = 0.007432
grad_step = 000028, loss = 0.007942
grad_step = 000029, loss = 0.009673
grad_step = 000030, loss = 0.011405
grad_step = 000031, loss = 0.012344
grad_step = 000032, loss = 0.012245
grad_step = 000033, loss = 0.011333
grad_step = 000034, loss = 0.010084
grad_step = 000035, loss = 0.008946
grad_step = 000036, loss = 0.008143
grad_step = 000037, loss = 0.007635
grad_step = 000038, loss = 0.007270
grad_step = 000039, loss = 0.006973
grad_step = 000040, loss = 0.006797
grad_step = 000041, loss = 0.006832
grad_step = 000042, loss = 0.007073
grad_step = 000043, loss = 0.007369
grad_step = 000044, loss = 0.007535
grad_step = 000045, loss = 0.007456
grad_step = 000046, loss = 0.007157
grad_step = 000047, loss = 0.006779
grad_step = 000048, loss = 0.006491
grad_step = 000049, loss = 0.006386
grad_step = 000050, loss = 0.006426
grad_step = 000051, loss = 0.006483
grad_step = 000052, loss = 0.006452
grad_step = 000053, loss = 0.006331
grad_step = 000054, loss = 0.006199
grad_step = 000055, loss = 0.006129
grad_step = 000056, loss = 0.006129
grad_step = 000057, loss = 0.006151
grad_step = 000058, loss = 0.006145
grad_step = 000059, loss = 0.006100
grad_step = 000060, loss = 0.006043
grad_step = 000061, loss = 0.006005
grad_step = 000062, loss = 0.005986
grad_step = 000063, loss = 0.005962
grad_step = 000064, loss = 0.005921
grad_step = 000065, loss = 0.005859
grad_step = 000066, loss = 0.005803
grad_step = 000067, loss = 0.005778
grad_step = 000068, loss = 0.005779
grad_step = 000069, loss = 0.005783
grad_step = 000070, loss = 0.005770
grad_step = 000071, loss = 0.005738
grad_step = 000072, loss = 0.005704
grad_step = 000073, loss = 0.005684
grad_step = 000074, loss = 0.005673
grad_step = 000075, loss = 0.005656
grad_step = 000076, loss = 0.005627
grad_step = 000077, loss = 0.005595
grad_step = 000078, loss = 0.005570
grad_step = 000079, loss = 0.005553
grad_step = 000080, loss = 0.005533
grad_step = 000081, loss = 0.005510
grad_step = 000082, loss = 0.005489
grad_step = 000083, loss = 0.005474
grad_step = 000084, loss = 0.005462
grad_step = 000085, loss = 0.005444
grad_step = 000086, loss = 0.005418
grad_step = 000087, loss = 0.005391
grad_step = 000088, loss = 0.005368
grad_step = 000089, loss = 0.005348
grad_step = 000090, loss = 0.005328
grad_step = 000091, loss = 0.005306
grad_step = 000092, loss = 0.005285
grad_step = 000093, loss = 0.005264
grad_step = 000094, loss = 0.005239
grad_step = 000095, loss = 0.005213
grad_step = 000096, loss = 0.005187
grad_step = 000097, loss = 0.005163
grad_step = 000098, loss = 0.005136
grad_step = 000099, loss = 0.005109
grad_step = 000100, loss = 0.005081
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.005054
grad_step = 000102, loss = 0.005024
grad_step = 000103, loss = 0.004992
grad_step = 000104, loss = 0.004960
grad_step = 000105, loss = 0.004927
grad_step = 000106, loss = 0.004893
grad_step = 000107, loss = 0.004857
grad_step = 000108, loss = 0.004821
grad_step = 000109, loss = 0.004783
grad_step = 000110, loss = 0.004744
grad_step = 000111, loss = 0.004703
grad_step = 000112, loss = 0.004663
grad_step = 000113, loss = 0.004620
grad_step = 000114, loss = 0.004575
grad_step = 000115, loss = 0.004530
grad_step = 000116, loss = 0.004483
grad_step = 000117, loss = 0.004434
grad_step = 000118, loss = 0.004384
grad_step = 000119, loss = 0.004333
grad_step = 000120, loss = 0.004280
grad_step = 000121, loss = 0.004227
grad_step = 000122, loss = 0.004172
grad_step = 000123, loss = 0.004115
grad_step = 000124, loss = 0.004057
grad_step = 000125, loss = 0.003998
grad_step = 000126, loss = 0.003937
grad_step = 000127, loss = 0.003876
grad_step = 000128, loss = 0.003813
grad_step = 000129, loss = 0.003749
grad_step = 000130, loss = 0.003684
grad_step = 000131, loss = 0.003617
grad_step = 000132, loss = 0.003549
grad_step = 000133, loss = 0.003479
grad_step = 000134, loss = 0.003409
grad_step = 000135, loss = 0.003337
grad_step = 000136, loss = 0.003265
grad_step = 000137, loss = 0.003191
grad_step = 000138, loss = 0.003116
grad_step = 000139, loss = 0.003039
grad_step = 000140, loss = 0.002961
grad_step = 000141, loss = 0.002882
grad_step = 000142, loss = 0.002798
grad_step = 000143, loss = 0.002714
grad_step = 000144, loss = 0.002629
grad_step = 000145, loss = 0.002543
grad_step = 000146, loss = 0.002457
grad_step = 000147, loss = 0.002370
grad_step = 000148, loss = 0.002283
grad_step = 000149, loss = 0.002197
grad_step = 000150, loss = 0.002112
grad_step = 000151, loss = 0.002029
grad_step = 000152, loss = 0.001948
grad_step = 000153, loss = 0.001867
grad_step = 000154, loss = 0.001788
grad_step = 000155, loss = 0.001713
grad_step = 000156, loss = 0.001643
grad_step = 000157, loss = 0.001579
grad_step = 000158, loss = 0.001520
grad_step = 000159, loss = 0.001466
grad_step = 000160, loss = 0.001415
grad_step = 000161, loss = 0.001369
grad_step = 000162, loss = 0.001330
grad_step = 000163, loss = 0.001295
grad_step = 000164, loss = 0.001264
grad_step = 000165, loss = 0.001239
grad_step = 000166, loss = 0.001226
grad_step = 000167, loss = 0.001196
grad_step = 000168, loss = 0.001149
grad_step = 000169, loss = 0.001130
grad_step = 000170, loss = 0.001106
grad_step = 000171, loss = 0.001076
grad_step = 000172, loss = 0.001056
grad_step = 000173, loss = 0.001027
grad_step = 000174, loss = 0.001013
grad_step = 000175, loss = 0.000994
grad_step = 000176, loss = 0.000972
grad_step = 000177, loss = 0.000961
grad_step = 000178, loss = 0.000945
grad_step = 000179, loss = 0.000932
grad_step = 000180, loss = 0.000921
grad_step = 000181, loss = 0.000908
grad_step = 000182, loss = 0.000898
grad_step = 000183, loss = 0.000885
grad_step = 000184, loss = 0.000874
grad_step = 000185, loss = 0.000865
grad_step = 000186, loss = 0.000851
grad_step = 000187, loss = 0.000841
grad_step = 000188, loss = 0.000830
grad_step = 000189, loss = 0.000817
grad_step = 000190, loss = 0.000806
grad_step = 000191, loss = 0.000793
grad_step = 000192, loss = 0.000782
grad_step = 000193, loss = 0.000771
grad_step = 000194, loss = 0.000760
grad_step = 000195, loss = 0.000749
grad_step = 000196, loss = 0.000737
grad_step = 000197, loss = 0.000727
grad_step = 000198, loss = 0.000717
grad_step = 000199, loss = 0.000709
grad_step = 000200, loss = 0.000705
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.000697
grad_step = 000202, loss = 0.000686
grad_step = 000203, loss = 0.000682
grad_step = 000204, loss = 0.000676
grad_step = 000205, loss = 0.000670
grad_step = 000206, loss = 0.000667
grad_step = 000207, loss = 0.000662
grad_step = 000208, loss = 0.000656
grad_step = 000209, loss = 0.000654
grad_step = 000210, loss = 0.000650
grad_step = 000211, loss = 0.000645
grad_step = 000212, loss = 0.000642
grad_step = 000213, loss = 0.000637
grad_step = 000214, loss = 0.000633
grad_step = 000215, loss = 0.000629
grad_step = 000216, loss = 0.000624
grad_step = 000217, loss = 0.000620
grad_step = 000218, loss = 0.000616
grad_step = 000219, loss = 0.000611
grad_step = 000220, loss = 0.000607
grad_step = 000221, loss = 0.000603
grad_step = 000222, loss = 0.000598
grad_step = 000223, loss = 0.000595
grad_step = 000224, loss = 0.000591
grad_step = 000225, loss = 0.000587
grad_step = 000226, loss = 0.000584
grad_step = 000227, loss = 0.000580
grad_step = 000228, loss = 0.000576
grad_step = 000229, loss = 0.000574
grad_step = 000230, loss = 0.000572
grad_step = 000231, loss = 0.000572
grad_step = 000232, loss = 0.000573
grad_step = 000233, loss = 0.000573
grad_step = 000234, loss = 0.000572
grad_step = 000235, loss = 0.000563
grad_step = 000236, loss = 0.000551
grad_step = 000237, loss = 0.000546
grad_step = 000238, loss = 0.000548
grad_step = 000239, loss = 0.000550
grad_step = 000240, loss = 0.000546
grad_step = 000241, loss = 0.000539
grad_step = 000242, loss = 0.000532
grad_step = 000243, loss = 0.000530
grad_step = 000244, loss = 0.000531
grad_step = 000245, loss = 0.000531
grad_step = 000246, loss = 0.000525
grad_step = 000247, loss = 0.000519
grad_step = 000248, loss = 0.000516
grad_step = 000249, loss = 0.000516
grad_step = 000250, loss = 0.000515
grad_step = 000251, loss = 0.000511
grad_step = 000252, loss = 0.000507
grad_step = 000253, loss = 0.000503
grad_step = 000254, loss = 0.000502
grad_step = 000255, loss = 0.000502
grad_step = 000256, loss = 0.000500
grad_step = 000257, loss = 0.000496
grad_step = 000258, loss = 0.000492
grad_step = 000259, loss = 0.000489
grad_step = 000260, loss = 0.000487
grad_step = 000261, loss = 0.000484
grad_step = 000262, loss = 0.000483
grad_step = 000263, loss = 0.000482
grad_step = 000264, loss = 0.000481
grad_step = 000265, loss = 0.000479
grad_step = 000266, loss = 0.000478
grad_step = 000267, loss = 0.000480
grad_step = 000268, loss = 0.000483
grad_step = 000269, loss = 0.000485
grad_step = 000270, loss = 0.000482
grad_step = 000271, loss = 0.000478
grad_step = 000272, loss = 0.000468
grad_step = 000273, loss = 0.000459
grad_step = 000274, loss = 0.000458
grad_step = 000275, loss = 0.000462
grad_step = 000276, loss = 0.000464
grad_step = 000277, loss = 0.000458
grad_step = 000278, loss = 0.000455
grad_step = 000279, loss = 0.000452
grad_step = 000280, loss = 0.000446
grad_step = 000281, loss = 0.000442
grad_step = 000282, loss = 0.000443
grad_step = 000283, loss = 0.000445
grad_step = 000284, loss = 0.000444
grad_step = 000285, loss = 0.000445
grad_step = 000286, loss = 0.000445
grad_step = 000287, loss = 0.000444
grad_step = 000288, loss = 0.000439
grad_step = 000289, loss = 0.000435
grad_step = 000290, loss = 0.000430
grad_step = 000291, loss = 0.000425
grad_step = 000292, loss = 0.000423
grad_step = 000293, loss = 0.000423
grad_step = 000294, loss = 0.000424
grad_step = 000295, loss = 0.000423
grad_step = 000296, loss = 0.000423
grad_step = 000297, loss = 0.000425
grad_step = 000298, loss = 0.000426
grad_step = 000299, loss = 0.000426
grad_step = 000300, loss = 0.000424
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000418
grad_step = 000302, loss = 0.000411
grad_step = 000303, loss = 0.000406
grad_step = 000304, loss = 0.000404
grad_step = 000305, loss = 0.000402
grad_step = 000306, loss = 0.000401
grad_step = 000307, loss = 0.000400
grad_step = 000308, loss = 0.000398
grad_step = 000309, loss = 0.000399
grad_step = 000310, loss = 0.000403
grad_step = 000311, loss = 0.000409
grad_step = 000312, loss = 0.000416
grad_step = 000313, loss = 0.000425
grad_step = 000314, loss = 0.000425
grad_step = 000315, loss = 0.000413
grad_step = 000316, loss = 0.000398
grad_step = 000317, loss = 0.000389
grad_step = 000318, loss = 0.000388
grad_step = 000319, loss = 0.000390
grad_step = 000320, loss = 0.000393
grad_step = 000321, loss = 0.000394
grad_step = 000322, loss = 0.000386
grad_step = 000323, loss = 0.000380
grad_step = 000324, loss = 0.000378
grad_step = 000325, loss = 0.000378
grad_step = 000326, loss = 0.000379
grad_step = 000327, loss = 0.000381
grad_step = 000328, loss = 0.000383
grad_step = 000329, loss = 0.000384
grad_step = 000330, loss = 0.000380
grad_step = 000331, loss = 0.000376
grad_step = 000332, loss = 0.000373
grad_step = 000333, loss = 0.000369
grad_step = 000334, loss = 0.000366
grad_step = 000335, loss = 0.000365
grad_step = 000336, loss = 0.000364
grad_step = 000337, loss = 0.000365
grad_step = 000338, loss = 0.000365
grad_step = 000339, loss = 0.000367
grad_step = 000340, loss = 0.000369
grad_step = 000341, loss = 0.000374
grad_step = 000342, loss = 0.000382
grad_step = 000343, loss = 0.000388
grad_step = 000344, loss = 0.000393
grad_step = 000345, loss = 0.000395
grad_step = 000346, loss = 0.000383
grad_step = 000347, loss = 0.000364
grad_step = 000348, loss = 0.000353
grad_step = 000349, loss = 0.000356
grad_step = 000350, loss = 0.000367
grad_step = 000351, loss = 0.000374
grad_step = 000352, loss = 0.000369
grad_step = 000353, loss = 0.000354
grad_step = 000354, loss = 0.000347
grad_step = 000355, loss = 0.000348
grad_step = 000356, loss = 0.000352
grad_step = 000357, loss = 0.000356
grad_step = 000358, loss = 0.000356
grad_step = 000359, loss = 0.000351
grad_step = 000360, loss = 0.000345
grad_step = 000361, loss = 0.000341
grad_step = 000362, loss = 0.000341
grad_step = 000363, loss = 0.000342
grad_step = 000364, loss = 0.000344
grad_step = 000365, loss = 0.000346
grad_step = 000366, loss = 0.000345
grad_step = 000367, loss = 0.000342
grad_step = 000368, loss = 0.000339
grad_step = 000369, loss = 0.000336
grad_step = 000370, loss = 0.000334
grad_step = 000371, loss = 0.000333
grad_step = 000372, loss = 0.000333
grad_step = 000373, loss = 0.000333
grad_step = 000374, loss = 0.000333
grad_step = 000375, loss = 0.000331
grad_step = 000376, loss = 0.000331
grad_step = 000377, loss = 0.000331
grad_step = 000378, loss = 0.000333
grad_step = 000379, loss = 0.000336
grad_step = 000380, loss = 0.000341
grad_step = 000381, loss = 0.000348
grad_step = 000382, loss = 0.000354
grad_step = 000383, loss = 0.000356
grad_step = 000384, loss = 0.000349
grad_step = 000385, loss = 0.000338
grad_step = 000386, loss = 0.000327
grad_step = 000387, loss = 0.000321
grad_step = 000388, loss = 0.000319
grad_step = 000389, loss = 0.000321
grad_step = 000390, loss = 0.000323
grad_step = 000391, loss = 0.000327
grad_step = 000392, loss = 0.000330
grad_step = 000393, loss = 0.000333
grad_step = 000394, loss = 0.000329
grad_step = 000395, loss = 0.000323
grad_step = 000396, loss = 0.000317
grad_step = 000397, loss = 0.000314
grad_step = 000398, loss = 0.000312
grad_step = 000399, loss = 0.000311
grad_step = 000400, loss = 0.000312
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000315
grad_step = 000402, loss = 0.000319
grad_step = 000403, loss = 0.000324
grad_step = 000404, loss = 0.000328
grad_step = 000405, loss = 0.000328
grad_step = 000406, loss = 0.000326
grad_step = 000407, loss = 0.000320
grad_step = 000408, loss = 0.000312
grad_step = 000409, loss = 0.000306
grad_step = 000410, loss = 0.000302
grad_step = 000411, loss = 0.000302
grad_step = 000412, loss = 0.000303
grad_step = 000413, loss = 0.000305
grad_step = 000414, loss = 0.000308
grad_step = 000415, loss = 0.000313
grad_step = 000416, loss = 0.000315
grad_step = 000417, loss = 0.000316
grad_step = 000418, loss = 0.000311
grad_step = 000419, loss = 0.000306
grad_step = 000420, loss = 0.000301
grad_step = 000421, loss = 0.000297
grad_step = 000422, loss = 0.000294
grad_step = 000423, loss = 0.000292
grad_step = 000424, loss = 0.000293
grad_step = 000425, loss = 0.000294
grad_step = 000426, loss = 0.000294
grad_step = 000427, loss = 0.000293
grad_step = 000428, loss = 0.000290
grad_step = 000429, loss = 0.000289
grad_step = 000430, loss = 0.000290
grad_step = 000431, loss = 0.000291
grad_step = 000432, loss = 0.000294
grad_step = 000433, loss = 0.000297
grad_step = 000434, loss = 0.000303
grad_step = 000435, loss = 0.000307
grad_step = 000436, loss = 0.000306
grad_step = 000437, loss = 0.000302
grad_step = 000438, loss = 0.000295
grad_step = 000439, loss = 0.000288
grad_step = 000440, loss = 0.000283
grad_step = 000441, loss = 0.000279
grad_step = 000442, loss = 0.000277
grad_step = 000443, loss = 0.000277
grad_step = 000444, loss = 0.000278
grad_step = 000445, loss = 0.000281
grad_step = 000446, loss = 0.000284
grad_step = 000447, loss = 0.000290
grad_step = 000448, loss = 0.000296
grad_step = 000449, loss = 0.000300
grad_step = 000450, loss = 0.000298
grad_step = 000451, loss = 0.000291
grad_step = 000452, loss = 0.000281
grad_step = 000453, loss = 0.000274
grad_step = 000454, loss = 0.000269
grad_step = 000455, loss = 0.000268
grad_step = 000456, loss = 0.000268
grad_step = 000457, loss = 0.000270
grad_step = 000458, loss = 0.000274
grad_step = 000459, loss = 0.000278
grad_step = 000460, loss = 0.000281
grad_step = 000461, loss = 0.000282
grad_step = 000462, loss = 0.000281
grad_step = 000463, loss = 0.000276
grad_step = 000464, loss = 0.000270
grad_step = 000465, loss = 0.000265
grad_step = 000466, loss = 0.000261
grad_step = 000467, loss = 0.000258
grad_step = 000468, loss = 0.000259
grad_step = 000469, loss = 0.000260
grad_step = 000470, loss = 0.000262
grad_step = 000471, loss = 0.000263
grad_step = 000472, loss = 0.000263
grad_step = 000473, loss = 0.000264
grad_step = 000474, loss = 0.000266
grad_step = 000475, loss = 0.000266
grad_step = 000476, loss = 0.000264
grad_step = 000477, loss = 0.000259
grad_step = 000478, loss = 0.000255
grad_step = 000479, loss = 0.000252
grad_step = 000480, loss = 0.000250
grad_step = 000481, loss = 0.000250
grad_step = 000482, loss = 0.000250
grad_step = 000483, loss = 0.000250
grad_step = 000484, loss = 0.000250
grad_step = 000485, loss = 0.000250
grad_step = 000486, loss = 0.000251
grad_step = 000487, loss = 0.000254
grad_step = 000488, loss = 0.000258
grad_step = 000489, loss = 0.000264
grad_step = 000490, loss = 0.000272
grad_step = 000491, loss = 0.000275
grad_step = 000492, loss = 0.000275
grad_step = 000493, loss = 0.000265
grad_step = 000494, loss = 0.000254
grad_step = 000495, loss = 0.000244
grad_step = 000496, loss = 0.000240
grad_step = 000497, loss = 0.000241
grad_step = 000498, loss = 0.000245
grad_step = 000499, loss = 0.000249
grad_step = 000500, loss = 0.000250
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000250
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
[[0.86868775 0.8590134  0.9314568  0.95502007 0.9960254 ]
 [0.8160746  0.92954254 0.9688376  1.0134124  0.9839613 ]
 [0.8644196  0.94087946 0.9969963  0.9923471  0.9368766 ]
 [0.9342594  0.9684053  0.9877432  0.9502361  0.91603434]
 [0.9980328  0.9672813  0.9329598  0.9172879  0.860641  ]
 [0.96075815 0.95219034 0.9160119  0.8549378  0.8588066 ]
 [0.92041063 0.9073429  0.8678137  0.85832953 0.80758333]
 [0.88159615 0.8461431  0.84817004 0.8195433  0.8320575 ]
 [0.81858456 0.82328194 0.8103647  0.8352635  0.85838616]
 [0.829311   0.7986069  0.86087936 0.8637502  0.8201443 ]
 [0.8303769  0.8352466  0.85091317 0.81972516 0.91914624]
 [0.8068949  0.8234322  0.81622845 0.9169383  0.9470016 ]
 [0.8569798  0.8567261  0.92507654 0.95422995 0.99588823]
 [0.8194857  0.9404276  0.9623577  1.018042   0.97724056]
 [0.89514536 0.944354   0.99043965 0.97950333 0.91711855]
 [0.9456033  0.97135794 0.96824443 0.9310769  0.900247  ]
 [1.0026844  0.9617269  0.92083794 0.9009186  0.84462863]
 [0.9628145  0.9361862  0.89872146 0.83732986 0.8451452 ]
 [0.9258471  0.8938798  0.8506768  0.85023856 0.8037501 ]
 [0.895141   0.8451858  0.8363432  0.81707823 0.82814395]
 [0.82921344 0.8323183  0.809029   0.83512264 0.8680919 ]
 [0.845017   0.80739206 0.8641984  0.868994   0.8283719 ]
 [0.84599745 0.8463491  0.8565522  0.8248069  0.92514515]
 [0.8209143  0.83229154 0.82287383 0.918867   0.94769853]
 [0.87265795 0.864541   0.92963386 0.9574276  1.0041853 ]
 [0.82460034 0.93700576 0.9710914  1.0216876  0.9978126 ]
 [0.87528336 0.95237756 1.0036286  1.0056541  0.9482519 ]
 [0.95231855 0.9771376  0.9897399  0.96576124 0.9268198 ]
 [1.0140815  0.9728104  0.9380429  0.931201   0.87048876]
 [0.97279453 0.9630884  0.9276149  0.8621366  0.86552817]
 [0.9290413  0.9164433  0.87577736 0.86381066 0.8149207 ]]

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
[master 410b3f5] ml_store
 1 file changed, 1122 insertions(+)
To github.com:arita37/mlmodels_store.git
   bcc60b3..410b3f5  master -> master





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
[master 5fe1e8c] ml_store
 1 file changed, 37 insertions(+)
To github.com:arita37/mlmodels_store.git
   410b3f5..5fe1e8c  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//matchzoo_models.py 

  #### Loading params   ############################################## 

  {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'dataset_pars': {'data_pack': '', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'resample': True, 'sort': False, 'callbacks': 'PADDING'}, 'dataloader': '', 'dataloader_pars': {'device': 'cpu', 'dataset': 'None', 'stage': 'train', 'callback': 'PADDING'}, 'preprocess': {'train': {'transform': True, 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'stage': 'train', 'resample': True, 'sort': False, 'dataloader_callback': 'PADDING'}, 'test': {'transform': True, 'batch_size': 20, 'stage': 'dev', 'dataloader_callback': 'PADDING'}}} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Loading dataset   ############################################# 

  #### Model init   ################################################## 
  0%|          | 0/231508 [00:00<?, ?B/s]100%|██████████| 231508/231508 [00:00<00:00, 17827909.71B/s]
  0%|          | 0/433 [00:00<?, ?B/s]100%|██████████| 433/433 [00:00<00:00, 518007.31B/s]
  0%|          | 0/440473133 [00:00<?, ?B/s]  1%|          | 4007936/440473133 [00:00<00:10, 40057336.75B/s]  2%|▏         | 9366528/440473133 [00:00<00:09, 43338010.78B/s]  3%|▎         | 14613504/440473133 [00:00<00:09, 45724900.92B/s]  4%|▍         | 19770368/440473133 [00:00<00:08, 47333672.93B/s]  6%|▌         | 24805376/440473133 [00:00<00:08, 48199040.78B/s]  7%|▋         | 29153280/440473133 [00:00<00:08, 46673756.57B/s]  8%|▊         | 34052096/440473133 [00:00<00:08, 47344635.11B/s]  9%|▉         | 39271424/440473133 [00:00<00:08, 48700022.12B/s] 10%|█         | 44340224/440473133 [00:00<00:08, 49275817.56B/s] 11%|█         | 49282048/440473133 [00:01<00:07, 49315339.13B/s] 12%|█▏        | 54528000/440473133 [00:01<00:07, 50216933.21B/s] 14%|█▎        | 59813888/440473133 [00:01<00:07, 50978706.47B/s] 15%|█▍        | 65105920/440473133 [00:01<00:07, 51545037.32B/s] 16%|█▌        | 70249472/440473133 [00:01<00:07, 51509786.50B/s] 17%|█▋        | 75500544/440473133 [00:01<00:07, 51804552.67B/s] 18%|█▊        | 80668672/440473133 [00:01<00:06, 51472137.80B/s] 20%|█▉        | 85942272/440473133 [00:01<00:06, 51840712.11B/s] 21%|██        | 91182080/440473133 [00:01<00:06, 52003144.10B/s] 22%|██▏       | 96594944/440473133 [00:01<00:06, 52622632.26B/s] 23%|██▎       | 101923840/440473133 [00:02<00:06, 52816483.08B/s] 24%|██▍       | 107406336/440473133 [00:02<00:06, 53403194.45B/s] 26%|██▌       | 112748544/440473133 [00:02<00:06, 52953273.05B/s] 27%|██▋       | 118045696/440473133 [00:02<00:06, 51539115.27B/s] 28%|██▊       | 123479040/440473133 [00:02<00:06, 52345404.42B/s] 29%|██▉       | 129024000/440473133 [00:02<00:05, 53238134.66B/s] 31%|███       | 134359040/440473133 [00:02<00:05, 52391739.57B/s] 32%|███▏      | 139761664/440473133 [00:02<00:05, 52870181.62B/s] 33%|███▎      | 145274880/440473133 [00:02<00:05, 53528872.19B/s] 34%|███▍      | 150636544/440473133 [00:02<00:05, 53142788.81B/s] 35%|███▌      | 156066816/440473133 [00:03<00:05, 53482984.45B/s] 37%|███▋      | 161420288/440473133 [00:03<00:05, 53119321.49B/s] 38%|███▊      | 166885376/440473133 [00:03<00:05, 53567235.51B/s] 39%|███▉      | 172246016/440473133 [00:03<00:05, 53345068.81B/s] 40%|████      | 177732608/440473133 [00:03<00:04, 53791355.81B/s] 42%|████▏     | 183114752/440473133 [00:03<00:04, 53665179.45B/s] 43%|████▎     | 188483584/440473133 [00:03<00:04, 53296795.91B/s] 44%|████▍     | 193988608/440473133 [00:03<00:04, 53809648.13B/s] 45%|████▌     | 199372800/440473133 [00:03<00:04, 52612923.42B/s] 46%|████▋     | 204642304/440473133 [00:03<00:04, 51444956.40B/s] 48%|████▊     | 209798144/440473133 [00:04<00:04, 50163792.94B/s] 49%|████▉     | 214830080/440473133 [00:04<00:04, 49771041.18B/s] 50%|█████     | 220263424/440473133 [00:04<00:04, 51055012.67B/s] 51%|█████     | 225491968/440473133 [00:04<00:04, 51416874.13B/s] 52%|█████▏    | 230645760/440473133 [00:04<00:04, 51255208.54B/s] 54%|█████▎    | 236208128/440473133 [00:04<00:03, 52484928.36B/s] 55%|█████▍    | 241597440/440473133 [00:04<00:03, 52898832.77B/s] 56%|█████▌    | 247363584/440473133 [00:04<00:03, 54240659.62B/s] 58%|█████▊    | 253388800/440473133 [00:04<00:03, 55913096.48B/s] 59%|█████▉    | 259511296/440473133 [00:04<00:03, 57399759.75B/s] 60%|██████    | 265505792/440473133 [00:05<00:03, 58133372.36B/s] 62%|██████▏   | 271340544/440473133 [00:05<00:02, 57715716.62B/s] 63%|██████▎   | 277341184/440473133 [00:05<00:02, 58382777.24B/s] 64%|██████▍   | 283192320/440473133 [00:05<00:02, 58138186.72B/s] 66%|██████▌   | 289115136/440473133 [00:05<00:02, 58459721.34B/s] 67%|██████▋   | 294968320/440473133 [00:05<00:02, 57804868.93B/s] 68%|██████▊   | 300755968/440473133 [00:05<00:02, 57185840.53B/s] 70%|██████▉   | 307008512/440473133 [00:05<00:02, 58689104.63B/s] 71%|███████   | 313051136/440473133 [00:05<00:02, 59196364.88B/s] 72%|███████▏  | 318982144/440473133 [00:05<00:02, 56821938.69B/s] 74%|███████▎  | 324692992/440473133 [00:06<00:02, 55108675.33B/s] 75%|███████▍  | 330234880/440473133 [00:06<00:02, 54650756.28B/s] 76%|███████▌  | 335722496/440473133 [00:06<00:01, 54687425.05B/s] 77%|███████▋  | 341207040/440473133 [00:06<00:01, 54589287.54B/s] 79%|███████▊  | 346677248/440473133 [00:06<00:01, 54400845.24B/s] 80%|███████▉  | 352125952/440473133 [00:06<00:01, 54393075.06B/s] 81%|████████  | 357776384/440473133 [00:06<00:01, 55009442.03B/s] 83%|████████▎ | 363410432/440473133 [00:06<00:01, 55401144.14B/s] 84%|████████▍ | 369157120/440473133 [00:06<00:01, 56003750.41B/s] 85%|████████▌ | 375003136/440473133 [00:07<00:01, 56718658.51B/s] 86%|████████▋ | 380766208/440473133 [00:07<00:01, 56989045.89B/s] 88%|████████▊ | 386469888/440473133 [00:07<00:00, 56058388.20B/s] 89%|████████▉ | 392197120/440473133 [00:07<00:00, 56415833.99B/s] 90%|█████████ | 398150656/440473133 [00:07<00:00, 57316359.90B/s] 92%|█████████▏| 404092928/440473133 [00:07<00:00, 57928921.71B/s] 93%|█████████▎| 409892864/440473133 [00:07<00:00, 56998108.38B/s] 94%|█████████▍| 415600640/440473133 [00:07<00:00, 55921797.16B/s] 96%|█████████▌| 421202944/440473133 [00:07<00:00, 55642508.60B/s] 97%|█████████▋| 426775552/440473133 [00:07<00:00, 55667092.30B/s] 98%|█████████▊| 432348160/440473133 [00:08<00:00, 55522526.07B/s] 99%|█████████▉| 437904384/440473133 [00:08<00:00, 55089418.71B/s]100%|██████████| 440473133/440473133 [00:08<00:00, 53877083.61B/s]Downloading data from https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip

   8192/7094233 [..............................] - ETA: 0s
 917504/7094233 [==>...........................] - ETA: 0s
1843200/7094233 [======>.......................] - ETA: 0s
2883584/7094233 [===========>..................] - ETA: 0s
3932160/7094233 [===============>..............] - ETA: 0s
4972544/7094233 [====================>.........] - ETA: 0s
6012928/7094233 [========================>.....] - ETA: 0s
7053312/7094233 [============================>.] - ETA: 0s
7094272/7094233 [==============================] - 0s 0us/step

Processing text_left with encode:   0%|          | 0/2118 [00:00<?, ?it/s]Processing text_left with encode:   3%|▎         | 61/2118 [00:00<00:03, 609.50it/s]Processing text_left with encode:  28%|██▊       | 590/2118 [00:00<00:01, 829.67it/s]Processing text_left with encode:  52%|█████▏    | 1099/2118 [00:00<00:00, 1107.82it/s]Processing text_left with encode:  78%|███████▊  | 1656/2118 [00:00<00:00, 1458.21it/s]Processing text_left with encode: 100%|██████████| 2118/2118 [00:00<00:00, 4315.34it/s]
Processing text_right with encode:   0%|          | 0/18841 [00:00<?, ?it/s]Processing text_right with encode:   1%|          | 178/18841 [00:00<00:10, 1779.90it/s]Processing text_right with encode:   2%|▏         | 369/18841 [00:00<00:10, 1814.45it/s]Processing text_right with encode:   3%|▎         | 580/18841 [00:00<00:09, 1893.43it/s]Processing text_right with encode:   4%|▍         | 777/18841 [00:00<00:09, 1915.36it/s]Processing text_right with encode:   5%|▌         | 958/18841 [00:00<00:09, 1879.27it/s]Processing text_right with encode:   6%|▌         | 1140/18841 [00:00<00:09, 1860.47it/s]Processing text_right with encode:   7%|▋         | 1340/18841 [00:00<00:09, 1898.62it/s]Processing text_right with encode:   8%|▊         | 1532/18841 [00:00<00:09, 1903.68it/s]Processing text_right with encode:   9%|▉         | 1749/18841 [00:00<00:08, 1973.84it/s]Processing text_right with encode:  10%|█         | 1941/18841 [00:01<00:08, 1950.42it/s]Processing text_right with encode:  11%|█▏        | 2132/18841 [00:01<00:08, 1893.38it/s]Processing text_right with encode:  12%|█▏        | 2329/18841 [00:01<00:08, 1913.16it/s]Processing text_right with encode:  14%|█▎        | 2545/18841 [00:01<00:08, 1980.96it/s]Processing text_right with encode:  15%|█▍        | 2757/18841 [00:01<00:07, 2020.28it/s]Processing text_right with encode:  16%|█▌        | 2959/18841 [00:01<00:08, 1978.20it/s]Processing text_right with encode:  17%|█▋        | 3157/18841 [00:01<00:07, 1972.19it/s]Processing text_right with encode:  18%|█▊        | 3355/18841 [00:01<00:08, 1929.57it/s]Processing text_right with encode:  19%|█▉        | 3549/18841 [00:01<00:08, 1868.74it/s]Processing text_right with encode:  20%|█▉        | 3745/18841 [00:01<00:07, 1894.17it/s]Processing text_right with encode:  21%|██        | 3935/18841 [00:02<00:07, 1879.96it/s]Processing text_right with encode:  22%|██▏       | 4136/18841 [00:02<00:07, 1914.55it/s]Processing text_right with encode:  23%|██▎       | 4328/18841 [00:02<00:07, 1900.85it/s]Processing text_right with encode:  24%|██▍       | 4520/18841 [00:02<00:07, 1904.61it/s]Processing text_right with encode:  25%|██▌       | 4718/18841 [00:02<00:07, 1926.48it/s]Processing text_right with encode:  26%|██▌       | 4928/18841 [00:02<00:07, 1973.79it/s]Processing text_right with encode:  27%|██▋       | 5126/18841 [00:02<00:06, 1964.09it/s]Processing text_right with encode:  28%|██▊       | 5327/18841 [00:02<00:06, 1974.91it/s]Processing text_right with encode:  29%|██▉       | 5525/18841 [00:02<00:06, 1959.46it/s]Processing text_right with encode:  30%|███       | 5722/18841 [00:02<00:06, 1930.34it/s]Processing text_right with encode:  31%|███▏      | 5916/18841 [00:03<00:06, 1912.40it/s]Processing text_right with encode:  32%|███▏      | 6117/18841 [00:03<00:06, 1940.29it/s]Processing text_right with encode:  34%|███▎      | 6313/18841 [00:03<00:06, 1945.31it/s]Processing text_right with encode:  35%|███▍      | 6531/18841 [00:03<00:06, 2008.62it/s]Processing text_right with encode:  36%|███▌      | 6749/18841 [00:03<00:05, 2054.63it/s]Processing text_right with encode:  37%|███▋      | 6956/18841 [00:03<00:06, 1950.85it/s]Processing text_right with encode:  38%|███▊      | 7153/18841 [00:03<00:06, 1933.98it/s]Processing text_right with encode:  39%|███▉      | 7381/18841 [00:03<00:05, 2025.35it/s]Processing text_right with encode:  40%|████      | 7586/18841 [00:03<00:05, 1959.08it/s]Processing text_right with encode:  41%|████▏     | 7784/18841 [00:04<00:05, 1918.57it/s]Processing text_right with encode:  42%|████▏     | 7978/18841 [00:04<00:05, 1862.75it/s]Processing text_right with encode:  43%|████▎     | 8166/18841 [00:04<00:05, 1840.35it/s]Processing text_right with encode:  44%|████▍     | 8359/18841 [00:04<00:05, 1864.66it/s]Processing text_right with encode:  45%|████▌     | 8547/18841 [00:04<00:05, 1809.55it/s]Processing text_right with encode:  46%|████▋     | 8729/18841 [00:04<00:05, 1793.21it/s]Processing text_right with encode:  47%|████▋     | 8909/18841 [00:04<00:05, 1789.11it/s]Processing text_right with encode:  48%|████▊     | 9089/18841 [00:04<00:05, 1739.08it/s]Processing text_right with encode:  49%|████▉     | 9280/18841 [00:04<00:05, 1785.34it/s]Processing text_right with encode:  50%|█████     | 9460/18841 [00:04<00:05, 1776.21it/s]Processing text_right with encode:  51%|█████     | 9639/18841 [00:05<00:05, 1763.25it/s]Processing text_right with encode:  52%|█████▏    | 9816/18841 [00:05<00:05, 1737.88it/s]Processing text_right with encode:  53%|█████▎    | 10012/18841 [00:05<00:04, 1794.10it/s]Processing text_right with encode:  54%|█████▍    | 10203/18841 [00:05<00:04, 1823.05it/s]Processing text_right with encode:  55%|█████▌    | 10386/18841 [00:05<00:04, 1814.17it/s]Processing text_right with encode:  56%|█████▌    | 10574/18841 [00:05<00:04, 1832.34it/s]Processing text_right with encode:  57%|█████▋    | 10758/18841 [00:05<00:04, 1817.57it/s]Processing text_right with encode:  58%|█████▊    | 10941/18841 [00:05<00:04, 1781.44it/s]Processing text_right with encode:  59%|█████▉    | 11120/18841 [00:05<00:04, 1760.79it/s]Processing text_right with encode:  60%|█████▉    | 11297/18841 [00:05<00:04, 1744.57it/s]Processing text_right with encode:  61%|██████    | 11472/18841 [00:06<00:04, 1701.64it/s]Processing text_right with encode:  62%|██████▏   | 11643/18841 [00:06<00:04, 1676.99it/s]Processing text_right with encode:  63%|██████▎   | 11819/18841 [00:06<00:04, 1700.32it/s]Processing text_right with encode:  64%|██████▎   | 12003/18841 [00:06<00:03, 1737.73it/s]Processing text_right with encode:  65%|██████▍   | 12184/18841 [00:06<00:03, 1758.54it/s]Processing text_right with encode:  66%|██████▌   | 12365/18841 [00:06<00:03, 1773.57it/s]Processing text_right with encode:  67%|██████▋   | 12543/18841 [00:06<00:03, 1766.51it/s]Processing text_right with encode:  68%|██████▊   | 12720/18841 [00:06<00:03, 1764.21it/s]Processing text_right with encode:  68%|██████▊   | 12897/18841 [00:06<00:03, 1754.54it/s]Processing text_right with encode:  69%|██████▉   | 13073/18841 [00:07<00:03, 1752.39it/s]Processing text_right with encode:  70%|███████   | 13249/18841 [00:07<00:03, 1711.06it/s]Processing text_right with encode:  71%|███████▏  | 13439/18841 [00:07<00:03, 1763.55it/s]Processing text_right with encode:  72%|███████▏  | 13616/18841 [00:07<00:02, 1761.51it/s]Processing text_right with encode:  73%|███████▎  | 13805/18841 [00:07<00:02, 1797.68it/s]Processing text_right with encode:  74%|███████▍  | 13997/18841 [00:07<00:02, 1830.86it/s]Processing text_right with encode:  75%|███████▌  | 14181/18841 [00:07<00:02, 1833.55it/s]Processing text_right with encode:  76%|███████▋  | 14373/18841 [00:07<00:02, 1854.79it/s]Processing text_right with encode:  77%|███████▋  | 14559/18841 [00:07<00:02, 1825.33it/s]Processing text_right with encode:  78%|███████▊  | 14759/18841 [00:07<00:02, 1872.01it/s]Processing text_right with encode:  79%|███████▉  | 14955/18841 [00:08<00:02, 1895.71it/s]Processing text_right with encode:  80%|████████  | 15146/18841 [00:08<00:02, 1827.52it/s]Processing text_right with encode:  81%|████████▏ | 15330/18841 [00:08<00:02, 1747.34it/s]Processing text_right with encode:  82%|████████▏ | 15507/18841 [00:08<00:01, 1708.80it/s]Processing text_right with encode:  83%|████████▎ | 15686/18841 [00:08<00:01, 1727.80it/s]Processing text_right with encode:  84%|████████▍ | 15860/18841 [00:08<00:01, 1699.48it/s]Processing text_right with encode:  85%|████████▌ | 16031/18841 [00:08<00:01, 1672.74it/s]Processing text_right with encode:  86%|████████▌ | 16199/18841 [00:08<00:01, 1662.16it/s]Processing text_right with encode:  87%|████████▋ | 16369/18841 [00:08<00:01, 1671.88it/s]Processing text_right with encode:  88%|████████▊ | 16545/18841 [00:08<00:01, 1696.29it/s]Processing text_right with encode:  89%|████████▉ | 16736/18841 [00:09<00:01, 1751.76it/s]Processing text_right with encode:  90%|████████▉ | 16912/18841 [00:09<00:01, 1662.06it/s]Processing text_right with encode:  91%|█████████ | 17097/18841 [00:09<00:01, 1712.95it/s]Processing text_right with encode:  92%|█████████▏| 17270/18841 [00:09<00:00, 1710.03it/s]Processing text_right with encode:  93%|█████████▎| 17443/18841 [00:09<00:00, 1712.52it/s]Processing text_right with encode:  93%|█████████▎| 17615/18841 [00:09<00:00, 1697.02it/s]Processing text_right with encode:  95%|█████████▍| 17805/18841 [00:09<00:00, 1752.79it/s]Processing text_right with encode:  95%|█████████▌| 17985/18841 [00:09<00:00, 1766.31it/s]Processing text_right with encode:  97%|█████████▋| 18185/18841 [00:09<00:00, 1825.60it/s]Processing text_right with encode:  97%|█████████▋| 18369/18841 [00:10<00:00, 1718.99it/s]Processing text_right with encode:  98%|█████████▊| 18553/18841 [00:10<00:00, 1752.24it/s]Processing text_right with encode:  99%|█████████▉| 18730/18841 [00:10<00:00, 1708.89it/s]Processing text_right with encode: 100%|██████████| 18841/18841 [00:10<00:00, 1829.28it/s]
Processing length_left with len:   0%|          | 0/2118 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 2118/2118 [00:00<00:00, 681984.94it/s]
Processing length_right with len:   0%|          | 0/18841 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 18841/18841 [00:00<00:00, 808397.34it/s]
Processing text_left with encode:   0%|          | 0/633 [00:00<?, ?it/s]Processing text_left with encode:  77%|███████▋  | 488/633 [00:00<00:00, 4874.31it/s]Processing text_left with encode: 100%|██████████| 633/633 [00:00<00:00, 4740.83it/s]
Processing text_right with encode:   0%|          | 0/5961 [00:00<?, ?it/s]Processing text_right with encode:   3%|▎         | 180/5961 [00:00<00:03, 1799.33it/s]Processing text_right with encode:   6%|▌         | 358/5961 [00:00<00:03, 1790.29it/s]Processing text_right with encode:   9%|▊         | 515/5961 [00:00<00:03, 1714.83it/s]Processing text_right with encode:  12%|█▏        | 693/5961 [00:00<00:03, 1731.76it/s]Processing text_right with encode:  15%|█▍        | 881/5961 [00:00<00:02, 1769.01it/s]Processing text_right with encode:  18%|█▊        | 1054/5961 [00:00<00:02, 1756.96it/s]Processing text_right with encode:  20%|██        | 1218/5961 [00:00<00:02, 1719.65it/s]Processing text_right with encode:  23%|██▎       | 1390/5961 [00:00<00:02, 1719.56it/s]Processing text_right with encode:  26%|██▌       | 1560/5961 [00:00<00:02, 1713.33it/s]Processing text_right with encode:  29%|██▉       | 1728/5961 [00:01<00:02, 1702.44it/s]Processing text_right with encode:  32%|███▏      | 1919/5961 [00:01<00:02, 1757.58it/s]Processing text_right with encode:  36%|███▌      | 2121/5961 [00:01<00:02, 1826.17it/s]Processing text_right with encode:  39%|███▊      | 2303/5961 [00:01<00:02, 1801.24it/s]Processing text_right with encode:  42%|████▏     | 2484/5961 [00:01<00:01, 1803.43it/s]Processing text_right with encode:  45%|████▍     | 2680/5961 [00:01<00:01, 1847.10it/s]Processing text_right with encode:  48%|████▊     | 2885/5961 [00:01<00:01, 1902.78it/s]Processing text_right with encode:  52%|█████▏    | 3076/5961 [00:01<00:01, 1901.84it/s]Processing text_right with encode:  55%|█████▍    | 3272/5961 [00:01<00:01, 1916.39it/s]Processing text_right with encode:  58%|█████▊    | 3464/5961 [00:01<00:01, 1856.97it/s]Processing text_right with encode:  61%|██████    | 3651/5961 [00:02<00:01, 1839.48it/s]Processing text_right with encode:  64%|██████▍   | 3839/5961 [00:02<00:01, 1847.99it/s]Processing text_right with encode:  68%|██████▊   | 4033/5961 [00:02<00:01, 1872.27it/s]Processing text_right with encode:  71%|███████   | 4223/5961 [00:02<00:00, 1878.82it/s]Processing text_right with encode:  74%|███████▍  | 4412/5961 [00:02<00:00, 1862.03it/s]Processing text_right with encode:  77%|███████▋  | 4609/5961 [00:02<00:00, 1890.86it/s]Processing text_right with encode:  81%|████████  | 4799/5961 [00:02<00:00, 1853.48it/s]Processing text_right with encode:  84%|████████▎ | 4985/5961 [00:02<00:00, 1824.21it/s]Processing text_right with encode:  87%|████████▋ | 5168/5961 [00:02<00:00, 1807.69it/s]Processing text_right with encode:  90%|████████▉ | 5350/5961 [00:02<00:00, 1781.53it/s]Processing text_right with encode:  93%|█████████▎| 5529/5961 [00:03<00:00, 1749.39it/s]Processing text_right with encode:  96%|█████████▌| 5705/5961 [00:03<00:00, 1716.99it/s]Processing text_right with encode:  99%|█████████▉| 5892/5961 [00:03<00:00, 1757.48it/s]Processing text_right with encode: 100%|██████████| 5961/5961 [00:03<00:00, 1809.46it/s]
Processing length_left with len:   0%|          | 0/633 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 633/633 [00:00<00:00, 389924.28it/s]
Processing length_right with len:   0%|          | 0/5961 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 5961/5961 [00:00<00:00, 696887.87it/s]
  #### Model  fit   ############################################# 

  0%|          | 0/102 [00:00<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:27<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:27<?, ?it/s, loss=0.989]Epoch 1/1:   1%|          | 1/102 [00:27<46:40, 27.73s/it, loss=0.989]Epoch 1/1:   1%|          | 1/102 [01:40<46:40, 27.73s/it, loss=0.989]Epoch 1/1:   1%|          | 1/102 [01:40<46:40, 27.73s/it, loss=0.966]Epoch 1/1:   2%|▏         | 2/102 [01:40<1:08:56, 41.37s/it, loss=0.966]Epoch 1/1:   2%|▏         | 2/102 [02:03<1:08:56, 41.37s/it, loss=0.966]Epoch 1/1:   2%|▏         | 2/102 [02:03<1:08:56, 41.37s/it, loss=0.972]Epoch 1/1:   3%|▎         | 3/102 [02:03<58:44, 35.61s/it, loss=0.972]  Epoch 1/1:   3%|▎         | 3/102 [03:22<58:44, 35.61s/it, loss=0.972]Epoch 1/1:   3%|▎         | 3/102 [03:22<58:44, 35.61s/it, loss=0.940]Epoch 1/1:   4%|▍         | 4/102 [03:22<1:19:42, 48.80s/it, loss=0.940]Epoch 1/1:   4%|▍         | 4/102 [04:13<1:19:42, 48.80s/it, loss=0.940]Epoch 1/1:   4%|▍         | 4/102 [04:13<1:19:42, 48.80s/it, loss=0.848]Epoch 1/1:   5%|▍         | 5/102 [04:13<1:19:49, 49.37s/it, loss=0.848]Epoch 1/1:   5%|▍         | 5/102 [05:47<1:19:49, 49.37s/it, loss=0.848]Epoch 1/1:   5%|▍         | 5/102 [05:47<1:19:49, 49.37s/it, loss=0.864]Epoch 1/1:   6%|▌         | 6/102 [05:47<1:40:26, 62.78s/it, loss=0.864]Epoch 1/1:   6%|▌         | 6/102 [06:51<1:40:26, 62.78s/it, loss=0.864]Epoch 1/1:   6%|▌         | 6/102 [06:51<1:40:26, 62.78s/it, loss=0.754]Epoch 1/1:   7%|▋         | 7/102 [06:51<1:40:00, 63.16s/it, loss=0.754]Epoch 1/1:   7%|▋         | 7/102 [10:04<1:40:00, 63.16s/it, loss=0.754]Epoch 1/1:   7%|▋         | 7/102 [10:04<1:40:00, 63.16s/it, loss=0.963]Epoch 1/1:   8%|▊         | 8/102 [10:04<2:40:05, 102.19s/it, loss=0.963]Epoch 1/1:   8%|▊         | 8/102 [11:45<2:40:05, 102.19s/it, loss=0.963]Epoch 1/1:   8%|▊         | 8/102 [11:45<2:40:05, 102.19s/it, loss=0.649]Epoch 1/1:   9%|▉         | 9/102 [11:45<2:37:47, 101.80s/it, loss=0.649]Epoch 1/1:   9%|▉         | 9/102 [12:41<2:37:47, 101.80s/it, loss=0.649]Epoch 1/1:   9%|▉         | 9/102 [12:41<2:37:47, 101.80s/it, loss=0.730]Epoch 1/1:  10%|▉         | 10/102 [12:41<2:14:56, 88.00s/it, loss=0.730]Epoch 1/1:  10%|▉         | 10/102 [14:16<2:14:56, 88.00s/it, loss=0.730]Epoch 1/1:  10%|▉         | 10/102 [14:16<2:14:56, 88.00s/it, loss=0.558]Epoch 1/1:  11%|█         | 11/102 [14:16<2:16:54, 90.27s/it, loss=0.558]Epoch 1/1:  11%|█         | 11/102 [16:32<2:16:54, 90.27s/it, loss=0.558]Epoch 1/1:  11%|█         | 11/102 [16:32<2:16:54, 90.27s/it, loss=0.842]Epoch 1/1:  12%|█▏        | 12/102 [16:32<2:35:31, 103.69s/it, loss=0.842]Epoch 1/1:  12%|█▏        | 12/102 [17:17<2:35:31, 103.69s/it, loss=0.842]Epoch 1/1:  12%|█▏        | 12/102 [17:17<2:35:31, 103.69s/it, loss=0.540]Epoch 1/1:  13%|█▎        | 13/102 [17:17<2:08:02, 86.32s/it, loss=0.540] Epoch 1/1:  13%|█▎        | 13/102 [19:48<2:08:02, 86.32s/it, loss=0.540]Epoch 1/1:  13%|█▎        | 13/102 [19:48<2:08:02, 86.32s/it, loss=0.558]Epoch 1/1:  14%|█▎        | 14/102 [19:48<2:34:53, 105.60s/it, loss=0.558]Killed

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   5fe1e8c..cc75309  master     -> origin/master
Updating 5fe1e8c..cc75309
Fast-forward
 error_list/20200515/list_log_jupyter_20200515.md | 1749 +++++++++++-----------
 error_list/20200515/list_log_testall_20200515.md |  364 +++--
 2 files changed, 1084 insertions(+), 1029 deletions(-)
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
[master 49a7824] ml_store
 1 file changed, 70 insertions(+)
To github.com:arita37/mlmodels_store.git
   cc75309..49a7824  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py 

  #### Loading params   ############################################## 

  {'dataset': 'torchvision.datasets:MNIST', 'transform_uri': 'mlmodels.preprocess.image.py:torch_transform_mnist', '2nd___transform_uri': '/mnt/hgfs/d/gitdev/mlmodels/mlmodels/preprocess/image.py:torch_transform_mnist', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 4, 'test_batch_size': 1} {'checkpointdir': 'ztest/model_tch/torchhub/restnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18/'} 

  #### Loading dataset   ############################################# 

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 30%|███       | 2990080/9912422 [00:00<00:00, 29697689.33it/s]9920512it [00:00, 32142440.79it/s]                             
0it [00:00, ?it/s]32768it [00:00, 1431193.61it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1001698.96it/s]1654784it [00:00, 12349436.14it/s]                           
0it [00:00, ?it/s]8192it [00:00, 266472.30it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
Extracting /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz to /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw/train-labels-idx1-ubyte.gz to /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw/t10k-images-idx3-ubyte.gz to /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw/t10k-labels-idx1-ubyte.gz to /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw
Processing...
Done!

  #### Model init, fit   ############################################# 

  #### If transformer URI is Provided 

  #### Loading dataloader URI 

Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
dataset :  <class 'torchvision.datasets.mnist.MNIST'>
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py", line 406, in <module>
    test(data_path="model_tch/torchhub_cnn_list.json", pars_choice="json", config_mode="resnet18")
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py", line 338, in test
    model, session = fit(model, data_pars, compute_pars, out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py", line 207, in fit
    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py", line 46, in _train
    for i,batch in enumerate(train_itr):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
    return self.collate_fn(data)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
    return [default_collate(samples) for samples in transposed]
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
    return [default_collate(samples) for samples in transposed]
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
    raise TypeError(default_collate_err_msg_format.format(elem_type))
TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>

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
[master 9ee4220] ml_store
 1 file changed, 84 insertions(+)
To github.com:arita37/mlmodels_store.git
   49a7824..9ee4220  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//03_nbeats_dataloader.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//03_nbeats_dataloader.py", line 9, in <module>
    from dataloader import DataLoader
ModuleNotFoundError: No module named 'dataloader'

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
[master 0afa1c5] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   9ee4220..0afa1c5  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_sentence.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Epoch:   0%|          | 0/1 [00:00<?, ?it/s]
Iteration:   0%|          | 0/29440 [00:00<?, ?it/s][A
Iteration:   0%|          | 1/29440 [00:20<170:04:03, 20.80s/it][A
Iteration:   0%|          | 2/29440 [00:39<163:50:40, 20.04s/it][A
Iteration:   0%|          | 3/29440 [01:16<206:09:24, 25.21s/it][A
Iteration:   0%|          | 4/29440 [01:44<212:16:43, 25.96s/it][A
Iteration:   0%|          | 5/29440 [02:39<284:57:33, 34.85s/it][A
Iteration:   0%|          | 6/29440 [03:09<272:19:51, 33.31s/it][A
Iteration:   0%|          | 7/29440 [04:32<394:14:02, 48.22s/it][A
Iteration:   0%|          | 8/29440 [05:20<394:28:28, 48.25s/it][A
Iteration:   0%|          | 9/29440 [07:39<617:42:52, 75.56s/it][A
Iteration:   0%|          | 10/29440 [12:08<1090:31:44, 133.40s/it][A
Iteration:   0%|          | 11/29440 [13:31<966:37:33, 118.25s/it] [A
Iteration:   0%|          | 12/29440 [15:04<905:42:28, 110.80s/it][AKilled

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Warning: Permanently added the RSA host key for IP address '140.82.118.3' to the list of known hosts.
From github.com:arita37/mlmodels_store
   0afa1c5..c1a3369  master     -> origin/master
Updating 0afa1c5..c1a3369
Fast-forward
 .../20200515/list_log_dataloader_20200515.md       |    2 +-
 error_list/20200515/list_log_import_20200515.md    |    2 +-
 error_list/20200515/list_log_json_20200515.md      | 1146 +++---
 error_list/20200515/list_log_jupyter_20200515.md   | 1749 +++++----
 error_list/20200515/list_log_test_cli_20200515.md  |  364 +-
 error_list/20200515/list_log_testall_20200515.md   |  363 +-
 ...-16_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py | 3968 ++++++++++++++++++++
 ...-10_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py |  621 +++
 8 files changed, 6417 insertions(+), 1798 deletions(-)
 create mode 100644 log_json/log_json_2020-05-15-21-16_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py
 create mode 100644 log_pullrequest/log_pr_2020-05-15-21-10_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py
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
[master 38bd599] ml_store
 1 file changed, 68 insertions(+)
To github.com:arita37/mlmodels_store.git
   c1a3369..38bd599  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//pytorch_vae.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//pytorch_vae.py", line 34, in <module>
    "beta_vae": md.model.beta_vae,
AttributeError: module 'mlmodels.model_tch.raw.pytorch_vae' has no attribute 'model'

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
