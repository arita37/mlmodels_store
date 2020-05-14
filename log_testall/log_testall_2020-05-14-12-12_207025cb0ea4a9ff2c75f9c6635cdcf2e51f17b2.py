
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
Warning: Permanently added the RSA host key for IP address '192.30.255.113' to the list of known hosts.
From github.com:arita37/mlmodels_store
   51d337f..bc7f5c9  master     -> origin/master
Updating 51d337f..bc7f5c9
Fast-forward
 ...-08_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py | 373 +++++++++++++++++++++
 1 file changed, 373 insertions(+)
 create mode 100644 log_dataloader/log_2020-05-14-12-08_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py
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
[master c973efb] ml_store
 1 file changed, 66 insertions(+)
 create mode 100644 log_testall/log_testall_2020-05-14-12-12_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py
To github.com:arita37/mlmodels_store.git
   bc7f5c9..c973efb  master -> master





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
[master 5bb3029] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   c973efb..5bb3029  master -> master





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
[master 0b50c58] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   5bb3029..0b50c58  master -> master





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
weighted_sequence_layer_1 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         16          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         36          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         12          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-14 12:12:54.363619: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-14 12:12:54.379830: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-05-14 12:12:54.380173: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d522ce2b70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 12:12:54.380227: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 1s 1ms/sample - loss: 0.2501 - binary_crossentropy: 0.6933 - val_loss: 0.2516 - val_binary_crossentropy: 0.7227

  #### metrics   #################################################### 
{'MSE': 0.25060909900631906}

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
weighted_sequence_layer_1 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         16          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         36          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         12          sparse_feature_2[0][0]           
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         4           sequence_mean[0][0]              
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         1           sequence_mean[0][0]              
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
Total params: 383
Trainable params: 383
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2677 - binary_crossentropy: 0.7296500/500 [==============================] - 1s 2ms/sample - loss: 0.2632 - binary_crossentropy: 0.7467 - val_loss: 0.2615 - val_binary_crossentropy: 0.7695

  #### metrics   #################################################### 
{'MSE': 0.261759040180748}

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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         4           sequence_mean[0][0]              
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         1           sequence_mean[0][0]              
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
Total params: 383
Trainable params: 383
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
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
Total params: 642
Trainable params: 642
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2750 - binary_crossentropy: 1.5216500/500 [==============================] - 1s 2ms/sample - loss: 0.2846 - binary_crossentropy: 1.7004 - val_loss: 0.2878 - val_binary_crossentropy: 1.7087

  #### metrics   #################################################### 
{'MSE': 0.28583957778780744}

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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
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
Total params: 642
Trainable params: 642
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         28          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
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
100/500 [=====>........................] - ETA: 2s - loss: 0.2472 - binary_crossentropy: 0.6870500/500 [==============================] - 2s 3ms/sample - loss: 0.2617 - binary_crossentropy: 0.7170 - val_loss: 0.2549 - val_binary_crossentropy: 0.7029

  #### metrics   #################################################### 
{'MSE': 0.25475400085566513}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         28          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_max[0][0]               
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
Total params: 153
Trainable params: 153
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2714 - binary_crossentropy: 0.7388500/500 [==============================] - 2s 3ms/sample - loss: 0.2687 - binary_crossentropy: 0.7333 - val_loss: 0.2648 - val_binary_crossentropy: 0.7252

  #### metrics   #################################################### 
{'MSE': 0.265659207236391}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_max[0][0]               
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-14 12:14:19.569211: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 12:14:19.572018: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 12:14:19.577811: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-14 12:14:19.589522: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-14 12:14:19.592061: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-14 12:14:19.593683: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 12:14:19.595287: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2500 - val_binary_crossentropy: 0.6931
2020-05-14 12:14:20.996837: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 12:14:20.998595: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 12:14:21.002919: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-14 12:14:21.011725: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-14 12:14:21.013282: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-14 12:14:21.014584: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 12:14:21.015904: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2498561284046931}

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
2020-05-14 12:14:46.279885: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 12:14:46.281331: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 12:14:46.284991: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-14 12:14:46.291428: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-14 12:14:46.292595: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-14 12:14:46.293673: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 12:14:46.295058: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2524 - val_binary_crossentropy: 0.6980
2020-05-14 12:14:47.940522: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 12:14:47.941733: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 12:14:47.944411: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-14 12:14:47.949796: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-14 12:14:47.950807: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-14 12:14:47.951643: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 12:14:47.952434: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.25306458137512006}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-14 12:15:23.747025: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 12:15:23.751868: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 12:15:23.766454: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-14 12:15:23.791614: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-14 12:15:23.796479: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-14 12:15:23.800313: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 12:15:23.804253: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.0873 - binary_crossentropy: 0.3502 - val_loss: 0.2980 - val_binary_crossentropy: 0.7998
2020-05-14 12:15:26.212222: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 12:15:26.217793: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 12:15:26.232613: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-14 12:15:26.261214: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-14 12:15:26.266293: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-14 12:15:26.271146: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 12:15:26.276822: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.22501285138291158}

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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
Total params: 700
Trainable params: 700
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2530 - binary_crossentropy: 0.6991500/500 [==============================] - 4s 9ms/sample - loss: 0.2504 - binary_crossentropy: 0.6940 - val_loss: 0.2506 - val_binary_crossentropy: 0.6943

  #### metrics   #################################################### 
{'MSE': 0.25037437189869777}

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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 3, 2)         14          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 2)         10          sequence_mean[0][0]              
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         8           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         14          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         18          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         12          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         14          sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_mean[0][0]              
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
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_5[0][0]           
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
Total params: 284
Trainable params: 284
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.3211 - binary_crossentropy: 0.8800500/500 [==============================] - 5s 9ms/sample - loss: 0.3012 - binary_crossentropy: 0.8793 - val_loss: 0.2973 - val_binary_crossentropy: 0.8304

  #### metrics   #################################################### 
{'MSE': 0.29670681561162077}

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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 3, 2)         14          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 2)         10          sequence_mean[0][0]              
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         8           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         14          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         18          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         12          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         14          sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_mean[0][0]              
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
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_5[0][0]           
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
Total params: 284
Trainable params: 284
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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_max[0][0]               
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
Total params: 1,884
Trainable params: 1,884
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.3785 - binary_crossentropy: 3.9768500/500 [==============================] - 5s 10ms/sample - loss: 0.3599 - binary_crossentropy: 3.2204 - val_loss: 0.3604 - val_binary_crossentropy: 3.2418

  #### metrics   #################################################### 
{'MSE': 0.3586615791668037}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_max[0][0]               
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
Total params: 1,884
Trainable params: 1,884
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
regionsequence_mean (InputLayer [(None, 4)]          0                                            
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
region_10sparse_seq_emb_regions (None, 6, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 4, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 2, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 6, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 4, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 2, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 6, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 4, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 2, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 6, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 4, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 2, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 6, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 4, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 2, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 6, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 4, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 2, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 6, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 4, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 2, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 6, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 4, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 2, 1)         2           regionsequence_max[0][0]         
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
100/500 [=====>........................] - ETA: 9s - loss: 0.3395 - binary_crossentropy: 2.7043500/500 [==============================] - 6s 12ms/sample - loss: 0.3172 - binary_crossentropy: 2.5737 - val_loss: 0.2990 - val_binary_crossentropy: 2.1399

  #### metrics   #################################################### 
{'MSE': 0.307865797099945}

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
regionsequence_mean (InputLayer [(None, 4)]          0                                            
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
region_10sparse_seq_emb_regions (None, 6, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 4, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 2, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 6, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 4, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 2, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 6, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 4, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 2, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 6, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 4, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 2, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 6, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 4, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 2, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 6, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 4, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 2, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 6, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 4, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 2, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 6, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 4, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 2, 1)         2           regionsequence_max[0][0]         
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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
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
100/500 [=====>........................] - ETA: 7s - loss: 0.2484 - binary_crossentropy: 0.8199500/500 [==============================] - 6s 12ms/sample - loss: 0.2648 - binary_crossentropy: 1.0105 - val_loss: 0.2663 - val_binary_crossentropy: 1.0138

  #### metrics   #################################################### 
{'MSE': 0.2646512113661072}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_emb_sparse_feature_1_spa (None, 1, 4)         4           hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 6, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 6, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 6, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 6, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_max[0][0]               
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
Total params: 2,967
Trainable params: 2,887
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.5100 - binary_crossentropy: 7.8667500/500 [==============================] - 7s 13ms/sample - loss: 0.4360 - binary_crossentropy: 6.7253 - val_loss: 0.5020 - val_binary_crossentropy: 7.7433

  #### metrics   #################################################### 
{'MSE': 0.469}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_emb_sparse_feature_1_spa (None, 1, 4)         4           hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 6, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 6, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 6, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 6, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_max[0][0]               
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
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         20          sequence_max[0][0]               
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
Total params: 421
Trainable params: 421
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2590 - binary_crossentropy: 0.7124500/500 [==============================] - 7s 13ms/sample - loss: 0.2597 - binary_crossentropy: 0.7137 - val_loss: 0.2535 - val_binary_crossentropy: 0.7002

  #### metrics   #################################################### 
{'MSE': 0.2536811017846551}

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
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         20          sequence_max[0][0]               
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
Total params: 421
Trainable params: 421
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_44 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_125 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_194[0][0] 
                                                                 sequence_pooling_layer_195[0][0] 
                                                                 sequence_pooling_layer_196[0][0] 
                                                                 sequence_pooling_layer_197[0][0] 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
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
Total params: 2,009
Trainable params: 2,009
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 7s 14ms/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2500 - val_binary_crossentropy: 0.6930

  #### metrics   #################################################### 
{'MSE': 0.2498914015433677}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_44 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_125 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_194[0][0] 
                                                                 sequence_pooling_layer_195[0][0] 
                                                                 sequence_pooling_layer_196[0][0] 
                                                                 sequence_pooling_layer_197[0][0] 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
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
Total params: 2,009
Trainable params: 2,009
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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_47 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_90 (Concatenate)    (None, 1, 20)        0           no_mask_130[0][0]                
                                                                 no_mask_130[1][0]                
                                                                 no_mask_130[2][0]                
                                                                 no_mask_130[3][0]                
                                                                 no_mask_130[4][0]                
__________________________________________________________________________________________________
no_mask_131 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
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
Total params: 311
Trainable params: 311
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2688 - binary_crossentropy: 0.7323500/500 [==============================] - 7s 15ms/sample - loss: 0.2592 - binary_crossentropy: 0.7125 - val_loss: 0.2592 - val_binary_crossentropy: 0.7118

  #### metrics   #################################################### 
{'MSE': 0.25822723961813226}

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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_47 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_90 (Concatenate)    (None, 1, 20)        0           no_mask_130[0][0]                
                                                                 no_mask_130[1][0]                
                                                                 no_mask_130[2][0]                
                                                                 no_mask_130[3][0]                
                                                                 no_mask_130[4][0]                
__________________________________________________________________________________________________
no_mask_131 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
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
Total params: 311
Trainable params: 311
Non-trainable params: 0
__________________________________________________________________________________________________

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   0b50c58..aeccb11  master     -> origin/master
Updating 0b50c58..aeccb11
Fast-forward
 .../20200514/list_log_dataloader_20200514.md       |    2 +-
 error_list/20200514/list_log_json_20200514.md      | 1146 ++++++-------
 error_list/20200514/list_log_jupyter_20200514.md   | 1773 ++++++++++----------
 .../20200514/list_log_pullrequest_20200514.md      |    2 +-
 error_list/20200514/list_log_testall_20200514.md   |  809 +--------
 ...-10_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py |  611 +++++++
 6 files changed, 2077 insertions(+), 2266 deletions(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-14-12-10_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py
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
[master 288435a] ml_store
 1 file changed, 5672 insertions(+)
To github.com:arita37/mlmodels_store.git
   aeccb11..288435a  master -> master





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
[master cc7c80d] ml_store
 1 file changed, 50 insertions(+)
To github.com:arita37/mlmodels_store.git
   288435a..cc7c80d  master -> master





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
[master d525d4f] ml_store
 1 file changed, 46 insertions(+)
To github.com:arita37/mlmodels_store.git
   cc7c80d..d525d4f  master -> master





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
[master bc686c2] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   d525d4f..bc686c2  master -> master





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

2020-05-14 12:29:00.707799: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-14 12:29:00.713133: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-05-14 12:29:00.713306: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ec35d069f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 12:29:00.713324: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 8s - loss: 1.3817
256/354 [====================>.........] - ETA: 3s - loss: 1.1954
354/354 [==============================] - 14s 40ms/step - loss: 1.3607 - val_loss: 2.4322

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
[master 721d535] ml_store
 1 file changed, 149 insertions(+)
To github.com:arita37/mlmodels_store.git
   bc686c2..721d535  master -> master





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
[master f350101] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   721d535..f350101  master -> master





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
[master cafa3e8] ml_store
 1 file changed, 44 insertions(+)
To github.com:arita37/mlmodels_store.git
   f350101..cafa3e8  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 43s
   57344/17464789 [..............................] - ETA: 37s
   90112/17464789 [..............................] - ETA: 36s
  180224/17464789 [..............................] - ETA: 23s
  335872/17464789 [..............................] - ETA: 15s
  647168/17464789 [>.............................] - ETA: 9s 
 1294336/17464789 [=>............................] - ETA: 5s
 2572288/17464789 [===>..........................] - ETA: 2s
 5128192/17464789 [=======>......................] - ETA: 1s
 7864320/17464789 [============>.................] - ETA: 0s
10518528/17464789 [=================>............] - ETA: 0s
13434880/17464789 [======================>.......] - ETA: 0s
16269312/17464789 [==========================>...] - ETA: 0s
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
2020-05-14 12:30:07.021012: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-14 12:30:07.025775: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-05-14 12:30:07.025920: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55be3cae33a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 12:30:07.025936: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 13s - loss: 7.5593 - accuracy: 0.5070
 2000/25000 [=>............................] - ETA: 9s - loss: 7.5440 - accuracy: 0.5080 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6513 - accuracy: 0.5010
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5670 - accuracy: 0.5065
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5900 - accuracy: 0.5050
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5848 - accuracy: 0.5053
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5330 - accuracy: 0.5087
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5631 - accuracy: 0.5067
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5831 - accuracy: 0.5054
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5961 - accuracy: 0.5046
11000/25000 [============>.................] - ETA: 4s - loss: 7.6248 - accuracy: 0.5027
12000/25000 [=============>................] - ETA: 4s - loss: 7.6411 - accuracy: 0.5017
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6100 - accuracy: 0.5037
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6414 - accuracy: 0.5016
15000/25000 [=================>............] - ETA: 3s - loss: 7.6360 - accuracy: 0.5020
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6484 - accuracy: 0.5012
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6495 - accuracy: 0.5011
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6504 - accuracy: 0.5011
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6513 - accuracy: 0.5010
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6268 - accuracy: 0.5026
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6440 - accuracy: 0.5015
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6422 - accuracy: 0.5016
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6433 - accuracy: 0.5015
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6519 - accuracy: 0.5010
25000/25000 [==============================] - 9s 362us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7fb71319b908>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7fb7098b1390> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.6666 - accuracy: 0.5000
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6283 - accuracy: 0.5025 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.4622 - accuracy: 0.5133
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5171 - accuracy: 0.5098
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5225 - accuracy: 0.5094
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5516 - accuracy: 0.5075
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5834 - accuracy: 0.5054
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5880 - accuracy: 0.5051
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6087 - accuracy: 0.5038
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6237 - accuracy: 0.5028
11000/25000 [============>.................] - ETA: 4s - loss: 7.5983 - accuracy: 0.5045
12000/25000 [=============>................] - ETA: 3s - loss: 7.5989 - accuracy: 0.5044
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6112 - accuracy: 0.5036
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6338 - accuracy: 0.5021
15000/25000 [=================>............] - ETA: 3s - loss: 7.6441 - accuracy: 0.5015
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6321 - accuracy: 0.5023
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6450 - accuracy: 0.5014
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6726 - accuracy: 0.4996
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6715 - accuracy: 0.4997
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6735 - accuracy: 0.4996
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6659 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6715 - accuracy: 0.4997
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6733 - accuracy: 0.4996
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6653 - accuracy: 0.5001
25000/25000 [==============================] - 9s 360us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 13s - loss: 7.9120 - accuracy: 0.4840
 2000/25000 [=>............................] - ETA: 9s - loss: 7.7970 - accuracy: 0.4915 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.8046 - accuracy: 0.4910
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6896 - accuracy: 0.4985
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6973 - accuracy: 0.4980
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6513 - accuracy: 0.5010
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6360 - accuracy: 0.5020
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6245 - accuracy: 0.5027
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6172 - accuracy: 0.5032
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6283 - accuracy: 0.5025
11000/25000 [============>.................] - ETA: 4s - loss: 7.6680 - accuracy: 0.4999
12000/25000 [=============>................] - ETA: 4s - loss: 7.6513 - accuracy: 0.5010
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6348 - accuracy: 0.5021
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6491 - accuracy: 0.5011
15000/25000 [=================>............] - ETA: 3s - loss: 7.6595 - accuracy: 0.5005
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6657 - accuracy: 0.5001
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6621 - accuracy: 0.5003
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6513 - accuracy: 0.5010
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6505 - accuracy: 0.5011
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6444 - accuracy: 0.5015
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6601 - accuracy: 0.5004
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6548 - accuracy: 0.5008
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6546 - accuracy: 0.5008
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6634 - accuracy: 0.5002
25000/25000 [==============================] - 9s 373us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
   cafa3e8..1847ac0  master     -> origin/master
Updating cafa3e8..1847ac0
Fast-forward
 .../20200514/list_log_dataloader_20200514.md       |    2 +-
 error_list/20200514/list_log_json_20200514.md      | 1146 ++++++++++----------
 error_list/20200514/list_log_testall_20200514.md   |   89 ++
 3 files changed, 663 insertions(+), 574 deletions(-)
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
[master 59421be] ml_store
 1 file changed, 334 insertions(+)
To github.com:arita37/mlmodels_store.git
   1847ac0..59421be  master -> master





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

13/13 [==============================] - 2s 141ms/step - loss: nan
Epoch 2/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 3/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 4/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 5/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 6/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 7/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 8/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 9/10

13/13 [==============================] - 0s 5ms/step - loss: nan
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
[master 1d1320d] ml_store
 1 file changed, 125 insertions(+)
To github.com:arita37/mlmodels_store.git
   59421be..1d1320d  master -> master





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
   90112/11490434 [..............................] - ETA: 24s
  180224/11490434 [..............................] - ETA: 16s
  335872/11490434 [..............................] - ETA: 10s
  647168/11490434 [>.............................] - ETA: 6s 
 1286144/11490434 [==>...........................] - ETA: 3s
 2547712/11490434 [=====>........................] - ETA: 1s
 5038080/11490434 [============>.................] - ETA: 0s
 7987200/11490434 [===================>..........] - ETA: 0s
10985472/11490434 [===========================>..] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:33 - loss: 2.3092 - categorical_accuracy: 0.0312
   64/60000 [..............................] - ETA: 4:42 - loss: 2.2862 - categorical_accuracy: 0.0938
   96/60000 [..............................] - ETA: 3:43 - loss: 2.2523 - categorical_accuracy: 0.1354
  128/60000 [..............................] - ETA: 3:13 - loss: 2.2206 - categorical_accuracy: 0.1875
  160/60000 [..............................] - ETA: 2:55 - loss: 2.1619 - categorical_accuracy: 0.2000
  192/60000 [..............................] - ETA: 2:44 - loss: 2.1463 - categorical_accuracy: 0.2031
  224/60000 [..............................] - ETA: 2:36 - loss: 2.1123 - categorical_accuracy: 0.2143
  256/60000 [..............................] - ETA: 2:30 - loss: 2.0744 - categorical_accuracy: 0.2344
  288/60000 [..............................] - ETA: 2:24 - loss: 2.0211 - categorical_accuracy: 0.2743
  320/60000 [..............................] - ETA: 2:20 - loss: 1.9828 - categorical_accuracy: 0.3000
  352/60000 [..............................] - ETA: 2:18 - loss: 1.9702 - categorical_accuracy: 0.3153
  384/60000 [..............................] - ETA: 2:15 - loss: 1.9265 - categorical_accuracy: 0.3438
  416/60000 [..............................] - ETA: 2:12 - loss: 1.8849 - categorical_accuracy: 0.3558
  448/60000 [..............................] - ETA: 2:10 - loss: 1.8572 - categorical_accuracy: 0.3728
  480/60000 [..............................] - ETA: 2:08 - loss: 1.8356 - categorical_accuracy: 0.3812
  512/60000 [..............................] - ETA: 2:06 - loss: 1.8196 - categorical_accuracy: 0.3809
  544/60000 [..............................] - ETA: 2:05 - loss: 1.7844 - categorical_accuracy: 0.3989
  576/60000 [..............................] - ETA: 2:03 - loss: 1.7602 - categorical_accuracy: 0.4097
  608/60000 [..............................] - ETA: 2:02 - loss: 1.7291 - categorical_accuracy: 0.4211
  640/60000 [..............................] - ETA: 2:02 - loss: 1.7115 - categorical_accuracy: 0.4281
  672/60000 [..............................] - ETA: 2:01 - loss: 1.6932 - categorical_accuracy: 0.4390
  704/60000 [..............................] - ETA: 2:00 - loss: 1.6732 - categorical_accuracy: 0.4474
  736/60000 [..............................] - ETA: 1:59 - loss: 1.6442 - categorical_accuracy: 0.4579
  768/60000 [..............................] - ETA: 1:59 - loss: 1.6109 - categorical_accuracy: 0.4701
  800/60000 [..............................] - ETA: 1:58 - loss: 1.5898 - categorical_accuracy: 0.4750
  832/60000 [..............................] - ETA: 1:58 - loss: 1.5680 - categorical_accuracy: 0.4784
  864/60000 [..............................] - ETA: 1:57 - loss: 1.5417 - categorical_accuracy: 0.4861
  896/60000 [..............................] - ETA: 1:56 - loss: 1.5361 - categorical_accuracy: 0.4877
  928/60000 [..............................] - ETA: 1:56 - loss: 1.5288 - categorical_accuracy: 0.4881
  960/60000 [..............................] - ETA: 1:56 - loss: 1.5038 - categorical_accuracy: 0.4969
  992/60000 [..............................] - ETA: 1:55 - loss: 1.4756 - categorical_accuracy: 0.5071
 1024/60000 [..............................] - ETA: 1:55 - loss: 1.4528 - categorical_accuracy: 0.5166
 1056/60000 [..............................] - ETA: 1:55 - loss: 1.4376 - categorical_accuracy: 0.5208
 1088/60000 [..............................] - ETA: 1:55 - loss: 1.4222 - categorical_accuracy: 0.5230
 1120/60000 [..............................] - ETA: 1:55 - loss: 1.4084 - categorical_accuracy: 0.5259
 1152/60000 [..............................] - ETA: 1:54 - loss: 1.3932 - categorical_accuracy: 0.5286
 1184/60000 [..............................] - ETA: 1:54 - loss: 1.3903 - categorical_accuracy: 0.5312
 1216/60000 [..............................] - ETA: 1:54 - loss: 1.3718 - categorical_accuracy: 0.5403
 1248/60000 [..............................] - ETA: 1:54 - loss: 1.3554 - categorical_accuracy: 0.5457
 1280/60000 [..............................] - ETA: 1:53 - loss: 1.3504 - categorical_accuracy: 0.5453
 1312/60000 [..............................] - ETA: 1:53 - loss: 1.3406 - categorical_accuracy: 0.5488
 1344/60000 [..............................] - ETA: 1:53 - loss: 1.3287 - categorical_accuracy: 0.5536
 1376/60000 [..............................] - ETA: 1:53 - loss: 1.3122 - categorical_accuracy: 0.5589
 1408/60000 [..............................] - ETA: 1:53 - loss: 1.3012 - categorical_accuracy: 0.5604
 1440/60000 [..............................] - ETA: 1:52 - loss: 1.2912 - categorical_accuracy: 0.5646
 1472/60000 [..............................] - ETA: 1:52 - loss: 1.2711 - categorical_accuracy: 0.5720
 1504/60000 [..............................] - ETA: 1:52 - loss: 1.2537 - categorical_accuracy: 0.5785
 1536/60000 [..............................] - ETA: 1:51 - loss: 1.2365 - categorical_accuracy: 0.5840
 1568/60000 [..............................] - ETA: 1:51 - loss: 1.2209 - categorical_accuracy: 0.5893
 1600/60000 [..............................] - ETA: 1:51 - loss: 1.2073 - categorical_accuracy: 0.5938
 1632/60000 [..............................] - ETA: 1:51 - loss: 1.1978 - categorical_accuracy: 0.5980
 1664/60000 [..............................] - ETA: 1:50 - loss: 1.1853 - categorical_accuracy: 0.6016
 1696/60000 [..............................] - ETA: 1:50 - loss: 1.1744 - categorical_accuracy: 0.6055
 1728/60000 [..............................] - ETA: 1:50 - loss: 1.1687 - categorical_accuracy: 0.6082
 1760/60000 [..............................] - ETA: 1:49 - loss: 1.1599 - categorical_accuracy: 0.6114
 1792/60000 [..............................] - ETA: 1:49 - loss: 1.1457 - categorical_accuracy: 0.6155
 1824/60000 [..............................] - ETA: 1:49 - loss: 1.1353 - categorical_accuracy: 0.6190
 1856/60000 [..............................] - ETA: 1:48 - loss: 1.1224 - categorical_accuracy: 0.6228
 1888/60000 [..............................] - ETA: 1:48 - loss: 1.1187 - categorical_accuracy: 0.6245
 1920/60000 [..............................] - ETA: 1:48 - loss: 1.1156 - categorical_accuracy: 0.6255
 1952/60000 [..............................] - ETA: 1:47 - loss: 1.1052 - categorical_accuracy: 0.6281
 1984/60000 [..............................] - ETA: 1:47 - loss: 1.0964 - categorical_accuracy: 0.6316
 2016/60000 [>.............................] - ETA: 1:47 - loss: 1.0839 - categorical_accuracy: 0.6359
 2048/60000 [>.............................] - ETA: 1:47 - loss: 1.0823 - categorical_accuracy: 0.6382
 2080/60000 [>.............................] - ETA: 1:47 - loss: 1.0725 - categorical_accuracy: 0.6418
 2112/60000 [>.............................] - ETA: 1:47 - loss: 1.0667 - categorical_accuracy: 0.6449
 2144/60000 [>.............................] - ETA: 1:46 - loss: 1.0573 - categorical_accuracy: 0.6483
 2176/60000 [>.............................] - ETA: 1:46 - loss: 1.0469 - categorical_accuracy: 0.6517
 2208/60000 [>.............................] - ETA: 1:46 - loss: 1.0370 - categorical_accuracy: 0.6553
 2240/60000 [>.............................] - ETA: 1:46 - loss: 1.0331 - categorical_accuracy: 0.6576
 2272/60000 [>.............................] - ETA: 1:46 - loss: 1.0236 - categorical_accuracy: 0.6607
 2304/60000 [>.............................] - ETA: 1:46 - loss: 1.0149 - categorical_accuracy: 0.6632
 2336/60000 [>.............................] - ETA: 1:45 - loss: 1.0053 - categorical_accuracy: 0.6665
 2368/60000 [>.............................] - ETA: 1:45 - loss: 0.9934 - categorical_accuracy: 0.6710
 2400/60000 [>.............................] - ETA: 1:45 - loss: 0.9884 - categorical_accuracy: 0.6717
 2432/60000 [>.............................] - ETA: 1:45 - loss: 0.9873 - categorical_accuracy: 0.6723
 2464/60000 [>.............................] - ETA: 1:45 - loss: 0.9830 - categorical_accuracy: 0.6737
 2496/60000 [>.............................] - ETA: 1:45 - loss: 0.9794 - categorical_accuracy: 0.6755
 2528/60000 [>.............................] - ETA: 1:44 - loss: 0.9720 - categorical_accuracy: 0.6788
 2560/60000 [>.............................] - ETA: 1:44 - loss: 0.9655 - categorical_accuracy: 0.6820
 2592/60000 [>.............................] - ETA: 1:44 - loss: 0.9571 - categorical_accuracy: 0.6844
 2624/60000 [>.............................] - ETA: 1:44 - loss: 0.9494 - categorical_accuracy: 0.6860
 2656/60000 [>.............................] - ETA: 1:44 - loss: 0.9419 - categorical_accuracy: 0.6886
 2688/60000 [>.............................] - ETA: 1:44 - loss: 0.9369 - categorical_accuracy: 0.6901
 2720/60000 [>.............................] - ETA: 1:44 - loss: 0.9342 - categorical_accuracy: 0.6904
 2752/60000 [>.............................] - ETA: 1:44 - loss: 0.9260 - categorical_accuracy: 0.6937
 2784/60000 [>.............................] - ETA: 1:44 - loss: 0.9171 - categorical_accuracy: 0.6965
 2816/60000 [>.............................] - ETA: 1:44 - loss: 0.9106 - categorical_accuracy: 0.6996
 2848/60000 [>.............................] - ETA: 1:44 - loss: 0.9059 - categorical_accuracy: 0.7012
 2880/60000 [>.............................] - ETA: 1:44 - loss: 0.9016 - categorical_accuracy: 0.7024
 2912/60000 [>.............................] - ETA: 1:43 - loss: 0.8959 - categorical_accuracy: 0.7040
 2944/60000 [>.............................] - ETA: 1:43 - loss: 0.8891 - categorical_accuracy: 0.7062
 2976/60000 [>.............................] - ETA: 1:43 - loss: 0.8885 - categorical_accuracy: 0.7070
 3008/60000 [>.............................] - ETA: 1:43 - loss: 0.8830 - categorical_accuracy: 0.7084
 3040/60000 [>.............................] - ETA: 1:43 - loss: 0.8766 - categorical_accuracy: 0.7102
 3072/60000 [>.............................] - ETA: 1:43 - loss: 0.8697 - categorical_accuracy: 0.7129
 3104/60000 [>.............................] - ETA: 1:43 - loss: 0.8638 - categorical_accuracy: 0.7139
 3136/60000 [>.............................] - ETA: 1:43 - loss: 0.8594 - categorical_accuracy: 0.7149
 3168/60000 [>.............................] - ETA: 1:43 - loss: 0.8553 - categorical_accuracy: 0.7169
 3200/60000 [>.............................] - ETA: 1:42 - loss: 0.8513 - categorical_accuracy: 0.7175
 3232/60000 [>.............................] - ETA: 1:42 - loss: 0.8450 - categorical_accuracy: 0.7197
 3264/60000 [>.............................] - ETA: 1:42 - loss: 0.8431 - categorical_accuracy: 0.7206
 3296/60000 [>.............................] - ETA: 1:42 - loss: 0.8378 - categorical_accuracy: 0.7224
 3328/60000 [>.............................] - ETA: 1:42 - loss: 0.8321 - categorical_accuracy: 0.7245
 3360/60000 [>.............................] - ETA: 1:42 - loss: 0.8290 - categorical_accuracy: 0.7259
 3392/60000 [>.............................] - ETA: 1:42 - loss: 0.8245 - categorical_accuracy: 0.7276
 3424/60000 [>.............................] - ETA: 1:42 - loss: 0.8208 - categorical_accuracy: 0.7287
 3456/60000 [>.............................] - ETA: 1:42 - loss: 0.8167 - categorical_accuracy: 0.7300
 3488/60000 [>.............................] - ETA: 1:42 - loss: 0.8127 - categorical_accuracy: 0.7308
 3520/60000 [>.............................] - ETA: 1:41 - loss: 0.8112 - categorical_accuracy: 0.7315
 3552/60000 [>.............................] - ETA: 1:41 - loss: 0.8068 - categorical_accuracy: 0.7328
 3584/60000 [>.............................] - ETA: 1:41 - loss: 0.8022 - categorical_accuracy: 0.7341
 3616/60000 [>.............................] - ETA: 1:41 - loss: 0.7991 - categorical_accuracy: 0.7356
 3648/60000 [>.............................] - ETA: 1:41 - loss: 0.7960 - categorical_accuracy: 0.7368
 3680/60000 [>.............................] - ETA: 1:41 - loss: 0.7907 - categorical_accuracy: 0.7389
 3712/60000 [>.............................] - ETA: 1:41 - loss: 0.7870 - categorical_accuracy: 0.7400
 3744/60000 [>.............................] - ETA: 1:41 - loss: 0.7834 - categorical_accuracy: 0.7417
 3776/60000 [>.............................] - ETA: 1:41 - loss: 0.7796 - categorical_accuracy: 0.7428
 3808/60000 [>.............................] - ETA: 1:41 - loss: 0.7765 - categorical_accuracy: 0.7437
 3840/60000 [>.............................] - ETA: 1:41 - loss: 0.7716 - categorical_accuracy: 0.7456
 3872/60000 [>.............................] - ETA: 1:40 - loss: 0.7669 - categorical_accuracy: 0.7472
 3904/60000 [>.............................] - ETA: 1:40 - loss: 0.7612 - categorical_accuracy: 0.7492
 3936/60000 [>.............................] - ETA: 1:40 - loss: 0.7591 - categorical_accuracy: 0.7500
 3968/60000 [>.............................] - ETA: 1:40 - loss: 0.7556 - categorical_accuracy: 0.7510
 4000/60000 [=>............................] - ETA: 1:40 - loss: 0.7540 - categorical_accuracy: 0.7515
 4032/60000 [=>............................] - ETA: 1:40 - loss: 0.7500 - categorical_accuracy: 0.7530
 4064/60000 [=>............................] - ETA: 1:40 - loss: 0.7466 - categorical_accuracy: 0.7537
 4096/60000 [=>............................] - ETA: 1:40 - loss: 0.7420 - categorical_accuracy: 0.7556
 4128/60000 [=>............................] - ETA: 1:39 - loss: 0.7371 - categorical_accuracy: 0.7573
 4160/60000 [=>............................] - ETA: 1:39 - loss: 0.7340 - categorical_accuracy: 0.7579
 4192/60000 [=>............................] - ETA: 1:39 - loss: 0.7296 - categorical_accuracy: 0.7595
 4224/60000 [=>............................] - ETA: 1:39 - loss: 0.7279 - categorical_accuracy: 0.7602
 4256/60000 [=>............................] - ETA: 1:39 - loss: 0.7245 - categorical_accuracy: 0.7615
 4288/60000 [=>............................] - ETA: 1:39 - loss: 0.7209 - categorical_accuracy: 0.7631
 4320/60000 [=>............................] - ETA: 1:39 - loss: 0.7172 - categorical_accuracy: 0.7641
 4352/60000 [=>............................] - ETA: 1:39 - loss: 0.7144 - categorical_accuracy: 0.7649
 4384/60000 [=>............................] - ETA: 1:39 - loss: 0.7117 - categorical_accuracy: 0.7660
 4416/60000 [=>............................] - ETA: 1:38 - loss: 0.7077 - categorical_accuracy: 0.7672
 4448/60000 [=>............................] - ETA: 1:38 - loss: 0.7043 - categorical_accuracy: 0.7682
 4480/60000 [=>............................] - ETA: 1:38 - loss: 0.7022 - categorical_accuracy: 0.7688
 4512/60000 [=>............................] - ETA: 1:38 - loss: 0.6983 - categorical_accuracy: 0.7699
 4544/60000 [=>............................] - ETA: 1:38 - loss: 0.6951 - categorical_accuracy: 0.7711
 4576/60000 [=>............................] - ETA: 1:38 - loss: 0.6935 - categorical_accuracy: 0.7716
 4608/60000 [=>............................] - ETA: 1:38 - loss: 0.6914 - categorical_accuracy: 0.7724
 4640/60000 [=>............................] - ETA: 1:38 - loss: 0.6893 - categorical_accuracy: 0.7733
 4672/60000 [=>............................] - ETA: 1:38 - loss: 0.6862 - categorical_accuracy: 0.7746
 4704/60000 [=>............................] - ETA: 1:38 - loss: 0.6827 - categorical_accuracy: 0.7759
 4736/60000 [=>............................] - ETA: 1:38 - loss: 0.6800 - categorical_accuracy: 0.7768
 4768/60000 [=>............................] - ETA: 1:38 - loss: 0.6765 - categorical_accuracy: 0.7781
 4800/60000 [=>............................] - ETA: 1:38 - loss: 0.6742 - categorical_accuracy: 0.7790
 4832/60000 [=>............................] - ETA: 1:38 - loss: 0.6730 - categorical_accuracy: 0.7792
 4864/60000 [=>............................] - ETA: 1:38 - loss: 0.6705 - categorical_accuracy: 0.7800
 4896/60000 [=>............................] - ETA: 1:37 - loss: 0.6678 - categorical_accuracy: 0.7808
 4928/60000 [=>............................] - ETA: 1:37 - loss: 0.6662 - categorical_accuracy: 0.7819
 4960/60000 [=>............................] - ETA: 1:37 - loss: 0.6653 - categorical_accuracy: 0.7823
 4992/60000 [=>............................] - ETA: 1:37 - loss: 0.6621 - categorical_accuracy: 0.7833
 5024/60000 [=>............................] - ETA: 1:37 - loss: 0.6595 - categorical_accuracy: 0.7844
 5056/60000 [=>............................] - ETA: 1:37 - loss: 0.6575 - categorical_accuracy: 0.7850
 5088/60000 [=>............................] - ETA: 1:37 - loss: 0.6550 - categorical_accuracy: 0.7860
 5120/60000 [=>............................] - ETA: 1:37 - loss: 0.6540 - categorical_accuracy: 0.7865
 5152/60000 [=>............................] - ETA: 1:37 - loss: 0.6522 - categorical_accuracy: 0.7869
 5184/60000 [=>............................] - ETA: 1:37 - loss: 0.6486 - categorical_accuracy: 0.7882
 5216/60000 [=>............................] - ETA: 1:37 - loss: 0.6482 - categorical_accuracy: 0.7887
 5248/60000 [=>............................] - ETA: 1:37 - loss: 0.6466 - categorical_accuracy: 0.7894
 5280/60000 [=>............................] - ETA: 1:37 - loss: 0.6443 - categorical_accuracy: 0.7903
 5312/60000 [=>............................] - ETA: 1:37 - loss: 0.6427 - categorical_accuracy: 0.7910
 5344/60000 [=>............................] - ETA: 1:37 - loss: 0.6414 - categorical_accuracy: 0.7915
 5376/60000 [=>............................] - ETA: 1:36 - loss: 0.6389 - categorical_accuracy: 0.7924
 5408/60000 [=>............................] - ETA: 1:36 - loss: 0.6363 - categorical_accuracy: 0.7935
 5440/60000 [=>............................] - ETA: 1:36 - loss: 0.6341 - categorical_accuracy: 0.7941
 5472/60000 [=>............................] - ETA: 1:36 - loss: 0.6315 - categorical_accuracy: 0.7950
 5504/60000 [=>............................] - ETA: 1:36 - loss: 0.6315 - categorical_accuracy: 0.7952
 5536/60000 [=>............................] - ETA: 1:36 - loss: 0.6294 - categorical_accuracy: 0.7961
 5568/60000 [=>............................] - ETA: 1:36 - loss: 0.6275 - categorical_accuracy: 0.7967
 5600/60000 [=>............................] - ETA: 1:36 - loss: 0.6249 - categorical_accuracy: 0.7977
 5632/60000 [=>............................] - ETA: 1:36 - loss: 0.6242 - categorical_accuracy: 0.7983
 5664/60000 [=>............................] - ETA: 1:36 - loss: 0.6216 - categorical_accuracy: 0.7991
 5696/60000 [=>............................] - ETA: 1:36 - loss: 0.6194 - categorical_accuracy: 0.7999
 5728/60000 [=>............................] - ETA: 1:36 - loss: 0.6176 - categorical_accuracy: 0.8003
 5760/60000 [=>............................] - ETA: 1:36 - loss: 0.6144 - categorical_accuracy: 0.8014
 5792/60000 [=>............................] - ETA: 1:36 - loss: 0.6136 - categorical_accuracy: 0.8016
 5824/60000 [=>............................] - ETA: 1:36 - loss: 0.6136 - categorical_accuracy: 0.8017
 5856/60000 [=>............................] - ETA: 1:36 - loss: 0.6111 - categorical_accuracy: 0.8026
 5888/60000 [=>............................] - ETA: 1:36 - loss: 0.6095 - categorical_accuracy: 0.8030
 5920/60000 [=>............................] - ETA: 1:36 - loss: 0.6075 - categorical_accuracy: 0.8037
 5952/60000 [=>............................] - ETA: 1:36 - loss: 0.6076 - categorical_accuracy: 0.8038
 5984/60000 [=>............................] - ETA: 1:35 - loss: 0.6051 - categorical_accuracy: 0.8045
 6016/60000 [==>...........................] - ETA: 1:35 - loss: 0.6036 - categorical_accuracy: 0.8050
 6048/60000 [==>...........................] - ETA: 1:35 - loss: 0.6011 - categorical_accuracy: 0.8061
 6080/60000 [==>...........................] - ETA: 1:35 - loss: 0.6001 - categorical_accuracy: 0.8064
 6112/60000 [==>...........................] - ETA: 1:35 - loss: 0.5996 - categorical_accuracy: 0.8069
 6144/60000 [==>...........................] - ETA: 1:35 - loss: 0.5969 - categorical_accuracy: 0.8079
 6176/60000 [==>...........................] - ETA: 1:35 - loss: 0.5945 - categorical_accuracy: 0.8086
 6208/60000 [==>...........................] - ETA: 1:35 - loss: 0.5930 - categorical_accuracy: 0.8093
 6240/60000 [==>...........................] - ETA: 1:35 - loss: 0.5912 - categorical_accuracy: 0.8099
 6272/60000 [==>...........................] - ETA: 1:35 - loss: 0.5901 - categorical_accuracy: 0.8101
 6304/60000 [==>...........................] - ETA: 1:35 - loss: 0.5882 - categorical_accuracy: 0.8106
 6336/60000 [==>...........................] - ETA: 1:35 - loss: 0.5864 - categorical_accuracy: 0.8109
 6368/60000 [==>...........................] - ETA: 1:35 - loss: 0.5847 - categorical_accuracy: 0.8116
 6400/60000 [==>...........................] - ETA: 1:35 - loss: 0.5837 - categorical_accuracy: 0.8117
 6432/60000 [==>...........................] - ETA: 1:35 - loss: 0.5814 - categorical_accuracy: 0.8125
 6464/60000 [==>...........................] - ETA: 1:35 - loss: 0.5798 - categorical_accuracy: 0.8128
 6496/60000 [==>...........................] - ETA: 1:35 - loss: 0.5785 - categorical_accuracy: 0.8133
 6528/60000 [==>...........................] - ETA: 1:35 - loss: 0.5778 - categorical_accuracy: 0.8136
 6560/60000 [==>...........................] - ETA: 1:35 - loss: 0.5756 - categorical_accuracy: 0.8143
 6592/60000 [==>...........................] - ETA: 1:35 - loss: 0.5745 - categorical_accuracy: 0.8148
 6624/60000 [==>...........................] - ETA: 1:35 - loss: 0.5724 - categorical_accuracy: 0.8155
 6656/60000 [==>...........................] - ETA: 1:35 - loss: 0.5713 - categorical_accuracy: 0.8157
 6688/60000 [==>...........................] - ETA: 1:35 - loss: 0.5691 - categorical_accuracy: 0.8164
 6720/60000 [==>...........................] - ETA: 1:34 - loss: 0.5670 - categorical_accuracy: 0.8171
 6752/60000 [==>...........................] - ETA: 1:34 - loss: 0.5653 - categorical_accuracy: 0.8177
 6784/60000 [==>...........................] - ETA: 1:34 - loss: 0.5648 - categorical_accuracy: 0.8180
 6816/60000 [==>...........................] - ETA: 1:34 - loss: 0.5630 - categorical_accuracy: 0.8185
 6848/60000 [==>...........................] - ETA: 1:34 - loss: 0.5609 - categorical_accuracy: 0.8192
 6880/60000 [==>...........................] - ETA: 1:34 - loss: 0.5593 - categorical_accuracy: 0.8198
 6912/60000 [==>...........................] - ETA: 1:34 - loss: 0.5570 - categorical_accuracy: 0.8206
 6944/60000 [==>...........................] - ETA: 1:34 - loss: 0.5554 - categorical_accuracy: 0.8213
 6976/60000 [==>...........................] - ETA: 1:34 - loss: 0.5536 - categorical_accuracy: 0.8218
 7008/60000 [==>...........................] - ETA: 1:34 - loss: 0.5525 - categorical_accuracy: 0.8221
 7040/60000 [==>...........................] - ETA: 1:34 - loss: 0.5513 - categorical_accuracy: 0.8224
 7072/60000 [==>...........................] - ETA: 1:34 - loss: 0.5494 - categorical_accuracy: 0.8230
 7104/60000 [==>...........................] - ETA: 1:34 - loss: 0.5492 - categorical_accuracy: 0.8233
 7136/60000 [==>...........................] - ETA: 1:34 - loss: 0.5479 - categorical_accuracy: 0.8237
 7168/60000 [==>...........................] - ETA: 1:34 - loss: 0.5460 - categorical_accuracy: 0.8244
 7200/60000 [==>...........................] - ETA: 1:34 - loss: 0.5443 - categorical_accuracy: 0.8249
 7232/60000 [==>...........................] - ETA: 1:34 - loss: 0.5426 - categorical_accuracy: 0.8254
 7264/60000 [==>...........................] - ETA: 1:34 - loss: 0.5407 - categorical_accuracy: 0.8260
 7296/60000 [==>...........................] - ETA: 1:34 - loss: 0.5393 - categorical_accuracy: 0.8266
 7328/60000 [==>...........................] - ETA: 1:34 - loss: 0.5374 - categorical_accuracy: 0.8272
 7360/60000 [==>...........................] - ETA: 1:34 - loss: 0.5357 - categorical_accuracy: 0.8277
 7392/60000 [==>...........................] - ETA: 1:34 - loss: 0.5341 - categorical_accuracy: 0.8282
 7424/60000 [==>...........................] - ETA: 1:34 - loss: 0.5319 - categorical_accuracy: 0.8289
 7456/60000 [==>...........................] - ETA: 1:33 - loss: 0.5313 - categorical_accuracy: 0.8293
 7488/60000 [==>...........................] - ETA: 1:33 - loss: 0.5304 - categorical_accuracy: 0.8297
 7520/60000 [==>...........................] - ETA: 1:33 - loss: 0.5295 - categorical_accuracy: 0.8302
 7552/60000 [==>...........................] - ETA: 1:33 - loss: 0.5295 - categorical_accuracy: 0.8302
 7584/60000 [==>...........................] - ETA: 1:33 - loss: 0.5292 - categorical_accuracy: 0.8304
 7616/60000 [==>...........................] - ETA: 1:33 - loss: 0.5290 - categorical_accuracy: 0.8306
 7648/60000 [==>...........................] - ETA: 1:33 - loss: 0.5282 - categorical_accuracy: 0.8305
 7680/60000 [==>...........................] - ETA: 1:33 - loss: 0.5284 - categorical_accuracy: 0.8306
 7712/60000 [==>...........................] - ETA: 1:33 - loss: 0.5273 - categorical_accuracy: 0.8309
 7744/60000 [==>...........................] - ETA: 1:33 - loss: 0.5262 - categorical_accuracy: 0.8312
 7776/60000 [==>...........................] - ETA: 1:33 - loss: 0.5247 - categorical_accuracy: 0.8317
 7808/60000 [==>...........................] - ETA: 1:33 - loss: 0.5238 - categorical_accuracy: 0.8321
 7840/60000 [==>...........................] - ETA: 1:33 - loss: 0.5220 - categorical_accuracy: 0.8325
 7872/60000 [==>...........................] - ETA: 1:33 - loss: 0.5225 - categorical_accuracy: 0.8328
 7904/60000 [==>...........................] - ETA: 1:33 - loss: 0.5215 - categorical_accuracy: 0.8330
 7936/60000 [==>...........................] - ETA: 1:33 - loss: 0.5201 - categorical_accuracy: 0.8334
 7968/60000 [==>...........................] - ETA: 1:33 - loss: 0.5194 - categorical_accuracy: 0.8340
 8000/60000 [===>..........................] - ETA: 1:32 - loss: 0.5187 - categorical_accuracy: 0.8341
 8032/60000 [===>..........................] - ETA: 1:32 - loss: 0.5173 - categorical_accuracy: 0.8345
 8064/60000 [===>..........................] - ETA: 1:32 - loss: 0.5173 - categorical_accuracy: 0.8346
 8096/60000 [===>..........................] - ETA: 1:32 - loss: 0.5162 - categorical_accuracy: 0.8350
 8128/60000 [===>..........................] - ETA: 1:32 - loss: 0.5146 - categorical_accuracy: 0.8355
 8160/60000 [===>..........................] - ETA: 1:32 - loss: 0.5130 - categorical_accuracy: 0.8362
 8192/60000 [===>..........................] - ETA: 1:32 - loss: 0.5125 - categorical_accuracy: 0.8361
 8224/60000 [===>..........................] - ETA: 1:32 - loss: 0.5108 - categorical_accuracy: 0.8366
 8256/60000 [===>..........................] - ETA: 1:32 - loss: 0.5101 - categorical_accuracy: 0.8367
 8288/60000 [===>..........................] - ETA: 1:32 - loss: 0.5086 - categorical_accuracy: 0.8371
 8320/60000 [===>..........................] - ETA: 1:32 - loss: 0.5071 - categorical_accuracy: 0.8376
 8352/60000 [===>..........................] - ETA: 1:32 - loss: 0.5058 - categorical_accuracy: 0.8380
 8384/60000 [===>..........................] - ETA: 1:32 - loss: 0.5048 - categorical_accuracy: 0.8383
 8416/60000 [===>..........................] - ETA: 1:32 - loss: 0.5037 - categorical_accuracy: 0.8384
 8448/60000 [===>..........................] - ETA: 1:32 - loss: 0.5022 - categorical_accuracy: 0.8389
 8480/60000 [===>..........................] - ETA: 1:32 - loss: 0.5009 - categorical_accuracy: 0.8394
 8512/60000 [===>..........................] - ETA: 1:32 - loss: 0.5003 - categorical_accuracy: 0.8394
 8544/60000 [===>..........................] - ETA: 1:32 - loss: 0.4994 - categorical_accuracy: 0.8398
 8576/60000 [===>..........................] - ETA: 1:32 - loss: 0.4987 - categorical_accuracy: 0.8399
 8608/60000 [===>..........................] - ETA: 1:32 - loss: 0.4974 - categorical_accuracy: 0.8404
 8640/60000 [===>..........................] - ETA: 1:32 - loss: 0.4969 - categorical_accuracy: 0.8405
 8672/60000 [===>..........................] - ETA: 1:31 - loss: 0.4956 - categorical_accuracy: 0.8410
 8704/60000 [===>..........................] - ETA: 1:31 - loss: 0.4952 - categorical_accuracy: 0.8413
 8736/60000 [===>..........................] - ETA: 1:31 - loss: 0.4947 - categorical_accuracy: 0.8415
 8768/60000 [===>..........................] - ETA: 1:31 - loss: 0.4940 - categorical_accuracy: 0.8417
 8800/60000 [===>..........................] - ETA: 1:31 - loss: 0.4940 - categorical_accuracy: 0.8418
 8832/60000 [===>..........................] - ETA: 1:31 - loss: 0.4932 - categorical_accuracy: 0.8419
 8864/60000 [===>..........................] - ETA: 1:31 - loss: 0.4918 - categorical_accuracy: 0.8425
 8896/60000 [===>..........................] - ETA: 1:31 - loss: 0.4911 - categorical_accuracy: 0.8427
 8928/60000 [===>..........................] - ETA: 1:31 - loss: 0.4907 - categorical_accuracy: 0.8427
 8960/60000 [===>..........................] - ETA: 1:31 - loss: 0.4893 - categorical_accuracy: 0.8433
 8992/60000 [===>..........................] - ETA: 1:31 - loss: 0.4884 - categorical_accuracy: 0.8438
 9024/60000 [===>..........................] - ETA: 1:31 - loss: 0.4872 - categorical_accuracy: 0.8442
 9056/60000 [===>..........................] - ETA: 1:31 - loss: 0.4866 - categorical_accuracy: 0.8443
 9088/60000 [===>..........................] - ETA: 1:30 - loss: 0.4861 - categorical_accuracy: 0.8446
 9120/60000 [===>..........................] - ETA: 1:30 - loss: 0.4849 - categorical_accuracy: 0.8450
 9152/60000 [===>..........................] - ETA: 1:30 - loss: 0.4833 - categorical_accuracy: 0.8455
 9184/60000 [===>..........................] - ETA: 1:30 - loss: 0.4820 - categorical_accuracy: 0.8460
 9216/60000 [===>..........................] - ETA: 1:30 - loss: 0.4805 - categorical_accuracy: 0.8466
 9248/60000 [===>..........................] - ETA: 1:30 - loss: 0.4795 - categorical_accuracy: 0.8469
 9280/60000 [===>..........................] - ETA: 1:30 - loss: 0.4796 - categorical_accuracy: 0.8472
 9312/60000 [===>..........................] - ETA: 1:30 - loss: 0.4783 - categorical_accuracy: 0.8476
 9344/60000 [===>..........................] - ETA: 1:30 - loss: 0.4772 - categorical_accuracy: 0.8478
 9376/60000 [===>..........................] - ETA: 1:30 - loss: 0.4762 - categorical_accuracy: 0.8480
 9408/60000 [===>..........................] - ETA: 1:30 - loss: 0.4751 - categorical_accuracy: 0.8484
 9440/60000 [===>..........................] - ETA: 1:30 - loss: 0.4747 - categorical_accuracy: 0.8485
 9472/60000 [===>..........................] - ETA: 1:30 - loss: 0.4738 - categorical_accuracy: 0.8488
 9504/60000 [===>..........................] - ETA: 1:30 - loss: 0.4728 - categorical_accuracy: 0.8492
 9536/60000 [===>..........................] - ETA: 1:30 - loss: 0.4722 - categorical_accuracy: 0.8493
 9568/60000 [===>..........................] - ETA: 1:29 - loss: 0.4720 - categorical_accuracy: 0.8496
 9600/60000 [===>..........................] - ETA: 1:29 - loss: 0.4710 - categorical_accuracy: 0.8500
 9632/60000 [===>..........................] - ETA: 1:29 - loss: 0.4710 - categorical_accuracy: 0.8501
 9664/60000 [===>..........................] - ETA: 1:29 - loss: 0.4705 - categorical_accuracy: 0.8503
 9696/60000 [===>..........................] - ETA: 1:29 - loss: 0.4696 - categorical_accuracy: 0.8507
 9728/60000 [===>..........................] - ETA: 1:29 - loss: 0.4692 - categorical_accuracy: 0.8506
 9760/60000 [===>..........................] - ETA: 1:29 - loss: 0.4684 - categorical_accuracy: 0.8509
 9792/60000 [===>..........................] - ETA: 1:29 - loss: 0.4670 - categorical_accuracy: 0.8514
 9824/60000 [===>..........................] - ETA: 1:29 - loss: 0.4668 - categorical_accuracy: 0.8514
 9856/60000 [===>..........................] - ETA: 1:29 - loss: 0.4660 - categorical_accuracy: 0.8516
 9888/60000 [===>..........................] - ETA: 1:29 - loss: 0.4647 - categorical_accuracy: 0.8520
 9920/60000 [===>..........................] - ETA: 1:29 - loss: 0.4644 - categorical_accuracy: 0.8521
 9952/60000 [===>..........................] - ETA: 1:29 - loss: 0.4634 - categorical_accuracy: 0.8525
 9984/60000 [===>..........................] - ETA: 1:29 - loss: 0.4620 - categorical_accuracy: 0.8530
10016/60000 [====>.........................] - ETA: 1:29 - loss: 0.4611 - categorical_accuracy: 0.8532
10048/60000 [====>.........................] - ETA: 1:28 - loss: 0.4605 - categorical_accuracy: 0.8532
10080/60000 [====>.........................] - ETA: 1:28 - loss: 0.4594 - categorical_accuracy: 0.8535
10112/60000 [====>.........................] - ETA: 1:28 - loss: 0.4586 - categorical_accuracy: 0.8536
10144/60000 [====>.........................] - ETA: 1:28 - loss: 0.4576 - categorical_accuracy: 0.8540
10176/60000 [====>.........................] - ETA: 1:28 - loss: 0.4568 - categorical_accuracy: 0.8543
10208/60000 [====>.........................] - ETA: 1:28 - loss: 0.4562 - categorical_accuracy: 0.8546
10240/60000 [====>.........................] - ETA: 1:28 - loss: 0.4554 - categorical_accuracy: 0.8549
10272/60000 [====>.........................] - ETA: 1:28 - loss: 0.4551 - categorical_accuracy: 0.8548
10304/60000 [====>.........................] - ETA: 1:28 - loss: 0.4540 - categorical_accuracy: 0.8552
10336/60000 [====>.........................] - ETA: 1:28 - loss: 0.4537 - categorical_accuracy: 0.8555
10368/60000 [====>.........................] - ETA: 1:28 - loss: 0.4535 - categorical_accuracy: 0.8556
10400/60000 [====>.........................] - ETA: 1:28 - loss: 0.4531 - categorical_accuracy: 0.8558
10432/60000 [====>.........................] - ETA: 1:28 - loss: 0.4521 - categorical_accuracy: 0.8561
10464/60000 [====>.........................] - ETA: 1:28 - loss: 0.4517 - categorical_accuracy: 0.8562
10496/60000 [====>.........................] - ETA: 1:27 - loss: 0.4507 - categorical_accuracy: 0.8565
10528/60000 [====>.........................] - ETA: 1:27 - loss: 0.4498 - categorical_accuracy: 0.8568
10560/60000 [====>.........................] - ETA: 1:27 - loss: 0.4491 - categorical_accuracy: 0.8569
10592/60000 [====>.........................] - ETA: 1:27 - loss: 0.4484 - categorical_accuracy: 0.8572
10624/60000 [====>.........................] - ETA: 1:27 - loss: 0.4473 - categorical_accuracy: 0.8575
10656/60000 [====>.........................] - ETA: 1:27 - loss: 0.4463 - categorical_accuracy: 0.8578
10688/60000 [====>.........................] - ETA: 1:27 - loss: 0.4456 - categorical_accuracy: 0.8582
10720/60000 [====>.........................] - ETA: 1:27 - loss: 0.4445 - categorical_accuracy: 0.8584
10752/60000 [====>.........................] - ETA: 1:27 - loss: 0.4437 - categorical_accuracy: 0.8587
10784/60000 [====>.........................] - ETA: 1:27 - loss: 0.4436 - categorical_accuracy: 0.8588
10816/60000 [====>.........................] - ETA: 1:27 - loss: 0.4427 - categorical_accuracy: 0.8591
10848/60000 [====>.........................] - ETA: 1:27 - loss: 0.4418 - categorical_accuracy: 0.8593
10880/60000 [====>.........................] - ETA: 1:27 - loss: 0.4407 - categorical_accuracy: 0.8597
10912/60000 [====>.........................] - ETA: 1:27 - loss: 0.4398 - categorical_accuracy: 0.8599
10944/60000 [====>.........................] - ETA: 1:27 - loss: 0.4396 - categorical_accuracy: 0.8600
10976/60000 [====>.........................] - ETA: 1:27 - loss: 0.4389 - categorical_accuracy: 0.8602
11008/60000 [====>.........................] - ETA: 1:26 - loss: 0.4378 - categorical_accuracy: 0.8606
11040/60000 [====>.........................] - ETA: 1:26 - loss: 0.4370 - categorical_accuracy: 0.8609
11072/60000 [====>.........................] - ETA: 1:26 - loss: 0.4366 - categorical_accuracy: 0.8612
11104/60000 [====>.........................] - ETA: 1:26 - loss: 0.4362 - categorical_accuracy: 0.8612
11136/60000 [====>.........................] - ETA: 1:26 - loss: 0.4351 - categorical_accuracy: 0.8615
11168/60000 [====>.........................] - ETA: 1:26 - loss: 0.4341 - categorical_accuracy: 0.8619
11232/60000 [====>.........................] - ETA: 1:26 - loss: 0.4332 - categorical_accuracy: 0.8622
11264/60000 [====>.........................] - ETA: 1:26 - loss: 0.4331 - categorical_accuracy: 0.8623
11296/60000 [====>.........................] - ETA: 1:26 - loss: 0.4324 - categorical_accuracy: 0.8626
11328/60000 [====>.........................] - ETA: 1:26 - loss: 0.4318 - categorical_accuracy: 0.8628
11360/60000 [====>.........................] - ETA: 1:26 - loss: 0.4308 - categorical_accuracy: 0.8631
11392/60000 [====>.........................] - ETA: 1:26 - loss: 0.4304 - categorical_accuracy: 0.8632
11456/60000 [====>.........................] - ETA: 1:25 - loss: 0.4294 - categorical_accuracy: 0.8637
11488/60000 [====>.........................] - ETA: 1:25 - loss: 0.4285 - categorical_accuracy: 0.8639
11520/60000 [====>.........................] - ETA: 1:25 - loss: 0.4279 - categorical_accuracy: 0.8641
11552/60000 [====>.........................] - ETA: 1:25 - loss: 0.4271 - categorical_accuracy: 0.8644
11584/60000 [====>.........................] - ETA: 1:25 - loss: 0.4268 - categorical_accuracy: 0.8643
11616/60000 [====>.........................] - ETA: 1:25 - loss: 0.4259 - categorical_accuracy: 0.8647
11648/60000 [====>.........................] - ETA: 1:25 - loss: 0.4249 - categorical_accuracy: 0.8650
11680/60000 [====>.........................] - ETA: 1:25 - loss: 0.4239 - categorical_accuracy: 0.8653
11712/60000 [====>.........................] - ETA: 1:25 - loss: 0.4235 - categorical_accuracy: 0.8654
11744/60000 [====>.........................] - ETA: 1:25 - loss: 0.4228 - categorical_accuracy: 0.8655
11776/60000 [====>.........................] - ETA: 1:25 - loss: 0.4222 - categorical_accuracy: 0.8658
11808/60000 [====>.........................] - ETA: 1:25 - loss: 0.4217 - categorical_accuracy: 0.8659
11840/60000 [====>.........................] - ETA: 1:25 - loss: 0.4213 - categorical_accuracy: 0.8660
11872/60000 [====>.........................] - ETA: 1:25 - loss: 0.4203 - categorical_accuracy: 0.8664
11904/60000 [====>.........................] - ETA: 1:25 - loss: 0.4195 - categorical_accuracy: 0.8667
11936/60000 [====>.........................] - ETA: 1:24 - loss: 0.4189 - categorical_accuracy: 0.8670
11968/60000 [====>.........................] - ETA: 1:24 - loss: 0.4183 - categorical_accuracy: 0.8671
12032/60000 [=====>........................] - ETA: 1:24 - loss: 0.4174 - categorical_accuracy: 0.8674
12064/60000 [=====>........................] - ETA: 1:24 - loss: 0.4168 - categorical_accuracy: 0.8675
12096/60000 [=====>........................] - ETA: 1:24 - loss: 0.4158 - categorical_accuracy: 0.8678
12128/60000 [=====>........................] - ETA: 1:24 - loss: 0.4154 - categorical_accuracy: 0.8678
12160/60000 [=====>........................] - ETA: 1:24 - loss: 0.4146 - categorical_accuracy: 0.8681
12192/60000 [=====>........................] - ETA: 1:24 - loss: 0.4137 - categorical_accuracy: 0.8684
12224/60000 [=====>........................] - ETA: 1:24 - loss: 0.4129 - categorical_accuracy: 0.8686
12256/60000 [=====>........................] - ETA: 1:24 - loss: 0.4120 - categorical_accuracy: 0.8689
12288/60000 [=====>........................] - ETA: 1:24 - loss: 0.4111 - categorical_accuracy: 0.8691
12320/60000 [=====>........................] - ETA: 1:24 - loss: 0.4111 - categorical_accuracy: 0.8692
12352/60000 [=====>........................] - ETA: 1:24 - loss: 0.4103 - categorical_accuracy: 0.8695
12384/60000 [=====>........................] - ETA: 1:23 - loss: 0.4097 - categorical_accuracy: 0.8697
12416/60000 [=====>........................] - ETA: 1:23 - loss: 0.4093 - categorical_accuracy: 0.8698
12448/60000 [=====>........................] - ETA: 1:23 - loss: 0.4089 - categorical_accuracy: 0.8700
12480/60000 [=====>........................] - ETA: 1:23 - loss: 0.4080 - categorical_accuracy: 0.8704
12512/60000 [=====>........................] - ETA: 1:23 - loss: 0.4075 - categorical_accuracy: 0.8704
12544/60000 [=====>........................] - ETA: 1:23 - loss: 0.4065 - categorical_accuracy: 0.8708
12576/60000 [=====>........................] - ETA: 1:23 - loss: 0.4062 - categorical_accuracy: 0.8709
12608/60000 [=====>........................] - ETA: 1:23 - loss: 0.4053 - categorical_accuracy: 0.8712
12640/60000 [=====>........................] - ETA: 1:23 - loss: 0.4045 - categorical_accuracy: 0.8715
12672/60000 [=====>........................] - ETA: 1:23 - loss: 0.4036 - categorical_accuracy: 0.8718
12704/60000 [=====>........................] - ETA: 1:23 - loss: 0.4029 - categorical_accuracy: 0.8720
12736/60000 [=====>........................] - ETA: 1:23 - loss: 0.4022 - categorical_accuracy: 0.8723
12768/60000 [=====>........................] - ETA: 1:23 - loss: 0.4013 - categorical_accuracy: 0.8726
12800/60000 [=====>........................] - ETA: 1:23 - loss: 0.4010 - categorical_accuracy: 0.8727
12832/60000 [=====>........................] - ETA: 1:23 - loss: 0.4006 - categorical_accuracy: 0.8728
12864/60000 [=====>........................] - ETA: 1:23 - loss: 0.3999 - categorical_accuracy: 0.8731
12896/60000 [=====>........................] - ETA: 1:23 - loss: 0.3991 - categorical_accuracy: 0.8734
12928/60000 [=====>........................] - ETA: 1:22 - loss: 0.3990 - categorical_accuracy: 0.8735
12960/60000 [=====>........................] - ETA: 1:22 - loss: 0.3981 - categorical_accuracy: 0.8738
12992/60000 [=====>........................] - ETA: 1:22 - loss: 0.3973 - categorical_accuracy: 0.8742
13024/60000 [=====>........................] - ETA: 1:22 - loss: 0.3969 - categorical_accuracy: 0.8743
13056/60000 [=====>........................] - ETA: 1:22 - loss: 0.3967 - categorical_accuracy: 0.8745
13088/60000 [=====>........................] - ETA: 1:22 - loss: 0.3959 - categorical_accuracy: 0.8748
13120/60000 [=====>........................] - ETA: 1:22 - loss: 0.3951 - categorical_accuracy: 0.8750
13152/60000 [=====>........................] - ETA: 1:22 - loss: 0.3947 - categorical_accuracy: 0.8752
13184/60000 [=====>........................] - ETA: 1:22 - loss: 0.3945 - categorical_accuracy: 0.8751
13216/60000 [=====>........................] - ETA: 1:22 - loss: 0.3938 - categorical_accuracy: 0.8753
13248/60000 [=====>........................] - ETA: 1:22 - loss: 0.3939 - categorical_accuracy: 0.8752
13280/60000 [=====>........................] - ETA: 1:22 - loss: 0.3932 - categorical_accuracy: 0.8754
13312/60000 [=====>........................] - ETA: 1:22 - loss: 0.3926 - categorical_accuracy: 0.8755
13344/60000 [=====>........................] - ETA: 1:22 - loss: 0.3926 - categorical_accuracy: 0.8756
13376/60000 [=====>........................] - ETA: 1:22 - loss: 0.3922 - categorical_accuracy: 0.8757
13408/60000 [=====>........................] - ETA: 1:22 - loss: 0.3917 - categorical_accuracy: 0.8759
13440/60000 [=====>........................] - ETA: 1:22 - loss: 0.3913 - categorical_accuracy: 0.8761
13472/60000 [=====>........................] - ETA: 1:22 - loss: 0.3908 - categorical_accuracy: 0.8763
13504/60000 [=====>........................] - ETA: 1:21 - loss: 0.3900 - categorical_accuracy: 0.8765
13536/60000 [=====>........................] - ETA: 1:21 - loss: 0.3895 - categorical_accuracy: 0.8766
13568/60000 [=====>........................] - ETA: 1:21 - loss: 0.3887 - categorical_accuracy: 0.8768
13600/60000 [=====>........................] - ETA: 1:21 - loss: 0.3880 - categorical_accuracy: 0.8771
13632/60000 [=====>........................] - ETA: 1:21 - loss: 0.3881 - categorical_accuracy: 0.8771
13664/60000 [=====>........................] - ETA: 1:21 - loss: 0.3878 - categorical_accuracy: 0.8771
13696/60000 [=====>........................] - ETA: 1:21 - loss: 0.3872 - categorical_accuracy: 0.8773
13728/60000 [=====>........................] - ETA: 1:21 - loss: 0.3874 - categorical_accuracy: 0.8774
13760/60000 [=====>........................] - ETA: 1:21 - loss: 0.3866 - categorical_accuracy: 0.8777
13792/60000 [=====>........................] - ETA: 1:21 - loss: 0.3862 - categorical_accuracy: 0.8778
13824/60000 [=====>........................] - ETA: 1:21 - loss: 0.3855 - categorical_accuracy: 0.8780
13856/60000 [=====>........................] - ETA: 1:21 - loss: 0.3849 - categorical_accuracy: 0.8782
13888/60000 [=====>........................] - ETA: 1:21 - loss: 0.3845 - categorical_accuracy: 0.8783
13920/60000 [=====>........................] - ETA: 1:21 - loss: 0.3840 - categorical_accuracy: 0.8784
13952/60000 [=====>........................] - ETA: 1:21 - loss: 0.3838 - categorical_accuracy: 0.8785
13984/60000 [=====>........................] - ETA: 1:21 - loss: 0.3836 - categorical_accuracy: 0.8786
14016/60000 [======>.......................] - ETA: 1:21 - loss: 0.3833 - categorical_accuracy: 0.8787
14048/60000 [======>.......................] - ETA: 1:20 - loss: 0.3829 - categorical_accuracy: 0.8789
14080/60000 [======>.......................] - ETA: 1:20 - loss: 0.3822 - categorical_accuracy: 0.8791
14112/60000 [======>.......................] - ETA: 1:20 - loss: 0.3816 - categorical_accuracy: 0.8793
14144/60000 [======>.......................] - ETA: 1:20 - loss: 0.3813 - categorical_accuracy: 0.8795
14176/60000 [======>.......................] - ETA: 1:20 - loss: 0.3810 - categorical_accuracy: 0.8796
14208/60000 [======>.......................] - ETA: 1:20 - loss: 0.3803 - categorical_accuracy: 0.8798
14240/60000 [======>.......................] - ETA: 1:20 - loss: 0.3803 - categorical_accuracy: 0.8798
14272/60000 [======>.......................] - ETA: 1:20 - loss: 0.3802 - categorical_accuracy: 0.8799
14304/60000 [======>.......................] - ETA: 1:20 - loss: 0.3796 - categorical_accuracy: 0.8801
14336/60000 [======>.......................] - ETA: 1:20 - loss: 0.3790 - categorical_accuracy: 0.8803
14368/60000 [======>.......................] - ETA: 1:20 - loss: 0.3787 - categorical_accuracy: 0.8804
14400/60000 [======>.......................] - ETA: 1:20 - loss: 0.3782 - categorical_accuracy: 0.8804
14432/60000 [======>.......................] - ETA: 1:20 - loss: 0.3775 - categorical_accuracy: 0.8807
14464/60000 [======>.......................] - ETA: 1:20 - loss: 0.3772 - categorical_accuracy: 0.8807
14496/60000 [======>.......................] - ETA: 1:20 - loss: 0.3766 - categorical_accuracy: 0.8809
14528/60000 [======>.......................] - ETA: 1:19 - loss: 0.3761 - categorical_accuracy: 0.8811
14560/60000 [======>.......................] - ETA: 1:19 - loss: 0.3761 - categorical_accuracy: 0.8811
14592/60000 [======>.......................] - ETA: 1:19 - loss: 0.3753 - categorical_accuracy: 0.8814
14624/60000 [======>.......................] - ETA: 1:19 - loss: 0.3749 - categorical_accuracy: 0.8815
14656/60000 [======>.......................] - ETA: 1:19 - loss: 0.3749 - categorical_accuracy: 0.8816
14688/60000 [======>.......................] - ETA: 1:19 - loss: 0.3744 - categorical_accuracy: 0.8817
14720/60000 [======>.......................] - ETA: 1:19 - loss: 0.3739 - categorical_accuracy: 0.8819
14752/60000 [======>.......................] - ETA: 1:19 - loss: 0.3735 - categorical_accuracy: 0.8820
14784/60000 [======>.......................] - ETA: 1:19 - loss: 0.3732 - categorical_accuracy: 0.8821
14816/60000 [======>.......................] - ETA: 1:19 - loss: 0.3725 - categorical_accuracy: 0.8824
14848/60000 [======>.......................] - ETA: 1:19 - loss: 0.3718 - categorical_accuracy: 0.8826
14880/60000 [======>.......................] - ETA: 1:19 - loss: 0.3713 - categorical_accuracy: 0.8827
14912/60000 [======>.......................] - ETA: 1:19 - loss: 0.3708 - categorical_accuracy: 0.8828
14944/60000 [======>.......................] - ETA: 1:19 - loss: 0.3704 - categorical_accuracy: 0.8830
14976/60000 [======>.......................] - ETA: 1:19 - loss: 0.3702 - categorical_accuracy: 0.8831
15008/60000 [======>.......................] - ETA: 1:19 - loss: 0.3702 - categorical_accuracy: 0.8832
15040/60000 [======>.......................] - ETA: 1:18 - loss: 0.3697 - categorical_accuracy: 0.8833
15072/60000 [======>.......................] - ETA: 1:18 - loss: 0.3690 - categorical_accuracy: 0.8836
15104/60000 [======>.......................] - ETA: 1:18 - loss: 0.3687 - categorical_accuracy: 0.8837
15136/60000 [======>.......................] - ETA: 1:18 - loss: 0.3683 - categorical_accuracy: 0.8838
15168/60000 [======>.......................] - ETA: 1:18 - loss: 0.3685 - categorical_accuracy: 0.8838
15200/60000 [======>.......................] - ETA: 1:18 - loss: 0.3678 - categorical_accuracy: 0.8841
15232/60000 [======>.......................] - ETA: 1:18 - loss: 0.3677 - categorical_accuracy: 0.8841
15264/60000 [======>.......................] - ETA: 1:18 - loss: 0.3672 - categorical_accuracy: 0.8842
15296/60000 [======>.......................] - ETA: 1:18 - loss: 0.3671 - categorical_accuracy: 0.8843
15328/60000 [======>.......................] - ETA: 1:18 - loss: 0.3675 - categorical_accuracy: 0.8843
15360/60000 [======>.......................] - ETA: 1:18 - loss: 0.3669 - categorical_accuracy: 0.8846
15392/60000 [======>.......................] - ETA: 1:18 - loss: 0.3669 - categorical_accuracy: 0.8846
15424/60000 [======>.......................] - ETA: 1:18 - loss: 0.3667 - categorical_accuracy: 0.8846
15456/60000 [======>.......................] - ETA: 1:18 - loss: 0.3663 - categorical_accuracy: 0.8847
15488/60000 [======>.......................] - ETA: 1:18 - loss: 0.3658 - categorical_accuracy: 0.8849
15520/60000 [======>.......................] - ETA: 1:18 - loss: 0.3655 - categorical_accuracy: 0.8850
15552/60000 [======>.......................] - ETA: 1:18 - loss: 0.3648 - categorical_accuracy: 0.8852
15584/60000 [======>.......................] - ETA: 1:17 - loss: 0.3645 - categorical_accuracy: 0.8854
15616/60000 [======>.......................] - ETA: 1:17 - loss: 0.3640 - categorical_accuracy: 0.8856
15648/60000 [======>.......................] - ETA: 1:17 - loss: 0.3633 - categorical_accuracy: 0.8858
15680/60000 [======>.......................] - ETA: 1:17 - loss: 0.3631 - categorical_accuracy: 0.8859
15712/60000 [======>.......................] - ETA: 1:17 - loss: 0.3631 - categorical_accuracy: 0.8860
15744/60000 [======>.......................] - ETA: 1:17 - loss: 0.3636 - categorical_accuracy: 0.8859
15776/60000 [======>.......................] - ETA: 1:17 - loss: 0.3632 - categorical_accuracy: 0.8860
15808/60000 [======>.......................] - ETA: 1:17 - loss: 0.3626 - categorical_accuracy: 0.8862
15840/60000 [======>.......................] - ETA: 1:17 - loss: 0.3621 - categorical_accuracy: 0.8864
15872/60000 [======>.......................] - ETA: 1:17 - loss: 0.3616 - categorical_accuracy: 0.8865
15904/60000 [======>.......................] - ETA: 1:17 - loss: 0.3610 - categorical_accuracy: 0.8867
15936/60000 [======>.......................] - ETA: 1:17 - loss: 0.3604 - categorical_accuracy: 0.8869
15968/60000 [======>.......................] - ETA: 1:17 - loss: 0.3604 - categorical_accuracy: 0.8870
16000/60000 [=======>......................] - ETA: 1:17 - loss: 0.3599 - categorical_accuracy: 0.8871
16032/60000 [=======>......................] - ETA: 1:17 - loss: 0.3594 - categorical_accuracy: 0.8872
16064/60000 [=======>......................] - ETA: 1:17 - loss: 0.3589 - categorical_accuracy: 0.8874
16096/60000 [=======>......................] - ETA: 1:16 - loss: 0.3586 - categorical_accuracy: 0.8875
16128/60000 [=======>......................] - ETA: 1:16 - loss: 0.3581 - categorical_accuracy: 0.8876
16160/60000 [=======>......................] - ETA: 1:16 - loss: 0.3578 - categorical_accuracy: 0.8878
16192/60000 [=======>......................] - ETA: 1:16 - loss: 0.3572 - categorical_accuracy: 0.8880
16256/60000 [=======>......................] - ETA: 1:16 - loss: 0.3562 - categorical_accuracy: 0.8883
16288/60000 [=======>......................] - ETA: 1:16 - loss: 0.3557 - categorical_accuracy: 0.8885
16320/60000 [=======>......................] - ETA: 1:16 - loss: 0.3559 - categorical_accuracy: 0.8886
16352/60000 [=======>......................] - ETA: 1:16 - loss: 0.3554 - categorical_accuracy: 0.8888
16384/60000 [=======>......................] - ETA: 1:16 - loss: 0.3549 - categorical_accuracy: 0.8889
16416/60000 [=======>......................] - ETA: 1:16 - loss: 0.3543 - categorical_accuracy: 0.8891
16448/60000 [=======>......................] - ETA: 1:16 - loss: 0.3539 - categorical_accuracy: 0.8892
16480/60000 [=======>......................] - ETA: 1:16 - loss: 0.3535 - categorical_accuracy: 0.8893
16512/60000 [=======>......................] - ETA: 1:16 - loss: 0.3530 - categorical_accuracy: 0.8894
16544/60000 [=======>......................] - ETA: 1:16 - loss: 0.3527 - categorical_accuracy: 0.8895
16576/60000 [=======>......................] - ETA: 1:16 - loss: 0.3524 - categorical_accuracy: 0.8896
16608/60000 [=======>......................] - ETA: 1:15 - loss: 0.3522 - categorical_accuracy: 0.8896
16640/60000 [=======>......................] - ETA: 1:15 - loss: 0.3516 - categorical_accuracy: 0.8898
16672/60000 [=======>......................] - ETA: 1:15 - loss: 0.3511 - categorical_accuracy: 0.8899
16736/60000 [=======>......................] - ETA: 1:15 - loss: 0.3510 - categorical_accuracy: 0.8901
16768/60000 [=======>......................] - ETA: 1:15 - loss: 0.3505 - categorical_accuracy: 0.8902
16800/60000 [=======>......................] - ETA: 1:15 - loss: 0.3499 - categorical_accuracy: 0.8904
16832/60000 [=======>......................] - ETA: 1:15 - loss: 0.3494 - categorical_accuracy: 0.8906
16864/60000 [=======>......................] - ETA: 1:15 - loss: 0.3490 - categorical_accuracy: 0.8907
16896/60000 [=======>......................] - ETA: 1:15 - loss: 0.3493 - categorical_accuracy: 0.8907
16928/60000 [=======>......................] - ETA: 1:15 - loss: 0.3487 - categorical_accuracy: 0.8909
16960/60000 [=======>......................] - ETA: 1:15 - loss: 0.3486 - categorical_accuracy: 0.8910
16992/60000 [=======>......................] - ETA: 1:15 - loss: 0.3482 - categorical_accuracy: 0.8911
17024/60000 [=======>......................] - ETA: 1:15 - loss: 0.3479 - categorical_accuracy: 0.8911
17056/60000 [=======>......................] - ETA: 1:15 - loss: 0.3474 - categorical_accuracy: 0.8913
17088/60000 [=======>......................] - ETA: 1:15 - loss: 0.3470 - categorical_accuracy: 0.8914
17120/60000 [=======>......................] - ETA: 1:14 - loss: 0.3465 - categorical_accuracy: 0.8916
17152/60000 [=======>......................] - ETA: 1:14 - loss: 0.3464 - categorical_accuracy: 0.8916
17216/60000 [=======>......................] - ETA: 1:14 - loss: 0.3454 - categorical_accuracy: 0.8919
17248/60000 [=======>......................] - ETA: 1:14 - loss: 0.3448 - categorical_accuracy: 0.8921
17280/60000 [=======>......................] - ETA: 1:14 - loss: 0.3444 - categorical_accuracy: 0.8922
17312/60000 [=======>......................] - ETA: 1:14 - loss: 0.3439 - categorical_accuracy: 0.8923
17344/60000 [=======>......................] - ETA: 1:14 - loss: 0.3437 - categorical_accuracy: 0.8924
17376/60000 [=======>......................] - ETA: 1:14 - loss: 0.3436 - categorical_accuracy: 0.8925
17408/60000 [=======>......................] - ETA: 1:14 - loss: 0.3435 - categorical_accuracy: 0.8926
17440/60000 [=======>......................] - ETA: 1:14 - loss: 0.3433 - categorical_accuracy: 0.8926
17472/60000 [=======>......................] - ETA: 1:14 - loss: 0.3429 - categorical_accuracy: 0.8927
17504/60000 [=======>......................] - ETA: 1:14 - loss: 0.3428 - categorical_accuracy: 0.8928
17536/60000 [=======>......................] - ETA: 1:14 - loss: 0.3422 - categorical_accuracy: 0.8930
17568/60000 [=======>......................] - ETA: 1:14 - loss: 0.3417 - categorical_accuracy: 0.8932
17600/60000 [=======>......................] - ETA: 1:14 - loss: 0.3413 - categorical_accuracy: 0.8932
17632/60000 [=======>......................] - ETA: 1:14 - loss: 0.3408 - categorical_accuracy: 0.8934
17664/60000 [=======>......................] - ETA: 1:13 - loss: 0.3410 - categorical_accuracy: 0.8935
17696/60000 [=======>......................] - ETA: 1:13 - loss: 0.3406 - categorical_accuracy: 0.8936
17728/60000 [=======>......................] - ETA: 1:13 - loss: 0.3402 - categorical_accuracy: 0.8936
17760/60000 [=======>......................] - ETA: 1:13 - loss: 0.3398 - categorical_accuracy: 0.8938
17792/60000 [=======>......................] - ETA: 1:13 - loss: 0.3393 - categorical_accuracy: 0.8939
17824/60000 [=======>......................] - ETA: 1:13 - loss: 0.3391 - categorical_accuracy: 0.8939
17856/60000 [=======>......................] - ETA: 1:13 - loss: 0.3386 - categorical_accuracy: 0.8940
17888/60000 [=======>......................] - ETA: 1:13 - loss: 0.3385 - categorical_accuracy: 0.8941
17920/60000 [=======>......................] - ETA: 1:13 - loss: 0.3379 - categorical_accuracy: 0.8943
17952/60000 [=======>......................] - ETA: 1:13 - loss: 0.3377 - categorical_accuracy: 0.8943
17984/60000 [=======>......................] - ETA: 1:13 - loss: 0.3372 - categorical_accuracy: 0.8944
18016/60000 [========>.....................] - ETA: 1:13 - loss: 0.3368 - categorical_accuracy: 0.8945
18048/60000 [========>.....................] - ETA: 1:13 - loss: 0.3366 - categorical_accuracy: 0.8946
18080/60000 [========>.....................] - ETA: 1:13 - loss: 0.3361 - categorical_accuracy: 0.8946
18112/60000 [========>.....................] - ETA: 1:13 - loss: 0.3356 - categorical_accuracy: 0.8948
18144/60000 [========>.....................] - ETA: 1:12 - loss: 0.3353 - categorical_accuracy: 0.8949
18176/60000 [========>.....................] - ETA: 1:12 - loss: 0.3352 - categorical_accuracy: 0.8950
18208/60000 [========>.....................] - ETA: 1:12 - loss: 0.3346 - categorical_accuracy: 0.8952
18240/60000 [========>.....................] - ETA: 1:12 - loss: 0.3341 - categorical_accuracy: 0.8954
18272/60000 [========>.....................] - ETA: 1:12 - loss: 0.3336 - categorical_accuracy: 0.8956
18304/60000 [========>.....................] - ETA: 1:12 - loss: 0.3333 - categorical_accuracy: 0.8956
18336/60000 [========>.....................] - ETA: 1:12 - loss: 0.3330 - categorical_accuracy: 0.8957
18368/60000 [========>.....................] - ETA: 1:12 - loss: 0.3326 - categorical_accuracy: 0.8959
18400/60000 [========>.....................] - ETA: 1:12 - loss: 0.3322 - categorical_accuracy: 0.8960
18432/60000 [========>.....................] - ETA: 1:12 - loss: 0.3320 - categorical_accuracy: 0.8961
18464/60000 [========>.....................] - ETA: 1:12 - loss: 0.3316 - categorical_accuracy: 0.8962
18496/60000 [========>.....................] - ETA: 1:12 - loss: 0.3312 - categorical_accuracy: 0.8963
18528/60000 [========>.....................] - ETA: 1:12 - loss: 0.3310 - categorical_accuracy: 0.8963
18592/60000 [========>.....................] - ETA: 1:12 - loss: 0.3302 - categorical_accuracy: 0.8966
18624/60000 [========>.....................] - ETA: 1:12 - loss: 0.3300 - categorical_accuracy: 0.8966
18656/60000 [========>.....................] - ETA: 1:12 - loss: 0.3296 - categorical_accuracy: 0.8968
18688/60000 [========>.....................] - ETA: 1:11 - loss: 0.3291 - categorical_accuracy: 0.8969
18720/60000 [========>.....................] - ETA: 1:11 - loss: 0.3286 - categorical_accuracy: 0.8971
18752/60000 [========>.....................] - ETA: 1:11 - loss: 0.3285 - categorical_accuracy: 0.8971
18784/60000 [========>.....................] - ETA: 1:11 - loss: 0.3282 - categorical_accuracy: 0.8972
18816/60000 [========>.....................] - ETA: 1:11 - loss: 0.3278 - categorical_accuracy: 0.8973
18848/60000 [========>.....................] - ETA: 1:11 - loss: 0.3279 - categorical_accuracy: 0.8973
18880/60000 [========>.....................] - ETA: 1:11 - loss: 0.3277 - categorical_accuracy: 0.8973
18912/60000 [========>.....................] - ETA: 1:11 - loss: 0.3273 - categorical_accuracy: 0.8974
18944/60000 [========>.....................] - ETA: 1:11 - loss: 0.3269 - categorical_accuracy: 0.8975
18976/60000 [========>.....................] - ETA: 1:11 - loss: 0.3265 - categorical_accuracy: 0.8976
19008/60000 [========>.....................] - ETA: 1:11 - loss: 0.3261 - categorical_accuracy: 0.8977
19040/60000 [========>.....................] - ETA: 1:11 - loss: 0.3257 - categorical_accuracy: 0.8978
19072/60000 [========>.....................] - ETA: 1:11 - loss: 0.3252 - categorical_accuracy: 0.8980
19104/60000 [========>.....................] - ETA: 1:11 - loss: 0.3247 - categorical_accuracy: 0.8981
19136/60000 [========>.....................] - ETA: 1:11 - loss: 0.3245 - categorical_accuracy: 0.8982
19168/60000 [========>.....................] - ETA: 1:11 - loss: 0.3240 - categorical_accuracy: 0.8983
19232/60000 [========>.....................] - ETA: 1:11 - loss: 0.3240 - categorical_accuracy: 0.8984
19296/60000 [========>.....................] - ETA: 1:10 - loss: 0.3237 - categorical_accuracy: 0.8985
19328/60000 [========>.....................] - ETA: 1:10 - loss: 0.3232 - categorical_accuracy: 0.8986
19360/60000 [========>.....................] - ETA: 1:10 - loss: 0.3230 - categorical_accuracy: 0.8988
19392/60000 [========>.....................] - ETA: 1:10 - loss: 0.3226 - categorical_accuracy: 0.8989
19424/60000 [========>.....................] - ETA: 1:10 - loss: 0.3226 - categorical_accuracy: 0.8989
19456/60000 [========>.....................] - ETA: 1:10 - loss: 0.3222 - categorical_accuracy: 0.8990
19488/60000 [========>.....................] - ETA: 1:10 - loss: 0.3219 - categorical_accuracy: 0.8991
19520/60000 [========>.....................] - ETA: 1:10 - loss: 0.3219 - categorical_accuracy: 0.8992
19552/60000 [========>.....................] - ETA: 1:10 - loss: 0.3218 - categorical_accuracy: 0.8992
19616/60000 [========>.....................] - ETA: 1:10 - loss: 0.3225 - categorical_accuracy: 0.8993
19648/60000 [========>.....................] - ETA: 1:10 - loss: 0.3226 - categorical_accuracy: 0.8993
19680/60000 [========>.....................] - ETA: 1:10 - loss: 0.3222 - categorical_accuracy: 0.8994
19712/60000 [========>.....................] - ETA: 1:10 - loss: 0.3219 - categorical_accuracy: 0.8996
19744/60000 [========>.....................] - ETA: 1:10 - loss: 0.3218 - categorical_accuracy: 0.8997
19776/60000 [========>.....................] - ETA: 1:10 - loss: 0.3217 - categorical_accuracy: 0.8997
19808/60000 [========>.....................] - ETA: 1:09 - loss: 0.3215 - categorical_accuracy: 0.8997
19840/60000 [========>.....................] - ETA: 1:09 - loss: 0.3212 - categorical_accuracy: 0.8998
19872/60000 [========>.....................] - ETA: 1:09 - loss: 0.3208 - categorical_accuracy: 0.9000
19904/60000 [========>.....................] - ETA: 1:09 - loss: 0.3206 - categorical_accuracy: 0.9000
19936/60000 [========>.....................] - ETA: 1:09 - loss: 0.3208 - categorical_accuracy: 0.9000
19968/60000 [========>.....................] - ETA: 1:09 - loss: 0.3204 - categorical_accuracy: 0.9001
20000/60000 [=========>....................] - ETA: 1:09 - loss: 0.3201 - categorical_accuracy: 0.9002
20032/60000 [=========>....................] - ETA: 1:09 - loss: 0.3199 - categorical_accuracy: 0.9002
20064/60000 [=========>....................] - ETA: 1:09 - loss: 0.3196 - categorical_accuracy: 0.9003
20096/60000 [=========>....................] - ETA: 1:09 - loss: 0.3197 - categorical_accuracy: 0.9004
20128/60000 [=========>....................] - ETA: 1:09 - loss: 0.3195 - categorical_accuracy: 0.9004
20160/60000 [=========>....................] - ETA: 1:09 - loss: 0.3192 - categorical_accuracy: 0.9005
20192/60000 [=========>....................] - ETA: 1:09 - loss: 0.3187 - categorical_accuracy: 0.9007
20224/60000 [=========>....................] - ETA: 1:09 - loss: 0.3185 - categorical_accuracy: 0.9008
20256/60000 [=========>....................] - ETA: 1:09 - loss: 0.3181 - categorical_accuracy: 0.9009
20288/60000 [=========>....................] - ETA: 1:09 - loss: 0.3178 - categorical_accuracy: 0.9010
20320/60000 [=========>....................] - ETA: 1:08 - loss: 0.3174 - categorical_accuracy: 0.9011
20352/60000 [=========>....................] - ETA: 1:08 - loss: 0.3170 - categorical_accuracy: 0.9013
20384/60000 [=========>....................] - ETA: 1:08 - loss: 0.3167 - categorical_accuracy: 0.9014
20416/60000 [=========>....................] - ETA: 1:08 - loss: 0.3164 - categorical_accuracy: 0.9014
20448/60000 [=========>....................] - ETA: 1:08 - loss: 0.3160 - categorical_accuracy: 0.9016
20480/60000 [=========>....................] - ETA: 1:08 - loss: 0.3162 - categorical_accuracy: 0.9016
20512/60000 [=========>....................] - ETA: 1:08 - loss: 0.3158 - categorical_accuracy: 0.9018
20544/60000 [=========>....................] - ETA: 1:08 - loss: 0.3154 - categorical_accuracy: 0.9018
20576/60000 [=========>....................] - ETA: 1:08 - loss: 0.3150 - categorical_accuracy: 0.9020
20608/60000 [=========>....................] - ETA: 1:08 - loss: 0.3145 - categorical_accuracy: 0.9021
20640/60000 [=========>....................] - ETA: 1:08 - loss: 0.3141 - categorical_accuracy: 0.9023
20672/60000 [=========>....................] - ETA: 1:08 - loss: 0.3139 - categorical_accuracy: 0.9023
20704/60000 [=========>....................] - ETA: 1:08 - loss: 0.3136 - categorical_accuracy: 0.9025
20736/60000 [=========>....................] - ETA: 1:08 - loss: 0.3132 - categorical_accuracy: 0.9026
20768/60000 [=========>....................] - ETA: 1:08 - loss: 0.3132 - categorical_accuracy: 0.9026
20800/60000 [=========>....................] - ETA: 1:08 - loss: 0.3129 - categorical_accuracy: 0.9027
20832/60000 [=========>....................] - ETA: 1:08 - loss: 0.3125 - categorical_accuracy: 0.9028
20864/60000 [=========>....................] - ETA: 1:07 - loss: 0.3121 - categorical_accuracy: 0.9029
20896/60000 [=========>....................] - ETA: 1:07 - loss: 0.3117 - categorical_accuracy: 0.9030
20928/60000 [=========>....................] - ETA: 1:07 - loss: 0.3116 - categorical_accuracy: 0.9030
20960/60000 [=========>....................] - ETA: 1:07 - loss: 0.3116 - categorical_accuracy: 0.9031
20992/60000 [=========>....................] - ETA: 1:07 - loss: 0.3114 - categorical_accuracy: 0.9032
21024/60000 [=========>....................] - ETA: 1:07 - loss: 0.3112 - categorical_accuracy: 0.9033
21056/60000 [=========>....................] - ETA: 1:07 - loss: 0.3112 - categorical_accuracy: 0.9033
21088/60000 [=========>....................] - ETA: 1:07 - loss: 0.3111 - categorical_accuracy: 0.9033
21120/60000 [=========>....................] - ETA: 1:07 - loss: 0.3107 - categorical_accuracy: 0.9034
21152/60000 [=========>....................] - ETA: 1:07 - loss: 0.3106 - categorical_accuracy: 0.9034
21184/60000 [=========>....................] - ETA: 1:07 - loss: 0.3102 - categorical_accuracy: 0.9035
21248/60000 [=========>....................] - ETA: 1:07 - loss: 0.3097 - categorical_accuracy: 0.9037
21280/60000 [=========>....................] - ETA: 1:07 - loss: 0.3097 - categorical_accuracy: 0.9037
21312/60000 [=========>....................] - ETA: 1:07 - loss: 0.3093 - categorical_accuracy: 0.9039
21344/60000 [=========>....................] - ETA: 1:07 - loss: 0.3091 - categorical_accuracy: 0.9040
21376/60000 [=========>....................] - ETA: 1:07 - loss: 0.3088 - categorical_accuracy: 0.9041
21408/60000 [=========>....................] - ETA: 1:06 - loss: 0.3084 - categorical_accuracy: 0.9042
21440/60000 [=========>....................] - ETA: 1:06 - loss: 0.3082 - categorical_accuracy: 0.9042
21472/60000 [=========>....................] - ETA: 1:06 - loss: 0.3077 - categorical_accuracy: 0.9043
21504/60000 [=========>....................] - ETA: 1:06 - loss: 0.3073 - categorical_accuracy: 0.9045
21536/60000 [=========>....................] - ETA: 1:06 - loss: 0.3069 - categorical_accuracy: 0.9046
21568/60000 [=========>....................] - ETA: 1:06 - loss: 0.3068 - categorical_accuracy: 0.9047
21600/60000 [=========>....................] - ETA: 1:06 - loss: 0.3064 - categorical_accuracy: 0.9048
21632/60000 [=========>....................] - ETA: 1:06 - loss: 0.3062 - categorical_accuracy: 0.9049
21664/60000 [=========>....................] - ETA: 1:06 - loss: 0.3058 - categorical_accuracy: 0.9050
21696/60000 [=========>....................] - ETA: 1:06 - loss: 0.3054 - categorical_accuracy: 0.9051
21728/60000 [=========>....................] - ETA: 1:06 - loss: 0.3051 - categorical_accuracy: 0.9052
21760/60000 [=========>....................] - ETA: 1:06 - loss: 0.3047 - categorical_accuracy: 0.9053
21792/60000 [=========>....................] - ETA: 1:06 - loss: 0.3044 - categorical_accuracy: 0.9054
21824/60000 [=========>....................] - ETA: 1:06 - loss: 0.3045 - categorical_accuracy: 0.9054
21856/60000 [=========>....................] - ETA: 1:06 - loss: 0.3044 - categorical_accuracy: 0.9054
21888/60000 [=========>....................] - ETA: 1:06 - loss: 0.3041 - categorical_accuracy: 0.9055
21920/60000 [=========>....................] - ETA: 1:06 - loss: 0.3039 - categorical_accuracy: 0.9055
21952/60000 [=========>....................] - ETA: 1:06 - loss: 0.3038 - categorical_accuracy: 0.9055
21984/60000 [=========>....................] - ETA: 1:05 - loss: 0.3034 - categorical_accuracy: 0.9056
22016/60000 [==========>...................] - ETA: 1:05 - loss: 0.3033 - categorical_accuracy: 0.9057
22048/60000 [==========>...................] - ETA: 1:05 - loss: 0.3030 - categorical_accuracy: 0.9058
22080/60000 [==========>...................] - ETA: 1:05 - loss: 0.3028 - categorical_accuracy: 0.9058
22112/60000 [==========>...................] - ETA: 1:05 - loss: 0.3026 - categorical_accuracy: 0.9059
22144/60000 [==========>...................] - ETA: 1:05 - loss: 0.3024 - categorical_accuracy: 0.9060
22176/60000 [==========>...................] - ETA: 1:05 - loss: 0.3020 - categorical_accuracy: 0.9061
22208/60000 [==========>...................] - ETA: 1:05 - loss: 0.3018 - categorical_accuracy: 0.9062
22240/60000 [==========>...................] - ETA: 1:05 - loss: 0.3017 - categorical_accuracy: 0.9062
22272/60000 [==========>...................] - ETA: 1:05 - loss: 0.3013 - categorical_accuracy: 0.9064
22304/60000 [==========>...................] - ETA: 1:05 - loss: 0.3009 - categorical_accuracy: 0.9065
22336/60000 [==========>...................] - ETA: 1:05 - loss: 0.3008 - categorical_accuracy: 0.9065
22368/60000 [==========>...................] - ETA: 1:05 - loss: 0.3004 - categorical_accuracy: 0.9067
22400/60000 [==========>...................] - ETA: 1:05 - loss: 0.3001 - categorical_accuracy: 0.9068
22432/60000 [==========>...................] - ETA: 1:05 - loss: 0.2997 - categorical_accuracy: 0.9069
22464/60000 [==========>...................] - ETA: 1:05 - loss: 0.2994 - categorical_accuracy: 0.9070
22496/60000 [==========>...................] - ETA: 1:05 - loss: 0.2992 - categorical_accuracy: 0.9071
22528/60000 [==========>...................] - ETA: 1:04 - loss: 0.2989 - categorical_accuracy: 0.9072
22560/60000 [==========>...................] - ETA: 1:04 - loss: 0.2988 - categorical_accuracy: 0.9072
22592/60000 [==========>...................] - ETA: 1:04 - loss: 0.2985 - categorical_accuracy: 0.9073
22624/60000 [==========>...................] - ETA: 1:04 - loss: 0.2984 - categorical_accuracy: 0.9073
22656/60000 [==========>...................] - ETA: 1:04 - loss: 0.2980 - categorical_accuracy: 0.9074
22688/60000 [==========>...................] - ETA: 1:04 - loss: 0.2980 - categorical_accuracy: 0.9075
22720/60000 [==========>...................] - ETA: 1:04 - loss: 0.2978 - categorical_accuracy: 0.9075
22784/60000 [==========>...................] - ETA: 1:04 - loss: 0.2980 - categorical_accuracy: 0.9076
22816/60000 [==========>...................] - ETA: 1:04 - loss: 0.2976 - categorical_accuracy: 0.9077
22848/60000 [==========>...................] - ETA: 1:04 - loss: 0.2973 - categorical_accuracy: 0.9078
22880/60000 [==========>...................] - ETA: 1:04 - loss: 0.2971 - categorical_accuracy: 0.9078
22912/60000 [==========>...................] - ETA: 1:04 - loss: 0.2969 - categorical_accuracy: 0.9079
22944/60000 [==========>...................] - ETA: 1:04 - loss: 0.2966 - categorical_accuracy: 0.9079
22976/60000 [==========>...................] - ETA: 1:04 - loss: 0.2964 - categorical_accuracy: 0.9080
23008/60000 [==========>...................] - ETA: 1:04 - loss: 0.2965 - categorical_accuracy: 0.9080
23040/60000 [==========>...................] - ETA: 1:04 - loss: 0.2961 - categorical_accuracy: 0.9082
23072/60000 [==========>...................] - ETA: 1:03 - loss: 0.2958 - categorical_accuracy: 0.9082
23104/60000 [==========>...................] - ETA: 1:03 - loss: 0.2955 - categorical_accuracy: 0.9083
23136/60000 [==========>...................] - ETA: 1:03 - loss: 0.2955 - categorical_accuracy: 0.9084
23168/60000 [==========>...................] - ETA: 1:03 - loss: 0.2954 - categorical_accuracy: 0.9085
23200/60000 [==========>...................] - ETA: 1:03 - loss: 0.2951 - categorical_accuracy: 0.9085
23232/60000 [==========>...................] - ETA: 1:03 - loss: 0.2951 - categorical_accuracy: 0.9085
23264/60000 [==========>...................] - ETA: 1:03 - loss: 0.2948 - categorical_accuracy: 0.9086
23296/60000 [==========>...................] - ETA: 1:03 - loss: 0.2945 - categorical_accuracy: 0.9087
23328/60000 [==========>...................] - ETA: 1:03 - loss: 0.2941 - categorical_accuracy: 0.9089
23360/60000 [==========>...................] - ETA: 1:03 - loss: 0.2943 - categorical_accuracy: 0.9089
23392/60000 [==========>...................] - ETA: 1:03 - loss: 0.2939 - categorical_accuracy: 0.9090
23424/60000 [==========>...................] - ETA: 1:03 - loss: 0.2938 - categorical_accuracy: 0.9090
23456/60000 [==========>...................] - ETA: 1:03 - loss: 0.2935 - categorical_accuracy: 0.9091
23488/60000 [==========>...................] - ETA: 1:03 - loss: 0.2932 - categorical_accuracy: 0.9092
23520/60000 [==========>...................] - ETA: 1:03 - loss: 0.2929 - categorical_accuracy: 0.9093
23552/60000 [==========>...................] - ETA: 1:03 - loss: 0.2928 - categorical_accuracy: 0.9093
23584/60000 [==========>...................] - ETA: 1:03 - loss: 0.2925 - categorical_accuracy: 0.9094
23616/60000 [==========>...................] - ETA: 1:02 - loss: 0.2923 - categorical_accuracy: 0.9095
23648/60000 [==========>...................] - ETA: 1:02 - loss: 0.2919 - categorical_accuracy: 0.9096
23680/60000 [==========>...................] - ETA: 1:02 - loss: 0.2916 - categorical_accuracy: 0.9097
23712/60000 [==========>...................] - ETA: 1:02 - loss: 0.2913 - categorical_accuracy: 0.9098
23744/60000 [==========>...................] - ETA: 1:02 - loss: 0.2912 - categorical_accuracy: 0.9099
23776/60000 [==========>...................] - ETA: 1:02 - loss: 0.2909 - categorical_accuracy: 0.9100
23808/60000 [==========>...................] - ETA: 1:02 - loss: 0.2911 - categorical_accuracy: 0.9099
23840/60000 [==========>...................] - ETA: 1:02 - loss: 0.2909 - categorical_accuracy: 0.9100
23872/60000 [==========>...................] - ETA: 1:02 - loss: 0.2908 - categorical_accuracy: 0.9100
23904/60000 [==========>...................] - ETA: 1:02 - loss: 0.2906 - categorical_accuracy: 0.9101
23936/60000 [==========>...................] - ETA: 1:02 - loss: 0.2905 - categorical_accuracy: 0.9101
23968/60000 [==========>...................] - ETA: 1:02 - loss: 0.2901 - categorical_accuracy: 0.9103
24000/60000 [===========>..................] - ETA: 1:02 - loss: 0.2901 - categorical_accuracy: 0.9103
24032/60000 [===========>..................] - ETA: 1:02 - loss: 0.2897 - categorical_accuracy: 0.9104
24064/60000 [===========>..................] - ETA: 1:02 - loss: 0.2894 - categorical_accuracy: 0.9105
24096/60000 [===========>..................] - ETA: 1:02 - loss: 0.2893 - categorical_accuracy: 0.9106
24128/60000 [===========>..................] - ETA: 1:02 - loss: 0.2889 - categorical_accuracy: 0.9107
24160/60000 [===========>..................] - ETA: 1:02 - loss: 0.2887 - categorical_accuracy: 0.9107
24192/60000 [===========>..................] - ETA: 1:02 - loss: 0.2885 - categorical_accuracy: 0.9108
24224/60000 [===========>..................] - ETA: 1:01 - loss: 0.2887 - categorical_accuracy: 0.9107
24256/60000 [===========>..................] - ETA: 1:01 - loss: 0.2885 - categorical_accuracy: 0.9108
24288/60000 [===========>..................] - ETA: 1:01 - loss: 0.2882 - categorical_accuracy: 0.9109
24320/60000 [===========>..................] - ETA: 1:01 - loss: 0.2879 - categorical_accuracy: 0.9109
24352/60000 [===========>..................] - ETA: 1:01 - loss: 0.2876 - categorical_accuracy: 0.9111
24384/60000 [===========>..................] - ETA: 1:01 - loss: 0.2872 - categorical_accuracy: 0.9112
24416/60000 [===========>..................] - ETA: 1:01 - loss: 0.2869 - categorical_accuracy: 0.9113
24448/60000 [===========>..................] - ETA: 1:01 - loss: 0.2869 - categorical_accuracy: 0.9113
24480/60000 [===========>..................] - ETA: 1:01 - loss: 0.2868 - categorical_accuracy: 0.9113
24512/60000 [===========>..................] - ETA: 1:01 - loss: 0.2865 - categorical_accuracy: 0.9113
24544/60000 [===========>..................] - ETA: 1:01 - loss: 0.2864 - categorical_accuracy: 0.9114
24576/60000 [===========>..................] - ETA: 1:01 - loss: 0.2861 - categorical_accuracy: 0.9115
24608/60000 [===========>..................] - ETA: 1:01 - loss: 0.2858 - categorical_accuracy: 0.9115
24640/60000 [===========>..................] - ETA: 1:01 - loss: 0.2855 - categorical_accuracy: 0.9116
24672/60000 [===========>..................] - ETA: 1:01 - loss: 0.2854 - categorical_accuracy: 0.9116
24704/60000 [===========>..................] - ETA: 1:01 - loss: 0.2854 - categorical_accuracy: 0.9116
24736/60000 [===========>..................] - ETA: 1:01 - loss: 0.2851 - categorical_accuracy: 0.9117
24768/60000 [===========>..................] - ETA: 1:01 - loss: 0.2852 - categorical_accuracy: 0.9117
24800/60000 [===========>..................] - ETA: 1:00 - loss: 0.2850 - categorical_accuracy: 0.9118
24832/60000 [===========>..................] - ETA: 1:00 - loss: 0.2848 - categorical_accuracy: 0.9118
24864/60000 [===========>..................] - ETA: 1:00 - loss: 0.2846 - categorical_accuracy: 0.9119
24896/60000 [===========>..................] - ETA: 1:00 - loss: 0.2843 - categorical_accuracy: 0.9120
24928/60000 [===========>..................] - ETA: 1:00 - loss: 0.2840 - categorical_accuracy: 0.9121
24960/60000 [===========>..................] - ETA: 1:00 - loss: 0.2843 - categorical_accuracy: 0.9121
24992/60000 [===========>..................] - ETA: 1:00 - loss: 0.2840 - categorical_accuracy: 0.9121
25024/60000 [===========>..................] - ETA: 1:00 - loss: 0.2838 - categorical_accuracy: 0.9122
25056/60000 [===========>..................] - ETA: 1:00 - loss: 0.2838 - categorical_accuracy: 0.9122
25088/60000 [===========>..................] - ETA: 1:00 - loss: 0.2838 - categorical_accuracy: 0.9122
25120/60000 [===========>..................] - ETA: 1:00 - loss: 0.2835 - categorical_accuracy: 0.9123
25152/60000 [===========>..................] - ETA: 1:00 - loss: 0.2833 - categorical_accuracy: 0.9124
25184/60000 [===========>..................] - ETA: 1:00 - loss: 0.2831 - categorical_accuracy: 0.9124
25216/60000 [===========>..................] - ETA: 1:00 - loss: 0.2829 - categorical_accuracy: 0.9125
25248/60000 [===========>..................] - ETA: 1:00 - loss: 0.2826 - categorical_accuracy: 0.9126
25280/60000 [===========>..................] - ETA: 1:00 - loss: 0.2825 - categorical_accuracy: 0.9126
25312/60000 [===========>..................] - ETA: 1:00 - loss: 0.2825 - categorical_accuracy: 0.9127
25344/60000 [===========>..................] - ETA: 1:00 - loss: 0.2824 - categorical_accuracy: 0.9127
25376/60000 [===========>..................] - ETA: 59s - loss: 0.2821 - categorical_accuracy: 0.9128 
25408/60000 [===========>..................] - ETA: 59s - loss: 0.2819 - categorical_accuracy: 0.9128
25440/60000 [===========>..................] - ETA: 59s - loss: 0.2818 - categorical_accuracy: 0.9128
25472/60000 [===========>..................] - ETA: 59s - loss: 0.2815 - categorical_accuracy: 0.9129
25504/60000 [===========>..................] - ETA: 59s - loss: 0.2813 - categorical_accuracy: 0.9130
25536/60000 [===========>..................] - ETA: 59s - loss: 0.2810 - categorical_accuracy: 0.9131
25568/60000 [===========>..................] - ETA: 59s - loss: 0.2807 - categorical_accuracy: 0.9132
25600/60000 [===========>..................] - ETA: 59s - loss: 0.2803 - categorical_accuracy: 0.9133
25632/60000 [===========>..................] - ETA: 59s - loss: 0.2802 - categorical_accuracy: 0.9134
25664/60000 [===========>..................] - ETA: 59s - loss: 0.2799 - categorical_accuracy: 0.9135
25696/60000 [===========>..................] - ETA: 59s - loss: 0.2800 - categorical_accuracy: 0.9136
25728/60000 [===========>..................] - ETA: 59s - loss: 0.2800 - categorical_accuracy: 0.9136
25760/60000 [===========>..................] - ETA: 59s - loss: 0.2798 - categorical_accuracy: 0.9137
25792/60000 [===========>..................] - ETA: 59s - loss: 0.2795 - categorical_accuracy: 0.9138
25824/60000 [===========>..................] - ETA: 59s - loss: 0.2794 - categorical_accuracy: 0.9138
25856/60000 [===========>..................] - ETA: 59s - loss: 0.2793 - categorical_accuracy: 0.9138
25888/60000 [===========>..................] - ETA: 59s - loss: 0.2790 - categorical_accuracy: 0.9139
25920/60000 [===========>..................] - ETA: 59s - loss: 0.2787 - categorical_accuracy: 0.9140
25952/60000 [===========>..................] - ETA: 58s - loss: 0.2787 - categorical_accuracy: 0.9140
25984/60000 [===========>..................] - ETA: 58s - loss: 0.2784 - categorical_accuracy: 0.9141
26016/60000 [============>.................] - ETA: 58s - loss: 0.2782 - categorical_accuracy: 0.9142
26048/60000 [============>.................] - ETA: 58s - loss: 0.2779 - categorical_accuracy: 0.9142
26080/60000 [============>.................] - ETA: 58s - loss: 0.2778 - categorical_accuracy: 0.9143
26112/60000 [============>.................] - ETA: 58s - loss: 0.2776 - categorical_accuracy: 0.9144
26144/60000 [============>.................] - ETA: 58s - loss: 0.2775 - categorical_accuracy: 0.9144
26176/60000 [============>.................] - ETA: 58s - loss: 0.2775 - categorical_accuracy: 0.9145
26208/60000 [============>.................] - ETA: 58s - loss: 0.2774 - categorical_accuracy: 0.9145
26240/60000 [============>.................] - ETA: 58s - loss: 0.2771 - categorical_accuracy: 0.9146
26272/60000 [============>.................] - ETA: 58s - loss: 0.2768 - categorical_accuracy: 0.9147
26304/60000 [============>.................] - ETA: 58s - loss: 0.2768 - categorical_accuracy: 0.9147
26336/60000 [============>.................] - ETA: 58s - loss: 0.2765 - categorical_accuracy: 0.9148
26368/60000 [============>.................] - ETA: 58s - loss: 0.2765 - categorical_accuracy: 0.9148
26400/60000 [============>.................] - ETA: 58s - loss: 0.2764 - categorical_accuracy: 0.9148
26432/60000 [============>.................] - ETA: 58s - loss: 0.2761 - categorical_accuracy: 0.9149
26464/60000 [============>.................] - ETA: 58s - loss: 0.2763 - categorical_accuracy: 0.9149
26496/60000 [============>.................] - ETA: 58s - loss: 0.2761 - categorical_accuracy: 0.9149
26528/60000 [============>.................] - ETA: 57s - loss: 0.2760 - categorical_accuracy: 0.9149
26592/60000 [============>.................] - ETA: 57s - loss: 0.2760 - categorical_accuracy: 0.9149
26624/60000 [============>.................] - ETA: 57s - loss: 0.2759 - categorical_accuracy: 0.9150
26656/60000 [============>.................] - ETA: 57s - loss: 0.2756 - categorical_accuracy: 0.9150
26688/60000 [============>.................] - ETA: 57s - loss: 0.2753 - categorical_accuracy: 0.9151
26720/60000 [============>.................] - ETA: 57s - loss: 0.2751 - categorical_accuracy: 0.9152
26752/60000 [============>.................] - ETA: 57s - loss: 0.2748 - categorical_accuracy: 0.9153
26784/60000 [============>.................] - ETA: 57s - loss: 0.2745 - categorical_accuracy: 0.9154
26816/60000 [============>.................] - ETA: 57s - loss: 0.2744 - categorical_accuracy: 0.9155
26848/60000 [============>.................] - ETA: 57s - loss: 0.2741 - categorical_accuracy: 0.9155
26880/60000 [============>.................] - ETA: 57s - loss: 0.2740 - categorical_accuracy: 0.9156
26944/60000 [============>.................] - ETA: 57s - loss: 0.2736 - categorical_accuracy: 0.9157
26976/60000 [============>.................] - ETA: 57s - loss: 0.2733 - categorical_accuracy: 0.9158
27008/60000 [============>.................] - ETA: 57s - loss: 0.2730 - categorical_accuracy: 0.9158
27040/60000 [============>.................] - ETA: 57s - loss: 0.2728 - categorical_accuracy: 0.9159
27072/60000 [============>.................] - ETA: 56s - loss: 0.2727 - categorical_accuracy: 0.9159
27104/60000 [============>.................] - ETA: 56s - loss: 0.2724 - categorical_accuracy: 0.9160
27136/60000 [============>.................] - ETA: 56s - loss: 0.2722 - categorical_accuracy: 0.9161
27168/60000 [============>.................] - ETA: 56s - loss: 0.2720 - categorical_accuracy: 0.9162
27200/60000 [============>.................] - ETA: 56s - loss: 0.2718 - categorical_accuracy: 0.9162
27232/60000 [============>.................] - ETA: 56s - loss: 0.2718 - categorical_accuracy: 0.9162
27264/60000 [============>.................] - ETA: 56s - loss: 0.2715 - categorical_accuracy: 0.9163
27296/60000 [============>.................] - ETA: 56s - loss: 0.2713 - categorical_accuracy: 0.9164
27328/60000 [============>.................] - ETA: 56s - loss: 0.2710 - categorical_accuracy: 0.9165
27360/60000 [============>.................] - ETA: 56s - loss: 0.2708 - categorical_accuracy: 0.9165
27392/60000 [============>.................] - ETA: 56s - loss: 0.2706 - categorical_accuracy: 0.9166
27424/60000 [============>.................] - ETA: 56s - loss: 0.2703 - categorical_accuracy: 0.9167
27456/60000 [============>.................] - ETA: 56s - loss: 0.2700 - categorical_accuracy: 0.9167
27488/60000 [============>.................] - ETA: 56s - loss: 0.2699 - categorical_accuracy: 0.9168
27520/60000 [============>.................] - ETA: 56s - loss: 0.2696 - categorical_accuracy: 0.9168
27552/60000 [============>.................] - ETA: 56s - loss: 0.2695 - categorical_accuracy: 0.9168
27616/60000 [============>.................] - ETA: 55s - loss: 0.2692 - categorical_accuracy: 0.9170
27648/60000 [============>.................] - ETA: 55s - loss: 0.2689 - categorical_accuracy: 0.9171
27680/60000 [============>.................] - ETA: 55s - loss: 0.2687 - categorical_accuracy: 0.9171
27712/60000 [============>.................] - ETA: 55s - loss: 0.2685 - categorical_accuracy: 0.9172
27744/60000 [============>.................] - ETA: 55s - loss: 0.2684 - categorical_accuracy: 0.9172
27776/60000 [============>.................] - ETA: 55s - loss: 0.2683 - categorical_accuracy: 0.9172
27808/60000 [============>.................] - ETA: 55s - loss: 0.2681 - categorical_accuracy: 0.9173
27840/60000 [============>.................] - ETA: 55s - loss: 0.2680 - categorical_accuracy: 0.9173
27872/60000 [============>.................] - ETA: 55s - loss: 0.2678 - categorical_accuracy: 0.9174
27904/60000 [============>.................] - ETA: 55s - loss: 0.2675 - categorical_accuracy: 0.9175
27936/60000 [============>.................] - ETA: 55s - loss: 0.2673 - categorical_accuracy: 0.9176
27968/60000 [============>.................] - ETA: 55s - loss: 0.2670 - categorical_accuracy: 0.9177
28000/60000 [=============>................] - ETA: 55s - loss: 0.2667 - categorical_accuracy: 0.9178
28032/60000 [=============>................] - ETA: 55s - loss: 0.2668 - categorical_accuracy: 0.9178
28064/60000 [=============>................] - ETA: 55s - loss: 0.2665 - categorical_accuracy: 0.9179
28096/60000 [=============>................] - ETA: 55s - loss: 0.2664 - categorical_accuracy: 0.9179
28128/60000 [=============>................] - ETA: 55s - loss: 0.2662 - categorical_accuracy: 0.9180
28160/60000 [=============>................] - ETA: 55s - loss: 0.2661 - categorical_accuracy: 0.9180
28192/60000 [=============>................] - ETA: 54s - loss: 0.2662 - categorical_accuracy: 0.9179
28224/60000 [=============>................] - ETA: 54s - loss: 0.2660 - categorical_accuracy: 0.9180
28256/60000 [=============>................] - ETA: 54s - loss: 0.2658 - categorical_accuracy: 0.9180
28288/60000 [=============>................] - ETA: 54s - loss: 0.2656 - categorical_accuracy: 0.9181
28320/60000 [=============>................] - ETA: 54s - loss: 0.2654 - categorical_accuracy: 0.9181
28352/60000 [=============>................] - ETA: 54s - loss: 0.2651 - categorical_accuracy: 0.9182
28384/60000 [=============>................] - ETA: 54s - loss: 0.2650 - categorical_accuracy: 0.9183
28416/60000 [=============>................] - ETA: 54s - loss: 0.2647 - categorical_accuracy: 0.9184
28448/60000 [=============>................] - ETA: 54s - loss: 0.2645 - categorical_accuracy: 0.9184
28512/60000 [=============>................] - ETA: 54s - loss: 0.2641 - categorical_accuracy: 0.9185
28544/60000 [=============>................] - ETA: 54s - loss: 0.2638 - categorical_accuracy: 0.9186
28576/60000 [=============>................] - ETA: 54s - loss: 0.2638 - categorical_accuracy: 0.9186
28608/60000 [=============>................] - ETA: 54s - loss: 0.2635 - categorical_accuracy: 0.9187
28640/60000 [=============>................] - ETA: 54s - loss: 0.2633 - categorical_accuracy: 0.9187
28672/60000 [=============>................] - ETA: 54s - loss: 0.2631 - categorical_accuracy: 0.9188
28704/60000 [=============>................] - ETA: 54s - loss: 0.2630 - categorical_accuracy: 0.9188
28768/60000 [=============>................] - ETA: 53s - loss: 0.2626 - categorical_accuracy: 0.9189
28800/60000 [=============>................] - ETA: 53s - loss: 0.2625 - categorical_accuracy: 0.9190
28832/60000 [=============>................] - ETA: 53s - loss: 0.2624 - categorical_accuracy: 0.9190
28896/60000 [=============>................] - ETA: 53s - loss: 0.2620 - categorical_accuracy: 0.9191
28928/60000 [=============>................] - ETA: 53s - loss: 0.2619 - categorical_accuracy: 0.9191
28960/60000 [=============>................] - ETA: 53s - loss: 0.2616 - categorical_accuracy: 0.9191
28992/60000 [=============>................] - ETA: 53s - loss: 0.2618 - categorical_accuracy: 0.9192
29024/60000 [=============>................] - ETA: 53s - loss: 0.2616 - categorical_accuracy: 0.9192
29056/60000 [=============>................] - ETA: 53s - loss: 0.2614 - categorical_accuracy: 0.9193
29088/60000 [=============>................] - ETA: 53s - loss: 0.2612 - categorical_accuracy: 0.9193
29152/60000 [=============>................] - ETA: 53s - loss: 0.2609 - categorical_accuracy: 0.9195
29184/60000 [=============>................] - ETA: 53s - loss: 0.2607 - categorical_accuracy: 0.9195
29216/60000 [=============>................] - ETA: 53s - loss: 0.2604 - categorical_accuracy: 0.9196
29248/60000 [=============>................] - ETA: 53s - loss: 0.2602 - categorical_accuracy: 0.9197
29280/60000 [=============>................] - ETA: 52s - loss: 0.2600 - categorical_accuracy: 0.9198
29312/60000 [=============>................] - ETA: 52s - loss: 0.2598 - categorical_accuracy: 0.9199
29376/60000 [=============>................] - ETA: 52s - loss: 0.2596 - categorical_accuracy: 0.9199
29408/60000 [=============>................] - ETA: 52s - loss: 0.2594 - categorical_accuracy: 0.9200
29440/60000 [=============>................] - ETA: 52s - loss: 0.2592 - categorical_accuracy: 0.9200
29472/60000 [=============>................] - ETA: 52s - loss: 0.2592 - categorical_accuracy: 0.9200
29536/60000 [=============>................] - ETA: 52s - loss: 0.2586 - categorical_accuracy: 0.9202
29568/60000 [=============>................] - ETA: 52s - loss: 0.2584 - categorical_accuracy: 0.9203
29600/60000 [=============>................] - ETA: 52s - loss: 0.2582 - categorical_accuracy: 0.9203
29632/60000 [=============>................] - ETA: 52s - loss: 0.2581 - categorical_accuracy: 0.9203
29664/60000 [=============>................] - ETA: 52s - loss: 0.2580 - categorical_accuracy: 0.9203
29696/60000 [=============>................] - ETA: 52s - loss: 0.2578 - categorical_accuracy: 0.9204
29728/60000 [=============>................] - ETA: 52s - loss: 0.2576 - categorical_accuracy: 0.9204
29760/60000 [=============>................] - ETA: 52s - loss: 0.2577 - categorical_accuracy: 0.9205
29792/60000 [=============>................] - ETA: 52s - loss: 0.2577 - categorical_accuracy: 0.9205
29824/60000 [=============>................] - ETA: 52s - loss: 0.2575 - categorical_accuracy: 0.9206
29856/60000 [=============>................] - ETA: 51s - loss: 0.2573 - categorical_accuracy: 0.9207
29888/60000 [=============>................] - ETA: 51s - loss: 0.2570 - categorical_accuracy: 0.9207
29920/60000 [=============>................] - ETA: 51s - loss: 0.2571 - categorical_accuracy: 0.9208
29952/60000 [=============>................] - ETA: 51s - loss: 0.2571 - categorical_accuracy: 0.9208
29984/60000 [=============>................] - ETA: 51s - loss: 0.2570 - categorical_accuracy: 0.9208
30016/60000 [==============>...............] - ETA: 51s - loss: 0.2569 - categorical_accuracy: 0.9208
30048/60000 [==============>...............] - ETA: 51s - loss: 0.2571 - categorical_accuracy: 0.9207
30080/60000 [==============>...............] - ETA: 51s - loss: 0.2569 - categorical_accuracy: 0.9208
30112/60000 [==============>...............] - ETA: 51s - loss: 0.2567 - categorical_accuracy: 0.9209
30144/60000 [==============>...............] - ETA: 51s - loss: 0.2564 - categorical_accuracy: 0.9209
30176/60000 [==============>...............] - ETA: 51s - loss: 0.2564 - categorical_accuracy: 0.9210
30208/60000 [==============>...............] - ETA: 51s - loss: 0.2564 - categorical_accuracy: 0.9209
30240/60000 [==============>...............] - ETA: 51s - loss: 0.2562 - categorical_accuracy: 0.9210
30272/60000 [==============>...............] - ETA: 51s - loss: 0.2560 - categorical_accuracy: 0.9210
30304/60000 [==============>...............] - ETA: 51s - loss: 0.2558 - categorical_accuracy: 0.9211
30336/60000 [==============>...............] - ETA: 51s - loss: 0.2559 - categorical_accuracy: 0.9211
30368/60000 [==============>...............] - ETA: 51s - loss: 0.2558 - categorical_accuracy: 0.9212
30400/60000 [==============>...............] - ETA: 51s - loss: 0.2555 - categorical_accuracy: 0.9213
30432/60000 [==============>...............] - ETA: 50s - loss: 0.2553 - categorical_accuracy: 0.9214
30464/60000 [==============>...............] - ETA: 50s - loss: 0.2550 - categorical_accuracy: 0.9214
30496/60000 [==============>...............] - ETA: 50s - loss: 0.2548 - categorical_accuracy: 0.9215
30528/60000 [==============>...............] - ETA: 50s - loss: 0.2547 - categorical_accuracy: 0.9216
30560/60000 [==============>...............] - ETA: 50s - loss: 0.2545 - categorical_accuracy: 0.9216
30592/60000 [==============>...............] - ETA: 50s - loss: 0.2543 - categorical_accuracy: 0.9217
30624/60000 [==============>...............] - ETA: 50s - loss: 0.2542 - categorical_accuracy: 0.9218
30656/60000 [==============>...............] - ETA: 50s - loss: 0.2539 - categorical_accuracy: 0.9218
30688/60000 [==============>...............] - ETA: 50s - loss: 0.2537 - categorical_accuracy: 0.9219
30720/60000 [==============>...............] - ETA: 50s - loss: 0.2536 - categorical_accuracy: 0.9219
30752/60000 [==============>...............] - ETA: 50s - loss: 0.2533 - categorical_accuracy: 0.9220
30784/60000 [==============>...............] - ETA: 50s - loss: 0.2535 - categorical_accuracy: 0.9220
30816/60000 [==============>...............] - ETA: 50s - loss: 0.2533 - categorical_accuracy: 0.9221
30848/60000 [==============>...............] - ETA: 50s - loss: 0.2531 - categorical_accuracy: 0.9222
30880/60000 [==============>...............] - ETA: 50s - loss: 0.2529 - categorical_accuracy: 0.9222
30912/60000 [==============>...............] - ETA: 50s - loss: 0.2527 - categorical_accuracy: 0.9223
30944/60000 [==============>...............] - ETA: 50s - loss: 0.2524 - categorical_accuracy: 0.9224
30976/60000 [==============>...............] - ETA: 50s - loss: 0.2522 - categorical_accuracy: 0.9225
31008/60000 [==============>...............] - ETA: 50s - loss: 0.2520 - categorical_accuracy: 0.9225
31040/60000 [==============>...............] - ETA: 49s - loss: 0.2518 - categorical_accuracy: 0.9226
31072/60000 [==============>...............] - ETA: 49s - loss: 0.2516 - categorical_accuracy: 0.9227
31104/60000 [==============>...............] - ETA: 49s - loss: 0.2514 - categorical_accuracy: 0.9227
31136/60000 [==============>...............] - ETA: 49s - loss: 0.2512 - categorical_accuracy: 0.9228
31168/60000 [==============>...............] - ETA: 49s - loss: 0.2510 - categorical_accuracy: 0.9229
31200/60000 [==============>...............] - ETA: 49s - loss: 0.2509 - categorical_accuracy: 0.9229
31232/60000 [==============>...............] - ETA: 49s - loss: 0.2508 - categorical_accuracy: 0.9229
31264/60000 [==============>...............] - ETA: 49s - loss: 0.2507 - categorical_accuracy: 0.9230
31296/60000 [==============>...............] - ETA: 49s - loss: 0.2505 - categorical_accuracy: 0.9230
31328/60000 [==============>...............] - ETA: 49s - loss: 0.2503 - categorical_accuracy: 0.9231
31360/60000 [==============>...............] - ETA: 49s - loss: 0.2503 - categorical_accuracy: 0.9231
31392/60000 [==============>...............] - ETA: 49s - loss: 0.2501 - categorical_accuracy: 0.9232
31424/60000 [==============>...............] - ETA: 49s - loss: 0.2498 - categorical_accuracy: 0.9232
31456/60000 [==============>...............] - ETA: 49s - loss: 0.2497 - categorical_accuracy: 0.9233
31488/60000 [==============>...............] - ETA: 49s - loss: 0.2495 - categorical_accuracy: 0.9234
31520/60000 [==============>...............] - ETA: 49s - loss: 0.2493 - categorical_accuracy: 0.9234
31584/60000 [==============>...............] - ETA: 49s - loss: 0.2491 - categorical_accuracy: 0.9235
31616/60000 [==============>...............] - ETA: 48s - loss: 0.2489 - categorical_accuracy: 0.9236
31648/60000 [==============>...............] - ETA: 48s - loss: 0.2486 - categorical_accuracy: 0.9236
31680/60000 [==============>...............] - ETA: 48s - loss: 0.2485 - categorical_accuracy: 0.9236
31712/60000 [==============>...............] - ETA: 48s - loss: 0.2483 - categorical_accuracy: 0.9237
31744/60000 [==============>...............] - ETA: 48s - loss: 0.2481 - categorical_accuracy: 0.9237
31776/60000 [==============>...............] - ETA: 48s - loss: 0.2480 - categorical_accuracy: 0.9238
31808/60000 [==============>...............] - ETA: 48s - loss: 0.2477 - categorical_accuracy: 0.9239
31840/60000 [==============>...............] - ETA: 48s - loss: 0.2476 - categorical_accuracy: 0.9239
31872/60000 [==============>...............] - ETA: 48s - loss: 0.2474 - categorical_accuracy: 0.9239
31904/60000 [==============>...............] - ETA: 48s - loss: 0.2471 - categorical_accuracy: 0.9240
31936/60000 [==============>...............] - ETA: 48s - loss: 0.2469 - categorical_accuracy: 0.9241
31968/60000 [==============>...............] - ETA: 48s - loss: 0.2469 - categorical_accuracy: 0.9241
32000/60000 [===============>..............] - ETA: 48s - loss: 0.2467 - categorical_accuracy: 0.9241
32032/60000 [===============>..............] - ETA: 48s - loss: 0.2467 - categorical_accuracy: 0.9242
32064/60000 [===============>..............] - ETA: 48s - loss: 0.2465 - categorical_accuracy: 0.9242
32096/60000 [===============>..............] - ETA: 48s - loss: 0.2464 - categorical_accuracy: 0.9243
32128/60000 [===============>..............] - ETA: 48s - loss: 0.2463 - categorical_accuracy: 0.9243
32160/60000 [===============>..............] - ETA: 48s - loss: 0.2461 - categorical_accuracy: 0.9244
32192/60000 [===============>..............] - ETA: 47s - loss: 0.2459 - categorical_accuracy: 0.9244
32224/60000 [===============>..............] - ETA: 47s - loss: 0.2459 - categorical_accuracy: 0.9245
32256/60000 [===============>..............] - ETA: 47s - loss: 0.2456 - categorical_accuracy: 0.9245
32288/60000 [===============>..............] - ETA: 47s - loss: 0.2454 - categorical_accuracy: 0.9246
32320/60000 [===============>..............] - ETA: 47s - loss: 0.2452 - categorical_accuracy: 0.9247
32352/60000 [===============>..............] - ETA: 47s - loss: 0.2450 - categorical_accuracy: 0.9247
32384/60000 [===============>..............] - ETA: 47s - loss: 0.2448 - categorical_accuracy: 0.9248
32416/60000 [===============>..............] - ETA: 47s - loss: 0.2450 - categorical_accuracy: 0.9248
32448/60000 [===============>..............] - ETA: 47s - loss: 0.2448 - categorical_accuracy: 0.9248
32480/60000 [===============>..............] - ETA: 47s - loss: 0.2446 - categorical_accuracy: 0.9249
32512/60000 [===============>..............] - ETA: 47s - loss: 0.2446 - categorical_accuracy: 0.9249
32544/60000 [===============>..............] - ETA: 47s - loss: 0.2444 - categorical_accuracy: 0.9250
32576/60000 [===============>..............] - ETA: 47s - loss: 0.2441 - categorical_accuracy: 0.9250
32608/60000 [===============>..............] - ETA: 47s - loss: 0.2439 - categorical_accuracy: 0.9251
32640/60000 [===============>..............] - ETA: 47s - loss: 0.2438 - categorical_accuracy: 0.9252
32672/60000 [===============>..............] - ETA: 47s - loss: 0.2438 - categorical_accuracy: 0.9252
32704/60000 [===============>..............] - ETA: 47s - loss: 0.2435 - categorical_accuracy: 0.9253
32736/60000 [===============>..............] - ETA: 47s - loss: 0.2434 - categorical_accuracy: 0.9253
32768/60000 [===============>..............] - ETA: 46s - loss: 0.2432 - categorical_accuracy: 0.9254
32800/60000 [===============>..............] - ETA: 46s - loss: 0.2430 - categorical_accuracy: 0.9255
32832/60000 [===============>..............] - ETA: 46s - loss: 0.2428 - categorical_accuracy: 0.9255
32864/60000 [===============>..............] - ETA: 46s - loss: 0.2429 - categorical_accuracy: 0.9255
32896/60000 [===============>..............] - ETA: 46s - loss: 0.2430 - categorical_accuracy: 0.9255
32928/60000 [===============>..............] - ETA: 46s - loss: 0.2429 - categorical_accuracy: 0.9255
32960/60000 [===============>..............] - ETA: 46s - loss: 0.2427 - categorical_accuracy: 0.9256
32992/60000 [===============>..............] - ETA: 46s - loss: 0.2425 - categorical_accuracy: 0.9257
33024/60000 [===============>..............] - ETA: 46s - loss: 0.2425 - categorical_accuracy: 0.9257
33056/60000 [===============>..............] - ETA: 46s - loss: 0.2424 - categorical_accuracy: 0.9257
33088/60000 [===============>..............] - ETA: 46s - loss: 0.2424 - categorical_accuracy: 0.9257
33120/60000 [===============>..............] - ETA: 46s - loss: 0.2423 - categorical_accuracy: 0.9257
33152/60000 [===============>..............] - ETA: 46s - loss: 0.2422 - categorical_accuracy: 0.9258
33184/60000 [===============>..............] - ETA: 46s - loss: 0.2420 - categorical_accuracy: 0.9258
33216/60000 [===============>..............] - ETA: 46s - loss: 0.2418 - categorical_accuracy: 0.9259
33248/60000 [===============>..............] - ETA: 46s - loss: 0.2416 - categorical_accuracy: 0.9259
33280/60000 [===============>..............] - ETA: 46s - loss: 0.2415 - categorical_accuracy: 0.9260
33312/60000 [===============>..............] - ETA: 46s - loss: 0.2413 - categorical_accuracy: 0.9260
33344/60000 [===============>..............] - ETA: 46s - loss: 0.2411 - categorical_accuracy: 0.9261
33376/60000 [===============>..............] - ETA: 45s - loss: 0.2411 - categorical_accuracy: 0.9261
33408/60000 [===============>..............] - ETA: 45s - loss: 0.2409 - categorical_accuracy: 0.9261
33440/60000 [===============>..............] - ETA: 45s - loss: 0.2407 - categorical_accuracy: 0.9262
33472/60000 [===============>..............] - ETA: 45s - loss: 0.2405 - categorical_accuracy: 0.9262
33504/60000 [===============>..............] - ETA: 45s - loss: 0.2405 - categorical_accuracy: 0.9262
33536/60000 [===============>..............] - ETA: 45s - loss: 0.2403 - categorical_accuracy: 0.9263
33568/60000 [===============>..............] - ETA: 45s - loss: 0.2402 - categorical_accuracy: 0.9264
33600/60000 [===============>..............] - ETA: 45s - loss: 0.2401 - categorical_accuracy: 0.9264
33632/60000 [===============>..............] - ETA: 45s - loss: 0.2399 - categorical_accuracy: 0.9264
33664/60000 [===============>..............] - ETA: 45s - loss: 0.2398 - categorical_accuracy: 0.9265
33696/60000 [===============>..............] - ETA: 45s - loss: 0.2401 - categorical_accuracy: 0.9265
33728/60000 [===============>..............] - ETA: 45s - loss: 0.2400 - categorical_accuracy: 0.9266
33760/60000 [===============>..............] - ETA: 45s - loss: 0.2400 - categorical_accuracy: 0.9265
33792/60000 [===============>..............] - ETA: 45s - loss: 0.2399 - categorical_accuracy: 0.9266
33824/60000 [===============>..............] - ETA: 45s - loss: 0.2397 - categorical_accuracy: 0.9267
33856/60000 [===============>..............] - ETA: 45s - loss: 0.2396 - categorical_accuracy: 0.9267
33888/60000 [===============>..............] - ETA: 45s - loss: 0.2397 - categorical_accuracy: 0.9267
33920/60000 [===============>..............] - ETA: 45s - loss: 0.2396 - categorical_accuracy: 0.9267
33952/60000 [===============>..............] - ETA: 44s - loss: 0.2394 - categorical_accuracy: 0.9268
33984/60000 [===============>..............] - ETA: 44s - loss: 0.2392 - categorical_accuracy: 0.9268
34016/60000 [================>.............] - ETA: 44s - loss: 0.2391 - categorical_accuracy: 0.9269
34048/60000 [================>.............] - ETA: 44s - loss: 0.2389 - categorical_accuracy: 0.9270
34080/60000 [================>.............] - ETA: 44s - loss: 0.2388 - categorical_accuracy: 0.9270
34112/60000 [================>.............] - ETA: 44s - loss: 0.2386 - categorical_accuracy: 0.9270
34144/60000 [================>.............] - ETA: 44s - loss: 0.2385 - categorical_accuracy: 0.9270
34176/60000 [================>.............] - ETA: 44s - loss: 0.2384 - categorical_accuracy: 0.9271
34208/60000 [================>.............] - ETA: 44s - loss: 0.2384 - categorical_accuracy: 0.9271
34240/60000 [================>.............] - ETA: 44s - loss: 0.2382 - categorical_accuracy: 0.9271
34272/60000 [================>.............] - ETA: 44s - loss: 0.2381 - categorical_accuracy: 0.9272
34304/60000 [================>.............] - ETA: 44s - loss: 0.2380 - categorical_accuracy: 0.9272
34368/60000 [================>.............] - ETA: 44s - loss: 0.2379 - categorical_accuracy: 0.9272
34400/60000 [================>.............] - ETA: 44s - loss: 0.2378 - categorical_accuracy: 0.9272
34432/60000 [================>.............] - ETA: 44s - loss: 0.2378 - categorical_accuracy: 0.9272
34464/60000 [================>.............] - ETA: 44s - loss: 0.2379 - categorical_accuracy: 0.9273
34496/60000 [================>.............] - ETA: 44s - loss: 0.2377 - categorical_accuracy: 0.9273
34528/60000 [================>.............] - ETA: 43s - loss: 0.2376 - categorical_accuracy: 0.9274
34560/60000 [================>.............] - ETA: 43s - loss: 0.2374 - categorical_accuracy: 0.9274
34592/60000 [================>.............] - ETA: 43s - loss: 0.2372 - categorical_accuracy: 0.9275
34624/60000 [================>.............] - ETA: 43s - loss: 0.2373 - categorical_accuracy: 0.9275
34656/60000 [================>.............] - ETA: 43s - loss: 0.2372 - categorical_accuracy: 0.9275
34688/60000 [================>.............] - ETA: 43s - loss: 0.2371 - categorical_accuracy: 0.9275
34720/60000 [================>.............] - ETA: 43s - loss: 0.2371 - categorical_accuracy: 0.9274
34752/60000 [================>.............] - ETA: 43s - loss: 0.2369 - categorical_accuracy: 0.9275
34784/60000 [================>.............] - ETA: 43s - loss: 0.2367 - categorical_accuracy: 0.9276
34816/60000 [================>.............] - ETA: 43s - loss: 0.2367 - categorical_accuracy: 0.9276
34848/60000 [================>.............] - ETA: 43s - loss: 0.2365 - categorical_accuracy: 0.9276
34880/60000 [================>.............] - ETA: 43s - loss: 0.2364 - categorical_accuracy: 0.9277
34912/60000 [================>.............] - ETA: 43s - loss: 0.2364 - categorical_accuracy: 0.9277
34976/60000 [================>.............] - ETA: 43s - loss: 0.2360 - categorical_accuracy: 0.9278
35008/60000 [================>.............] - ETA: 43s - loss: 0.2361 - categorical_accuracy: 0.9278
35040/60000 [================>.............] - ETA: 43s - loss: 0.2364 - categorical_accuracy: 0.9278
35072/60000 [================>.............] - ETA: 43s - loss: 0.2363 - categorical_accuracy: 0.9278
35104/60000 [================>.............] - ETA: 42s - loss: 0.2361 - categorical_accuracy: 0.9279
35136/60000 [================>.............] - ETA: 42s - loss: 0.2360 - categorical_accuracy: 0.9280
35168/60000 [================>.............] - ETA: 42s - loss: 0.2359 - categorical_accuracy: 0.9280
35200/60000 [================>.............] - ETA: 42s - loss: 0.2357 - categorical_accuracy: 0.9281
35232/60000 [================>.............] - ETA: 42s - loss: 0.2356 - categorical_accuracy: 0.9281
35264/60000 [================>.............] - ETA: 42s - loss: 0.2356 - categorical_accuracy: 0.9281
35296/60000 [================>.............] - ETA: 42s - loss: 0.2354 - categorical_accuracy: 0.9282
35328/60000 [================>.............] - ETA: 42s - loss: 0.2352 - categorical_accuracy: 0.9282
35360/60000 [================>.............] - ETA: 42s - loss: 0.2351 - categorical_accuracy: 0.9283
35392/60000 [================>.............] - ETA: 42s - loss: 0.2350 - categorical_accuracy: 0.9283
35424/60000 [================>.............] - ETA: 42s - loss: 0.2348 - categorical_accuracy: 0.9284
35456/60000 [================>.............] - ETA: 42s - loss: 0.2348 - categorical_accuracy: 0.9284
35488/60000 [================>.............] - ETA: 42s - loss: 0.2347 - categorical_accuracy: 0.9284
35520/60000 [================>.............] - ETA: 42s - loss: 0.2347 - categorical_accuracy: 0.9284
35552/60000 [================>.............] - ETA: 42s - loss: 0.2347 - categorical_accuracy: 0.9284
35584/60000 [================>.............] - ETA: 42s - loss: 0.2345 - categorical_accuracy: 0.9285
35616/60000 [================>.............] - ETA: 42s - loss: 0.2346 - categorical_accuracy: 0.9284
35648/60000 [================>.............] - ETA: 42s - loss: 0.2345 - categorical_accuracy: 0.9285
35680/60000 [================>.............] - ETA: 41s - loss: 0.2343 - categorical_accuracy: 0.9285
35712/60000 [================>.............] - ETA: 41s - loss: 0.2342 - categorical_accuracy: 0.9286
35744/60000 [================>.............] - ETA: 41s - loss: 0.2340 - categorical_accuracy: 0.9286
35776/60000 [================>.............] - ETA: 41s - loss: 0.2339 - categorical_accuracy: 0.9287
35808/60000 [================>.............] - ETA: 41s - loss: 0.2337 - categorical_accuracy: 0.9287
35840/60000 [================>.............] - ETA: 41s - loss: 0.2336 - categorical_accuracy: 0.9288
35872/60000 [================>.............] - ETA: 41s - loss: 0.2336 - categorical_accuracy: 0.9287
35904/60000 [================>.............] - ETA: 41s - loss: 0.2335 - categorical_accuracy: 0.9288
35936/60000 [================>.............] - ETA: 41s - loss: 0.2336 - categorical_accuracy: 0.9288
35968/60000 [================>.............] - ETA: 41s - loss: 0.2335 - categorical_accuracy: 0.9289
36000/60000 [=================>............] - ETA: 41s - loss: 0.2336 - categorical_accuracy: 0.9289
36032/60000 [=================>............] - ETA: 41s - loss: 0.2334 - categorical_accuracy: 0.9289
36064/60000 [=================>............] - ETA: 41s - loss: 0.2332 - categorical_accuracy: 0.9290
36096/60000 [=================>............] - ETA: 41s - loss: 0.2332 - categorical_accuracy: 0.9290
36128/60000 [=================>............] - ETA: 41s - loss: 0.2332 - categorical_accuracy: 0.9290
36160/60000 [=================>............] - ETA: 41s - loss: 0.2330 - categorical_accuracy: 0.9291
36192/60000 [=================>............] - ETA: 41s - loss: 0.2328 - categorical_accuracy: 0.9291
36256/60000 [=================>............] - ETA: 40s - loss: 0.2327 - categorical_accuracy: 0.9291
36288/60000 [=================>............] - ETA: 40s - loss: 0.2325 - categorical_accuracy: 0.9292
36320/60000 [=================>............] - ETA: 40s - loss: 0.2323 - categorical_accuracy: 0.9292
36352/60000 [=================>............] - ETA: 40s - loss: 0.2321 - categorical_accuracy: 0.9293
36384/60000 [=================>............] - ETA: 40s - loss: 0.2323 - categorical_accuracy: 0.9293
36416/60000 [=================>............] - ETA: 40s - loss: 0.2322 - categorical_accuracy: 0.9293
36448/60000 [=================>............] - ETA: 40s - loss: 0.2320 - categorical_accuracy: 0.9294
36480/60000 [=================>............] - ETA: 40s - loss: 0.2318 - categorical_accuracy: 0.9294
36512/60000 [=================>............] - ETA: 40s - loss: 0.2316 - categorical_accuracy: 0.9295
36544/60000 [=================>............] - ETA: 40s - loss: 0.2316 - categorical_accuracy: 0.9295
36576/60000 [=================>............] - ETA: 40s - loss: 0.2315 - categorical_accuracy: 0.9295
36608/60000 [=================>............] - ETA: 40s - loss: 0.2314 - categorical_accuracy: 0.9296
36640/60000 [=================>............] - ETA: 40s - loss: 0.2312 - categorical_accuracy: 0.9296
36672/60000 [=================>............] - ETA: 40s - loss: 0.2310 - categorical_accuracy: 0.9297
36704/60000 [=================>............] - ETA: 40s - loss: 0.2308 - categorical_accuracy: 0.9297
36736/60000 [=================>............] - ETA: 40s - loss: 0.2307 - categorical_accuracy: 0.9298
36768/60000 [=================>............] - ETA: 40s - loss: 0.2305 - categorical_accuracy: 0.9298
36800/60000 [=================>............] - ETA: 40s - loss: 0.2304 - categorical_accuracy: 0.9299
36832/60000 [=================>............] - ETA: 39s - loss: 0.2302 - categorical_accuracy: 0.9299
36896/60000 [=================>............] - ETA: 39s - loss: 0.2300 - categorical_accuracy: 0.9300
36928/60000 [=================>............] - ETA: 39s - loss: 0.2299 - categorical_accuracy: 0.9300
36960/60000 [=================>............] - ETA: 39s - loss: 0.2298 - categorical_accuracy: 0.9301
36992/60000 [=================>............] - ETA: 39s - loss: 0.2296 - categorical_accuracy: 0.9301
37024/60000 [=================>............] - ETA: 39s - loss: 0.2294 - categorical_accuracy: 0.9302
37056/60000 [=================>............] - ETA: 39s - loss: 0.2294 - categorical_accuracy: 0.9302
37088/60000 [=================>............] - ETA: 39s - loss: 0.2293 - categorical_accuracy: 0.9302
37120/60000 [=================>............] - ETA: 39s - loss: 0.2292 - categorical_accuracy: 0.9303
37152/60000 [=================>............] - ETA: 39s - loss: 0.2290 - categorical_accuracy: 0.9303
37184/60000 [=================>............] - ETA: 39s - loss: 0.2289 - categorical_accuracy: 0.9303
37216/60000 [=================>............] - ETA: 39s - loss: 0.2288 - categorical_accuracy: 0.9304
37248/60000 [=================>............] - ETA: 39s - loss: 0.2287 - categorical_accuracy: 0.9304
37280/60000 [=================>............] - ETA: 39s - loss: 0.2287 - categorical_accuracy: 0.9304
37312/60000 [=================>............] - ETA: 39s - loss: 0.2285 - categorical_accuracy: 0.9304
37344/60000 [=================>............] - ETA: 39s - loss: 0.2284 - categorical_accuracy: 0.9305
37376/60000 [=================>............] - ETA: 39s - loss: 0.2283 - categorical_accuracy: 0.9305
37408/60000 [=================>............] - ETA: 38s - loss: 0.2281 - categorical_accuracy: 0.9305
37440/60000 [=================>............] - ETA: 38s - loss: 0.2279 - categorical_accuracy: 0.9306
37472/60000 [=================>............] - ETA: 38s - loss: 0.2278 - categorical_accuracy: 0.9306
37504/60000 [=================>............] - ETA: 38s - loss: 0.2278 - categorical_accuracy: 0.9306
37536/60000 [=================>............] - ETA: 38s - loss: 0.2276 - categorical_accuracy: 0.9307
37568/60000 [=================>............] - ETA: 38s - loss: 0.2275 - categorical_accuracy: 0.9306
37600/60000 [=================>............] - ETA: 38s - loss: 0.2273 - categorical_accuracy: 0.9307
37632/60000 [=================>............] - ETA: 38s - loss: 0.2273 - categorical_accuracy: 0.9307
37696/60000 [=================>............] - ETA: 38s - loss: 0.2273 - categorical_accuracy: 0.9307
37728/60000 [=================>............] - ETA: 38s - loss: 0.2272 - categorical_accuracy: 0.9307
37792/60000 [=================>............] - ETA: 38s - loss: 0.2269 - categorical_accuracy: 0.9308
37824/60000 [=================>............] - ETA: 38s - loss: 0.2267 - categorical_accuracy: 0.9308
37856/60000 [=================>............] - ETA: 38s - loss: 0.2265 - categorical_accuracy: 0.9309
37888/60000 [=================>............] - ETA: 38s - loss: 0.2267 - categorical_accuracy: 0.9309
37920/60000 [=================>............] - ETA: 38s - loss: 0.2267 - categorical_accuracy: 0.9308
37952/60000 [=================>............] - ETA: 37s - loss: 0.2265 - categorical_accuracy: 0.9309
37984/60000 [=================>............] - ETA: 37s - loss: 0.2264 - categorical_accuracy: 0.9309
38016/60000 [==================>...........] - ETA: 37s - loss: 0.2262 - categorical_accuracy: 0.9310
38048/60000 [==================>...........] - ETA: 37s - loss: 0.2261 - categorical_accuracy: 0.9310
38112/60000 [==================>...........] - ETA: 37s - loss: 0.2258 - categorical_accuracy: 0.9311
38144/60000 [==================>...........] - ETA: 37s - loss: 0.2259 - categorical_accuracy: 0.9311
38176/60000 [==================>...........] - ETA: 37s - loss: 0.2257 - categorical_accuracy: 0.9312
38208/60000 [==================>...........] - ETA: 37s - loss: 0.2256 - categorical_accuracy: 0.9312
38240/60000 [==================>...........] - ETA: 37s - loss: 0.2255 - categorical_accuracy: 0.9312
38272/60000 [==================>...........] - ETA: 37s - loss: 0.2253 - categorical_accuracy: 0.9312
38304/60000 [==================>...........] - ETA: 37s - loss: 0.2252 - categorical_accuracy: 0.9313
38336/60000 [==================>...........] - ETA: 37s - loss: 0.2255 - categorical_accuracy: 0.9312
38368/60000 [==================>...........] - ETA: 37s - loss: 0.2254 - categorical_accuracy: 0.9312
38400/60000 [==================>...........] - ETA: 37s - loss: 0.2253 - categorical_accuracy: 0.9313
38432/60000 [==================>...........] - ETA: 37s - loss: 0.2251 - categorical_accuracy: 0.9313
38464/60000 [==================>...........] - ETA: 37s - loss: 0.2250 - categorical_accuracy: 0.9314
38496/60000 [==================>...........] - ETA: 37s - loss: 0.2248 - categorical_accuracy: 0.9314
38528/60000 [==================>...........] - ETA: 36s - loss: 0.2246 - categorical_accuracy: 0.9315
38560/60000 [==================>...........] - ETA: 36s - loss: 0.2246 - categorical_accuracy: 0.9315
38592/60000 [==================>...........] - ETA: 36s - loss: 0.2246 - categorical_accuracy: 0.9315
38624/60000 [==================>...........] - ETA: 36s - loss: 0.2245 - categorical_accuracy: 0.9315
38656/60000 [==================>...........] - ETA: 36s - loss: 0.2244 - categorical_accuracy: 0.9315
38688/60000 [==================>...........] - ETA: 36s - loss: 0.2242 - categorical_accuracy: 0.9316
38720/60000 [==================>...........] - ETA: 36s - loss: 0.2241 - categorical_accuracy: 0.9316
38752/60000 [==================>...........] - ETA: 36s - loss: 0.2242 - categorical_accuracy: 0.9316
38784/60000 [==================>...........] - ETA: 36s - loss: 0.2240 - categorical_accuracy: 0.9316
38816/60000 [==================>...........] - ETA: 36s - loss: 0.2239 - categorical_accuracy: 0.9317
38848/60000 [==================>...........] - ETA: 36s - loss: 0.2240 - categorical_accuracy: 0.9317
38880/60000 [==================>...........] - ETA: 36s - loss: 0.2239 - categorical_accuracy: 0.9317
38912/60000 [==================>...........] - ETA: 36s - loss: 0.2241 - categorical_accuracy: 0.9317
38944/60000 [==================>...........] - ETA: 36s - loss: 0.2239 - categorical_accuracy: 0.9317
38976/60000 [==================>...........] - ETA: 36s - loss: 0.2238 - categorical_accuracy: 0.9318
39008/60000 [==================>...........] - ETA: 36s - loss: 0.2238 - categorical_accuracy: 0.9318
39040/60000 [==================>...........] - ETA: 36s - loss: 0.2236 - categorical_accuracy: 0.9318
39072/60000 [==================>...........] - ETA: 36s - loss: 0.2238 - categorical_accuracy: 0.9318
39104/60000 [==================>...........] - ETA: 35s - loss: 0.2237 - categorical_accuracy: 0.9318
39136/60000 [==================>...........] - ETA: 35s - loss: 0.2236 - categorical_accuracy: 0.9318
39200/60000 [==================>...........] - ETA: 35s - loss: 0.2233 - categorical_accuracy: 0.9319
39232/60000 [==================>...........] - ETA: 35s - loss: 0.2231 - categorical_accuracy: 0.9320
39264/60000 [==================>...........] - ETA: 35s - loss: 0.2230 - categorical_accuracy: 0.9320
39296/60000 [==================>...........] - ETA: 35s - loss: 0.2229 - categorical_accuracy: 0.9321
39328/60000 [==================>...........] - ETA: 35s - loss: 0.2227 - categorical_accuracy: 0.9321
39360/60000 [==================>...........] - ETA: 35s - loss: 0.2226 - categorical_accuracy: 0.9321
39392/60000 [==================>...........] - ETA: 35s - loss: 0.2225 - categorical_accuracy: 0.9322
39424/60000 [==================>...........] - ETA: 35s - loss: 0.2223 - categorical_accuracy: 0.9322
39456/60000 [==================>...........] - ETA: 35s - loss: 0.2223 - categorical_accuracy: 0.9323
39488/60000 [==================>...........] - ETA: 35s - loss: 0.2222 - categorical_accuracy: 0.9323
39520/60000 [==================>...........] - ETA: 35s - loss: 0.2221 - categorical_accuracy: 0.9323
39552/60000 [==================>...........] - ETA: 35s - loss: 0.2220 - categorical_accuracy: 0.9323
39584/60000 [==================>...........] - ETA: 35s - loss: 0.2218 - categorical_accuracy: 0.9324
39616/60000 [==================>...........] - ETA: 35s - loss: 0.2217 - categorical_accuracy: 0.9324
39648/60000 [==================>...........] - ETA: 35s - loss: 0.2218 - categorical_accuracy: 0.9324
39680/60000 [==================>...........] - ETA: 34s - loss: 0.2218 - categorical_accuracy: 0.9324
39712/60000 [==================>...........] - ETA: 34s - loss: 0.2218 - categorical_accuracy: 0.9324
39744/60000 [==================>...........] - ETA: 34s - loss: 0.2219 - categorical_accuracy: 0.9323
39776/60000 [==================>...........] - ETA: 34s - loss: 0.2219 - categorical_accuracy: 0.9323
39808/60000 [==================>...........] - ETA: 34s - loss: 0.2219 - categorical_accuracy: 0.9323
39840/60000 [==================>...........] - ETA: 34s - loss: 0.2218 - categorical_accuracy: 0.9323
39872/60000 [==================>...........] - ETA: 34s - loss: 0.2217 - categorical_accuracy: 0.9324
39904/60000 [==================>...........] - ETA: 34s - loss: 0.2216 - categorical_accuracy: 0.9324
39936/60000 [==================>...........] - ETA: 34s - loss: 0.2214 - categorical_accuracy: 0.9324
39968/60000 [==================>...........] - ETA: 34s - loss: 0.2216 - categorical_accuracy: 0.9324
40000/60000 [===================>..........] - ETA: 34s - loss: 0.2214 - categorical_accuracy: 0.9325
40032/60000 [===================>..........] - ETA: 34s - loss: 0.2213 - categorical_accuracy: 0.9325
40064/60000 [===================>..........] - ETA: 34s - loss: 0.2211 - categorical_accuracy: 0.9326
40096/60000 [===================>..........] - ETA: 34s - loss: 0.2210 - categorical_accuracy: 0.9326
40128/60000 [===================>..........] - ETA: 34s - loss: 0.2209 - categorical_accuracy: 0.9326
40160/60000 [===================>..........] - ETA: 34s - loss: 0.2209 - categorical_accuracy: 0.9326
40192/60000 [===================>..........] - ETA: 34s - loss: 0.2208 - categorical_accuracy: 0.9326
40224/60000 [===================>..........] - ETA: 34s - loss: 0.2207 - categorical_accuracy: 0.9327
40256/60000 [===================>..........] - ETA: 33s - loss: 0.2206 - categorical_accuracy: 0.9327
40288/60000 [===================>..........] - ETA: 33s - loss: 0.2206 - categorical_accuracy: 0.9327
40320/60000 [===================>..........] - ETA: 33s - loss: 0.2204 - categorical_accuracy: 0.9328
40352/60000 [===================>..........] - ETA: 33s - loss: 0.2204 - categorical_accuracy: 0.9328
40384/60000 [===================>..........] - ETA: 33s - loss: 0.2204 - categorical_accuracy: 0.9328
40416/60000 [===================>..........] - ETA: 33s - loss: 0.2202 - categorical_accuracy: 0.9328
40448/60000 [===================>..........] - ETA: 33s - loss: 0.2201 - categorical_accuracy: 0.9329
40480/60000 [===================>..........] - ETA: 33s - loss: 0.2200 - categorical_accuracy: 0.9329
40512/60000 [===================>..........] - ETA: 33s - loss: 0.2198 - categorical_accuracy: 0.9330
40544/60000 [===================>..........] - ETA: 33s - loss: 0.2197 - categorical_accuracy: 0.9330
40576/60000 [===================>..........] - ETA: 33s - loss: 0.2196 - categorical_accuracy: 0.9331
40608/60000 [===================>..........] - ETA: 33s - loss: 0.2195 - categorical_accuracy: 0.9331
40640/60000 [===================>..........] - ETA: 33s - loss: 0.2195 - categorical_accuracy: 0.9331
40672/60000 [===================>..........] - ETA: 33s - loss: 0.2193 - categorical_accuracy: 0.9331
40704/60000 [===================>..........] - ETA: 33s - loss: 0.2192 - categorical_accuracy: 0.9332
40736/60000 [===================>..........] - ETA: 33s - loss: 0.2191 - categorical_accuracy: 0.9332
40768/60000 [===================>..........] - ETA: 33s - loss: 0.2190 - categorical_accuracy: 0.9333
40800/60000 [===================>..........] - ETA: 33s - loss: 0.2189 - categorical_accuracy: 0.9333
40832/60000 [===================>..........] - ETA: 32s - loss: 0.2189 - categorical_accuracy: 0.9332
40864/60000 [===================>..........] - ETA: 32s - loss: 0.2188 - categorical_accuracy: 0.9333
40896/60000 [===================>..........] - ETA: 32s - loss: 0.2187 - categorical_accuracy: 0.9333
40928/60000 [===================>..........] - ETA: 32s - loss: 0.2189 - categorical_accuracy: 0.9332
40960/60000 [===================>..........] - ETA: 32s - loss: 0.2188 - categorical_accuracy: 0.9332
40992/60000 [===================>..........] - ETA: 32s - loss: 0.2187 - categorical_accuracy: 0.9333
41024/60000 [===================>..........] - ETA: 32s - loss: 0.2186 - categorical_accuracy: 0.9333
41056/60000 [===================>..........] - ETA: 32s - loss: 0.2186 - categorical_accuracy: 0.9333
41088/60000 [===================>..........] - ETA: 32s - loss: 0.2184 - categorical_accuracy: 0.9333
41120/60000 [===================>..........] - ETA: 32s - loss: 0.2184 - categorical_accuracy: 0.9333
41152/60000 [===================>..........] - ETA: 32s - loss: 0.2183 - categorical_accuracy: 0.9333
41184/60000 [===================>..........] - ETA: 32s - loss: 0.2181 - categorical_accuracy: 0.9334
41216/60000 [===================>..........] - ETA: 32s - loss: 0.2180 - categorical_accuracy: 0.9334
41248/60000 [===================>..........] - ETA: 32s - loss: 0.2179 - categorical_accuracy: 0.9334
41280/60000 [===================>..........] - ETA: 32s - loss: 0.2178 - categorical_accuracy: 0.9335
41312/60000 [===================>..........] - ETA: 32s - loss: 0.2177 - categorical_accuracy: 0.9335
41344/60000 [===================>..........] - ETA: 32s - loss: 0.2178 - categorical_accuracy: 0.9335
41376/60000 [===================>..........] - ETA: 32s - loss: 0.2177 - categorical_accuracy: 0.9335
41408/60000 [===================>..........] - ETA: 32s - loss: 0.2176 - categorical_accuracy: 0.9336
41440/60000 [===================>..........] - ETA: 31s - loss: 0.2175 - categorical_accuracy: 0.9336
41472/60000 [===================>..........] - ETA: 31s - loss: 0.2173 - categorical_accuracy: 0.9336
41504/60000 [===================>..........] - ETA: 31s - loss: 0.2174 - categorical_accuracy: 0.9336
41536/60000 [===================>..........] - ETA: 31s - loss: 0.2172 - categorical_accuracy: 0.9337
41568/60000 [===================>..........] - ETA: 31s - loss: 0.2171 - categorical_accuracy: 0.9337
41600/60000 [===================>..........] - ETA: 31s - loss: 0.2170 - categorical_accuracy: 0.9338
41632/60000 [===================>..........] - ETA: 31s - loss: 0.2168 - categorical_accuracy: 0.9338
41664/60000 [===================>..........] - ETA: 31s - loss: 0.2167 - categorical_accuracy: 0.9339
41696/60000 [===================>..........] - ETA: 31s - loss: 0.2165 - categorical_accuracy: 0.9339
41728/60000 [===================>..........] - ETA: 31s - loss: 0.2165 - categorical_accuracy: 0.9340
41760/60000 [===================>..........] - ETA: 31s - loss: 0.2164 - categorical_accuracy: 0.9340
41792/60000 [===================>..........] - ETA: 31s - loss: 0.2163 - categorical_accuracy: 0.9340
41824/60000 [===================>..........] - ETA: 31s - loss: 0.2161 - categorical_accuracy: 0.9341
41856/60000 [===================>..........] - ETA: 31s - loss: 0.2161 - categorical_accuracy: 0.9341
41888/60000 [===================>..........] - ETA: 31s - loss: 0.2160 - categorical_accuracy: 0.9341
41920/60000 [===================>..........] - ETA: 31s - loss: 0.2161 - categorical_accuracy: 0.9341
41952/60000 [===================>..........] - ETA: 31s - loss: 0.2161 - categorical_accuracy: 0.9341
41984/60000 [===================>..........] - ETA: 31s - loss: 0.2161 - categorical_accuracy: 0.9341
42016/60000 [====================>.........] - ETA: 30s - loss: 0.2162 - categorical_accuracy: 0.9341
42048/60000 [====================>.........] - ETA: 30s - loss: 0.2161 - categorical_accuracy: 0.9341
42080/60000 [====================>.........] - ETA: 30s - loss: 0.2160 - categorical_accuracy: 0.9341
42112/60000 [====================>.........] - ETA: 30s - loss: 0.2159 - categorical_accuracy: 0.9342
42144/60000 [====================>.........] - ETA: 30s - loss: 0.2158 - categorical_accuracy: 0.9342
42176/60000 [====================>.........] - ETA: 30s - loss: 0.2156 - categorical_accuracy: 0.9342
42208/60000 [====================>.........] - ETA: 30s - loss: 0.2155 - categorical_accuracy: 0.9342
42240/60000 [====================>.........] - ETA: 30s - loss: 0.2154 - categorical_accuracy: 0.9343
42272/60000 [====================>.........] - ETA: 30s - loss: 0.2152 - categorical_accuracy: 0.9343
42304/60000 [====================>.........] - ETA: 30s - loss: 0.2151 - categorical_accuracy: 0.9343
42336/60000 [====================>.........] - ETA: 30s - loss: 0.2150 - categorical_accuracy: 0.9344
42368/60000 [====================>.........] - ETA: 30s - loss: 0.2149 - categorical_accuracy: 0.9344
42400/60000 [====================>.........] - ETA: 30s - loss: 0.2148 - categorical_accuracy: 0.9344
42432/60000 [====================>.........] - ETA: 30s - loss: 0.2147 - categorical_accuracy: 0.9345
42464/60000 [====================>.........] - ETA: 30s - loss: 0.2146 - categorical_accuracy: 0.9345
42496/60000 [====================>.........] - ETA: 30s - loss: 0.2144 - categorical_accuracy: 0.9346
42528/60000 [====================>.........] - ETA: 30s - loss: 0.2143 - categorical_accuracy: 0.9346
42560/60000 [====================>.........] - ETA: 30s - loss: 0.2142 - categorical_accuracy: 0.9347
42592/60000 [====================>.........] - ETA: 29s - loss: 0.2140 - categorical_accuracy: 0.9347
42624/60000 [====================>.........] - ETA: 29s - loss: 0.2139 - categorical_accuracy: 0.9347
42656/60000 [====================>.........] - ETA: 29s - loss: 0.2139 - categorical_accuracy: 0.9347
42688/60000 [====================>.........] - ETA: 29s - loss: 0.2138 - categorical_accuracy: 0.9347
42720/60000 [====================>.........] - ETA: 29s - loss: 0.2137 - categorical_accuracy: 0.9348
42752/60000 [====================>.........] - ETA: 29s - loss: 0.2138 - categorical_accuracy: 0.9348
42784/60000 [====================>.........] - ETA: 29s - loss: 0.2137 - categorical_accuracy: 0.9348
42816/60000 [====================>.........] - ETA: 29s - loss: 0.2136 - categorical_accuracy: 0.9349
42848/60000 [====================>.........] - ETA: 29s - loss: 0.2135 - categorical_accuracy: 0.9349
42880/60000 [====================>.........] - ETA: 29s - loss: 0.2134 - categorical_accuracy: 0.9349
42912/60000 [====================>.........] - ETA: 29s - loss: 0.2133 - categorical_accuracy: 0.9350
42944/60000 [====================>.........] - ETA: 29s - loss: 0.2132 - categorical_accuracy: 0.9350
43008/60000 [====================>.........] - ETA: 29s - loss: 0.2129 - categorical_accuracy: 0.9351
43040/60000 [====================>.........] - ETA: 29s - loss: 0.2130 - categorical_accuracy: 0.9350
43072/60000 [====================>.........] - ETA: 29s - loss: 0.2129 - categorical_accuracy: 0.9350
43104/60000 [====================>.........] - ETA: 29s - loss: 0.2129 - categorical_accuracy: 0.9351
43136/60000 [====================>.........] - ETA: 29s - loss: 0.2127 - categorical_accuracy: 0.9351
43168/60000 [====================>.........] - ETA: 28s - loss: 0.2128 - categorical_accuracy: 0.9351
43200/60000 [====================>.........] - ETA: 28s - loss: 0.2126 - categorical_accuracy: 0.9352
43232/60000 [====================>.........] - ETA: 28s - loss: 0.2126 - categorical_accuracy: 0.9352
43264/60000 [====================>.........] - ETA: 28s - loss: 0.2126 - categorical_accuracy: 0.9352
43296/60000 [====================>.........] - ETA: 28s - loss: 0.2125 - categorical_accuracy: 0.9352
43328/60000 [====================>.........] - ETA: 28s - loss: 0.2124 - categorical_accuracy: 0.9353
43360/60000 [====================>.........] - ETA: 28s - loss: 0.2123 - categorical_accuracy: 0.9353
43392/60000 [====================>.........] - ETA: 28s - loss: 0.2123 - categorical_accuracy: 0.9353
43424/60000 [====================>.........] - ETA: 28s - loss: 0.2123 - categorical_accuracy: 0.9353
43456/60000 [====================>.........] - ETA: 28s - loss: 0.2123 - categorical_accuracy: 0.9353
43488/60000 [====================>.........] - ETA: 28s - loss: 0.2122 - categorical_accuracy: 0.9353
43520/60000 [====================>.........] - ETA: 28s - loss: 0.2122 - categorical_accuracy: 0.9353
43552/60000 [====================>.........] - ETA: 28s - loss: 0.2120 - categorical_accuracy: 0.9353
43584/60000 [====================>.........] - ETA: 28s - loss: 0.2121 - categorical_accuracy: 0.9353
43616/60000 [====================>.........] - ETA: 28s - loss: 0.2121 - categorical_accuracy: 0.9353
43648/60000 [====================>.........] - ETA: 28s - loss: 0.2119 - categorical_accuracy: 0.9354
43680/60000 [====================>.........] - ETA: 28s - loss: 0.2119 - categorical_accuracy: 0.9353
43712/60000 [====================>.........] - ETA: 28s - loss: 0.2118 - categorical_accuracy: 0.9354
43744/60000 [====================>.........] - ETA: 27s - loss: 0.2119 - categorical_accuracy: 0.9354
43776/60000 [====================>.........] - ETA: 27s - loss: 0.2118 - categorical_accuracy: 0.9354
43808/60000 [====================>.........] - ETA: 27s - loss: 0.2117 - categorical_accuracy: 0.9355
43840/60000 [====================>.........] - ETA: 27s - loss: 0.2116 - categorical_accuracy: 0.9355
43872/60000 [====================>.........] - ETA: 27s - loss: 0.2115 - categorical_accuracy: 0.9355
43904/60000 [====================>.........] - ETA: 27s - loss: 0.2114 - categorical_accuracy: 0.9356
43936/60000 [====================>.........] - ETA: 27s - loss: 0.2113 - categorical_accuracy: 0.9356
43968/60000 [====================>.........] - ETA: 27s - loss: 0.2112 - categorical_accuracy: 0.9356
44000/60000 [=====================>........] - ETA: 27s - loss: 0.2111 - categorical_accuracy: 0.9356
44032/60000 [=====================>........] - ETA: 27s - loss: 0.2111 - categorical_accuracy: 0.9357
44064/60000 [=====================>........] - ETA: 27s - loss: 0.2111 - categorical_accuracy: 0.9357
44096/60000 [=====================>........] - ETA: 27s - loss: 0.2113 - categorical_accuracy: 0.9356
44128/60000 [=====================>........] - ETA: 27s - loss: 0.2112 - categorical_accuracy: 0.9357
44160/60000 [=====================>........] - ETA: 27s - loss: 0.2111 - categorical_accuracy: 0.9357
44192/60000 [=====================>........] - ETA: 27s - loss: 0.2111 - categorical_accuracy: 0.9356
44224/60000 [=====================>........] - ETA: 27s - loss: 0.2110 - categorical_accuracy: 0.9356
44256/60000 [=====================>........] - ETA: 27s - loss: 0.2109 - categorical_accuracy: 0.9357
44288/60000 [=====================>........] - ETA: 27s - loss: 0.2108 - categorical_accuracy: 0.9357
44320/60000 [=====================>........] - ETA: 27s - loss: 0.2109 - categorical_accuracy: 0.9357
44352/60000 [=====================>........] - ETA: 26s - loss: 0.2108 - categorical_accuracy: 0.9357
44384/60000 [=====================>........] - ETA: 26s - loss: 0.2107 - categorical_accuracy: 0.9357
44416/60000 [=====================>........] - ETA: 26s - loss: 0.2106 - categorical_accuracy: 0.9357
44448/60000 [=====================>........] - ETA: 26s - loss: 0.2107 - categorical_accuracy: 0.9357
44480/60000 [=====================>........] - ETA: 26s - loss: 0.2106 - categorical_accuracy: 0.9357
44512/60000 [=====================>........] - ETA: 26s - loss: 0.2105 - categorical_accuracy: 0.9358
44544/60000 [=====================>........] - ETA: 26s - loss: 0.2104 - categorical_accuracy: 0.9358
44576/60000 [=====================>........] - ETA: 26s - loss: 0.2103 - categorical_accuracy: 0.9358
44608/60000 [=====================>........] - ETA: 26s - loss: 0.2102 - categorical_accuracy: 0.9358
44640/60000 [=====================>........] - ETA: 26s - loss: 0.2101 - categorical_accuracy: 0.9359
44672/60000 [=====================>........] - ETA: 26s - loss: 0.2101 - categorical_accuracy: 0.9359
44704/60000 [=====================>........] - ETA: 26s - loss: 0.2100 - categorical_accuracy: 0.9359
44736/60000 [=====================>........] - ETA: 26s - loss: 0.2098 - categorical_accuracy: 0.9360
44768/60000 [=====================>........] - ETA: 26s - loss: 0.2097 - categorical_accuracy: 0.9360
44800/60000 [=====================>........] - ETA: 26s - loss: 0.2096 - categorical_accuracy: 0.9360
44832/60000 [=====================>........] - ETA: 26s - loss: 0.2096 - categorical_accuracy: 0.9361
44864/60000 [=====================>........] - ETA: 26s - loss: 0.2096 - categorical_accuracy: 0.9361
44896/60000 [=====================>........] - ETA: 26s - loss: 0.2096 - categorical_accuracy: 0.9360
44928/60000 [=====================>........] - ETA: 25s - loss: 0.2096 - categorical_accuracy: 0.9360
44960/60000 [=====================>........] - ETA: 25s - loss: 0.2095 - categorical_accuracy: 0.9361
44992/60000 [=====================>........] - ETA: 25s - loss: 0.2094 - categorical_accuracy: 0.9361
45024/60000 [=====================>........] - ETA: 25s - loss: 0.2093 - categorical_accuracy: 0.9361
45056/60000 [=====================>........] - ETA: 25s - loss: 0.2094 - categorical_accuracy: 0.9361
45088/60000 [=====================>........] - ETA: 25s - loss: 0.2094 - categorical_accuracy: 0.9360
45120/60000 [=====================>........] - ETA: 25s - loss: 0.2093 - categorical_accuracy: 0.9361
45152/60000 [=====================>........] - ETA: 25s - loss: 0.2093 - categorical_accuracy: 0.9361
45184/60000 [=====================>........] - ETA: 25s - loss: 0.2093 - categorical_accuracy: 0.9361
45216/60000 [=====================>........] - ETA: 25s - loss: 0.2092 - categorical_accuracy: 0.9361
45248/60000 [=====================>........] - ETA: 25s - loss: 0.2092 - categorical_accuracy: 0.9361
45280/60000 [=====================>........] - ETA: 25s - loss: 0.2091 - categorical_accuracy: 0.9361
45312/60000 [=====================>........] - ETA: 25s - loss: 0.2090 - categorical_accuracy: 0.9361
45344/60000 [=====================>........] - ETA: 25s - loss: 0.2089 - categorical_accuracy: 0.9361
45376/60000 [=====================>........] - ETA: 25s - loss: 0.2089 - categorical_accuracy: 0.9362
45408/60000 [=====================>........] - ETA: 25s - loss: 0.2087 - categorical_accuracy: 0.9362
45440/60000 [=====================>........] - ETA: 25s - loss: 0.2087 - categorical_accuracy: 0.9362
45472/60000 [=====================>........] - ETA: 25s - loss: 0.2086 - categorical_accuracy: 0.9362
45504/60000 [=====================>........] - ETA: 24s - loss: 0.2085 - categorical_accuracy: 0.9363
45536/60000 [=====================>........] - ETA: 24s - loss: 0.2084 - categorical_accuracy: 0.9363
45568/60000 [=====================>........] - ETA: 24s - loss: 0.2085 - categorical_accuracy: 0.9363
45600/60000 [=====================>........] - ETA: 24s - loss: 0.2084 - categorical_accuracy: 0.9364
45632/60000 [=====================>........] - ETA: 24s - loss: 0.2083 - categorical_accuracy: 0.9364
45664/60000 [=====================>........] - ETA: 24s - loss: 0.2082 - categorical_accuracy: 0.9364
45696/60000 [=====================>........] - ETA: 24s - loss: 0.2080 - categorical_accuracy: 0.9365
45728/60000 [=====================>........] - ETA: 24s - loss: 0.2080 - categorical_accuracy: 0.9365
45760/60000 [=====================>........] - ETA: 24s - loss: 0.2080 - categorical_accuracy: 0.9365
45792/60000 [=====================>........] - ETA: 24s - loss: 0.2079 - categorical_accuracy: 0.9365
45824/60000 [=====================>........] - ETA: 24s - loss: 0.2079 - categorical_accuracy: 0.9365
45856/60000 [=====================>........] - ETA: 24s - loss: 0.2078 - categorical_accuracy: 0.9366
45888/60000 [=====================>........] - ETA: 24s - loss: 0.2077 - categorical_accuracy: 0.9366
45920/60000 [=====================>........] - ETA: 24s - loss: 0.2076 - categorical_accuracy: 0.9366
45984/60000 [=====================>........] - ETA: 24s - loss: 0.2075 - categorical_accuracy: 0.9367
46016/60000 [======================>.......] - ETA: 24s - loss: 0.2073 - categorical_accuracy: 0.9367
46048/60000 [======================>.......] - ETA: 24s - loss: 0.2073 - categorical_accuracy: 0.9367
46080/60000 [======================>.......] - ETA: 24s - loss: 0.2073 - categorical_accuracy: 0.9367
46112/60000 [======================>.......] - ETA: 23s - loss: 0.2074 - categorical_accuracy: 0.9367
46144/60000 [======================>.......] - ETA: 23s - loss: 0.2073 - categorical_accuracy: 0.9367
46176/60000 [======================>.......] - ETA: 23s - loss: 0.2072 - categorical_accuracy: 0.9367
46208/60000 [======================>.......] - ETA: 23s - loss: 0.2072 - categorical_accuracy: 0.9367
46240/60000 [======================>.......] - ETA: 23s - loss: 0.2071 - categorical_accuracy: 0.9367
46272/60000 [======================>.......] - ETA: 23s - loss: 0.2071 - categorical_accuracy: 0.9367
46304/60000 [======================>.......] - ETA: 23s - loss: 0.2069 - categorical_accuracy: 0.9368
46336/60000 [======================>.......] - ETA: 23s - loss: 0.2068 - categorical_accuracy: 0.9368
46368/60000 [======================>.......] - ETA: 23s - loss: 0.2067 - categorical_accuracy: 0.9369
46400/60000 [======================>.......] - ETA: 23s - loss: 0.2067 - categorical_accuracy: 0.9369
46432/60000 [======================>.......] - ETA: 23s - loss: 0.2066 - categorical_accuracy: 0.9369
46464/60000 [======================>.......] - ETA: 23s - loss: 0.2066 - categorical_accuracy: 0.9369
46496/60000 [======================>.......] - ETA: 23s - loss: 0.2065 - categorical_accuracy: 0.9369
46528/60000 [======================>.......] - ETA: 23s - loss: 0.2064 - categorical_accuracy: 0.9369
46560/60000 [======================>.......] - ETA: 23s - loss: 0.2063 - categorical_accuracy: 0.9369
46592/60000 [======================>.......] - ETA: 23s - loss: 0.2064 - categorical_accuracy: 0.9369
46624/60000 [======================>.......] - ETA: 23s - loss: 0.2064 - categorical_accuracy: 0.9369
46656/60000 [======================>.......] - ETA: 23s - loss: 0.2063 - categorical_accuracy: 0.9370
46688/60000 [======================>.......] - ETA: 22s - loss: 0.2062 - categorical_accuracy: 0.9370
46720/60000 [======================>.......] - ETA: 22s - loss: 0.2061 - categorical_accuracy: 0.9370
46752/60000 [======================>.......] - ETA: 22s - loss: 0.2060 - categorical_accuracy: 0.9371
46784/60000 [======================>.......] - ETA: 22s - loss: 0.2059 - categorical_accuracy: 0.9371
46816/60000 [======================>.......] - ETA: 22s - loss: 0.2058 - categorical_accuracy: 0.9371
46848/60000 [======================>.......] - ETA: 22s - loss: 0.2058 - categorical_accuracy: 0.9371
46880/60000 [======================>.......] - ETA: 22s - loss: 0.2058 - categorical_accuracy: 0.9371
46912/60000 [======================>.......] - ETA: 22s - loss: 0.2057 - categorical_accuracy: 0.9372
46944/60000 [======================>.......] - ETA: 22s - loss: 0.2056 - categorical_accuracy: 0.9372
46976/60000 [======================>.......] - ETA: 22s - loss: 0.2055 - categorical_accuracy: 0.9372
47008/60000 [======================>.......] - ETA: 22s - loss: 0.2054 - categorical_accuracy: 0.9373
47040/60000 [======================>.......] - ETA: 22s - loss: 0.2052 - categorical_accuracy: 0.9373
47072/60000 [======================>.......] - ETA: 22s - loss: 0.2052 - categorical_accuracy: 0.9373
47104/60000 [======================>.......] - ETA: 22s - loss: 0.2051 - categorical_accuracy: 0.9374
47136/60000 [======================>.......] - ETA: 22s - loss: 0.2051 - categorical_accuracy: 0.9374
47168/60000 [======================>.......] - ETA: 22s - loss: 0.2050 - categorical_accuracy: 0.9374
47200/60000 [======================>.......] - ETA: 22s - loss: 0.2049 - categorical_accuracy: 0.9374
47232/60000 [======================>.......] - ETA: 22s - loss: 0.2048 - categorical_accuracy: 0.9375
47264/60000 [======================>.......] - ETA: 21s - loss: 0.2048 - categorical_accuracy: 0.9375
47296/60000 [======================>.......] - ETA: 21s - loss: 0.2047 - categorical_accuracy: 0.9375
47328/60000 [======================>.......] - ETA: 21s - loss: 0.2045 - categorical_accuracy: 0.9375
47360/60000 [======================>.......] - ETA: 21s - loss: 0.2049 - categorical_accuracy: 0.9374
47392/60000 [======================>.......] - ETA: 21s - loss: 0.2048 - categorical_accuracy: 0.9374
47424/60000 [======================>.......] - ETA: 21s - loss: 0.2048 - categorical_accuracy: 0.9375
47456/60000 [======================>.......] - ETA: 21s - loss: 0.2047 - categorical_accuracy: 0.9375
47488/60000 [======================>.......] - ETA: 21s - loss: 0.2046 - categorical_accuracy: 0.9375
47520/60000 [======================>.......] - ETA: 21s - loss: 0.2046 - categorical_accuracy: 0.9375
47552/60000 [======================>.......] - ETA: 21s - loss: 0.2045 - categorical_accuracy: 0.9375
47584/60000 [======================>.......] - ETA: 21s - loss: 0.2044 - categorical_accuracy: 0.9376
47616/60000 [======================>.......] - ETA: 21s - loss: 0.2043 - categorical_accuracy: 0.9376
47648/60000 [======================>.......] - ETA: 21s - loss: 0.2042 - categorical_accuracy: 0.9376
47680/60000 [======================>.......] - ETA: 21s - loss: 0.2042 - categorical_accuracy: 0.9376
47712/60000 [======================>.......] - ETA: 21s - loss: 0.2041 - categorical_accuracy: 0.9377
47744/60000 [======================>.......] - ETA: 21s - loss: 0.2039 - categorical_accuracy: 0.9377
47776/60000 [======================>.......] - ETA: 21s - loss: 0.2041 - categorical_accuracy: 0.9377
47808/60000 [======================>.......] - ETA: 21s - loss: 0.2040 - categorical_accuracy: 0.9377
47840/60000 [======================>.......] - ETA: 20s - loss: 0.2038 - categorical_accuracy: 0.9378
47872/60000 [======================>.......] - ETA: 20s - loss: 0.2037 - categorical_accuracy: 0.9378
47904/60000 [======================>.......] - ETA: 20s - loss: 0.2036 - categorical_accuracy: 0.9378
47936/60000 [======================>.......] - ETA: 20s - loss: 0.2035 - categorical_accuracy: 0.9379
47968/60000 [======================>.......] - ETA: 20s - loss: 0.2034 - categorical_accuracy: 0.9379
48000/60000 [=======================>......] - ETA: 20s - loss: 0.2033 - categorical_accuracy: 0.9380
48032/60000 [=======================>......] - ETA: 20s - loss: 0.2031 - categorical_accuracy: 0.9380
48064/60000 [=======================>......] - ETA: 20s - loss: 0.2030 - categorical_accuracy: 0.9380
48096/60000 [=======================>......] - ETA: 20s - loss: 0.2029 - categorical_accuracy: 0.9381
48128/60000 [=======================>......] - ETA: 20s - loss: 0.2029 - categorical_accuracy: 0.9381
48160/60000 [=======================>......] - ETA: 20s - loss: 0.2028 - categorical_accuracy: 0.9381
48192/60000 [=======================>......] - ETA: 20s - loss: 0.2028 - categorical_accuracy: 0.9381
48224/60000 [=======================>......] - ETA: 20s - loss: 0.2027 - categorical_accuracy: 0.9381
48256/60000 [=======================>......] - ETA: 20s - loss: 0.2026 - categorical_accuracy: 0.9381
48288/60000 [=======================>......] - ETA: 20s - loss: 0.2025 - categorical_accuracy: 0.9382
48320/60000 [=======================>......] - ETA: 20s - loss: 0.2023 - categorical_accuracy: 0.9382
48352/60000 [=======================>......] - ETA: 20s - loss: 0.2022 - categorical_accuracy: 0.9383
48384/60000 [=======================>......] - ETA: 20s - loss: 0.2021 - categorical_accuracy: 0.9383
48416/60000 [=======================>......] - ETA: 19s - loss: 0.2020 - categorical_accuracy: 0.9383
48448/60000 [=======================>......] - ETA: 19s - loss: 0.2020 - categorical_accuracy: 0.9383
48480/60000 [=======================>......] - ETA: 19s - loss: 0.2019 - categorical_accuracy: 0.9384
48512/60000 [=======================>......] - ETA: 19s - loss: 0.2018 - categorical_accuracy: 0.9384
48544/60000 [=======================>......] - ETA: 19s - loss: 0.2017 - categorical_accuracy: 0.9384
48576/60000 [=======================>......] - ETA: 19s - loss: 0.2016 - categorical_accuracy: 0.9384
48608/60000 [=======================>......] - ETA: 19s - loss: 0.2016 - categorical_accuracy: 0.9384
48640/60000 [=======================>......] - ETA: 19s - loss: 0.2016 - categorical_accuracy: 0.9384
48672/60000 [=======================>......] - ETA: 19s - loss: 0.2015 - categorical_accuracy: 0.9385
48704/60000 [=======================>......] - ETA: 19s - loss: 0.2015 - categorical_accuracy: 0.9385
48736/60000 [=======================>......] - ETA: 19s - loss: 0.2014 - categorical_accuracy: 0.9385
48768/60000 [=======================>......] - ETA: 19s - loss: 0.2013 - categorical_accuracy: 0.9385
48800/60000 [=======================>......] - ETA: 19s - loss: 0.2013 - categorical_accuracy: 0.9386
48832/60000 [=======================>......] - ETA: 19s - loss: 0.2012 - categorical_accuracy: 0.9386
48864/60000 [=======================>......] - ETA: 19s - loss: 0.2011 - categorical_accuracy: 0.9386
48896/60000 [=======================>......] - ETA: 19s - loss: 0.2010 - categorical_accuracy: 0.9386
48928/60000 [=======================>......] - ETA: 19s - loss: 0.2010 - categorical_accuracy: 0.9386
48960/60000 [=======================>......] - ETA: 19s - loss: 0.2009 - categorical_accuracy: 0.9386
48992/60000 [=======================>......] - ETA: 18s - loss: 0.2009 - categorical_accuracy: 0.9386
49024/60000 [=======================>......] - ETA: 18s - loss: 0.2008 - categorical_accuracy: 0.9386
49056/60000 [=======================>......] - ETA: 18s - loss: 0.2008 - categorical_accuracy: 0.9386
49088/60000 [=======================>......] - ETA: 18s - loss: 0.2006 - categorical_accuracy: 0.9387
49120/60000 [=======================>......] - ETA: 18s - loss: 0.2005 - categorical_accuracy: 0.9387
49152/60000 [=======================>......] - ETA: 18s - loss: 0.2004 - categorical_accuracy: 0.9387
49184/60000 [=======================>......] - ETA: 18s - loss: 0.2004 - categorical_accuracy: 0.9388
49216/60000 [=======================>......] - ETA: 18s - loss: 0.2004 - categorical_accuracy: 0.9388
49248/60000 [=======================>......] - ETA: 18s - loss: 0.2004 - categorical_accuracy: 0.9388
49280/60000 [=======================>......] - ETA: 18s - loss: 0.2004 - categorical_accuracy: 0.9388
49312/60000 [=======================>......] - ETA: 18s - loss: 0.2004 - categorical_accuracy: 0.9388
49344/60000 [=======================>......] - ETA: 18s - loss: 0.2004 - categorical_accuracy: 0.9388
49376/60000 [=======================>......] - ETA: 18s - loss: 0.2003 - categorical_accuracy: 0.9388
49408/60000 [=======================>......] - ETA: 18s - loss: 0.2003 - categorical_accuracy: 0.9388
49440/60000 [=======================>......] - ETA: 18s - loss: 0.2002 - categorical_accuracy: 0.9389
49472/60000 [=======================>......] - ETA: 18s - loss: 0.2002 - categorical_accuracy: 0.9389
49504/60000 [=======================>......] - ETA: 18s - loss: 0.2000 - categorical_accuracy: 0.9389
49568/60000 [=======================>......] - ETA: 17s - loss: 0.1998 - categorical_accuracy: 0.9390
49600/60000 [=======================>......] - ETA: 17s - loss: 0.1997 - categorical_accuracy: 0.9390
49632/60000 [=======================>......] - ETA: 17s - loss: 0.1996 - categorical_accuracy: 0.9391
49664/60000 [=======================>......] - ETA: 17s - loss: 0.1995 - categorical_accuracy: 0.9391
49696/60000 [=======================>......] - ETA: 17s - loss: 0.1995 - categorical_accuracy: 0.9391
49760/60000 [=======================>......] - ETA: 17s - loss: 0.1994 - categorical_accuracy: 0.9391
49792/60000 [=======================>......] - ETA: 17s - loss: 0.1993 - categorical_accuracy: 0.9392
49824/60000 [=======================>......] - ETA: 17s - loss: 0.1991 - categorical_accuracy: 0.9392
49856/60000 [=======================>......] - ETA: 17s - loss: 0.1990 - categorical_accuracy: 0.9392
49888/60000 [=======================>......] - ETA: 17s - loss: 0.1990 - categorical_accuracy: 0.9393
49920/60000 [=======================>......] - ETA: 17s - loss: 0.1989 - categorical_accuracy: 0.9393
49952/60000 [=======================>......] - ETA: 17s - loss: 0.1988 - categorical_accuracy: 0.9393
49984/60000 [=======================>......] - ETA: 17s - loss: 0.1987 - categorical_accuracy: 0.9394
50016/60000 [========================>.....] - ETA: 17s - loss: 0.1986 - categorical_accuracy: 0.9394
50048/60000 [========================>.....] - ETA: 17s - loss: 0.1985 - categorical_accuracy: 0.9394
50112/60000 [========================>.....] - ETA: 17s - loss: 0.1983 - categorical_accuracy: 0.9395
50144/60000 [========================>.....] - ETA: 16s - loss: 0.1982 - categorical_accuracy: 0.9395
50176/60000 [========================>.....] - ETA: 16s - loss: 0.1982 - categorical_accuracy: 0.9395
50208/60000 [========================>.....] - ETA: 16s - loss: 0.1982 - categorical_accuracy: 0.9395
50240/60000 [========================>.....] - ETA: 16s - loss: 0.1981 - categorical_accuracy: 0.9396
50272/60000 [========================>.....] - ETA: 16s - loss: 0.1980 - categorical_accuracy: 0.9396
50304/60000 [========================>.....] - ETA: 16s - loss: 0.1979 - categorical_accuracy: 0.9396
50336/60000 [========================>.....] - ETA: 16s - loss: 0.1978 - categorical_accuracy: 0.9396
50368/60000 [========================>.....] - ETA: 16s - loss: 0.1977 - categorical_accuracy: 0.9397
50400/60000 [========================>.....] - ETA: 16s - loss: 0.1976 - categorical_accuracy: 0.9397
50432/60000 [========================>.....] - ETA: 16s - loss: 0.1976 - categorical_accuracy: 0.9397
50496/60000 [========================>.....] - ETA: 16s - loss: 0.1976 - categorical_accuracy: 0.9397
50528/60000 [========================>.....] - ETA: 16s - loss: 0.1975 - categorical_accuracy: 0.9397
50560/60000 [========================>.....] - ETA: 16s - loss: 0.1976 - categorical_accuracy: 0.9397
50592/60000 [========================>.....] - ETA: 16s - loss: 0.1977 - categorical_accuracy: 0.9397
50624/60000 [========================>.....] - ETA: 16s - loss: 0.1976 - categorical_accuracy: 0.9397
50656/60000 [========================>.....] - ETA: 16s - loss: 0.1975 - categorical_accuracy: 0.9398
50688/60000 [========================>.....] - ETA: 16s - loss: 0.1974 - categorical_accuracy: 0.9397
50720/60000 [========================>.....] - ETA: 15s - loss: 0.1973 - categorical_accuracy: 0.9398
50752/60000 [========================>.....] - ETA: 15s - loss: 0.1973 - categorical_accuracy: 0.9398
50784/60000 [========================>.....] - ETA: 15s - loss: 0.1972 - categorical_accuracy: 0.9398
50816/60000 [========================>.....] - ETA: 15s - loss: 0.1971 - categorical_accuracy: 0.9398
50848/60000 [========================>.....] - ETA: 15s - loss: 0.1970 - categorical_accuracy: 0.9399
50880/60000 [========================>.....] - ETA: 15s - loss: 0.1969 - categorical_accuracy: 0.9399
50912/60000 [========================>.....] - ETA: 15s - loss: 0.1968 - categorical_accuracy: 0.9399
50944/60000 [========================>.....] - ETA: 15s - loss: 0.1969 - categorical_accuracy: 0.9400
50976/60000 [========================>.....] - ETA: 15s - loss: 0.1968 - categorical_accuracy: 0.9400
51008/60000 [========================>.....] - ETA: 15s - loss: 0.1967 - categorical_accuracy: 0.9400
51040/60000 [========================>.....] - ETA: 15s - loss: 0.1966 - categorical_accuracy: 0.9400
51072/60000 [========================>.....] - ETA: 15s - loss: 0.1965 - categorical_accuracy: 0.9400
51104/60000 [========================>.....] - ETA: 15s - loss: 0.1964 - categorical_accuracy: 0.9401
51136/60000 [========================>.....] - ETA: 15s - loss: 0.1966 - categorical_accuracy: 0.9401
51168/60000 [========================>.....] - ETA: 15s - loss: 0.1965 - categorical_accuracy: 0.9401
51200/60000 [========================>.....] - ETA: 15s - loss: 0.1964 - categorical_accuracy: 0.9401
51232/60000 [========================>.....] - ETA: 15s - loss: 0.1963 - categorical_accuracy: 0.9402
51264/60000 [========================>.....] - ETA: 15s - loss: 0.1962 - categorical_accuracy: 0.9402
51296/60000 [========================>.....] - ETA: 14s - loss: 0.1962 - categorical_accuracy: 0.9402
51328/60000 [========================>.....] - ETA: 14s - loss: 0.1962 - categorical_accuracy: 0.9402
51360/60000 [========================>.....] - ETA: 14s - loss: 0.1961 - categorical_accuracy: 0.9402
51392/60000 [========================>.....] - ETA: 14s - loss: 0.1960 - categorical_accuracy: 0.9403
51424/60000 [========================>.....] - ETA: 14s - loss: 0.1960 - categorical_accuracy: 0.9402
51456/60000 [========================>.....] - ETA: 14s - loss: 0.1960 - categorical_accuracy: 0.9403
51488/60000 [========================>.....] - ETA: 14s - loss: 0.1960 - categorical_accuracy: 0.9403
51520/60000 [========================>.....] - ETA: 14s - loss: 0.1959 - categorical_accuracy: 0.9403
51552/60000 [========================>.....] - ETA: 14s - loss: 0.1958 - categorical_accuracy: 0.9403
51584/60000 [========================>.....] - ETA: 14s - loss: 0.1957 - categorical_accuracy: 0.9404
51616/60000 [========================>.....] - ETA: 14s - loss: 0.1956 - categorical_accuracy: 0.9404
51648/60000 [========================>.....] - ETA: 14s - loss: 0.1955 - categorical_accuracy: 0.9404
51680/60000 [========================>.....] - ETA: 14s - loss: 0.1956 - categorical_accuracy: 0.9404
51712/60000 [========================>.....] - ETA: 14s - loss: 0.1955 - categorical_accuracy: 0.9405
51744/60000 [========================>.....] - ETA: 14s - loss: 0.1954 - categorical_accuracy: 0.9405
51776/60000 [========================>.....] - ETA: 14s - loss: 0.1954 - categorical_accuracy: 0.9405
51808/60000 [========================>.....] - ETA: 14s - loss: 0.1953 - categorical_accuracy: 0.9405
51840/60000 [========================>.....] - ETA: 14s - loss: 0.1954 - categorical_accuracy: 0.9405
51872/60000 [========================>.....] - ETA: 13s - loss: 0.1953 - categorical_accuracy: 0.9405
51904/60000 [========================>.....] - ETA: 13s - loss: 0.1952 - categorical_accuracy: 0.9406
51936/60000 [========================>.....] - ETA: 13s - loss: 0.1952 - categorical_accuracy: 0.9406
51968/60000 [========================>.....] - ETA: 13s - loss: 0.1951 - categorical_accuracy: 0.9406
52000/60000 [=========================>....] - ETA: 13s - loss: 0.1950 - categorical_accuracy: 0.9406
52032/60000 [=========================>....] - ETA: 13s - loss: 0.1950 - categorical_accuracy: 0.9406
52064/60000 [=========================>....] - ETA: 13s - loss: 0.1949 - categorical_accuracy: 0.9406
52096/60000 [=========================>....] - ETA: 13s - loss: 0.1948 - categorical_accuracy: 0.9407
52128/60000 [=========================>....] - ETA: 13s - loss: 0.1947 - categorical_accuracy: 0.9407
52160/60000 [=========================>....] - ETA: 13s - loss: 0.1946 - categorical_accuracy: 0.9407
52192/60000 [=========================>....] - ETA: 13s - loss: 0.1945 - categorical_accuracy: 0.9408
52224/60000 [=========================>....] - ETA: 13s - loss: 0.1945 - categorical_accuracy: 0.9408
52256/60000 [=========================>....] - ETA: 13s - loss: 0.1944 - categorical_accuracy: 0.9408
52288/60000 [=========================>....] - ETA: 13s - loss: 0.1945 - categorical_accuracy: 0.9408
52320/60000 [=========================>....] - ETA: 13s - loss: 0.1944 - categorical_accuracy: 0.9408
52352/60000 [=========================>....] - ETA: 13s - loss: 0.1943 - categorical_accuracy: 0.9408
52384/60000 [=========================>....] - ETA: 13s - loss: 0.1943 - categorical_accuracy: 0.9408
52416/60000 [=========================>....] - ETA: 13s - loss: 0.1942 - categorical_accuracy: 0.9409
52448/60000 [=========================>....] - ETA: 12s - loss: 0.1941 - categorical_accuracy: 0.9409
52480/60000 [=========================>....] - ETA: 12s - loss: 0.1940 - categorical_accuracy: 0.9409
52512/60000 [=========================>....] - ETA: 12s - loss: 0.1939 - categorical_accuracy: 0.9409
52544/60000 [=========================>....] - ETA: 12s - loss: 0.1939 - categorical_accuracy: 0.9410
52576/60000 [=========================>....] - ETA: 12s - loss: 0.1939 - categorical_accuracy: 0.9409
52608/60000 [=========================>....] - ETA: 12s - loss: 0.1939 - categorical_accuracy: 0.9410
52640/60000 [=========================>....] - ETA: 12s - loss: 0.1938 - categorical_accuracy: 0.9410
52704/60000 [=========================>....] - ETA: 12s - loss: 0.1936 - categorical_accuracy: 0.9410
52736/60000 [=========================>....] - ETA: 12s - loss: 0.1937 - categorical_accuracy: 0.9410
52768/60000 [=========================>....] - ETA: 12s - loss: 0.1937 - categorical_accuracy: 0.9410
52800/60000 [=========================>....] - ETA: 12s - loss: 0.1936 - categorical_accuracy: 0.9411
52832/60000 [=========================>....] - ETA: 12s - loss: 0.1935 - categorical_accuracy: 0.9411
52864/60000 [=========================>....] - ETA: 12s - loss: 0.1934 - categorical_accuracy: 0.9411
52896/60000 [=========================>....] - ETA: 12s - loss: 0.1934 - categorical_accuracy: 0.9411
52928/60000 [=========================>....] - ETA: 12s - loss: 0.1933 - categorical_accuracy: 0.9411
52960/60000 [=========================>....] - ETA: 12s - loss: 0.1932 - categorical_accuracy: 0.9411
52992/60000 [=========================>....] - ETA: 12s - loss: 0.1932 - categorical_accuracy: 0.9411
53024/60000 [=========================>....] - ETA: 12s - loss: 0.1931 - categorical_accuracy: 0.9412
53056/60000 [=========================>....] - ETA: 11s - loss: 0.1931 - categorical_accuracy: 0.9412
53088/60000 [=========================>....] - ETA: 11s - loss: 0.1930 - categorical_accuracy: 0.9412
53120/60000 [=========================>....] - ETA: 11s - loss: 0.1930 - categorical_accuracy: 0.9412
53152/60000 [=========================>....] - ETA: 11s - loss: 0.1930 - categorical_accuracy: 0.9412
53184/60000 [=========================>....] - ETA: 11s - loss: 0.1929 - categorical_accuracy: 0.9412
53216/60000 [=========================>....] - ETA: 11s - loss: 0.1928 - categorical_accuracy: 0.9412
53248/60000 [=========================>....] - ETA: 11s - loss: 0.1927 - categorical_accuracy: 0.9413
53280/60000 [=========================>....] - ETA: 11s - loss: 0.1926 - categorical_accuracy: 0.9413
53312/60000 [=========================>....] - ETA: 11s - loss: 0.1925 - categorical_accuracy: 0.9413
53344/60000 [=========================>....] - ETA: 11s - loss: 0.1926 - categorical_accuracy: 0.9413
53376/60000 [=========================>....] - ETA: 11s - loss: 0.1925 - categorical_accuracy: 0.9413
53408/60000 [=========================>....] - ETA: 11s - loss: 0.1924 - categorical_accuracy: 0.9414
53440/60000 [=========================>....] - ETA: 11s - loss: 0.1927 - categorical_accuracy: 0.9414
53472/60000 [=========================>....] - ETA: 11s - loss: 0.1926 - categorical_accuracy: 0.9414
53504/60000 [=========================>....] - ETA: 11s - loss: 0.1927 - categorical_accuracy: 0.9414
53536/60000 [=========================>....] - ETA: 11s - loss: 0.1927 - categorical_accuracy: 0.9414
53568/60000 [=========================>....] - ETA: 11s - loss: 0.1926 - categorical_accuracy: 0.9414
53600/60000 [=========================>....] - ETA: 11s - loss: 0.1925 - categorical_accuracy: 0.9414
53632/60000 [=========================>....] - ETA: 10s - loss: 0.1924 - categorical_accuracy: 0.9415
53664/60000 [=========================>....] - ETA: 10s - loss: 0.1924 - categorical_accuracy: 0.9415
53696/60000 [=========================>....] - ETA: 10s - loss: 0.1923 - categorical_accuracy: 0.9415
53728/60000 [=========================>....] - ETA: 10s - loss: 0.1922 - categorical_accuracy: 0.9415
53760/60000 [=========================>....] - ETA: 10s - loss: 0.1922 - categorical_accuracy: 0.9415
53792/60000 [=========================>....] - ETA: 10s - loss: 0.1923 - categorical_accuracy: 0.9415
53824/60000 [=========================>....] - ETA: 10s - loss: 0.1923 - categorical_accuracy: 0.9415
53856/60000 [=========================>....] - ETA: 10s - loss: 0.1922 - categorical_accuracy: 0.9415
53888/60000 [=========================>....] - ETA: 10s - loss: 0.1923 - categorical_accuracy: 0.9415
53920/60000 [=========================>....] - ETA: 10s - loss: 0.1922 - categorical_accuracy: 0.9415
53952/60000 [=========================>....] - ETA: 10s - loss: 0.1922 - categorical_accuracy: 0.9415
53984/60000 [=========================>....] - ETA: 10s - loss: 0.1921 - categorical_accuracy: 0.9415
54016/60000 [==========================>...] - ETA: 10s - loss: 0.1921 - categorical_accuracy: 0.9415
54048/60000 [==========================>...] - ETA: 10s - loss: 0.1920 - categorical_accuracy: 0.9416
54080/60000 [==========================>...] - ETA: 10s - loss: 0.1920 - categorical_accuracy: 0.9416
54112/60000 [==========================>...] - ETA: 10s - loss: 0.1919 - categorical_accuracy: 0.9416
54144/60000 [==========================>...] - ETA: 10s - loss: 0.1918 - categorical_accuracy: 0.9416
54176/60000 [==========================>...] - ETA: 10s - loss: 0.1917 - categorical_accuracy: 0.9417
54208/60000 [==========================>...] - ETA: 9s - loss: 0.1916 - categorical_accuracy: 0.9417 
54240/60000 [==========================>...] - ETA: 9s - loss: 0.1915 - categorical_accuracy: 0.9417
54272/60000 [==========================>...] - ETA: 9s - loss: 0.1915 - categorical_accuracy: 0.9417
54304/60000 [==========================>...] - ETA: 9s - loss: 0.1914 - categorical_accuracy: 0.9417
54336/60000 [==========================>...] - ETA: 9s - loss: 0.1916 - categorical_accuracy: 0.9417
54368/60000 [==========================>...] - ETA: 9s - loss: 0.1916 - categorical_accuracy: 0.9417
54400/60000 [==========================>...] - ETA: 9s - loss: 0.1916 - categorical_accuracy: 0.9417
54432/60000 [==========================>...] - ETA: 9s - loss: 0.1917 - categorical_accuracy: 0.9417
54464/60000 [==========================>...] - ETA: 9s - loss: 0.1916 - categorical_accuracy: 0.9417
54496/60000 [==========================>...] - ETA: 9s - loss: 0.1915 - categorical_accuracy: 0.9417
54528/60000 [==========================>...] - ETA: 9s - loss: 0.1916 - categorical_accuracy: 0.9417
54560/60000 [==========================>...] - ETA: 9s - loss: 0.1915 - categorical_accuracy: 0.9418
54592/60000 [==========================>...] - ETA: 9s - loss: 0.1914 - categorical_accuracy: 0.9418
54624/60000 [==========================>...] - ETA: 9s - loss: 0.1913 - categorical_accuracy: 0.9418
54656/60000 [==========================>...] - ETA: 9s - loss: 0.1912 - categorical_accuracy: 0.9419
54688/60000 [==========================>...] - ETA: 9s - loss: 0.1911 - categorical_accuracy: 0.9419
54720/60000 [==========================>...] - ETA: 9s - loss: 0.1911 - categorical_accuracy: 0.9419
54752/60000 [==========================>...] - ETA: 9s - loss: 0.1911 - categorical_accuracy: 0.9419
54784/60000 [==========================>...] - ETA: 8s - loss: 0.1911 - categorical_accuracy: 0.9419
54816/60000 [==========================>...] - ETA: 8s - loss: 0.1910 - categorical_accuracy: 0.9419
54848/60000 [==========================>...] - ETA: 8s - loss: 0.1910 - categorical_accuracy: 0.9419
54880/60000 [==========================>...] - ETA: 8s - loss: 0.1911 - categorical_accuracy: 0.9419
54912/60000 [==========================>...] - ETA: 8s - loss: 0.1910 - categorical_accuracy: 0.9419
54944/60000 [==========================>...] - ETA: 8s - loss: 0.1909 - categorical_accuracy: 0.9420
54976/60000 [==========================>...] - ETA: 8s - loss: 0.1909 - categorical_accuracy: 0.9420
55040/60000 [==========================>...] - ETA: 8s - loss: 0.1908 - categorical_accuracy: 0.9420
55072/60000 [==========================>...] - ETA: 8s - loss: 0.1908 - categorical_accuracy: 0.9420
55104/60000 [==========================>...] - ETA: 8s - loss: 0.1907 - categorical_accuracy: 0.9420
55136/60000 [==========================>...] - ETA: 8s - loss: 0.1907 - categorical_accuracy: 0.9420
55168/60000 [==========================>...] - ETA: 8s - loss: 0.1906 - categorical_accuracy: 0.9420
55200/60000 [==========================>...] - ETA: 8s - loss: 0.1906 - categorical_accuracy: 0.9420
55232/60000 [==========================>...] - ETA: 8s - loss: 0.1905 - categorical_accuracy: 0.9420
55264/60000 [==========================>...] - ETA: 8s - loss: 0.1904 - categorical_accuracy: 0.9421
55296/60000 [==========================>...] - ETA: 8s - loss: 0.1903 - categorical_accuracy: 0.9421
55328/60000 [==========================>...] - ETA: 8s - loss: 0.1902 - categorical_accuracy: 0.9421
55360/60000 [==========================>...] - ETA: 7s - loss: 0.1902 - categorical_accuracy: 0.9422
55392/60000 [==========================>...] - ETA: 7s - loss: 0.1902 - categorical_accuracy: 0.9422
55424/60000 [==========================>...] - ETA: 7s - loss: 0.1901 - categorical_accuracy: 0.9422
55456/60000 [==========================>...] - ETA: 7s - loss: 0.1901 - categorical_accuracy: 0.9422
55488/60000 [==========================>...] - ETA: 7s - loss: 0.1901 - categorical_accuracy: 0.9422
55520/60000 [==========================>...] - ETA: 7s - loss: 0.1900 - categorical_accuracy: 0.9422
55552/60000 [==========================>...] - ETA: 7s - loss: 0.1900 - categorical_accuracy: 0.9422
55584/60000 [==========================>...] - ETA: 7s - loss: 0.1899 - categorical_accuracy: 0.9422
55616/60000 [==========================>...] - ETA: 7s - loss: 0.1898 - categorical_accuracy: 0.9422
55648/60000 [==========================>...] - ETA: 7s - loss: 0.1900 - categorical_accuracy: 0.9422
55680/60000 [==========================>...] - ETA: 7s - loss: 0.1899 - categorical_accuracy: 0.9422
55712/60000 [==========================>...] - ETA: 7s - loss: 0.1898 - categorical_accuracy: 0.9423
55744/60000 [==========================>...] - ETA: 7s - loss: 0.1897 - categorical_accuracy: 0.9423
55776/60000 [==========================>...] - ETA: 7s - loss: 0.1896 - categorical_accuracy: 0.9423
55808/60000 [==========================>...] - ETA: 7s - loss: 0.1896 - categorical_accuracy: 0.9424
55840/60000 [==========================>...] - ETA: 7s - loss: 0.1896 - categorical_accuracy: 0.9423
55872/60000 [==========================>...] - ETA: 7s - loss: 0.1895 - categorical_accuracy: 0.9424
55904/60000 [==========================>...] - ETA: 7s - loss: 0.1894 - categorical_accuracy: 0.9424
55936/60000 [==========================>...] - ETA: 6s - loss: 0.1894 - categorical_accuracy: 0.9424
55968/60000 [==========================>...] - ETA: 6s - loss: 0.1894 - categorical_accuracy: 0.9424
56000/60000 [===========================>..] - ETA: 6s - loss: 0.1893 - categorical_accuracy: 0.9424
56032/60000 [===========================>..] - ETA: 6s - loss: 0.1892 - categorical_accuracy: 0.9425
56064/60000 [===========================>..] - ETA: 6s - loss: 0.1891 - categorical_accuracy: 0.9425
56096/60000 [===========================>..] - ETA: 6s - loss: 0.1890 - categorical_accuracy: 0.9425
56128/60000 [===========================>..] - ETA: 6s - loss: 0.1890 - categorical_accuracy: 0.9426
56160/60000 [===========================>..] - ETA: 6s - loss: 0.1889 - categorical_accuracy: 0.9426
56192/60000 [===========================>..] - ETA: 6s - loss: 0.1888 - categorical_accuracy: 0.9426
56224/60000 [===========================>..] - ETA: 6s - loss: 0.1888 - categorical_accuracy: 0.9426
56256/60000 [===========================>..] - ETA: 6s - loss: 0.1887 - categorical_accuracy: 0.9426
56288/60000 [===========================>..] - ETA: 6s - loss: 0.1886 - categorical_accuracy: 0.9427
56320/60000 [===========================>..] - ETA: 6s - loss: 0.1886 - categorical_accuracy: 0.9427
56352/60000 [===========================>..] - ETA: 6s - loss: 0.1885 - categorical_accuracy: 0.9427
56384/60000 [===========================>..] - ETA: 6s - loss: 0.1885 - categorical_accuracy: 0.9427
56416/60000 [===========================>..] - ETA: 6s - loss: 0.1885 - categorical_accuracy: 0.9428
56448/60000 [===========================>..] - ETA: 6s - loss: 0.1884 - categorical_accuracy: 0.9428
56480/60000 [===========================>..] - ETA: 6s - loss: 0.1884 - categorical_accuracy: 0.9428
56512/60000 [===========================>..] - ETA: 6s - loss: 0.1884 - categorical_accuracy: 0.9428
56544/60000 [===========================>..] - ETA: 5s - loss: 0.1883 - categorical_accuracy: 0.9428
56576/60000 [===========================>..] - ETA: 5s - loss: 0.1883 - categorical_accuracy: 0.9428
56608/60000 [===========================>..] - ETA: 5s - loss: 0.1883 - categorical_accuracy: 0.9428
56640/60000 [===========================>..] - ETA: 5s - loss: 0.1882 - categorical_accuracy: 0.9428
56672/60000 [===========================>..] - ETA: 5s - loss: 0.1881 - categorical_accuracy: 0.9428
56704/60000 [===========================>..] - ETA: 5s - loss: 0.1881 - categorical_accuracy: 0.9428
56736/60000 [===========================>..] - ETA: 5s - loss: 0.1880 - categorical_accuracy: 0.9429
56768/60000 [===========================>..] - ETA: 5s - loss: 0.1880 - categorical_accuracy: 0.9429
56800/60000 [===========================>..] - ETA: 5s - loss: 0.1880 - categorical_accuracy: 0.9429
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1880 - categorical_accuracy: 0.9429
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1879 - categorical_accuracy: 0.9429
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1879 - categorical_accuracy: 0.9429
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1879 - categorical_accuracy: 0.9429
56960/60000 [===========================>..] - ETA: 5s - loss: 0.1878 - categorical_accuracy: 0.9429
56992/60000 [===========================>..] - ETA: 5s - loss: 0.1877 - categorical_accuracy: 0.9429
57024/60000 [===========================>..] - ETA: 5s - loss: 0.1876 - categorical_accuracy: 0.9430
57056/60000 [===========================>..] - ETA: 5s - loss: 0.1877 - categorical_accuracy: 0.9429
57088/60000 [===========================>..] - ETA: 5s - loss: 0.1877 - categorical_accuracy: 0.9429
57120/60000 [===========================>..] - ETA: 4s - loss: 0.1876 - categorical_accuracy: 0.9429
57152/60000 [===========================>..] - ETA: 4s - loss: 0.1875 - categorical_accuracy: 0.9430
57184/60000 [===========================>..] - ETA: 4s - loss: 0.1874 - categorical_accuracy: 0.9430
57216/60000 [===========================>..] - ETA: 4s - loss: 0.1874 - categorical_accuracy: 0.9430
57248/60000 [===========================>..] - ETA: 4s - loss: 0.1873 - categorical_accuracy: 0.9430
57312/60000 [===========================>..] - ETA: 4s - loss: 0.1872 - categorical_accuracy: 0.9431
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1871 - categorical_accuracy: 0.9431
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1871 - categorical_accuracy: 0.9431
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1870 - categorical_accuracy: 0.9431
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1869 - categorical_accuracy: 0.9431
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1869 - categorical_accuracy: 0.9432
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1869 - categorical_accuracy: 0.9432
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1868 - categorical_accuracy: 0.9431
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1868 - categorical_accuracy: 0.9431
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1867 - categorical_accuracy: 0.9432
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1867 - categorical_accuracy: 0.9432
57664/60000 [===========================>..] - ETA: 4s - loss: 0.1866 - categorical_accuracy: 0.9432
57696/60000 [===========================>..] - ETA: 3s - loss: 0.1866 - categorical_accuracy: 0.9432
57728/60000 [===========================>..] - ETA: 3s - loss: 0.1865 - categorical_accuracy: 0.9432
57760/60000 [===========================>..] - ETA: 3s - loss: 0.1866 - categorical_accuracy: 0.9432
57792/60000 [===========================>..] - ETA: 3s - loss: 0.1865 - categorical_accuracy: 0.9433
57824/60000 [===========================>..] - ETA: 3s - loss: 0.1864 - categorical_accuracy: 0.9433
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1863 - categorical_accuracy: 0.9433
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1863 - categorical_accuracy: 0.9433
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1862 - categorical_accuracy: 0.9433
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1861 - categorical_accuracy: 0.9434
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1860 - categorical_accuracy: 0.9434
58016/60000 [============================>.] - ETA: 3s - loss: 0.1860 - categorical_accuracy: 0.9434
58048/60000 [============================>.] - ETA: 3s - loss: 0.1859 - categorical_accuracy: 0.9434
58080/60000 [============================>.] - ETA: 3s - loss: 0.1858 - categorical_accuracy: 0.9434
58112/60000 [============================>.] - ETA: 3s - loss: 0.1857 - categorical_accuracy: 0.9435
58144/60000 [============================>.] - ETA: 3s - loss: 0.1856 - categorical_accuracy: 0.9435
58176/60000 [============================>.] - ETA: 3s - loss: 0.1857 - categorical_accuracy: 0.9435
58208/60000 [============================>.] - ETA: 3s - loss: 0.1857 - categorical_accuracy: 0.9435
58240/60000 [============================>.] - ETA: 3s - loss: 0.1856 - categorical_accuracy: 0.9435
58272/60000 [============================>.] - ETA: 2s - loss: 0.1855 - categorical_accuracy: 0.9436
58304/60000 [============================>.] - ETA: 2s - loss: 0.1855 - categorical_accuracy: 0.9436
58336/60000 [============================>.] - ETA: 2s - loss: 0.1854 - categorical_accuracy: 0.9436
58368/60000 [============================>.] - ETA: 2s - loss: 0.1853 - categorical_accuracy: 0.9436
58400/60000 [============================>.] - ETA: 2s - loss: 0.1852 - categorical_accuracy: 0.9436
58432/60000 [============================>.] - ETA: 2s - loss: 0.1852 - categorical_accuracy: 0.9437
58464/60000 [============================>.] - ETA: 2s - loss: 0.1851 - categorical_accuracy: 0.9437
58496/60000 [============================>.] - ETA: 2s - loss: 0.1850 - categorical_accuracy: 0.9437
58528/60000 [============================>.] - ETA: 2s - loss: 0.1849 - categorical_accuracy: 0.9437
58560/60000 [============================>.] - ETA: 2s - loss: 0.1849 - categorical_accuracy: 0.9438
58592/60000 [============================>.] - ETA: 2s - loss: 0.1849 - categorical_accuracy: 0.9437
58624/60000 [============================>.] - ETA: 2s - loss: 0.1849 - categorical_accuracy: 0.9438
58656/60000 [============================>.] - ETA: 2s - loss: 0.1849 - categorical_accuracy: 0.9437
58688/60000 [============================>.] - ETA: 2s - loss: 0.1849 - categorical_accuracy: 0.9438
58720/60000 [============================>.] - ETA: 2s - loss: 0.1848 - categorical_accuracy: 0.9438
58752/60000 [============================>.] - ETA: 2s - loss: 0.1847 - categorical_accuracy: 0.9438
58784/60000 [============================>.] - ETA: 2s - loss: 0.1846 - categorical_accuracy: 0.9438
58816/60000 [============================>.] - ETA: 2s - loss: 0.1846 - categorical_accuracy: 0.9439
58848/60000 [============================>.] - ETA: 1s - loss: 0.1845 - categorical_accuracy: 0.9439
58880/60000 [============================>.] - ETA: 1s - loss: 0.1844 - categorical_accuracy: 0.9439
58912/60000 [============================>.] - ETA: 1s - loss: 0.1843 - categorical_accuracy: 0.9439
58944/60000 [============================>.] - ETA: 1s - loss: 0.1843 - categorical_accuracy: 0.9439
58976/60000 [============================>.] - ETA: 1s - loss: 0.1842 - categorical_accuracy: 0.9440
59008/60000 [============================>.] - ETA: 1s - loss: 0.1842 - categorical_accuracy: 0.9440
59040/60000 [============================>.] - ETA: 1s - loss: 0.1841 - categorical_accuracy: 0.9440
59072/60000 [============================>.] - ETA: 1s - loss: 0.1841 - categorical_accuracy: 0.9440
59104/60000 [============================>.] - ETA: 1s - loss: 0.1840 - categorical_accuracy: 0.9440
59136/60000 [============================>.] - ETA: 1s - loss: 0.1840 - categorical_accuracy: 0.9440
59168/60000 [============================>.] - ETA: 1s - loss: 0.1839 - categorical_accuracy: 0.9440
59232/60000 [============================>.] - ETA: 1s - loss: 0.1838 - categorical_accuracy: 0.9441
59264/60000 [============================>.] - ETA: 1s - loss: 0.1838 - categorical_accuracy: 0.9441
59296/60000 [============================>.] - ETA: 1s - loss: 0.1837 - categorical_accuracy: 0.9441
59328/60000 [============================>.] - ETA: 1s - loss: 0.1837 - categorical_accuracy: 0.9441
59360/60000 [============================>.] - ETA: 1s - loss: 0.1836 - categorical_accuracy: 0.9441
59392/60000 [============================>.] - ETA: 1s - loss: 0.1835 - categorical_accuracy: 0.9441
59424/60000 [============================>.] - ETA: 0s - loss: 0.1834 - categorical_accuracy: 0.9441
59456/60000 [============================>.] - ETA: 0s - loss: 0.1834 - categorical_accuracy: 0.9442
59488/60000 [============================>.] - ETA: 0s - loss: 0.1834 - categorical_accuracy: 0.9442
59520/60000 [============================>.] - ETA: 0s - loss: 0.1833 - categorical_accuracy: 0.9442
59552/60000 [============================>.] - ETA: 0s - loss: 0.1832 - categorical_accuracy: 0.9442
59584/60000 [============================>.] - ETA: 0s - loss: 0.1833 - categorical_accuracy: 0.9442
59616/60000 [============================>.] - ETA: 0s - loss: 0.1832 - categorical_accuracy: 0.9442
59648/60000 [============================>.] - ETA: 0s - loss: 0.1832 - categorical_accuracy: 0.9443
59680/60000 [============================>.] - ETA: 0s - loss: 0.1831 - categorical_accuracy: 0.9443
59712/60000 [============================>.] - ETA: 0s - loss: 0.1832 - categorical_accuracy: 0.9443
59744/60000 [============================>.] - ETA: 0s - loss: 0.1831 - categorical_accuracy: 0.9443
59776/60000 [============================>.] - ETA: 0s - loss: 0.1830 - categorical_accuracy: 0.9443
59808/60000 [============================>.] - ETA: 0s - loss: 0.1829 - categorical_accuracy: 0.9443
59840/60000 [============================>.] - ETA: 0s - loss: 0.1829 - categorical_accuracy: 0.9444
59872/60000 [============================>.] - ETA: 0s - loss: 0.1828 - categorical_accuracy: 0.9444
59904/60000 [============================>.] - ETA: 0s - loss: 0.1829 - categorical_accuracy: 0.9444
59936/60000 [============================>.] - ETA: 0s - loss: 0.1828 - categorical_accuracy: 0.9444
59968/60000 [============================>.] - ETA: 0s - loss: 0.1827 - categorical_accuracy: 0.9444
60000/60000 [==============================] - 107s 2ms/step - loss: 0.1827 - categorical_accuracy: 0.9444 - val_loss: 0.0490 - val_categorical_accuracy: 0.9838

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 17s
  160/10000 [..............................] - ETA: 6s 
  320/10000 [..............................] - ETA: 5s
  480/10000 [>.............................] - ETA: 4s
  640/10000 [>.............................] - ETA: 4s
  832/10000 [=>............................] - ETA: 3s
  992/10000 [=>............................] - ETA: 3s
 1152/10000 [==>...........................] - ETA: 3s
 1312/10000 [==>...........................] - ETA: 3s
 1472/10000 [===>..........................] - ETA: 3s
 1632/10000 [===>..........................] - ETA: 3s
 1792/10000 [====>.........................] - ETA: 3s
 1952/10000 [====>.........................] - ETA: 2s
 2112/10000 [=====>........................] - ETA: 2s
 2272/10000 [=====>........................] - ETA: 2s
 2432/10000 [======>.......................] - ETA: 2s
 2592/10000 [======>.......................] - ETA: 2s
 2752/10000 [=======>......................] - ETA: 2s
 2912/10000 [=======>......................] - ETA: 2s
 3072/10000 [========>.....................] - ETA: 2s
 3264/10000 [========>.....................] - ETA: 2s
 3424/10000 [=========>....................] - ETA: 2s
 3616/10000 [=========>....................] - ETA: 2s
 3808/10000 [==========>...................] - ETA: 2s
 3968/10000 [==========>...................] - ETA: 2s
 4128/10000 [===========>..................] - ETA: 2s
 4288/10000 [===========>..................] - ETA: 1s
 4448/10000 [============>.................] - ETA: 1s
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
 7168/10000 [====================>.........] - ETA: 0s
 7328/10000 [====================>.........] - ETA: 0s
 7488/10000 [=====================>........] - ETA: 0s
 7648/10000 [=====================>........] - ETA: 0s
 7840/10000 [======================>.......] - ETA: 0s
 8000/10000 [=======================>......] - ETA: 0s
 8160/10000 [=======================>......] - ETA: 0s
 8352/10000 [========================>.....] - ETA: 0s
 8512/10000 [========================>.....] - ETA: 0s
 8672/10000 [=========================>....] - ETA: 0s
 8832/10000 [=========================>....] - ETA: 0s
 9024/10000 [==========================>...] - ETA: 0s
 9184/10000 [==========================>...] - ETA: 0s
 9344/10000 [===========================>..] - ETA: 0s
 9504/10000 [===========================>..] - ETA: 0s
 9664/10000 [===========================>..] - ETA: 0s
 9824/10000 [============================>.] - ETA: 0s
 9984/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 3s 346us/step
[[4.8217981e-09 2.1221840e-08 4.8589786e-08 ... 9.9999702e-01
  2.7961196e-09 7.8930526e-07]
 [7.5798591e-05 2.0618218e-05 9.9985576e-01 ... 1.0178781e-07
  3.5882589e-05 8.8276958e-10]
 [4.2425955e-07 9.9984062e-01 2.3105806e-05 ... 3.9718449e-05
  1.1421776e-05 5.3600860e-07]
 ...
 [6.9834849e-10 2.5617129e-07 1.8315951e-08 ... 2.9391604e-06
  1.2472903e-07 7.0626384e-06]
 [1.2640124e-06 5.9885558e-08 1.2322918e-08 ... 1.3291184e-07
  4.1680920e-04 1.9353891e-07]
 [4.8092274e-06 6.0298197e-07 2.9160290e-06 ... 1.7008031e-09
  1.8907116e-06 7.9418081e-09]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.04896668555822689, 'accuracy_test:': 0.9837999939918518}

  ('#### Save   #######################################################',) 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/charcnn/result'}

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   1d1320d..da833b9  master     -> origin/master
Updating 1d1320d..da833b9
Fast-forward
 .../20200514/list_log_dataloader_20200514.md       |    2 +-
 error_list/20200514/list_log_json_20200514.md      | 1146 ++++++-------
 error_list/20200514/list_log_jupyter_20200514.md   | 1788 ++++++++++----------
 .../20200514/list_log_pullrequest_20200514.md      |    2 +-
 error_list/20200514/list_log_testall_20200514.md   |  443 +++++
 5 files changed, 1918 insertions(+), 1463 deletions(-)
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
[master 503dfe6] ml_store
 1 file changed, 2014 insertions(+)
To github.com:arita37/mlmodels_store.git
   da833b9..503dfe6  master -> master





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
{'loss': 0.4734189249575138, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-14 12:33:55.009913: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
