
  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_all 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077

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
Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
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
[master 03f0077] ml_store
 1 file changed, 60 insertions(+)
 create mode 100644 log_testall/log_testall_2020-05-15-16-10_169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077.py
To github.com:arita37/mlmodels_store.git
   8b0bebd..03f0077  master -> master





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
[master 59a269e] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   03f0077..59a269e  master -> master





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
[master 1a51214] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   59a269e..1a51214  master -> master





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
weighted_sequence_layer_1 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         36          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         32          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-15 16:11:20.298230: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 16:11:20.317620: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-15 16:11:20.318389: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x556808a4d450 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 16:11:20.318408: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 263
Trainable params: 263
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 1ms/sample - loss: 0.2501 - binary_crossentropy: 0.6933 - val_loss: 0.2502 - val_binary_crossentropy: 0.6934

  #### metrics   #################################################### 
{'MSE': 0.24997271921622324}

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
weighted_sequence_layer_1 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         36          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         32          sparse_feature_2[0][0]           
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
Total params: 263
Trainable params: 263
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_max[0][0]               
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
Total params: 488
Trainable params: 488
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.3083 - binary_crossentropy: 2.1180500/500 [==============================] - 1s 1ms/sample - loss: 0.3174 - binary_crossentropy: 2.3440 - val_loss: 0.3034 - val_binary_crossentropy: 1.9703

  #### metrics   #################################################### 
{'MSE': 0.31018330986422277}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_max[0][0]               
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         36          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_2[0][0]           
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
Total params: 622
Trainable params: 622
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 2ms/sample - loss: 0.2531 - binary_crossentropy: 0.7523 - val_loss: 0.2506 - val_binary_crossentropy: 0.7203

  #### metrics   #################################################### 
{'MSE': 0.25161685478239376}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         36          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_2[0][0]           
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
Total params: 622
Trainable params: 622
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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         8           sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_2[0][0]           
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
Total params: 413
Trainable params: 413
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2620 - binary_crossentropy: 0.7214500/500 [==============================] - 1s 2ms/sample - loss: 0.2685 - binary_crossentropy: 0.7358 - val_loss: 0.2599 - val_binary_crossentropy: 0.7143

  #### metrics   #################################################### 
{'MSE': 0.2585574761431876}

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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         8           sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_2[0][0]           
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
Total params: 413
Trainable params: 413
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         6           sequence_max[0][0]               
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
Total params: 143
Trainable params: 143
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2763 - binary_crossentropy: 0.7582500/500 [==============================] - 1s 3ms/sample - loss: 0.2849 - binary_crossentropy: 0.7760 - val_loss: 0.2993 - val_binary_crossentropy: 0.8063

  #### metrics   #################################################### 
{'MSE': 0.28996362274891424}

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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         6           sequence_max[0][0]               
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
Total params: 143
Trainable params: 143
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-15 16:12:28.577247: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 16:12:28.579646: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 16:12:28.586548: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-15 16:12:28.595955: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-15 16:12:28.597542: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-15 16:12:28.599105: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 16:12:28.600580: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2490 - val_binary_crossentropy: 0.6911
2020-05-15 16:12:29.689739: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 16:12:29.691397: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 16:12:29.695192: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-15 16:12:29.703691: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-15 16:12:29.705147: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-15 16:12:29.706557: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 16:12:29.707746: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24852301575993016}

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
2020-05-15 16:12:49.920972: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 16:12:49.922216: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 16:12:49.925413: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-15 16:12:49.931109: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-15 16:12:49.932534: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-15 16:12:49.933505: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 16:12:49.934417: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2501 - val_binary_crossentropy: 0.6934
2020-05-15 16:12:51.222653: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 16:12:51.223670: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 16:12:51.225983: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-15 16:12:51.230926: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-15 16:12:51.231875: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-15 16:12:51.232669: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 16:12:51.233331: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2500889340947398}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-15 16:13:21.256006: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 16:13:21.260301: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 16:13:21.272932: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-15 16:13:21.294987: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-15 16:13:21.298757: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-15 16:13:21.302345: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 16:13:21.305782: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 4s 4s/sample - loss: 0.0414 - binary_crossentropy: 0.2276 - val_loss: 0.3499 - val_binary_crossentropy: 0.9483
2020-05-15 16:13:23.269342: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 16:13:23.274672: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 16:13:23.286117: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-15 16:13:23.308086: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-15 16:13:23.311752: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-15 16:13:23.315194: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 16:13:23.318628: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24454517849292282}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
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
Total params: 665
Trainable params: 665
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 5s - loss: 0.2525 - binary_crossentropy: 0.8297500/500 [==============================] - 4s 7ms/sample - loss: 0.2701 - binary_crossentropy: 0.8152 - val_loss: 0.2707 - val_binary_crossentropy: 0.8151

  #### metrics   #################################################### 
{'MSE': 0.2700916625912879}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
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
Total params: 665
Trainable params: 665
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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 2)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 2)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 2)         10          sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         18          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         14          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         16          sparse_feature_4[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_4[0][0]           
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
Total params: 266
Trainable params: 266
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 5s - loss: 0.2867 - binary_crossentropy: 1.1679500/500 [==============================] - 4s 8ms/sample - loss: 0.2714 - binary_crossentropy: 0.8226 - val_loss: 0.2653 - val_binary_crossentropy: 0.7512

  #### metrics   #################################################### 
{'MSE': 0.26395168703666577}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 2)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 2)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 2)         10          sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         18          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         14          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         16          sparse_feature_4[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_4[0][0]           
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
Total params: 266
Trainable params: 266
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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
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
Total params: 1,914
Trainable params: 1,914
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 5s - loss: 0.2664 - binary_crossentropy: 0.7269500/500 [==============================] - 4s 8ms/sample - loss: 0.2629 - binary_crossentropy: 0.7196 - val_loss: 0.2512 - val_binary_crossentropy: 0.6956

  #### metrics   #################################################### 
{'MSE': 0.2540262763265608}

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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
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
Total params: 1,914
Trainable params: 1,914
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
regionsequence_mean (InputLayer [(None, 1)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 4)]          0                                            
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
region_10sparse_seq_emb_regions (None, 2, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 1, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 4, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 2, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 4, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 2, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 4, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 2, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 4, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 2, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 4, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 2, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 4, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 2, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 4, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 2, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 4, 1)         7           regionsequence_max[0][0]         
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
100/500 [=====>........................] - ETA: 7s - loss: 0.2852 - binary_crossentropy: 0.7731500/500 [==============================] - 5s 10ms/sample - loss: 0.2752 - binary_crossentropy: 0.7484 - val_loss: 0.2655 - val_binary_crossentropy: 0.7538

  #### metrics   #################################################### 
{'MSE': 0.26961035224298635}

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
regionsequence_mean (InputLayer [(None, 1)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 4)]          0                                            
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
region_10sparse_seq_emb_regions (None, 2, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 1, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 4, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 2, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 4, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 2, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 4, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 2, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 4, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 2, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 4, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 2, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 4, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 2, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 4, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 2, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 4, 1)         7           regionsequence_max[0][0]         
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_max[0][0]               
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
Total params: 1,377
Trainable params: 1,377
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2575 - binary_crossentropy: 0.7089500/500 [==============================] - 5s 10ms/sample - loss: 0.2594 - binary_crossentropy: 0.7129 - val_loss: 0.2529 - val_binary_crossentropy: 0.6991

  #### metrics   #################################################### 
{'MSE': 0.2546191823833981}

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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_max[0][0]               
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         12          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         24          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         12          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 6, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         12          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         12          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 6, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 6, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 6, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           hash_10[0][0]                    
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
Total params: 3,018
Trainable params: 2,938
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.5400 - binary_crossentropy: 8.3295500/500 [==============================] - 6s 12ms/sample - loss: 0.5260 - binary_crossentropy: 8.1135 - val_loss: 0.5180 - val_binary_crossentropy: 7.9901

  #### metrics   #################################################### 
{'MSE': 0.522}

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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         12          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         24          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         12          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 6, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         12          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         12          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 6, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 6, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 6, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           hash_10[0][0]                    
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
Total params: 3,018
Trainable params: 2,938
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         8           sequence_max[0][0]               
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
Total params: 465
Trainable params: 465
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2532 - binary_crossentropy: 0.6997500/500 [==============================] - 6s 12ms/sample - loss: 0.2527 - binary_crossentropy: 0.6988 - val_loss: 0.2505 - val_binary_crossentropy: 0.6942

  #### metrics   #################################################### 
{'MSE': 0.25053865976754097}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         8           sequence_max[0][0]               
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
Total params: 465
Trainable params: 465
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
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_44 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
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
Total params: 1,989
Trainable params: 1,989
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 6s 12ms/sample - loss: 0.2502 - binary_crossentropy: 0.6936 - val_loss: 0.2500 - val_binary_crossentropy: 0.6932

  #### metrics   #################################################### 
{'MSE': 0.24993686291225073}

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
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_44 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
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
Total params: 1,989
Trainable params: 1,989
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_47 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_90 (Concatenate)    (None, 1, 20)        0           no_mask_130[0][0]                
                                                                 no_mask_130[1][0]                
                                                                 no_mask_130[2][0]                
                                                                 no_mask_130[3][0]                
                                                                 no_mask_130[4][0]                
__________________________________________________________________________________________________
no_mask_131 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
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
Total params: 296
Trainable params: 296
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2609 - binary_crossentropy: 0.7164500/500 [==============================] - 6s 13ms/sample - loss: 0.2570 - binary_crossentropy: 0.7077 - val_loss: 0.2580 - val_binary_crossentropy: 0.7096

  #### metrics   #################################################### 
{'MSE': 0.25653900737014157}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_47 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_90 (Concatenate)    (None, 1, 20)        0           no_mask_130[0][0]                
                                                                 no_mask_130[1][0]                
                                                                 no_mask_130[2][0]                
                                                                 no_mask_130[3][0]                
                                                                 no_mask_130[4][0]                
__________________________________________________________________________________________________
no_mask_131 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
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
Total params: 296
Trainable params: 296
Non-trainable params: 0
__________________________________________________________________________________________________

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Warning: Permanently added the RSA host key for IP address '140.82.113.3' to the list of known hosts.
From github.com:arita37/mlmodels_store
   1a51214..24d1ba9  master     -> origin/master
Updating 1a51214..24d1ba9
Fast-forward
 error_list/20200515/list_log_benchmark_20200515.md |  434 ++++----
 .../20200515/list_log_dataloader_20200515.md       |    2 +-
 error_list/20200515/list_log_import_20200515.md    |    2 +-
 error_list/20200515/list_log_json_20200515.md      | 1146 ++++++++++----------
 error_list/20200515/list_log_testall_20200515.md   |  820 +-------------
 ...-11_169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077.py |  611 +++++++++++
 6 files changed, 1410 insertions(+), 1605 deletions(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-15-16-11_169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077.py
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
[master d5f8a68] ml_store
 1 file changed, 5673 insertions(+)
To github.com:arita37/mlmodels_store.git
   24d1ba9..d5f8a68  master -> master





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
[master cdb823f] ml_store
 1 file changed, 50 insertions(+)
To github.com:arita37/mlmodels_store.git
   d5f8a68..cdb823f  master -> master





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
[master 87f9c35] ml_store
 1 file changed, 46 insertions(+)
To github.com:arita37/mlmodels_store.git
   cdb823f..87f9c35  master -> master





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
[master 827482f] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   87f9c35..827482f  master -> master





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

2020-05-15 16:25:15.451493: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 16:25:15.458075: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-15 16:25:15.458229: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56284071f7b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 16:25:15.458245: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 8s - loss: 1.3828
256/354 [====================>.........] - ETA: 3s - loss: 1.1750
354/354 [==============================] - 14s 40ms/step - loss: 1.3636 - val_loss: 2.8594

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
[master 978f2e2] ml_store
 1 file changed, 149 insertions(+)
To github.com:arita37/mlmodels_store.git
   827482f..978f2e2  master -> master





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
[master 6d6b3f1] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   978f2e2..6d6b3f1  master -> master





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
[master 97549a0] ml_store
 1 file changed, 44 insertions(+)
To github.com:arita37/mlmodels_store.git
   6d6b3f1..97549a0  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2703360/17464789 [===>..........................] - ETA: 0s
 9822208/17464789 [===============>..............] - ETA: 0s
13115392/17464789 [=====================>........] - ETA: 0s
16580608/17464789 [===========================>..] - ETA: 0s
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
2020-05-15 16:26:13.365711: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 16:26:13.369365: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-15 16:26:13.369500: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563c34fd3db0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 16:26:13.369514: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.8813 - accuracy: 0.4860
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8966 - accuracy: 0.4850 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7637 - accuracy: 0.4937
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7165 - accuracy: 0.4967
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6513 - accuracy: 0.5010
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6104 - accuracy: 0.5037
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5681 - accuracy: 0.5064
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6053 - accuracy: 0.5040
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6291 - accuracy: 0.5024
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5869 - accuracy: 0.5052
11000/25000 [============>.................] - ETA: 4s - loss: 7.6206 - accuracy: 0.5030
12000/25000 [=============>................] - ETA: 3s - loss: 7.6015 - accuracy: 0.5042
13000/25000 [==============>...............] - ETA: 3s - loss: 7.5911 - accuracy: 0.5049
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6053 - accuracy: 0.5040
15000/25000 [=================>............] - ETA: 3s - loss: 7.5920 - accuracy: 0.5049
16000/25000 [==================>...........] - ETA: 2s - loss: 7.5976 - accuracy: 0.5045
17000/25000 [===================>..........] - ETA: 2s - loss: 7.5972 - accuracy: 0.5045
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6428 - accuracy: 0.5016
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6440 - accuracy: 0.5015
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6697 - accuracy: 0.4998
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6630 - accuracy: 0.5002
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6652 - accuracy: 0.5001
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 9s 358us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7ffbac124d68>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7ffba2851ac8> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.9426 - accuracy: 0.4820
 2000/25000 [=>............................] - ETA: 8s - loss: 7.8583 - accuracy: 0.4875 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.9171 - accuracy: 0.4837
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6858 - accuracy: 0.4988
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6544 - accuracy: 0.5008
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6360 - accuracy: 0.5020
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6535 - accuracy: 0.5009
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7471 - accuracy: 0.4947
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7211 - accuracy: 0.4964
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6973 - accuracy: 0.4980
11000/25000 [============>.................] - ETA: 4s - loss: 7.7182 - accuracy: 0.4966
12000/25000 [=============>................] - ETA: 3s - loss: 7.7011 - accuracy: 0.4978
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7185 - accuracy: 0.4966
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7422 - accuracy: 0.4951
15000/25000 [=================>............] - ETA: 2s - loss: 7.7474 - accuracy: 0.4947
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7117 - accuracy: 0.4971
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7126 - accuracy: 0.4970
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6998 - accuracy: 0.4978
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6933 - accuracy: 0.4983
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6743 - accuracy: 0.4995
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6761 - accuracy: 0.4994
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6638 - accuracy: 0.5002
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6615 - accuracy: 0.5003
25000/25000 [==============================] - 9s 357us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.1300 - accuracy: 0.5350
 2000/25000 [=>............................] - ETA: 8s - loss: 7.4750 - accuracy: 0.5125 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.4775 - accuracy: 0.5123
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5631 - accuracy: 0.5067
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5624 - accuracy: 0.5068
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5721 - accuracy: 0.5062
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6009 - accuracy: 0.5043
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5612 - accuracy: 0.5069
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5678 - accuracy: 0.5064
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5838 - accuracy: 0.5054
11000/25000 [============>.................] - ETA: 4s - loss: 7.5900 - accuracy: 0.5050
12000/25000 [=============>................] - ETA: 3s - loss: 7.6053 - accuracy: 0.5040
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6277 - accuracy: 0.5025
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6272 - accuracy: 0.5026
15000/25000 [=================>............] - ETA: 2s - loss: 7.6441 - accuracy: 0.5015
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6398 - accuracy: 0.5017
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6351 - accuracy: 0.5021
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6573 - accuracy: 0.5006
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6658 - accuracy: 0.5001
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6544 - accuracy: 0.5008
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6476 - accuracy: 0.5012
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6520 - accuracy: 0.5010
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6540 - accuracy: 0.5008
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6653 - accuracy: 0.5001
25000/25000 [==============================] - 9s 356us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
From github.com:arita37/mlmodels_store
   97549a0..1f3dae6  master     -> origin/master
Updating 97549a0..1f3dae6
Fast-forward
 error_list/20200515/list_log_testall_20200515.md | 89 ++++++++++++++++++++++++
 1 file changed, 89 insertions(+)
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
[master 7a1f68b] ml_store
 1 file changed, 324 insertions(+)
To github.com:arita37/mlmodels_store.git
   1f3dae6..7a1f68b  master -> master





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

13/13 [==============================] - 1s 103ms/step - loss: nan
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
[master 2e2c008] ml_store
 1 file changed, 125 insertions(+)
To github.com:arita37/mlmodels_store.git
   7a1f68b..2e2c008  master -> master





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
 1613824/11490434 [===>..........................] - ETA: 0s
 6356992/11490434 [===============>..............] - ETA: 0s
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

   32/60000 [..............................] - ETA: 6:47 - loss: 2.2979 - categorical_accuracy: 0.0312
   64/60000 [..............................] - ETA: 4:18 - loss: 2.2762 - categorical_accuracy: 0.1406
   96/60000 [..............................] - ETA: 3:27 - loss: 2.2716 - categorical_accuracy: 0.1354
  128/60000 [..............................] - ETA: 2:59 - loss: 2.2226 - categorical_accuracy: 0.1875
  160/60000 [..............................] - ETA: 2:42 - loss: 2.1773 - categorical_accuracy: 0.2250
  192/60000 [..............................] - ETA: 2:31 - loss: 2.1544 - categorical_accuracy: 0.2292
  224/60000 [..............................] - ETA: 2:23 - loss: 2.1423 - categorical_accuracy: 0.2232
  256/60000 [..............................] - ETA: 2:18 - loss: 2.1258 - categorical_accuracy: 0.2305
  288/60000 [..............................] - ETA: 2:14 - loss: 2.0618 - categorical_accuracy: 0.2604
  320/60000 [..............................] - ETA: 2:12 - loss: 2.0259 - categorical_accuracy: 0.2781
  352/60000 [..............................] - ETA: 2:09 - loss: 1.9924 - categorical_accuracy: 0.2926
  384/60000 [..............................] - ETA: 2:07 - loss: 1.9597 - categorical_accuracy: 0.2969
  416/60000 [..............................] - ETA: 2:04 - loss: 1.9363 - categorical_accuracy: 0.3173
  448/60000 [..............................] - ETA: 2:03 - loss: 1.8968 - categorical_accuracy: 0.3304
  480/60000 [..............................] - ETA: 2:01 - loss: 1.8810 - categorical_accuracy: 0.3417
  512/60000 [..............................] - ETA: 2:00 - loss: 1.8561 - categorical_accuracy: 0.3438
  544/60000 [..............................] - ETA: 1:59 - loss: 1.8213 - categorical_accuracy: 0.3621
  576/60000 [..............................] - ETA: 1:58 - loss: 1.7920 - categorical_accuracy: 0.3698
  608/60000 [..............................] - ETA: 1:57 - loss: 1.7742 - categorical_accuracy: 0.3799
  640/60000 [..............................] - ETA: 1:56 - loss: 1.7479 - categorical_accuracy: 0.3844
  672/60000 [..............................] - ETA: 1:55 - loss: 1.7284 - categorical_accuracy: 0.3914
  704/60000 [..............................] - ETA: 1:54 - loss: 1.6944 - categorical_accuracy: 0.4062
  736/60000 [..............................] - ETA: 1:54 - loss: 1.6705 - categorical_accuracy: 0.4130
  768/60000 [..............................] - ETA: 1:53 - loss: 1.6607 - categorical_accuracy: 0.4193
  800/60000 [..............................] - ETA: 1:52 - loss: 1.6418 - categorical_accuracy: 0.4288
  832/60000 [..............................] - ETA: 1:51 - loss: 1.6200 - categorical_accuracy: 0.4387
  864/60000 [..............................] - ETA: 1:51 - loss: 1.6009 - categorical_accuracy: 0.4444
  896/60000 [..............................] - ETA: 1:51 - loss: 1.5828 - categorical_accuracy: 0.4542
  928/60000 [..............................] - ETA: 1:51 - loss: 1.5566 - categorical_accuracy: 0.4666
  960/60000 [..............................] - ETA: 1:51 - loss: 1.5354 - categorical_accuracy: 0.4750
  992/60000 [..............................] - ETA: 1:50 - loss: 1.5351 - categorical_accuracy: 0.4748
 1024/60000 [..............................] - ETA: 1:50 - loss: 1.5201 - categorical_accuracy: 0.4824
 1056/60000 [..............................] - ETA: 1:50 - loss: 1.5020 - categorical_accuracy: 0.4905
 1088/60000 [..............................] - ETA: 1:49 - loss: 1.4856 - categorical_accuracy: 0.4991
 1120/60000 [..............................] - ETA: 1:49 - loss: 1.4623 - categorical_accuracy: 0.5071
 1152/60000 [..............................] - ETA: 1:48 - loss: 1.4352 - categorical_accuracy: 0.5165
 1184/60000 [..............................] - ETA: 1:48 - loss: 1.4175 - categorical_accuracy: 0.5220
 1216/60000 [..............................] - ETA: 1:48 - loss: 1.4155 - categorical_accuracy: 0.5263
 1248/60000 [..............................] - ETA: 1:48 - loss: 1.4010 - categorical_accuracy: 0.5304
 1280/60000 [..............................] - ETA: 1:47 - loss: 1.3975 - categorical_accuracy: 0.5312
 1312/60000 [..............................] - ETA: 1:47 - loss: 1.3820 - categorical_accuracy: 0.5389
 1344/60000 [..............................] - ETA: 1:47 - loss: 1.3654 - categorical_accuracy: 0.5432
 1376/60000 [..............................] - ETA: 1:47 - loss: 1.3427 - categorical_accuracy: 0.5523
 1408/60000 [..............................] - ETA: 1:46 - loss: 1.3210 - categorical_accuracy: 0.5604
 1440/60000 [..............................] - ETA: 1:46 - loss: 1.3099 - categorical_accuracy: 0.5639
 1472/60000 [..............................] - ETA: 1:46 - loss: 1.2970 - categorical_accuracy: 0.5679
 1504/60000 [..............................] - ETA: 1:46 - loss: 1.2800 - categorical_accuracy: 0.5745
 1536/60000 [..............................] - ETA: 1:46 - loss: 1.2666 - categorical_accuracy: 0.5801
 1568/60000 [..............................] - ETA: 1:45 - loss: 1.2496 - categorical_accuracy: 0.5861
 1600/60000 [..............................] - ETA: 1:45 - loss: 1.2406 - categorical_accuracy: 0.5894
 1632/60000 [..............................] - ETA: 1:45 - loss: 1.2306 - categorical_accuracy: 0.5931
 1664/60000 [..............................] - ETA: 1:45 - loss: 1.2213 - categorical_accuracy: 0.5956
 1696/60000 [..............................] - ETA: 1:45 - loss: 1.2107 - categorical_accuracy: 0.6002
 1728/60000 [..............................] - ETA: 1:45 - loss: 1.1998 - categorical_accuracy: 0.6036
 1760/60000 [..............................] - ETA: 1:45 - loss: 1.1883 - categorical_accuracy: 0.6080
 1792/60000 [..............................] - ETA: 1:45 - loss: 1.1754 - categorical_accuracy: 0.6122
 1824/60000 [..............................] - ETA: 1:45 - loss: 1.1687 - categorical_accuracy: 0.6146
 1856/60000 [..............................] - ETA: 1:44 - loss: 1.1578 - categorical_accuracy: 0.6175
 1888/60000 [..............................] - ETA: 1:44 - loss: 1.1442 - categorical_accuracy: 0.6218
 1920/60000 [..............................] - ETA: 1:44 - loss: 1.1308 - categorical_accuracy: 0.6266
 1984/60000 [..............................] - ETA: 1:43 - loss: 1.1131 - categorical_accuracy: 0.6341
 2016/60000 [>.............................] - ETA: 1:43 - loss: 1.1024 - categorical_accuracy: 0.6359
 2048/60000 [>.............................] - ETA: 1:43 - loss: 1.0916 - categorical_accuracy: 0.6387
 2080/60000 [>.............................] - ETA: 1:43 - loss: 1.0842 - categorical_accuracy: 0.6423
 2112/60000 [>.............................] - ETA: 1:43 - loss: 1.0758 - categorical_accuracy: 0.6454
 2144/60000 [>.............................] - ETA: 1:43 - loss: 1.0661 - categorical_accuracy: 0.6483
 2176/60000 [>.............................] - ETA: 1:43 - loss: 1.0615 - categorical_accuracy: 0.6489
 2208/60000 [>.............................] - ETA: 1:43 - loss: 1.0566 - categorical_accuracy: 0.6513
 2240/60000 [>.............................] - ETA: 1:43 - loss: 1.0473 - categorical_accuracy: 0.6536
 2272/60000 [>.............................] - ETA: 1:42 - loss: 1.0369 - categorical_accuracy: 0.6567
 2304/60000 [>.............................] - ETA: 1:42 - loss: 1.0294 - categorical_accuracy: 0.6593
 2336/60000 [>.............................] - ETA: 1:42 - loss: 1.0197 - categorical_accuracy: 0.6627
 2368/60000 [>.............................] - ETA: 1:42 - loss: 1.0107 - categorical_accuracy: 0.6664
 2400/60000 [>.............................] - ETA: 1:42 - loss: 1.0029 - categorical_accuracy: 0.6692
 2432/60000 [>.............................] - ETA: 1:42 - loss: 0.9978 - categorical_accuracy: 0.6706
 2464/60000 [>.............................] - ETA: 1:42 - loss: 0.9998 - categorical_accuracy: 0.6700
 2496/60000 [>.............................] - ETA: 1:42 - loss: 0.9958 - categorical_accuracy: 0.6715
 2528/60000 [>.............................] - ETA: 1:41 - loss: 0.9910 - categorical_accuracy: 0.6729
 2560/60000 [>.............................] - ETA: 1:41 - loss: 0.9849 - categorical_accuracy: 0.6746
 2592/60000 [>.............................] - ETA: 1:41 - loss: 0.9788 - categorical_accuracy: 0.6767
 2624/60000 [>.............................] - ETA: 1:41 - loss: 0.9701 - categorical_accuracy: 0.6799
 2656/60000 [>.............................] - ETA: 1:41 - loss: 0.9634 - categorical_accuracy: 0.6826
 2688/60000 [>.............................] - ETA: 1:41 - loss: 0.9586 - categorical_accuracy: 0.6849
 2720/60000 [>.............................] - ETA: 1:41 - loss: 0.9535 - categorical_accuracy: 0.6868
 2752/60000 [>.............................] - ETA: 1:41 - loss: 0.9491 - categorical_accuracy: 0.6882
 2784/60000 [>.............................] - ETA: 1:41 - loss: 0.9488 - categorical_accuracy: 0.6893
 2816/60000 [>.............................] - ETA: 1:41 - loss: 0.9438 - categorical_accuracy: 0.6900
 2848/60000 [>.............................] - ETA: 1:41 - loss: 0.9368 - categorical_accuracy: 0.6917
 2880/60000 [>.............................] - ETA: 1:41 - loss: 0.9301 - categorical_accuracy: 0.6938
 2912/60000 [>.............................] - ETA: 1:40 - loss: 0.9262 - categorical_accuracy: 0.6951
 2944/60000 [>.............................] - ETA: 1:40 - loss: 0.9229 - categorical_accuracy: 0.6957
 2976/60000 [>.............................] - ETA: 1:40 - loss: 0.9171 - categorical_accuracy: 0.6976
 3008/60000 [>.............................] - ETA: 1:40 - loss: 0.9133 - categorical_accuracy: 0.6988
 3040/60000 [>.............................] - ETA: 1:40 - loss: 0.9081 - categorical_accuracy: 0.7016
 3072/60000 [>.............................] - ETA: 1:40 - loss: 0.9018 - categorical_accuracy: 0.7038
 3104/60000 [>.............................] - ETA: 1:40 - loss: 0.8953 - categorical_accuracy: 0.7062
 3168/60000 [>.............................] - ETA: 1:39 - loss: 0.8840 - categorical_accuracy: 0.7105
 3200/60000 [>.............................] - ETA: 1:39 - loss: 0.8777 - categorical_accuracy: 0.7128
 3232/60000 [>.............................] - ETA: 1:39 - loss: 0.8716 - categorical_accuracy: 0.7144
 3264/60000 [>.............................] - ETA: 1:39 - loss: 0.8672 - categorical_accuracy: 0.7160
 3296/60000 [>.............................] - ETA: 1:39 - loss: 0.8625 - categorical_accuracy: 0.7172
 3328/60000 [>.............................] - ETA: 1:39 - loss: 0.8574 - categorical_accuracy: 0.7188
 3360/60000 [>.............................] - ETA: 1:39 - loss: 0.8509 - categorical_accuracy: 0.7208
 3392/60000 [>.............................] - ETA: 1:39 - loss: 0.8454 - categorical_accuracy: 0.7226
 3424/60000 [>.............................] - ETA: 1:39 - loss: 0.8395 - categorical_accuracy: 0.7246
 3456/60000 [>.............................] - ETA: 1:39 - loss: 0.8348 - categorical_accuracy: 0.7257
 3488/60000 [>.............................] - ETA: 1:38 - loss: 0.8314 - categorical_accuracy: 0.7274
 3520/60000 [>.............................] - ETA: 1:38 - loss: 0.8268 - categorical_accuracy: 0.7287
 3552/60000 [>.............................] - ETA: 1:38 - loss: 0.8211 - categorical_accuracy: 0.7303
 3584/60000 [>.............................] - ETA: 1:38 - loss: 0.8171 - categorical_accuracy: 0.7319
 3616/60000 [>.............................] - ETA: 1:38 - loss: 0.8160 - categorical_accuracy: 0.7334
 3648/60000 [>.............................] - ETA: 1:38 - loss: 0.8122 - categorical_accuracy: 0.7346
 3680/60000 [>.............................] - ETA: 1:38 - loss: 0.8078 - categorical_accuracy: 0.7364
 3712/60000 [>.............................] - ETA: 1:38 - loss: 0.8025 - categorical_accuracy: 0.7381
 3744/60000 [>.............................] - ETA: 1:38 - loss: 0.7978 - categorical_accuracy: 0.7399
 3776/60000 [>.............................] - ETA: 1:38 - loss: 0.7926 - categorical_accuracy: 0.7413
 3808/60000 [>.............................] - ETA: 1:38 - loss: 0.7933 - categorical_accuracy: 0.7424
 3840/60000 [>.............................] - ETA: 1:38 - loss: 0.7882 - categorical_accuracy: 0.7443
 3872/60000 [>.............................] - ETA: 1:38 - loss: 0.7840 - categorical_accuracy: 0.7456
 3904/60000 [>.............................] - ETA: 1:38 - loss: 0.7803 - categorical_accuracy: 0.7467
 3936/60000 [>.............................] - ETA: 1:38 - loss: 0.7783 - categorical_accuracy: 0.7472
 3968/60000 [>.............................] - ETA: 1:37 - loss: 0.7754 - categorical_accuracy: 0.7480
 4000/60000 [=>............................] - ETA: 1:37 - loss: 0.7745 - categorical_accuracy: 0.7490
 4032/60000 [=>............................] - ETA: 1:37 - loss: 0.7698 - categorical_accuracy: 0.7505
 4096/60000 [=>............................] - ETA: 1:37 - loss: 0.7638 - categorical_accuracy: 0.7529
 4128/60000 [=>............................] - ETA: 1:37 - loss: 0.7612 - categorical_accuracy: 0.7539
 4160/60000 [=>............................] - ETA: 1:37 - loss: 0.7607 - categorical_accuracy: 0.7543
 4192/60000 [=>............................] - ETA: 1:37 - loss: 0.7569 - categorical_accuracy: 0.7557
 4224/60000 [=>............................] - ETA: 1:37 - loss: 0.7544 - categorical_accuracy: 0.7566
 4288/60000 [=>............................] - ETA: 1:36 - loss: 0.7504 - categorical_accuracy: 0.7584
 4320/60000 [=>............................] - ETA: 1:36 - loss: 0.7473 - categorical_accuracy: 0.7597
 4352/60000 [=>............................] - ETA: 1:36 - loss: 0.7440 - categorical_accuracy: 0.7606
 4384/60000 [=>............................] - ETA: 1:36 - loss: 0.7408 - categorical_accuracy: 0.7619
 4416/60000 [=>............................] - ETA: 1:36 - loss: 0.7387 - categorical_accuracy: 0.7627
 4448/60000 [=>............................] - ETA: 1:36 - loss: 0.7348 - categorical_accuracy: 0.7637
 4480/60000 [=>............................] - ETA: 1:36 - loss: 0.7310 - categorical_accuracy: 0.7650
 4512/60000 [=>............................] - ETA: 1:36 - loss: 0.7271 - categorical_accuracy: 0.7657
 4544/60000 [=>............................] - ETA: 1:36 - loss: 0.7262 - categorical_accuracy: 0.7663
 4576/60000 [=>............................] - ETA: 1:36 - loss: 0.7227 - categorical_accuracy: 0.7675
 4608/60000 [=>............................] - ETA: 1:36 - loss: 0.7206 - categorical_accuracy: 0.7680
 4640/60000 [=>............................] - ETA: 1:36 - loss: 0.7176 - categorical_accuracy: 0.7688
 4672/60000 [=>............................] - ETA: 1:36 - loss: 0.7148 - categorical_accuracy: 0.7701
 4704/60000 [=>............................] - ETA: 1:36 - loss: 0.7129 - categorical_accuracy: 0.7708
 4736/60000 [=>............................] - ETA: 1:36 - loss: 0.7091 - categorical_accuracy: 0.7720
 4768/60000 [=>............................] - ETA: 1:36 - loss: 0.7074 - categorical_accuracy: 0.7724
 4800/60000 [=>............................] - ETA: 1:36 - loss: 0.7051 - categorical_accuracy: 0.7733
 4832/60000 [=>............................] - ETA: 1:35 - loss: 0.7025 - categorical_accuracy: 0.7742
 4864/60000 [=>............................] - ETA: 1:35 - loss: 0.7013 - categorical_accuracy: 0.7741
 4896/60000 [=>............................] - ETA: 1:35 - loss: 0.6990 - categorical_accuracy: 0.7745
 4928/60000 [=>............................] - ETA: 1:35 - loss: 0.6986 - categorical_accuracy: 0.7746
 4992/60000 [=>............................] - ETA: 1:35 - loss: 0.6946 - categorical_accuracy: 0.7764
 5024/60000 [=>............................] - ETA: 1:35 - loss: 0.6924 - categorical_accuracy: 0.7771
 5056/60000 [=>............................] - ETA: 1:35 - loss: 0.6887 - categorical_accuracy: 0.7785
 5088/60000 [=>............................] - ETA: 1:35 - loss: 0.6859 - categorical_accuracy: 0.7795
 5120/60000 [=>............................] - ETA: 1:35 - loss: 0.6829 - categorical_accuracy: 0.7803
 5152/60000 [=>............................] - ETA: 1:35 - loss: 0.6799 - categorical_accuracy: 0.7812
 5184/60000 [=>............................] - ETA: 1:35 - loss: 0.6763 - categorical_accuracy: 0.7826
 5216/60000 [=>............................] - ETA: 1:35 - loss: 0.6731 - categorical_accuracy: 0.7836
 5248/60000 [=>............................] - ETA: 1:35 - loss: 0.6713 - categorical_accuracy: 0.7841
 5280/60000 [=>............................] - ETA: 1:34 - loss: 0.6676 - categorical_accuracy: 0.7854
 5312/60000 [=>............................] - ETA: 1:34 - loss: 0.6646 - categorical_accuracy: 0.7863
 5344/60000 [=>............................] - ETA: 1:34 - loss: 0.6620 - categorical_accuracy: 0.7874
 5376/60000 [=>............................] - ETA: 1:34 - loss: 0.6584 - categorical_accuracy: 0.7887
 5408/60000 [=>............................] - ETA: 1:34 - loss: 0.6569 - categorical_accuracy: 0.7896
 5440/60000 [=>............................] - ETA: 1:34 - loss: 0.6543 - categorical_accuracy: 0.7904
 5472/60000 [=>............................] - ETA: 1:34 - loss: 0.6521 - categorical_accuracy: 0.7913
 5504/60000 [=>............................] - ETA: 1:34 - loss: 0.6505 - categorical_accuracy: 0.7920
 5536/60000 [=>............................] - ETA: 1:34 - loss: 0.6475 - categorical_accuracy: 0.7930
 5568/60000 [=>............................] - ETA: 1:34 - loss: 0.6465 - categorical_accuracy: 0.7936
 5600/60000 [=>............................] - ETA: 1:34 - loss: 0.6433 - categorical_accuracy: 0.7948
 5632/60000 [=>............................] - ETA: 1:34 - loss: 0.6424 - categorical_accuracy: 0.7953
 5664/60000 [=>............................] - ETA: 1:34 - loss: 0.6397 - categorical_accuracy: 0.7961
 5696/60000 [=>............................] - ETA: 1:34 - loss: 0.6379 - categorical_accuracy: 0.7969
 5728/60000 [=>............................] - ETA: 1:34 - loss: 0.6349 - categorical_accuracy: 0.7980
 5760/60000 [=>............................] - ETA: 1:34 - loss: 0.6325 - categorical_accuracy: 0.7988
 5792/60000 [=>............................] - ETA: 1:33 - loss: 0.6314 - categorical_accuracy: 0.7994
 5824/60000 [=>............................] - ETA: 1:33 - loss: 0.6289 - categorical_accuracy: 0.8001
 5856/60000 [=>............................] - ETA: 1:33 - loss: 0.6272 - categorical_accuracy: 0.8009
 5888/60000 [=>............................] - ETA: 1:33 - loss: 0.6259 - categorical_accuracy: 0.8015
 5920/60000 [=>............................] - ETA: 1:33 - loss: 0.6242 - categorical_accuracy: 0.8020
 5952/60000 [=>............................] - ETA: 1:33 - loss: 0.6212 - categorical_accuracy: 0.8031
 5984/60000 [=>............................] - ETA: 1:33 - loss: 0.6187 - categorical_accuracy: 0.8038
 6048/60000 [==>...........................] - ETA: 1:33 - loss: 0.6140 - categorical_accuracy: 0.8052
 6112/60000 [==>...........................] - ETA: 1:33 - loss: 0.6110 - categorical_accuracy: 0.8063
 6144/60000 [==>...........................] - ETA: 1:33 - loss: 0.6087 - categorical_accuracy: 0.8068
 6176/60000 [==>...........................] - ETA: 1:33 - loss: 0.6076 - categorical_accuracy: 0.8073
 6208/60000 [==>...........................] - ETA: 1:33 - loss: 0.6073 - categorical_accuracy: 0.8077
 6240/60000 [==>...........................] - ETA: 1:33 - loss: 0.6053 - categorical_accuracy: 0.8082
 6272/60000 [==>...........................] - ETA: 1:32 - loss: 0.6037 - categorical_accuracy: 0.8090
 6304/60000 [==>...........................] - ETA: 1:32 - loss: 0.6014 - categorical_accuracy: 0.8096
 6336/60000 [==>...........................] - ETA: 1:32 - loss: 0.5989 - categorical_accuracy: 0.8104
 6368/60000 [==>...........................] - ETA: 1:32 - loss: 0.5966 - categorical_accuracy: 0.8112
 6400/60000 [==>...........................] - ETA: 1:32 - loss: 0.5955 - categorical_accuracy: 0.8117
 6432/60000 [==>...........................] - ETA: 1:32 - loss: 0.5934 - categorical_accuracy: 0.8123
 6464/60000 [==>...........................] - ETA: 1:32 - loss: 0.5913 - categorical_accuracy: 0.8131
 6496/60000 [==>...........................] - ETA: 1:32 - loss: 0.5891 - categorical_accuracy: 0.8137
 6528/60000 [==>...........................] - ETA: 1:32 - loss: 0.5872 - categorical_accuracy: 0.8143
 6560/60000 [==>...........................] - ETA: 1:32 - loss: 0.5854 - categorical_accuracy: 0.8149
 6592/60000 [==>...........................] - ETA: 1:32 - loss: 0.5834 - categorical_accuracy: 0.8155
 6624/60000 [==>...........................] - ETA: 1:32 - loss: 0.5825 - categorical_accuracy: 0.8158
 6656/60000 [==>...........................] - ETA: 1:32 - loss: 0.5816 - categorical_accuracy: 0.8161
 6688/60000 [==>...........................] - ETA: 1:32 - loss: 0.5803 - categorical_accuracy: 0.8165
 6720/60000 [==>...........................] - ETA: 1:31 - loss: 0.5789 - categorical_accuracy: 0.8171
 6752/60000 [==>...........................] - ETA: 1:31 - loss: 0.5771 - categorical_accuracy: 0.8178
 6784/60000 [==>...........................] - ETA: 1:31 - loss: 0.5751 - categorical_accuracy: 0.8185
 6816/60000 [==>...........................] - ETA: 1:31 - loss: 0.5745 - categorical_accuracy: 0.8191
 6848/60000 [==>...........................] - ETA: 1:31 - loss: 0.5735 - categorical_accuracy: 0.8192
 6880/60000 [==>...........................] - ETA: 1:31 - loss: 0.5719 - categorical_accuracy: 0.8198
 6912/60000 [==>...........................] - ETA: 1:31 - loss: 0.5698 - categorical_accuracy: 0.8205
 6944/60000 [==>...........................] - ETA: 1:31 - loss: 0.5681 - categorical_accuracy: 0.8211
 6976/60000 [==>...........................] - ETA: 1:31 - loss: 0.5671 - categorical_accuracy: 0.8214
 7008/60000 [==>...........................] - ETA: 1:31 - loss: 0.5651 - categorical_accuracy: 0.8219
 7040/60000 [==>...........................] - ETA: 1:31 - loss: 0.5640 - categorical_accuracy: 0.8219
 7072/60000 [==>...........................] - ETA: 1:31 - loss: 0.5620 - categorical_accuracy: 0.8225
 7104/60000 [==>...........................] - ETA: 1:31 - loss: 0.5606 - categorical_accuracy: 0.8232
 7136/60000 [==>...........................] - ETA: 1:31 - loss: 0.5587 - categorical_accuracy: 0.8237
 7168/60000 [==>...........................] - ETA: 1:31 - loss: 0.5580 - categorical_accuracy: 0.8239
 7232/60000 [==>...........................] - ETA: 1:30 - loss: 0.5553 - categorical_accuracy: 0.8249
 7264/60000 [==>...........................] - ETA: 1:30 - loss: 0.5539 - categorical_accuracy: 0.8254
 7296/60000 [==>...........................] - ETA: 1:30 - loss: 0.5524 - categorical_accuracy: 0.8259
 7328/60000 [==>...........................] - ETA: 1:30 - loss: 0.5520 - categorical_accuracy: 0.8261
 7360/60000 [==>...........................] - ETA: 1:30 - loss: 0.5497 - categorical_accuracy: 0.8269
 7392/60000 [==>...........................] - ETA: 1:30 - loss: 0.5488 - categorical_accuracy: 0.8274
 7424/60000 [==>...........................] - ETA: 1:30 - loss: 0.5471 - categorical_accuracy: 0.8279
 7456/60000 [==>...........................] - ETA: 1:30 - loss: 0.5455 - categorical_accuracy: 0.8282
 7488/60000 [==>...........................] - ETA: 1:30 - loss: 0.5443 - categorical_accuracy: 0.8285
 7520/60000 [==>...........................] - ETA: 1:30 - loss: 0.5423 - categorical_accuracy: 0.8293
 7552/60000 [==>...........................] - ETA: 1:30 - loss: 0.5403 - categorical_accuracy: 0.8300
 7584/60000 [==>...........................] - ETA: 1:30 - loss: 0.5389 - categorical_accuracy: 0.8302
 7616/60000 [==>...........................] - ETA: 1:30 - loss: 0.5380 - categorical_accuracy: 0.8302
 7648/60000 [==>...........................] - ETA: 1:30 - loss: 0.5369 - categorical_accuracy: 0.8308
 7680/60000 [==>...........................] - ETA: 1:30 - loss: 0.5356 - categorical_accuracy: 0.8313
 7712/60000 [==>...........................] - ETA: 1:30 - loss: 0.5355 - categorical_accuracy: 0.8314
 7744/60000 [==>...........................] - ETA: 1:30 - loss: 0.5339 - categorical_accuracy: 0.8319
 7776/60000 [==>...........................] - ETA: 1:29 - loss: 0.5322 - categorical_accuracy: 0.8324
 7808/60000 [==>...........................] - ETA: 1:29 - loss: 0.5315 - categorical_accuracy: 0.8326
 7872/60000 [==>...........................] - ETA: 1:29 - loss: 0.5286 - categorical_accuracy: 0.8333
 7904/60000 [==>...........................] - ETA: 1:29 - loss: 0.5293 - categorical_accuracy: 0.8334
 7936/60000 [==>...........................] - ETA: 1:29 - loss: 0.5278 - categorical_accuracy: 0.8337
 7968/60000 [==>...........................] - ETA: 1:29 - loss: 0.5269 - categorical_accuracy: 0.8340
 8000/60000 [===>..........................] - ETA: 1:29 - loss: 0.5253 - categorical_accuracy: 0.8345
 8032/60000 [===>..........................] - ETA: 1:29 - loss: 0.5240 - categorical_accuracy: 0.8348
 8064/60000 [===>..........................] - ETA: 1:29 - loss: 0.5227 - categorical_accuracy: 0.8352
 8096/60000 [===>..........................] - ETA: 1:29 - loss: 0.5210 - categorical_accuracy: 0.8357
 8128/60000 [===>..........................] - ETA: 1:29 - loss: 0.5198 - categorical_accuracy: 0.8361
 8160/60000 [===>..........................] - ETA: 1:29 - loss: 0.5179 - categorical_accuracy: 0.8368
 8192/60000 [===>..........................] - ETA: 1:29 - loss: 0.5172 - categorical_accuracy: 0.8373
 8256/60000 [===>..........................] - ETA: 1:28 - loss: 0.5151 - categorical_accuracy: 0.8379
 8288/60000 [===>..........................] - ETA: 1:28 - loss: 0.5150 - categorical_accuracy: 0.8381
 8320/60000 [===>..........................] - ETA: 1:28 - loss: 0.5132 - categorical_accuracy: 0.8387
 8352/60000 [===>..........................] - ETA: 1:28 - loss: 0.5119 - categorical_accuracy: 0.8392
 8384/60000 [===>..........................] - ETA: 1:28 - loss: 0.5108 - categorical_accuracy: 0.8393
 8416/60000 [===>..........................] - ETA: 1:28 - loss: 0.5092 - categorical_accuracy: 0.8398
 8448/60000 [===>..........................] - ETA: 1:28 - loss: 0.5076 - categorical_accuracy: 0.8403
 8480/60000 [===>..........................] - ETA: 1:28 - loss: 0.5061 - categorical_accuracy: 0.8408
 8512/60000 [===>..........................] - ETA: 1:28 - loss: 0.5046 - categorical_accuracy: 0.8413
 8544/60000 [===>..........................] - ETA: 1:28 - loss: 0.5033 - categorical_accuracy: 0.8416
 8576/60000 [===>..........................] - ETA: 1:28 - loss: 0.5026 - categorical_accuracy: 0.8420
 8608/60000 [===>..........................] - ETA: 1:28 - loss: 0.5017 - categorical_accuracy: 0.8422
 8640/60000 [===>..........................] - ETA: 1:28 - loss: 0.5002 - categorical_accuracy: 0.8428
 8672/60000 [===>..........................] - ETA: 1:28 - loss: 0.4987 - categorical_accuracy: 0.8433
 8704/60000 [===>..........................] - ETA: 1:28 - loss: 0.4979 - categorical_accuracy: 0.8438
 8736/60000 [===>..........................] - ETA: 1:28 - loss: 0.4964 - categorical_accuracy: 0.8442
 8768/60000 [===>..........................] - ETA: 1:28 - loss: 0.4963 - categorical_accuracy: 0.8444
 8800/60000 [===>..........................] - ETA: 1:28 - loss: 0.4948 - categorical_accuracy: 0.8449
 8832/60000 [===>..........................] - ETA: 1:28 - loss: 0.4931 - categorical_accuracy: 0.8454
 8864/60000 [===>..........................] - ETA: 1:28 - loss: 0.4919 - categorical_accuracy: 0.8458
 8896/60000 [===>..........................] - ETA: 1:27 - loss: 0.4904 - categorical_accuracy: 0.8462
 8928/60000 [===>..........................] - ETA: 1:27 - loss: 0.4893 - categorical_accuracy: 0.8464
 8960/60000 [===>..........................] - ETA: 1:27 - loss: 0.4891 - categorical_accuracy: 0.8465
 8992/60000 [===>..........................] - ETA: 1:27 - loss: 0.4879 - categorical_accuracy: 0.8469
 9024/60000 [===>..........................] - ETA: 1:27 - loss: 0.4866 - categorical_accuracy: 0.8473
 9056/60000 [===>..........................] - ETA: 1:27 - loss: 0.4858 - categorical_accuracy: 0.8475
 9088/60000 [===>..........................] - ETA: 1:27 - loss: 0.4846 - categorical_accuracy: 0.8478
 9120/60000 [===>..........................] - ETA: 1:27 - loss: 0.4833 - categorical_accuracy: 0.8482
 9152/60000 [===>..........................] - ETA: 1:27 - loss: 0.4821 - categorical_accuracy: 0.8487
 9184/60000 [===>..........................] - ETA: 1:27 - loss: 0.4817 - categorical_accuracy: 0.8489
 9216/60000 [===>..........................] - ETA: 1:27 - loss: 0.4809 - categorical_accuracy: 0.8491
 9248/60000 [===>..........................] - ETA: 1:27 - loss: 0.4810 - categorical_accuracy: 0.8493
 9280/60000 [===>..........................] - ETA: 1:27 - loss: 0.4808 - categorical_accuracy: 0.8494
 9312/60000 [===>..........................] - ETA: 1:27 - loss: 0.4802 - categorical_accuracy: 0.8497
 9344/60000 [===>..........................] - ETA: 1:27 - loss: 0.4791 - categorical_accuracy: 0.8501
 9376/60000 [===>..........................] - ETA: 1:27 - loss: 0.4779 - categorical_accuracy: 0.8505
 9408/60000 [===>..........................] - ETA: 1:27 - loss: 0.4767 - categorical_accuracy: 0.8509
 9440/60000 [===>..........................] - ETA: 1:26 - loss: 0.4756 - categorical_accuracy: 0.8512
 9472/60000 [===>..........................] - ETA: 1:26 - loss: 0.4746 - categorical_accuracy: 0.8516
 9504/60000 [===>..........................] - ETA: 1:26 - loss: 0.4731 - categorical_accuracy: 0.8521
 9536/60000 [===>..........................] - ETA: 1:26 - loss: 0.4720 - categorical_accuracy: 0.8525
 9568/60000 [===>..........................] - ETA: 1:26 - loss: 0.4714 - categorical_accuracy: 0.8527
 9600/60000 [===>..........................] - ETA: 1:26 - loss: 0.4700 - categorical_accuracy: 0.8532
 9632/60000 [===>..........................] - ETA: 1:26 - loss: 0.4689 - categorical_accuracy: 0.8535
 9664/60000 [===>..........................] - ETA: 1:26 - loss: 0.4675 - categorical_accuracy: 0.8540
 9696/60000 [===>..........................] - ETA: 1:26 - loss: 0.4674 - categorical_accuracy: 0.8540
 9728/60000 [===>..........................] - ETA: 1:26 - loss: 0.4666 - categorical_accuracy: 0.8542
 9760/60000 [===>..........................] - ETA: 1:26 - loss: 0.4666 - categorical_accuracy: 0.8541
 9792/60000 [===>..........................] - ETA: 1:26 - loss: 0.4659 - categorical_accuracy: 0.8544
 9824/60000 [===>..........................] - ETA: 1:26 - loss: 0.4654 - categorical_accuracy: 0.8545
 9856/60000 [===>..........................] - ETA: 1:26 - loss: 0.4645 - categorical_accuracy: 0.8548
 9888/60000 [===>..........................] - ETA: 1:26 - loss: 0.4636 - categorical_accuracy: 0.8551
 9952/60000 [===>..........................] - ETA: 1:26 - loss: 0.4614 - categorical_accuracy: 0.8559
 9984/60000 [===>..........................] - ETA: 1:25 - loss: 0.4603 - categorical_accuracy: 0.8562
10016/60000 [====>.........................] - ETA: 1:25 - loss: 0.4595 - categorical_accuracy: 0.8564
10048/60000 [====>.........................] - ETA: 1:25 - loss: 0.4582 - categorical_accuracy: 0.8569
10080/60000 [====>.........................] - ETA: 1:25 - loss: 0.4572 - categorical_accuracy: 0.8572
10112/60000 [====>.........................] - ETA: 1:25 - loss: 0.4562 - categorical_accuracy: 0.8576
10144/60000 [====>.........................] - ETA: 1:25 - loss: 0.4551 - categorical_accuracy: 0.8578
10176/60000 [====>.........................] - ETA: 1:25 - loss: 0.4540 - categorical_accuracy: 0.8582
10208/60000 [====>.........................] - ETA: 1:25 - loss: 0.4530 - categorical_accuracy: 0.8584
10240/60000 [====>.........................] - ETA: 1:25 - loss: 0.4525 - categorical_accuracy: 0.8586
10272/60000 [====>.........................] - ETA: 1:25 - loss: 0.4525 - categorical_accuracy: 0.8587
10304/60000 [====>.........................] - ETA: 1:25 - loss: 0.4513 - categorical_accuracy: 0.8591
10336/60000 [====>.........................] - ETA: 1:25 - loss: 0.4507 - categorical_accuracy: 0.8590
10368/60000 [====>.........................] - ETA: 1:25 - loss: 0.4498 - categorical_accuracy: 0.8593
10400/60000 [====>.........................] - ETA: 1:25 - loss: 0.4488 - categorical_accuracy: 0.8596
10432/60000 [====>.........................] - ETA: 1:25 - loss: 0.4476 - categorical_accuracy: 0.8600
10464/60000 [====>.........................] - ETA: 1:25 - loss: 0.4464 - categorical_accuracy: 0.8604
10496/60000 [====>.........................] - ETA: 1:25 - loss: 0.4458 - categorical_accuracy: 0.8606
10528/60000 [====>.........................] - ETA: 1:25 - loss: 0.4450 - categorical_accuracy: 0.8608
10560/60000 [====>.........................] - ETA: 1:25 - loss: 0.4442 - categorical_accuracy: 0.8610
10592/60000 [====>.........................] - ETA: 1:25 - loss: 0.4433 - categorical_accuracy: 0.8612
10624/60000 [====>.........................] - ETA: 1:24 - loss: 0.4424 - categorical_accuracy: 0.8615
10656/60000 [====>.........................] - ETA: 1:24 - loss: 0.4415 - categorical_accuracy: 0.8619
10688/60000 [====>.........................] - ETA: 1:24 - loss: 0.4410 - categorical_accuracy: 0.8621
10720/60000 [====>.........................] - ETA: 1:24 - loss: 0.4399 - categorical_accuracy: 0.8625
10752/60000 [====>.........................] - ETA: 1:24 - loss: 0.4398 - categorical_accuracy: 0.8626
10784/60000 [====>.........................] - ETA: 1:24 - loss: 0.4392 - categorical_accuracy: 0.8629
10816/60000 [====>.........................] - ETA: 1:24 - loss: 0.4391 - categorical_accuracy: 0.8630
10848/60000 [====>.........................] - ETA: 1:24 - loss: 0.4381 - categorical_accuracy: 0.8632
10880/60000 [====>.........................] - ETA: 1:24 - loss: 0.4374 - categorical_accuracy: 0.8634
10912/60000 [====>.........................] - ETA: 1:24 - loss: 0.4365 - categorical_accuracy: 0.8637
10944/60000 [====>.........................] - ETA: 1:24 - loss: 0.4353 - categorical_accuracy: 0.8641
10976/60000 [====>.........................] - ETA: 1:24 - loss: 0.4345 - categorical_accuracy: 0.8642
11008/60000 [====>.........................] - ETA: 1:24 - loss: 0.4337 - categorical_accuracy: 0.8646
11040/60000 [====>.........................] - ETA: 1:24 - loss: 0.4327 - categorical_accuracy: 0.8649
11072/60000 [====>.........................] - ETA: 1:24 - loss: 0.4321 - categorical_accuracy: 0.8651
11104/60000 [====>.........................] - ETA: 1:24 - loss: 0.4312 - categorical_accuracy: 0.8654
11136/60000 [====>.........................] - ETA: 1:24 - loss: 0.4310 - categorical_accuracy: 0.8655
11168/60000 [====>.........................] - ETA: 1:23 - loss: 0.4314 - categorical_accuracy: 0.8655
11200/60000 [====>.........................] - ETA: 1:23 - loss: 0.4314 - categorical_accuracy: 0.8655
11232/60000 [====>.........................] - ETA: 1:23 - loss: 0.4307 - categorical_accuracy: 0.8657
11264/60000 [====>.........................] - ETA: 1:23 - loss: 0.4298 - categorical_accuracy: 0.8659
11328/60000 [====>.........................] - ETA: 1:23 - loss: 0.4285 - categorical_accuracy: 0.8663
11360/60000 [====>.........................] - ETA: 1:23 - loss: 0.4276 - categorical_accuracy: 0.8666
11392/60000 [====>.........................] - ETA: 1:23 - loss: 0.4269 - categorical_accuracy: 0.8668
11424/60000 [====>.........................] - ETA: 1:23 - loss: 0.4259 - categorical_accuracy: 0.8671
11456/60000 [====>.........................] - ETA: 1:23 - loss: 0.4250 - categorical_accuracy: 0.8673
11488/60000 [====>.........................] - ETA: 1:23 - loss: 0.4242 - categorical_accuracy: 0.8675
11520/60000 [====>.........................] - ETA: 1:23 - loss: 0.4232 - categorical_accuracy: 0.8678
11552/60000 [====>.........................] - ETA: 1:23 - loss: 0.4222 - categorical_accuracy: 0.8682
11584/60000 [====>.........................] - ETA: 1:23 - loss: 0.4214 - categorical_accuracy: 0.8684
11616/60000 [====>.........................] - ETA: 1:23 - loss: 0.4206 - categorical_accuracy: 0.8686
11648/60000 [====>.........................] - ETA: 1:23 - loss: 0.4195 - categorical_accuracy: 0.8690
11680/60000 [====>.........................] - ETA: 1:23 - loss: 0.4190 - categorical_accuracy: 0.8691
11712/60000 [====>.........................] - ETA: 1:22 - loss: 0.4184 - categorical_accuracy: 0.8692
11744/60000 [====>.........................] - ETA: 1:22 - loss: 0.4173 - categorical_accuracy: 0.8696
11776/60000 [====>.........................] - ETA: 1:22 - loss: 0.4171 - categorical_accuracy: 0.8697
11808/60000 [====>.........................] - ETA: 1:22 - loss: 0.4161 - categorical_accuracy: 0.8700
11840/60000 [====>.........................] - ETA: 1:22 - loss: 0.4155 - categorical_accuracy: 0.8703
11872/60000 [====>.........................] - ETA: 1:22 - loss: 0.4150 - categorical_accuracy: 0.8705
11904/60000 [====>.........................] - ETA: 1:22 - loss: 0.4145 - categorical_accuracy: 0.8706
11936/60000 [====>.........................] - ETA: 1:22 - loss: 0.4138 - categorical_accuracy: 0.8707
11968/60000 [====>.........................] - ETA: 1:22 - loss: 0.4133 - categorical_accuracy: 0.8709
12000/60000 [=====>........................] - ETA: 1:22 - loss: 0.4124 - categorical_accuracy: 0.8712
12032/60000 [=====>........................] - ETA: 1:22 - loss: 0.4116 - categorical_accuracy: 0.8714
12064/60000 [=====>........................] - ETA: 1:22 - loss: 0.4111 - categorical_accuracy: 0.8714
12096/60000 [=====>........................] - ETA: 1:22 - loss: 0.4101 - categorical_accuracy: 0.8718
12128/60000 [=====>........................] - ETA: 1:22 - loss: 0.4092 - categorical_accuracy: 0.8721
12192/60000 [=====>........................] - ETA: 1:22 - loss: 0.4074 - categorical_accuracy: 0.8727
12224/60000 [=====>........................] - ETA: 1:22 - loss: 0.4067 - categorical_accuracy: 0.8730
12256/60000 [=====>........................] - ETA: 1:21 - loss: 0.4062 - categorical_accuracy: 0.8730
12320/60000 [=====>........................] - ETA: 1:21 - loss: 0.4048 - categorical_accuracy: 0.8733
12352/60000 [=====>........................] - ETA: 1:21 - loss: 0.4040 - categorical_accuracy: 0.8735
12384/60000 [=====>........................] - ETA: 1:21 - loss: 0.4041 - categorical_accuracy: 0.8735
12416/60000 [=====>........................] - ETA: 1:21 - loss: 0.4034 - categorical_accuracy: 0.8736
12448/60000 [=====>........................] - ETA: 1:21 - loss: 0.4025 - categorical_accuracy: 0.8740
12480/60000 [=====>........................] - ETA: 1:21 - loss: 0.4023 - categorical_accuracy: 0.8740
12512/60000 [=====>........................] - ETA: 1:21 - loss: 0.4018 - categorical_accuracy: 0.8742
12544/60000 [=====>........................] - ETA: 1:21 - loss: 0.4020 - categorical_accuracy: 0.8742
12576/60000 [=====>........................] - ETA: 1:21 - loss: 0.4014 - categorical_accuracy: 0.8743
12608/60000 [=====>........................] - ETA: 1:21 - loss: 0.4010 - categorical_accuracy: 0.8743
12640/60000 [=====>........................] - ETA: 1:21 - loss: 0.4006 - categorical_accuracy: 0.8744
12672/60000 [=====>........................] - ETA: 1:21 - loss: 0.3999 - categorical_accuracy: 0.8747
12704/60000 [=====>........................] - ETA: 1:21 - loss: 0.3993 - categorical_accuracy: 0.8749
12736/60000 [=====>........................] - ETA: 1:21 - loss: 0.3988 - categorical_accuracy: 0.8749
12768/60000 [=====>........................] - ETA: 1:21 - loss: 0.3980 - categorical_accuracy: 0.8752
12800/60000 [=====>........................] - ETA: 1:21 - loss: 0.3977 - categorical_accuracy: 0.8753
12832/60000 [=====>........................] - ETA: 1:20 - loss: 0.3970 - categorical_accuracy: 0.8755
12864/60000 [=====>........................] - ETA: 1:20 - loss: 0.3965 - categorical_accuracy: 0.8756
12896/60000 [=====>........................] - ETA: 1:20 - loss: 0.3959 - categorical_accuracy: 0.8758
12928/60000 [=====>........................] - ETA: 1:20 - loss: 0.3958 - categorical_accuracy: 0.8758
12960/60000 [=====>........................] - ETA: 1:20 - loss: 0.3954 - categorical_accuracy: 0.8758
12992/60000 [=====>........................] - ETA: 1:20 - loss: 0.3948 - categorical_accuracy: 0.8760
13024/60000 [=====>........................] - ETA: 1:20 - loss: 0.3948 - categorical_accuracy: 0.8761
13056/60000 [=====>........................] - ETA: 1:20 - loss: 0.3938 - categorical_accuracy: 0.8764
13088/60000 [=====>........................] - ETA: 1:20 - loss: 0.3937 - categorical_accuracy: 0.8766
13120/60000 [=====>........................] - ETA: 1:20 - loss: 0.3943 - categorical_accuracy: 0.8765
13152/60000 [=====>........................] - ETA: 1:20 - loss: 0.3935 - categorical_accuracy: 0.8768
13184/60000 [=====>........................] - ETA: 1:20 - loss: 0.3930 - categorical_accuracy: 0.8770
13216/60000 [=====>........................] - ETA: 1:20 - loss: 0.3923 - categorical_accuracy: 0.8772
13280/60000 [=====>........................] - ETA: 1:20 - loss: 0.3916 - categorical_accuracy: 0.8775
13312/60000 [=====>........................] - ETA: 1:20 - loss: 0.3910 - categorical_accuracy: 0.8777
13344/60000 [=====>........................] - ETA: 1:20 - loss: 0.3902 - categorical_accuracy: 0.8780
13376/60000 [=====>........................] - ETA: 1:20 - loss: 0.3904 - categorical_accuracy: 0.8781
13408/60000 [=====>........................] - ETA: 1:19 - loss: 0.3903 - categorical_accuracy: 0.8781
13440/60000 [=====>........................] - ETA: 1:19 - loss: 0.3899 - categorical_accuracy: 0.8783
13472/60000 [=====>........................] - ETA: 1:19 - loss: 0.3892 - categorical_accuracy: 0.8785
13504/60000 [=====>........................] - ETA: 1:19 - loss: 0.3886 - categorical_accuracy: 0.8787
13536/60000 [=====>........................] - ETA: 1:19 - loss: 0.3883 - categorical_accuracy: 0.8788
13568/60000 [=====>........................] - ETA: 1:19 - loss: 0.3875 - categorical_accuracy: 0.8791
13600/60000 [=====>........................] - ETA: 1:19 - loss: 0.3870 - categorical_accuracy: 0.8791
13632/60000 [=====>........................] - ETA: 1:19 - loss: 0.3864 - categorical_accuracy: 0.8793
13664/60000 [=====>........................] - ETA: 1:19 - loss: 0.3864 - categorical_accuracy: 0.8792
13696/60000 [=====>........................] - ETA: 1:19 - loss: 0.3860 - categorical_accuracy: 0.8795
13728/60000 [=====>........................] - ETA: 1:19 - loss: 0.3864 - categorical_accuracy: 0.8793
13760/60000 [=====>........................] - ETA: 1:19 - loss: 0.3857 - categorical_accuracy: 0.8796
13792/60000 [=====>........................] - ETA: 1:19 - loss: 0.3856 - categorical_accuracy: 0.8798
13824/60000 [=====>........................] - ETA: 1:19 - loss: 0.3851 - categorical_accuracy: 0.8798
13856/60000 [=====>........................] - ETA: 1:19 - loss: 0.3844 - categorical_accuracy: 0.8801
13888/60000 [=====>........................] - ETA: 1:19 - loss: 0.3842 - categorical_accuracy: 0.8800
13920/60000 [=====>........................] - ETA: 1:19 - loss: 0.3835 - categorical_accuracy: 0.8802
13952/60000 [=====>........................] - ETA: 1:19 - loss: 0.3829 - categorical_accuracy: 0.8804
13984/60000 [=====>........................] - ETA: 1:18 - loss: 0.3825 - categorical_accuracy: 0.8806
14016/60000 [======>.......................] - ETA: 1:18 - loss: 0.3817 - categorical_accuracy: 0.8809
14048/60000 [======>.......................] - ETA: 1:18 - loss: 0.3812 - categorical_accuracy: 0.8811
14080/60000 [======>.......................] - ETA: 1:18 - loss: 0.3806 - categorical_accuracy: 0.8813
14112/60000 [======>.......................] - ETA: 1:18 - loss: 0.3802 - categorical_accuracy: 0.8814
14144/60000 [======>.......................] - ETA: 1:18 - loss: 0.3794 - categorical_accuracy: 0.8816
14176/60000 [======>.......................] - ETA: 1:18 - loss: 0.3790 - categorical_accuracy: 0.8817
14208/60000 [======>.......................] - ETA: 1:18 - loss: 0.3786 - categorical_accuracy: 0.8818
14240/60000 [======>.......................] - ETA: 1:18 - loss: 0.3783 - categorical_accuracy: 0.8820
14272/60000 [======>.......................] - ETA: 1:18 - loss: 0.3779 - categorical_accuracy: 0.8821
14304/60000 [======>.......................] - ETA: 1:18 - loss: 0.3777 - categorical_accuracy: 0.8822
14336/60000 [======>.......................] - ETA: 1:18 - loss: 0.3771 - categorical_accuracy: 0.8824
14368/60000 [======>.......................] - ETA: 1:18 - loss: 0.3767 - categorical_accuracy: 0.8826
14400/60000 [======>.......................] - ETA: 1:18 - loss: 0.3764 - categorical_accuracy: 0.8827
14432/60000 [======>.......................] - ETA: 1:18 - loss: 0.3762 - categorical_accuracy: 0.8828
14464/60000 [======>.......................] - ETA: 1:18 - loss: 0.3755 - categorical_accuracy: 0.8830
14496/60000 [======>.......................] - ETA: 1:18 - loss: 0.3752 - categorical_accuracy: 0.8831
14528/60000 [======>.......................] - ETA: 1:18 - loss: 0.3744 - categorical_accuracy: 0.8834
14560/60000 [======>.......................] - ETA: 1:18 - loss: 0.3743 - categorical_accuracy: 0.8835
14592/60000 [======>.......................] - ETA: 1:18 - loss: 0.3740 - categorical_accuracy: 0.8836
14624/60000 [======>.......................] - ETA: 1:18 - loss: 0.3735 - categorical_accuracy: 0.8837
14656/60000 [======>.......................] - ETA: 1:17 - loss: 0.3729 - categorical_accuracy: 0.8839
14688/60000 [======>.......................] - ETA: 1:17 - loss: 0.3730 - categorical_accuracy: 0.8840
14720/60000 [======>.......................] - ETA: 1:17 - loss: 0.3726 - categorical_accuracy: 0.8841
14752/60000 [======>.......................] - ETA: 1:17 - loss: 0.3726 - categorical_accuracy: 0.8842
14784/60000 [======>.......................] - ETA: 1:17 - loss: 0.3720 - categorical_accuracy: 0.8845
14816/60000 [======>.......................] - ETA: 1:17 - loss: 0.3715 - categorical_accuracy: 0.8847
14880/60000 [======>.......................] - ETA: 1:17 - loss: 0.3702 - categorical_accuracy: 0.8851
14912/60000 [======>.......................] - ETA: 1:17 - loss: 0.3694 - categorical_accuracy: 0.8853
14944/60000 [======>.......................] - ETA: 1:17 - loss: 0.3689 - categorical_accuracy: 0.8855
14976/60000 [======>.......................] - ETA: 1:17 - loss: 0.3686 - categorical_accuracy: 0.8856
15008/60000 [======>.......................] - ETA: 1:17 - loss: 0.3683 - categorical_accuracy: 0.8857
15040/60000 [======>.......................] - ETA: 1:17 - loss: 0.3679 - categorical_accuracy: 0.8858
15072/60000 [======>.......................] - ETA: 1:17 - loss: 0.3674 - categorical_accuracy: 0.8859
15104/60000 [======>.......................] - ETA: 1:17 - loss: 0.3669 - categorical_accuracy: 0.8861
15136/60000 [======>.......................] - ETA: 1:17 - loss: 0.3668 - categorical_accuracy: 0.8862
15168/60000 [======>.......................] - ETA: 1:17 - loss: 0.3668 - categorical_accuracy: 0.8862
15200/60000 [======>.......................] - ETA: 1:17 - loss: 0.3667 - categorical_accuracy: 0.8862
15232/60000 [======>.......................] - ETA: 1:16 - loss: 0.3660 - categorical_accuracy: 0.8864
15264/60000 [======>.......................] - ETA: 1:16 - loss: 0.3655 - categorical_accuracy: 0.8865
15296/60000 [======>.......................] - ETA: 1:16 - loss: 0.3650 - categorical_accuracy: 0.8866
15328/60000 [======>.......................] - ETA: 1:16 - loss: 0.3647 - categorical_accuracy: 0.8867
15360/60000 [======>.......................] - ETA: 1:16 - loss: 0.3647 - categorical_accuracy: 0.8868
15392/60000 [======>.......................] - ETA: 1:16 - loss: 0.3642 - categorical_accuracy: 0.8869
15424/60000 [======>.......................] - ETA: 1:16 - loss: 0.3637 - categorical_accuracy: 0.8870
15456/60000 [======>.......................] - ETA: 1:16 - loss: 0.3631 - categorical_accuracy: 0.8872
15488/60000 [======>.......................] - ETA: 1:16 - loss: 0.3625 - categorical_accuracy: 0.8874
15520/60000 [======>.......................] - ETA: 1:16 - loss: 0.3624 - categorical_accuracy: 0.8875
15552/60000 [======>.......................] - ETA: 1:16 - loss: 0.3622 - categorical_accuracy: 0.8876
15584/60000 [======>.......................] - ETA: 1:16 - loss: 0.3615 - categorical_accuracy: 0.8878
15616/60000 [======>.......................] - ETA: 1:16 - loss: 0.3616 - categorical_accuracy: 0.8879
15680/60000 [======>.......................] - ETA: 1:16 - loss: 0.3609 - categorical_accuracy: 0.8880
15712/60000 [======>.......................] - ETA: 1:16 - loss: 0.3605 - categorical_accuracy: 0.8881
15744/60000 [======>.......................] - ETA: 1:16 - loss: 0.3603 - categorical_accuracy: 0.8881
15776/60000 [======>.......................] - ETA: 1:15 - loss: 0.3597 - categorical_accuracy: 0.8884
15840/60000 [======>.......................] - ETA: 1:15 - loss: 0.3587 - categorical_accuracy: 0.8887
15872/60000 [======>.......................] - ETA: 1:15 - loss: 0.3585 - categorical_accuracy: 0.8887
15904/60000 [======>.......................] - ETA: 1:15 - loss: 0.3585 - categorical_accuracy: 0.8887
15936/60000 [======>.......................] - ETA: 1:15 - loss: 0.3579 - categorical_accuracy: 0.8889
15968/60000 [======>.......................] - ETA: 1:15 - loss: 0.3573 - categorical_accuracy: 0.8891
16000/60000 [=======>......................] - ETA: 1:15 - loss: 0.3569 - categorical_accuracy: 0.8891
16032/60000 [=======>......................] - ETA: 1:15 - loss: 0.3565 - categorical_accuracy: 0.8893
16064/60000 [=======>......................] - ETA: 1:15 - loss: 0.3560 - categorical_accuracy: 0.8894
16096/60000 [=======>......................] - ETA: 1:15 - loss: 0.3557 - categorical_accuracy: 0.8895
16128/60000 [=======>......................] - ETA: 1:15 - loss: 0.3553 - categorical_accuracy: 0.8896
16160/60000 [=======>......................] - ETA: 1:15 - loss: 0.3547 - categorical_accuracy: 0.8898
16192/60000 [=======>......................] - ETA: 1:15 - loss: 0.3541 - categorical_accuracy: 0.8900
16224/60000 [=======>......................] - ETA: 1:15 - loss: 0.3535 - categorical_accuracy: 0.8902
16256/60000 [=======>......................] - ETA: 1:15 - loss: 0.3535 - categorical_accuracy: 0.8902
16288/60000 [=======>......................] - ETA: 1:15 - loss: 0.3531 - categorical_accuracy: 0.8903
16352/60000 [=======>......................] - ETA: 1:14 - loss: 0.3519 - categorical_accuracy: 0.8907
16384/60000 [=======>......................] - ETA: 1:14 - loss: 0.3513 - categorical_accuracy: 0.8909
16416/60000 [=======>......................] - ETA: 1:14 - loss: 0.3509 - categorical_accuracy: 0.8910
16448/60000 [=======>......................] - ETA: 1:14 - loss: 0.3504 - categorical_accuracy: 0.8912
16480/60000 [=======>......................] - ETA: 1:14 - loss: 0.3501 - categorical_accuracy: 0.8913
16512/60000 [=======>......................] - ETA: 1:14 - loss: 0.3495 - categorical_accuracy: 0.8915
16544/60000 [=======>......................] - ETA: 1:14 - loss: 0.3495 - categorical_accuracy: 0.8914
16576/60000 [=======>......................] - ETA: 1:14 - loss: 0.3493 - categorical_accuracy: 0.8915
16608/60000 [=======>......................] - ETA: 1:14 - loss: 0.3487 - categorical_accuracy: 0.8916
16640/60000 [=======>......................] - ETA: 1:14 - loss: 0.3491 - categorical_accuracy: 0.8915
16672/60000 [=======>......................] - ETA: 1:14 - loss: 0.3492 - categorical_accuracy: 0.8916
16704/60000 [=======>......................] - ETA: 1:14 - loss: 0.3491 - categorical_accuracy: 0.8916
16736/60000 [=======>......................] - ETA: 1:14 - loss: 0.3485 - categorical_accuracy: 0.8918
16768/60000 [=======>......................] - ETA: 1:14 - loss: 0.3479 - categorical_accuracy: 0.8921
16800/60000 [=======>......................] - ETA: 1:14 - loss: 0.3479 - categorical_accuracy: 0.8920
16832/60000 [=======>......................] - ETA: 1:14 - loss: 0.3476 - categorical_accuracy: 0.8921
16864/60000 [=======>......................] - ETA: 1:14 - loss: 0.3477 - categorical_accuracy: 0.8921
16896/60000 [=======>......................] - ETA: 1:13 - loss: 0.3473 - categorical_accuracy: 0.8923
16928/60000 [=======>......................] - ETA: 1:13 - loss: 0.3476 - categorical_accuracy: 0.8923
16960/60000 [=======>......................] - ETA: 1:13 - loss: 0.3471 - categorical_accuracy: 0.8925
16992/60000 [=======>......................] - ETA: 1:13 - loss: 0.3467 - categorical_accuracy: 0.8926
17024/60000 [=======>......................] - ETA: 1:13 - loss: 0.3462 - categorical_accuracy: 0.8927
17056/60000 [=======>......................] - ETA: 1:13 - loss: 0.3458 - categorical_accuracy: 0.8928
17088/60000 [=======>......................] - ETA: 1:13 - loss: 0.3455 - categorical_accuracy: 0.8929
17120/60000 [=======>......................] - ETA: 1:13 - loss: 0.3452 - categorical_accuracy: 0.8930
17184/60000 [=======>......................] - ETA: 1:13 - loss: 0.3448 - categorical_accuracy: 0.8932
17216/60000 [=======>......................] - ETA: 1:13 - loss: 0.3444 - categorical_accuracy: 0.8933
17248/60000 [=======>......................] - ETA: 1:13 - loss: 0.3441 - categorical_accuracy: 0.8934
17280/60000 [=======>......................] - ETA: 1:13 - loss: 0.3437 - categorical_accuracy: 0.8935
17312/60000 [=======>......................] - ETA: 1:13 - loss: 0.3432 - categorical_accuracy: 0.8937
17344/60000 [=======>......................] - ETA: 1:13 - loss: 0.3427 - categorical_accuracy: 0.8938
17376/60000 [=======>......................] - ETA: 1:13 - loss: 0.3423 - categorical_accuracy: 0.8939
17408/60000 [=======>......................] - ETA: 1:13 - loss: 0.3419 - categorical_accuracy: 0.8940
17440/60000 [=======>......................] - ETA: 1:13 - loss: 0.3415 - categorical_accuracy: 0.8941
17472/60000 [=======>......................] - ETA: 1:12 - loss: 0.3412 - categorical_accuracy: 0.8942
17504/60000 [=======>......................] - ETA: 1:12 - loss: 0.3407 - categorical_accuracy: 0.8944
17536/60000 [=======>......................] - ETA: 1:12 - loss: 0.3403 - categorical_accuracy: 0.8945
17568/60000 [=======>......................] - ETA: 1:12 - loss: 0.3402 - categorical_accuracy: 0.8945
17600/60000 [=======>......................] - ETA: 1:12 - loss: 0.3400 - categorical_accuracy: 0.8946
17632/60000 [=======>......................] - ETA: 1:12 - loss: 0.3397 - categorical_accuracy: 0.8946
17664/60000 [=======>......................] - ETA: 1:12 - loss: 0.3395 - categorical_accuracy: 0.8946
17696/60000 [=======>......................] - ETA: 1:12 - loss: 0.3391 - categorical_accuracy: 0.8948
17728/60000 [=======>......................] - ETA: 1:12 - loss: 0.3388 - categorical_accuracy: 0.8949
17760/60000 [=======>......................] - ETA: 1:12 - loss: 0.3386 - categorical_accuracy: 0.8949
17792/60000 [=======>......................] - ETA: 1:12 - loss: 0.3380 - categorical_accuracy: 0.8951
17856/60000 [=======>......................] - ETA: 1:12 - loss: 0.3375 - categorical_accuracy: 0.8952
17888/60000 [=======>......................] - ETA: 1:12 - loss: 0.3374 - categorical_accuracy: 0.8953
17920/60000 [=======>......................] - ETA: 1:12 - loss: 0.3375 - categorical_accuracy: 0.8954
17984/60000 [=======>......................] - ETA: 1:12 - loss: 0.3368 - categorical_accuracy: 0.8956
18016/60000 [========>.....................] - ETA: 1:11 - loss: 0.3363 - categorical_accuracy: 0.8958
18048/60000 [========>.....................] - ETA: 1:11 - loss: 0.3360 - categorical_accuracy: 0.8958
18080/60000 [========>.....................] - ETA: 1:11 - loss: 0.3355 - categorical_accuracy: 0.8960
18112/60000 [========>.....................] - ETA: 1:11 - loss: 0.3352 - categorical_accuracy: 0.8960
18144/60000 [========>.....................] - ETA: 1:11 - loss: 0.3349 - categorical_accuracy: 0.8961
18176/60000 [========>.....................] - ETA: 1:11 - loss: 0.3345 - categorical_accuracy: 0.8962
18208/60000 [========>.....................] - ETA: 1:11 - loss: 0.3344 - categorical_accuracy: 0.8961
18240/60000 [========>.....................] - ETA: 1:11 - loss: 0.3339 - categorical_accuracy: 0.8963
18272/60000 [========>.....................] - ETA: 1:11 - loss: 0.3334 - categorical_accuracy: 0.8964
18304/60000 [========>.....................] - ETA: 1:11 - loss: 0.3330 - categorical_accuracy: 0.8965
18336/60000 [========>.....................] - ETA: 1:11 - loss: 0.3326 - categorical_accuracy: 0.8966
18368/60000 [========>.....................] - ETA: 1:11 - loss: 0.3323 - categorical_accuracy: 0.8967
18400/60000 [========>.....................] - ETA: 1:11 - loss: 0.3321 - categorical_accuracy: 0.8967
18432/60000 [========>.....................] - ETA: 1:11 - loss: 0.3315 - categorical_accuracy: 0.8969
18464/60000 [========>.....................] - ETA: 1:11 - loss: 0.3315 - categorical_accuracy: 0.8969
18496/60000 [========>.....................] - ETA: 1:11 - loss: 0.3311 - categorical_accuracy: 0.8971
18528/60000 [========>.....................] - ETA: 1:11 - loss: 0.3308 - categorical_accuracy: 0.8972
18560/60000 [========>.....................] - ETA: 1:11 - loss: 0.3309 - categorical_accuracy: 0.8973
18592/60000 [========>.....................] - ETA: 1:11 - loss: 0.3307 - categorical_accuracy: 0.8973
18624/60000 [========>.....................] - ETA: 1:11 - loss: 0.3303 - categorical_accuracy: 0.8973
18656/60000 [========>.....................] - ETA: 1:10 - loss: 0.3299 - categorical_accuracy: 0.8975
18688/60000 [========>.....................] - ETA: 1:10 - loss: 0.3295 - categorical_accuracy: 0.8976
18720/60000 [========>.....................] - ETA: 1:10 - loss: 0.3291 - categorical_accuracy: 0.8976
18752/60000 [========>.....................] - ETA: 1:10 - loss: 0.3293 - categorical_accuracy: 0.8977
18784/60000 [========>.....................] - ETA: 1:10 - loss: 0.3288 - categorical_accuracy: 0.8978
18816/60000 [========>.....................] - ETA: 1:10 - loss: 0.3283 - categorical_accuracy: 0.8980
18848/60000 [========>.....................] - ETA: 1:10 - loss: 0.3279 - categorical_accuracy: 0.8982
18880/60000 [========>.....................] - ETA: 1:10 - loss: 0.3275 - categorical_accuracy: 0.8983
18912/60000 [========>.....................] - ETA: 1:10 - loss: 0.3271 - categorical_accuracy: 0.8984
18944/60000 [========>.....................] - ETA: 1:10 - loss: 0.3268 - categorical_accuracy: 0.8985
18976/60000 [========>.....................] - ETA: 1:10 - loss: 0.3266 - categorical_accuracy: 0.8986
19008/60000 [========>.....................] - ETA: 1:10 - loss: 0.3261 - categorical_accuracy: 0.8987
19040/60000 [========>.....................] - ETA: 1:10 - loss: 0.3260 - categorical_accuracy: 0.8987
19072/60000 [========>.....................] - ETA: 1:10 - loss: 0.3257 - categorical_accuracy: 0.8988
19104/60000 [========>.....................] - ETA: 1:10 - loss: 0.3255 - categorical_accuracy: 0.8989
19136/60000 [========>.....................] - ETA: 1:10 - loss: 0.3250 - categorical_accuracy: 0.8990
19168/60000 [========>.....................] - ETA: 1:10 - loss: 0.3248 - categorical_accuracy: 0.8991
19200/60000 [========>.....................] - ETA: 1:10 - loss: 0.3246 - categorical_accuracy: 0.8992
19232/60000 [========>.....................] - ETA: 1:09 - loss: 0.3242 - categorical_accuracy: 0.8993
19264/60000 [========>.....................] - ETA: 1:09 - loss: 0.3238 - categorical_accuracy: 0.8994
19296/60000 [========>.....................] - ETA: 1:09 - loss: 0.3234 - categorical_accuracy: 0.8996
19328/60000 [========>.....................] - ETA: 1:09 - loss: 0.3230 - categorical_accuracy: 0.8997
19360/60000 [========>.....................] - ETA: 1:09 - loss: 0.3226 - categorical_accuracy: 0.8998
19392/60000 [========>.....................] - ETA: 1:09 - loss: 0.3222 - categorical_accuracy: 0.8999
19424/60000 [========>.....................] - ETA: 1:09 - loss: 0.3220 - categorical_accuracy: 0.9000
19456/60000 [========>.....................] - ETA: 1:09 - loss: 0.3218 - categorical_accuracy: 0.9000
19488/60000 [========>.....................] - ETA: 1:09 - loss: 0.3213 - categorical_accuracy: 0.9002
19520/60000 [========>.....................] - ETA: 1:09 - loss: 0.3209 - categorical_accuracy: 0.9003
19552/60000 [========>.....................] - ETA: 1:09 - loss: 0.3205 - categorical_accuracy: 0.9004
19584/60000 [========>.....................] - ETA: 1:09 - loss: 0.3201 - categorical_accuracy: 0.9005
19616/60000 [========>.....................] - ETA: 1:09 - loss: 0.3198 - categorical_accuracy: 0.9006
19648/60000 [========>.....................] - ETA: 1:09 - loss: 0.3195 - categorical_accuracy: 0.9007
19680/60000 [========>.....................] - ETA: 1:09 - loss: 0.3192 - categorical_accuracy: 0.9008
19712/60000 [========>.....................] - ETA: 1:09 - loss: 0.3188 - categorical_accuracy: 0.9009
19744/60000 [========>.....................] - ETA: 1:09 - loss: 0.3187 - categorical_accuracy: 0.9010
19776/60000 [========>.....................] - ETA: 1:09 - loss: 0.3185 - categorical_accuracy: 0.9010
19808/60000 [========>.....................] - ETA: 1:08 - loss: 0.3180 - categorical_accuracy: 0.9012
19840/60000 [========>.....................] - ETA: 1:08 - loss: 0.3176 - categorical_accuracy: 0.9013
19872/60000 [========>.....................] - ETA: 1:08 - loss: 0.3174 - categorical_accuracy: 0.9014
19904/60000 [========>.....................] - ETA: 1:08 - loss: 0.3173 - categorical_accuracy: 0.9014
19936/60000 [========>.....................] - ETA: 1:08 - loss: 0.3171 - categorical_accuracy: 0.9015
19968/60000 [========>.....................] - ETA: 1:08 - loss: 0.3167 - categorical_accuracy: 0.9016
20000/60000 [=========>....................] - ETA: 1:08 - loss: 0.3166 - categorical_accuracy: 0.9017
20032/60000 [=========>....................] - ETA: 1:08 - loss: 0.3162 - categorical_accuracy: 0.9018
20064/60000 [=========>....................] - ETA: 1:08 - loss: 0.3157 - categorical_accuracy: 0.9020
20096/60000 [=========>....................] - ETA: 1:08 - loss: 0.3156 - categorical_accuracy: 0.9020
20128/60000 [=========>....................] - ETA: 1:08 - loss: 0.3151 - categorical_accuracy: 0.9021
20160/60000 [=========>....................] - ETA: 1:08 - loss: 0.3149 - categorical_accuracy: 0.9022
20192/60000 [=========>....................] - ETA: 1:08 - loss: 0.3145 - categorical_accuracy: 0.9024
20224/60000 [=========>....................] - ETA: 1:08 - loss: 0.3142 - categorical_accuracy: 0.9025
20256/60000 [=========>....................] - ETA: 1:08 - loss: 0.3139 - categorical_accuracy: 0.9026
20288/60000 [=========>....................] - ETA: 1:08 - loss: 0.3136 - categorical_accuracy: 0.9027
20320/60000 [=========>....................] - ETA: 1:08 - loss: 0.3136 - categorical_accuracy: 0.9027
20352/60000 [=========>....................] - ETA: 1:08 - loss: 0.3134 - categorical_accuracy: 0.9028
20384/60000 [=========>....................] - ETA: 1:07 - loss: 0.3135 - categorical_accuracy: 0.9027
20416/60000 [=========>....................] - ETA: 1:07 - loss: 0.3133 - categorical_accuracy: 0.9028
20448/60000 [=========>....................] - ETA: 1:07 - loss: 0.3130 - categorical_accuracy: 0.9029
20480/60000 [=========>....................] - ETA: 1:07 - loss: 0.3126 - categorical_accuracy: 0.9030
20512/60000 [=========>....................] - ETA: 1:07 - loss: 0.3123 - categorical_accuracy: 0.9031
20544/60000 [=========>....................] - ETA: 1:07 - loss: 0.3122 - categorical_accuracy: 0.9031
20576/60000 [=========>....................] - ETA: 1:07 - loss: 0.3121 - categorical_accuracy: 0.9032
20608/60000 [=========>....................] - ETA: 1:07 - loss: 0.3117 - categorical_accuracy: 0.9033
20640/60000 [=========>....................] - ETA: 1:07 - loss: 0.3114 - categorical_accuracy: 0.9034
20672/60000 [=========>....................] - ETA: 1:07 - loss: 0.3109 - categorical_accuracy: 0.9035
20704/60000 [=========>....................] - ETA: 1:07 - loss: 0.3107 - categorical_accuracy: 0.9036
20736/60000 [=========>....................] - ETA: 1:07 - loss: 0.3102 - categorical_accuracy: 0.9037
20768/60000 [=========>....................] - ETA: 1:07 - loss: 0.3099 - categorical_accuracy: 0.9038
20800/60000 [=========>....................] - ETA: 1:07 - loss: 0.3097 - categorical_accuracy: 0.9038
20832/60000 [=========>....................] - ETA: 1:07 - loss: 0.3094 - categorical_accuracy: 0.9039
20864/60000 [=========>....................] - ETA: 1:07 - loss: 0.3092 - categorical_accuracy: 0.9040
20896/60000 [=========>....................] - ETA: 1:07 - loss: 0.3094 - categorical_accuracy: 0.9040
20928/60000 [=========>....................] - ETA: 1:06 - loss: 0.3090 - categorical_accuracy: 0.9042
20960/60000 [=========>....................] - ETA: 1:06 - loss: 0.3093 - categorical_accuracy: 0.9041
20992/60000 [=========>....................] - ETA: 1:06 - loss: 0.3094 - categorical_accuracy: 0.9040
21024/60000 [=========>....................] - ETA: 1:06 - loss: 0.3090 - categorical_accuracy: 0.9041
21056/60000 [=========>....................] - ETA: 1:06 - loss: 0.3087 - categorical_accuracy: 0.9042
21088/60000 [=========>....................] - ETA: 1:06 - loss: 0.3086 - categorical_accuracy: 0.9043
21120/60000 [=========>....................] - ETA: 1:06 - loss: 0.3083 - categorical_accuracy: 0.9044
21152/60000 [=========>....................] - ETA: 1:06 - loss: 0.3084 - categorical_accuracy: 0.9044
21184/60000 [=========>....................] - ETA: 1:06 - loss: 0.3082 - categorical_accuracy: 0.9044
21216/60000 [=========>....................] - ETA: 1:06 - loss: 0.3077 - categorical_accuracy: 0.9046
21248/60000 [=========>....................] - ETA: 1:06 - loss: 0.3075 - categorical_accuracy: 0.9046
21280/60000 [=========>....................] - ETA: 1:06 - loss: 0.3073 - categorical_accuracy: 0.9047
21312/60000 [=========>....................] - ETA: 1:06 - loss: 0.3073 - categorical_accuracy: 0.9047
21344/60000 [=========>....................] - ETA: 1:06 - loss: 0.3069 - categorical_accuracy: 0.9048
21376/60000 [=========>....................] - ETA: 1:06 - loss: 0.3066 - categorical_accuracy: 0.9049
21408/60000 [=========>....................] - ETA: 1:06 - loss: 0.3065 - categorical_accuracy: 0.9049
21440/60000 [=========>....................] - ETA: 1:06 - loss: 0.3062 - categorical_accuracy: 0.9050
21472/60000 [=========>....................] - ETA: 1:06 - loss: 0.3058 - categorical_accuracy: 0.9051
21504/60000 [=========>....................] - ETA: 1:06 - loss: 0.3054 - categorical_accuracy: 0.9053
21536/60000 [=========>....................] - ETA: 1:06 - loss: 0.3050 - categorical_accuracy: 0.9054
21568/60000 [=========>....................] - ETA: 1:06 - loss: 0.3048 - categorical_accuracy: 0.9055
21600/60000 [=========>....................] - ETA: 1:05 - loss: 0.3045 - categorical_accuracy: 0.9056
21632/60000 [=========>....................] - ETA: 1:05 - loss: 0.3044 - categorical_accuracy: 0.9056
21664/60000 [=========>....................] - ETA: 1:05 - loss: 0.3042 - categorical_accuracy: 0.9056
21696/60000 [=========>....................] - ETA: 1:05 - loss: 0.3041 - categorical_accuracy: 0.9057
21728/60000 [=========>....................] - ETA: 1:05 - loss: 0.3042 - categorical_accuracy: 0.9057
21760/60000 [=========>....................] - ETA: 1:05 - loss: 0.3041 - categorical_accuracy: 0.9057
21792/60000 [=========>....................] - ETA: 1:05 - loss: 0.3038 - categorical_accuracy: 0.9057
21824/60000 [=========>....................] - ETA: 1:05 - loss: 0.3035 - categorical_accuracy: 0.9058
21856/60000 [=========>....................] - ETA: 1:05 - loss: 0.3037 - categorical_accuracy: 0.9058
21888/60000 [=========>....................] - ETA: 1:05 - loss: 0.3036 - categorical_accuracy: 0.9058
21920/60000 [=========>....................] - ETA: 1:05 - loss: 0.3033 - categorical_accuracy: 0.9059
21952/60000 [=========>....................] - ETA: 1:05 - loss: 0.3029 - categorical_accuracy: 0.9060
21984/60000 [=========>....................] - ETA: 1:05 - loss: 0.3027 - categorical_accuracy: 0.9060
22016/60000 [==========>...................] - ETA: 1:05 - loss: 0.3027 - categorical_accuracy: 0.9060
22048/60000 [==========>...................] - ETA: 1:05 - loss: 0.3024 - categorical_accuracy: 0.9061
22080/60000 [==========>...................] - ETA: 1:05 - loss: 0.3021 - categorical_accuracy: 0.9062
22112/60000 [==========>...................] - ETA: 1:05 - loss: 0.3019 - categorical_accuracy: 0.9063
22144/60000 [==========>...................] - ETA: 1:04 - loss: 0.3019 - categorical_accuracy: 0.9062
22176/60000 [==========>...................] - ETA: 1:04 - loss: 0.3020 - categorical_accuracy: 0.9062
22208/60000 [==========>...................] - ETA: 1:04 - loss: 0.3016 - categorical_accuracy: 0.9064
22240/60000 [==========>...................] - ETA: 1:04 - loss: 0.3013 - categorical_accuracy: 0.9065
22272/60000 [==========>...................] - ETA: 1:04 - loss: 0.3010 - categorical_accuracy: 0.9066
22304/60000 [==========>...................] - ETA: 1:04 - loss: 0.3007 - categorical_accuracy: 0.9067
22336/60000 [==========>...................] - ETA: 1:04 - loss: 0.3006 - categorical_accuracy: 0.9067
22368/60000 [==========>...................] - ETA: 1:04 - loss: 0.3003 - categorical_accuracy: 0.9068
22400/60000 [==========>...................] - ETA: 1:04 - loss: 0.3000 - categorical_accuracy: 0.9069
22432/60000 [==========>...................] - ETA: 1:04 - loss: 0.3005 - categorical_accuracy: 0.9070
22464/60000 [==========>...................] - ETA: 1:04 - loss: 0.3004 - categorical_accuracy: 0.9069
22496/60000 [==========>...................] - ETA: 1:04 - loss: 0.3002 - categorical_accuracy: 0.9070
22528/60000 [==========>...................] - ETA: 1:04 - loss: 0.2999 - categorical_accuracy: 0.9071
22560/60000 [==========>...................] - ETA: 1:04 - loss: 0.2998 - categorical_accuracy: 0.9071
22592/60000 [==========>...................] - ETA: 1:04 - loss: 0.2995 - categorical_accuracy: 0.9072
22624/60000 [==========>...................] - ETA: 1:04 - loss: 0.2996 - categorical_accuracy: 0.9072
22656/60000 [==========>...................] - ETA: 1:04 - loss: 0.2992 - categorical_accuracy: 0.9073
22688/60000 [==========>...................] - ETA: 1:04 - loss: 0.2992 - categorical_accuracy: 0.9074
22720/60000 [==========>...................] - ETA: 1:03 - loss: 0.2988 - categorical_accuracy: 0.9075
22752/60000 [==========>...................] - ETA: 1:03 - loss: 0.2987 - categorical_accuracy: 0.9075
22784/60000 [==========>...................] - ETA: 1:03 - loss: 0.2984 - categorical_accuracy: 0.9077
22816/60000 [==========>...................] - ETA: 1:03 - loss: 0.2981 - categorical_accuracy: 0.9077
22848/60000 [==========>...................] - ETA: 1:03 - loss: 0.2979 - categorical_accuracy: 0.9078
22880/60000 [==========>...................] - ETA: 1:03 - loss: 0.2976 - categorical_accuracy: 0.9079
22912/60000 [==========>...................] - ETA: 1:03 - loss: 0.2973 - categorical_accuracy: 0.9080
22944/60000 [==========>...................] - ETA: 1:03 - loss: 0.2972 - categorical_accuracy: 0.9079
22976/60000 [==========>...................] - ETA: 1:03 - loss: 0.2970 - categorical_accuracy: 0.9079
23040/60000 [==========>...................] - ETA: 1:03 - loss: 0.2963 - categorical_accuracy: 0.9082
23072/60000 [==========>...................] - ETA: 1:03 - loss: 0.2960 - categorical_accuracy: 0.9082
23104/60000 [==========>...................] - ETA: 1:03 - loss: 0.2957 - categorical_accuracy: 0.9084
23136/60000 [==========>...................] - ETA: 1:03 - loss: 0.2955 - categorical_accuracy: 0.9084
23168/60000 [==========>...................] - ETA: 1:03 - loss: 0.2953 - categorical_accuracy: 0.9085
23200/60000 [==========>...................] - ETA: 1:03 - loss: 0.2953 - categorical_accuracy: 0.9084
23232/60000 [==========>...................] - ETA: 1:03 - loss: 0.2950 - categorical_accuracy: 0.9085
23264/60000 [==========>...................] - ETA: 1:03 - loss: 0.2948 - categorical_accuracy: 0.9086
23296/60000 [==========>...................] - ETA: 1:03 - loss: 0.2948 - categorical_accuracy: 0.9087
23328/60000 [==========>...................] - ETA: 1:02 - loss: 0.2947 - categorical_accuracy: 0.9087
23360/60000 [==========>...................] - ETA: 1:02 - loss: 0.2945 - categorical_accuracy: 0.9087
23392/60000 [==========>...................] - ETA: 1:02 - loss: 0.2943 - categorical_accuracy: 0.9088
23424/60000 [==========>...................] - ETA: 1:02 - loss: 0.2940 - categorical_accuracy: 0.9089
23456/60000 [==========>...................] - ETA: 1:02 - loss: 0.2937 - categorical_accuracy: 0.9090
23488/60000 [==========>...................] - ETA: 1:02 - loss: 0.2934 - categorical_accuracy: 0.9091
23520/60000 [==========>...................] - ETA: 1:02 - loss: 0.2933 - categorical_accuracy: 0.9091
23552/60000 [==========>...................] - ETA: 1:02 - loss: 0.2930 - categorical_accuracy: 0.9092
23584/60000 [==========>...................] - ETA: 1:02 - loss: 0.2928 - categorical_accuracy: 0.9093
23616/60000 [==========>...................] - ETA: 1:02 - loss: 0.2925 - categorical_accuracy: 0.9093
23648/60000 [==========>...................] - ETA: 1:02 - loss: 0.2922 - categorical_accuracy: 0.9094
23680/60000 [==========>...................] - ETA: 1:02 - loss: 0.2923 - categorical_accuracy: 0.9095
23712/60000 [==========>...................] - ETA: 1:02 - loss: 0.2919 - categorical_accuracy: 0.9095
23744/60000 [==========>...................] - ETA: 1:02 - loss: 0.2917 - categorical_accuracy: 0.9096
23776/60000 [==========>...................] - ETA: 1:02 - loss: 0.2916 - categorical_accuracy: 0.9097
23808/60000 [==========>...................] - ETA: 1:02 - loss: 0.2913 - categorical_accuracy: 0.9098
23840/60000 [==========>...................] - ETA: 1:02 - loss: 0.2910 - categorical_accuracy: 0.9099
23872/60000 [==========>...................] - ETA: 1:02 - loss: 0.2909 - categorical_accuracy: 0.9100
23904/60000 [==========>...................] - ETA: 1:01 - loss: 0.2908 - categorical_accuracy: 0.9101
23936/60000 [==========>...................] - ETA: 1:01 - loss: 0.2910 - categorical_accuracy: 0.9101
24000/60000 [===========>..................] - ETA: 1:01 - loss: 0.2906 - categorical_accuracy: 0.9103
24032/60000 [===========>..................] - ETA: 1:01 - loss: 0.2904 - categorical_accuracy: 0.9102
24064/60000 [===========>..................] - ETA: 1:01 - loss: 0.2901 - categorical_accuracy: 0.9104
24096/60000 [===========>..................] - ETA: 1:01 - loss: 0.2904 - categorical_accuracy: 0.9103
24128/60000 [===========>..................] - ETA: 1:01 - loss: 0.2902 - categorical_accuracy: 0.9103
24160/60000 [===========>..................] - ETA: 1:01 - loss: 0.2900 - categorical_accuracy: 0.9104
24192/60000 [===========>..................] - ETA: 1:01 - loss: 0.2900 - categorical_accuracy: 0.9105
24224/60000 [===========>..................] - ETA: 1:01 - loss: 0.2897 - categorical_accuracy: 0.9106
24256/60000 [===========>..................] - ETA: 1:01 - loss: 0.2895 - categorical_accuracy: 0.9107
24288/60000 [===========>..................] - ETA: 1:01 - loss: 0.2891 - categorical_accuracy: 0.9108
24320/60000 [===========>..................] - ETA: 1:01 - loss: 0.2889 - categorical_accuracy: 0.9109
24352/60000 [===========>..................] - ETA: 1:01 - loss: 0.2887 - categorical_accuracy: 0.9109
24384/60000 [===========>..................] - ETA: 1:01 - loss: 0.2885 - categorical_accuracy: 0.9110
24416/60000 [===========>..................] - ETA: 1:01 - loss: 0.2884 - categorical_accuracy: 0.9111
24448/60000 [===========>..................] - ETA: 1:01 - loss: 0.2884 - categorical_accuracy: 0.9111
24480/60000 [===========>..................] - ETA: 1:00 - loss: 0.2881 - categorical_accuracy: 0.9112
24512/60000 [===========>..................] - ETA: 1:00 - loss: 0.2879 - categorical_accuracy: 0.9113
24544/60000 [===========>..................] - ETA: 1:00 - loss: 0.2878 - categorical_accuracy: 0.9113
24576/60000 [===========>..................] - ETA: 1:00 - loss: 0.2875 - categorical_accuracy: 0.9114
24608/60000 [===========>..................] - ETA: 1:00 - loss: 0.2872 - categorical_accuracy: 0.9115
24640/60000 [===========>..................] - ETA: 1:00 - loss: 0.2869 - categorical_accuracy: 0.9116
24672/60000 [===========>..................] - ETA: 1:00 - loss: 0.2871 - categorical_accuracy: 0.9117
24704/60000 [===========>..................] - ETA: 1:00 - loss: 0.2867 - categorical_accuracy: 0.9118
24736/60000 [===========>..................] - ETA: 1:00 - loss: 0.2866 - categorical_accuracy: 0.9118
24768/60000 [===========>..................] - ETA: 1:00 - loss: 0.2862 - categorical_accuracy: 0.9119
24800/60000 [===========>..................] - ETA: 1:00 - loss: 0.2863 - categorical_accuracy: 0.9119
24832/60000 [===========>..................] - ETA: 1:00 - loss: 0.2860 - categorical_accuracy: 0.9120
24864/60000 [===========>..................] - ETA: 1:00 - loss: 0.2857 - categorical_accuracy: 0.9121
24896/60000 [===========>..................] - ETA: 1:00 - loss: 0.2856 - categorical_accuracy: 0.9122
24928/60000 [===========>..................] - ETA: 1:00 - loss: 0.2854 - categorical_accuracy: 0.9122
24960/60000 [===========>..................] - ETA: 1:00 - loss: 0.2852 - categorical_accuracy: 0.9123
24992/60000 [===========>..................] - ETA: 1:00 - loss: 0.2853 - categorical_accuracy: 0.9123
25024/60000 [===========>..................] - ETA: 1:00 - loss: 0.2851 - categorical_accuracy: 0.9124
25056/60000 [===========>..................] - ETA: 59s - loss: 0.2848 - categorical_accuracy: 0.9125 
25088/60000 [===========>..................] - ETA: 59s - loss: 0.2845 - categorical_accuracy: 0.9125
25120/60000 [===========>..................] - ETA: 59s - loss: 0.2844 - categorical_accuracy: 0.9126
25152/60000 [===========>..................] - ETA: 59s - loss: 0.2841 - categorical_accuracy: 0.9127
25184/60000 [===========>..................] - ETA: 59s - loss: 0.2838 - categorical_accuracy: 0.9128
25216/60000 [===========>..................] - ETA: 59s - loss: 0.2835 - categorical_accuracy: 0.9129
25248/60000 [===========>..................] - ETA: 59s - loss: 0.2834 - categorical_accuracy: 0.9129
25280/60000 [===========>..................] - ETA: 59s - loss: 0.2831 - categorical_accuracy: 0.9130
25312/60000 [===========>..................] - ETA: 59s - loss: 0.2830 - categorical_accuracy: 0.9131
25344/60000 [===========>..................] - ETA: 59s - loss: 0.2827 - categorical_accuracy: 0.9132
25376/60000 [===========>..................] - ETA: 59s - loss: 0.2824 - categorical_accuracy: 0.9133
25408/60000 [===========>..................] - ETA: 59s - loss: 0.2821 - categorical_accuracy: 0.9134
25440/60000 [===========>..................] - ETA: 59s - loss: 0.2819 - categorical_accuracy: 0.9134
25472/60000 [===========>..................] - ETA: 59s - loss: 0.2818 - categorical_accuracy: 0.9135
25504/60000 [===========>..................] - ETA: 59s - loss: 0.2816 - categorical_accuracy: 0.9135
25536/60000 [===========>..................] - ETA: 59s - loss: 0.2814 - categorical_accuracy: 0.9136
25568/60000 [===========>..................] - ETA: 59s - loss: 0.2812 - categorical_accuracy: 0.9136
25600/60000 [===========>..................] - ETA: 59s - loss: 0.2809 - categorical_accuracy: 0.9137
25632/60000 [===========>..................] - ETA: 58s - loss: 0.2806 - categorical_accuracy: 0.9138
25664/60000 [===========>..................] - ETA: 58s - loss: 0.2803 - categorical_accuracy: 0.9139
25696/60000 [===========>..................] - ETA: 58s - loss: 0.2802 - categorical_accuracy: 0.9139
25728/60000 [===========>..................] - ETA: 58s - loss: 0.2801 - categorical_accuracy: 0.9139
25760/60000 [===========>..................] - ETA: 58s - loss: 0.2799 - categorical_accuracy: 0.9140
25792/60000 [===========>..................] - ETA: 58s - loss: 0.2798 - categorical_accuracy: 0.9140
25824/60000 [===========>..................] - ETA: 58s - loss: 0.2796 - categorical_accuracy: 0.9141
25856/60000 [===========>..................] - ETA: 58s - loss: 0.2794 - categorical_accuracy: 0.9141
25888/60000 [===========>..................] - ETA: 58s - loss: 0.2792 - categorical_accuracy: 0.9142
25920/60000 [===========>..................] - ETA: 58s - loss: 0.2791 - categorical_accuracy: 0.9142
25952/60000 [===========>..................] - ETA: 58s - loss: 0.2789 - categorical_accuracy: 0.9142
25984/60000 [===========>..................] - ETA: 58s - loss: 0.2786 - categorical_accuracy: 0.9143
26016/60000 [============>.................] - ETA: 58s - loss: 0.2783 - categorical_accuracy: 0.9144
26048/60000 [============>.................] - ETA: 58s - loss: 0.2780 - categorical_accuracy: 0.9145
26080/60000 [============>.................] - ETA: 58s - loss: 0.2778 - categorical_accuracy: 0.9146
26112/60000 [============>.................] - ETA: 58s - loss: 0.2775 - categorical_accuracy: 0.9147
26144/60000 [============>.................] - ETA: 58s - loss: 0.2772 - categorical_accuracy: 0.9148
26176/60000 [============>.................] - ETA: 58s - loss: 0.2769 - categorical_accuracy: 0.9149
26208/60000 [============>.................] - ETA: 57s - loss: 0.2765 - categorical_accuracy: 0.9150
26240/60000 [============>.................] - ETA: 57s - loss: 0.2763 - categorical_accuracy: 0.9151
26272/60000 [============>.................] - ETA: 57s - loss: 0.2760 - categorical_accuracy: 0.9152
26304/60000 [============>.................] - ETA: 57s - loss: 0.2759 - categorical_accuracy: 0.9152
26336/60000 [============>.................] - ETA: 57s - loss: 0.2757 - categorical_accuracy: 0.9152
26368/60000 [============>.................] - ETA: 57s - loss: 0.2755 - categorical_accuracy: 0.9153
26400/60000 [============>.................] - ETA: 57s - loss: 0.2751 - categorical_accuracy: 0.9154
26432/60000 [============>.................] - ETA: 57s - loss: 0.2749 - categorical_accuracy: 0.9155
26464/60000 [============>.................] - ETA: 57s - loss: 0.2752 - categorical_accuracy: 0.9154
26496/60000 [============>.................] - ETA: 57s - loss: 0.2749 - categorical_accuracy: 0.9155
26528/60000 [============>.................] - ETA: 57s - loss: 0.2747 - categorical_accuracy: 0.9156
26560/60000 [============>.................] - ETA: 57s - loss: 0.2744 - categorical_accuracy: 0.9157
26624/60000 [============>.................] - ETA: 57s - loss: 0.2741 - categorical_accuracy: 0.9158
26656/60000 [============>.................] - ETA: 57s - loss: 0.2739 - categorical_accuracy: 0.9159
26688/60000 [============>.................] - ETA: 57s - loss: 0.2736 - categorical_accuracy: 0.9160
26720/60000 [============>.................] - ETA: 57s - loss: 0.2734 - categorical_accuracy: 0.9160
26752/60000 [============>.................] - ETA: 57s - loss: 0.2733 - categorical_accuracy: 0.9160
26784/60000 [============>.................] - ETA: 56s - loss: 0.2731 - categorical_accuracy: 0.9160
26848/60000 [============>.................] - ETA: 56s - loss: 0.2726 - categorical_accuracy: 0.9161
26880/60000 [============>.................] - ETA: 56s - loss: 0.2724 - categorical_accuracy: 0.9162
26912/60000 [============>.................] - ETA: 56s - loss: 0.2720 - categorical_accuracy: 0.9163
26944/60000 [============>.................] - ETA: 56s - loss: 0.2721 - categorical_accuracy: 0.9163
26976/60000 [============>.................] - ETA: 56s - loss: 0.2720 - categorical_accuracy: 0.9163
27008/60000 [============>.................] - ETA: 56s - loss: 0.2716 - categorical_accuracy: 0.9164
27040/60000 [============>.................] - ETA: 56s - loss: 0.2715 - categorical_accuracy: 0.9165
27072/60000 [============>.................] - ETA: 56s - loss: 0.2713 - categorical_accuracy: 0.9165
27104/60000 [============>.................] - ETA: 56s - loss: 0.2712 - categorical_accuracy: 0.9166
27136/60000 [============>.................] - ETA: 56s - loss: 0.2709 - categorical_accuracy: 0.9166
27168/60000 [============>.................] - ETA: 56s - loss: 0.2706 - categorical_accuracy: 0.9167
27200/60000 [============>.................] - ETA: 56s - loss: 0.2704 - categorical_accuracy: 0.9168
27232/60000 [============>.................] - ETA: 56s - loss: 0.2702 - categorical_accuracy: 0.9169
27264/60000 [============>.................] - ETA: 56s - loss: 0.2702 - categorical_accuracy: 0.9169
27296/60000 [============>.................] - ETA: 56s - loss: 0.2702 - categorical_accuracy: 0.9169
27328/60000 [============>.................] - ETA: 56s - loss: 0.2700 - categorical_accuracy: 0.9170
27360/60000 [============>.................] - ETA: 55s - loss: 0.2698 - categorical_accuracy: 0.9171
27392/60000 [============>.................] - ETA: 55s - loss: 0.2695 - categorical_accuracy: 0.9172
27424/60000 [============>.................] - ETA: 55s - loss: 0.2692 - categorical_accuracy: 0.9172
27456/60000 [============>.................] - ETA: 55s - loss: 0.2695 - categorical_accuracy: 0.9172
27488/60000 [============>.................] - ETA: 55s - loss: 0.2693 - categorical_accuracy: 0.9172
27520/60000 [============>.................] - ETA: 55s - loss: 0.2692 - categorical_accuracy: 0.9172
27552/60000 [============>.................] - ETA: 55s - loss: 0.2689 - categorical_accuracy: 0.9173
27584/60000 [============>.................] - ETA: 55s - loss: 0.2687 - categorical_accuracy: 0.9173
27616/60000 [============>.................] - ETA: 55s - loss: 0.2686 - categorical_accuracy: 0.9174
27648/60000 [============>.................] - ETA: 55s - loss: 0.2683 - categorical_accuracy: 0.9175
27680/60000 [============>.................] - ETA: 55s - loss: 0.2682 - categorical_accuracy: 0.9175
27712/60000 [============>.................] - ETA: 55s - loss: 0.2681 - categorical_accuracy: 0.9175
27744/60000 [============>.................] - ETA: 55s - loss: 0.2679 - categorical_accuracy: 0.9176
27776/60000 [============>.................] - ETA: 55s - loss: 0.2678 - categorical_accuracy: 0.9176
27808/60000 [============>.................] - ETA: 55s - loss: 0.2677 - categorical_accuracy: 0.9177
27840/60000 [============>.................] - ETA: 55s - loss: 0.2675 - categorical_accuracy: 0.9177
27872/60000 [============>.................] - ETA: 55s - loss: 0.2673 - categorical_accuracy: 0.9178
27936/60000 [============>.................] - ETA: 54s - loss: 0.2671 - categorical_accuracy: 0.9179
27968/60000 [============>.................] - ETA: 54s - loss: 0.2669 - categorical_accuracy: 0.9180
28000/60000 [=============>................] - ETA: 54s - loss: 0.2671 - categorical_accuracy: 0.9180
28032/60000 [=============>................] - ETA: 54s - loss: 0.2671 - categorical_accuracy: 0.9180
28064/60000 [=============>................] - ETA: 54s - loss: 0.2670 - categorical_accuracy: 0.9180
28096/60000 [=============>................] - ETA: 54s - loss: 0.2670 - categorical_accuracy: 0.9181
28128/60000 [=============>................] - ETA: 54s - loss: 0.2669 - categorical_accuracy: 0.9181
28160/60000 [=============>................] - ETA: 54s - loss: 0.2667 - categorical_accuracy: 0.9182
28192/60000 [=============>................] - ETA: 54s - loss: 0.2667 - categorical_accuracy: 0.9182
28224/60000 [=============>................] - ETA: 54s - loss: 0.2669 - categorical_accuracy: 0.9182
28256/60000 [=============>................] - ETA: 54s - loss: 0.2668 - categorical_accuracy: 0.9182
28288/60000 [=============>................] - ETA: 54s - loss: 0.2669 - categorical_accuracy: 0.9182
28352/60000 [=============>................] - ETA: 54s - loss: 0.2668 - categorical_accuracy: 0.9182
28416/60000 [=============>................] - ETA: 54s - loss: 0.2668 - categorical_accuracy: 0.9181
28448/60000 [=============>................] - ETA: 54s - loss: 0.2666 - categorical_accuracy: 0.9182
28480/60000 [=============>................] - ETA: 54s - loss: 0.2663 - categorical_accuracy: 0.9183
28512/60000 [=============>................] - ETA: 53s - loss: 0.2661 - categorical_accuracy: 0.9184
28544/60000 [=============>................] - ETA: 53s - loss: 0.2659 - categorical_accuracy: 0.9184
28576/60000 [=============>................] - ETA: 53s - loss: 0.2658 - categorical_accuracy: 0.9185
28608/60000 [=============>................] - ETA: 53s - loss: 0.2656 - categorical_accuracy: 0.9185
28640/60000 [=============>................] - ETA: 53s - loss: 0.2653 - categorical_accuracy: 0.9186
28672/60000 [=============>................] - ETA: 53s - loss: 0.2651 - categorical_accuracy: 0.9186
28704/60000 [=============>................] - ETA: 53s - loss: 0.2649 - categorical_accuracy: 0.9187
28736/60000 [=============>................] - ETA: 53s - loss: 0.2647 - categorical_accuracy: 0.9187
28768/60000 [=============>................] - ETA: 53s - loss: 0.2646 - categorical_accuracy: 0.9188
28800/60000 [=============>................] - ETA: 53s - loss: 0.2644 - categorical_accuracy: 0.9188
28832/60000 [=============>................] - ETA: 53s - loss: 0.2643 - categorical_accuracy: 0.9188
28864/60000 [=============>................] - ETA: 53s - loss: 0.2643 - categorical_accuracy: 0.9188
28896/60000 [=============>................] - ETA: 53s - loss: 0.2642 - categorical_accuracy: 0.9189
28960/60000 [=============>................] - ETA: 53s - loss: 0.2640 - categorical_accuracy: 0.9190
28992/60000 [=============>................] - ETA: 53s - loss: 0.2640 - categorical_accuracy: 0.9190
29024/60000 [=============>................] - ETA: 53s - loss: 0.2639 - categorical_accuracy: 0.9190
29056/60000 [=============>................] - ETA: 53s - loss: 0.2636 - categorical_accuracy: 0.9191
29088/60000 [=============>................] - ETA: 52s - loss: 0.2634 - categorical_accuracy: 0.9192
29120/60000 [=============>................] - ETA: 52s - loss: 0.2632 - categorical_accuracy: 0.9193
29152/60000 [=============>................] - ETA: 52s - loss: 0.2630 - categorical_accuracy: 0.9194
29184/60000 [=============>................] - ETA: 52s - loss: 0.2627 - categorical_accuracy: 0.9194
29216/60000 [=============>................] - ETA: 52s - loss: 0.2626 - categorical_accuracy: 0.9195
29248/60000 [=============>................] - ETA: 52s - loss: 0.2624 - categorical_accuracy: 0.9196
29280/60000 [=============>................] - ETA: 52s - loss: 0.2622 - categorical_accuracy: 0.9196
29312/60000 [=============>................] - ETA: 52s - loss: 0.2620 - categorical_accuracy: 0.9197
29344/60000 [=============>................] - ETA: 52s - loss: 0.2619 - categorical_accuracy: 0.9197
29376/60000 [=============>................] - ETA: 52s - loss: 0.2617 - categorical_accuracy: 0.9198
29408/60000 [=============>................] - ETA: 52s - loss: 0.2615 - categorical_accuracy: 0.9198
29440/60000 [=============>................] - ETA: 52s - loss: 0.2613 - categorical_accuracy: 0.9199
29504/60000 [=============>................] - ETA: 52s - loss: 0.2609 - categorical_accuracy: 0.9200
29568/60000 [=============>................] - ETA: 52s - loss: 0.2605 - categorical_accuracy: 0.9201
29600/60000 [=============>................] - ETA: 52s - loss: 0.2603 - categorical_accuracy: 0.9202
29632/60000 [=============>................] - ETA: 52s - loss: 0.2601 - categorical_accuracy: 0.9203
29664/60000 [=============>................] - ETA: 51s - loss: 0.2600 - categorical_accuracy: 0.9203
29696/60000 [=============>................] - ETA: 51s - loss: 0.2597 - categorical_accuracy: 0.9204
29728/60000 [=============>................] - ETA: 51s - loss: 0.2595 - categorical_accuracy: 0.9205
29760/60000 [=============>................] - ETA: 51s - loss: 0.2593 - categorical_accuracy: 0.9205
29792/60000 [=============>................] - ETA: 51s - loss: 0.2591 - categorical_accuracy: 0.9206
29824/60000 [=============>................] - ETA: 51s - loss: 0.2588 - categorical_accuracy: 0.9207
29856/60000 [=============>................] - ETA: 51s - loss: 0.2588 - categorical_accuracy: 0.9207
29888/60000 [=============>................] - ETA: 51s - loss: 0.2587 - categorical_accuracy: 0.9207
29920/60000 [=============>................] - ETA: 51s - loss: 0.2585 - categorical_accuracy: 0.9208
29952/60000 [=============>................] - ETA: 51s - loss: 0.2582 - categorical_accuracy: 0.9208
29984/60000 [=============>................] - ETA: 51s - loss: 0.2580 - categorical_accuracy: 0.9209
30016/60000 [==============>...............] - ETA: 51s - loss: 0.2577 - categorical_accuracy: 0.9210
30048/60000 [==============>...............] - ETA: 51s - loss: 0.2574 - categorical_accuracy: 0.9211
30080/60000 [==============>...............] - ETA: 51s - loss: 0.2573 - categorical_accuracy: 0.9211
30112/60000 [==============>...............] - ETA: 51s - loss: 0.2574 - categorical_accuracy: 0.9211
30144/60000 [==============>...............] - ETA: 51s - loss: 0.2573 - categorical_accuracy: 0.9212
30176/60000 [==============>...............] - ETA: 51s - loss: 0.2571 - categorical_accuracy: 0.9213
30208/60000 [==============>...............] - ETA: 51s - loss: 0.2569 - categorical_accuracy: 0.9213
30240/60000 [==============>...............] - ETA: 50s - loss: 0.2566 - categorical_accuracy: 0.9214
30272/60000 [==============>...............] - ETA: 50s - loss: 0.2565 - categorical_accuracy: 0.9214
30304/60000 [==============>...............] - ETA: 50s - loss: 0.2563 - categorical_accuracy: 0.9215
30336/60000 [==============>...............] - ETA: 50s - loss: 0.2563 - categorical_accuracy: 0.9215
30368/60000 [==============>...............] - ETA: 50s - loss: 0.2561 - categorical_accuracy: 0.9215
30400/60000 [==============>...............] - ETA: 50s - loss: 0.2559 - categorical_accuracy: 0.9216
30432/60000 [==============>...............] - ETA: 50s - loss: 0.2557 - categorical_accuracy: 0.9216
30464/60000 [==============>...............] - ETA: 50s - loss: 0.2556 - categorical_accuracy: 0.9216
30496/60000 [==============>...............] - ETA: 50s - loss: 0.2553 - categorical_accuracy: 0.9217
30528/60000 [==============>...............] - ETA: 50s - loss: 0.2551 - categorical_accuracy: 0.9218
30560/60000 [==============>...............] - ETA: 50s - loss: 0.2549 - categorical_accuracy: 0.9219
30592/60000 [==============>...............] - ETA: 50s - loss: 0.2547 - categorical_accuracy: 0.9219
30624/60000 [==============>...............] - ETA: 50s - loss: 0.2545 - categorical_accuracy: 0.9220
30656/60000 [==============>...............] - ETA: 50s - loss: 0.2545 - categorical_accuracy: 0.9220
30720/60000 [==============>...............] - ETA: 50s - loss: 0.2542 - categorical_accuracy: 0.9220
30752/60000 [==============>...............] - ETA: 50s - loss: 0.2541 - categorical_accuracy: 0.9221
30784/60000 [==============>...............] - ETA: 50s - loss: 0.2539 - categorical_accuracy: 0.9221
30816/60000 [==============>...............] - ETA: 49s - loss: 0.2538 - categorical_accuracy: 0.9221
30848/60000 [==============>...............] - ETA: 49s - loss: 0.2538 - categorical_accuracy: 0.9221
30880/60000 [==============>...............] - ETA: 49s - loss: 0.2537 - categorical_accuracy: 0.9222
30912/60000 [==============>...............] - ETA: 49s - loss: 0.2536 - categorical_accuracy: 0.9222
30944/60000 [==============>...............] - ETA: 49s - loss: 0.2535 - categorical_accuracy: 0.9222
30976/60000 [==============>...............] - ETA: 49s - loss: 0.2536 - categorical_accuracy: 0.9223
31008/60000 [==============>...............] - ETA: 49s - loss: 0.2537 - categorical_accuracy: 0.9223
31040/60000 [==============>...............] - ETA: 49s - loss: 0.2535 - categorical_accuracy: 0.9224
31072/60000 [==============>...............] - ETA: 49s - loss: 0.2532 - categorical_accuracy: 0.9225
31104/60000 [==============>...............] - ETA: 49s - loss: 0.2531 - categorical_accuracy: 0.9225
31136/60000 [==============>...............] - ETA: 49s - loss: 0.2530 - categorical_accuracy: 0.9226
31168/60000 [==============>...............] - ETA: 49s - loss: 0.2528 - categorical_accuracy: 0.9226
31200/60000 [==============>...............] - ETA: 49s - loss: 0.2527 - categorical_accuracy: 0.9227
31232/60000 [==============>...............] - ETA: 49s - loss: 0.2526 - categorical_accuracy: 0.9227
31264/60000 [==============>...............] - ETA: 49s - loss: 0.2526 - categorical_accuracy: 0.9227
31296/60000 [==============>...............] - ETA: 49s - loss: 0.2525 - categorical_accuracy: 0.9227
31328/60000 [==============>...............] - ETA: 49s - loss: 0.2525 - categorical_accuracy: 0.9227
31360/60000 [==============>...............] - ETA: 49s - loss: 0.2523 - categorical_accuracy: 0.9228
31392/60000 [==============>...............] - ETA: 48s - loss: 0.2522 - categorical_accuracy: 0.9228
31424/60000 [==============>...............] - ETA: 48s - loss: 0.2519 - categorical_accuracy: 0.9229
31456/60000 [==============>...............] - ETA: 48s - loss: 0.2518 - categorical_accuracy: 0.9229
31488/60000 [==============>...............] - ETA: 48s - loss: 0.2516 - categorical_accuracy: 0.9230
31520/60000 [==============>...............] - ETA: 48s - loss: 0.2517 - categorical_accuracy: 0.9229
31552/60000 [==============>...............] - ETA: 48s - loss: 0.2515 - categorical_accuracy: 0.9230
31584/60000 [==============>...............] - ETA: 48s - loss: 0.2515 - categorical_accuracy: 0.9230
31616/60000 [==============>...............] - ETA: 48s - loss: 0.2513 - categorical_accuracy: 0.9231
31648/60000 [==============>...............] - ETA: 48s - loss: 0.2511 - categorical_accuracy: 0.9232
31680/60000 [==============>...............] - ETA: 48s - loss: 0.2509 - categorical_accuracy: 0.9232
31712/60000 [==============>...............] - ETA: 48s - loss: 0.2508 - categorical_accuracy: 0.9232
31744/60000 [==============>...............] - ETA: 48s - loss: 0.2506 - categorical_accuracy: 0.9233
31776/60000 [==============>...............] - ETA: 48s - loss: 0.2504 - categorical_accuracy: 0.9233
31808/60000 [==============>...............] - ETA: 48s - loss: 0.2502 - categorical_accuracy: 0.9234
31840/60000 [==============>...............] - ETA: 48s - loss: 0.2502 - categorical_accuracy: 0.9234
31872/60000 [==============>...............] - ETA: 48s - loss: 0.2501 - categorical_accuracy: 0.9235
31904/60000 [==============>...............] - ETA: 48s - loss: 0.2501 - categorical_accuracy: 0.9235
31936/60000 [==============>...............] - ETA: 48s - loss: 0.2500 - categorical_accuracy: 0.9235
31968/60000 [==============>...............] - ETA: 47s - loss: 0.2497 - categorical_accuracy: 0.9236
32000/60000 [===============>..............] - ETA: 47s - loss: 0.2496 - categorical_accuracy: 0.9236
32064/60000 [===============>..............] - ETA: 47s - loss: 0.2495 - categorical_accuracy: 0.9237
32096/60000 [===============>..............] - ETA: 47s - loss: 0.2496 - categorical_accuracy: 0.9237
32128/60000 [===============>..............] - ETA: 47s - loss: 0.2494 - categorical_accuracy: 0.9237
32160/60000 [===============>..............] - ETA: 47s - loss: 0.2492 - categorical_accuracy: 0.9238
32192/60000 [===============>..............] - ETA: 47s - loss: 0.2490 - categorical_accuracy: 0.9239
32256/60000 [===============>..............] - ETA: 47s - loss: 0.2486 - categorical_accuracy: 0.9240
32288/60000 [===============>..............] - ETA: 47s - loss: 0.2484 - categorical_accuracy: 0.9241
32320/60000 [===============>..............] - ETA: 47s - loss: 0.2483 - categorical_accuracy: 0.9241
32352/60000 [===============>..............] - ETA: 47s - loss: 0.2481 - categorical_accuracy: 0.9242
32384/60000 [===============>..............] - ETA: 47s - loss: 0.2479 - categorical_accuracy: 0.9242
32448/60000 [===============>..............] - ETA: 47s - loss: 0.2479 - categorical_accuracy: 0.9243
32480/60000 [===============>..............] - ETA: 47s - loss: 0.2477 - categorical_accuracy: 0.9244
32544/60000 [===============>..............] - ETA: 46s - loss: 0.2474 - categorical_accuracy: 0.9245
32576/60000 [===============>..............] - ETA: 46s - loss: 0.2472 - categorical_accuracy: 0.9245
32608/60000 [===============>..............] - ETA: 46s - loss: 0.2470 - categorical_accuracy: 0.9246
32640/60000 [===============>..............] - ETA: 46s - loss: 0.2470 - categorical_accuracy: 0.9246
32672/60000 [===============>..............] - ETA: 46s - loss: 0.2470 - categorical_accuracy: 0.9246
32704/60000 [===============>..............] - ETA: 46s - loss: 0.2469 - categorical_accuracy: 0.9246
32736/60000 [===============>..............] - ETA: 46s - loss: 0.2466 - categorical_accuracy: 0.9247
32768/60000 [===============>..............] - ETA: 46s - loss: 0.2466 - categorical_accuracy: 0.9247
32800/60000 [===============>..............] - ETA: 46s - loss: 0.2465 - categorical_accuracy: 0.9248
32832/60000 [===============>..............] - ETA: 46s - loss: 0.2463 - categorical_accuracy: 0.9248
32864/60000 [===============>..............] - ETA: 46s - loss: 0.2461 - categorical_accuracy: 0.9248
32896/60000 [===============>..............] - ETA: 46s - loss: 0.2459 - categorical_accuracy: 0.9249
32928/60000 [===============>..............] - ETA: 46s - loss: 0.2458 - categorical_accuracy: 0.9250
32960/60000 [===============>..............] - ETA: 46s - loss: 0.2456 - categorical_accuracy: 0.9250
32992/60000 [===============>..............] - ETA: 46s - loss: 0.2455 - categorical_accuracy: 0.9250
33024/60000 [===============>..............] - ETA: 46s - loss: 0.2453 - categorical_accuracy: 0.9251
33056/60000 [===============>..............] - ETA: 46s - loss: 0.2452 - categorical_accuracy: 0.9251
33088/60000 [===============>..............] - ETA: 46s - loss: 0.2452 - categorical_accuracy: 0.9251
33120/60000 [===============>..............] - ETA: 45s - loss: 0.2450 - categorical_accuracy: 0.9252
33152/60000 [===============>..............] - ETA: 45s - loss: 0.2449 - categorical_accuracy: 0.9252
33184/60000 [===============>..............] - ETA: 45s - loss: 0.2449 - categorical_accuracy: 0.9251
33216/60000 [===============>..............] - ETA: 45s - loss: 0.2446 - categorical_accuracy: 0.9252
33280/60000 [===============>..............] - ETA: 45s - loss: 0.2444 - categorical_accuracy: 0.9252
33312/60000 [===============>..............] - ETA: 45s - loss: 0.2442 - categorical_accuracy: 0.9253
33344/60000 [===============>..............] - ETA: 45s - loss: 0.2441 - categorical_accuracy: 0.9253
33376/60000 [===============>..............] - ETA: 45s - loss: 0.2440 - categorical_accuracy: 0.9253
33408/60000 [===============>..............] - ETA: 45s - loss: 0.2439 - categorical_accuracy: 0.9253
33440/60000 [===============>..............] - ETA: 45s - loss: 0.2438 - categorical_accuracy: 0.9254
33472/60000 [===============>..............] - ETA: 45s - loss: 0.2436 - categorical_accuracy: 0.9254
33504/60000 [===============>..............] - ETA: 45s - loss: 0.2434 - categorical_accuracy: 0.9255
33536/60000 [===============>..............] - ETA: 45s - loss: 0.2432 - categorical_accuracy: 0.9256
33568/60000 [===============>..............] - ETA: 45s - loss: 0.2430 - categorical_accuracy: 0.9256
33600/60000 [===============>..............] - ETA: 45s - loss: 0.2428 - categorical_accuracy: 0.9257
33632/60000 [===============>..............] - ETA: 45s - loss: 0.2427 - categorical_accuracy: 0.9257
33664/60000 [===============>..............] - ETA: 45s - loss: 0.2425 - categorical_accuracy: 0.9257
33696/60000 [===============>..............] - ETA: 44s - loss: 0.2424 - categorical_accuracy: 0.9258
33728/60000 [===============>..............] - ETA: 44s - loss: 0.2424 - categorical_accuracy: 0.9258
33760/60000 [===============>..............] - ETA: 44s - loss: 0.2422 - categorical_accuracy: 0.9258
33792/60000 [===============>..............] - ETA: 44s - loss: 0.2423 - categorical_accuracy: 0.9258
33824/60000 [===============>..............] - ETA: 44s - loss: 0.2421 - categorical_accuracy: 0.9259
33856/60000 [===============>..............] - ETA: 44s - loss: 0.2420 - categorical_accuracy: 0.9259
33888/60000 [===============>..............] - ETA: 44s - loss: 0.2418 - categorical_accuracy: 0.9260
33920/60000 [===============>..............] - ETA: 44s - loss: 0.2417 - categorical_accuracy: 0.9260
33952/60000 [===============>..............] - ETA: 44s - loss: 0.2415 - categorical_accuracy: 0.9260
33984/60000 [===============>..............] - ETA: 44s - loss: 0.2413 - categorical_accuracy: 0.9261
34016/60000 [================>.............] - ETA: 44s - loss: 0.2411 - categorical_accuracy: 0.9262
34048/60000 [================>.............] - ETA: 44s - loss: 0.2409 - categorical_accuracy: 0.9262
34080/60000 [================>.............] - ETA: 44s - loss: 0.2407 - categorical_accuracy: 0.9263
34112/60000 [================>.............] - ETA: 44s - loss: 0.2407 - categorical_accuracy: 0.9263
34144/60000 [================>.............] - ETA: 44s - loss: 0.2406 - categorical_accuracy: 0.9264
34176/60000 [================>.............] - ETA: 44s - loss: 0.2405 - categorical_accuracy: 0.9264
34208/60000 [================>.............] - ETA: 44s - loss: 0.2402 - categorical_accuracy: 0.9265
34240/60000 [================>.............] - ETA: 44s - loss: 0.2402 - categorical_accuracy: 0.9265
34272/60000 [================>.............] - ETA: 43s - loss: 0.2400 - categorical_accuracy: 0.9265
34304/60000 [================>.............] - ETA: 43s - loss: 0.2399 - categorical_accuracy: 0.9266
34336/60000 [================>.............] - ETA: 43s - loss: 0.2398 - categorical_accuracy: 0.9266
34368/60000 [================>.............] - ETA: 43s - loss: 0.2397 - categorical_accuracy: 0.9266
34400/60000 [================>.............] - ETA: 43s - loss: 0.2397 - categorical_accuracy: 0.9267
34432/60000 [================>.............] - ETA: 43s - loss: 0.2395 - categorical_accuracy: 0.9267
34464/60000 [================>.............] - ETA: 43s - loss: 0.2393 - categorical_accuracy: 0.9268
34496/60000 [================>.............] - ETA: 43s - loss: 0.2392 - categorical_accuracy: 0.9268
34528/60000 [================>.............] - ETA: 43s - loss: 0.2391 - categorical_accuracy: 0.9268
34560/60000 [================>.............] - ETA: 43s - loss: 0.2389 - categorical_accuracy: 0.9269
34592/60000 [================>.............] - ETA: 43s - loss: 0.2387 - categorical_accuracy: 0.9269
34624/60000 [================>.............] - ETA: 43s - loss: 0.2385 - categorical_accuracy: 0.9270
34656/60000 [================>.............] - ETA: 43s - loss: 0.2384 - categorical_accuracy: 0.9270
34688/60000 [================>.............] - ETA: 43s - loss: 0.2383 - categorical_accuracy: 0.9271
34720/60000 [================>.............] - ETA: 43s - loss: 0.2381 - categorical_accuracy: 0.9271
34752/60000 [================>.............] - ETA: 43s - loss: 0.2381 - categorical_accuracy: 0.9271
34784/60000 [================>.............] - ETA: 43s - loss: 0.2380 - categorical_accuracy: 0.9272
34816/60000 [================>.............] - ETA: 43s - loss: 0.2378 - categorical_accuracy: 0.9273
34848/60000 [================>.............] - ETA: 42s - loss: 0.2376 - categorical_accuracy: 0.9273
34880/60000 [================>.............] - ETA: 42s - loss: 0.2375 - categorical_accuracy: 0.9273
34912/60000 [================>.............] - ETA: 42s - loss: 0.2376 - categorical_accuracy: 0.9273
34944/60000 [================>.............] - ETA: 42s - loss: 0.2375 - categorical_accuracy: 0.9273
34976/60000 [================>.............] - ETA: 42s - loss: 0.2373 - categorical_accuracy: 0.9274
35008/60000 [================>.............] - ETA: 42s - loss: 0.2372 - categorical_accuracy: 0.9274
35040/60000 [================>.............] - ETA: 42s - loss: 0.2371 - categorical_accuracy: 0.9275
35072/60000 [================>.............] - ETA: 42s - loss: 0.2371 - categorical_accuracy: 0.9274
35104/60000 [================>.............] - ETA: 42s - loss: 0.2369 - categorical_accuracy: 0.9275
35136/60000 [================>.............] - ETA: 42s - loss: 0.2369 - categorical_accuracy: 0.9275
35168/60000 [================>.............] - ETA: 42s - loss: 0.2367 - categorical_accuracy: 0.9276
35200/60000 [================>.............] - ETA: 42s - loss: 0.2366 - categorical_accuracy: 0.9276
35232/60000 [================>.............] - ETA: 42s - loss: 0.2364 - categorical_accuracy: 0.9277
35264/60000 [================>.............] - ETA: 42s - loss: 0.2363 - categorical_accuracy: 0.9277
35296/60000 [================>.............] - ETA: 42s - loss: 0.2361 - categorical_accuracy: 0.9278
35328/60000 [================>.............] - ETA: 42s - loss: 0.2359 - categorical_accuracy: 0.9279
35360/60000 [================>.............] - ETA: 42s - loss: 0.2357 - categorical_accuracy: 0.9279
35392/60000 [================>.............] - ETA: 42s - loss: 0.2356 - categorical_accuracy: 0.9280
35424/60000 [================>.............] - ETA: 42s - loss: 0.2354 - categorical_accuracy: 0.9281
35456/60000 [================>.............] - ETA: 41s - loss: 0.2355 - categorical_accuracy: 0.9281
35488/60000 [================>.............] - ETA: 41s - loss: 0.2353 - categorical_accuracy: 0.9281
35520/60000 [================>.............] - ETA: 41s - loss: 0.2352 - categorical_accuracy: 0.9281
35552/60000 [================>.............] - ETA: 41s - loss: 0.2351 - categorical_accuracy: 0.9282
35616/60000 [================>.............] - ETA: 41s - loss: 0.2348 - categorical_accuracy: 0.9283
35648/60000 [================>.............] - ETA: 41s - loss: 0.2346 - categorical_accuracy: 0.9283
35680/60000 [================>.............] - ETA: 41s - loss: 0.2345 - categorical_accuracy: 0.9283
35712/60000 [================>.............] - ETA: 41s - loss: 0.2344 - categorical_accuracy: 0.9283
35744/60000 [================>.............] - ETA: 41s - loss: 0.2343 - categorical_accuracy: 0.9284
35776/60000 [================>.............] - ETA: 41s - loss: 0.2343 - categorical_accuracy: 0.9284
35840/60000 [================>.............] - ETA: 41s - loss: 0.2340 - categorical_accuracy: 0.9285
35872/60000 [================>.............] - ETA: 41s - loss: 0.2338 - categorical_accuracy: 0.9285
35936/60000 [================>.............] - ETA: 41s - loss: 0.2335 - categorical_accuracy: 0.9286
35968/60000 [================>.............] - ETA: 41s - loss: 0.2334 - categorical_accuracy: 0.9286
36032/60000 [=================>............] - ETA: 40s - loss: 0.2331 - categorical_accuracy: 0.9287
36064/60000 [=================>............] - ETA: 40s - loss: 0.2333 - categorical_accuracy: 0.9287
36096/60000 [=================>............] - ETA: 40s - loss: 0.2332 - categorical_accuracy: 0.9287
36128/60000 [=================>............] - ETA: 40s - loss: 0.2330 - categorical_accuracy: 0.9288
36160/60000 [=================>............] - ETA: 40s - loss: 0.2328 - categorical_accuracy: 0.9289
36192/60000 [=================>............] - ETA: 40s - loss: 0.2327 - categorical_accuracy: 0.9289
36224/60000 [=================>............] - ETA: 40s - loss: 0.2325 - categorical_accuracy: 0.9290
36256/60000 [=================>............] - ETA: 40s - loss: 0.2324 - categorical_accuracy: 0.9290
36288/60000 [=================>............] - ETA: 40s - loss: 0.2323 - categorical_accuracy: 0.9290
36320/60000 [=================>............] - ETA: 40s - loss: 0.2321 - categorical_accuracy: 0.9291
36352/60000 [=================>............] - ETA: 40s - loss: 0.2322 - categorical_accuracy: 0.9291
36384/60000 [=================>............] - ETA: 40s - loss: 0.2320 - categorical_accuracy: 0.9291
36416/60000 [=================>............] - ETA: 40s - loss: 0.2319 - categorical_accuracy: 0.9292
36448/60000 [=================>............] - ETA: 40s - loss: 0.2317 - categorical_accuracy: 0.9292
36480/60000 [=================>............] - ETA: 40s - loss: 0.2315 - categorical_accuracy: 0.9293
36512/60000 [=================>............] - ETA: 40s - loss: 0.2316 - categorical_accuracy: 0.9293
36544/60000 [=================>............] - ETA: 40s - loss: 0.2314 - categorical_accuracy: 0.9293
36576/60000 [=================>............] - ETA: 40s - loss: 0.2312 - categorical_accuracy: 0.9294
36608/60000 [=================>............] - ETA: 39s - loss: 0.2311 - categorical_accuracy: 0.9294
36640/60000 [=================>............] - ETA: 39s - loss: 0.2309 - categorical_accuracy: 0.9295
36672/60000 [=================>............] - ETA: 39s - loss: 0.2308 - categorical_accuracy: 0.9295
36704/60000 [=================>............] - ETA: 39s - loss: 0.2307 - categorical_accuracy: 0.9295
36736/60000 [=================>............] - ETA: 39s - loss: 0.2307 - categorical_accuracy: 0.9295
36768/60000 [=================>............] - ETA: 39s - loss: 0.2306 - categorical_accuracy: 0.9296
36800/60000 [=================>............] - ETA: 39s - loss: 0.2305 - categorical_accuracy: 0.9296
36832/60000 [=================>............] - ETA: 39s - loss: 0.2304 - categorical_accuracy: 0.9296
36864/60000 [=================>............] - ETA: 39s - loss: 0.2303 - categorical_accuracy: 0.9296
36896/60000 [=================>............] - ETA: 39s - loss: 0.2301 - categorical_accuracy: 0.9297
36928/60000 [=================>............] - ETA: 39s - loss: 0.2300 - categorical_accuracy: 0.9297
36960/60000 [=================>............] - ETA: 39s - loss: 0.2298 - categorical_accuracy: 0.9298
36992/60000 [=================>............] - ETA: 39s - loss: 0.2300 - categorical_accuracy: 0.9298
37024/60000 [=================>............] - ETA: 39s - loss: 0.2298 - categorical_accuracy: 0.9298
37056/60000 [=================>............] - ETA: 39s - loss: 0.2298 - categorical_accuracy: 0.9299
37088/60000 [=================>............] - ETA: 39s - loss: 0.2298 - categorical_accuracy: 0.9299
37120/60000 [=================>............] - ETA: 39s - loss: 0.2297 - categorical_accuracy: 0.9299
37152/60000 [=================>............] - ETA: 39s - loss: 0.2295 - categorical_accuracy: 0.9300
37184/60000 [=================>............] - ETA: 38s - loss: 0.2295 - categorical_accuracy: 0.9300
37216/60000 [=================>............] - ETA: 38s - loss: 0.2293 - categorical_accuracy: 0.9301
37248/60000 [=================>............] - ETA: 38s - loss: 0.2291 - categorical_accuracy: 0.9301
37280/60000 [=================>............] - ETA: 38s - loss: 0.2289 - categorical_accuracy: 0.9302
37312/60000 [=================>............] - ETA: 38s - loss: 0.2289 - categorical_accuracy: 0.9302
37344/60000 [=================>............] - ETA: 38s - loss: 0.2287 - categorical_accuracy: 0.9303
37376/60000 [=================>............] - ETA: 38s - loss: 0.2287 - categorical_accuracy: 0.9303
37408/60000 [=================>............] - ETA: 38s - loss: 0.2286 - categorical_accuracy: 0.9303
37440/60000 [=================>............] - ETA: 38s - loss: 0.2285 - categorical_accuracy: 0.9303
37472/60000 [=================>............] - ETA: 38s - loss: 0.2284 - categorical_accuracy: 0.9303
37504/60000 [=================>............] - ETA: 38s - loss: 0.2284 - categorical_accuracy: 0.9303
37536/60000 [=================>............] - ETA: 38s - loss: 0.2284 - categorical_accuracy: 0.9303
37568/60000 [=================>............] - ETA: 38s - loss: 0.2283 - categorical_accuracy: 0.9303
37600/60000 [=================>............] - ETA: 38s - loss: 0.2282 - categorical_accuracy: 0.9303
37632/60000 [=================>............] - ETA: 38s - loss: 0.2280 - categorical_accuracy: 0.9304
37664/60000 [=================>............] - ETA: 38s - loss: 0.2281 - categorical_accuracy: 0.9304
37696/60000 [=================>............] - ETA: 38s - loss: 0.2279 - categorical_accuracy: 0.9305
37728/60000 [=================>............] - ETA: 38s - loss: 0.2279 - categorical_accuracy: 0.9305
37760/60000 [=================>............] - ETA: 38s - loss: 0.2278 - categorical_accuracy: 0.9305
37792/60000 [=================>............] - ETA: 37s - loss: 0.2277 - categorical_accuracy: 0.9305
37856/60000 [=================>............] - ETA: 37s - loss: 0.2273 - categorical_accuracy: 0.9307
37888/60000 [=================>............] - ETA: 37s - loss: 0.2272 - categorical_accuracy: 0.9307
37920/60000 [=================>............] - ETA: 37s - loss: 0.2270 - categorical_accuracy: 0.9307
37984/60000 [=================>............] - ETA: 37s - loss: 0.2269 - categorical_accuracy: 0.9307
38016/60000 [==================>...........] - ETA: 37s - loss: 0.2268 - categorical_accuracy: 0.9308
38048/60000 [==================>...........] - ETA: 37s - loss: 0.2266 - categorical_accuracy: 0.9309
38080/60000 [==================>...........] - ETA: 37s - loss: 0.2266 - categorical_accuracy: 0.9309
38112/60000 [==================>...........] - ETA: 37s - loss: 0.2265 - categorical_accuracy: 0.9309
38144/60000 [==================>...........] - ETA: 37s - loss: 0.2264 - categorical_accuracy: 0.9309
38176/60000 [==================>...........] - ETA: 37s - loss: 0.2263 - categorical_accuracy: 0.9310
38208/60000 [==================>...........] - ETA: 37s - loss: 0.2261 - categorical_accuracy: 0.9310
38240/60000 [==================>...........] - ETA: 37s - loss: 0.2260 - categorical_accuracy: 0.9310
38272/60000 [==================>...........] - ETA: 37s - loss: 0.2259 - categorical_accuracy: 0.9311
38304/60000 [==================>...........] - ETA: 37s - loss: 0.2257 - categorical_accuracy: 0.9311
38368/60000 [==================>...........] - ETA: 36s - loss: 0.2255 - categorical_accuracy: 0.9311
38432/60000 [==================>...........] - ETA: 36s - loss: 0.2256 - categorical_accuracy: 0.9311
38464/60000 [==================>...........] - ETA: 36s - loss: 0.2254 - categorical_accuracy: 0.9312
38496/60000 [==================>...........] - ETA: 36s - loss: 0.2253 - categorical_accuracy: 0.9312
38528/60000 [==================>...........] - ETA: 36s - loss: 0.2251 - categorical_accuracy: 0.9313
38560/60000 [==================>...........] - ETA: 36s - loss: 0.2252 - categorical_accuracy: 0.9312
38592/60000 [==================>...........] - ETA: 36s - loss: 0.2251 - categorical_accuracy: 0.9313
38624/60000 [==================>...........] - ETA: 36s - loss: 0.2250 - categorical_accuracy: 0.9313
38656/60000 [==================>...........] - ETA: 36s - loss: 0.2248 - categorical_accuracy: 0.9314
38720/60000 [==================>...........] - ETA: 36s - loss: 0.2245 - categorical_accuracy: 0.9314
38752/60000 [==================>...........] - ETA: 36s - loss: 0.2244 - categorical_accuracy: 0.9315
38784/60000 [==================>...........] - ETA: 36s - loss: 0.2242 - categorical_accuracy: 0.9315
38816/60000 [==================>...........] - ETA: 36s - loss: 0.2241 - categorical_accuracy: 0.9316
38848/60000 [==================>...........] - ETA: 36s - loss: 0.2240 - categorical_accuracy: 0.9316
38912/60000 [==================>...........] - ETA: 36s - loss: 0.2239 - categorical_accuracy: 0.9317
38944/60000 [==================>...........] - ETA: 35s - loss: 0.2240 - categorical_accuracy: 0.9316
38976/60000 [==================>...........] - ETA: 35s - loss: 0.2240 - categorical_accuracy: 0.9317
39008/60000 [==================>...........] - ETA: 35s - loss: 0.2238 - categorical_accuracy: 0.9317
39040/60000 [==================>...........] - ETA: 35s - loss: 0.2237 - categorical_accuracy: 0.9317
39072/60000 [==================>...........] - ETA: 35s - loss: 0.2236 - categorical_accuracy: 0.9317
39104/60000 [==================>...........] - ETA: 35s - loss: 0.2235 - categorical_accuracy: 0.9318
39136/60000 [==================>...........] - ETA: 35s - loss: 0.2233 - categorical_accuracy: 0.9318
39168/60000 [==================>...........] - ETA: 35s - loss: 0.2232 - categorical_accuracy: 0.9319
39200/60000 [==================>...........] - ETA: 35s - loss: 0.2231 - categorical_accuracy: 0.9319
39232/60000 [==================>...........] - ETA: 35s - loss: 0.2229 - categorical_accuracy: 0.9320
39264/60000 [==================>...........] - ETA: 35s - loss: 0.2228 - categorical_accuracy: 0.9320
39296/60000 [==================>...........] - ETA: 35s - loss: 0.2229 - categorical_accuracy: 0.9320
39328/60000 [==================>...........] - ETA: 35s - loss: 0.2228 - categorical_accuracy: 0.9320
39360/60000 [==================>...........] - ETA: 35s - loss: 0.2227 - categorical_accuracy: 0.9321
39392/60000 [==================>...........] - ETA: 35s - loss: 0.2225 - categorical_accuracy: 0.9321
39424/60000 [==================>...........] - ETA: 35s - loss: 0.2224 - categorical_accuracy: 0.9322
39456/60000 [==================>...........] - ETA: 35s - loss: 0.2223 - categorical_accuracy: 0.9322
39488/60000 [==================>...........] - ETA: 35s - loss: 0.2223 - categorical_accuracy: 0.9322
39520/60000 [==================>...........] - ETA: 34s - loss: 0.2223 - categorical_accuracy: 0.9322
39552/60000 [==================>...........] - ETA: 34s - loss: 0.2221 - categorical_accuracy: 0.9322
39584/60000 [==================>...........] - ETA: 34s - loss: 0.2220 - categorical_accuracy: 0.9322
39616/60000 [==================>...........] - ETA: 34s - loss: 0.2219 - categorical_accuracy: 0.9323
39648/60000 [==================>...........] - ETA: 34s - loss: 0.2218 - categorical_accuracy: 0.9323
39680/60000 [==================>...........] - ETA: 34s - loss: 0.2218 - categorical_accuracy: 0.9323
39712/60000 [==================>...........] - ETA: 34s - loss: 0.2217 - categorical_accuracy: 0.9324
39744/60000 [==================>...........] - ETA: 34s - loss: 0.2215 - categorical_accuracy: 0.9324
39776/60000 [==================>...........] - ETA: 34s - loss: 0.2213 - categorical_accuracy: 0.9325
39808/60000 [==================>...........] - ETA: 34s - loss: 0.2212 - categorical_accuracy: 0.9325
39840/60000 [==================>...........] - ETA: 34s - loss: 0.2210 - categorical_accuracy: 0.9325
39872/60000 [==================>...........] - ETA: 34s - loss: 0.2209 - categorical_accuracy: 0.9326
39904/60000 [==================>...........] - ETA: 34s - loss: 0.2207 - categorical_accuracy: 0.9326
39936/60000 [==================>...........] - ETA: 34s - loss: 0.2207 - categorical_accuracy: 0.9326
39968/60000 [==================>...........] - ETA: 34s - loss: 0.2206 - categorical_accuracy: 0.9326
40000/60000 [===================>..........] - ETA: 34s - loss: 0.2204 - categorical_accuracy: 0.9327
40032/60000 [===================>..........] - ETA: 34s - loss: 0.2204 - categorical_accuracy: 0.9327
40064/60000 [===================>..........] - ETA: 34s - loss: 0.2203 - categorical_accuracy: 0.9328
40096/60000 [===================>..........] - ETA: 33s - loss: 0.2203 - categorical_accuracy: 0.9328
40128/60000 [===================>..........] - ETA: 33s - loss: 0.2204 - categorical_accuracy: 0.9328
40160/60000 [===================>..........] - ETA: 33s - loss: 0.2203 - categorical_accuracy: 0.9328
40192/60000 [===================>..........] - ETA: 33s - loss: 0.2202 - categorical_accuracy: 0.9328
40224/60000 [===================>..........] - ETA: 33s - loss: 0.2202 - categorical_accuracy: 0.9328
40256/60000 [===================>..........] - ETA: 33s - loss: 0.2201 - categorical_accuracy: 0.9329
40288/60000 [===================>..........] - ETA: 33s - loss: 0.2201 - categorical_accuracy: 0.9329
40320/60000 [===================>..........] - ETA: 33s - loss: 0.2200 - categorical_accuracy: 0.9329
40352/60000 [===================>..........] - ETA: 33s - loss: 0.2199 - categorical_accuracy: 0.9329
40384/60000 [===================>..........] - ETA: 33s - loss: 0.2199 - categorical_accuracy: 0.9329
40416/60000 [===================>..........] - ETA: 33s - loss: 0.2198 - categorical_accuracy: 0.9330
40448/60000 [===================>..........] - ETA: 33s - loss: 0.2197 - categorical_accuracy: 0.9330
40480/60000 [===================>..........] - ETA: 33s - loss: 0.2196 - categorical_accuracy: 0.9330
40512/60000 [===================>..........] - ETA: 33s - loss: 0.2195 - categorical_accuracy: 0.9330
40544/60000 [===================>..........] - ETA: 33s - loss: 0.2195 - categorical_accuracy: 0.9330
40576/60000 [===================>..........] - ETA: 33s - loss: 0.2194 - categorical_accuracy: 0.9331
40608/60000 [===================>..........] - ETA: 33s - loss: 0.2193 - categorical_accuracy: 0.9331
40640/60000 [===================>..........] - ETA: 33s - loss: 0.2192 - categorical_accuracy: 0.9331
40672/60000 [===================>..........] - ETA: 33s - loss: 0.2192 - categorical_accuracy: 0.9332
40704/60000 [===================>..........] - ETA: 32s - loss: 0.2191 - categorical_accuracy: 0.9332
40736/60000 [===================>..........] - ETA: 32s - loss: 0.2190 - categorical_accuracy: 0.9332
40768/60000 [===================>..........] - ETA: 32s - loss: 0.2189 - categorical_accuracy: 0.9332
40800/60000 [===================>..........] - ETA: 32s - loss: 0.2188 - categorical_accuracy: 0.9333
40832/60000 [===================>..........] - ETA: 32s - loss: 0.2188 - categorical_accuracy: 0.9333
40864/60000 [===================>..........] - ETA: 32s - loss: 0.2187 - categorical_accuracy: 0.9333
40896/60000 [===================>..........] - ETA: 32s - loss: 0.2185 - categorical_accuracy: 0.9333
40928/60000 [===================>..........] - ETA: 32s - loss: 0.2186 - categorical_accuracy: 0.9333
40960/60000 [===================>..........] - ETA: 32s - loss: 0.2185 - categorical_accuracy: 0.9334
40992/60000 [===================>..........] - ETA: 32s - loss: 0.2184 - categorical_accuracy: 0.9334
41024/60000 [===================>..........] - ETA: 32s - loss: 0.2183 - categorical_accuracy: 0.9334
41056/60000 [===================>..........] - ETA: 32s - loss: 0.2183 - categorical_accuracy: 0.9334
41088/60000 [===================>..........] - ETA: 32s - loss: 0.2181 - categorical_accuracy: 0.9335
41120/60000 [===================>..........] - ETA: 32s - loss: 0.2180 - categorical_accuracy: 0.9335
41152/60000 [===================>..........] - ETA: 32s - loss: 0.2179 - categorical_accuracy: 0.9335
41184/60000 [===================>..........] - ETA: 32s - loss: 0.2180 - categorical_accuracy: 0.9335
41216/60000 [===================>..........] - ETA: 32s - loss: 0.2178 - categorical_accuracy: 0.9336
41248/60000 [===================>..........] - ETA: 32s - loss: 0.2176 - categorical_accuracy: 0.9336
41280/60000 [===================>..........] - ETA: 31s - loss: 0.2177 - categorical_accuracy: 0.9337
41312/60000 [===================>..........] - ETA: 31s - loss: 0.2177 - categorical_accuracy: 0.9337
41344/60000 [===================>..........] - ETA: 31s - loss: 0.2175 - categorical_accuracy: 0.9337
41376/60000 [===================>..........] - ETA: 31s - loss: 0.2174 - categorical_accuracy: 0.9338
41408/60000 [===================>..........] - ETA: 31s - loss: 0.2173 - categorical_accuracy: 0.9338
41440/60000 [===================>..........] - ETA: 31s - loss: 0.2173 - categorical_accuracy: 0.9338
41472/60000 [===================>..........] - ETA: 31s - loss: 0.2171 - categorical_accuracy: 0.9338
41504/60000 [===================>..........] - ETA: 31s - loss: 0.2170 - categorical_accuracy: 0.9339
41568/60000 [===================>..........] - ETA: 31s - loss: 0.2167 - categorical_accuracy: 0.9340
41600/60000 [===================>..........] - ETA: 31s - loss: 0.2168 - categorical_accuracy: 0.9339
41632/60000 [===================>..........] - ETA: 31s - loss: 0.2168 - categorical_accuracy: 0.9339
41664/60000 [===================>..........] - ETA: 31s - loss: 0.2167 - categorical_accuracy: 0.9339
41696/60000 [===================>..........] - ETA: 31s - loss: 0.2166 - categorical_accuracy: 0.9339
41728/60000 [===================>..........] - ETA: 31s - loss: 0.2168 - categorical_accuracy: 0.9339
41760/60000 [===================>..........] - ETA: 31s - loss: 0.2167 - categorical_accuracy: 0.9339
41792/60000 [===================>..........] - ETA: 31s - loss: 0.2167 - categorical_accuracy: 0.9339
41824/60000 [===================>..........] - ETA: 31s - loss: 0.2166 - categorical_accuracy: 0.9340
41856/60000 [===================>..........] - ETA: 30s - loss: 0.2164 - categorical_accuracy: 0.9340
41888/60000 [===================>..........] - ETA: 30s - loss: 0.2163 - categorical_accuracy: 0.9341
41920/60000 [===================>..........] - ETA: 30s - loss: 0.2162 - categorical_accuracy: 0.9341
41952/60000 [===================>..........] - ETA: 30s - loss: 0.2161 - categorical_accuracy: 0.9342
41984/60000 [===================>..........] - ETA: 30s - loss: 0.2160 - categorical_accuracy: 0.9342
42016/60000 [====================>.........] - ETA: 30s - loss: 0.2160 - categorical_accuracy: 0.9342
42048/60000 [====================>.........] - ETA: 30s - loss: 0.2159 - categorical_accuracy: 0.9342
42080/60000 [====================>.........] - ETA: 30s - loss: 0.2157 - categorical_accuracy: 0.9343
42112/60000 [====================>.........] - ETA: 30s - loss: 0.2156 - categorical_accuracy: 0.9343
42144/60000 [====================>.........] - ETA: 30s - loss: 0.2155 - categorical_accuracy: 0.9344
42176/60000 [====================>.........] - ETA: 30s - loss: 0.2154 - categorical_accuracy: 0.9344
42208/60000 [====================>.........] - ETA: 30s - loss: 0.2152 - categorical_accuracy: 0.9344
42240/60000 [====================>.........] - ETA: 30s - loss: 0.2152 - categorical_accuracy: 0.9344
42272/60000 [====================>.........] - ETA: 30s - loss: 0.2152 - categorical_accuracy: 0.9344
42304/60000 [====================>.........] - ETA: 30s - loss: 0.2150 - categorical_accuracy: 0.9345
42336/60000 [====================>.........] - ETA: 30s - loss: 0.2149 - categorical_accuracy: 0.9345
42368/60000 [====================>.........] - ETA: 30s - loss: 0.2149 - categorical_accuracy: 0.9346
42400/60000 [====================>.........] - ETA: 30s - loss: 0.2148 - categorical_accuracy: 0.9346
42432/60000 [====================>.........] - ETA: 29s - loss: 0.2148 - categorical_accuracy: 0.9346
42464/60000 [====================>.........] - ETA: 29s - loss: 0.2147 - categorical_accuracy: 0.9347
42496/60000 [====================>.........] - ETA: 29s - loss: 0.2145 - categorical_accuracy: 0.9347
42528/60000 [====================>.........] - ETA: 29s - loss: 0.2144 - categorical_accuracy: 0.9347
42560/60000 [====================>.........] - ETA: 29s - loss: 0.2143 - categorical_accuracy: 0.9348
42592/60000 [====================>.........] - ETA: 29s - loss: 0.2142 - categorical_accuracy: 0.9348
42624/60000 [====================>.........] - ETA: 29s - loss: 0.2141 - categorical_accuracy: 0.9348
42656/60000 [====================>.........] - ETA: 29s - loss: 0.2141 - categorical_accuracy: 0.9348
42688/60000 [====================>.........] - ETA: 29s - loss: 0.2140 - categorical_accuracy: 0.9349
42720/60000 [====================>.........] - ETA: 29s - loss: 0.2139 - categorical_accuracy: 0.9349
42752/60000 [====================>.........] - ETA: 29s - loss: 0.2138 - categorical_accuracy: 0.9349
42784/60000 [====================>.........] - ETA: 29s - loss: 0.2137 - categorical_accuracy: 0.9350
42816/60000 [====================>.........] - ETA: 29s - loss: 0.2135 - categorical_accuracy: 0.9350
42848/60000 [====================>.........] - ETA: 29s - loss: 0.2135 - categorical_accuracy: 0.9350
42880/60000 [====================>.........] - ETA: 29s - loss: 0.2134 - categorical_accuracy: 0.9350
42912/60000 [====================>.........] - ETA: 29s - loss: 0.2132 - categorical_accuracy: 0.9351
42944/60000 [====================>.........] - ETA: 29s - loss: 0.2132 - categorical_accuracy: 0.9351
42976/60000 [====================>.........] - ETA: 29s - loss: 0.2130 - categorical_accuracy: 0.9351
43008/60000 [====================>.........] - ETA: 29s - loss: 0.2130 - categorical_accuracy: 0.9352
43040/60000 [====================>.........] - ETA: 28s - loss: 0.2128 - categorical_accuracy: 0.9352
43072/60000 [====================>.........] - ETA: 28s - loss: 0.2127 - categorical_accuracy: 0.9353
43104/60000 [====================>.........] - ETA: 28s - loss: 0.2125 - categorical_accuracy: 0.9353
43168/60000 [====================>.........] - ETA: 28s - loss: 0.2124 - categorical_accuracy: 0.9353
43200/60000 [====================>.........] - ETA: 28s - loss: 0.2122 - categorical_accuracy: 0.9354
43232/60000 [====================>.........] - ETA: 28s - loss: 0.2121 - categorical_accuracy: 0.9354
43264/60000 [====================>.........] - ETA: 28s - loss: 0.2120 - categorical_accuracy: 0.9354
43296/60000 [====================>.........] - ETA: 28s - loss: 0.2120 - categorical_accuracy: 0.9354
43328/60000 [====================>.........] - ETA: 28s - loss: 0.2119 - categorical_accuracy: 0.9355
43360/60000 [====================>.........] - ETA: 28s - loss: 0.2117 - categorical_accuracy: 0.9355
43392/60000 [====================>.........] - ETA: 28s - loss: 0.2116 - categorical_accuracy: 0.9356
43424/60000 [====================>.........] - ETA: 28s - loss: 0.2115 - categorical_accuracy: 0.9356
43456/60000 [====================>.........] - ETA: 28s - loss: 0.2113 - categorical_accuracy: 0.9357
43488/60000 [====================>.........] - ETA: 28s - loss: 0.2113 - categorical_accuracy: 0.9357
43552/60000 [====================>.........] - ETA: 28s - loss: 0.2111 - categorical_accuracy: 0.9358
43584/60000 [====================>.........] - ETA: 28s - loss: 0.2110 - categorical_accuracy: 0.9358
43616/60000 [====================>.........] - ETA: 27s - loss: 0.2109 - categorical_accuracy: 0.9358
43648/60000 [====================>.........] - ETA: 27s - loss: 0.2109 - categorical_accuracy: 0.9359
43680/60000 [====================>.........] - ETA: 27s - loss: 0.2108 - categorical_accuracy: 0.9359
43712/60000 [====================>.........] - ETA: 27s - loss: 0.2108 - categorical_accuracy: 0.9359
43744/60000 [====================>.........] - ETA: 27s - loss: 0.2107 - categorical_accuracy: 0.9359
43776/60000 [====================>.........] - ETA: 27s - loss: 0.2105 - categorical_accuracy: 0.9359
43808/60000 [====================>.........] - ETA: 27s - loss: 0.2105 - categorical_accuracy: 0.9359
43840/60000 [====================>.........] - ETA: 27s - loss: 0.2104 - categorical_accuracy: 0.9360
43872/60000 [====================>.........] - ETA: 27s - loss: 0.2103 - categorical_accuracy: 0.9360
43904/60000 [====================>.........] - ETA: 27s - loss: 0.2102 - categorical_accuracy: 0.9361
43936/60000 [====================>.........] - ETA: 27s - loss: 0.2101 - categorical_accuracy: 0.9361
43968/60000 [====================>.........] - ETA: 27s - loss: 0.2100 - categorical_accuracy: 0.9361
44000/60000 [=====================>........] - ETA: 27s - loss: 0.2099 - categorical_accuracy: 0.9361
44032/60000 [=====================>........] - ETA: 27s - loss: 0.2098 - categorical_accuracy: 0.9361
44064/60000 [=====================>........] - ETA: 27s - loss: 0.2099 - categorical_accuracy: 0.9361
44096/60000 [=====================>........] - ETA: 27s - loss: 0.2099 - categorical_accuracy: 0.9361
44128/60000 [=====================>........] - ETA: 27s - loss: 0.2098 - categorical_accuracy: 0.9361
44160/60000 [=====================>........] - ETA: 27s - loss: 0.2097 - categorical_accuracy: 0.9361
44192/60000 [=====================>........] - ETA: 26s - loss: 0.2096 - categorical_accuracy: 0.9362
44224/60000 [=====================>........] - ETA: 26s - loss: 0.2095 - categorical_accuracy: 0.9362
44256/60000 [=====================>........] - ETA: 26s - loss: 0.2094 - categorical_accuracy: 0.9362
44288/60000 [=====================>........] - ETA: 26s - loss: 0.2096 - categorical_accuracy: 0.9362
44352/60000 [=====================>........] - ETA: 26s - loss: 0.2096 - categorical_accuracy: 0.9363
44384/60000 [=====================>........] - ETA: 26s - loss: 0.2095 - categorical_accuracy: 0.9363
44416/60000 [=====================>........] - ETA: 26s - loss: 0.2095 - categorical_accuracy: 0.9363
44448/60000 [=====================>........] - ETA: 26s - loss: 0.2093 - categorical_accuracy: 0.9363
44480/60000 [=====================>........] - ETA: 26s - loss: 0.2092 - categorical_accuracy: 0.9364
44512/60000 [=====================>........] - ETA: 26s - loss: 0.2091 - categorical_accuracy: 0.9364
44544/60000 [=====================>........] - ETA: 26s - loss: 0.2091 - categorical_accuracy: 0.9364
44576/60000 [=====================>........] - ETA: 26s - loss: 0.2090 - categorical_accuracy: 0.9364
44608/60000 [=====================>........] - ETA: 26s - loss: 0.2089 - categorical_accuracy: 0.9365
44640/60000 [=====================>........] - ETA: 26s - loss: 0.2088 - categorical_accuracy: 0.9365
44672/60000 [=====================>........] - ETA: 26s - loss: 0.2088 - categorical_accuracy: 0.9365
44704/60000 [=====================>........] - ETA: 26s - loss: 0.2087 - categorical_accuracy: 0.9365
44736/60000 [=====================>........] - ETA: 26s - loss: 0.2086 - categorical_accuracy: 0.9366
44768/60000 [=====================>........] - ETA: 26s - loss: 0.2086 - categorical_accuracy: 0.9366
44800/60000 [=====================>........] - ETA: 25s - loss: 0.2084 - categorical_accuracy: 0.9367
44832/60000 [=====================>........] - ETA: 25s - loss: 0.2083 - categorical_accuracy: 0.9367
44864/60000 [=====================>........] - ETA: 25s - loss: 0.2082 - categorical_accuracy: 0.9367
44896/60000 [=====================>........] - ETA: 25s - loss: 0.2081 - categorical_accuracy: 0.9367
44928/60000 [=====================>........] - ETA: 25s - loss: 0.2080 - categorical_accuracy: 0.9368
44960/60000 [=====================>........] - ETA: 25s - loss: 0.2080 - categorical_accuracy: 0.9368
44992/60000 [=====================>........] - ETA: 25s - loss: 0.2080 - categorical_accuracy: 0.9368
45024/60000 [=====================>........] - ETA: 25s - loss: 0.2079 - categorical_accuracy: 0.9368
45056/60000 [=====================>........] - ETA: 25s - loss: 0.2078 - categorical_accuracy: 0.9369
45088/60000 [=====================>........] - ETA: 25s - loss: 0.2077 - categorical_accuracy: 0.9369
45120/60000 [=====================>........] - ETA: 25s - loss: 0.2076 - categorical_accuracy: 0.9369
45152/60000 [=====================>........] - ETA: 25s - loss: 0.2075 - categorical_accuracy: 0.9370
45184/60000 [=====================>........] - ETA: 25s - loss: 0.2074 - categorical_accuracy: 0.9370
45216/60000 [=====================>........] - ETA: 25s - loss: 0.2073 - categorical_accuracy: 0.9370
45248/60000 [=====================>........] - ETA: 25s - loss: 0.2072 - categorical_accuracy: 0.9370
45280/60000 [=====================>........] - ETA: 25s - loss: 0.2072 - categorical_accuracy: 0.9371
45312/60000 [=====================>........] - ETA: 25s - loss: 0.2071 - categorical_accuracy: 0.9371
45344/60000 [=====================>........] - ETA: 25s - loss: 0.2070 - categorical_accuracy: 0.9371
45376/60000 [=====================>........] - ETA: 24s - loss: 0.2070 - categorical_accuracy: 0.9371
45408/60000 [=====================>........] - ETA: 24s - loss: 0.2069 - categorical_accuracy: 0.9371
45440/60000 [=====================>........] - ETA: 24s - loss: 0.2068 - categorical_accuracy: 0.9371
45472/60000 [=====================>........] - ETA: 24s - loss: 0.2066 - categorical_accuracy: 0.9372
45504/60000 [=====================>........] - ETA: 24s - loss: 0.2065 - categorical_accuracy: 0.9372
45536/60000 [=====================>........] - ETA: 24s - loss: 0.2064 - categorical_accuracy: 0.9373
45568/60000 [=====================>........] - ETA: 24s - loss: 0.2063 - categorical_accuracy: 0.9373
45600/60000 [=====================>........] - ETA: 24s - loss: 0.2062 - categorical_accuracy: 0.9373
45632/60000 [=====================>........] - ETA: 24s - loss: 0.2062 - categorical_accuracy: 0.9373
45664/60000 [=====================>........] - ETA: 24s - loss: 0.2062 - categorical_accuracy: 0.9373
45696/60000 [=====================>........] - ETA: 24s - loss: 0.2061 - categorical_accuracy: 0.9373
45728/60000 [=====================>........] - ETA: 24s - loss: 0.2060 - categorical_accuracy: 0.9374
45760/60000 [=====================>........] - ETA: 24s - loss: 0.2059 - categorical_accuracy: 0.9374
45792/60000 [=====================>........] - ETA: 24s - loss: 0.2060 - categorical_accuracy: 0.9374
45824/60000 [=====================>........] - ETA: 24s - loss: 0.2059 - categorical_accuracy: 0.9374
45856/60000 [=====================>........] - ETA: 24s - loss: 0.2058 - categorical_accuracy: 0.9375
45888/60000 [=====================>........] - ETA: 24s - loss: 0.2056 - categorical_accuracy: 0.9375
45920/60000 [=====================>........] - ETA: 24s - loss: 0.2055 - categorical_accuracy: 0.9375
45952/60000 [=====================>........] - ETA: 23s - loss: 0.2055 - categorical_accuracy: 0.9376
45984/60000 [=====================>........] - ETA: 23s - loss: 0.2054 - categorical_accuracy: 0.9376
46016/60000 [======================>.......] - ETA: 23s - loss: 0.2053 - categorical_accuracy: 0.9376
46048/60000 [======================>.......] - ETA: 23s - loss: 0.2052 - categorical_accuracy: 0.9377
46080/60000 [======================>.......] - ETA: 23s - loss: 0.2051 - categorical_accuracy: 0.9377
46112/60000 [======================>.......] - ETA: 23s - loss: 0.2051 - categorical_accuracy: 0.9377
46144/60000 [======================>.......] - ETA: 23s - loss: 0.2049 - categorical_accuracy: 0.9377
46176/60000 [======================>.......] - ETA: 23s - loss: 0.2048 - categorical_accuracy: 0.9378
46208/60000 [======================>.......] - ETA: 23s - loss: 0.2047 - categorical_accuracy: 0.9378
46240/60000 [======================>.......] - ETA: 23s - loss: 0.2047 - categorical_accuracy: 0.9378
46272/60000 [======================>.......] - ETA: 23s - loss: 0.2046 - categorical_accuracy: 0.9378
46304/60000 [======================>.......] - ETA: 23s - loss: 0.2046 - categorical_accuracy: 0.9378
46336/60000 [======================>.......] - ETA: 23s - loss: 0.2045 - categorical_accuracy: 0.9378
46368/60000 [======================>.......] - ETA: 23s - loss: 0.2043 - categorical_accuracy: 0.9379
46400/60000 [======================>.......] - ETA: 23s - loss: 0.2043 - categorical_accuracy: 0.9379
46432/60000 [======================>.......] - ETA: 23s - loss: 0.2042 - categorical_accuracy: 0.9380
46464/60000 [======================>.......] - ETA: 23s - loss: 0.2041 - categorical_accuracy: 0.9380
46496/60000 [======================>.......] - ETA: 23s - loss: 0.2040 - categorical_accuracy: 0.9380
46528/60000 [======================>.......] - ETA: 23s - loss: 0.2039 - categorical_accuracy: 0.9380
46560/60000 [======================>.......] - ETA: 22s - loss: 0.2039 - categorical_accuracy: 0.9381
46592/60000 [======================>.......] - ETA: 22s - loss: 0.2042 - categorical_accuracy: 0.9380
46624/60000 [======================>.......] - ETA: 22s - loss: 0.2042 - categorical_accuracy: 0.9380
46656/60000 [======================>.......] - ETA: 22s - loss: 0.2041 - categorical_accuracy: 0.9381
46688/60000 [======================>.......] - ETA: 22s - loss: 0.2040 - categorical_accuracy: 0.9381
46720/60000 [======================>.......] - ETA: 22s - loss: 0.2039 - categorical_accuracy: 0.9381
46784/60000 [======================>.......] - ETA: 22s - loss: 0.2038 - categorical_accuracy: 0.9382
46816/60000 [======================>.......] - ETA: 22s - loss: 0.2037 - categorical_accuracy: 0.9382
46848/60000 [======================>.......] - ETA: 22s - loss: 0.2036 - categorical_accuracy: 0.9382
46880/60000 [======================>.......] - ETA: 22s - loss: 0.2036 - categorical_accuracy: 0.9382
46912/60000 [======================>.......] - ETA: 22s - loss: 0.2035 - categorical_accuracy: 0.9382
46944/60000 [======================>.......] - ETA: 22s - loss: 0.2034 - categorical_accuracy: 0.9383
46976/60000 [======================>.......] - ETA: 22s - loss: 0.2034 - categorical_accuracy: 0.9383
47040/60000 [======================>.......] - ETA: 22s - loss: 0.2032 - categorical_accuracy: 0.9384
47072/60000 [======================>.......] - ETA: 22s - loss: 0.2031 - categorical_accuracy: 0.9384
47104/60000 [======================>.......] - ETA: 22s - loss: 0.2030 - categorical_accuracy: 0.9384
47136/60000 [======================>.......] - ETA: 21s - loss: 0.2030 - categorical_accuracy: 0.9384
47168/60000 [======================>.......] - ETA: 21s - loss: 0.2029 - categorical_accuracy: 0.9385
47200/60000 [======================>.......] - ETA: 21s - loss: 0.2028 - categorical_accuracy: 0.9385
47232/60000 [======================>.......] - ETA: 21s - loss: 0.2027 - categorical_accuracy: 0.9385
47264/60000 [======================>.......] - ETA: 21s - loss: 0.2028 - categorical_accuracy: 0.9385
47296/60000 [======================>.......] - ETA: 21s - loss: 0.2026 - categorical_accuracy: 0.9385
47328/60000 [======================>.......] - ETA: 21s - loss: 0.2025 - categorical_accuracy: 0.9386
47360/60000 [======================>.......] - ETA: 21s - loss: 0.2026 - categorical_accuracy: 0.9386
47392/60000 [======================>.......] - ETA: 21s - loss: 0.2024 - categorical_accuracy: 0.9386
47424/60000 [======================>.......] - ETA: 21s - loss: 0.2024 - categorical_accuracy: 0.9386
47456/60000 [======================>.......] - ETA: 21s - loss: 0.2023 - categorical_accuracy: 0.9387
47488/60000 [======================>.......] - ETA: 21s - loss: 0.2022 - categorical_accuracy: 0.9387
47552/60000 [======================>.......] - ETA: 21s - loss: 0.2022 - categorical_accuracy: 0.9387
47584/60000 [======================>.......] - ETA: 21s - loss: 0.2020 - categorical_accuracy: 0.9387
47616/60000 [======================>.......] - ETA: 21s - loss: 0.2020 - categorical_accuracy: 0.9387
47648/60000 [======================>.......] - ETA: 21s - loss: 0.2019 - categorical_accuracy: 0.9388
47680/60000 [======================>.......] - ETA: 21s - loss: 0.2019 - categorical_accuracy: 0.9388
47712/60000 [======================>.......] - ETA: 20s - loss: 0.2018 - categorical_accuracy: 0.9388
47744/60000 [======================>.......] - ETA: 20s - loss: 0.2017 - categorical_accuracy: 0.9388
47776/60000 [======================>.......] - ETA: 20s - loss: 0.2017 - categorical_accuracy: 0.9388
47808/60000 [======================>.......] - ETA: 20s - loss: 0.2015 - categorical_accuracy: 0.9389
47840/60000 [======================>.......] - ETA: 20s - loss: 0.2014 - categorical_accuracy: 0.9389
47872/60000 [======================>.......] - ETA: 20s - loss: 0.2015 - categorical_accuracy: 0.9389
47904/60000 [======================>.......] - ETA: 20s - loss: 0.2014 - categorical_accuracy: 0.9389
47936/60000 [======================>.......] - ETA: 20s - loss: 0.2013 - categorical_accuracy: 0.9390
47968/60000 [======================>.......] - ETA: 20s - loss: 0.2015 - categorical_accuracy: 0.9390
48000/60000 [=======================>......] - ETA: 20s - loss: 0.2014 - categorical_accuracy: 0.9390
48032/60000 [=======================>......] - ETA: 20s - loss: 0.2013 - categorical_accuracy: 0.9390
48064/60000 [=======================>......] - ETA: 20s - loss: 0.2012 - categorical_accuracy: 0.9391
48096/60000 [=======================>......] - ETA: 20s - loss: 0.2012 - categorical_accuracy: 0.9391
48128/60000 [=======================>......] - ETA: 20s - loss: 0.2010 - categorical_accuracy: 0.9391
48160/60000 [=======================>......] - ETA: 20s - loss: 0.2010 - categorical_accuracy: 0.9391
48192/60000 [=======================>......] - ETA: 20s - loss: 0.2009 - categorical_accuracy: 0.9392
48224/60000 [=======================>......] - ETA: 20s - loss: 0.2008 - categorical_accuracy: 0.9392
48256/60000 [=======================>......] - ETA: 20s - loss: 0.2009 - categorical_accuracy: 0.9392
48288/60000 [=======================>......] - ETA: 20s - loss: 0.2008 - categorical_accuracy: 0.9392
48320/60000 [=======================>......] - ETA: 19s - loss: 0.2007 - categorical_accuracy: 0.9392
48352/60000 [=======================>......] - ETA: 19s - loss: 0.2007 - categorical_accuracy: 0.9392
48384/60000 [=======================>......] - ETA: 19s - loss: 0.2006 - categorical_accuracy: 0.9392
48416/60000 [=======================>......] - ETA: 19s - loss: 0.2005 - categorical_accuracy: 0.9393
48448/60000 [=======================>......] - ETA: 19s - loss: 0.2004 - categorical_accuracy: 0.9393
48480/60000 [=======================>......] - ETA: 19s - loss: 0.2004 - categorical_accuracy: 0.9393
48512/60000 [=======================>......] - ETA: 19s - loss: 0.2003 - categorical_accuracy: 0.9393
48544/60000 [=======================>......] - ETA: 19s - loss: 0.2002 - categorical_accuracy: 0.9394
48576/60000 [=======================>......] - ETA: 19s - loss: 0.2002 - categorical_accuracy: 0.9394
48608/60000 [=======================>......] - ETA: 19s - loss: 0.2001 - categorical_accuracy: 0.9394
48640/60000 [=======================>......] - ETA: 19s - loss: 0.2000 - categorical_accuracy: 0.9394
48672/60000 [=======================>......] - ETA: 19s - loss: 0.1999 - categorical_accuracy: 0.9395
48704/60000 [=======================>......] - ETA: 19s - loss: 0.1999 - categorical_accuracy: 0.9395
48736/60000 [=======================>......] - ETA: 19s - loss: 0.2000 - categorical_accuracy: 0.9394
48768/60000 [=======================>......] - ETA: 19s - loss: 0.1999 - categorical_accuracy: 0.9395
48800/60000 [=======================>......] - ETA: 19s - loss: 0.1999 - categorical_accuracy: 0.9395
48832/60000 [=======================>......] - ETA: 19s - loss: 0.1998 - categorical_accuracy: 0.9395
48864/60000 [=======================>......] - ETA: 19s - loss: 0.1997 - categorical_accuracy: 0.9395
48896/60000 [=======================>......] - ETA: 18s - loss: 0.1997 - categorical_accuracy: 0.9395
48928/60000 [=======================>......] - ETA: 18s - loss: 0.1996 - categorical_accuracy: 0.9396
48960/60000 [=======================>......] - ETA: 18s - loss: 0.1994 - categorical_accuracy: 0.9396
48992/60000 [=======================>......] - ETA: 18s - loss: 0.1994 - categorical_accuracy: 0.9396
49024/60000 [=======================>......] - ETA: 18s - loss: 0.1993 - categorical_accuracy: 0.9396
49056/60000 [=======================>......] - ETA: 18s - loss: 0.1992 - categorical_accuracy: 0.9397
49088/60000 [=======================>......] - ETA: 18s - loss: 0.1991 - categorical_accuracy: 0.9397
49120/60000 [=======================>......] - ETA: 18s - loss: 0.1990 - categorical_accuracy: 0.9397
49152/60000 [=======================>......] - ETA: 18s - loss: 0.1990 - categorical_accuracy: 0.9397
49184/60000 [=======================>......] - ETA: 18s - loss: 0.1989 - categorical_accuracy: 0.9397
49216/60000 [=======================>......] - ETA: 18s - loss: 0.1988 - categorical_accuracy: 0.9398
49248/60000 [=======================>......] - ETA: 18s - loss: 0.1988 - categorical_accuracy: 0.9398
49280/60000 [=======================>......] - ETA: 18s - loss: 0.1987 - categorical_accuracy: 0.9398
49312/60000 [=======================>......] - ETA: 18s - loss: 0.1987 - categorical_accuracy: 0.9398
49344/60000 [=======================>......] - ETA: 18s - loss: 0.1987 - categorical_accuracy: 0.9398
49376/60000 [=======================>......] - ETA: 18s - loss: 0.1986 - categorical_accuracy: 0.9399
49408/60000 [=======================>......] - ETA: 18s - loss: 0.1984 - categorical_accuracy: 0.9399
49440/60000 [=======================>......] - ETA: 18s - loss: 0.1985 - categorical_accuracy: 0.9399
49472/60000 [=======================>......] - ETA: 17s - loss: 0.1984 - categorical_accuracy: 0.9399
49504/60000 [=======================>......] - ETA: 17s - loss: 0.1983 - categorical_accuracy: 0.9399
49536/60000 [=======================>......] - ETA: 17s - loss: 0.1982 - categorical_accuracy: 0.9400
49568/60000 [=======================>......] - ETA: 17s - loss: 0.1982 - categorical_accuracy: 0.9400
49600/60000 [=======================>......] - ETA: 17s - loss: 0.1981 - categorical_accuracy: 0.9400
49632/60000 [=======================>......] - ETA: 17s - loss: 0.1980 - categorical_accuracy: 0.9400
49664/60000 [=======================>......] - ETA: 17s - loss: 0.1979 - categorical_accuracy: 0.9400
49696/60000 [=======================>......] - ETA: 17s - loss: 0.1979 - categorical_accuracy: 0.9400
49728/60000 [=======================>......] - ETA: 17s - loss: 0.1979 - categorical_accuracy: 0.9400
49760/60000 [=======================>......] - ETA: 17s - loss: 0.1978 - categorical_accuracy: 0.9401
49792/60000 [=======================>......] - ETA: 17s - loss: 0.1977 - categorical_accuracy: 0.9401
49824/60000 [=======================>......] - ETA: 17s - loss: 0.1976 - categorical_accuracy: 0.9401
49856/60000 [=======================>......] - ETA: 17s - loss: 0.1976 - categorical_accuracy: 0.9401
49888/60000 [=======================>......] - ETA: 17s - loss: 0.1976 - categorical_accuracy: 0.9401
49920/60000 [=======================>......] - ETA: 17s - loss: 0.1975 - categorical_accuracy: 0.9401
49952/60000 [=======================>......] - ETA: 17s - loss: 0.1974 - categorical_accuracy: 0.9401
49984/60000 [=======================>......] - ETA: 17s - loss: 0.1973 - categorical_accuracy: 0.9402
50016/60000 [========================>.....] - ETA: 17s - loss: 0.1974 - categorical_accuracy: 0.9401
50048/60000 [========================>.....] - ETA: 17s - loss: 0.1973 - categorical_accuracy: 0.9402
50080/60000 [========================>.....] - ETA: 16s - loss: 0.1973 - categorical_accuracy: 0.9402
50112/60000 [========================>.....] - ETA: 16s - loss: 0.1972 - categorical_accuracy: 0.9402
50144/60000 [========================>.....] - ETA: 16s - loss: 0.1971 - categorical_accuracy: 0.9402
50176/60000 [========================>.....] - ETA: 16s - loss: 0.1971 - categorical_accuracy: 0.9402
50208/60000 [========================>.....] - ETA: 16s - loss: 0.1970 - categorical_accuracy: 0.9402
50240/60000 [========================>.....] - ETA: 16s - loss: 0.1969 - categorical_accuracy: 0.9403
50272/60000 [========================>.....] - ETA: 16s - loss: 0.1968 - categorical_accuracy: 0.9403
50304/60000 [========================>.....] - ETA: 16s - loss: 0.1967 - categorical_accuracy: 0.9403
50336/60000 [========================>.....] - ETA: 16s - loss: 0.1966 - categorical_accuracy: 0.9404
50368/60000 [========================>.....] - ETA: 16s - loss: 0.1965 - categorical_accuracy: 0.9404
50400/60000 [========================>.....] - ETA: 16s - loss: 0.1964 - categorical_accuracy: 0.9404
50432/60000 [========================>.....] - ETA: 16s - loss: 0.1963 - categorical_accuracy: 0.9405
50464/60000 [========================>.....] - ETA: 16s - loss: 0.1962 - categorical_accuracy: 0.9405
50496/60000 [========================>.....] - ETA: 16s - loss: 0.1961 - categorical_accuracy: 0.9405
50528/60000 [========================>.....] - ETA: 16s - loss: 0.1962 - categorical_accuracy: 0.9405
50560/60000 [========================>.....] - ETA: 16s - loss: 0.1961 - categorical_accuracy: 0.9406
50592/60000 [========================>.....] - ETA: 16s - loss: 0.1960 - categorical_accuracy: 0.9406
50624/60000 [========================>.....] - ETA: 16s - loss: 0.1959 - categorical_accuracy: 0.9406
50656/60000 [========================>.....] - ETA: 15s - loss: 0.1959 - categorical_accuracy: 0.9406
50688/60000 [========================>.....] - ETA: 15s - loss: 0.1958 - categorical_accuracy: 0.9406
50720/60000 [========================>.....] - ETA: 15s - loss: 0.1958 - categorical_accuracy: 0.9407
50752/60000 [========================>.....] - ETA: 15s - loss: 0.1957 - categorical_accuracy: 0.9407
50784/60000 [========================>.....] - ETA: 15s - loss: 0.1959 - categorical_accuracy: 0.9407
50816/60000 [========================>.....] - ETA: 15s - loss: 0.1958 - categorical_accuracy: 0.9407
50848/60000 [========================>.....] - ETA: 15s - loss: 0.1957 - categorical_accuracy: 0.9407
50880/60000 [========================>.....] - ETA: 15s - loss: 0.1956 - categorical_accuracy: 0.9407
50944/60000 [========================>.....] - ETA: 15s - loss: 0.1954 - categorical_accuracy: 0.9408
50976/60000 [========================>.....] - ETA: 15s - loss: 0.1954 - categorical_accuracy: 0.9408
51040/60000 [========================>.....] - ETA: 15s - loss: 0.1951 - categorical_accuracy: 0.9409
51104/60000 [========================>.....] - ETA: 15s - loss: 0.1949 - categorical_accuracy: 0.9409
51168/60000 [========================>.....] - ETA: 15s - loss: 0.1948 - categorical_accuracy: 0.9410
51200/60000 [========================>.....] - ETA: 15s - loss: 0.1946 - categorical_accuracy: 0.9410
51232/60000 [========================>.....] - ETA: 14s - loss: 0.1946 - categorical_accuracy: 0.9410
51264/60000 [========================>.....] - ETA: 14s - loss: 0.1945 - categorical_accuracy: 0.9411
51296/60000 [========================>.....] - ETA: 14s - loss: 0.1944 - categorical_accuracy: 0.9411
51328/60000 [========================>.....] - ETA: 14s - loss: 0.1943 - categorical_accuracy: 0.9411
51360/60000 [========================>.....] - ETA: 14s - loss: 0.1942 - categorical_accuracy: 0.9411
51392/60000 [========================>.....] - ETA: 14s - loss: 0.1942 - categorical_accuracy: 0.9412
51424/60000 [========================>.....] - ETA: 14s - loss: 0.1941 - categorical_accuracy: 0.9412
51456/60000 [========================>.....] - ETA: 14s - loss: 0.1940 - categorical_accuracy: 0.9412
51488/60000 [========================>.....] - ETA: 14s - loss: 0.1939 - categorical_accuracy: 0.9412
51520/60000 [========================>.....] - ETA: 14s - loss: 0.1940 - categorical_accuracy: 0.9412
51552/60000 [========================>.....] - ETA: 14s - loss: 0.1938 - categorical_accuracy: 0.9412
51584/60000 [========================>.....] - ETA: 14s - loss: 0.1938 - categorical_accuracy: 0.9412
51616/60000 [========================>.....] - ETA: 14s - loss: 0.1937 - categorical_accuracy: 0.9413
51648/60000 [========================>.....] - ETA: 14s - loss: 0.1937 - categorical_accuracy: 0.9413
51680/60000 [========================>.....] - ETA: 14s - loss: 0.1937 - categorical_accuracy: 0.9413
51744/60000 [========================>.....] - ETA: 14s - loss: 0.1935 - categorical_accuracy: 0.9414
51776/60000 [========================>.....] - ETA: 14s - loss: 0.1935 - categorical_accuracy: 0.9414
51808/60000 [========================>.....] - ETA: 13s - loss: 0.1934 - categorical_accuracy: 0.9414
51840/60000 [========================>.....] - ETA: 13s - loss: 0.1933 - categorical_accuracy: 0.9414
51872/60000 [========================>.....] - ETA: 13s - loss: 0.1932 - categorical_accuracy: 0.9415
51904/60000 [========================>.....] - ETA: 13s - loss: 0.1932 - categorical_accuracy: 0.9415
51936/60000 [========================>.....] - ETA: 13s - loss: 0.1930 - categorical_accuracy: 0.9415
51968/60000 [========================>.....] - ETA: 13s - loss: 0.1930 - categorical_accuracy: 0.9415
52000/60000 [=========================>....] - ETA: 13s - loss: 0.1929 - categorical_accuracy: 0.9416
52032/60000 [=========================>....] - ETA: 13s - loss: 0.1928 - categorical_accuracy: 0.9416
52064/60000 [=========================>....] - ETA: 13s - loss: 0.1929 - categorical_accuracy: 0.9416
52096/60000 [=========================>....] - ETA: 13s - loss: 0.1928 - categorical_accuracy: 0.9416
52160/60000 [=========================>....] - ETA: 13s - loss: 0.1926 - categorical_accuracy: 0.9417
52192/60000 [=========================>....] - ETA: 13s - loss: 0.1925 - categorical_accuracy: 0.9417
52224/60000 [=========================>....] - ETA: 13s - loss: 0.1924 - categorical_accuracy: 0.9417
52256/60000 [=========================>....] - ETA: 13s - loss: 0.1923 - categorical_accuracy: 0.9417
52288/60000 [=========================>....] - ETA: 13s - loss: 0.1925 - categorical_accuracy: 0.9418
52320/60000 [=========================>....] - ETA: 13s - loss: 0.1924 - categorical_accuracy: 0.9418
52352/60000 [=========================>....] - ETA: 13s - loss: 0.1923 - categorical_accuracy: 0.9418
52416/60000 [=========================>....] - ETA: 12s - loss: 0.1922 - categorical_accuracy: 0.9418
52448/60000 [=========================>....] - ETA: 12s - loss: 0.1922 - categorical_accuracy: 0.9419
52480/60000 [=========================>....] - ETA: 12s - loss: 0.1921 - categorical_accuracy: 0.9419
52512/60000 [=========================>....] - ETA: 12s - loss: 0.1921 - categorical_accuracy: 0.9419
52544/60000 [=========================>....] - ETA: 12s - loss: 0.1923 - categorical_accuracy: 0.9419
52576/60000 [=========================>....] - ETA: 12s - loss: 0.1922 - categorical_accuracy: 0.9419
52608/60000 [=========================>....] - ETA: 12s - loss: 0.1921 - categorical_accuracy: 0.9419
52640/60000 [=========================>....] - ETA: 12s - loss: 0.1921 - categorical_accuracy: 0.9419
52672/60000 [=========================>....] - ETA: 12s - loss: 0.1921 - categorical_accuracy: 0.9419
52704/60000 [=========================>....] - ETA: 12s - loss: 0.1920 - categorical_accuracy: 0.9420
52736/60000 [=========================>....] - ETA: 12s - loss: 0.1919 - categorical_accuracy: 0.9420
52768/60000 [=========================>....] - ETA: 12s - loss: 0.1920 - categorical_accuracy: 0.9419
52800/60000 [=========================>....] - ETA: 12s - loss: 0.1920 - categorical_accuracy: 0.9419
52832/60000 [=========================>....] - ETA: 12s - loss: 0.1919 - categorical_accuracy: 0.9419
52864/60000 [=========================>....] - ETA: 12s - loss: 0.1919 - categorical_accuracy: 0.9420
52896/60000 [=========================>....] - ETA: 12s - loss: 0.1919 - categorical_accuracy: 0.9420
52928/60000 [=========================>....] - ETA: 12s - loss: 0.1918 - categorical_accuracy: 0.9420
52960/60000 [=========================>....] - ETA: 12s - loss: 0.1918 - categorical_accuracy: 0.9420
52992/60000 [=========================>....] - ETA: 11s - loss: 0.1918 - categorical_accuracy: 0.9420
53024/60000 [=========================>....] - ETA: 11s - loss: 0.1917 - categorical_accuracy: 0.9420
53056/60000 [=========================>....] - ETA: 11s - loss: 0.1916 - categorical_accuracy: 0.9421
53088/60000 [=========================>....] - ETA: 11s - loss: 0.1915 - categorical_accuracy: 0.9421
53120/60000 [=========================>....] - ETA: 11s - loss: 0.1914 - categorical_accuracy: 0.9421
53152/60000 [=========================>....] - ETA: 11s - loss: 0.1913 - categorical_accuracy: 0.9421
53184/60000 [=========================>....] - ETA: 11s - loss: 0.1912 - categorical_accuracy: 0.9422
53216/60000 [=========================>....] - ETA: 11s - loss: 0.1911 - categorical_accuracy: 0.9422
53248/60000 [=========================>....] - ETA: 11s - loss: 0.1911 - categorical_accuracy: 0.9422
53280/60000 [=========================>....] - ETA: 11s - loss: 0.1911 - categorical_accuracy: 0.9422
53312/60000 [=========================>....] - ETA: 11s - loss: 0.1912 - categorical_accuracy: 0.9422
53344/60000 [=========================>....] - ETA: 11s - loss: 0.1911 - categorical_accuracy: 0.9423
53376/60000 [=========================>....] - ETA: 11s - loss: 0.1910 - categorical_accuracy: 0.9423
53408/60000 [=========================>....] - ETA: 11s - loss: 0.1911 - categorical_accuracy: 0.9423
53440/60000 [=========================>....] - ETA: 11s - loss: 0.1910 - categorical_accuracy: 0.9423
53504/60000 [=========================>....] - ETA: 11s - loss: 0.1908 - categorical_accuracy: 0.9424
53536/60000 [=========================>....] - ETA: 11s - loss: 0.1907 - categorical_accuracy: 0.9424
53568/60000 [=========================>....] - ETA: 10s - loss: 0.1908 - categorical_accuracy: 0.9424
53600/60000 [=========================>....] - ETA: 10s - loss: 0.1907 - categorical_accuracy: 0.9424
53632/60000 [=========================>....] - ETA: 10s - loss: 0.1907 - categorical_accuracy: 0.9424
53664/60000 [=========================>....] - ETA: 10s - loss: 0.1906 - categorical_accuracy: 0.9424
53696/60000 [=========================>....] - ETA: 10s - loss: 0.1905 - categorical_accuracy: 0.9425
53728/60000 [=========================>....] - ETA: 10s - loss: 0.1904 - categorical_accuracy: 0.9425
53760/60000 [=========================>....] - ETA: 10s - loss: 0.1904 - categorical_accuracy: 0.9425
53792/60000 [=========================>....] - ETA: 10s - loss: 0.1904 - categorical_accuracy: 0.9425
53824/60000 [=========================>....] - ETA: 10s - loss: 0.1903 - categorical_accuracy: 0.9425
53856/60000 [=========================>....] - ETA: 10s - loss: 0.1902 - categorical_accuracy: 0.9426
53888/60000 [=========================>....] - ETA: 10s - loss: 0.1901 - categorical_accuracy: 0.9426
53920/60000 [=========================>....] - ETA: 10s - loss: 0.1901 - categorical_accuracy: 0.9426
53952/60000 [=========================>....] - ETA: 10s - loss: 0.1900 - categorical_accuracy: 0.9426
53984/60000 [=========================>....] - ETA: 10s - loss: 0.1899 - categorical_accuracy: 0.9426
54016/60000 [==========================>...] - ETA: 10s - loss: 0.1898 - categorical_accuracy: 0.9427
54080/60000 [==========================>...] - ETA: 10s - loss: 0.1897 - categorical_accuracy: 0.9427
54112/60000 [==========================>...] - ETA: 10s - loss: 0.1897 - categorical_accuracy: 0.9427
54176/60000 [==========================>...] - ETA: 9s - loss: 0.1896 - categorical_accuracy: 0.9428 
54208/60000 [==========================>...] - ETA: 9s - loss: 0.1895 - categorical_accuracy: 0.9428
54240/60000 [==========================>...] - ETA: 9s - loss: 0.1896 - categorical_accuracy: 0.9428
54272/60000 [==========================>...] - ETA: 9s - loss: 0.1896 - categorical_accuracy: 0.9428
54304/60000 [==========================>...] - ETA: 9s - loss: 0.1895 - categorical_accuracy: 0.9428
54368/60000 [==========================>...] - ETA: 9s - loss: 0.1895 - categorical_accuracy: 0.9429
54400/60000 [==========================>...] - ETA: 9s - loss: 0.1894 - categorical_accuracy: 0.9429
54432/60000 [==========================>...] - ETA: 9s - loss: 0.1894 - categorical_accuracy: 0.9429
54464/60000 [==========================>...] - ETA: 9s - loss: 0.1893 - categorical_accuracy: 0.9429
54496/60000 [==========================>...] - ETA: 9s - loss: 0.1893 - categorical_accuracy: 0.9429
54528/60000 [==========================>...] - ETA: 9s - loss: 0.1893 - categorical_accuracy: 0.9429
54560/60000 [==========================>...] - ETA: 9s - loss: 0.1892 - categorical_accuracy: 0.9429
54592/60000 [==========================>...] - ETA: 9s - loss: 0.1891 - categorical_accuracy: 0.9429
54624/60000 [==========================>...] - ETA: 9s - loss: 0.1891 - categorical_accuracy: 0.9429
54656/60000 [==========================>...] - ETA: 9s - loss: 0.1890 - categorical_accuracy: 0.9430
54688/60000 [==========================>...] - ETA: 9s - loss: 0.1889 - categorical_accuracy: 0.9430
54720/60000 [==========================>...] - ETA: 9s - loss: 0.1888 - categorical_accuracy: 0.9430
54752/60000 [==========================>...] - ETA: 8s - loss: 0.1888 - categorical_accuracy: 0.9430
54784/60000 [==========================>...] - ETA: 8s - loss: 0.1887 - categorical_accuracy: 0.9430
54816/60000 [==========================>...] - ETA: 8s - loss: 0.1886 - categorical_accuracy: 0.9431
54848/60000 [==========================>...] - ETA: 8s - loss: 0.1886 - categorical_accuracy: 0.9431
54880/60000 [==========================>...] - ETA: 8s - loss: 0.1885 - categorical_accuracy: 0.9431
54912/60000 [==========================>...] - ETA: 8s - loss: 0.1884 - categorical_accuracy: 0.9431
54944/60000 [==========================>...] - ETA: 8s - loss: 0.1883 - categorical_accuracy: 0.9432
54976/60000 [==========================>...] - ETA: 8s - loss: 0.1883 - categorical_accuracy: 0.9432
55008/60000 [==========================>...] - ETA: 8s - loss: 0.1882 - categorical_accuracy: 0.9432
55040/60000 [==========================>...] - ETA: 8s - loss: 0.1881 - categorical_accuracy: 0.9432
55072/60000 [==========================>...] - ETA: 8s - loss: 0.1880 - categorical_accuracy: 0.9432
55104/60000 [==========================>...] - ETA: 8s - loss: 0.1880 - categorical_accuracy: 0.9433
55136/60000 [==========================>...] - ETA: 8s - loss: 0.1879 - categorical_accuracy: 0.9433
55168/60000 [==========================>...] - ETA: 8s - loss: 0.1878 - categorical_accuracy: 0.9433
55200/60000 [==========================>...] - ETA: 8s - loss: 0.1878 - categorical_accuracy: 0.9433
55232/60000 [==========================>...] - ETA: 8s - loss: 0.1877 - categorical_accuracy: 0.9433
55264/60000 [==========================>...] - ETA: 8s - loss: 0.1877 - categorical_accuracy: 0.9433
55296/60000 [==========================>...] - ETA: 8s - loss: 0.1876 - categorical_accuracy: 0.9434
55328/60000 [==========================>...] - ETA: 7s - loss: 0.1875 - categorical_accuracy: 0.9434
55360/60000 [==========================>...] - ETA: 7s - loss: 0.1875 - categorical_accuracy: 0.9434
55392/60000 [==========================>...] - ETA: 7s - loss: 0.1875 - categorical_accuracy: 0.9434
55424/60000 [==========================>...] - ETA: 7s - loss: 0.1875 - categorical_accuracy: 0.9434
55456/60000 [==========================>...] - ETA: 7s - loss: 0.1874 - categorical_accuracy: 0.9434
55488/60000 [==========================>...] - ETA: 7s - loss: 0.1873 - categorical_accuracy: 0.9435
55520/60000 [==========================>...] - ETA: 7s - loss: 0.1872 - categorical_accuracy: 0.9435
55584/60000 [==========================>...] - ETA: 7s - loss: 0.1871 - categorical_accuracy: 0.9435
55648/60000 [==========================>...] - ETA: 7s - loss: 0.1870 - categorical_accuracy: 0.9436
55680/60000 [==========================>...] - ETA: 7s - loss: 0.1869 - categorical_accuracy: 0.9436
55712/60000 [==========================>...] - ETA: 7s - loss: 0.1869 - categorical_accuracy: 0.9435
55744/60000 [==========================>...] - ETA: 7s - loss: 0.1869 - categorical_accuracy: 0.9436
55776/60000 [==========================>...] - ETA: 7s - loss: 0.1868 - categorical_accuracy: 0.9436
55808/60000 [==========================>...] - ETA: 7s - loss: 0.1868 - categorical_accuracy: 0.9436
55840/60000 [==========================>...] - ETA: 7s - loss: 0.1867 - categorical_accuracy: 0.9436
55872/60000 [==========================>...] - ETA: 7s - loss: 0.1866 - categorical_accuracy: 0.9436
55936/60000 [==========================>...] - ETA: 6s - loss: 0.1864 - categorical_accuracy: 0.9437
55968/60000 [==========================>...] - ETA: 6s - loss: 0.1863 - categorical_accuracy: 0.9437
56000/60000 [===========================>..] - ETA: 6s - loss: 0.1863 - categorical_accuracy: 0.9437
56032/60000 [===========================>..] - ETA: 6s - loss: 0.1862 - categorical_accuracy: 0.9437
56064/60000 [===========================>..] - ETA: 6s - loss: 0.1861 - categorical_accuracy: 0.9437
56096/60000 [===========================>..] - ETA: 6s - loss: 0.1860 - categorical_accuracy: 0.9438
56128/60000 [===========================>..] - ETA: 6s - loss: 0.1860 - categorical_accuracy: 0.9438
56160/60000 [===========================>..] - ETA: 6s - loss: 0.1859 - categorical_accuracy: 0.9438
56192/60000 [===========================>..] - ETA: 6s - loss: 0.1859 - categorical_accuracy: 0.9438
56224/60000 [===========================>..] - ETA: 6s - loss: 0.1859 - categorical_accuracy: 0.9438
56288/60000 [===========================>..] - ETA: 6s - loss: 0.1858 - categorical_accuracy: 0.9438
56320/60000 [===========================>..] - ETA: 6s - loss: 0.1858 - categorical_accuracy: 0.9439
56384/60000 [===========================>..] - ETA: 6s - loss: 0.1856 - categorical_accuracy: 0.9439
56448/60000 [===========================>..] - ETA: 6s - loss: 0.1856 - categorical_accuracy: 0.9439
56480/60000 [===========================>..] - ETA: 6s - loss: 0.1855 - categorical_accuracy: 0.9439
56512/60000 [===========================>..] - ETA: 5s - loss: 0.1854 - categorical_accuracy: 0.9440
56544/60000 [===========================>..] - ETA: 5s - loss: 0.1854 - categorical_accuracy: 0.9440
56576/60000 [===========================>..] - ETA: 5s - loss: 0.1853 - categorical_accuracy: 0.9440
56608/60000 [===========================>..] - ETA: 5s - loss: 0.1852 - categorical_accuracy: 0.9440
56640/60000 [===========================>..] - ETA: 5s - loss: 0.1851 - categorical_accuracy: 0.9441
56672/60000 [===========================>..] - ETA: 5s - loss: 0.1850 - categorical_accuracy: 0.9441
56704/60000 [===========================>..] - ETA: 5s - loss: 0.1849 - categorical_accuracy: 0.9441
56736/60000 [===========================>..] - ETA: 5s - loss: 0.1849 - categorical_accuracy: 0.9442
56768/60000 [===========================>..] - ETA: 5s - loss: 0.1848 - categorical_accuracy: 0.9442
56800/60000 [===========================>..] - ETA: 5s - loss: 0.1848 - categorical_accuracy: 0.9442
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1847 - categorical_accuracy: 0.9442
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1848 - categorical_accuracy: 0.9442
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1847 - categorical_accuracy: 0.9442
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1846 - categorical_accuracy: 0.9442
56992/60000 [===========================>..] - ETA: 5s - loss: 0.1845 - categorical_accuracy: 0.9442
57024/60000 [===========================>..] - ETA: 5s - loss: 0.1844 - categorical_accuracy: 0.9443
57056/60000 [===========================>..] - ETA: 5s - loss: 0.1844 - categorical_accuracy: 0.9443
57120/60000 [===========================>..] - ETA: 4s - loss: 0.1842 - categorical_accuracy: 0.9443
57152/60000 [===========================>..] - ETA: 4s - loss: 0.1842 - categorical_accuracy: 0.9443
57184/60000 [===========================>..] - ETA: 4s - loss: 0.1842 - categorical_accuracy: 0.9443
57216/60000 [===========================>..] - ETA: 4s - loss: 0.1842 - categorical_accuracy: 0.9443
57248/60000 [===========================>..] - ETA: 4s - loss: 0.1841 - categorical_accuracy: 0.9443
57280/60000 [===========================>..] - ETA: 4s - loss: 0.1840 - categorical_accuracy: 0.9443
57312/60000 [===========================>..] - ETA: 4s - loss: 0.1841 - categorical_accuracy: 0.9443
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1841 - categorical_accuracy: 0.9443
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1840 - categorical_accuracy: 0.9444
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1839 - categorical_accuracy: 0.9444
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1838 - categorical_accuracy: 0.9444
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1838 - categorical_accuracy: 0.9444
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1838 - categorical_accuracy: 0.9444
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1837 - categorical_accuracy: 0.9444
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1837 - categorical_accuracy: 0.9444
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1836 - categorical_accuracy: 0.9444
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1836 - categorical_accuracy: 0.9445
57664/60000 [===========================>..] - ETA: 3s - loss: 0.1835 - categorical_accuracy: 0.9445
57696/60000 [===========================>..] - ETA: 3s - loss: 0.1834 - categorical_accuracy: 0.9445
57728/60000 [===========================>..] - ETA: 3s - loss: 0.1833 - categorical_accuracy: 0.9445
57760/60000 [===========================>..] - ETA: 3s - loss: 0.1833 - categorical_accuracy: 0.9445
57792/60000 [===========================>..] - ETA: 3s - loss: 0.1832 - categorical_accuracy: 0.9446
57824/60000 [===========================>..] - ETA: 3s - loss: 0.1832 - categorical_accuracy: 0.9446
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1832 - categorical_accuracy: 0.9446
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1831 - categorical_accuracy: 0.9446
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1830 - categorical_accuracy: 0.9446
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1830 - categorical_accuracy: 0.9446
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1829 - categorical_accuracy: 0.9446
58016/60000 [============================>.] - ETA: 3s - loss: 0.1828 - categorical_accuracy: 0.9447
58048/60000 [============================>.] - ETA: 3s - loss: 0.1827 - categorical_accuracy: 0.9447
58080/60000 [============================>.] - ETA: 3s - loss: 0.1826 - categorical_accuracy: 0.9447
58112/60000 [============================>.] - ETA: 3s - loss: 0.1826 - categorical_accuracy: 0.9447
58144/60000 [============================>.] - ETA: 3s - loss: 0.1825 - categorical_accuracy: 0.9448
58176/60000 [============================>.] - ETA: 3s - loss: 0.1824 - categorical_accuracy: 0.9448
58240/60000 [============================>.] - ETA: 3s - loss: 0.1823 - categorical_accuracy: 0.9448
58272/60000 [============================>.] - ETA: 2s - loss: 0.1823 - categorical_accuracy: 0.9448
58304/60000 [============================>.] - ETA: 2s - loss: 0.1822 - categorical_accuracy: 0.9449
58336/60000 [============================>.] - ETA: 2s - loss: 0.1821 - categorical_accuracy: 0.9449
58368/60000 [============================>.] - ETA: 2s - loss: 0.1821 - categorical_accuracy: 0.9449
58400/60000 [============================>.] - ETA: 2s - loss: 0.1821 - categorical_accuracy: 0.9449
58432/60000 [============================>.] - ETA: 2s - loss: 0.1820 - categorical_accuracy: 0.9449
58464/60000 [============================>.] - ETA: 2s - loss: 0.1819 - categorical_accuracy: 0.9450
58496/60000 [============================>.] - ETA: 2s - loss: 0.1818 - categorical_accuracy: 0.9450
58528/60000 [============================>.] - ETA: 2s - loss: 0.1818 - categorical_accuracy: 0.9450
58560/60000 [============================>.] - ETA: 2s - loss: 0.1817 - categorical_accuracy: 0.9450
58592/60000 [============================>.] - ETA: 2s - loss: 0.1816 - categorical_accuracy: 0.9450
58624/60000 [============================>.] - ETA: 2s - loss: 0.1816 - categorical_accuracy: 0.9450
58656/60000 [============================>.] - ETA: 2s - loss: 0.1816 - categorical_accuracy: 0.9450
58688/60000 [============================>.] - ETA: 2s - loss: 0.1816 - categorical_accuracy: 0.9450
58720/60000 [============================>.] - ETA: 2s - loss: 0.1817 - categorical_accuracy: 0.9450
58752/60000 [============================>.] - ETA: 2s - loss: 0.1816 - categorical_accuracy: 0.9450
58784/60000 [============================>.] - ETA: 2s - loss: 0.1815 - categorical_accuracy: 0.9450
58816/60000 [============================>.] - ETA: 2s - loss: 0.1815 - categorical_accuracy: 0.9450
58848/60000 [============================>.] - ETA: 1s - loss: 0.1814 - categorical_accuracy: 0.9451
58880/60000 [============================>.] - ETA: 1s - loss: 0.1814 - categorical_accuracy: 0.9451
58912/60000 [============================>.] - ETA: 1s - loss: 0.1814 - categorical_accuracy: 0.9451
58944/60000 [============================>.] - ETA: 1s - loss: 0.1813 - categorical_accuracy: 0.9451
58976/60000 [============================>.] - ETA: 1s - loss: 0.1812 - categorical_accuracy: 0.9451
59008/60000 [============================>.] - ETA: 1s - loss: 0.1812 - categorical_accuracy: 0.9451
59040/60000 [============================>.] - ETA: 1s - loss: 0.1811 - categorical_accuracy: 0.9452
59072/60000 [============================>.] - ETA: 1s - loss: 0.1810 - categorical_accuracy: 0.9452
59104/60000 [============================>.] - ETA: 1s - loss: 0.1809 - categorical_accuracy: 0.9452
59136/60000 [============================>.] - ETA: 1s - loss: 0.1810 - categorical_accuracy: 0.9452
59200/60000 [============================>.] - ETA: 1s - loss: 0.1809 - categorical_accuracy: 0.9453
59232/60000 [============================>.] - ETA: 1s - loss: 0.1808 - categorical_accuracy: 0.9453
59264/60000 [============================>.] - ETA: 1s - loss: 0.1807 - categorical_accuracy: 0.9453
59296/60000 [============================>.] - ETA: 1s - loss: 0.1806 - categorical_accuracy: 0.9453
59328/60000 [============================>.] - ETA: 1s - loss: 0.1805 - categorical_accuracy: 0.9454
59360/60000 [============================>.] - ETA: 1s - loss: 0.1804 - categorical_accuracy: 0.9454
59424/60000 [============================>.] - ETA: 0s - loss: 0.1804 - categorical_accuracy: 0.9454
59456/60000 [============================>.] - ETA: 0s - loss: 0.1804 - categorical_accuracy: 0.9454
59488/60000 [============================>.] - ETA: 0s - loss: 0.1804 - categorical_accuracy: 0.9454
59520/60000 [============================>.] - ETA: 0s - loss: 0.1805 - categorical_accuracy: 0.9454
59552/60000 [============================>.] - ETA: 0s - loss: 0.1805 - categorical_accuracy: 0.9454
59584/60000 [============================>.] - ETA: 0s - loss: 0.1805 - categorical_accuracy: 0.9454
59616/60000 [============================>.] - ETA: 0s - loss: 0.1804 - categorical_accuracy: 0.9454
59648/60000 [============================>.] - ETA: 0s - loss: 0.1804 - categorical_accuracy: 0.9454
59680/60000 [============================>.] - ETA: 0s - loss: 0.1804 - categorical_accuracy: 0.9454
59712/60000 [============================>.] - ETA: 0s - loss: 0.1803 - categorical_accuracy: 0.9455
59744/60000 [============================>.] - ETA: 0s - loss: 0.1802 - categorical_accuracy: 0.9455
59776/60000 [============================>.] - ETA: 0s - loss: 0.1802 - categorical_accuracy: 0.9455
59808/60000 [============================>.] - ETA: 0s - loss: 0.1801 - categorical_accuracy: 0.9455
59840/60000 [============================>.] - ETA: 0s - loss: 0.1800 - categorical_accuracy: 0.9455
59872/60000 [============================>.] - ETA: 0s - loss: 0.1799 - categorical_accuracy: 0.9456
59904/60000 [============================>.] - ETA: 0s - loss: 0.1799 - categorical_accuracy: 0.9456
59936/60000 [============================>.] - ETA: 0s - loss: 0.1798 - categorical_accuracy: 0.9456
59968/60000 [============================>.] - ETA: 0s - loss: 0.1798 - categorical_accuracy: 0.9456
60000/60000 [==============================] - 106s 2ms/step - loss: 0.1797 - categorical_accuracy: 0.9456 - val_loss: 0.0508 - val_categorical_accuracy: 0.9823

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 13s
  192/10000 [..............................] - ETA: 5s 
  384/10000 [>.............................] - ETA: 4s
  544/10000 [>.............................] - ETA: 3s
  704/10000 [=>............................] - ETA: 3s
  864/10000 [=>............................] - ETA: 3s
 1024/10000 [==>...........................] - ETA: 3s
 1184/10000 [==>...........................] - ETA: 3s
 1344/10000 [===>..........................] - ETA: 3s
 1504/10000 [===>..........................] - ETA: 3s
 1664/10000 [===>..........................] - ETA: 3s
 1824/10000 [====>.........................] - ETA: 2s
 1984/10000 [====>.........................] - ETA: 2s
 2144/10000 [=====>........................] - ETA: 2s
 2304/10000 [=====>........................] - ETA: 2s
 2464/10000 [======>.......................] - ETA: 2s
 2624/10000 [======>.......................] - ETA: 2s
 2784/10000 [=======>......................] - ETA: 2s
 2944/10000 [=======>......................] - ETA: 2s
 3136/10000 [========>.....................] - ETA: 2s
 3328/10000 [========>.....................] - ETA: 2s
 3488/10000 [=========>....................] - ETA: 2s
 3648/10000 [=========>....................] - ETA: 2s
 3840/10000 [==========>...................] - ETA: 2s
 4000/10000 [===========>..................] - ETA: 2s
 4192/10000 [===========>..................] - ETA: 1s
 4384/10000 [============>.................] - ETA: 1s
 4544/10000 [============>.................] - ETA: 1s
 4704/10000 [=============>................] - ETA: 1s
 4864/10000 [=============>................] - ETA: 1s
 5024/10000 [==============>...............] - ETA: 1s
 5184/10000 [==============>...............] - ETA: 1s
 5344/10000 [===============>..............] - ETA: 1s
 5504/10000 [===============>..............] - ETA: 1s
 5664/10000 [===============>..............] - ETA: 1s
 5856/10000 [================>.............] - ETA: 1s
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
 7808/10000 [======================>.......] - ETA: 0s
 7968/10000 [======================>.......] - ETA: 0s
 8160/10000 [=======================>......] - ETA: 0s
 8288/10000 [=======================>......] - ETA: 0s
 8448/10000 [========================>.....] - ETA: 0s
 8608/10000 [========================>.....] - ETA: 0s
 8768/10000 [=========================>....] - ETA: 0s
 8928/10000 [=========================>....] - ETA: 0s
 9088/10000 [==========================>...] - ETA: 0s
 9248/10000 [==========================>...] - ETA: 0s
 9408/10000 [===========================>..] - ETA: 0s
 9568/10000 [===========================>..] - ETA: 0s
 9696/10000 [============================>.] - ETA: 0s
 9856/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 3s 339us/step
[[4.1998792e-08 1.5050729e-08 8.0720283e-06 ... 9.9998605e-01
  1.7412424e-08 3.0439803e-06]
 [2.5612786e-05 7.9630998e-05 9.9986112e-01 ... 3.2562507e-08
  1.0849430e-05 1.0250679e-09]
 [5.8826885e-08 9.9983549e-01 2.5024743e-05 ... 2.3689405e-05
  7.8563380e-06 1.7761445e-06]
 ...
 [1.2181848e-09 1.9352840e-06 1.1576524e-08 ... 1.4455748e-06
  2.7270646e-06 8.5407664e-05]
 [6.0845713e-07 5.7714578e-07 2.8738286e-08 ... 2.5220555e-07
  1.4642676e-03 4.3074863e-07]
 [1.0900052e-05 1.4085806e-07 5.6744229e-06 ... 1.2699126e-08
  4.6293858e-06 8.8104358e-08]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.0508459737409139, 'accuracy_test:': 0.9822999835014343}

  ('#### Save   #######################################################',) 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/charcnn/result'}

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   2e2c008..e298ae1  master     -> origin/master
Updating 2e2c008..e298ae1
Fast-forward
 error_list/20200515/list_log_testall_20200515.md | 433 +++++++++++++++++++++++
 1 file changed, 433 insertions(+)
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
[master 053f605] ml_store
 1 file changed, 1963 insertions(+)
To github.com:arita37/mlmodels_store.git
   e298ae1..053f605  master -> master





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
{'loss': 0.4473135210573673, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-15 16:29:46.154236: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
[master 37f0980] ml_store
 1 file changed, 233 insertions(+)
To github.com:arita37/mlmodels_store.git
   053f605..37f0980  master -> master





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
[master 1cfae01] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   37f0980..1cfae01  master -> master





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
 40%|████      | 2/5 [00:18<00:28,  9.49s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.7990714277434179, 'learning_rate': 0.11381922471379974, 'min_data_in_leaf': 12, 'num_leaves': 46} and reward: 0.3906
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9\x91\xfe>*n\xa6X\r\x00\x00\x00learning_rateq\x02G?\xbd#A\xb7\xcdE\xb4X\x10\x00\x00\x00min_data_in_leafq\x03K\x0cX\n\x00\x00\x00num_leavesq\x04K.u.' and reward: 0.3906
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9\x91\xfe>*n\xa6X\r\x00\x00\x00learning_rateq\x02G?\xbd#A\xb7\xcdE\xb4X\x10\x00\x00\x00min_data_in_leafq\x03K\x0cX\n\x00\x00\x00num_leavesq\x04K.u.' and reward: 0.3906
 60%|██████    | 3/5 [00:41<00:26, 13.27s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.9795003957935997, 'learning_rate': 0.03957279632956312, 'min_data_in_leaf': 17, 'num_leaves': 55} and reward: 0.3904
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xefX\x116\xcbH:X\r\x00\x00\x00learning_rateq\x02G?\xa4B\xe2\xb4\x17\xee\xa0X\x10\x00\x00\x00min_data_in_leafq\x03K\x11X\n\x00\x00\x00num_leavesq\x04K7u.' and reward: 0.3904
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xefX\x116\xcbH:X\r\x00\x00\x00learning_rateq\x02G?\xa4B\xe2\xb4\x17\xee\xa0X\x10\x00\x00\x00min_data_in_leafq\x03K\x11X\n\x00\x00\x00num_leavesq\x04K7u.' and reward: 0.3904
 80%|████████  | 4/5 [01:07<00:17, 17.24s/it] 80%|████████  | 4/5 [01:07<00:16, 16.90s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.9232812260515846, 'learning_rate': 0.010600185077305318, 'min_data_in_leaf': 7, 'num_leaves': 61} and reward: 0.3896
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\x8b\x85\x11\xdc\xdf\xf6X\r\x00\x00\x00learning_rateq\x02G?\x85\xb5\x8c\xc1\xe8\x99CX\x10\x00\x00\x00min_data_in_leafq\x03K\x07X\n\x00\x00\x00num_leavesq\x04K=u.' and reward: 0.3896
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\x8b\x85\x11\xdc\xdf\xf6X\r\x00\x00\x00learning_rateq\x02G?\x85\xb5\x8c\xc1\xe8\x99CX\x10\x00\x00\x00min_data_in_leafq\x03K\x07X\n\x00\x00\x00num_leavesq\x04K=u.' and reward: 0.3896
Time for Gradient Boosting hyperparameter optimization: 96.20826458930969
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
Saving dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3894
 40%|████      | 2/5 [00:45<01:08, 22.81s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.21096863413133093, 'embedding_size_factor': 0.5407140973693028, 'layers.choice': 0, 'learning_rate': 0.00021857882105029577, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 1.3981541436221955e-06} and reward: 0.3702
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xcb\x01\x05,\t\xb5\xa4X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe1M\x87\xa6\x95\xfe|X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?,\xa6I\xc6\xa9&8X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb7u\x06\xbd\x03\x930u.' and reward: 0.3702
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xcb\x01\x05,\t\xb5\xa4X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe1M\x87\xa6\x95\xfe|X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?,\xa6I\xc6\xa9&8X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb7u\x06\xbd\x03\x930u.' and reward: 0.3702
 60%|██████    | 3/5 [01:31<00:59, 29.64s/it] 60%|██████    | 3/5 [01:31<01:00, 30.39s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.2381631579254962, 'embedding_size_factor': 1.1124654092735844, 'layers.choice': 2, 'learning_rate': 0.00023006440736519552, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 9.473544213469078e-11} and reward: 0.3596
Finished Task with config: b"\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xce|!_3w\xaaX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf1\xcc\xa8\x87l.IX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?.'\xae6\x12#yX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\xda\nj\x01\xf1,\xd5u." and reward: 0.3596
Finished Task with config: b"\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xce|!_3w\xaaX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf1\xcc\xa8\x87l.IX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?.'\xae6\x12#yX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\xda\nj\x01\xf1,\xd5u." and reward: 0.3596
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 138.478586435318
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
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.79s of the -118.62s of remaining time.
Ensemble size: 33
Ensemble weights: 
[0.36363636 0.03030303 0.03030303 0.09090909 0.3030303  0.09090909
 0.09090909]
	0.4006	 = Validation accuracy score
	1.45s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 240.11s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7ff2d63e3668>

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
From github.com:arita37/mlmodels_store
   1cfae01..c981b74  master     -> origin/master
Updating 1cfae01..c981b74
Fast-forward
 error_list/20200515/list_log_benchmark_20200515.md |  434 ++++----
 error_list/20200515/list_log_import_20200515.md    |    2 +-
 error_list/20200515/list_log_json_20200515.md      | 1146 ++++++++++----------
 error_list/20200515/list_log_testall_20200515.md   |  174 +++
 4 files changed, 970 insertions(+), 786 deletions(-)
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
[master 81f4db8] ml_store
 1 file changed, 217 insertions(+)
To github.com:arita37/mlmodels_store.git
   c981b74..81f4db8  master -> master





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
[master c9144f5] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   81f4db8..c9144f5  master -> master





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
