
  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_all 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/76b7a81be9b27c2e92c4951280c0a8da664b997c', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '76b7a81be9b27c2e92c4951280c0a8da664b997c', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/76b7a81be9b27c2e92c4951280c0a8da664b997c

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/76b7a81be9b27c2e92c4951280c0a8da664b997c

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
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
Already up to date.
[master 12c8998] ml_store
 2 files changed, 63 insertions(+), 10429 deletions(-)
 rewrite log_testall/log_testall.py (99%)
To github.com:arita37/mlmodels_store.git
   ef82d22..12c8998  master -> master





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
[master e3db9c9] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   12c8998..e3db9c9  master -> master





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
Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
Already up to date.
[master ad9d8e2] ml_store
 1 file changed, 49 insertions(+)
To github.com:arita37/mlmodels_store.git
   e3db9c9..ad9d8e2  master -> master





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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         28          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-17 16:12:08.987646: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-17 16:12:09.001123: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-05-17 16:12:09.001404: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ec9d95a9b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-17 16:12:09.001423: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
100/500 [=====>........................] - ETA: 2s - loss: 0.2475 - binary_crossentropy: 0.6862500/500 [==============================] - 1s 1ms/sample - loss: 0.2505 - binary_crossentropy: 0.7704 - val_loss: 0.2515 - val_binary_crossentropy: 0.7477

  #### metrics   #################################################### 
{'MSE': 0.25076894297596747}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         28          sparse_feature_1[0][0]           
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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_max[0][0]               
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
Total params: 463
Trainable params: 463
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2621 - binary_crossentropy: 0.7216500/500 [==============================] - 1s 2ms/sample - loss: 0.2753 - binary_crossentropy: 0.7494 - val_loss: 0.2738 - val_binary_crossentropy: 0.7456

  #### metrics   #################################################### 
{'MSE': 0.27405183451639803}

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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_max[0][0]               
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         28          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
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
Total params: 647
Trainable params: 647
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2450 - binary_crossentropy: 0.6793500/500 [==============================] - 1s 2ms/sample - loss: 0.2489 - binary_crossentropy: 0.7154 - val_loss: 0.2558 - val_binary_crossentropy: 0.8355

  #### metrics   #################################################### 
{'MSE': 0.2520014295268053}

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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         28          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
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
Total params: 647
Trainable params: 647
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
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         4           sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_2[0][0]           
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
Total params: 458
Trainable params: 458
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.3219 - binary_crossentropy: 0.8696500/500 [==============================] - 1s 3ms/sample - loss: 0.3053 - binary_crossentropy: 0.8313 - val_loss: 0.2971 - val_binary_crossentropy: 0.8071

  #### metrics   #################################################### 
{'MSE': 0.2988980366812539}

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
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         4           sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_2[0][0]           
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
Total params: 458
Trainable params: 458
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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_max[0][0]               
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
Total params: 183
Trainable params: 183
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.5000 - binary_crossentropy: 7.7125500/500 [==============================] - 2s 3ms/sample - loss: 0.5120 - binary_crossentropy: 7.8976 - val_loss: 0.5020 - val_binary_crossentropy: 7.7433

  #### metrics   #################################################### 
{'MSE': 0.507}

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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_max[0][0]               
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-17 16:13:31.540968: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 16:13:31.543099: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 16:13:31.550822: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-17 16:13:31.561608: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-17 16:13:31.563454: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-17 16:13:31.565157: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 16:13:31.566775: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2501 - val_binary_crossentropy: 0.6933
2020-05-17 16:13:32.888197: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 16:13:32.893269: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 16:13:32.898133: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-17 16:13:32.910744: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-17 16:13:32.912322: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-17 16:13:32.913961: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 16:13:32.915349: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2500289846260264}

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
2020-05-17 16:13:57.172602: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 16:13:57.174038: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 16:13:57.177915: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-17 16:13:57.184529: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-17 16:13:57.186019: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-17 16:13:57.187060: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 16:13:57.188123: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
2020-05-17 16:13:58.782633: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 16:13:58.783903: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 16:13:58.786730: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-17 16:13:58.792350: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-17 16:13:58.793302: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-17 16:13:58.794174: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 16:13:58.795132: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24851024375436026}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-17 16:14:34.533345: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 16:14:34.538289: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 16:14:34.554852: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-17 16:14:34.593067: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-17 16:14:34.597738: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-17 16:14:34.601817: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 16:14:34.605908: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.4459 - binary_crossentropy: 1.1019 - val_loss: 0.2587 - val_binary_crossentropy: 0.7109
2020-05-17 16:14:37.066678: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 16:14:37.071470: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 16:14:37.084219: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-17 16:14:37.110695: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-17 16:14:37.115824: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-17 16:14:37.119832: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 16:14:37.124139: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.22762075377683408}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
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
Total params: 700
Trainable params: 700
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.3099 - binary_crossentropy: 1.9979500/500 [==============================] - 4s 9ms/sample - loss: 0.2731 - binary_crossentropy: 1.4408 - val_loss: 0.2748 - val_binary_crossentropy: 1.3659

  #### metrics   #################################################### 
{'MSE': 0.27351417453586013}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 2)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 2)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 2)         12          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         10          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         6           sparse_feature_4[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_4[0][0]           
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
Total params: 230
Trainable params: 230
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2710 - binary_crossentropy: 0.7410500/500 [==============================] - 5s 9ms/sample - loss: 0.2659 - binary_crossentropy: 0.7304 - val_loss: 0.2634 - val_binary_crossentropy: 0.7255

  #### metrics   #################################################### 
{'MSE': 0.26140341092705577}

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
sparse_seq_emb_sequence_sum (Em (None, 9, 2)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 2)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 2)         12          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         10          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         6           sparse_feature_4[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_4[0][0]           
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
Total params: 230
Trainable params: 230
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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_max[0][0]               
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
Total params: 1,899
Trainable params: 1,899
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2654 - binary_crossentropy: 0.8545500/500 [==============================] - 5s 9ms/sample - loss: 0.2590 - binary_crossentropy: 0.8424 - val_loss: 0.2529 - val_binary_crossentropy: 0.8022

  #### metrics   #################################################### 
{'MSE': 0.25612139061789607}

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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_max[0][0]               
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
Total params: 1,899
Trainable params: 1,899
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
regionsequence_mean (InputLayer [(None, 1)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 3)]          0                                            
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
region_10sparse_seq_emb_regions (None, 7, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 1, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 7, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 7, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 7, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 7, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 7, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 7, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 7, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         9           regionsequence_max[0][0]         
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
100/500 [=====>........................] - ETA: 10s - loss: 0.2449 - binary_crossentropy: 0.6827500/500 [==============================] - 6s 13ms/sample - loss: 0.2547 - binary_crossentropy: 0.7029 - val_loss: 0.2648 - val_binary_crossentropy: 0.7229

  #### metrics   #################################################### 
{'MSE': 0.25945189018016956}

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
regionsequence_mean (InputLayer [(None, 1)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 3)]          0                                            
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
region_10sparse_seq_emb_regions (None, 7, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 1, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 7, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 7, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 7, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 7, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 7, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 7, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 7, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         9           regionsequence_max[0][0]         
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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_max[0][0]               
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
Total params: 1,377
Trainable params: 1,377
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2667 - binary_crossentropy: 0.8598500/500 [==============================] - 6s 12ms/sample - loss: 0.2555 - binary_crossentropy: 0.7309 - val_loss: 0.2531 - val_binary_crossentropy: 0.6994

  #### metrics   #################################################### 
{'MSE': 0.25196645457971534}

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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_max[0][0]               
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_emb_sparse_feature_1_spa (None, 1, 4)         28          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 7, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 1, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         28          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 7, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         28          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         28          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 1, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 7, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 7, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 1, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 1, 4)         8           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           hash_11[0][0]                    
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
Total params: 3,103
Trainable params: 3,023
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.3956 - binary_crossentropy: 4.5082500/500 [==============================] - 7s 14ms/sample - loss: 0.3898 - binary_crossentropy: 4.3656 - val_loss: 0.3819 - val_binary_crossentropy: 4.2449

  #### metrics   #################################################### 
{'MSE': 0.3827995949400753}

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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_emb_sparse_feature_1_spa (None, 1, 4)         28          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 7, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 1, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         28          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 7, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         28          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         28          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 1, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 7, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 7, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 1, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 1, 4)         8           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           hash_11[0][0]                    
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
Total params: 3,103
Trainable params: 3,023
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
Warning: Permanently added the RSA host key for IP address '140.82.113.3' to the list of known hosts.
From github.com:arita37/mlmodels_store
   ad9d8e2..168d6d8  master     -> origin/master
Updating ad9d8e2..168d6d8
Fast-forward
 error_list/20200517/list_log_benchmark_20200517.md |  182 +-
 error_list/20200517/list_log_jupyter_20200517.md   | 1749 ++++++++++----------
 .../20200517/list_log_pullrequest_20200517.md      |    2 +-
 error_list/20200517/list_log_test_cli_20200517.md  |  138 +-
 4 files changed, 1031 insertions(+), 1040 deletions(-)
[master 3e85ee2] ml_store
 1 file changed, 4956 insertions(+)
To github.com:arita37/mlmodels_store.git
   168d6d8..3e85ee2  master -> master





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
[master b26023d] ml_store
 1 file changed, 51 insertions(+)
To github.com:arita37/mlmodels_store.git
   3e85ee2..b26023d  master -> master





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
[master 1bcdae9] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   b26023d..1bcdae9  master -> master





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
[master 1193689] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   1bcdae9..1193689  master -> master





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

2020-05-17 16:24:05.218740: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-17 16:24:05.223837: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-05-17 16:24:05.224306: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b07d168cf0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-17 16:24:05.224616: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 7s - loss: 1.3810
256/354 [====================>.........] - ETA: 3s - loss: 1.1821
354/354 [==============================] - 14s 38ms/step - loss: 1.5519 - val_loss: 2.3282

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
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
Already up to date.
[master dacd521] ml_store
 1 file changed, 151 insertions(+)
To github.com:arita37/mlmodels_store.git
   1193689..dacd521  master -> master





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
[master 0126de6] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   dacd521..0126de6  master -> master





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
[master d612805] ml_store
 1 file changed, 45 insertions(+)
To github.com:arita37/mlmodels_store.git
   0126de6..d612805  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2883584/17464789 [===>..........................] - ETA: 0s
 9388032/17464789 [===============>..............] - ETA: 0s
16613376/17464789 [===========================>..] - ETA: 0s
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
2020-05-17 16:25:07.838550: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-17 16:25:07.842906: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-05-17 16:25:07.843049: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f88f1fc990 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-17 16:25:07.843063: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.6973 - accuracy: 0.4980
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6896 - accuracy: 0.4985 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6206 - accuracy: 0.5030
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6820 - accuracy: 0.4990
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7280 - accuracy: 0.4960
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6768 - accuracy: 0.4993
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6666 - accuracy: 0.5000
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6340 - accuracy: 0.5021
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6888 - accuracy: 0.4986
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6866 - accuracy: 0.4987
11000/25000 [============>.................] - ETA: 4s - loss: 7.6792 - accuracy: 0.4992
12000/25000 [=============>................] - ETA: 3s - loss: 7.6628 - accuracy: 0.5002
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6631 - accuracy: 0.5002
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6644 - accuracy: 0.5001
15000/25000 [=================>............] - ETA: 2s - loss: 7.6584 - accuracy: 0.5005
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6800 - accuracy: 0.4991
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6991 - accuracy: 0.4979
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6794 - accuracy: 0.4992
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6586 - accuracy: 0.5005
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6521 - accuracy: 0.5009
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6622 - accuracy: 0.5003
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6576 - accuracy: 0.5006
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6700 - accuracy: 0.4998
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6807 - accuracy: 0.4991
25000/25000 [==============================] - 9s 353us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7fdedd3e0d68>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7fdeeab41c18> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.9886 - accuracy: 0.4790
 2000/25000 [=>............................] - ETA: 9s - loss: 7.5823 - accuracy: 0.5055 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5746 - accuracy: 0.5060
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6360 - accuracy: 0.5020
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7402 - accuracy: 0.4952
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7101 - accuracy: 0.4972
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6929 - accuracy: 0.4983
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6321 - accuracy: 0.5023
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6598 - accuracy: 0.5004
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6712 - accuracy: 0.4997
11000/25000 [============>.................] - ETA: 4s - loss: 7.6945 - accuracy: 0.4982
12000/25000 [=============>................] - ETA: 3s - loss: 7.7382 - accuracy: 0.4953
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7362 - accuracy: 0.4955
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7444 - accuracy: 0.4949
15000/25000 [=================>............] - ETA: 2s - loss: 7.7177 - accuracy: 0.4967
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6762 - accuracy: 0.4994
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6657 - accuracy: 0.5001
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6547 - accuracy: 0.5008
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6618 - accuracy: 0.5003
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6620 - accuracy: 0.5003
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6469 - accuracy: 0.5013
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6687 - accuracy: 0.4999
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6615 - accuracy: 0.5003
25000/25000 [==============================] - 9s 354us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 12s - loss: 8.1420 - accuracy: 0.4690
 2000/25000 [=>............................] - ETA: 9s - loss: 7.9043 - accuracy: 0.4845 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.8915 - accuracy: 0.4853
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.8583 - accuracy: 0.4875
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.8230 - accuracy: 0.4898
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7612 - accuracy: 0.4938
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7367 - accuracy: 0.4954
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7490 - accuracy: 0.4946
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7620 - accuracy: 0.4938
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7249 - accuracy: 0.4962
11000/25000 [============>.................] - ETA: 4s - loss: 7.7098 - accuracy: 0.4972
12000/25000 [=============>................] - ETA: 3s - loss: 7.7331 - accuracy: 0.4957
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7480 - accuracy: 0.4947
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7411 - accuracy: 0.4951
15000/25000 [=================>............] - ETA: 2s - loss: 7.7372 - accuracy: 0.4954
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7327 - accuracy: 0.4957
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7379 - accuracy: 0.4954
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7365 - accuracy: 0.4954
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7062 - accuracy: 0.4974
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7218 - accuracy: 0.4964
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6980 - accuracy: 0.4980
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6729 - accuracy: 0.4996
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6693 - accuracy: 0.4998
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 9s 350us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
   d612805..3b0143b  master     -> origin/master
Updating d612805..3b0143b
Fast-forward
 error_list/20200517/list_log_benchmark_20200517.md | 182 +++++-----
 .../20200517/list_log_pullrequest_20200517.md      |   2 +-
 error_list/20200517/list_log_testall_20200517.md   | 386 ++++++++++++---------
 3 files changed, 310 insertions(+), 260 deletions(-)
[master cbcb40b] ml_store
 1 file changed, 325 insertions(+)
To github.com:arita37/mlmodels_store.git
   3b0143b..cbcb40b  master -> master





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

13/13 [==============================] - 2s 134ms/step - loss: nan
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
[master bb1eb78] ml_store
 1 file changed, 126 insertions(+)
To github.com:arita37/mlmodels_store.git
   cbcb40b..bb1eb78  master -> master





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

    8192/11490434 [..............................] - ETA: 6s
 4243456/11490434 [==========>...................] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:27 - loss: 2.3101 - categorical_accuracy: 0.1250
   64/60000 [..............................] - ETA: 4:36 - loss: 2.2732 - categorical_accuracy: 0.1719
   96/60000 [..............................] - ETA: 3:37 - loss: 2.2585 - categorical_accuracy: 0.1458
  128/60000 [..............................] - ETA: 3:09 - loss: 2.2667 - categorical_accuracy: 0.1406
  160/60000 [..............................] - ETA: 2:53 - loss: 2.2519 - categorical_accuracy: 0.1625
  192/60000 [..............................] - ETA: 2:40 - loss: 2.2378 - categorical_accuracy: 0.1771
  224/60000 [..............................] - ETA: 2:32 - loss: 2.2016 - categorical_accuracy: 0.2188
  256/60000 [..............................] - ETA: 2:25 - loss: 2.1704 - categorical_accuracy: 0.2383
  288/60000 [..............................] - ETA: 2:20 - loss: 2.1490 - categorical_accuracy: 0.2535
  320/60000 [..............................] - ETA: 2:15 - loss: 2.1128 - categorical_accuracy: 0.2750
  352/60000 [..............................] - ETA: 2:12 - loss: 2.0726 - categorical_accuracy: 0.2955
  384/60000 [..............................] - ETA: 2:09 - loss: 2.0424 - categorical_accuracy: 0.3073
  416/60000 [..............................] - ETA: 2:08 - loss: 1.9948 - categorical_accuracy: 0.3293
  448/60000 [..............................] - ETA: 2:05 - loss: 2.0017 - categorical_accuracy: 0.3326
  480/60000 [..............................] - ETA: 2:03 - loss: 1.9617 - categorical_accuracy: 0.3500
  512/60000 [..............................] - ETA: 2:02 - loss: 1.9491 - categorical_accuracy: 0.3516
  544/60000 [..............................] - ETA: 2:00 - loss: 1.9114 - categorical_accuracy: 0.3640
  576/60000 [..............................] - ETA: 1:59 - loss: 1.8816 - categorical_accuracy: 0.3767
  608/60000 [..............................] - ETA: 1:58 - loss: 1.8678 - categorical_accuracy: 0.3832
  640/60000 [..............................] - ETA: 1:57 - loss: 1.8366 - categorical_accuracy: 0.3984
  672/60000 [..............................] - ETA: 1:56 - loss: 1.8053 - categorical_accuracy: 0.4107
  704/60000 [..............................] - ETA: 1:55 - loss: 1.7829 - categorical_accuracy: 0.4162
  736/60000 [..............................] - ETA: 1:55 - loss: 1.7597 - categorical_accuracy: 0.4212
  768/60000 [..............................] - ETA: 1:54 - loss: 1.7483 - categorical_accuracy: 0.4180
  800/60000 [..............................] - ETA: 1:54 - loss: 1.7385 - categorical_accuracy: 0.4175
  832/60000 [..............................] - ETA: 1:54 - loss: 1.7249 - categorical_accuracy: 0.4231
  864/60000 [..............................] - ETA: 1:53 - loss: 1.6997 - categorical_accuracy: 0.4317
  896/60000 [..............................] - ETA: 1:53 - loss: 1.6746 - categorical_accuracy: 0.4386
  928/60000 [..............................] - ETA: 1:52 - loss: 1.6534 - categorical_accuracy: 0.4472
  960/60000 [..............................] - ETA: 1:51 - loss: 1.6189 - categorical_accuracy: 0.4604
  992/60000 [..............................] - ETA: 1:51 - loss: 1.5940 - categorical_accuracy: 0.4657
 1024/60000 [..............................] - ETA: 1:50 - loss: 1.5656 - categorical_accuracy: 0.4736
 1056/60000 [..............................] - ETA: 1:50 - loss: 1.5403 - categorical_accuracy: 0.4811
 1088/60000 [..............................] - ETA: 1:49 - loss: 1.5305 - categorical_accuracy: 0.4881
 1120/60000 [..............................] - ETA: 1:49 - loss: 1.5088 - categorical_accuracy: 0.4973
 1152/60000 [..............................] - ETA: 1:49 - loss: 1.4887 - categorical_accuracy: 0.5026
 1184/60000 [..............................] - ETA: 1:48 - loss: 1.4811 - categorical_accuracy: 0.5034
 1216/60000 [..............................] - ETA: 1:48 - loss: 1.4660 - categorical_accuracy: 0.5107
 1248/60000 [..............................] - ETA: 1:48 - loss: 1.4431 - categorical_accuracy: 0.5200
 1280/60000 [..............................] - ETA: 1:47 - loss: 1.4235 - categorical_accuracy: 0.5273
 1312/60000 [..............................] - ETA: 1:47 - loss: 1.4024 - categorical_accuracy: 0.5320
 1376/60000 [..............................] - ETA: 1:46 - loss: 1.3771 - categorical_accuracy: 0.5400
 1408/60000 [..............................] - ETA: 1:46 - loss: 1.3623 - categorical_accuracy: 0.5440
 1440/60000 [..............................] - ETA: 1:46 - loss: 1.3493 - categorical_accuracy: 0.5479
 1472/60000 [..............................] - ETA: 1:45 - loss: 1.3368 - categorical_accuracy: 0.5516
 1504/60000 [..............................] - ETA: 1:45 - loss: 1.3195 - categorical_accuracy: 0.5585
 1536/60000 [..............................] - ETA: 1:45 - loss: 1.2997 - categorical_accuracy: 0.5645
 1568/60000 [..............................] - ETA: 1:45 - loss: 1.2942 - categorical_accuracy: 0.5670
 1600/60000 [..............................] - ETA: 1:44 - loss: 1.2904 - categorical_accuracy: 0.5688
 1632/60000 [..............................] - ETA: 1:44 - loss: 1.2786 - categorical_accuracy: 0.5735
 1664/60000 [..............................] - ETA: 1:44 - loss: 1.2627 - categorical_accuracy: 0.5781
 1696/60000 [..............................] - ETA: 1:44 - loss: 1.2487 - categorical_accuracy: 0.5820
 1728/60000 [..............................] - ETA: 1:43 - loss: 1.2377 - categorical_accuracy: 0.5851
 1760/60000 [..............................] - ETA: 1:43 - loss: 1.2200 - categorical_accuracy: 0.5920
 1792/60000 [..............................] - ETA: 1:43 - loss: 1.2050 - categorical_accuracy: 0.5971
 1824/60000 [..............................] - ETA: 1:43 - loss: 1.1938 - categorical_accuracy: 0.6014
 1856/60000 [..............................] - ETA: 1:42 - loss: 1.1845 - categorical_accuracy: 0.6056
 1888/60000 [..............................] - ETA: 1:42 - loss: 1.1751 - categorical_accuracy: 0.6070
 1920/60000 [..............................] - ETA: 1:42 - loss: 1.1630 - categorical_accuracy: 0.6109
 1952/60000 [..............................] - ETA: 1:42 - loss: 1.1522 - categorical_accuracy: 0.6148
 1984/60000 [..............................] - ETA: 1:42 - loss: 1.1370 - categorical_accuracy: 0.6205
 2016/60000 [>.............................] - ETA: 1:41 - loss: 1.1310 - categorical_accuracy: 0.6220
 2048/60000 [>.............................] - ETA: 1:41 - loss: 1.1261 - categorical_accuracy: 0.6255
 2080/60000 [>.............................] - ETA: 1:41 - loss: 1.1179 - categorical_accuracy: 0.6293
 2112/60000 [>.............................] - ETA: 1:41 - loss: 1.1112 - categorical_accuracy: 0.6307
 2144/60000 [>.............................] - ETA: 1:41 - loss: 1.1038 - categorical_accuracy: 0.6334
 2176/60000 [>.............................] - ETA: 1:40 - loss: 1.0981 - categorical_accuracy: 0.6356
 2208/60000 [>.............................] - ETA: 1:40 - loss: 1.0899 - categorical_accuracy: 0.6381
 2240/60000 [>.............................] - ETA: 1:40 - loss: 1.0808 - categorical_accuracy: 0.6411
 2272/60000 [>.............................] - ETA: 1:40 - loss: 1.0728 - categorical_accuracy: 0.6439
 2304/60000 [>.............................] - ETA: 1:40 - loss: 1.0656 - categorical_accuracy: 0.6458
 2336/60000 [>.............................] - ETA: 1:40 - loss: 1.0619 - categorical_accuracy: 0.6477
 2368/60000 [>.............................] - ETA: 1:40 - loss: 1.0550 - categorical_accuracy: 0.6495
 2400/60000 [>.............................] - ETA: 1:39 - loss: 1.0473 - categorical_accuracy: 0.6517
 2432/60000 [>.............................] - ETA: 1:39 - loss: 1.0395 - categorical_accuracy: 0.6546
 2464/60000 [>.............................] - ETA: 1:39 - loss: 1.0346 - categorical_accuracy: 0.6575
 2496/60000 [>.............................] - ETA: 1:39 - loss: 1.0270 - categorical_accuracy: 0.6595
 2528/60000 [>.............................] - ETA: 1:39 - loss: 1.0193 - categorical_accuracy: 0.6626
 2560/60000 [>.............................] - ETA: 1:39 - loss: 1.0136 - categorical_accuracy: 0.6652
 2592/60000 [>.............................] - ETA: 1:39 - loss: 1.0054 - categorical_accuracy: 0.6682
 2624/60000 [>.............................] - ETA: 1:39 - loss: 0.9993 - categorical_accuracy: 0.6707
 2656/60000 [>.............................] - ETA: 1:38 - loss: 0.9926 - categorical_accuracy: 0.6724
 2688/60000 [>.............................] - ETA: 1:38 - loss: 0.9876 - categorical_accuracy: 0.6745
 2720/60000 [>.............................] - ETA: 1:38 - loss: 0.9851 - categorical_accuracy: 0.6761
 2752/60000 [>.............................] - ETA: 1:38 - loss: 0.9802 - categorical_accuracy: 0.6788
 2784/60000 [>.............................] - ETA: 1:38 - loss: 0.9740 - categorical_accuracy: 0.6810
 2816/60000 [>.............................] - ETA: 1:38 - loss: 0.9682 - categorical_accuracy: 0.6832
 2848/60000 [>.............................] - ETA: 1:38 - loss: 0.9616 - categorical_accuracy: 0.6850
 2880/60000 [>.............................] - ETA: 1:38 - loss: 0.9564 - categorical_accuracy: 0.6865
 2912/60000 [>.............................] - ETA: 1:38 - loss: 0.9494 - categorical_accuracy: 0.6889
 2944/60000 [>.............................] - ETA: 1:38 - loss: 0.9439 - categorical_accuracy: 0.6909
 2976/60000 [>.............................] - ETA: 1:37 - loss: 0.9407 - categorical_accuracy: 0.6925
 3008/60000 [>.............................] - ETA: 1:37 - loss: 0.9356 - categorical_accuracy: 0.6945
 3040/60000 [>.............................] - ETA: 1:37 - loss: 0.9343 - categorical_accuracy: 0.6957
 3104/60000 [>.............................] - ETA: 1:37 - loss: 0.9265 - categorical_accuracy: 0.6994
 3136/60000 [>.............................] - ETA: 1:37 - loss: 0.9199 - categorical_accuracy: 0.7015
 3168/60000 [>.............................] - ETA: 1:37 - loss: 0.9175 - categorical_accuracy: 0.7020
 3200/60000 [>.............................] - ETA: 1:37 - loss: 0.9119 - categorical_accuracy: 0.7041
 3232/60000 [>.............................] - ETA: 1:37 - loss: 0.9082 - categorical_accuracy: 0.7051
 3264/60000 [>.............................] - ETA: 1:37 - loss: 0.9020 - categorical_accuracy: 0.7068
 3296/60000 [>.............................] - ETA: 1:36 - loss: 0.8986 - categorical_accuracy: 0.7078
 3328/60000 [>.............................] - ETA: 1:36 - loss: 0.8927 - categorical_accuracy: 0.7103
 3360/60000 [>.............................] - ETA: 1:36 - loss: 0.8879 - categorical_accuracy: 0.7122
 3392/60000 [>.............................] - ETA: 1:36 - loss: 0.8845 - categorical_accuracy: 0.7129
 3424/60000 [>.............................] - ETA: 1:36 - loss: 0.8795 - categorical_accuracy: 0.7147
 3456/60000 [>.............................] - ETA: 1:36 - loss: 0.8749 - categorical_accuracy: 0.7164
 3488/60000 [>.............................] - ETA: 1:36 - loss: 0.8712 - categorical_accuracy: 0.7170
 3520/60000 [>.............................] - ETA: 1:36 - loss: 0.8678 - categorical_accuracy: 0.7179
 3552/60000 [>.............................] - ETA: 1:36 - loss: 0.8617 - categorical_accuracy: 0.7202
 3584/60000 [>.............................] - ETA: 1:36 - loss: 0.8549 - categorical_accuracy: 0.7227
 3616/60000 [>.............................] - ETA: 1:36 - loss: 0.8555 - categorical_accuracy: 0.7235
 3648/60000 [>.............................] - ETA: 1:36 - loss: 0.8507 - categorical_accuracy: 0.7251
 3712/60000 [>.............................] - ETA: 1:35 - loss: 0.8436 - categorical_accuracy: 0.7268
 3744/60000 [>.............................] - ETA: 1:35 - loss: 0.8433 - categorical_accuracy: 0.7273
 3776/60000 [>.............................] - ETA: 1:35 - loss: 0.8394 - categorical_accuracy: 0.7285
 3808/60000 [>.............................] - ETA: 1:35 - loss: 0.8360 - categorical_accuracy: 0.7300
 3840/60000 [>.............................] - ETA: 1:35 - loss: 0.8337 - categorical_accuracy: 0.7305
 3872/60000 [>.............................] - ETA: 1:35 - loss: 0.8303 - categorical_accuracy: 0.7317
 3904/60000 [>.............................] - ETA: 1:35 - loss: 0.8261 - categorical_accuracy: 0.7331
 3936/60000 [>.............................] - ETA: 1:35 - loss: 0.8201 - categorical_accuracy: 0.7353
 3968/60000 [>.............................] - ETA: 1:35 - loss: 0.8152 - categorical_accuracy: 0.7371
 4000/60000 [=>............................] - ETA: 1:35 - loss: 0.8115 - categorical_accuracy: 0.7385
 4032/60000 [=>............................] - ETA: 1:34 - loss: 0.8088 - categorical_accuracy: 0.7388
 4064/60000 [=>............................] - ETA: 1:34 - loss: 0.8040 - categorical_accuracy: 0.7406
 4096/60000 [=>............................] - ETA: 1:34 - loss: 0.8016 - categorical_accuracy: 0.7415
 4128/60000 [=>............................] - ETA: 1:34 - loss: 0.7984 - categorical_accuracy: 0.7425
 4160/60000 [=>............................] - ETA: 1:34 - loss: 0.7963 - categorical_accuracy: 0.7435
 4192/60000 [=>............................] - ETA: 1:34 - loss: 0.7924 - categorical_accuracy: 0.7448
 4224/60000 [=>............................] - ETA: 1:34 - loss: 0.7912 - categorical_accuracy: 0.7448
 4256/60000 [=>............................] - ETA: 1:34 - loss: 0.7876 - categorical_accuracy: 0.7460
 4288/60000 [=>............................] - ETA: 1:34 - loss: 0.7848 - categorical_accuracy: 0.7467
 4320/60000 [=>............................] - ETA: 1:34 - loss: 0.7798 - categorical_accuracy: 0.7484
 4352/60000 [=>............................] - ETA: 1:34 - loss: 0.7776 - categorical_accuracy: 0.7493
 4384/60000 [=>............................] - ETA: 1:34 - loss: 0.7740 - categorical_accuracy: 0.7507
 4416/60000 [=>............................] - ETA: 1:34 - loss: 0.7704 - categorical_accuracy: 0.7518
 4480/60000 [=>............................] - ETA: 1:33 - loss: 0.7635 - categorical_accuracy: 0.7547
 4512/60000 [=>............................] - ETA: 1:33 - loss: 0.7603 - categorical_accuracy: 0.7558
 4544/60000 [=>............................] - ETA: 1:33 - loss: 0.7569 - categorical_accuracy: 0.7568
 4576/60000 [=>............................] - ETA: 1:33 - loss: 0.7544 - categorical_accuracy: 0.7579
 4640/60000 [=>............................] - ETA: 1:33 - loss: 0.7499 - categorical_accuracy: 0.7597
 4672/60000 [=>............................] - ETA: 1:33 - loss: 0.7474 - categorical_accuracy: 0.7607
 4704/60000 [=>............................] - ETA: 1:33 - loss: 0.7437 - categorical_accuracy: 0.7619
 4768/60000 [=>............................] - ETA: 1:33 - loss: 0.7398 - categorical_accuracy: 0.7634
 4800/60000 [=>............................] - ETA: 1:33 - loss: 0.7393 - categorical_accuracy: 0.7638
 4832/60000 [=>............................] - ETA: 1:33 - loss: 0.7362 - categorical_accuracy: 0.7647
 4864/60000 [=>............................] - ETA: 1:32 - loss: 0.7335 - categorical_accuracy: 0.7656
 4896/60000 [=>............................] - ETA: 1:32 - loss: 0.7313 - categorical_accuracy: 0.7661
 4960/60000 [=>............................] - ETA: 1:32 - loss: 0.7253 - categorical_accuracy: 0.7679
 4992/60000 [=>............................] - ETA: 1:32 - loss: 0.7252 - categorical_accuracy: 0.7680
 5024/60000 [=>............................] - ETA: 1:32 - loss: 0.7235 - categorical_accuracy: 0.7685
 5056/60000 [=>............................] - ETA: 1:32 - loss: 0.7211 - categorical_accuracy: 0.7692
 5120/60000 [=>............................] - ETA: 1:32 - loss: 0.7155 - categorical_accuracy: 0.7707
 5184/60000 [=>............................] - ETA: 1:32 - loss: 0.7100 - categorical_accuracy: 0.7728
 5216/60000 [=>............................] - ETA: 1:32 - loss: 0.7064 - categorical_accuracy: 0.7740
 5248/60000 [=>............................] - ETA: 1:32 - loss: 0.7041 - categorical_accuracy: 0.7746
 5280/60000 [=>............................] - ETA: 1:32 - loss: 0.7011 - categorical_accuracy: 0.7758
 5312/60000 [=>............................] - ETA: 1:31 - loss: 0.6994 - categorical_accuracy: 0.7767
 5344/60000 [=>............................] - ETA: 1:31 - loss: 0.6982 - categorical_accuracy: 0.7773
 5376/60000 [=>............................] - ETA: 1:31 - loss: 0.6956 - categorical_accuracy: 0.7779
 5408/60000 [=>............................] - ETA: 1:31 - loss: 0.6922 - categorical_accuracy: 0.7792
 5440/60000 [=>............................] - ETA: 1:31 - loss: 0.6899 - categorical_accuracy: 0.7798
 5472/60000 [=>............................] - ETA: 1:31 - loss: 0.6879 - categorical_accuracy: 0.7805
 5504/60000 [=>............................] - ETA: 1:31 - loss: 0.6854 - categorical_accuracy: 0.7816
 5536/60000 [=>............................] - ETA: 1:31 - loss: 0.6852 - categorical_accuracy: 0.7822
 5568/60000 [=>............................] - ETA: 1:31 - loss: 0.6827 - categorical_accuracy: 0.7829
 5600/60000 [=>............................] - ETA: 1:31 - loss: 0.6806 - categorical_accuracy: 0.7834
 5632/60000 [=>............................] - ETA: 1:31 - loss: 0.6784 - categorical_accuracy: 0.7841
 5696/60000 [=>............................] - ETA: 1:31 - loss: 0.6746 - categorical_accuracy: 0.7858
 5728/60000 [=>............................] - ETA: 1:31 - loss: 0.6728 - categorical_accuracy: 0.7860
 5760/60000 [=>............................] - ETA: 1:31 - loss: 0.6700 - categorical_accuracy: 0.7870
 5792/60000 [=>............................] - ETA: 1:30 - loss: 0.6673 - categorical_accuracy: 0.7878
 5824/60000 [=>............................] - ETA: 1:30 - loss: 0.6652 - categorical_accuracy: 0.7886
 5856/60000 [=>............................] - ETA: 1:30 - loss: 0.6628 - categorical_accuracy: 0.7896
 5888/60000 [=>............................] - ETA: 1:30 - loss: 0.6606 - categorical_accuracy: 0.7906
 5920/60000 [=>............................] - ETA: 1:30 - loss: 0.6592 - categorical_accuracy: 0.7910
 5952/60000 [=>............................] - ETA: 1:30 - loss: 0.6562 - categorical_accuracy: 0.7920
 5984/60000 [=>............................] - ETA: 1:30 - loss: 0.6549 - categorical_accuracy: 0.7924
 6016/60000 [==>...........................] - ETA: 1:30 - loss: 0.6543 - categorical_accuracy: 0.7929
 6048/60000 [==>...........................] - ETA: 1:30 - loss: 0.6516 - categorical_accuracy: 0.7940
 6080/60000 [==>...........................] - ETA: 1:30 - loss: 0.6494 - categorical_accuracy: 0.7944
 6112/60000 [==>...........................] - ETA: 1:30 - loss: 0.6479 - categorical_accuracy: 0.7948
 6144/60000 [==>...........................] - ETA: 1:30 - loss: 0.6456 - categorical_accuracy: 0.7957
 6176/60000 [==>...........................] - ETA: 1:30 - loss: 0.6428 - categorical_accuracy: 0.7968
 6208/60000 [==>...........................] - ETA: 1:30 - loss: 0.6398 - categorical_accuracy: 0.7978
 6240/60000 [==>...........................] - ETA: 1:30 - loss: 0.6370 - categorical_accuracy: 0.7987
 6272/60000 [==>...........................] - ETA: 1:30 - loss: 0.6345 - categorical_accuracy: 0.7997
 6304/60000 [==>...........................] - ETA: 1:30 - loss: 0.6319 - categorical_accuracy: 0.8004
 6336/60000 [==>...........................] - ETA: 1:30 - loss: 0.6303 - categorical_accuracy: 0.8010
 6368/60000 [==>...........................] - ETA: 1:30 - loss: 0.6274 - categorical_accuracy: 0.8020
 6400/60000 [==>...........................] - ETA: 1:29 - loss: 0.6249 - categorical_accuracy: 0.8028
 6432/60000 [==>...........................] - ETA: 1:29 - loss: 0.6249 - categorical_accuracy: 0.8033
 6464/60000 [==>...........................] - ETA: 1:29 - loss: 0.6228 - categorical_accuracy: 0.8040
 6496/60000 [==>...........................] - ETA: 1:29 - loss: 0.6212 - categorical_accuracy: 0.8046
 6528/60000 [==>...........................] - ETA: 1:29 - loss: 0.6205 - categorical_accuracy: 0.8045
 6560/60000 [==>...........................] - ETA: 1:29 - loss: 0.6181 - categorical_accuracy: 0.8053
 6624/60000 [==>...........................] - ETA: 1:29 - loss: 0.6154 - categorical_accuracy: 0.8066
 6656/60000 [==>...........................] - ETA: 1:29 - loss: 0.6129 - categorical_accuracy: 0.8072
 6688/60000 [==>...........................] - ETA: 1:29 - loss: 0.6116 - categorical_accuracy: 0.8076
 6752/60000 [==>...........................] - ETA: 1:29 - loss: 0.6078 - categorical_accuracy: 0.8084
 6784/60000 [==>...........................] - ETA: 1:29 - loss: 0.6058 - categorical_accuracy: 0.8091
 6816/60000 [==>...........................] - ETA: 1:29 - loss: 0.6035 - categorical_accuracy: 0.8100
 6848/60000 [==>...........................] - ETA: 1:29 - loss: 0.6022 - categorical_accuracy: 0.8102
 6880/60000 [==>...........................] - ETA: 1:28 - loss: 0.6009 - categorical_accuracy: 0.8108
 6912/60000 [==>...........................] - ETA: 1:28 - loss: 0.5995 - categorical_accuracy: 0.8111
 6944/60000 [==>...........................] - ETA: 1:28 - loss: 0.5972 - categorical_accuracy: 0.8118
 6976/60000 [==>...........................] - ETA: 1:28 - loss: 0.5953 - categorical_accuracy: 0.8122
 7008/60000 [==>...........................] - ETA: 1:28 - loss: 0.5930 - categorical_accuracy: 0.8129
 7040/60000 [==>...........................] - ETA: 1:28 - loss: 0.5915 - categorical_accuracy: 0.8132
 7072/60000 [==>...........................] - ETA: 1:28 - loss: 0.5900 - categorical_accuracy: 0.8138
 7104/60000 [==>...........................] - ETA: 1:28 - loss: 0.5885 - categorical_accuracy: 0.8142
 7136/60000 [==>...........................] - ETA: 1:28 - loss: 0.5881 - categorical_accuracy: 0.8140
 7168/60000 [==>...........................] - ETA: 1:28 - loss: 0.5857 - categorical_accuracy: 0.8149
 7200/60000 [==>...........................] - ETA: 1:28 - loss: 0.5837 - categorical_accuracy: 0.8154
 7264/60000 [==>...........................] - ETA: 1:28 - loss: 0.5806 - categorical_accuracy: 0.8164
 7296/60000 [==>...........................] - ETA: 1:28 - loss: 0.5807 - categorical_accuracy: 0.8167
 7328/60000 [==>...........................] - ETA: 1:28 - loss: 0.5805 - categorical_accuracy: 0.8171
 7360/60000 [==>...........................] - ETA: 1:28 - loss: 0.5788 - categorical_accuracy: 0.8177
 7392/60000 [==>...........................] - ETA: 1:27 - loss: 0.5768 - categorical_accuracy: 0.8183
 7424/60000 [==>...........................] - ETA: 1:27 - loss: 0.5757 - categorical_accuracy: 0.8188
 7456/60000 [==>...........................] - ETA: 1:27 - loss: 0.5750 - categorical_accuracy: 0.8192
 7488/60000 [==>...........................] - ETA: 1:27 - loss: 0.5728 - categorical_accuracy: 0.8200
 7520/60000 [==>...........................] - ETA: 1:27 - loss: 0.5718 - categorical_accuracy: 0.8205
 7552/60000 [==>...........................] - ETA: 1:27 - loss: 0.5704 - categorical_accuracy: 0.8208
 7584/60000 [==>...........................] - ETA: 1:27 - loss: 0.5685 - categorical_accuracy: 0.8215
 7616/60000 [==>...........................] - ETA: 1:27 - loss: 0.5682 - categorical_accuracy: 0.8214
 7648/60000 [==>...........................] - ETA: 1:27 - loss: 0.5679 - categorical_accuracy: 0.8218
 7680/60000 [==>...........................] - ETA: 1:27 - loss: 0.5681 - categorical_accuracy: 0.8216
 7712/60000 [==>...........................] - ETA: 1:27 - loss: 0.5678 - categorical_accuracy: 0.8218
 7776/60000 [==>...........................] - ETA: 1:27 - loss: 0.5657 - categorical_accuracy: 0.8227
 7808/60000 [==>...........................] - ETA: 1:27 - loss: 0.5636 - categorical_accuracy: 0.8234
 7840/60000 [==>...........................] - ETA: 1:27 - loss: 0.5628 - categorical_accuracy: 0.8239
 7872/60000 [==>...........................] - ETA: 1:26 - loss: 0.5612 - categorical_accuracy: 0.8244
 7904/60000 [==>...........................] - ETA: 1:26 - loss: 0.5596 - categorical_accuracy: 0.8249
 7936/60000 [==>...........................] - ETA: 1:26 - loss: 0.5589 - categorical_accuracy: 0.8250
 7968/60000 [==>...........................] - ETA: 1:26 - loss: 0.5570 - categorical_accuracy: 0.8256
 8000/60000 [===>..........................] - ETA: 1:26 - loss: 0.5557 - categorical_accuracy: 0.8259
 8032/60000 [===>..........................] - ETA: 1:26 - loss: 0.5548 - categorical_accuracy: 0.8262
 8064/60000 [===>..........................] - ETA: 1:26 - loss: 0.5540 - categorical_accuracy: 0.8266
 8096/60000 [===>..........................] - ETA: 1:26 - loss: 0.5528 - categorical_accuracy: 0.8268
 8128/60000 [===>..........................] - ETA: 1:26 - loss: 0.5522 - categorical_accuracy: 0.8271
 8192/60000 [===>..........................] - ETA: 1:26 - loss: 0.5503 - categorical_accuracy: 0.8278
 8224/60000 [===>..........................] - ETA: 1:26 - loss: 0.5490 - categorical_accuracy: 0.8282
 8256/60000 [===>..........................] - ETA: 1:26 - loss: 0.5484 - categorical_accuracy: 0.8282
 8288/60000 [===>..........................] - ETA: 1:26 - loss: 0.5467 - categorical_accuracy: 0.8288
 8320/60000 [===>..........................] - ETA: 1:26 - loss: 0.5455 - categorical_accuracy: 0.8291
 8352/60000 [===>..........................] - ETA: 1:26 - loss: 0.5444 - categorical_accuracy: 0.8293
 8384/60000 [===>..........................] - ETA: 1:25 - loss: 0.5428 - categorical_accuracy: 0.8298
 8416/60000 [===>..........................] - ETA: 1:25 - loss: 0.5420 - categorical_accuracy: 0.8301
 8448/60000 [===>..........................] - ETA: 1:25 - loss: 0.5404 - categorical_accuracy: 0.8307
 8480/60000 [===>..........................] - ETA: 1:25 - loss: 0.5391 - categorical_accuracy: 0.8310
 8544/60000 [===>..........................] - ETA: 1:25 - loss: 0.5375 - categorical_accuracy: 0.8317
 8576/60000 [===>..........................] - ETA: 1:25 - loss: 0.5359 - categorical_accuracy: 0.8322
 8608/60000 [===>..........................] - ETA: 1:25 - loss: 0.5348 - categorical_accuracy: 0.8325
 8640/60000 [===>..........................] - ETA: 1:25 - loss: 0.5338 - categorical_accuracy: 0.8328
 8672/60000 [===>..........................] - ETA: 1:25 - loss: 0.5323 - categorical_accuracy: 0.8331
 8704/60000 [===>..........................] - ETA: 1:25 - loss: 0.5308 - categorical_accuracy: 0.8336
 8736/60000 [===>..........................] - ETA: 1:25 - loss: 0.5292 - categorical_accuracy: 0.8342
 8768/60000 [===>..........................] - ETA: 1:25 - loss: 0.5290 - categorical_accuracy: 0.8342
 8800/60000 [===>..........................] - ETA: 1:25 - loss: 0.5272 - categorical_accuracy: 0.8348
 8832/60000 [===>..........................] - ETA: 1:25 - loss: 0.5267 - categorical_accuracy: 0.8351
 8864/60000 [===>..........................] - ETA: 1:25 - loss: 0.5256 - categorical_accuracy: 0.8354
 8896/60000 [===>..........................] - ETA: 1:25 - loss: 0.5253 - categorical_accuracy: 0.8357
 8928/60000 [===>..........................] - ETA: 1:24 - loss: 0.5246 - categorical_accuracy: 0.8360
 8960/60000 [===>..........................] - ETA: 1:24 - loss: 0.5234 - categorical_accuracy: 0.8363
 8992/60000 [===>..........................] - ETA: 1:24 - loss: 0.5219 - categorical_accuracy: 0.8367
 9024/60000 [===>..........................] - ETA: 1:24 - loss: 0.5213 - categorical_accuracy: 0.8370
 9056/60000 [===>..........................] - ETA: 1:24 - loss: 0.5209 - categorical_accuracy: 0.8373
 9120/60000 [===>..........................] - ETA: 1:24 - loss: 0.5183 - categorical_accuracy: 0.8380
 9152/60000 [===>..........................] - ETA: 1:24 - loss: 0.5170 - categorical_accuracy: 0.8385
 9216/60000 [===>..........................] - ETA: 1:24 - loss: 0.5147 - categorical_accuracy: 0.8392
 9248/60000 [===>..........................] - ETA: 1:24 - loss: 0.5132 - categorical_accuracy: 0.8395
 9280/60000 [===>..........................] - ETA: 1:24 - loss: 0.5117 - categorical_accuracy: 0.8400
 9312/60000 [===>..........................] - ETA: 1:24 - loss: 0.5102 - categorical_accuracy: 0.8405
 9344/60000 [===>..........................] - ETA: 1:24 - loss: 0.5095 - categorical_accuracy: 0.8409
 9376/60000 [===>..........................] - ETA: 1:24 - loss: 0.5088 - categorical_accuracy: 0.8411
 9408/60000 [===>..........................] - ETA: 1:24 - loss: 0.5078 - categorical_accuracy: 0.8414
 9440/60000 [===>..........................] - ETA: 1:23 - loss: 0.5068 - categorical_accuracy: 0.8418
 9504/60000 [===>..........................] - ETA: 1:23 - loss: 0.5050 - categorical_accuracy: 0.8425
 9536/60000 [===>..........................] - ETA: 1:23 - loss: 0.5036 - categorical_accuracy: 0.8430
 9568/60000 [===>..........................] - ETA: 1:23 - loss: 0.5024 - categorical_accuracy: 0.8433
 9600/60000 [===>..........................] - ETA: 1:23 - loss: 0.5010 - categorical_accuracy: 0.8438
 9632/60000 [===>..........................] - ETA: 1:23 - loss: 0.4995 - categorical_accuracy: 0.8443
 9664/60000 [===>..........................] - ETA: 1:23 - loss: 0.4988 - categorical_accuracy: 0.8445
 9696/60000 [===>..........................] - ETA: 1:23 - loss: 0.4974 - categorical_accuracy: 0.8449
 9728/60000 [===>..........................] - ETA: 1:23 - loss: 0.4961 - categorical_accuracy: 0.8453
 9760/60000 [===>..........................] - ETA: 1:23 - loss: 0.4954 - categorical_accuracy: 0.8457
 9792/60000 [===>..........................] - ETA: 1:23 - loss: 0.4942 - categorical_accuracy: 0.8461
 9856/60000 [===>..........................] - ETA: 1:23 - loss: 0.4917 - categorical_accuracy: 0.8468
 9888/60000 [===>..........................] - ETA: 1:23 - loss: 0.4913 - categorical_accuracy: 0.8468
 9920/60000 [===>..........................] - ETA: 1:23 - loss: 0.4906 - categorical_accuracy: 0.8471
 9952/60000 [===>..........................] - ETA: 1:23 - loss: 0.4899 - categorical_accuracy: 0.8472
 9984/60000 [===>..........................] - ETA: 1:22 - loss: 0.4889 - categorical_accuracy: 0.8475
10016/60000 [====>.........................] - ETA: 1:22 - loss: 0.4877 - categorical_accuracy: 0.8477
10048/60000 [====>.........................] - ETA: 1:22 - loss: 0.4869 - categorical_accuracy: 0.8479
10080/60000 [====>.........................] - ETA: 1:22 - loss: 0.4858 - categorical_accuracy: 0.8483
10144/60000 [====>.........................] - ETA: 1:22 - loss: 0.4833 - categorical_accuracy: 0.8493
10176/60000 [====>.........................] - ETA: 1:22 - loss: 0.4828 - categorical_accuracy: 0.8493
10208/60000 [====>.........................] - ETA: 1:22 - loss: 0.4821 - categorical_accuracy: 0.8493
10272/60000 [====>.........................] - ETA: 1:22 - loss: 0.4810 - categorical_accuracy: 0.8498
10304/60000 [====>.........................] - ETA: 1:22 - loss: 0.4808 - categorical_accuracy: 0.8501
10336/60000 [====>.........................] - ETA: 1:22 - loss: 0.4795 - categorical_accuracy: 0.8505
10368/60000 [====>.........................] - ETA: 1:22 - loss: 0.4783 - categorical_accuracy: 0.8510
10400/60000 [====>.........................] - ETA: 1:22 - loss: 0.4780 - categorical_accuracy: 0.8511
10432/60000 [====>.........................] - ETA: 1:22 - loss: 0.4774 - categorical_accuracy: 0.8514
10464/60000 [====>.........................] - ETA: 1:22 - loss: 0.4762 - categorical_accuracy: 0.8518
10496/60000 [====>.........................] - ETA: 1:22 - loss: 0.4753 - categorical_accuracy: 0.8520
10560/60000 [====>.........................] - ETA: 1:21 - loss: 0.4732 - categorical_accuracy: 0.8526
10592/60000 [====>.........................] - ETA: 1:21 - loss: 0.4722 - categorical_accuracy: 0.8529
10624/60000 [====>.........................] - ETA: 1:21 - loss: 0.4709 - categorical_accuracy: 0.8534
10656/60000 [====>.........................] - ETA: 1:21 - loss: 0.4699 - categorical_accuracy: 0.8536
10688/60000 [====>.........................] - ETA: 1:21 - loss: 0.4691 - categorical_accuracy: 0.8539
10752/60000 [====>.........................] - ETA: 1:21 - loss: 0.4672 - categorical_accuracy: 0.8544
10784/60000 [====>.........................] - ETA: 1:21 - loss: 0.4660 - categorical_accuracy: 0.8549
10816/60000 [====>.........................] - ETA: 1:21 - loss: 0.4654 - categorical_accuracy: 0.8552
10848/60000 [====>.........................] - ETA: 1:21 - loss: 0.4644 - categorical_accuracy: 0.8555
10880/60000 [====>.........................] - ETA: 1:21 - loss: 0.4641 - categorical_accuracy: 0.8556
10912/60000 [====>.........................] - ETA: 1:21 - loss: 0.4629 - categorical_accuracy: 0.8559
10944/60000 [====>.........................] - ETA: 1:21 - loss: 0.4622 - categorical_accuracy: 0.8563
11008/60000 [====>.........................] - ETA: 1:21 - loss: 0.4615 - categorical_accuracy: 0.8568
11040/60000 [====>.........................] - ETA: 1:21 - loss: 0.4607 - categorical_accuracy: 0.8571
11072/60000 [====>.........................] - ETA: 1:21 - loss: 0.4604 - categorical_accuracy: 0.8570
11104/60000 [====>.........................] - ETA: 1:20 - loss: 0.4597 - categorical_accuracy: 0.8573
11136/60000 [====>.........................] - ETA: 1:20 - loss: 0.4591 - categorical_accuracy: 0.8575
11168/60000 [====>.........................] - ETA: 1:20 - loss: 0.4592 - categorical_accuracy: 0.8574
11232/60000 [====>.........................] - ETA: 1:20 - loss: 0.4593 - categorical_accuracy: 0.8572
11264/60000 [====>.........................] - ETA: 1:20 - loss: 0.4582 - categorical_accuracy: 0.8575
11296/60000 [====>.........................] - ETA: 1:20 - loss: 0.4575 - categorical_accuracy: 0.8577
11328/60000 [====>.........................] - ETA: 1:20 - loss: 0.4564 - categorical_accuracy: 0.8581
11360/60000 [====>.........................] - ETA: 1:20 - loss: 0.4563 - categorical_accuracy: 0.8580
11424/60000 [====>.........................] - ETA: 1:20 - loss: 0.4551 - categorical_accuracy: 0.8585
11456/60000 [====>.........................] - ETA: 1:20 - loss: 0.4544 - categorical_accuracy: 0.8587
11488/60000 [====>.........................] - ETA: 1:20 - loss: 0.4534 - categorical_accuracy: 0.8590
11520/60000 [====>.........................] - ETA: 1:20 - loss: 0.4524 - categorical_accuracy: 0.8593
11552/60000 [====>.........................] - ETA: 1:20 - loss: 0.4522 - categorical_accuracy: 0.8593
11584/60000 [====>.........................] - ETA: 1:20 - loss: 0.4513 - categorical_accuracy: 0.8595
11616/60000 [====>.........................] - ETA: 1:20 - loss: 0.4504 - categorical_accuracy: 0.8598
11648/60000 [====>.........................] - ETA: 1:20 - loss: 0.4494 - categorical_accuracy: 0.8602
11680/60000 [====>.........................] - ETA: 1:20 - loss: 0.4490 - categorical_accuracy: 0.8604
11712/60000 [====>.........................] - ETA: 1:19 - loss: 0.4481 - categorical_accuracy: 0.8607
11744/60000 [====>.........................] - ETA: 1:19 - loss: 0.4479 - categorical_accuracy: 0.8607
11776/60000 [====>.........................] - ETA: 1:19 - loss: 0.4472 - categorical_accuracy: 0.8609
11808/60000 [====>.........................] - ETA: 1:19 - loss: 0.4462 - categorical_accuracy: 0.8612
11840/60000 [====>.........................] - ETA: 1:19 - loss: 0.4455 - categorical_accuracy: 0.8613
11872/60000 [====>.........................] - ETA: 1:19 - loss: 0.4452 - categorical_accuracy: 0.8613
11904/60000 [====>.........................] - ETA: 1:19 - loss: 0.4444 - categorical_accuracy: 0.8616
11968/60000 [====>.........................] - ETA: 1:19 - loss: 0.4434 - categorical_accuracy: 0.8619
12000/60000 [=====>........................] - ETA: 1:19 - loss: 0.4426 - categorical_accuracy: 0.8622
12032/60000 [=====>........................] - ETA: 1:19 - loss: 0.4415 - categorical_accuracy: 0.8625
12064/60000 [=====>........................] - ETA: 1:19 - loss: 0.4405 - categorical_accuracy: 0.8629
12096/60000 [=====>........................] - ETA: 1:19 - loss: 0.4400 - categorical_accuracy: 0.8632
12128/60000 [=====>........................] - ETA: 1:19 - loss: 0.4394 - categorical_accuracy: 0.8633
12160/60000 [=====>........................] - ETA: 1:19 - loss: 0.4387 - categorical_accuracy: 0.8635
12192/60000 [=====>........................] - ETA: 1:19 - loss: 0.4382 - categorical_accuracy: 0.8637
12224/60000 [=====>........................] - ETA: 1:19 - loss: 0.4375 - categorical_accuracy: 0.8639
12256/60000 [=====>........................] - ETA: 1:19 - loss: 0.4367 - categorical_accuracy: 0.8641
12288/60000 [=====>........................] - ETA: 1:18 - loss: 0.4364 - categorical_accuracy: 0.8643
12320/60000 [=====>........................] - ETA: 1:18 - loss: 0.4354 - categorical_accuracy: 0.8647
12352/60000 [=====>........................] - ETA: 1:18 - loss: 0.4354 - categorical_accuracy: 0.8648
12384/60000 [=====>........................] - ETA: 1:18 - loss: 0.4347 - categorical_accuracy: 0.8650
12416/60000 [=====>........................] - ETA: 1:18 - loss: 0.4339 - categorical_accuracy: 0.8653
12448/60000 [=====>........................] - ETA: 1:18 - loss: 0.4330 - categorical_accuracy: 0.8655
12480/60000 [=====>........................] - ETA: 1:18 - loss: 0.4326 - categorical_accuracy: 0.8656
12512/60000 [=====>........................] - ETA: 1:18 - loss: 0.4323 - categorical_accuracy: 0.8656
12544/60000 [=====>........................] - ETA: 1:18 - loss: 0.4320 - categorical_accuracy: 0.8657
12576/60000 [=====>........................] - ETA: 1:18 - loss: 0.4310 - categorical_accuracy: 0.8660
12608/60000 [=====>........................] - ETA: 1:18 - loss: 0.4313 - categorical_accuracy: 0.8660
12640/60000 [=====>........................] - ETA: 1:18 - loss: 0.4308 - categorical_accuracy: 0.8661
12672/60000 [=====>........................] - ETA: 1:18 - loss: 0.4302 - categorical_accuracy: 0.8662
12704/60000 [=====>........................] - ETA: 1:18 - loss: 0.4295 - categorical_accuracy: 0.8664
12768/60000 [=====>........................] - ETA: 1:18 - loss: 0.4278 - categorical_accuracy: 0.8670
12800/60000 [=====>........................] - ETA: 1:18 - loss: 0.4270 - categorical_accuracy: 0.8673
12832/60000 [=====>........................] - ETA: 1:18 - loss: 0.4263 - categorical_accuracy: 0.8675
12896/60000 [=====>........................] - ETA: 1:17 - loss: 0.4261 - categorical_accuracy: 0.8676
12928/60000 [=====>........................] - ETA: 1:17 - loss: 0.4252 - categorical_accuracy: 0.8679
12960/60000 [=====>........................] - ETA: 1:17 - loss: 0.4245 - categorical_accuracy: 0.8681
12992/60000 [=====>........................] - ETA: 1:17 - loss: 0.4236 - categorical_accuracy: 0.8685
13056/60000 [=====>........................] - ETA: 1:17 - loss: 0.4223 - categorical_accuracy: 0.8688
13088/60000 [=====>........................] - ETA: 1:17 - loss: 0.4214 - categorical_accuracy: 0.8690
13120/60000 [=====>........................] - ETA: 1:17 - loss: 0.4206 - categorical_accuracy: 0.8694
13152/60000 [=====>........................] - ETA: 1:17 - loss: 0.4200 - categorical_accuracy: 0.8694
13216/60000 [=====>........................] - ETA: 1:17 - loss: 0.4191 - categorical_accuracy: 0.8697
13248/60000 [=====>........................] - ETA: 1:17 - loss: 0.4185 - categorical_accuracy: 0.8699
13280/60000 [=====>........................] - ETA: 1:17 - loss: 0.4187 - categorical_accuracy: 0.8699
13312/60000 [=====>........................] - ETA: 1:17 - loss: 0.4182 - categorical_accuracy: 0.8700
13376/60000 [=====>........................] - ETA: 1:17 - loss: 0.4168 - categorical_accuracy: 0.8704
13408/60000 [=====>........................] - ETA: 1:16 - loss: 0.4162 - categorical_accuracy: 0.8707
13440/60000 [=====>........................] - ETA: 1:16 - loss: 0.4154 - categorical_accuracy: 0.8708
13472/60000 [=====>........................] - ETA: 1:16 - loss: 0.4150 - categorical_accuracy: 0.8710
13504/60000 [=====>........................] - ETA: 1:16 - loss: 0.4142 - categorical_accuracy: 0.8713
13536/60000 [=====>........................] - ETA: 1:16 - loss: 0.4145 - categorical_accuracy: 0.8713
13568/60000 [=====>........................] - ETA: 1:16 - loss: 0.4138 - categorical_accuracy: 0.8715
13600/60000 [=====>........................] - ETA: 1:16 - loss: 0.4132 - categorical_accuracy: 0.8716
13632/60000 [=====>........................] - ETA: 1:16 - loss: 0.4124 - categorical_accuracy: 0.8718
13696/60000 [=====>........................] - ETA: 1:16 - loss: 0.4111 - categorical_accuracy: 0.8722
13728/60000 [=====>........................] - ETA: 1:16 - loss: 0.4102 - categorical_accuracy: 0.8725
13792/60000 [=====>........................] - ETA: 1:16 - loss: 0.4087 - categorical_accuracy: 0.8730
13824/60000 [=====>........................] - ETA: 1:16 - loss: 0.4080 - categorical_accuracy: 0.8732
13856/60000 [=====>........................] - ETA: 1:16 - loss: 0.4074 - categorical_accuracy: 0.8733
13888/60000 [=====>........................] - ETA: 1:16 - loss: 0.4068 - categorical_accuracy: 0.8734
13920/60000 [=====>........................] - ETA: 1:16 - loss: 0.4067 - categorical_accuracy: 0.8734
13952/60000 [=====>........................] - ETA: 1:16 - loss: 0.4062 - categorical_accuracy: 0.8736
14016/60000 [======>.......................] - ETA: 1:15 - loss: 0.4053 - categorical_accuracy: 0.8741
14048/60000 [======>.......................] - ETA: 1:15 - loss: 0.4055 - categorical_accuracy: 0.8741
14080/60000 [======>.......................] - ETA: 1:15 - loss: 0.4048 - categorical_accuracy: 0.8744
14112/60000 [======>.......................] - ETA: 1:15 - loss: 0.4041 - categorical_accuracy: 0.8746
14144/60000 [======>.......................] - ETA: 1:15 - loss: 0.4033 - categorical_accuracy: 0.8749
14176/60000 [======>.......................] - ETA: 1:15 - loss: 0.4038 - categorical_accuracy: 0.8749
14240/60000 [======>.......................] - ETA: 1:15 - loss: 0.4026 - categorical_accuracy: 0.8751
14272/60000 [======>.......................] - ETA: 1:15 - loss: 0.4019 - categorical_accuracy: 0.8754
14336/60000 [======>.......................] - ETA: 1:15 - loss: 0.4009 - categorical_accuracy: 0.8758
14368/60000 [======>.......................] - ETA: 1:15 - loss: 0.4002 - categorical_accuracy: 0.8760
14400/60000 [======>.......................] - ETA: 1:15 - loss: 0.3996 - categorical_accuracy: 0.8762
14432/60000 [======>.......................] - ETA: 1:15 - loss: 0.3990 - categorical_accuracy: 0.8764
14464/60000 [======>.......................] - ETA: 1:15 - loss: 0.3983 - categorical_accuracy: 0.8767
14496/60000 [======>.......................] - ETA: 1:15 - loss: 0.3975 - categorical_accuracy: 0.8769
14528/60000 [======>.......................] - ETA: 1:15 - loss: 0.3967 - categorical_accuracy: 0.8771
14560/60000 [======>.......................] - ETA: 1:15 - loss: 0.3960 - categorical_accuracy: 0.8773
14592/60000 [======>.......................] - ETA: 1:15 - loss: 0.3953 - categorical_accuracy: 0.8775
14624/60000 [======>.......................] - ETA: 1:15 - loss: 0.3953 - categorical_accuracy: 0.8777
14656/60000 [======>.......................] - ETA: 1:14 - loss: 0.3949 - categorical_accuracy: 0.8777
14688/60000 [======>.......................] - ETA: 1:14 - loss: 0.3946 - categorical_accuracy: 0.8778
14720/60000 [======>.......................] - ETA: 1:14 - loss: 0.3941 - categorical_accuracy: 0.8779
14752/60000 [======>.......................] - ETA: 1:14 - loss: 0.3937 - categorical_accuracy: 0.8780
14784/60000 [======>.......................] - ETA: 1:14 - loss: 0.3931 - categorical_accuracy: 0.8782
14816/60000 [======>.......................] - ETA: 1:14 - loss: 0.3924 - categorical_accuracy: 0.8783
14848/60000 [======>.......................] - ETA: 1:14 - loss: 0.3922 - categorical_accuracy: 0.8783
14880/60000 [======>.......................] - ETA: 1:14 - loss: 0.3914 - categorical_accuracy: 0.8785
14912/60000 [======>.......................] - ETA: 1:14 - loss: 0.3907 - categorical_accuracy: 0.8787
14944/60000 [======>.......................] - ETA: 1:14 - loss: 0.3900 - categorical_accuracy: 0.8789
14976/60000 [======>.......................] - ETA: 1:14 - loss: 0.3896 - categorical_accuracy: 0.8791
15008/60000 [======>.......................] - ETA: 1:14 - loss: 0.3890 - categorical_accuracy: 0.8793
15040/60000 [======>.......................] - ETA: 1:14 - loss: 0.3886 - categorical_accuracy: 0.8794
15072/60000 [======>.......................] - ETA: 1:14 - loss: 0.3878 - categorical_accuracy: 0.8796
15104/60000 [======>.......................] - ETA: 1:14 - loss: 0.3878 - categorical_accuracy: 0.8797
15136/60000 [======>.......................] - ETA: 1:14 - loss: 0.3874 - categorical_accuracy: 0.8798
15168/60000 [======>.......................] - ETA: 1:14 - loss: 0.3868 - categorical_accuracy: 0.8799
15200/60000 [======>.......................] - ETA: 1:14 - loss: 0.3861 - categorical_accuracy: 0.8801
15232/60000 [======>.......................] - ETA: 1:13 - loss: 0.3855 - categorical_accuracy: 0.8803
15264/60000 [======>.......................] - ETA: 1:13 - loss: 0.3852 - categorical_accuracy: 0.8804
15296/60000 [======>.......................] - ETA: 1:13 - loss: 0.3851 - categorical_accuracy: 0.8806
15328/60000 [======>.......................] - ETA: 1:13 - loss: 0.3846 - categorical_accuracy: 0.8807
15360/60000 [======>.......................] - ETA: 1:13 - loss: 0.3838 - categorical_accuracy: 0.8809
15392/60000 [======>.......................] - ETA: 1:13 - loss: 0.3831 - categorical_accuracy: 0.8810
15424/60000 [======>.......................] - ETA: 1:13 - loss: 0.3826 - categorical_accuracy: 0.8812
15456/60000 [======>.......................] - ETA: 1:13 - loss: 0.3822 - categorical_accuracy: 0.8813
15488/60000 [======>.......................] - ETA: 1:13 - loss: 0.3824 - categorical_accuracy: 0.8813
15520/60000 [======>.......................] - ETA: 1:13 - loss: 0.3820 - categorical_accuracy: 0.8814
15552/60000 [======>.......................] - ETA: 1:13 - loss: 0.3817 - categorical_accuracy: 0.8815
15584/60000 [======>.......................] - ETA: 1:13 - loss: 0.3810 - categorical_accuracy: 0.8817
15616/60000 [======>.......................] - ETA: 1:13 - loss: 0.3803 - categorical_accuracy: 0.8820
15680/60000 [======>.......................] - ETA: 1:13 - loss: 0.3789 - categorical_accuracy: 0.8825
15712/60000 [======>.......................] - ETA: 1:13 - loss: 0.3783 - categorical_accuracy: 0.8827
15744/60000 [======>.......................] - ETA: 1:13 - loss: 0.3779 - categorical_accuracy: 0.8828
15776/60000 [======>.......................] - ETA: 1:13 - loss: 0.3775 - categorical_accuracy: 0.8830
15840/60000 [======>.......................] - ETA: 1:12 - loss: 0.3771 - categorical_accuracy: 0.8831
15872/60000 [======>.......................] - ETA: 1:12 - loss: 0.3768 - categorical_accuracy: 0.8833
15904/60000 [======>.......................] - ETA: 1:12 - loss: 0.3766 - categorical_accuracy: 0.8834
15936/60000 [======>.......................] - ETA: 1:12 - loss: 0.3766 - categorical_accuracy: 0.8835
15968/60000 [======>.......................] - ETA: 1:12 - loss: 0.3762 - categorical_accuracy: 0.8835
16000/60000 [=======>......................] - ETA: 1:12 - loss: 0.3756 - categorical_accuracy: 0.8837
16032/60000 [=======>......................] - ETA: 1:12 - loss: 0.3754 - categorical_accuracy: 0.8838
16064/60000 [=======>......................] - ETA: 1:12 - loss: 0.3750 - categorical_accuracy: 0.8838
16096/60000 [=======>......................] - ETA: 1:12 - loss: 0.3754 - categorical_accuracy: 0.8838
16128/60000 [=======>......................] - ETA: 1:12 - loss: 0.3749 - categorical_accuracy: 0.8839
16160/60000 [=======>......................] - ETA: 1:12 - loss: 0.3743 - categorical_accuracy: 0.8841
16192/60000 [=======>......................] - ETA: 1:12 - loss: 0.3737 - categorical_accuracy: 0.8842
16224/60000 [=======>......................] - ETA: 1:12 - loss: 0.3734 - categorical_accuracy: 0.8842
16256/60000 [=======>......................] - ETA: 1:12 - loss: 0.3729 - categorical_accuracy: 0.8844
16288/60000 [=======>......................] - ETA: 1:12 - loss: 0.3726 - categorical_accuracy: 0.8845
16320/60000 [=======>......................] - ETA: 1:12 - loss: 0.3720 - categorical_accuracy: 0.8847
16352/60000 [=======>......................] - ETA: 1:12 - loss: 0.3718 - categorical_accuracy: 0.8848
16384/60000 [=======>......................] - ETA: 1:12 - loss: 0.3715 - categorical_accuracy: 0.8850
16416/60000 [=======>......................] - ETA: 1:12 - loss: 0.3708 - categorical_accuracy: 0.8852
16448/60000 [=======>......................] - ETA: 1:11 - loss: 0.3701 - categorical_accuracy: 0.8855
16480/60000 [=======>......................] - ETA: 1:11 - loss: 0.3697 - categorical_accuracy: 0.8856
16512/60000 [=======>......................] - ETA: 1:11 - loss: 0.3692 - categorical_accuracy: 0.8857
16544/60000 [=======>......................] - ETA: 1:11 - loss: 0.3687 - categorical_accuracy: 0.8859
16576/60000 [=======>......................] - ETA: 1:11 - loss: 0.3688 - categorical_accuracy: 0.8858
16608/60000 [=======>......................] - ETA: 1:11 - loss: 0.3687 - categorical_accuracy: 0.8858
16640/60000 [=======>......................] - ETA: 1:11 - loss: 0.3681 - categorical_accuracy: 0.8861
16672/60000 [=======>......................] - ETA: 1:11 - loss: 0.3676 - categorical_accuracy: 0.8863
16704/60000 [=======>......................] - ETA: 1:11 - loss: 0.3671 - categorical_accuracy: 0.8864
16736/60000 [=======>......................] - ETA: 1:11 - loss: 0.3669 - categorical_accuracy: 0.8865
16768/60000 [=======>......................] - ETA: 1:11 - loss: 0.3664 - categorical_accuracy: 0.8867
16800/60000 [=======>......................] - ETA: 1:11 - loss: 0.3659 - categorical_accuracy: 0.8868
16832/60000 [=======>......................] - ETA: 1:11 - loss: 0.3656 - categorical_accuracy: 0.8869
16864/60000 [=======>......................] - ETA: 1:11 - loss: 0.3651 - categorical_accuracy: 0.8871
16896/60000 [=======>......................] - ETA: 1:11 - loss: 0.3646 - categorical_accuracy: 0.8873
16928/60000 [=======>......................] - ETA: 1:11 - loss: 0.3643 - categorical_accuracy: 0.8873
16960/60000 [=======>......................] - ETA: 1:11 - loss: 0.3640 - categorical_accuracy: 0.8874
16992/60000 [=======>......................] - ETA: 1:11 - loss: 0.3638 - categorical_accuracy: 0.8876
17024/60000 [=======>......................] - ETA: 1:10 - loss: 0.3632 - categorical_accuracy: 0.8878
17056/60000 [=======>......................] - ETA: 1:10 - loss: 0.3630 - categorical_accuracy: 0.8879
17088/60000 [=======>......................] - ETA: 1:10 - loss: 0.3626 - categorical_accuracy: 0.8880
17120/60000 [=======>......................] - ETA: 1:10 - loss: 0.3619 - categorical_accuracy: 0.8882
17152/60000 [=======>......................] - ETA: 1:10 - loss: 0.3617 - categorical_accuracy: 0.8882
17184/60000 [=======>......................] - ETA: 1:10 - loss: 0.3616 - categorical_accuracy: 0.8883
17216/60000 [=======>......................] - ETA: 1:10 - loss: 0.3611 - categorical_accuracy: 0.8885
17248/60000 [=======>......................] - ETA: 1:10 - loss: 0.3609 - categorical_accuracy: 0.8886
17312/60000 [=======>......................] - ETA: 1:10 - loss: 0.3599 - categorical_accuracy: 0.8889
17376/60000 [=======>......................] - ETA: 1:10 - loss: 0.3592 - categorical_accuracy: 0.8891
17408/60000 [=======>......................] - ETA: 1:10 - loss: 0.3586 - categorical_accuracy: 0.8893
17440/60000 [=======>......................] - ETA: 1:10 - loss: 0.3580 - categorical_accuracy: 0.8895
17472/60000 [=======>......................] - ETA: 1:10 - loss: 0.3575 - categorical_accuracy: 0.8897
17504/60000 [=======>......................] - ETA: 1:10 - loss: 0.3569 - categorical_accuracy: 0.8898
17536/60000 [=======>......................] - ETA: 1:10 - loss: 0.3564 - categorical_accuracy: 0.8900
17568/60000 [=======>......................] - ETA: 1:10 - loss: 0.3559 - categorical_accuracy: 0.8901
17600/60000 [=======>......................] - ETA: 1:10 - loss: 0.3554 - categorical_accuracy: 0.8902
17632/60000 [=======>......................] - ETA: 1:09 - loss: 0.3551 - categorical_accuracy: 0.8903
17664/60000 [=======>......................] - ETA: 1:09 - loss: 0.3545 - categorical_accuracy: 0.8905
17696/60000 [=======>......................] - ETA: 1:09 - loss: 0.3542 - categorical_accuracy: 0.8905
17728/60000 [=======>......................] - ETA: 1:09 - loss: 0.3540 - categorical_accuracy: 0.8906
17760/60000 [=======>......................] - ETA: 1:09 - loss: 0.3536 - categorical_accuracy: 0.8908
17792/60000 [=======>......................] - ETA: 1:09 - loss: 0.3538 - categorical_accuracy: 0.8908
17824/60000 [=======>......................] - ETA: 1:09 - loss: 0.3541 - categorical_accuracy: 0.8907
17856/60000 [=======>......................] - ETA: 1:09 - loss: 0.3536 - categorical_accuracy: 0.8908
17888/60000 [=======>......................] - ETA: 1:09 - loss: 0.3532 - categorical_accuracy: 0.8909
17920/60000 [=======>......................] - ETA: 1:09 - loss: 0.3527 - categorical_accuracy: 0.8911
17952/60000 [=======>......................] - ETA: 1:09 - loss: 0.3522 - categorical_accuracy: 0.8912
17984/60000 [=======>......................] - ETA: 1:09 - loss: 0.3517 - categorical_accuracy: 0.8914
18016/60000 [========>.....................] - ETA: 1:09 - loss: 0.3512 - categorical_accuracy: 0.8916
18048/60000 [========>.....................] - ETA: 1:09 - loss: 0.3510 - categorical_accuracy: 0.8917
18080/60000 [========>.....................] - ETA: 1:09 - loss: 0.3505 - categorical_accuracy: 0.8918
18112/60000 [========>.....................] - ETA: 1:09 - loss: 0.3500 - categorical_accuracy: 0.8920
18144/60000 [========>.....................] - ETA: 1:09 - loss: 0.3498 - categorical_accuracy: 0.8921
18176/60000 [========>.....................] - ETA: 1:09 - loss: 0.3493 - categorical_accuracy: 0.8923
18208/60000 [========>.....................] - ETA: 1:09 - loss: 0.3487 - categorical_accuracy: 0.8925
18240/60000 [========>.....................] - ETA: 1:09 - loss: 0.3482 - categorical_accuracy: 0.8927
18272/60000 [========>.....................] - ETA: 1:08 - loss: 0.3481 - categorical_accuracy: 0.8927
18304/60000 [========>.....................] - ETA: 1:08 - loss: 0.3479 - categorical_accuracy: 0.8927
18336/60000 [========>.....................] - ETA: 1:08 - loss: 0.3480 - categorical_accuracy: 0.8927
18368/60000 [========>.....................] - ETA: 1:08 - loss: 0.3476 - categorical_accuracy: 0.8929
18432/60000 [========>.....................] - ETA: 1:08 - loss: 0.3465 - categorical_accuracy: 0.8932
18496/60000 [========>.....................] - ETA: 1:08 - loss: 0.3455 - categorical_accuracy: 0.8935
18528/60000 [========>.....................] - ETA: 1:08 - loss: 0.3451 - categorical_accuracy: 0.8936
18560/60000 [========>.....................] - ETA: 1:08 - loss: 0.3447 - categorical_accuracy: 0.8937
18592/60000 [========>.....................] - ETA: 1:08 - loss: 0.3443 - categorical_accuracy: 0.8938
18624/60000 [========>.....................] - ETA: 1:08 - loss: 0.3443 - categorical_accuracy: 0.8938
18656/60000 [========>.....................] - ETA: 1:08 - loss: 0.3443 - categorical_accuracy: 0.8939
18688/60000 [========>.....................] - ETA: 1:08 - loss: 0.3438 - categorical_accuracy: 0.8940
18720/60000 [========>.....................] - ETA: 1:08 - loss: 0.3433 - categorical_accuracy: 0.8942
18752/60000 [========>.....................] - ETA: 1:08 - loss: 0.3428 - categorical_accuracy: 0.8944
18784/60000 [========>.....................] - ETA: 1:08 - loss: 0.3428 - categorical_accuracy: 0.8945
18816/60000 [========>.....................] - ETA: 1:08 - loss: 0.3426 - categorical_accuracy: 0.8946
18848/60000 [========>.....................] - ETA: 1:07 - loss: 0.3422 - categorical_accuracy: 0.8947
18880/60000 [========>.....................] - ETA: 1:07 - loss: 0.3420 - categorical_accuracy: 0.8948
18912/60000 [========>.....................] - ETA: 1:07 - loss: 0.3415 - categorical_accuracy: 0.8950
18944/60000 [========>.....................] - ETA: 1:07 - loss: 0.3411 - categorical_accuracy: 0.8951
18976/60000 [========>.....................] - ETA: 1:07 - loss: 0.3406 - categorical_accuracy: 0.8952
19008/60000 [========>.....................] - ETA: 1:07 - loss: 0.3403 - categorical_accuracy: 0.8953
19040/60000 [========>.....................] - ETA: 1:07 - loss: 0.3399 - categorical_accuracy: 0.8954
19072/60000 [========>.....................] - ETA: 1:07 - loss: 0.3394 - categorical_accuracy: 0.8955
19104/60000 [========>.....................] - ETA: 1:07 - loss: 0.3389 - categorical_accuracy: 0.8957
19136/60000 [========>.....................] - ETA: 1:07 - loss: 0.3387 - categorical_accuracy: 0.8956
19168/60000 [========>.....................] - ETA: 1:07 - loss: 0.3387 - categorical_accuracy: 0.8957
19200/60000 [========>.....................] - ETA: 1:07 - loss: 0.3385 - categorical_accuracy: 0.8957
19232/60000 [========>.....................] - ETA: 1:07 - loss: 0.3383 - categorical_accuracy: 0.8957
19264/60000 [========>.....................] - ETA: 1:07 - loss: 0.3380 - categorical_accuracy: 0.8959
19296/60000 [========>.....................] - ETA: 1:07 - loss: 0.3376 - categorical_accuracy: 0.8959
19328/60000 [========>.....................] - ETA: 1:07 - loss: 0.3372 - categorical_accuracy: 0.8960
19360/60000 [========>.....................] - ETA: 1:07 - loss: 0.3371 - categorical_accuracy: 0.8960
19392/60000 [========>.....................] - ETA: 1:07 - loss: 0.3367 - categorical_accuracy: 0.8962
19424/60000 [========>.....................] - ETA: 1:07 - loss: 0.3362 - categorical_accuracy: 0.8964
19456/60000 [========>.....................] - ETA: 1:06 - loss: 0.3367 - categorical_accuracy: 0.8964
19488/60000 [========>.....................] - ETA: 1:06 - loss: 0.3366 - categorical_accuracy: 0.8964
19520/60000 [========>.....................] - ETA: 1:06 - loss: 0.3362 - categorical_accuracy: 0.8965
19552/60000 [========>.....................] - ETA: 1:06 - loss: 0.3361 - categorical_accuracy: 0.8965
19584/60000 [========>.....................] - ETA: 1:06 - loss: 0.3358 - categorical_accuracy: 0.8967
19616/60000 [========>.....................] - ETA: 1:06 - loss: 0.3354 - categorical_accuracy: 0.8968
19648/60000 [========>.....................] - ETA: 1:06 - loss: 0.3354 - categorical_accuracy: 0.8968
19680/60000 [========>.....................] - ETA: 1:06 - loss: 0.3351 - categorical_accuracy: 0.8969
19712/60000 [========>.....................] - ETA: 1:06 - loss: 0.3347 - categorical_accuracy: 0.8971
19744/60000 [========>.....................] - ETA: 1:06 - loss: 0.3344 - categorical_accuracy: 0.8972
19776/60000 [========>.....................] - ETA: 1:06 - loss: 0.3341 - categorical_accuracy: 0.8972
19808/60000 [========>.....................] - ETA: 1:06 - loss: 0.3338 - categorical_accuracy: 0.8973
19872/60000 [========>.....................] - ETA: 1:06 - loss: 0.3335 - categorical_accuracy: 0.8973
19904/60000 [========>.....................] - ETA: 1:06 - loss: 0.3330 - categorical_accuracy: 0.8975
19936/60000 [========>.....................] - ETA: 1:06 - loss: 0.3329 - categorical_accuracy: 0.8975
19968/60000 [========>.....................] - ETA: 1:06 - loss: 0.3325 - categorical_accuracy: 0.8976
20000/60000 [=========>....................] - ETA: 1:06 - loss: 0.3320 - categorical_accuracy: 0.8978
20032/60000 [=========>....................] - ETA: 1:06 - loss: 0.3324 - categorical_accuracy: 0.8978
20064/60000 [=========>....................] - ETA: 1:05 - loss: 0.3320 - categorical_accuracy: 0.8979
20096/60000 [=========>....................] - ETA: 1:05 - loss: 0.3315 - categorical_accuracy: 0.8981
20128/60000 [=========>....................] - ETA: 1:05 - loss: 0.3313 - categorical_accuracy: 0.8982
20160/60000 [=========>....................] - ETA: 1:05 - loss: 0.3308 - categorical_accuracy: 0.8983
20192/60000 [=========>....................] - ETA: 1:05 - loss: 0.3307 - categorical_accuracy: 0.8983
20224/60000 [=========>....................] - ETA: 1:05 - loss: 0.3304 - categorical_accuracy: 0.8984
20256/60000 [=========>....................] - ETA: 1:05 - loss: 0.3303 - categorical_accuracy: 0.8984
20288/60000 [=========>....................] - ETA: 1:05 - loss: 0.3298 - categorical_accuracy: 0.8986
20320/60000 [=========>....................] - ETA: 1:05 - loss: 0.3297 - categorical_accuracy: 0.8987
20352/60000 [=========>....................] - ETA: 1:05 - loss: 0.3292 - categorical_accuracy: 0.8988
20384/60000 [=========>....................] - ETA: 1:05 - loss: 0.3290 - categorical_accuracy: 0.8989
20416/60000 [=========>....................] - ETA: 1:05 - loss: 0.3286 - categorical_accuracy: 0.8990
20448/60000 [=========>....................] - ETA: 1:05 - loss: 0.3283 - categorical_accuracy: 0.8990
20480/60000 [=========>....................] - ETA: 1:05 - loss: 0.3279 - categorical_accuracy: 0.8992
20512/60000 [=========>....................] - ETA: 1:05 - loss: 0.3276 - categorical_accuracy: 0.8992
20544/60000 [=========>....................] - ETA: 1:05 - loss: 0.3272 - categorical_accuracy: 0.8994
20576/60000 [=========>....................] - ETA: 1:05 - loss: 0.3271 - categorical_accuracy: 0.8994
20640/60000 [=========>....................] - ETA: 1:04 - loss: 0.3264 - categorical_accuracy: 0.8996
20672/60000 [=========>....................] - ETA: 1:04 - loss: 0.3260 - categorical_accuracy: 0.8997
20704/60000 [=========>....................] - ETA: 1:04 - loss: 0.3255 - categorical_accuracy: 0.8999
20736/60000 [=========>....................] - ETA: 1:04 - loss: 0.3252 - categorical_accuracy: 0.9000
20768/60000 [=========>....................] - ETA: 1:04 - loss: 0.3251 - categorical_accuracy: 0.9000
20800/60000 [=========>....................] - ETA: 1:04 - loss: 0.3248 - categorical_accuracy: 0.9001
20832/60000 [=========>....................] - ETA: 1:04 - loss: 0.3244 - categorical_accuracy: 0.9002
20864/60000 [=========>....................] - ETA: 1:04 - loss: 0.3239 - categorical_accuracy: 0.9004
20896/60000 [=========>....................] - ETA: 1:04 - loss: 0.3239 - categorical_accuracy: 0.9004
20928/60000 [=========>....................] - ETA: 1:04 - loss: 0.3237 - categorical_accuracy: 0.9004
20960/60000 [=========>....................] - ETA: 1:04 - loss: 0.3234 - categorical_accuracy: 0.9005
20992/60000 [=========>....................] - ETA: 1:04 - loss: 0.3231 - categorical_accuracy: 0.9006
21024/60000 [=========>....................] - ETA: 1:04 - loss: 0.3228 - categorical_accuracy: 0.9007
21056/60000 [=========>....................] - ETA: 1:04 - loss: 0.3224 - categorical_accuracy: 0.9008
21120/60000 [=========>....................] - ETA: 1:04 - loss: 0.3216 - categorical_accuracy: 0.9010
21152/60000 [=========>....................] - ETA: 1:04 - loss: 0.3216 - categorical_accuracy: 0.9010
21184/60000 [=========>....................] - ETA: 1:04 - loss: 0.3215 - categorical_accuracy: 0.9011
21216/60000 [=========>....................] - ETA: 1:04 - loss: 0.3211 - categorical_accuracy: 0.9012
21248/60000 [=========>....................] - ETA: 1:04 - loss: 0.3211 - categorical_accuracy: 0.9012
21312/60000 [=========>....................] - ETA: 1:03 - loss: 0.3204 - categorical_accuracy: 0.9014
21344/60000 [=========>....................] - ETA: 1:03 - loss: 0.3203 - categorical_accuracy: 0.9014
21376/60000 [=========>....................] - ETA: 1:03 - loss: 0.3200 - categorical_accuracy: 0.9014
21408/60000 [=========>....................] - ETA: 1:03 - loss: 0.3196 - categorical_accuracy: 0.9016
21440/60000 [=========>....................] - ETA: 1:03 - loss: 0.3193 - categorical_accuracy: 0.9016
21472/60000 [=========>....................] - ETA: 1:03 - loss: 0.3189 - categorical_accuracy: 0.9017
21536/60000 [=========>....................] - ETA: 1:03 - loss: 0.3181 - categorical_accuracy: 0.9020
21568/60000 [=========>....................] - ETA: 1:03 - loss: 0.3179 - categorical_accuracy: 0.9020
21600/60000 [=========>....................] - ETA: 1:03 - loss: 0.3177 - categorical_accuracy: 0.9021
21632/60000 [=========>....................] - ETA: 1:03 - loss: 0.3172 - categorical_accuracy: 0.9022
21664/60000 [=========>....................] - ETA: 1:03 - loss: 0.3170 - categorical_accuracy: 0.9023
21696/60000 [=========>....................] - ETA: 1:03 - loss: 0.3165 - categorical_accuracy: 0.9025
21728/60000 [=========>....................] - ETA: 1:03 - loss: 0.3161 - categorical_accuracy: 0.9026
21760/60000 [=========>....................] - ETA: 1:03 - loss: 0.3160 - categorical_accuracy: 0.9026
21792/60000 [=========>....................] - ETA: 1:03 - loss: 0.3157 - categorical_accuracy: 0.9026
21824/60000 [=========>....................] - ETA: 1:03 - loss: 0.3154 - categorical_accuracy: 0.9028
21856/60000 [=========>....................] - ETA: 1:03 - loss: 0.3150 - categorical_accuracy: 0.9029
21888/60000 [=========>....................] - ETA: 1:02 - loss: 0.3149 - categorical_accuracy: 0.9029
21920/60000 [=========>....................] - ETA: 1:02 - loss: 0.3145 - categorical_accuracy: 0.9031
21952/60000 [=========>....................] - ETA: 1:02 - loss: 0.3142 - categorical_accuracy: 0.9032
21984/60000 [=========>....................] - ETA: 1:02 - loss: 0.3138 - categorical_accuracy: 0.9033
22016/60000 [==========>...................] - ETA: 1:02 - loss: 0.3138 - categorical_accuracy: 0.9033
22048/60000 [==========>...................] - ETA: 1:02 - loss: 0.3138 - categorical_accuracy: 0.9033
22080/60000 [==========>...................] - ETA: 1:02 - loss: 0.3135 - categorical_accuracy: 0.9034
22112/60000 [==========>...................] - ETA: 1:02 - loss: 0.3132 - categorical_accuracy: 0.9035
22176/60000 [==========>...................] - ETA: 1:02 - loss: 0.3129 - categorical_accuracy: 0.9036
22208/60000 [==========>...................] - ETA: 1:02 - loss: 0.3128 - categorical_accuracy: 0.9036
22240/60000 [==========>...................] - ETA: 1:02 - loss: 0.3132 - categorical_accuracy: 0.9036
22272/60000 [==========>...................] - ETA: 1:02 - loss: 0.3128 - categorical_accuracy: 0.9037
22304/60000 [==========>...................] - ETA: 1:02 - loss: 0.3127 - categorical_accuracy: 0.9037
22336/60000 [==========>...................] - ETA: 1:02 - loss: 0.3123 - categorical_accuracy: 0.9038
22368/60000 [==========>...................] - ETA: 1:02 - loss: 0.3122 - categorical_accuracy: 0.9038
22432/60000 [==========>...................] - ETA: 1:02 - loss: 0.3118 - categorical_accuracy: 0.9039
22464/60000 [==========>...................] - ETA: 1:01 - loss: 0.3116 - categorical_accuracy: 0.9040
22528/60000 [==========>...................] - ETA: 1:01 - loss: 0.3116 - categorical_accuracy: 0.9040
22560/60000 [==========>...................] - ETA: 1:01 - loss: 0.3115 - categorical_accuracy: 0.9041
22624/60000 [==========>...................] - ETA: 1:01 - loss: 0.3109 - categorical_accuracy: 0.9043
22656/60000 [==========>...................] - ETA: 1:01 - loss: 0.3108 - categorical_accuracy: 0.9043
22688/60000 [==========>...................] - ETA: 1:01 - loss: 0.3105 - categorical_accuracy: 0.9044
22720/60000 [==========>...................] - ETA: 1:01 - loss: 0.3103 - categorical_accuracy: 0.9044
22752/60000 [==========>...................] - ETA: 1:01 - loss: 0.3101 - categorical_accuracy: 0.9044
22784/60000 [==========>...................] - ETA: 1:01 - loss: 0.3097 - categorical_accuracy: 0.9046
22816/60000 [==========>...................] - ETA: 1:01 - loss: 0.3098 - categorical_accuracy: 0.9045
22848/60000 [==========>...................] - ETA: 1:01 - loss: 0.3094 - categorical_accuracy: 0.9047
22880/60000 [==========>...................] - ETA: 1:01 - loss: 0.3093 - categorical_accuracy: 0.9047
22912/60000 [==========>...................] - ETA: 1:01 - loss: 0.3089 - categorical_accuracy: 0.9048
22944/60000 [==========>...................] - ETA: 1:01 - loss: 0.3091 - categorical_accuracy: 0.9049
22976/60000 [==========>...................] - ETA: 1:01 - loss: 0.3088 - categorical_accuracy: 0.9049
23008/60000 [==========>...................] - ETA: 1:01 - loss: 0.3085 - categorical_accuracy: 0.9050
23040/60000 [==========>...................] - ETA: 1:01 - loss: 0.3082 - categorical_accuracy: 0.9051
23072/60000 [==========>...................] - ETA: 1:00 - loss: 0.3078 - categorical_accuracy: 0.9052
23104/60000 [==========>...................] - ETA: 1:00 - loss: 0.3076 - categorical_accuracy: 0.9053
23136/60000 [==========>...................] - ETA: 1:00 - loss: 0.3073 - categorical_accuracy: 0.9053
23168/60000 [==========>...................] - ETA: 1:00 - loss: 0.3071 - categorical_accuracy: 0.9054
23200/60000 [==========>...................] - ETA: 1:00 - loss: 0.3069 - categorical_accuracy: 0.9055
23232/60000 [==========>...................] - ETA: 1:00 - loss: 0.3067 - categorical_accuracy: 0.9055
23264/60000 [==========>...................] - ETA: 1:00 - loss: 0.3065 - categorical_accuracy: 0.9056
23296/60000 [==========>...................] - ETA: 1:00 - loss: 0.3062 - categorical_accuracy: 0.9057
23328/60000 [==========>...................] - ETA: 1:00 - loss: 0.3058 - categorical_accuracy: 0.9058
23360/60000 [==========>...................] - ETA: 1:00 - loss: 0.3057 - categorical_accuracy: 0.9059
23392/60000 [==========>...................] - ETA: 1:00 - loss: 0.3054 - categorical_accuracy: 0.9060
23424/60000 [==========>...................] - ETA: 1:00 - loss: 0.3053 - categorical_accuracy: 0.9060
23456/60000 [==========>...................] - ETA: 1:00 - loss: 0.3050 - categorical_accuracy: 0.9061
23488/60000 [==========>...................] - ETA: 1:00 - loss: 0.3048 - categorical_accuracy: 0.9061
23520/60000 [==========>...................] - ETA: 1:00 - loss: 0.3047 - categorical_accuracy: 0.9062
23552/60000 [==========>...................] - ETA: 1:00 - loss: 0.3044 - categorical_accuracy: 0.9063
23584/60000 [==========>...................] - ETA: 1:00 - loss: 0.3042 - categorical_accuracy: 0.9063
23616/60000 [==========>...................] - ETA: 1:00 - loss: 0.3039 - categorical_accuracy: 0.9064
23648/60000 [==========>...................] - ETA: 1:00 - loss: 0.3035 - categorical_accuracy: 0.9065
23712/60000 [==========>...................] - ETA: 59s - loss: 0.3028 - categorical_accuracy: 0.9068 
23744/60000 [==========>...................] - ETA: 59s - loss: 0.3028 - categorical_accuracy: 0.9068
23776/60000 [==========>...................] - ETA: 59s - loss: 0.3025 - categorical_accuracy: 0.9069
23808/60000 [==========>...................] - ETA: 59s - loss: 0.3024 - categorical_accuracy: 0.9069
23872/60000 [==========>...................] - ETA: 59s - loss: 0.3020 - categorical_accuracy: 0.9070
23904/60000 [==========>...................] - ETA: 59s - loss: 0.3016 - categorical_accuracy: 0.9072
23936/60000 [==========>...................] - ETA: 59s - loss: 0.3013 - categorical_accuracy: 0.9073
24000/60000 [===========>..................] - ETA: 59s - loss: 0.3009 - categorical_accuracy: 0.9074
24032/60000 [===========>..................] - ETA: 59s - loss: 0.3008 - categorical_accuracy: 0.9074
24064/60000 [===========>..................] - ETA: 59s - loss: 0.3005 - categorical_accuracy: 0.9075
24128/60000 [===========>..................] - ETA: 59s - loss: 0.3000 - categorical_accuracy: 0.9077
24192/60000 [===========>..................] - ETA: 59s - loss: 0.2993 - categorical_accuracy: 0.9079
24224/60000 [===========>..................] - ETA: 59s - loss: 0.2990 - categorical_accuracy: 0.9080
24256/60000 [===========>..................] - ETA: 58s - loss: 0.2988 - categorical_accuracy: 0.9080
24288/60000 [===========>..................] - ETA: 58s - loss: 0.2987 - categorical_accuracy: 0.9081
24320/60000 [===========>..................] - ETA: 58s - loss: 0.2984 - categorical_accuracy: 0.9081
24352/60000 [===========>..................] - ETA: 58s - loss: 0.2982 - categorical_accuracy: 0.9082
24384/60000 [===========>..................] - ETA: 58s - loss: 0.2980 - categorical_accuracy: 0.9083
24416/60000 [===========>..................] - ETA: 58s - loss: 0.2977 - categorical_accuracy: 0.9083
24448/60000 [===========>..................] - ETA: 58s - loss: 0.2973 - categorical_accuracy: 0.9085
24480/60000 [===========>..................] - ETA: 58s - loss: 0.2978 - categorical_accuracy: 0.9084
24512/60000 [===========>..................] - ETA: 58s - loss: 0.2979 - categorical_accuracy: 0.9084
24544/60000 [===========>..................] - ETA: 58s - loss: 0.2975 - categorical_accuracy: 0.9085
24576/60000 [===========>..................] - ETA: 58s - loss: 0.2971 - categorical_accuracy: 0.9087
24608/60000 [===========>..................] - ETA: 58s - loss: 0.2969 - categorical_accuracy: 0.9087
24640/60000 [===========>..................] - ETA: 58s - loss: 0.2966 - categorical_accuracy: 0.9088
24672/60000 [===========>..................] - ETA: 58s - loss: 0.2962 - categorical_accuracy: 0.9089
24704/60000 [===========>..................] - ETA: 58s - loss: 0.2963 - categorical_accuracy: 0.9089
24736/60000 [===========>..................] - ETA: 58s - loss: 0.2961 - categorical_accuracy: 0.9089
24768/60000 [===========>..................] - ETA: 58s - loss: 0.2958 - categorical_accuracy: 0.9090
24800/60000 [===========>..................] - ETA: 58s - loss: 0.2954 - categorical_accuracy: 0.9092
24832/60000 [===========>..................] - ETA: 58s - loss: 0.2953 - categorical_accuracy: 0.9091
24864/60000 [===========>..................] - ETA: 58s - loss: 0.2950 - categorical_accuracy: 0.9092
24896/60000 [===========>..................] - ETA: 57s - loss: 0.2948 - categorical_accuracy: 0.9092
24928/60000 [===========>..................] - ETA: 57s - loss: 0.2945 - categorical_accuracy: 0.9093
24960/60000 [===========>..................] - ETA: 57s - loss: 0.2942 - categorical_accuracy: 0.9094
24992/60000 [===========>..................] - ETA: 57s - loss: 0.2940 - categorical_accuracy: 0.9095
25024/60000 [===========>..................] - ETA: 57s - loss: 0.2939 - categorical_accuracy: 0.9095
25056/60000 [===========>..................] - ETA: 57s - loss: 0.2937 - categorical_accuracy: 0.9096
25088/60000 [===========>..................] - ETA: 57s - loss: 0.2935 - categorical_accuracy: 0.9097
25120/60000 [===========>..................] - ETA: 57s - loss: 0.2932 - categorical_accuracy: 0.9098
25152/60000 [===========>..................] - ETA: 57s - loss: 0.2929 - categorical_accuracy: 0.9098
25184/60000 [===========>..................] - ETA: 57s - loss: 0.2926 - categorical_accuracy: 0.9099
25216/60000 [===========>..................] - ETA: 57s - loss: 0.2923 - categorical_accuracy: 0.9101
25248/60000 [===========>..................] - ETA: 57s - loss: 0.2920 - categorical_accuracy: 0.9102
25280/60000 [===========>..................] - ETA: 57s - loss: 0.2917 - categorical_accuracy: 0.9102
25312/60000 [===========>..................] - ETA: 57s - loss: 0.2916 - categorical_accuracy: 0.9103
25344/60000 [===========>..................] - ETA: 57s - loss: 0.2913 - categorical_accuracy: 0.9104
25376/60000 [===========>..................] - ETA: 57s - loss: 0.2910 - categorical_accuracy: 0.9104
25408/60000 [===========>..................] - ETA: 57s - loss: 0.2908 - categorical_accuracy: 0.9105
25440/60000 [===========>..................] - ETA: 57s - loss: 0.2907 - categorical_accuracy: 0.9105
25472/60000 [===========>..................] - ETA: 57s - loss: 0.2906 - categorical_accuracy: 0.9105
25504/60000 [===========>..................] - ETA: 56s - loss: 0.2903 - categorical_accuracy: 0.9106
25536/60000 [===========>..................] - ETA: 56s - loss: 0.2903 - categorical_accuracy: 0.9107
25568/60000 [===========>..................] - ETA: 56s - loss: 0.2899 - categorical_accuracy: 0.9108
25600/60000 [===========>..................] - ETA: 56s - loss: 0.2896 - categorical_accuracy: 0.9109
25664/60000 [===========>..................] - ETA: 56s - loss: 0.2891 - categorical_accuracy: 0.9110
25696/60000 [===========>..................] - ETA: 56s - loss: 0.2890 - categorical_accuracy: 0.9111
25728/60000 [===========>..................] - ETA: 56s - loss: 0.2890 - categorical_accuracy: 0.9111
25760/60000 [===========>..................] - ETA: 56s - loss: 0.2888 - categorical_accuracy: 0.9112
25792/60000 [===========>..................] - ETA: 56s - loss: 0.2888 - categorical_accuracy: 0.9113
25824/60000 [===========>..................] - ETA: 56s - loss: 0.2891 - categorical_accuracy: 0.9112
25856/60000 [===========>..................] - ETA: 56s - loss: 0.2889 - categorical_accuracy: 0.9112
25920/60000 [===========>..................] - ETA: 56s - loss: 0.2883 - categorical_accuracy: 0.9114
25952/60000 [===========>..................] - ETA: 56s - loss: 0.2882 - categorical_accuracy: 0.9115
25984/60000 [===========>..................] - ETA: 56s - loss: 0.2879 - categorical_accuracy: 0.9116
26016/60000 [============>.................] - ETA: 56s - loss: 0.2876 - categorical_accuracy: 0.9116
26048/60000 [============>.................] - ETA: 56s - loss: 0.2874 - categorical_accuracy: 0.9117
26080/60000 [============>.................] - ETA: 56s - loss: 0.2872 - categorical_accuracy: 0.9117
26112/60000 [============>.................] - ETA: 55s - loss: 0.2869 - categorical_accuracy: 0.9118
26144/60000 [============>.................] - ETA: 55s - loss: 0.2866 - categorical_accuracy: 0.9119
26176/60000 [============>.................] - ETA: 55s - loss: 0.2865 - categorical_accuracy: 0.9119
26208/60000 [============>.................] - ETA: 55s - loss: 0.2862 - categorical_accuracy: 0.9120
26240/60000 [============>.................] - ETA: 55s - loss: 0.2860 - categorical_accuracy: 0.9121
26272/60000 [============>.................] - ETA: 55s - loss: 0.2856 - categorical_accuracy: 0.9122
26304/60000 [============>.................] - ETA: 55s - loss: 0.2855 - categorical_accuracy: 0.9122
26336/60000 [============>.................] - ETA: 55s - loss: 0.2853 - categorical_accuracy: 0.9123
26368/60000 [============>.................] - ETA: 55s - loss: 0.2851 - categorical_accuracy: 0.9124
26400/60000 [============>.................] - ETA: 55s - loss: 0.2850 - categorical_accuracy: 0.9124
26432/60000 [============>.................] - ETA: 55s - loss: 0.2851 - categorical_accuracy: 0.9124
26464/60000 [============>.................] - ETA: 55s - loss: 0.2848 - categorical_accuracy: 0.9125
26528/60000 [============>.................] - ETA: 55s - loss: 0.2847 - categorical_accuracy: 0.9125
26592/60000 [============>.................] - ETA: 55s - loss: 0.2848 - categorical_accuracy: 0.9126
26624/60000 [============>.................] - ETA: 55s - loss: 0.2845 - categorical_accuracy: 0.9127
26688/60000 [============>.................] - ETA: 54s - loss: 0.2840 - categorical_accuracy: 0.9129
26720/60000 [============>.................] - ETA: 54s - loss: 0.2837 - categorical_accuracy: 0.9130
26784/60000 [============>.................] - ETA: 54s - loss: 0.2832 - categorical_accuracy: 0.9132
26816/60000 [============>.................] - ETA: 54s - loss: 0.2830 - categorical_accuracy: 0.9132
26848/60000 [============>.................] - ETA: 54s - loss: 0.2827 - categorical_accuracy: 0.9133
26880/60000 [============>.................] - ETA: 54s - loss: 0.2825 - categorical_accuracy: 0.9134
26912/60000 [============>.................] - ETA: 54s - loss: 0.2824 - categorical_accuracy: 0.9134
26944/60000 [============>.................] - ETA: 54s - loss: 0.2821 - categorical_accuracy: 0.9135
26976/60000 [============>.................] - ETA: 54s - loss: 0.2820 - categorical_accuracy: 0.9136
27008/60000 [============>.................] - ETA: 54s - loss: 0.2817 - categorical_accuracy: 0.9137
27040/60000 [============>.................] - ETA: 54s - loss: 0.2815 - categorical_accuracy: 0.9137
27072/60000 [============>.................] - ETA: 54s - loss: 0.2814 - categorical_accuracy: 0.9137
27104/60000 [============>.................] - ETA: 54s - loss: 0.2814 - categorical_accuracy: 0.9137
27136/60000 [============>.................] - ETA: 54s - loss: 0.2811 - categorical_accuracy: 0.9138
27200/60000 [============>.................] - ETA: 54s - loss: 0.2810 - categorical_accuracy: 0.9138
27232/60000 [============>.................] - ETA: 54s - loss: 0.2807 - categorical_accuracy: 0.9139
27296/60000 [============>.................] - ETA: 53s - loss: 0.2802 - categorical_accuracy: 0.9141
27328/60000 [============>.................] - ETA: 53s - loss: 0.2799 - categorical_accuracy: 0.9142
27360/60000 [============>.................] - ETA: 53s - loss: 0.2796 - categorical_accuracy: 0.9143
27392/60000 [============>.................] - ETA: 53s - loss: 0.2794 - categorical_accuracy: 0.9144
27424/60000 [============>.................] - ETA: 53s - loss: 0.2791 - categorical_accuracy: 0.9144
27456/60000 [============>.................] - ETA: 53s - loss: 0.2788 - categorical_accuracy: 0.9145
27488/60000 [============>.................] - ETA: 53s - loss: 0.2785 - categorical_accuracy: 0.9146
27520/60000 [============>.................] - ETA: 53s - loss: 0.2783 - categorical_accuracy: 0.9147
27552/60000 [============>.................] - ETA: 53s - loss: 0.2782 - categorical_accuracy: 0.9147
27584/60000 [============>.................] - ETA: 53s - loss: 0.2782 - categorical_accuracy: 0.9148
27616/60000 [============>.................] - ETA: 53s - loss: 0.2780 - categorical_accuracy: 0.9149
27648/60000 [============>.................] - ETA: 53s - loss: 0.2777 - categorical_accuracy: 0.9150
27680/60000 [============>.................] - ETA: 53s - loss: 0.2774 - categorical_accuracy: 0.9151
27712/60000 [============>.................] - ETA: 53s - loss: 0.2773 - categorical_accuracy: 0.9152
27744/60000 [============>.................] - ETA: 53s - loss: 0.2772 - categorical_accuracy: 0.9152
27776/60000 [============>.................] - ETA: 53s - loss: 0.2770 - categorical_accuracy: 0.9153
27808/60000 [============>.................] - ETA: 53s - loss: 0.2769 - categorical_accuracy: 0.9153
27840/60000 [============>.................] - ETA: 53s - loss: 0.2766 - categorical_accuracy: 0.9153
27872/60000 [============>.................] - ETA: 53s - loss: 0.2764 - categorical_accuracy: 0.9154
27904/60000 [============>.................] - ETA: 52s - loss: 0.2761 - categorical_accuracy: 0.9155
27936/60000 [============>.................] - ETA: 52s - loss: 0.2760 - categorical_accuracy: 0.9156
27968/60000 [============>.................] - ETA: 52s - loss: 0.2758 - categorical_accuracy: 0.9156
28000/60000 [=============>................] - ETA: 52s - loss: 0.2756 - categorical_accuracy: 0.9156
28032/60000 [=============>................] - ETA: 52s - loss: 0.2755 - categorical_accuracy: 0.9157
28064/60000 [=============>................] - ETA: 52s - loss: 0.2756 - categorical_accuracy: 0.9157
28096/60000 [=============>................] - ETA: 52s - loss: 0.2754 - categorical_accuracy: 0.9158
28128/60000 [=============>................] - ETA: 52s - loss: 0.2753 - categorical_accuracy: 0.9157
28160/60000 [=============>................] - ETA: 52s - loss: 0.2751 - categorical_accuracy: 0.9158
28192/60000 [=============>................] - ETA: 52s - loss: 0.2748 - categorical_accuracy: 0.9159
28224/60000 [=============>................] - ETA: 52s - loss: 0.2746 - categorical_accuracy: 0.9159
28256/60000 [=============>................] - ETA: 52s - loss: 0.2744 - categorical_accuracy: 0.9160
28288/60000 [=============>................] - ETA: 52s - loss: 0.2742 - categorical_accuracy: 0.9160
28320/60000 [=============>................] - ETA: 52s - loss: 0.2741 - categorical_accuracy: 0.9161
28352/60000 [=============>................] - ETA: 52s - loss: 0.2738 - categorical_accuracy: 0.9161
28416/60000 [=============>................] - ETA: 52s - loss: 0.2737 - categorical_accuracy: 0.9162
28448/60000 [=============>................] - ETA: 52s - loss: 0.2737 - categorical_accuracy: 0.9162
28480/60000 [=============>................] - ETA: 52s - loss: 0.2735 - categorical_accuracy: 0.9163
28512/60000 [=============>................] - ETA: 51s - loss: 0.2734 - categorical_accuracy: 0.9163
28544/60000 [=============>................] - ETA: 51s - loss: 0.2732 - categorical_accuracy: 0.9163
28576/60000 [=============>................] - ETA: 51s - loss: 0.2730 - categorical_accuracy: 0.9164
28608/60000 [=============>................] - ETA: 51s - loss: 0.2728 - categorical_accuracy: 0.9164
28640/60000 [=============>................] - ETA: 51s - loss: 0.2726 - categorical_accuracy: 0.9165
28672/60000 [=============>................] - ETA: 51s - loss: 0.2724 - categorical_accuracy: 0.9165
28736/60000 [=============>................] - ETA: 51s - loss: 0.2724 - categorical_accuracy: 0.9165
28800/60000 [=============>................] - ETA: 51s - loss: 0.2723 - categorical_accuracy: 0.9165
28832/60000 [=============>................] - ETA: 51s - loss: 0.2720 - categorical_accuracy: 0.9166
28864/60000 [=============>................] - ETA: 51s - loss: 0.2719 - categorical_accuracy: 0.9166
28896/60000 [=============>................] - ETA: 51s - loss: 0.2717 - categorical_accuracy: 0.9167
28928/60000 [=============>................] - ETA: 51s - loss: 0.2714 - categorical_accuracy: 0.9168
28960/60000 [=============>................] - ETA: 51s - loss: 0.2712 - categorical_accuracy: 0.9168
28992/60000 [=============>................] - ETA: 51s - loss: 0.2711 - categorical_accuracy: 0.9168
29024/60000 [=============>................] - ETA: 51s - loss: 0.2710 - categorical_accuracy: 0.9169
29056/60000 [=============>................] - ETA: 51s - loss: 0.2709 - categorical_accuracy: 0.9169
29088/60000 [=============>................] - ETA: 50s - loss: 0.2707 - categorical_accuracy: 0.9169
29120/60000 [=============>................] - ETA: 50s - loss: 0.2706 - categorical_accuracy: 0.9170
29152/60000 [=============>................] - ETA: 50s - loss: 0.2705 - categorical_accuracy: 0.9170
29184/60000 [=============>................] - ETA: 50s - loss: 0.2703 - categorical_accuracy: 0.9170
29216/60000 [=============>................] - ETA: 50s - loss: 0.2702 - categorical_accuracy: 0.9171
29248/60000 [=============>................] - ETA: 50s - loss: 0.2699 - categorical_accuracy: 0.9172
29280/60000 [=============>................] - ETA: 50s - loss: 0.2698 - categorical_accuracy: 0.9172
29312/60000 [=============>................] - ETA: 50s - loss: 0.2696 - categorical_accuracy: 0.9173
29344/60000 [=============>................] - ETA: 50s - loss: 0.2694 - categorical_accuracy: 0.9174
29376/60000 [=============>................] - ETA: 50s - loss: 0.2692 - categorical_accuracy: 0.9174
29440/60000 [=============>................] - ETA: 50s - loss: 0.2688 - categorical_accuracy: 0.9175
29504/60000 [=============>................] - ETA: 50s - loss: 0.2685 - categorical_accuracy: 0.9176
29536/60000 [=============>................] - ETA: 50s - loss: 0.2683 - categorical_accuracy: 0.9177
29568/60000 [=============>................] - ETA: 50s - loss: 0.2681 - categorical_accuracy: 0.9178
29632/60000 [=============>................] - ETA: 50s - loss: 0.2676 - categorical_accuracy: 0.9180
29664/60000 [=============>................] - ETA: 50s - loss: 0.2673 - categorical_accuracy: 0.9180
29696/60000 [=============>................] - ETA: 49s - loss: 0.2672 - categorical_accuracy: 0.9181
29728/60000 [=============>................] - ETA: 49s - loss: 0.2670 - categorical_accuracy: 0.9182
29792/60000 [=============>................] - ETA: 49s - loss: 0.2668 - categorical_accuracy: 0.9182
29824/60000 [=============>................] - ETA: 49s - loss: 0.2667 - categorical_accuracy: 0.9183
29856/60000 [=============>................] - ETA: 49s - loss: 0.2665 - categorical_accuracy: 0.9183
29888/60000 [=============>................] - ETA: 49s - loss: 0.2663 - categorical_accuracy: 0.9184
29920/60000 [=============>................] - ETA: 49s - loss: 0.2665 - categorical_accuracy: 0.9183
29952/60000 [=============>................] - ETA: 49s - loss: 0.2662 - categorical_accuracy: 0.9184
29984/60000 [=============>................] - ETA: 49s - loss: 0.2662 - categorical_accuracy: 0.9185
30016/60000 [==============>...............] - ETA: 49s - loss: 0.2661 - categorical_accuracy: 0.9185
30048/60000 [==============>...............] - ETA: 49s - loss: 0.2661 - categorical_accuracy: 0.9185
30080/60000 [==============>...............] - ETA: 49s - loss: 0.2660 - categorical_accuracy: 0.9186
30112/60000 [==============>...............] - ETA: 49s - loss: 0.2658 - categorical_accuracy: 0.9186
30144/60000 [==============>...............] - ETA: 49s - loss: 0.2656 - categorical_accuracy: 0.9187
30176/60000 [==============>...............] - ETA: 49s - loss: 0.2654 - categorical_accuracy: 0.9187
30240/60000 [==============>...............] - ETA: 49s - loss: 0.2649 - categorical_accuracy: 0.9189
30304/60000 [==============>...............] - ETA: 48s - loss: 0.2646 - categorical_accuracy: 0.9190
30368/60000 [==============>...............] - ETA: 48s - loss: 0.2642 - categorical_accuracy: 0.9192
30432/60000 [==============>...............] - ETA: 48s - loss: 0.2638 - categorical_accuracy: 0.9193
30464/60000 [==============>...............] - ETA: 48s - loss: 0.2637 - categorical_accuracy: 0.9193
30528/60000 [==============>...............] - ETA: 48s - loss: 0.2634 - categorical_accuracy: 0.9195
30592/60000 [==============>...............] - ETA: 48s - loss: 0.2629 - categorical_accuracy: 0.9196
30624/60000 [==============>...............] - ETA: 48s - loss: 0.2629 - categorical_accuracy: 0.9196
30656/60000 [==============>...............] - ETA: 48s - loss: 0.2627 - categorical_accuracy: 0.9197
30688/60000 [==============>...............] - ETA: 48s - loss: 0.2624 - categorical_accuracy: 0.9198
30720/60000 [==============>...............] - ETA: 48s - loss: 0.2622 - categorical_accuracy: 0.9199
30752/60000 [==============>...............] - ETA: 48s - loss: 0.2620 - categorical_accuracy: 0.9199
30784/60000 [==============>...............] - ETA: 48s - loss: 0.2618 - categorical_accuracy: 0.9200
30816/60000 [==============>...............] - ETA: 48s - loss: 0.2616 - categorical_accuracy: 0.9200
30848/60000 [==============>...............] - ETA: 48s - loss: 0.2615 - categorical_accuracy: 0.9201
30880/60000 [==============>...............] - ETA: 47s - loss: 0.2613 - categorical_accuracy: 0.9201
30912/60000 [==============>...............] - ETA: 47s - loss: 0.2612 - categorical_accuracy: 0.9201
30944/60000 [==============>...............] - ETA: 47s - loss: 0.2609 - categorical_accuracy: 0.9201
30976/60000 [==============>...............] - ETA: 47s - loss: 0.2609 - categorical_accuracy: 0.9202
31008/60000 [==============>...............] - ETA: 47s - loss: 0.2608 - categorical_accuracy: 0.9202
31040/60000 [==============>...............] - ETA: 47s - loss: 0.2606 - categorical_accuracy: 0.9202
31072/60000 [==============>...............] - ETA: 47s - loss: 0.2605 - categorical_accuracy: 0.9203
31104/60000 [==============>...............] - ETA: 47s - loss: 0.2605 - categorical_accuracy: 0.9202
31136/60000 [==============>...............] - ETA: 47s - loss: 0.2604 - categorical_accuracy: 0.9203
31168/60000 [==============>...............] - ETA: 47s - loss: 0.2602 - categorical_accuracy: 0.9203
31200/60000 [==============>...............] - ETA: 47s - loss: 0.2601 - categorical_accuracy: 0.9203
31232/60000 [==============>...............] - ETA: 47s - loss: 0.2601 - categorical_accuracy: 0.9203
31296/60000 [==============>...............] - ETA: 47s - loss: 0.2597 - categorical_accuracy: 0.9205
31328/60000 [==============>...............] - ETA: 47s - loss: 0.2595 - categorical_accuracy: 0.9205
31360/60000 [==============>...............] - ETA: 47s - loss: 0.2593 - categorical_accuracy: 0.9206
31392/60000 [==============>...............] - ETA: 47s - loss: 0.2591 - categorical_accuracy: 0.9206
31424/60000 [==============>...............] - ETA: 47s - loss: 0.2589 - categorical_accuracy: 0.9207
31456/60000 [==============>...............] - ETA: 47s - loss: 0.2588 - categorical_accuracy: 0.9207
31488/60000 [==============>...............] - ETA: 46s - loss: 0.2589 - categorical_accuracy: 0.9207
31520/60000 [==============>...............] - ETA: 46s - loss: 0.2588 - categorical_accuracy: 0.9208
31552/60000 [==============>...............] - ETA: 46s - loss: 0.2587 - categorical_accuracy: 0.9208
31584/60000 [==============>...............] - ETA: 46s - loss: 0.2585 - categorical_accuracy: 0.9208
31616/60000 [==============>...............] - ETA: 46s - loss: 0.2587 - categorical_accuracy: 0.9208
31648/60000 [==============>...............] - ETA: 46s - loss: 0.2587 - categorical_accuracy: 0.9207
31680/60000 [==============>...............] - ETA: 46s - loss: 0.2586 - categorical_accuracy: 0.9207
31712/60000 [==============>...............] - ETA: 46s - loss: 0.2586 - categorical_accuracy: 0.9208
31776/60000 [==============>...............] - ETA: 46s - loss: 0.2582 - categorical_accuracy: 0.9209
31840/60000 [==============>...............] - ETA: 46s - loss: 0.2578 - categorical_accuracy: 0.9210
31872/60000 [==============>...............] - ETA: 46s - loss: 0.2577 - categorical_accuracy: 0.9210
31904/60000 [==============>...............] - ETA: 46s - loss: 0.2578 - categorical_accuracy: 0.9210
31936/60000 [==============>...............] - ETA: 46s - loss: 0.2580 - categorical_accuracy: 0.9210
31968/60000 [==============>...............] - ETA: 46s - loss: 0.2577 - categorical_accuracy: 0.9211
32000/60000 [===============>..............] - ETA: 46s - loss: 0.2575 - categorical_accuracy: 0.9212
32032/60000 [===============>..............] - ETA: 46s - loss: 0.2574 - categorical_accuracy: 0.9212
32064/60000 [===============>..............] - ETA: 46s - loss: 0.2572 - categorical_accuracy: 0.9213
32096/60000 [===============>..............] - ETA: 45s - loss: 0.2570 - categorical_accuracy: 0.9214
32128/60000 [===============>..............] - ETA: 45s - loss: 0.2568 - categorical_accuracy: 0.9214
32160/60000 [===============>..............] - ETA: 45s - loss: 0.2567 - categorical_accuracy: 0.9214
32192/60000 [===============>..............] - ETA: 45s - loss: 0.2565 - categorical_accuracy: 0.9215
32224/60000 [===============>..............] - ETA: 45s - loss: 0.2563 - categorical_accuracy: 0.9215
32256/60000 [===============>..............] - ETA: 45s - loss: 0.2560 - categorical_accuracy: 0.9216
32288/60000 [===============>..............] - ETA: 45s - loss: 0.2559 - categorical_accuracy: 0.9216
32320/60000 [===============>..............] - ETA: 45s - loss: 0.2556 - categorical_accuracy: 0.9217
32352/60000 [===============>..............] - ETA: 45s - loss: 0.2554 - categorical_accuracy: 0.9218
32384/60000 [===============>..............] - ETA: 45s - loss: 0.2552 - categorical_accuracy: 0.9219
32416/60000 [===============>..............] - ETA: 45s - loss: 0.2550 - categorical_accuracy: 0.9219
32448/60000 [===============>..............] - ETA: 45s - loss: 0.2549 - categorical_accuracy: 0.9220
32480/60000 [===============>..............] - ETA: 45s - loss: 0.2547 - categorical_accuracy: 0.9220
32544/60000 [===============>..............] - ETA: 45s - loss: 0.2545 - categorical_accuracy: 0.9221
32576/60000 [===============>..............] - ETA: 45s - loss: 0.2543 - categorical_accuracy: 0.9221
32608/60000 [===============>..............] - ETA: 45s - loss: 0.2541 - categorical_accuracy: 0.9222
32640/60000 [===============>..............] - ETA: 45s - loss: 0.2539 - categorical_accuracy: 0.9223
32672/60000 [===============>..............] - ETA: 45s - loss: 0.2538 - categorical_accuracy: 0.9223
32704/60000 [===============>..............] - ETA: 44s - loss: 0.2538 - categorical_accuracy: 0.9223
32736/60000 [===============>..............] - ETA: 44s - loss: 0.2536 - categorical_accuracy: 0.9223
32768/60000 [===============>..............] - ETA: 44s - loss: 0.2534 - categorical_accuracy: 0.9224
32800/60000 [===============>..............] - ETA: 44s - loss: 0.2532 - categorical_accuracy: 0.9225
32832/60000 [===============>..............] - ETA: 44s - loss: 0.2534 - categorical_accuracy: 0.9225
32864/60000 [===============>..............] - ETA: 44s - loss: 0.2533 - categorical_accuracy: 0.9225
32896/60000 [===============>..............] - ETA: 44s - loss: 0.2531 - categorical_accuracy: 0.9226
32928/60000 [===============>..............] - ETA: 44s - loss: 0.2530 - categorical_accuracy: 0.9226
32960/60000 [===============>..............] - ETA: 44s - loss: 0.2528 - categorical_accuracy: 0.9226
32992/60000 [===============>..............] - ETA: 44s - loss: 0.2529 - categorical_accuracy: 0.9226
33024/60000 [===============>..............] - ETA: 44s - loss: 0.2527 - categorical_accuracy: 0.9227
33056/60000 [===============>..............] - ETA: 44s - loss: 0.2527 - categorical_accuracy: 0.9227
33088/60000 [===============>..............] - ETA: 44s - loss: 0.2525 - categorical_accuracy: 0.9228
33152/60000 [===============>..............] - ETA: 44s - loss: 0.2522 - categorical_accuracy: 0.9229
33184/60000 [===============>..............] - ETA: 44s - loss: 0.2523 - categorical_accuracy: 0.9228
33216/60000 [===============>..............] - ETA: 44s - loss: 0.2522 - categorical_accuracy: 0.9229
33248/60000 [===============>..............] - ETA: 44s - loss: 0.2520 - categorical_accuracy: 0.9229
33280/60000 [===============>..............] - ETA: 44s - loss: 0.2523 - categorical_accuracy: 0.9228
33312/60000 [===============>..............] - ETA: 43s - loss: 0.2523 - categorical_accuracy: 0.9229
33344/60000 [===============>..............] - ETA: 43s - loss: 0.2523 - categorical_accuracy: 0.9228
33376/60000 [===============>..............] - ETA: 43s - loss: 0.2522 - categorical_accuracy: 0.9228
33440/60000 [===============>..............] - ETA: 43s - loss: 0.2518 - categorical_accuracy: 0.9230
33472/60000 [===============>..............] - ETA: 43s - loss: 0.2518 - categorical_accuracy: 0.9230
33504/60000 [===============>..............] - ETA: 43s - loss: 0.2516 - categorical_accuracy: 0.9231
33536/60000 [===============>..............] - ETA: 43s - loss: 0.2514 - categorical_accuracy: 0.9231
33568/60000 [===============>..............] - ETA: 43s - loss: 0.2514 - categorical_accuracy: 0.9231
33600/60000 [===============>..............] - ETA: 43s - loss: 0.2512 - categorical_accuracy: 0.9232
33632/60000 [===============>..............] - ETA: 43s - loss: 0.2512 - categorical_accuracy: 0.9231
33696/60000 [===============>..............] - ETA: 43s - loss: 0.2510 - categorical_accuracy: 0.9233
33760/60000 [===============>..............] - ETA: 43s - loss: 0.2506 - categorical_accuracy: 0.9234
33792/60000 [===============>..............] - ETA: 43s - loss: 0.2504 - categorical_accuracy: 0.9235
33824/60000 [===============>..............] - ETA: 43s - loss: 0.2503 - categorical_accuracy: 0.9235
33856/60000 [===============>..............] - ETA: 43s - loss: 0.2504 - categorical_accuracy: 0.9235
33888/60000 [===============>..............] - ETA: 43s - loss: 0.2504 - categorical_accuracy: 0.9235
33920/60000 [===============>..............] - ETA: 42s - loss: 0.2505 - categorical_accuracy: 0.9234
33952/60000 [===============>..............] - ETA: 42s - loss: 0.2504 - categorical_accuracy: 0.9235
33984/60000 [===============>..............] - ETA: 42s - loss: 0.2502 - categorical_accuracy: 0.9235
34016/60000 [================>.............] - ETA: 42s - loss: 0.2501 - categorical_accuracy: 0.9235
34048/60000 [================>.............] - ETA: 42s - loss: 0.2500 - categorical_accuracy: 0.9235
34080/60000 [================>.............] - ETA: 42s - loss: 0.2498 - categorical_accuracy: 0.9236
34112/60000 [================>.............] - ETA: 42s - loss: 0.2500 - categorical_accuracy: 0.9236
34144/60000 [================>.............] - ETA: 42s - loss: 0.2498 - categorical_accuracy: 0.9236
34176/60000 [================>.............] - ETA: 42s - loss: 0.2496 - categorical_accuracy: 0.9237
34208/60000 [================>.............] - ETA: 42s - loss: 0.2494 - categorical_accuracy: 0.9238
34240/60000 [================>.............] - ETA: 42s - loss: 0.2493 - categorical_accuracy: 0.9238
34272/60000 [================>.............] - ETA: 42s - loss: 0.2491 - categorical_accuracy: 0.9239
34304/60000 [================>.............] - ETA: 42s - loss: 0.2489 - categorical_accuracy: 0.9239
34336/60000 [================>.............] - ETA: 42s - loss: 0.2487 - categorical_accuracy: 0.9240
34368/60000 [================>.............] - ETA: 42s - loss: 0.2485 - categorical_accuracy: 0.9241
34400/60000 [================>.............] - ETA: 42s - loss: 0.2484 - categorical_accuracy: 0.9241
34432/60000 [================>.............] - ETA: 42s - loss: 0.2482 - categorical_accuracy: 0.9241
34464/60000 [================>.............] - ETA: 42s - loss: 0.2481 - categorical_accuracy: 0.9242
34496/60000 [================>.............] - ETA: 42s - loss: 0.2479 - categorical_accuracy: 0.9242
34528/60000 [================>.............] - ETA: 41s - loss: 0.2477 - categorical_accuracy: 0.9243
34560/60000 [================>.............] - ETA: 41s - loss: 0.2475 - categorical_accuracy: 0.9243
34592/60000 [================>.............] - ETA: 41s - loss: 0.2474 - categorical_accuracy: 0.9243
34624/60000 [================>.............] - ETA: 41s - loss: 0.2472 - categorical_accuracy: 0.9244
34656/60000 [================>.............] - ETA: 41s - loss: 0.2470 - categorical_accuracy: 0.9245
34688/60000 [================>.............] - ETA: 41s - loss: 0.2469 - categorical_accuracy: 0.9245
34720/60000 [================>.............] - ETA: 41s - loss: 0.2467 - categorical_accuracy: 0.9245
34752/60000 [================>.............] - ETA: 41s - loss: 0.2466 - categorical_accuracy: 0.9246
34784/60000 [================>.............] - ETA: 41s - loss: 0.2464 - categorical_accuracy: 0.9246
34816/60000 [================>.............] - ETA: 41s - loss: 0.2462 - categorical_accuracy: 0.9247
34848/60000 [================>.............] - ETA: 41s - loss: 0.2460 - categorical_accuracy: 0.9247
34880/60000 [================>.............] - ETA: 41s - loss: 0.2459 - categorical_accuracy: 0.9248
34912/60000 [================>.............] - ETA: 41s - loss: 0.2458 - categorical_accuracy: 0.9248
34944/60000 [================>.............] - ETA: 41s - loss: 0.2456 - categorical_accuracy: 0.9249
34976/60000 [================>.............] - ETA: 41s - loss: 0.2454 - categorical_accuracy: 0.9249
35040/60000 [================>.............] - ETA: 41s - loss: 0.2452 - categorical_accuracy: 0.9250
35072/60000 [================>.............] - ETA: 41s - loss: 0.2451 - categorical_accuracy: 0.9250
35104/60000 [================>.............] - ETA: 40s - loss: 0.2449 - categorical_accuracy: 0.9251
35136/60000 [================>.............] - ETA: 40s - loss: 0.2448 - categorical_accuracy: 0.9251
35168/60000 [================>.............] - ETA: 40s - loss: 0.2446 - categorical_accuracy: 0.9251
35200/60000 [================>.............] - ETA: 40s - loss: 0.2450 - categorical_accuracy: 0.9250
35232/60000 [================>.............] - ETA: 40s - loss: 0.2449 - categorical_accuracy: 0.9250
35264/60000 [================>.............] - ETA: 40s - loss: 0.2447 - categorical_accuracy: 0.9251
35296/60000 [================>.............] - ETA: 40s - loss: 0.2445 - categorical_accuracy: 0.9251
35328/60000 [================>.............] - ETA: 40s - loss: 0.2444 - categorical_accuracy: 0.9252
35360/60000 [================>.............] - ETA: 40s - loss: 0.2443 - categorical_accuracy: 0.9252
35392/60000 [================>.............] - ETA: 40s - loss: 0.2441 - categorical_accuracy: 0.9253
35424/60000 [================>.............] - ETA: 40s - loss: 0.2439 - categorical_accuracy: 0.9253
35456/60000 [================>.............] - ETA: 40s - loss: 0.2437 - categorical_accuracy: 0.9254
35488/60000 [================>.............] - ETA: 40s - loss: 0.2435 - categorical_accuracy: 0.9255
35520/60000 [================>.............] - ETA: 40s - loss: 0.2433 - categorical_accuracy: 0.9255
35552/60000 [================>.............] - ETA: 40s - loss: 0.2431 - categorical_accuracy: 0.9256
35584/60000 [================>.............] - ETA: 40s - loss: 0.2431 - categorical_accuracy: 0.9256
35616/60000 [================>.............] - ETA: 40s - loss: 0.2428 - categorical_accuracy: 0.9257
35648/60000 [================>.............] - ETA: 40s - loss: 0.2427 - categorical_accuracy: 0.9257
35712/60000 [================>.............] - ETA: 39s - loss: 0.2424 - categorical_accuracy: 0.9258
35776/60000 [================>.............] - ETA: 39s - loss: 0.2422 - categorical_accuracy: 0.9258
35840/60000 [================>.............] - ETA: 39s - loss: 0.2421 - categorical_accuracy: 0.9258
35872/60000 [================>.............] - ETA: 39s - loss: 0.2421 - categorical_accuracy: 0.9258
35904/60000 [================>.............] - ETA: 39s - loss: 0.2420 - categorical_accuracy: 0.9259
35936/60000 [================>.............] - ETA: 39s - loss: 0.2418 - categorical_accuracy: 0.9259
35968/60000 [================>.............] - ETA: 39s - loss: 0.2416 - categorical_accuracy: 0.9260
36000/60000 [=================>............] - ETA: 39s - loss: 0.2416 - categorical_accuracy: 0.9260
36032/60000 [=================>............] - ETA: 39s - loss: 0.2415 - categorical_accuracy: 0.9260
36064/60000 [=================>............] - ETA: 39s - loss: 0.2413 - categorical_accuracy: 0.9261
36096/60000 [=================>............] - ETA: 39s - loss: 0.2414 - categorical_accuracy: 0.9261
36128/60000 [=================>............] - ETA: 39s - loss: 0.2412 - categorical_accuracy: 0.9262
36160/60000 [=================>............] - ETA: 39s - loss: 0.2410 - categorical_accuracy: 0.9262
36192/60000 [=================>............] - ETA: 39s - loss: 0.2411 - categorical_accuracy: 0.9262
36224/60000 [=================>............] - ETA: 39s - loss: 0.2410 - categorical_accuracy: 0.9263
36256/60000 [=================>............] - ETA: 39s - loss: 0.2408 - categorical_accuracy: 0.9263
36288/60000 [=================>............] - ETA: 39s - loss: 0.2409 - categorical_accuracy: 0.9263
36320/60000 [=================>............] - ETA: 38s - loss: 0.2409 - categorical_accuracy: 0.9263
36352/60000 [=================>............] - ETA: 38s - loss: 0.2407 - categorical_accuracy: 0.9264
36384/60000 [=================>............] - ETA: 38s - loss: 0.2405 - categorical_accuracy: 0.9265
36416/60000 [=================>............] - ETA: 38s - loss: 0.2404 - categorical_accuracy: 0.9265
36448/60000 [=================>............] - ETA: 38s - loss: 0.2404 - categorical_accuracy: 0.9265
36480/60000 [=================>............] - ETA: 38s - loss: 0.2403 - categorical_accuracy: 0.9265
36512/60000 [=================>............] - ETA: 38s - loss: 0.2403 - categorical_accuracy: 0.9265
36544/60000 [=================>............] - ETA: 38s - loss: 0.2404 - categorical_accuracy: 0.9265
36576/60000 [=================>............] - ETA: 38s - loss: 0.2403 - categorical_accuracy: 0.9265
36608/60000 [=================>............] - ETA: 38s - loss: 0.2401 - categorical_accuracy: 0.9266
36640/60000 [=================>............] - ETA: 38s - loss: 0.2400 - categorical_accuracy: 0.9266
36672/60000 [=================>............] - ETA: 38s - loss: 0.2398 - categorical_accuracy: 0.9267
36704/60000 [=================>............] - ETA: 38s - loss: 0.2399 - categorical_accuracy: 0.9267
36736/60000 [=================>............] - ETA: 38s - loss: 0.2399 - categorical_accuracy: 0.9267
36768/60000 [=================>............] - ETA: 38s - loss: 0.2400 - categorical_accuracy: 0.9267
36800/60000 [=================>............] - ETA: 38s - loss: 0.2399 - categorical_accuracy: 0.9267
36832/60000 [=================>............] - ETA: 38s - loss: 0.2398 - categorical_accuracy: 0.9267
36864/60000 [=================>............] - ETA: 38s - loss: 0.2397 - categorical_accuracy: 0.9267
36896/60000 [=================>............] - ETA: 38s - loss: 0.2395 - categorical_accuracy: 0.9268
36928/60000 [=================>............] - ETA: 37s - loss: 0.2396 - categorical_accuracy: 0.9267
36960/60000 [=================>............] - ETA: 37s - loss: 0.2394 - categorical_accuracy: 0.9268
36992/60000 [=================>............] - ETA: 37s - loss: 0.2393 - categorical_accuracy: 0.9268
37024/60000 [=================>............] - ETA: 37s - loss: 0.2391 - categorical_accuracy: 0.9269
37056/60000 [=================>............] - ETA: 37s - loss: 0.2390 - categorical_accuracy: 0.9270
37088/60000 [=================>............] - ETA: 37s - loss: 0.2388 - categorical_accuracy: 0.9270
37120/60000 [=================>............] - ETA: 37s - loss: 0.2386 - categorical_accuracy: 0.9271
37152/60000 [=================>............] - ETA: 37s - loss: 0.2384 - categorical_accuracy: 0.9271
37184/60000 [=================>............] - ETA: 37s - loss: 0.2382 - categorical_accuracy: 0.9272
37248/60000 [=================>............] - ETA: 37s - loss: 0.2379 - categorical_accuracy: 0.9273
37280/60000 [=================>............] - ETA: 37s - loss: 0.2378 - categorical_accuracy: 0.9273
37312/60000 [=================>............] - ETA: 37s - loss: 0.2377 - categorical_accuracy: 0.9274
37344/60000 [=================>............] - ETA: 37s - loss: 0.2376 - categorical_accuracy: 0.9274
37376/60000 [=================>............] - ETA: 37s - loss: 0.2377 - categorical_accuracy: 0.9274
37408/60000 [=================>............] - ETA: 37s - loss: 0.2376 - categorical_accuracy: 0.9274
37440/60000 [=================>............] - ETA: 37s - loss: 0.2374 - categorical_accuracy: 0.9275
37472/60000 [=================>............] - ETA: 37s - loss: 0.2372 - categorical_accuracy: 0.9275
37504/60000 [=================>............] - ETA: 37s - loss: 0.2371 - categorical_accuracy: 0.9276
37536/60000 [=================>............] - ETA: 36s - loss: 0.2370 - categorical_accuracy: 0.9276
37568/60000 [=================>............] - ETA: 36s - loss: 0.2369 - categorical_accuracy: 0.9276
37600/60000 [=================>............] - ETA: 36s - loss: 0.2367 - categorical_accuracy: 0.9277
37632/60000 [=================>............] - ETA: 36s - loss: 0.2368 - categorical_accuracy: 0.9277
37664/60000 [=================>............] - ETA: 36s - loss: 0.2367 - categorical_accuracy: 0.9277
37696/60000 [=================>............] - ETA: 36s - loss: 0.2365 - categorical_accuracy: 0.9278
37728/60000 [=================>............] - ETA: 36s - loss: 0.2363 - categorical_accuracy: 0.9278
37760/60000 [=================>............] - ETA: 36s - loss: 0.2362 - categorical_accuracy: 0.9279
37792/60000 [=================>............] - ETA: 36s - loss: 0.2360 - categorical_accuracy: 0.9279
37824/60000 [=================>............] - ETA: 36s - loss: 0.2359 - categorical_accuracy: 0.9280
37856/60000 [=================>............] - ETA: 36s - loss: 0.2357 - categorical_accuracy: 0.9280
37888/60000 [=================>............] - ETA: 36s - loss: 0.2357 - categorical_accuracy: 0.9281
37920/60000 [=================>............] - ETA: 36s - loss: 0.2355 - categorical_accuracy: 0.9281
37952/60000 [=================>............] - ETA: 36s - loss: 0.2353 - categorical_accuracy: 0.9281
37984/60000 [=================>............] - ETA: 36s - loss: 0.2352 - categorical_accuracy: 0.9281
38016/60000 [==================>...........] - ETA: 36s - loss: 0.2352 - categorical_accuracy: 0.9281
38048/60000 [==================>...........] - ETA: 36s - loss: 0.2350 - categorical_accuracy: 0.9282
38112/60000 [==================>...........] - ETA: 36s - loss: 0.2347 - categorical_accuracy: 0.9283
38144/60000 [==================>...........] - ETA: 35s - loss: 0.2346 - categorical_accuracy: 0.9283
38176/60000 [==================>...........] - ETA: 35s - loss: 0.2345 - categorical_accuracy: 0.9283
38208/60000 [==================>...........] - ETA: 35s - loss: 0.2347 - categorical_accuracy: 0.9283
38272/60000 [==================>...........] - ETA: 35s - loss: 0.2345 - categorical_accuracy: 0.9284
38304/60000 [==================>...........] - ETA: 35s - loss: 0.2346 - categorical_accuracy: 0.9284
38368/60000 [==================>...........] - ETA: 35s - loss: 0.2343 - categorical_accuracy: 0.9284
38400/60000 [==================>...........] - ETA: 35s - loss: 0.2341 - categorical_accuracy: 0.9285
38432/60000 [==================>...........] - ETA: 35s - loss: 0.2339 - categorical_accuracy: 0.9285
38464/60000 [==================>...........] - ETA: 35s - loss: 0.2338 - categorical_accuracy: 0.9286
38496/60000 [==================>...........] - ETA: 35s - loss: 0.2337 - categorical_accuracy: 0.9286
38528/60000 [==================>...........] - ETA: 35s - loss: 0.2335 - categorical_accuracy: 0.9287
38560/60000 [==================>...........] - ETA: 35s - loss: 0.2333 - categorical_accuracy: 0.9288
38592/60000 [==================>...........] - ETA: 35s - loss: 0.2332 - categorical_accuracy: 0.9288
38624/60000 [==================>...........] - ETA: 35s - loss: 0.2331 - categorical_accuracy: 0.9288
38656/60000 [==================>...........] - ETA: 35s - loss: 0.2329 - categorical_accuracy: 0.9289
38720/60000 [==================>...........] - ETA: 35s - loss: 0.2328 - categorical_accuracy: 0.9289
38752/60000 [==================>...........] - ETA: 34s - loss: 0.2327 - categorical_accuracy: 0.9290
38784/60000 [==================>...........] - ETA: 34s - loss: 0.2326 - categorical_accuracy: 0.9290
38816/60000 [==================>...........] - ETA: 34s - loss: 0.2324 - categorical_accuracy: 0.9291
38880/60000 [==================>...........] - ETA: 34s - loss: 0.2321 - categorical_accuracy: 0.9292
38912/60000 [==================>...........] - ETA: 34s - loss: 0.2320 - categorical_accuracy: 0.9292
38944/60000 [==================>...........] - ETA: 34s - loss: 0.2319 - categorical_accuracy: 0.9293
38976/60000 [==================>...........] - ETA: 34s - loss: 0.2318 - categorical_accuracy: 0.9293
39008/60000 [==================>...........] - ETA: 34s - loss: 0.2317 - categorical_accuracy: 0.9293
39040/60000 [==================>...........] - ETA: 34s - loss: 0.2318 - categorical_accuracy: 0.9293
39072/60000 [==================>...........] - ETA: 34s - loss: 0.2318 - categorical_accuracy: 0.9293
39104/60000 [==================>...........] - ETA: 34s - loss: 0.2316 - categorical_accuracy: 0.9293
39136/60000 [==================>...........] - ETA: 34s - loss: 0.2316 - categorical_accuracy: 0.9293
39168/60000 [==================>...........] - ETA: 34s - loss: 0.2314 - categorical_accuracy: 0.9294
39200/60000 [==================>...........] - ETA: 34s - loss: 0.2313 - categorical_accuracy: 0.9294
39232/60000 [==================>...........] - ETA: 34s - loss: 0.2311 - categorical_accuracy: 0.9295
39264/60000 [==================>...........] - ETA: 34s - loss: 0.2310 - categorical_accuracy: 0.9295
39296/60000 [==================>...........] - ETA: 34s - loss: 0.2314 - categorical_accuracy: 0.9295
39328/60000 [==================>...........] - ETA: 34s - loss: 0.2313 - categorical_accuracy: 0.9295
39360/60000 [==================>...........] - ETA: 33s - loss: 0.2312 - categorical_accuracy: 0.9296
39392/60000 [==================>...........] - ETA: 33s - loss: 0.2312 - categorical_accuracy: 0.9296
39424/60000 [==================>...........] - ETA: 33s - loss: 0.2311 - categorical_accuracy: 0.9296
39456/60000 [==================>...........] - ETA: 33s - loss: 0.2310 - categorical_accuracy: 0.9297
39520/60000 [==================>...........] - ETA: 33s - loss: 0.2307 - categorical_accuracy: 0.9297
39552/60000 [==================>...........] - ETA: 33s - loss: 0.2307 - categorical_accuracy: 0.9297
39616/60000 [==================>...........] - ETA: 33s - loss: 0.2306 - categorical_accuracy: 0.9298
39680/60000 [==================>...........] - ETA: 33s - loss: 0.2304 - categorical_accuracy: 0.9298
39744/60000 [==================>...........] - ETA: 33s - loss: 0.2301 - categorical_accuracy: 0.9299
39776/60000 [==================>...........] - ETA: 33s - loss: 0.2300 - categorical_accuracy: 0.9299
39840/60000 [==================>...........] - ETA: 33s - loss: 0.2297 - categorical_accuracy: 0.9300
39872/60000 [==================>...........] - ETA: 33s - loss: 0.2296 - categorical_accuracy: 0.9301
39904/60000 [==================>...........] - ETA: 33s - loss: 0.2296 - categorical_accuracy: 0.9301
39936/60000 [==================>...........] - ETA: 32s - loss: 0.2295 - categorical_accuracy: 0.9301
39968/60000 [==================>...........] - ETA: 32s - loss: 0.2295 - categorical_accuracy: 0.9301
40000/60000 [===================>..........] - ETA: 32s - loss: 0.2294 - categorical_accuracy: 0.9301
40032/60000 [===================>..........] - ETA: 32s - loss: 0.2293 - categorical_accuracy: 0.9301
40064/60000 [===================>..........] - ETA: 32s - loss: 0.2292 - categorical_accuracy: 0.9302
40096/60000 [===================>..........] - ETA: 32s - loss: 0.2291 - categorical_accuracy: 0.9302
40160/60000 [===================>..........] - ETA: 32s - loss: 0.2288 - categorical_accuracy: 0.9303
40192/60000 [===================>..........] - ETA: 32s - loss: 0.2287 - categorical_accuracy: 0.9303
40224/60000 [===================>..........] - ETA: 32s - loss: 0.2286 - categorical_accuracy: 0.9303
40256/60000 [===================>..........] - ETA: 32s - loss: 0.2285 - categorical_accuracy: 0.9303
40288/60000 [===================>..........] - ETA: 32s - loss: 0.2283 - categorical_accuracy: 0.9304
40320/60000 [===================>..........] - ETA: 32s - loss: 0.2282 - categorical_accuracy: 0.9304
40384/60000 [===================>..........] - ETA: 32s - loss: 0.2281 - categorical_accuracy: 0.9304
40416/60000 [===================>..........] - ETA: 32s - loss: 0.2279 - categorical_accuracy: 0.9305
40448/60000 [===================>..........] - ETA: 32s - loss: 0.2278 - categorical_accuracy: 0.9305
40480/60000 [===================>..........] - ETA: 32s - loss: 0.2276 - categorical_accuracy: 0.9306
40512/60000 [===================>..........] - ETA: 32s - loss: 0.2276 - categorical_accuracy: 0.9306
40544/60000 [===================>..........] - ETA: 31s - loss: 0.2275 - categorical_accuracy: 0.9306
40576/60000 [===================>..........] - ETA: 31s - loss: 0.2274 - categorical_accuracy: 0.9306
40608/60000 [===================>..........] - ETA: 31s - loss: 0.2274 - categorical_accuracy: 0.9306
40640/60000 [===================>..........] - ETA: 31s - loss: 0.2272 - categorical_accuracy: 0.9307
40672/60000 [===================>..........] - ETA: 31s - loss: 0.2271 - categorical_accuracy: 0.9307
40704/60000 [===================>..........] - ETA: 31s - loss: 0.2272 - categorical_accuracy: 0.9307
40736/60000 [===================>..........] - ETA: 31s - loss: 0.2273 - categorical_accuracy: 0.9307
40768/60000 [===================>..........] - ETA: 31s - loss: 0.2272 - categorical_accuracy: 0.9307
40832/60000 [===================>..........] - ETA: 31s - loss: 0.2269 - categorical_accuracy: 0.9308
40864/60000 [===================>..........] - ETA: 31s - loss: 0.2268 - categorical_accuracy: 0.9308
40896/60000 [===================>..........] - ETA: 31s - loss: 0.2267 - categorical_accuracy: 0.9309
40928/60000 [===================>..........] - ETA: 31s - loss: 0.2266 - categorical_accuracy: 0.9309
40960/60000 [===================>..........] - ETA: 31s - loss: 0.2265 - categorical_accuracy: 0.9309
40992/60000 [===================>..........] - ETA: 31s - loss: 0.2263 - categorical_accuracy: 0.9310
41024/60000 [===================>..........] - ETA: 31s - loss: 0.2262 - categorical_accuracy: 0.9310
41056/60000 [===================>..........] - ETA: 31s - loss: 0.2261 - categorical_accuracy: 0.9310
41088/60000 [===================>..........] - ETA: 31s - loss: 0.2262 - categorical_accuracy: 0.9310
41120/60000 [===================>..........] - ETA: 31s - loss: 0.2264 - categorical_accuracy: 0.9310
41184/60000 [===================>..........] - ETA: 30s - loss: 0.2262 - categorical_accuracy: 0.9310
41216/60000 [===================>..........] - ETA: 30s - loss: 0.2261 - categorical_accuracy: 0.9310
41248/60000 [===================>..........] - ETA: 30s - loss: 0.2261 - categorical_accuracy: 0.9310
41280/60000 [===================>..........] - ETA: 30s - loss: 0.2261 - categorical_accuracy: 0.9310
41312/60000 [===================>..........] - ETA: 30s - loss: 0.2259 - categorical_accuracy: 0.9311
41344/60000 [===================>..........] - ETA: 30s - loss: 0.2260 - categorical_accuracy: 0.9310
41376/60000 [===================>..........] - ETA: 30s - loss: 0.2259 - categorical_accuracy: 0.9311
41408/60000 [===================>..........] - ETA: 30s - loss: 0.2258 - categorical_accuracy: 0.9311
41440/60000 [===================>..........] - ETA: 30s - loss: 0.2256 - categorical_accuracy: 0.9312
41472/60000 [===================>..........] - ETA: 30s - loss: 0.2255 - categorical_accuracy: 0.9312
41504/60000 [===================>..........] - ETA: 30s - loss: 0.2254 - categorical_accuracy: 0.9313
41568/60000 [===================>..........] - ETA: 30s - loss: 0.2257 - categorical_accuracy: 0.9312
41600/60000 [===================>..........] - ETA: 30s - loss: 0.2256 - categorical_accuracy: 0.9312
41632/60000 [===================>..........] - ETA: 30s - loss: 0.2255 - categorical_accuracy: 0.9313
41664/60000 [===================>..........] - ETA: 30s - loss: 0.2254 - categorical_accuracy: 0.9313
41696/60000 [===================>..........] - ETA: 30s - loss: 0.2253 - categorical_accuracy: 0.9313
41728/60000 [===================>..........] - ETA: 30s - loss: 0.2252 - categorical_accuracy: 0.9314
41792/60000 [===================>..........] - ETA: 29s - loss: 0.2249 - categorical_accuracy: 0.9315
41824/60000 [===================>..........] - ETA: 29s - loss: 0.2247 - categorical_accuracy: 0.9315
41856/60000 [===================>..........] - ETA: 29s - loss: 0.2246 - categorical_accuracy: 0.9316
41888/60000 [===================>..........] - ETA: 29s - loss: 0.2244 - categorical_accuracy: 0.9316
41920/60000 [===================>..........] - ETA: 29s - loss: 0.2245 - categorical_accuracy: 0.9316
41952/60000 [===================>..........] - ETA: 29s - loss: 0.2243 - categorical_accuracy: 0.9317
41984/60000 [===================>..........] - ETA: 29s - loss: 0.2243 - categorical_accuracy: 0.9317
42016/60000 [====================>.........] - ETA: 29s - loss: 0.2242 - categorical_accuracy: 0.9317
42048/60000 [====================>.........] - ETA: 29s - loss: 0.2241 - categorical_accuracy: 0.9317
42080/60000 [====================>.........] - ETA: 29s - loss: 0.2239 - categorical_accuracy: 0.9318
42112/60000 [====================>.........] - ETA: 29s - loss: 0.2239 - categorical_accuracy: 0.9318
42144/60000 [====================>.........] - ETA: 29s - loss: 0.2238 - categorical_accuracy: 0.9318
42176/60000 [====================>.........] - ETA: 29s - loss: 0.2236 - categorical_accuracy: 0.9319
42240/60000 [====================>.........] - ETA: 29s - loss: 0.2236 - categorical_accuracy: 0.9319
42272/60000 [====================>.........] - ETA: 29s - loss: 0.2236 - categorical_accuracy: 0.9320
42304/60000 [====================>.........] - ETA: 29s - loss: 0.2234 - categorical_accuracy: 0.9320
42336/60000 [====================>.........] - ETA: 29s - loss: 0.2233 - categorical_accuracy: 0.9321
42368/60000 [====================>.........] - ETA: 28s - loss: 0.2232 - categorical_accuracy: 0.9321
42400/60000 [====================>.........] - ETA: 28s - loss: 0.2230 - categorical_accuracy: 0.9321
42432/60000 [====================>.........] - ETA: 28s - loss: 0.2229 - categorical_accuracy: 0.9322
42464/60000 [====================>.........] - ETA: 28s - loss: 0.2228 - categorical_accuracy: 0.9322
42496/60000 [====================>.........] - ETA: 28s - loss: 0.2227 - categorical_accuracy: 0.9323
42560/60000 [====================>.........] - ETA: 28s - loss: 0.2224 - categorical_accuracy: 0.9323
42592/60000 [====================>.........] - ETA: 28s - loss: 0.2223 - categorical_accuracy: 0.9324
42624/60000 [====================>.........] - ETA: 28s - loss: 0.2221 - categorical_accuracy: 0.9324
42656/60000 [====================>.........] - ETA: 28s - loss: 0.2220 - categorical_accuracy: 0.9324
42720/60000 [====================>.........] - ETA: 28s - loss: 0.2220 - categorical_accuracy: 0.9325
42752/60000 [====================>.........] - ETA: 28s - loss: 0.2220 - categorical_accuracy: 0.9325
42784/60000 [====================>.........] - ETA: 28s - loss: 0.2219 - categorical_accuracy: 0.9325
42816/60000 [====================>.........] - ETA: 28s - loss: 0.2217 - categorical_accuracy: 0.9326
42848/60000 [====================>.........] - ETA: 28s - loss: 0.2216 - categorical_accuracy: 0.9326
42912/60000 [====================>.........] - ETA: 28s - loss: 0.2218 - categorical_accuracy: 0.9326
42944/60000 [====================>.........] - ETA: 28s - loss: 0.2217 - categorical_accuracy: 0.9327
42976/60000 [====================>.........] - ETA: 27s - loss: 0.2216 - categorical_accuracy: 0.9327
43040/60000 [====================>.........] - ETA: 27s - loss: 0.2213 - categorical_accuracy: 0.9328
43072/60000 [====================>.........] - ETA: 27s - loss: 0.2212 - categorical_accuracy: 0.9328
43136/60000 [====================>.........] - ETA: 27s - loss: 0.2212 - categorical_accuracy: 0.9328
43168/60000 [====================>.........] - ETA: 27s - loss: 0.2211 - categorical_accuracy: 0.9328
43200/60000 [====================>.........] - ETA: 27s - loss: 0.2209 - categorical_accuracy: 0.9329
43232/60000 [====================>.........] - ETA: 27s - loss: 0.2208 - categorical_accuracy: 0.9329
43264/60000 [====================>.........] - ETA: 27s - loss: 0.2207 - categorical_accuracy: 0.9330
43296/60000 [====================>.........] - ETA: 27s - loss: 0.2206 - categorical_accuracy: 0.9330
43328/60000 [====================>.........] - ETA: 27s - loss: 0.2205 - categorical_accuracy: 0.9330
43360/60000 [====================>.........] - ETA: 27s - loss: 0.2204 - categorical_accuracy: 0.9331
43392/60000 [====================>.........] - ETA: 27s - loss: 0.2204 - categorical_accuracy: 0.9331
43456/60000 [====================>.........] - ETA: 27s - loss: 0.2202 - categorical_accuracy: 0.9331
43520/60000 [====================>.........] - ETA: 27s - loss: 0.2199 - categorical_accuracy: 0.9332
43552/60000 [====================>.........] - ETA: 27s - loss: 0.2198 - categorical_accuracy: 0.9333
43616/60000 [====================>.........] - ETA: 26s - loss: 0.2195 - categorical_accuracy: 0.9334
43648/60000 [====================>.........] - ETA: 26s - loss: 0.2193 - categorical_accuracy: 0.9334
43680/60000 [====================>.........] - ETA: 26s - loss: 0.2192 - categorical_accuracy: 0.9334
43712/60000 [====================>.........] - ETA: 26s - loss: 0.2191 - categorical_accuracy: 0.9335
43744/60000 [====================>.........] - ETA: 26s - loss: 0.2192 - categorical_accuracy: 0.9334
43776/60000 [====================>.........] - ETA: 26s - loss: 0.2191 - categorical_accuracy: 0.9335
43808/60000 [====================>.........] - ETA: 26s - loss: 0.2192 - categorical_accuracy: 0.9334
43840/60000 [====================>.........] - ETA: 26s - loss: 0.2191 - categorical_accuracy: 0.9335
43872/60000 [====================>.........] - ETA: 26s - loss: 0.2192 - categorical_accuracy: 0.9335
43904/60000 [====================>.........] - ETA: 26s - loss: 0.2192 - categorical_accuracy: 0.9334
43936/60000 [====================>.........] - ETA: 26s - loss: 0.2191 - categorical_accuracy: 0.9335
43968/60000 [====================>.........] - ETA: 26s - loss: 0.2189 - categorical_accuracy: 0.9335
44000/60000 [=====================>........] - ETA: 26s - loss: 0.2188 - categorical_accuracy: 0.9336
44032/60000 [=====================>........] - ETA: 26s - loss: 0.2187 - categorical_accuracy: 0.9336
44064/60000 [=====================>........] - ETA: 26s - loss: 0.2187 - categorical_accuracy: 0.9336
44096/60000 [=====================>........] - ETA: 26s - loss: 0.2186 - categorical_accuracy: 0.9337
44128/60000 [=====================>........] - ETA: 26s - loss: 0.2185 - categorical_accuracy: 0.9337
44160/60000 [=====================>........] - ETA: 26s - loss: 0.2183 - categorical_accuracy: 0.9338
44224/60000 [=====================>........] - ETA: 25s - loss: 0.2181 - categorical_accuracy: 0.9338
44256/60000 [=====================>........] - ETA: 25s - loss: 0.2180 - categorical_accuracy: 0.9339
44288/60000 [=====================>........] - ETA: 25s - loss: 0.2180 - categorical_accuracy: 0.9339
44320/60000 [=====================>........] - ETA: 25s - loss: 0.2180 - categorical_accuracy: 0.9339
44352/60000 [=====================>........] - ETA: 25s - loss: 0.2180 - categorical_accuracy: 0.9339
44384/60000 [=====================>........] - ETA: 25s - loss: 0.2179 - categorical_accuracy: 0.9339
44416/60000 [=====================>........] - ETA: 25s - loss: 0.2178 - categorical_accuracy: 0.9339
44448/60000 [=====================>........] - ETA: 25s - loss: 0.2179 - categorical_accuracy: 0.9339
44480/60000 [=====================>........] - ETA: 25s - loss: 0.2178 - categorical_accuracy: 0.9339
44512/60000 [=====================>........] - ETA: 25s - loss: 0.2176 - categorical_accuracy: 0.9340
44544/60000 [=====================>........] - ETA: 25s - loss: 0.2177 - categorical_accuracy: 0.9340
44576/60000 [=====================>........] - ETA: 25s - loss: 0.2177 - categorical_accuracy: 0.9340
44608/60000 [=====================>........] - ETA: 25s - loss: 0.2176 - categorical_accuracy: 0.9340
44640/60000 [=====================>........] - ETA: 25s - loss: 0.2174 - categorical_accuracy: 0.9341
44704/60000 [=====================>........] - ETA: 25s - loss: 0.2173 - categorical_accuracy: 0.9341
44736/60000 [=====================>........] - ETA: 25s - loss: 0.2171 - categorical_accuracy: 0.9342
44768/60000 [=====================>........] - ETA: 25s - loss: 0.2170 - categorical_accuracy: 0.9342
44800/60000 [=====================>........] - ETA: 24s - loss: 0.2169 - categorical_accuracy: 0.9342
44832/60000 [=====================>........] - ETA: 24s - loss: 0.2169 - categorical_accuracy: 0.9342
44864/60000 [=====================>........] - ETA: 24s - loss: 0.2168 - categorical_accuracy: 0.9343
44928/60000 [=====================>........] - ETA: 24s - loss: 0.2166 - categorical_accuracy: 0.9343
44960/60000 [=====================>........] - ETA: 24s - loss: 0.2165 - categorical_accuracy: 0.9343
45024/60000 [=====================>........] - ETA: 24s - loss: 0.2164 - categorical_accuracy: 0.9344
45088/60000 [=====================>........] - ETA: 24s - loss: 0.2162 - categorical_accuracy: 0.9344
45120/60000 [=====================>........] - ETA: 24s - loss: 0.2160 - categorical_accuracy: 0.9345
45152/60000 [=====================>........] - ETA: 24s - loss: 0.2159 - categorical_accuracy: 0.9345
45184/60000 [=====================>........] - ETA: 24s - loss: 0.2158 - categorical_accuracy: 0.9346
45248/60000 [=====================>........] - ETA: 24s - loss: 0.2155 - categorical_accuracy: 0.9346
45280/60000 [=====================>........] - ETA: 24s - loss: 0.2154 - categorical_accuracy: 0.9347
45344/60000 [=====================>........] - ETA: 24s - loss: 0.2152 - categorical_accuracy: 0.9347
45376/60000 [=====================>........] - ETA: 24s - loss: 0.2151 - categorical_accuracy: 0.9347
45408/60000 [=====================>........] - ETA: 23s - loss: 0.2150 - categorical_accuracy: 0.9347
45440/60000 [=====================>........] - ETA: 23s - loss: 0.2149 - categorical_accuracy: 0.9348
45472/60000 [=====================>........] - ETA: 23s - loss: 0.2148 - categorical_accuracy: 0.9348
45504/60000 [=====================>........] - ETA: 23s - loss: 0.2146 - categorical_accuracy: 0.9349
45536/60000 [=====================>........] - ETA: 23s - loss: 0.2147 - categorical_accuracy: 0.9349
45600/60000 [=====================>........] - ETA: 23s - loss: 0.2146 - categorical_accuracy: 0.9349
45664/60000 [=====================>........] - ETA: 23s - loss: 0.2145 - categorical_accuracy: 0.9349
45696/60000 [=====================>........] - ETA: 23s - loss: 0.2145 - categorical_accuracy: 0.9349
45728/60000 [=====================>........] - ETA: 23s - loss: 0.2144 - categorical_accuracy: 0.9350
45792/60000 [=====================>........] - ETA: 23s - loss: 0.2142 - categorical_accuracy: 0.9350
45824/60000 [=====================>........] - ETA: 23s - loss: 0.2141 - categorical_accuracy: 0.9351
45856/60000 [=====================>........] - ETA: 23s - loss: 0.2139 - categorical_accuracy: 0.9351
45888/60000 [=====================>........] - ETA: 23s - loss: 0.2139 - categorical_accuracy: 0.9351
45920/60000 [=====================>........] - ETA: 23s - loss: 0.2138 - categorical_accuracy: 0.9351
45952/60000 [=====================>........] - ETA: 23s - loss: 0.2137 - categorical_accuracy: 0.9352
45984/60000 [=====================>........] - ETA: 23s - loss: 0.2136 - categorical_accuracy: 0.9352
46016/60000 [======================>.......] - ETA: 22s - loss: 0.2134 - categorical_accuracy: 0.9352
46048/60000 [======================>.......] - ETA: 22s - loss: 0.2134 - categorical_accuracy: 0.9352
46080/60000 [======================>.......] - ETA: 22s - loss: 0.2133 - categorical_accuracy: 0.9352
46112/60000 [======================>.......] - ETA: 22s - loss: 0.2133 - categorical_accuracy: 0.9353
46176/60000 [======================>.......] - ETA: 22s - loss: 0.2133 - categorical_accuracy: 0.9352
46208/60000 [======================>.......] - ETA: 22s - loss: 0.2133 - categorical_accuracy: 0.9352
46240/60000 [======================>.......] - ETA: 22s - loss: 0.2133 - categorical_accuracy: 0.9352
46272/60000 [======================>.......] - ETA: 22s - loss: 0.2131 - categorical_accuracy: 0.9353
46304/60000 [======================>.......] - ETA: 22s - loss: 0.2131 - categorical_accuracy: 0.9353
46336/60000 [======================>.......] - ETA: 22s - loss: 0.2131 - categorical_accuracy: 0.9353
46368/60000 [======================>.......] - ETA: 22s - loss: 0.2129 - categorical_accuracy: 0.9353
46400/60000 [======================>.......] - ETA: 22s - loss: 0.2128 - categorical_accuracy: 0.9354
46432/60000 [======================>.......] - ETA: 22s - loss: 0.2127 - categorical_accuracy: 0.9354
46464/60000 [======================>.......] - ETA: 22s - loss: 0.2126 - categorical_accuracy: 0.9355
46496/60000 [======================>.......] - ETA: 22s - loss: 0.2124 - categorical_accuracy: 0.9355
46528/60000 [======================>.......] - ETA: 22s - loss: 0.2123 - categorical_accuracy: 0.9355
46560/60000 [======================>.......] - ETA: 22s - loss: 0.2123 - categorical_accuracy: 0.9355
46592/60000 [======================>.......] - ETA: 22s - loss: 0.2122 - categorical_accuracy: 0.9356
46624/60000 [======================>.......] - ETA: 21s - loss: 0.2121 - categorical_accuracy: 0.9356
46688/60000 [======================>.......] - ETA: 21s - loss: 0.2119 - categorical_accuracy: 0.9356
46720/60000 [======================>.......] - ETA: 21s - loss: 0.2118 - categorical_accuracy: 0.9357
46784/60000 [======================>.......] - ETA: 21s - loss: 0.2116 - categorical_accuracy: 0.9357
46848/60000 [======================>.......] - ETA: 21s - loss: 0.2116 - categorical_accuracy: 0.9358
46880/60000 [======================>.......] - ETA: 21s - loss: 0.2114 - categorical_accuracy: 0.9358
46912/60000 [======================>.......] - ETA: 21s - loss: 0.2113 - categorical_accuracy: 0.9359
46944/60000 [======================>.......] - ETA: 21s - loss: 0.2113 - categorical_accuracy: 0.9359
47008/60000 [======================>.......] - ETA: 21s - loss: 0.2111 - categorical_accuracy: 0.9359
47040/60000 [======================>.......] - ETA: 21s - loss: 0.2110 - categorical_accuracy: 0.9360
47072/60000 [======================>.......] - ETA: 21s - loss: 0.2108 - categorical_accuracy: 0.9360
47104/60000 [======================>.......] - ETA: 21s - loss: 0.2107 - categorical_accuracy: 0.9361
47168/60000 [======================>.......] - ETA: 21s - loss: 0.2108 - categorical_accuracy: 0.9361
47200/60000 [======================>.......] - ETA: 21s - loss: 0.2108 - categorical_accuracy: 0.9361
47232/60000 [======================>.......] - ETA: 20s - loss: 0.2107 - categorical_accuracy: 0.9361
47264/60000 [======================>.......] - ETA: 20s - loss: 0.2107 - categorical_accuracy: 0.9361
47296/60000 [======================>.......] - ETA: 20s - loss: 0.2106 - categorical_accuracy: 0.9361
47328/60000 [======================>.......] - ETA: 20s - loss: 0.2105 - categorical_accuracy: 0.9361
47360/60000 [======================>.......] - ETA: 20s - loss: 0.2106 - categorical_accuracy: 0.9361
47392/60000 [======================>.......] - ETA: 20s - loss: 0.2106 - categorical_accuracy: 0.9361
47424/60000 [======================>.......] - ETA: 20s - loss: 0.2106 - categorical_accuracy: 0.9362
47456/60000 [======================>.......] - ETA: 20s - loss: 0.2105 - categorical_accuracy: 0.9362
47488/60000 [======================>.......] - ETA: 20s - loss: 0.2105 - categorical_accuracy: 0.9362
47520/60000 [======================>.......] - ETA: 20s - loss: 0.2103 - categorical_accuracy: 0.9363
47552/60000 [======================>.......] - ETA: 20s - loss: 0.2102 - categorical_accuracy: 0.9363
47584/60000 [======================>.......] - ETA: 20s - loss: 0.2101 - categorical_accuracy: 0.9363
47616/60000 [======================>.......] - ETA: 20s - loss: 0.2100 - categorical_accuracy: 0.9364
47648/60000 [======================>.......] - ETA: 20s - loss: 0.2099 - categorical_accuracy: 0.9364
47680/60000 [======================>.......] - ETA: 20s - loss: 0.2098 - categorical_accuracy: 0.9365
47712/60000 [======================>.......] - ETA: 20s - loss: 0.2099 - categorical_accuracy: 0.9365
47744/60000 [======================>.......] - ETA: 20s - loss: 0.2098 - categorical_accuracy: 0.9365
47808/60000 [======================>.......] - ETA: 20s - loss: 0.2097 - categorical_accuracy: 0.9365
47840/60000 [======================>.......] - ETA: 19s - loss: 0.2097 - categorical_accuracy: 0.9365
47872/60000 [======================>.......] - ETA: 19s - loss: 0.2096 - categorical_accuracy: 0.9365
47904/60000 [======================>.......] - ETA: 19s - loss: 0.2096 - categorical_accuracy: 0.9365
47936/60000 [======================>.......] - ETA: 19s - loss: 0.2096 - categorical_accuracy: 0.9366
47968/60000 [======================>.......] - ETA: 19s - loss: 0.2094 - categorical_accuracy: 0.9366
48000/60000 [=======================>......] - ETA: 19s - loss: 0.2093 - categorical_accuracy: 0.9367
48032/60000 [=======================>......] - ETA: 19s - loss: 0.2092 - categorical_accuracy: 0.9367
48064/60000 [=======================>......] - ETA: 19s - loss: 0.2091 - categorical_accuracy: 0.9368
48096/60000 [=======================>......] - ETA: 19s - loss: 0.2090 - categorical_accuracy: 0.9368
48128/60000 [=======================>......] - ETA: 19s - loss: 0.2090 - categorical_accuracy: 0.9368
48160/60000 [=======================>......] - ETA: 19s - loss: 0.2090 - categorical_accuracy: 0.9368
48192/60000 [=======================>......] - ETA: 19s - loss: 0.2089 - categorical_accuracy: 0.9368
48224/60000 [=======================>......] - ETA: 19s - loss: 0.2087 - categorical_accuracy: 0.9369
48256/60000 [=======================>......] - ETA: 19s - loss: 0.2087 - categorical_accuracy: 0.9369
48288/60000 [=======================>......] - ETA: 19s - loss: 0.2086 - categorical_accuracy: 0.9369
48352/60000 [=======================>......] - ETA: 19s - loss: 0.2084 - categorical_accuracy: 0.9370
48384/60000 [=======================>......] - ETA: 19s - loss: 0.2082 - categorical_accuracy: 0.9370
48416/60000 [=======================>......] - ETA: 19s - loss: 0.2083 - categorical_accuracy: 0.9370
48448/60000 [=======================>......] - ETA: 18s - loss: 0.2081 - categorical_accuracy: 0.9371
48480/60000 [=======================>......] - ETA: 18s - loss: 0.2081 - categorical_accuracy: 0.9371
48512/60000 [=======================>......] - ETA: 18s - loss: 0.2080 - categorical_accuracy: 0.9371
48544/60000 [=======================>......] - ETA: 18s - loss: 0.2080 - categorical_accuracy: 0.9371
48576/60000 [=======================>......] - ETA: 18s - loss: 0.2078 - categorical_accuracy: 0.9372
48608/60000 [=======================>......] - ETA: 18s - loss: 0.2077 - categorical_accuracy: 0.9372
48640/60000 [=======================>......] - ETA: 18s - loss: 0.2076 - categorical_accuracy: 0.9372
48672/60000 [=======================>......] - ETA: 18s - loss: 0.2076 - categorical_accuracy: 0.9372
48704/60000 [=======================>......] - ETA: 18s - loss: 0.2075 - categorical_accuracy: 0.9373
48736/60000 [=======================>......] - ETA: 18s - loss: 0.2074 - categorical_accuracy: 0.9373
48768/60000 [=======================>......] - ETA: 18s - loss: 0.2073 - categorical_accuracy: 0.9373
48800/60000 [=======================>......] - ETA: 18s - loss: 0.2074 - categorical_accuracy: 0.9373
48832/60000 [=======================>......] - ETA: 18s - loss: 0.2073 - categorical_accuracy: 0.9373
48864/60000 [=======================>......] - ETA: 18s - loss: 0.2072 - categorical_accuracy: 0.9373
48896/60000 [=======================>......] - ETA: 18s - loss: 0.2071 - categorical_accuracy: 0.9374
48928/60000 [=======================>......] - ETA: 18s - loss: 0.2071 - categorical_accuracy: 0.9373
48960/60000 [=======================>......] - ETA: 18s - loss: 0.2071 - categorical_accuracy: 0.9373
48992/60000 [=======================>......] - ETA: 18s - loss: 0.2070 - categorical_accuracy: 0.9374
49024/60000 [=======================>......] - ETA: 18s - loss: 0.2069 - categorical_accuracy: 0.9374
49056/60000 [=======================>......] - ETA: 17s - loss: 0.2067 - categorical_accuracy: 0.9375
49088/60000 [=======================>......] - ETA: 17s - loss: 0.2066 - categorical_accuracy: 0.9375
49120/60000 [=======================>......] - ETA: 17s - loss: 0.2066 - categorical_accuracy: 0.9375
49152/60000 [=======================>......] - ETA: 17s - loss: 0.2065 - categorical_accuracy: 0.9375
49184/60000 [=======================>......] - ETA: 17s - loss: 0.2064 - categorical_accuracy: 0.9375
49216/60000 [=======================>......] - ETA: 17s - loss: 0.2063 - categorical_accuracy: 0.9375
49248/60000 [=======================>......] - ETA: 17s - loss: 0.2063 - categorical_accuracy: 0.9375
49280/60000 [=======================>......] - ETA: 17s - loss: 0.2062 - categorical_accuracy: 0.9375
49312/60000 [=======================>......] - ETA: 17s - loss: 0.2061 - categorical_accuracy: 0.9376
49344/60000 [=======================>......] - ETA: 17s - loss: 0.2060 - categorical_accuracy: 0.9376
49376/60000 [=======================>......] - ETA: 17s - loss: 0.2059 - categorical_accuracy: 0.9376
49408/60000 [=======================>......] - ETA: 17s - loss: 0.2059 - categorical_accuracy: 0.9376
49440/60000 [=======================>......] - ETA: 17s - loss: 0.2058 - categorical_accuracy: 0.9377
49472/60000 [=======================>......] - ETA: 17s - loss: 0.2057 - categorical_accuracy: 0.9377
49504/60000 [=======================>......] - ETA: 17s - loss: 0.2057 - categorical_accuracy: 0.9377
49536/60000 [=======================>......] - ETA: 17s - loss: 0.2056 - categorical_accuracy: 0.9378
49568/60000 [=======================>......] - ETA: 17s - loss: 0.2056 - categorical_accuracy: 0.9378
49600/60000 [=======================>......] - ETA: 17s - loss: 0.2055 - categorical_accuracy: 0.9378
49632/60000 [=======================>......] - ETA: 17s - loss: 0.2054 - categorical_accuracy: 0.9378
49664/60000 [=======================>......] - ETA: 16s - loss: 0.2052 - categorical_accuracy: 0.9379
49696/60000 [=======================>......] - ETA: 16s - loss: 0.2051 - categorical_accuracy: 0.9379
49728/60000 [=======================>......] - ETA: 16s - loss: 0.2050 - categorical_accuracy: 0.9379
49760/60000 [=======================>......] - ETA: 16s - loss: 0.2050 - categorical_accuracy: 0.9379
49792/60000 [=======================>......] - ETA: 16s - loss: 0.2049 - categorical_accuracy: 0.9380
49824/60000 [=======================>......] - ETA: 16s - loss: 0.2047 - categorical_accuracy: 0.9380
49856/60000 [=======================>......] - ETA: 16s - loss: 0.2047 - categorical_accuracy: 0.9380
49888/60000 [=======================>......] - ETA: 16s - loss: 0.2046 - categorical_accuracy: 0.9381
49920/60000 [=======================>......] - ETA: 16s - loss: 0.2044 - categorical_accuracy: 0.9381
49952/60000 [=======================>......] - ETA: 16s - loss: 0.2044 - categorical_accuracy: 0.9381
49984/60000 [=======================>......] - ETA: 16s - loss: 0.2043 - categorical_accuracy: 0.9382
50016/60000 [========================>.....] - ETA: 16s - loss: 0.2042 - categorical_accuracy: 0.9382
50080/60000 [========================>.....] - ETA: 16s - loss: 0.2040 - categorical_accuracy: 0.9382
50112/60000 [========================>.....] - ETA: 16s - loss: 0.2039 - categorical_accuracy: 0.9383
50144/60000 [========================>.....] - ETA: 16s - loss: 0.2038 - categorical_accuracy: 0.9383
50176/60000 [========================>.....] - ETA: 16s - loss: 0.2038 - categorical_accuracy: 0.9383
50240/60000 [========================>.....] - ETA: 16s - loss: 0.2037 - categorical_accuracy: 0.9383
50272/60000 [========================>.....] - ETA: 15s - loss: 0.2036 - categorical_accuracy: 0.9383
50304/60000 [========================>.....] - ETA: 15s - loss: 0.2035 - categorical_accuracy: 0.9384
50336/60000 [========================>.....] - ETA: 15s - loss: 0.2034 - categorical_accuracy: 0.9384
50368/60000 [========================>.....] - ETA: 15s - loss: 0.2034 - categorical_accuracy: 0.9384
50400/60000 [========================>.....] - ETA: 15s - loss: 0.2034 - categorical_accuracy: 0.9384
50432/60000 [========================>.....] - ETA: 15s - loss: 0.2033 - categorical_accuracy: 0.9384
50464/60000 [========================>.....] - ETA: 15s - loss: 0.2034 - categorical_accuracy: 0.9384
50496/60000 [========================>.....] - ETA: 15s - loss: 0.2033 - categorical_accuracy: 0.9385
50528/60000 [========================>.....] - ETA: 15s - loss: 0.2032 - categorical_accuracy: 0.9385
50560/60000 [========================>.....] - ETA: 15s - loss: 0.2032 - categorical_accuracy: 0.9385
50592/60000 [========================>.....] - ETA: 15s - loss: 0.2031 - categorical_accuracy: 0.9385
50624/60000 [========================>.....] - ETA: 15s - loss: 0.2034 - categorical_accuracy: 0.9385
50656/60000 [========================>.....] - ETA: 15s - loss: 0.2033 - categorical_accuracy: 0.9385
50688/60000 [========================>.....] - ETA: 15s - loss: 0.2032 - categorical_accuracy: 0.9385
50720/60000 [========================>.....] - ETA: 15s - loss: 0.2031 - categorical_accuracy: 0.9386
50752/60000 [========================>.....] - ETA: 15s - loss: 0.2031 - categorical_accuracy: 0.9386
50816/60000 [========================>.....] - ETA: 15s - loss: 0.2029 - categorical_accuracy: 0.9386
50848/60000 [========================>.....] - ETA: 15s - loss: 0.2028 - categorical_accuracy: 0.9387
50912/60000 [========================>.....] - ETA: 14s - loss: 0.2026 - categorical_accuracy: 0.9387
50944/60000 [========================>.....] - ETA: 14s - loss: 0.2026 - categorical_accuracy: 0.9387
50976/60000 [========================>.....] - ETA: 14s - loss: 0.2026 - categorical_accuracy: 0.9387
51008/60000 [========================>.....] - ETA: 14s - loss: 0.2025 - categorical_accuracy: 0.9387
51040/60000 [========================>.....] - ETA: 14s - loss: 0.2026 - categorical_accuracy: 0.9388
51072/60000 [========================>.....] - ETA: 14s - loss: 0.2025 - categorical_accuracy: 0.9388
51104/60000 [========================>.....] - ETA: 14s - loss: 0.2024 - categorical_accuracy: 0.9388
51136/60000 [========================>.....] - ETA: 14s - loss: 0.2023 - categorical_accuracy: 0.9389
51168/60000 [========================>.....] - ETA: 14s - loss: 0.2022 - categorical_accuracy: 0.9389
51200/60000 [========================>.....] - ETA: 14s - loss: 0.2021 - categorical_accuracy: 0.9389
51232/60000 [========================>.....] - ETA: 14s - loss: 0.2022 - categorical_accuracy: 0.9389
51264/60000 [========================>.....] - ETA: 14s - loss: 0.2022 - categorical_accuracy: 0.9389
51296/60000 [========================>.....] - ETA: 14s - loss: 0.2021 - categorical_accuracy: 0.9390
51328/60000 [========================>.....] - ETA: 14s - loss: 0.2020 - categorical_accuracy: 0.9390
51392/60000 [========================>.....] - ETA: 14s - loss: 0.2019 - categorical_accuracy: 0.9390
51456/60000 [========================>.....] - ETA: 14s - loss: 0.2017 - categorical_accuracy: 0.9390
51488/60000 [========================>.....] - ETA: 13s - loss: 0.2016 - categorical_accuracy: 0.9391
51520/60000 [========================>.....] - ETA: 13s - loss: 0.2016 - categorical_accuracy: 0.9391
51552/60000 [========================>.....] - ETA: 13s - loss: 0.2014 - categorical_accuracy: 0.9391
51584/60000 [========================>.....] - ETA: 13s - loss: 0.2013 - categorical_accuracy: 0.9392
51616/60000 [========================>.....] - ETA: 13s - loss: 0.2012 - categorical_accuracy: 0.9392
51648/60000 [========================>.....] - ETA: 13s - loss: 0.2012 - categorical_accuracy: 0.9392
51680/60000 [========================>.....] - ETA: 13s - loss: 0.2012 - categorical_accuracy: 0.9392
51712/60000 [========================>.....] - ETA: 13s - loss: 0.2012 - categorical_accuracy: 0.9392
51744/60000 [========================>.....] - ETA: 13s - loss: 0.2012 - categorical_accuracy: 0.9392
51776/60000 [========================>.....] - ETA: 13s - loss: 0.2011 - categorical_accuracy: 0.9393
51840/60000 [========================>.....] - ETA: 13s - loss: 0.2009 - categorical_accuracy: 0.9394
51872/60000 [========================>.....] - ETA: 13s - loss: 0.2009 - categorical_accuracy: 0.9393
51904/60000 [========================>.....] - ETA: 13s - loss: 0.2009 - categorical_accuracy: 0.9393
51936/60000 [========================>.....] - ETA: 13s - loss: 0.2009 - categorical_accuracy: 0.9393
52000/60000 [=========================>....] - ETA: 13s - loss: 0.2008 - categorical_accuracy: 0.9393
52032/60000 [=========================>....] - ETA: 13s - loss: 0.2008 - categorical_accuracy: 0.9393
52064/60000 [=========================>....] - ETA: 13s - loss: 0.2007 - categorical_accuracy: 0.9394
52096/60000 [=========================>....] - ETA: 12s - loss: 0.2006 - categorical_accuracy: 0.9394
52160/60000 [=========================>....] - ETA: 12s - loss: 0.2006 - categorical_accuracy: 0.9394
52192/60000 [=========================>....] - ETA: 12s - loss: 0.2005 - categorical_accuracy: 0.9394
52224/60000 [=========================>....] - ETA: 12s - loss: 0.2006 - categorical_accuracy: 0.9394
52256/60000 [=========================>....] - ETA: 12s - loss: 0.2005 - categorical_accuracy: 0.9394
52288/60000 [=========================>....] - ETA: 12s - loss: 0.2005 - categorical_accuracy: 0.9394
52352/60000 [=========================>....] - ETA: 12s - loss: 0.2004 - categorical_accuracy: 0.9394
52384/60000 [=========================>....] - ETA: 12s - loss: 0.2004 - categorical_accuracy: 0.9395
52416/60000 [=========================>....] - ETA: 12s - loss: 0.2002 - categorical_accuracy: 0.9395
52448/60000 [=========================>....] - ETA: 12s - loss: 0.2002 - categorical_accuracy: 0.9395
52480/60000 [=========================>....] - ETA: 12s - loss: 0.2001 - categorical_accuracy: 0.9396
52512/60000 [=========================>....] - ETA: 12s - loss: 0.2000 - categorical_accuracy: 0.9396
52544/60000 [=========================>....] - ETA: 12s - loss: 0.2000 - categorical_accuracy: 0.9396
52576/60000 [=========================>....] - ETA: 12s - loss: 0.1999 - categorical_accuracy: 0.9396
52608/60000 [=========================>....] - ETA: 12s - loss: 0.1998 - categorical_accuracy: 0.9396
52640/60000 [=========================>....] - ETA: 12s - loss: 0.2000 - categorical_accuracy: 0.9396
52672/60000 [=========================>....] - ETA: 12s - loss: 0.1999 - categorical_accuracy: 0.9396
52704/60000 [=========================>....] - ETA: 11s - loss: 0.1999 - categorical_accuracy: 0.9396
52736/60000 [=========================>....] - ETA: 11s - loss: 0.1997 - categorical_accuracy: 0.9396
52768/60000 [=========================>....] - ETA: 11s - loss: 0.1996 - categorical_accuracy: 0.9397
52800/60000 [=========================>....] - ETA: 11s - loss: 0.1996 - categorical_accuracy: 0.9397
52832/60000 [=========================>....] - ETA: 11s - loss: 0.1995 - categorical_accuracy: 0.9397
52896/60000 [=========================>....] - ETA: 11s - loss: 0.1995 - categorical_accuracy: 0.9397
52928/60000 [=========================>....] - ETA: 11s - loss: 0.1994 - categorical_accuracy: 0.9397
52960/60000 [=========================>....] - ETA: 11s - loss: 0.1994 - categorical_accuracy: 0.9397
52992/60000 [=========================>....] - ETA: 11s - loss: 0.1993 - categorical_accuracy: 0.9397
53024/60000 [=========================>....] - ETA: 11s - loss: 0.1992 - categorical_accuracy: 0.9397
53056/60000 [=========================>....] - ETA: 11s - loss: 0.1992 - categorical_accuracy: 0.9398
53088/60000 [=========================>....] - ETA: 11s - loss: 0.1992 - categorical_accuracy: 0.9398
53120/60000 [=========================>....] - ETA: 11s - loss: 0.1991 - categorical_accuracy: 0.9398
53152/60000 [=========================>....] - ETA: 11s - loss: 0.1991 - categorical_accuracy: 0.9398
53216/60000 [=========================>....] - ETA: 11s - loss: 0.1991 - categorical_accuracy: 0.9398
53280/60000 [=========================>....] - ETA: 11s - loss: 0.1991 - categorical_accuracy: 0.9398
53312/60000 [=========================>....] - ETA: 10s - loss: 0.1991 - categorical_accuracy: 0.9398
53376/60000 [=========================>....] - ETA: 10s - loss: 0.1989 - categorical_accuracy: 0.9398
53408/60000 [=========================>....] - ETA: 10s - loss: 0.1989 - categorical_accuracy: 0.9398
53440/60000 [=========================>....] - ETA: 10s - loss: 0.1988 - categorical_accuracy: 0.9399
53472/60000 [=========================>....] - ETA: 10s - loss: 0.1988 - categorical_accuracy: 0.9399
53504/60000 [=========================>....] - ETA: 10s - loss: 0.1987 - categorical_accuracy: 0.9399
53568/60000 [=========================>....] - ETA: 10s - loss: 0.1985 - categorical_accuracy: 0.9400
53600/60000 [=========================>....] - ETA: 10s - loss: 0.1985 - categorical_accuracy: 0.9400
53632/60000 [=========================>....] - ETA: 10s - loss: 0.1984 - categorical_accuracy: 0.9400
53664/60000 [=========================>....] - ETA: 10s - loss: 0.1983 - categorical_accuracy: 0.9400
53696/60000 [=========================>....] - ETA: 10s - loss: 0.1982 - categorical_accuracy: 0.9400
53728/60000 [=========================>....] - ETA: 10s - loss: 0.1981 - categorical_accuracy: 0.9400
53760/60000 [=========================>....] - ETA: 10s - loss: 0.1981 - categorical_accuracy: 0.9400
53824/60000 [=========================>....] - ETA: 10s - loss: 0.1979 - categorical_accuracy: 0.9401
53856/60000 [=========================>....] - ETA: 10s - loss: 0.1980 - categorical_accuracy: 0.9401
53888/60000 [=========================>....] - ETA: 10s - loss: 0.1980 - categorical_accuracy: 0.9401
53920/60000 [=========================>....] - ETA: 9s - loss: 0.1979 - categorical_accuracy: 0.9401 
53952/60000 [=========================>....] - ETA: 9s - loss: 0.1979 - categorical_accuracy: 0.9401
53984/60000 [=========================>....] - ETA: 9s - loss: 0.1978 - categorical_accuracy: 0.9401
54016/60000 [==========================>...] - ETA: 9s - loss: 0.1977 - categorical_accuracy: 0.9402
54048/60000 [==========================>...] - ETA: 9s - loss: 0.1977 - categorical_accuracy: 0.9402
54080/60000 [==========================>...] - ETA: 9s - loss: 0.1976 - categorical_accuracy: 0.9402
54112/60000 [==========================>...] - ETA: 9s - loss: 0.1976 - categorical_accuracy: 0.9402
54144/60000 [==========================>...] - ETA: 9s - loss: 0.1975 - categorical_accuracy: 0.9402
54176/60000 [==========================>...] - ETA: 9s - loss: 0.1974 - categorical_accuracy: 0.9403
54208/60000 [==========================>...] - ETA: 9s - loss: 0.1973 - categorical_accuracy: 0.9403
54240/60000 [==========================>...] - ETA: 9s - loss: 0.1973 - categorical_accuracy: 0.9403
54272/60000 [==========================>...] - ETA: 9s - loss: 0.1973 - categorical_accuracy: 0.9403
54304/60000 [==========================>...] - ETA: 9s - loss: 0.1972 - categorical_accuracy: 0.9403
54336/60000 [==========================>...] - ETA: 9s - loss: 0.1972 - categorical_accuracy: 0.9403
54368/60000 [==========================>...] - ETA: 9s - loss: 0.1971 - categorical_accuracy: 0.9403
54400/60000 [==========================>...] - ETA: 9s - loss: 0.1970 - categorical_accuracy: 0.9404
54432/60000 [==========================>...] - ETA: 9s - loss: 0.1970 - categorical_accuracy: 0.9404
54464/60000 [==========================>...] - ETA: 9s - loss: 0.1969 - categorical_accuracy: 0.9404
54528/60000 [==========================>...] - ETA: 8s - loss: 0.1968 - categorical_accuracy: 0.9404
54592/60000 [==========================>...] - ETA: 8s - loss: 0.1966 - categorical_accuracy: 0.9405
54656/60000 [==========================>...] - ETA: 8s - loss: 0.1966 - categorical_accuracy: 0.9405
54688/60000 [==========================>...] - ETA: 8s - loss: 0.1968 - categorical_accuracy: 0.9405
54720/60000 [==========================>...] - ETA: 8s - loss: 0.1968 - categorical_accuracy: 0.9405
54752/60000 [==========================>...] - ETA: 8s - loss: 0.1967 - categorical_accuracy: 0.9405
54784/60000 [==========================>...] - ETA: 8s - loss: 0.1966 - categorical_accuracy: 0.9405
54816/60000 [==========================>...] - ETA: 8s - loss: 0.1965 - categorical_accuracy: 0.9406
54880/60000 [==========================>...] - ETA: 8s - loss: 0.1964 - categorical_accuracy: 0.9406
54912/60000 [==========================>...] - ETA: 8s - loss: 0.1963 - categorical_accuracy: 0.9406
54944/60000 [==========================>...] - ETA: 8s - loss: 0.1963 - categorical_accuracy: 0.9406
55008/60000 [==========================>...] - ETA: 8s - loss: 0.1961 - categorical_accuracy: 0.9407
55040/60000 [==========================>...] - ETA: 8s - loss: 0.1961 - categorical_accuracy: 0.9407
55072/60000 [==========================>...] - ETA: 8s - loss: 0.1961 - categorical_accuracy: 0.9407
55136/60000 [==========================>...] - ETA: 7s - loss: 0.1959 - categorical_accuracy: 0.9407
55168/60000 [==========================>...] - ETA: 7s - loss: 0.1958 - categorical_accuracy: 0.9407
55200/60000 [==========================>...] - ETA: 7s - loss: 0.1957 - categorical_accuracy: 0.9407
55232/60000 [==========================>...] - ETA: 7s - loss: 0.1956 - categorical_accuracy: 0.9408
55264/60000 [==========================>...] - ETA: 7s - loss: 0.1956 - categorical_accuracy: 0.9408
55296/60000 [==========================>...] - ETA: 7s - loss: 0.1955 - categorical_accuracy: 0.9408
55328/60000 [==========================>...] - ETA: 7s - loss: 0.1955 - categorical_accuracy: 0.9408
55360/60000 [==========================>...] - ETA: 7s - loss: 0.1954 - categorical_accuracy: 0.9408
55392/60000 [==========================>...] - ETA: 7s - loss: 0.1953 - categorical_accuracy: 0.9409
55424/60000 [==========================>...] - ETA: 7s - loss: 0.1952 - categorical_accuracy: 0.9409
55456/60000 [==========================>...] - ETA: 7s - loss: 0.1952 - categorical_accuracy: 0.9409
55488/60000 [==========================>...] - ETA: 7s - loss: 0.1952 - categorical_accuracy: 0.9409
55520/60000 [==========================>...] - ETA: 7s - loss: 0.1951 - categorical_accuracy: 0.9409
55552/60000 [==========================>...] - ETA: 7s - loss: 0.1950 - categorical_accuracy: 0.9410
55584/60000 [==========================>...] - ETA: 7s - loss: 0.1949 - categorical_accuracy: 0.9410
55616/60000 [==========================>...] - ETA: 7s - loss: 0.1949 - categorical_accuracy: 0.9410
55648/60000 [==========================>...] - ETA: 7s - loss: 0.1948 - categorical_accuracy: 0.9410
55680/60000 [==========================>...] - ETA: 7s - loss: 0.1947 - categorical_accuracy: 0.9411
55712/60000 [==========================>...] - ETA: 7s - loss: 0.1947 - categorical_accuracy: 0.9411
55744/60000 [==========================>...] - ETA: 6s - loss: 0.1947 - categorical_accuracy: 0.9411
55776/60000 [==========================>...] - ETA: 6s - loss: 0.1948 - categorical_accuracy: 0.9410
55808/60000 [==========================>...] - ETA: 6s - loss: 0.1947 - categorical_accuracy: 0.9411
55840/60000 [==========================>...] - ETA: 6s - loss: 0.1946 - categorical_accuracy: 0.9411
55904/60000 [==========================>...] - ETA: 6s - loss: 0.1944 - categorical_accuracy: 0.9411
55968/60000 [==========================>...] - ETA: 6s - loss: 0.1942 - categorical_accuracy: 0.9412
56000/60000 [===========================>..] - ETA: 6s - loss: 0.1942 - categorical_accuracy: 0.9412
56032/60000 [===========================>..] - ETA: 6s - loss: 0.1941 - categorical_accuracy: 0.9412
56064/60000 [===========================>..] - ETA: 6s - loss: 0.1942 - categorical_accuracy: 0.9412
56096/60000 [===========================>..] - ETA: 6s - loss: 0.1940 - categorical_accuracy: 0.9412
56128/60000 [===========================>..] - ETA: 6s - loss: 0.1940 - categorical_accuracy: 0.9413
56160/60000 [===========================>..] - ETA: 6s - loss: 0.1940 - categorical_accuracy: 0.9413
56224/60000 [===========================>..] - ETA: 6s - loss: 0.1939 - categorical_accuracy: 0.9413
56256/60000 [===========================>..] - ETA: 6s - loss: 0.1938 - categorical_accuracy: 0.9413
56288/60000 [===========================>..] - ETA: 6s - loss: 0.1938 - categorical_accuracy: 0.9413
56320/60000 [===========================>..] - ETA: 6s - loss: 0.1937 - categorical_accuracy: 0.9414
56352/60000 [===========================>..] - ETA: 5s - loss: 0.1937 - categorical_accuracy: 0.9414
56384/60000 [===========================>..] - ETA: 5s - loss: 0.1937 - categorical_accuracy: 0.9414
56416/60000 [===========================>..] - ETA: 5s - loss: 0.1937 - categorical_accuracy: 0.9414
56448/60000 [===========================>..] - ETA: 5s - loss: 0.1936 - categorical_accuracy: 0.9414
56512/60000 [===========================>..] - ETA: 5s - loss: 0.1934 - categorical_accuracy: 0.9415
56576/60000 [===========================>..] - ETA: 5s - loss: 0.1932 - categorical_accuracy: 0.9415
56608/60000 [===========================>..] - ETA: 5s - loss: 0.1931 - categorical_accuracy: 0.9415
56672/60000 [===========================>..] - ETA: 5s - loss: 0.1933 - categorical_accuracy: 0.9415
56736/60000 [===========================>..] - ETA: 5s - loss: 0.1932 - categorical_accuracy: 0.9416
56768/60000 [===========================>..] - ETA: 5s - loss: 0.1931 - categorical_accuracy: 0.9416
56800/60000 [===========================>..] - ETA: 5s - loss: 0.1932 - categorical_accuracy: 0.9415
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1932 - categorical_accuracy: 0.9415
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1931 - categorical_accuracy: 0.9415
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1930 - categorical_accuracy: 0.9416
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1931 - categorical_accuracy: 0.9416
56960/60000 [===========================>..] - ETA: 4s - loss: 0.1930 - categorical_accuracy: 0.9416
56992/60000 [===========================>..] - ETA: 4s - loss: 0.1929 - categorical_accuracy: 0.9416
57024/60000 [===========================>..] - ETA: 4s - loss: 0.1928 - categorical_accuracy: 0.9416
57056/60000 [===========================>..] - ETA: 4s - loss: 0.1928 - categorical_accuracy: 0.9416
57088/60000 [===========================>..] - ETA: 4s - loss: 0.1927 - categorical_accuracy: 0.9417
57120/60000 [===========================>..] - ETA: 4s - loss: 0.1927 - categorical_accuracy: 0.9417
57152/60000 [===========================>..] - ETA: 4s - loss: 0.1926 - categorical_accuracy: 0.9417
57184/60000 [===========================>..] - ETA: 4s - loss: 0.1925 - categorical_accuracy: 0.9417
57216/60000 [===========================>..] - ETA: 4s - loss: 0.1924 - categorical_accuracy: 0.9417
57248/60000 [===========================>..] - ETA: 4s - loss: 0.1923 - categorical_accuracy: 0.9418
57280/60000 [===========================>..] - ETA: 4s - loss: 0.1925 - categorical_accuracy: 0.9418
57312/60000 [===========================>..] - ETA: 4s - loss: 0.1925 - categorical_accuracy: 0.9418
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1924 - categorical_accuracy: 0.9418
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1924 - categorical_accuracy: 0.9418
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1924 - categorical_accuracy: 0.9418
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1923 - categorical_accuracy: 0.9418
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1922 - categorical_accuracy: 0.9418
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1922 - categorical_accuracy: 0.9419
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1922 - categorical_accuracy: 0.9419
57568/60000 [===========================>..] - ETA: 3s - loss: 0.1921 - categorical_accuracy: 0.9419
57632/60000 [===========================>..] - ETA: 3s - loss: 0.1920 - categorical_accuracy: 0.9419
57664/60000 [===========================>..] - ETA: 3s - loss: 0.1919 - categorical_accuracy: 0.9419
57696/60000 [===========================>..] - ETA: 3s - loss: 0.1919 - categorical_accuracy: 0.9420
57728/60000 [===========================>..] - ETA: 3s - loss: 0.1918 - categorical_accuracy: 0.9420
57760/60000 [===========================>..] - ETA: 3s - loss: 0.1917 - categorical_accuracy: 0.9420
57792/60000 [===========================>..] - ETA: 3s - loss: 0.1916 - categorical_accuracy: 0.9420
57824/60000 [===========================>..] - ETA: 3s - loss: 0.1917 - categorical_accuracy: 0.9420
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1916 - categorical_accuracy: 0.9421
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1917 - categorical_accuracy: 0.9420
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1916 - categorical_accuracy: 0.9421
58048/60000 [============================>.] - ETA: 3s - loss: 0.1915 - categorical_accuracy: 0.9421
58080/60000 [============================>.] - ETA: 3s - loss: 0.1914 - categorical_accuracy: 0.9421
58112/60000 [============================>.] - ETA: 3s - loss: 0.1913 - categorical_accuracy: 0.9422
58144/60000 [============================>.] - ETA: 3s - loss: 0.1912 - categorical_accuracy: 0.9422
58176/60000 [============================>.] - ETA: 2s - loss: 0.1912 - categorical_accuracy: 0.9422
58208/60000 [============================>.] - ETA: 2s - loss: 0.1911 - categorical_accuracy: 0.9422
58240/60000 [============================>.] - ETA: 2s - loss: 0.1911 - categorical_accuracy: 0.9422
58272/60000 [============================>.] - ETA: 2s - loss: 0.1910 - categorical_accuracy: 0.9423
58304/60000 [============================>.] - ETA: 2s - loss: 0.1910 - categorical_accuracy: 0.9423
58336/60000 [============================>.] - ETA: 2s - loss: 0.1909 - categorical_accuracy: 0.9423
58368/60000 [============================>.] - ETA: 2s - loss: 0.1910 - categorical_accuracy: 0.9423
58400/60000 [============================>.] - ETA: 2s - loss: 0.1909 - categorical_accuracy: 0.9423
58432/60000 [============================>.] - ETA: 2s - loss: 0.1909 - categorical_accuracy: 0.9423
58464/60000 [============================>.] - ETA: 2s - loss: 0.1908 - categorical_accuracy: 0.9424
58496/60000 [============================>.] - ETA: 2s - loss: 0.1907 - categorical_accuracy: 0.9424
58528/60000 [============================>.] - ETA: 2s - loss: 0.1906 - categorical_accuracy: 0.9424
58560/60000 [============================>.] - ETA: 2s - loss: 0.1905 - categorical_accuracy: 0.9424
58592/60000 [============================>.] - ETA: 2s - loss: 0.1905 - categorical_accuracy: 0.9424
58624/60000 [============================>.] - ETA: 2s - loss: 0.1904 - categorical_accuracy: 0.9425
58656/60000 [============================>.] - ETA: 2s - loss: 0.1903 - categorical_accuracy: 0.9425
58688/60000 [============================>.] - ETA: 2s - loss: 0.1902 - categorical_accuracy: 0.9425
58720/60000 [============================>.] - ETA: 2s - loss: 0.1903 - categorical_accuracy: 0.9425
58752/60000 [============================>.] - ETA: 2s - loss: 0.1903 - categorical_accuracy: 0.9425
58784/60000 [============================>.] - ETA: 1s - loss: 0.1903 - categorical_accuracy: 0.9425
58816/60000 [============================>.] - ETA: 1s - loss: 0.1902 - categorical_accuracy: 0.9425
58848/60000 [============================>.] - ETA: 1s - loss: 0.1901 - categorical_accuracy: 0.9425
58880/60000 [============================>.] - ETA: 1s - loss: 0.1901 - categorical_accuracy: 0.9426
58912/60000 [============================>.] - ETA: 1s - loss: 0.1900 - categorical_accuracy: 0.9426
58944/60000 [============================>.] - ETA: 1s - loss: 0.1900 - categorical_accuracy: 0.9426
58976/60000 [============================>.] - ETA: 1s - loss: 0.1900 - categorical_accuracy: 0.9426
59008/60000 [============================>.] - ETA: 1s - loss: 0.1899 - categorical_accuracy: 0.9426
59040/60000 [============================>.] - ETA: 1s - loss: 0.1898 - categorical_accuracy: 0.9426
59072/60000 [============================>.] - ETA: 1s - loss: 0.1898 - categorical_accuracy: 0.9427
59104/60000 [============================>.] - ETA: 1s - loss: 0.1898 - categorical_accuracy: 0.9427
59168/60000 [============================>.] - ETA: 1s - loss: 0.1897 - categorical_accuracy: 0.9426
59200/60000 [============================>.] - ETA: 1s - loss: 0.1897 - categorical_accuracy: 0.9427
59232/60000 [============================>.] - ETA: 1s - loss: 0.1898 - categorical_accuracy: 0.9426
59264/60000 [============================>.] - ETA: 1s - loss: 0.1897 - categorical_accuracy: 0.9427
59328/60000 [============================>.] - ETA: 1s - loss: 0.1896 - categorical_accuracy: 0.9427
59360/60000 [============================>.] - ETA: 1s - loss: 0.1895 - categorical_accuracy: 0.9427
59392/60000 [============================>.] - ETA: 0s - loss: 0.1895 - categorical_accuracy: 0.9427
59424/60000 [============================>.] - ETA: 0s - loss: 0.1894 - categorical_accuracy: 0.9428
59456/60000 [============================>.] - ETA: 0s - loss: 0.1894 - categorical_accuracy: 0.9428
59520/60000 [============================>.] - ETA: 0s - loss: 0.1892 - categorical_accuracy: 0.9428
59552/60000 [============================>.] - ETA: 0s - loss: 0.1892 - categorical_accuracy: 0.9429
59584/60000 [============================>.] - ETA: 0s - loss: 0.1891 - categorical_accuracy: 0.9429
59616/60000 [============================>.] - ETA: 0s - loss: 0.1891 - categorical_accuracy: 0.9429
59648/60000 [============================>.] - ETA: 0s - loss: 0.1890 - categorical_accuracy: 0.9429
59680/60000 [============================>.] - ETA: 0s - loss: 0.1891 - categorical_accuracy: 0.9429
59712/60000 [============================>.] - ETA: 0s - loss: 0.1890 - categorical_accuracy: 0.9429
59744/60000 [============================>.] - ETA: 0s - loss: 0.1890 - categorical_accuracy: 0.9429
59776/60000 [============================>.] - ETA: 0s - loss: 0.1889 - categorical_accuracy: 0.9429
59808/60000 [============================>.] - ETA: 0s - loss: 0.1889 - categorical_accuracy: 0.9429
59840/60000 [============================>.] - ETA: 0s - loss: 0.1888 - categorical_accuracy: 0.9430
59872/60000 [============================>.] - ETA: 0s - loss: 0.1887 - categorical_accuracy: 0.9430
59936/60000 [============================>.] - ETA: 0s - loss: 0.1886 - categorical_accuracy: 0.9430
59968/60000 [============================>.] - ETA: 0s - loss: 0.1885 - categorical_accuracy: 0.9430
60000/60000 [==============================] - 102s 2ms/step - loss: 0.1885 - categorical_accuracy: 0.9430 - val_loss: 0.0504 - val_categorical_accuracy: 0.9842

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 16s
  192/10000 [..............................] - ETA: 5s 
  352/10000 [>.............................] - ETA: 4s
  512/10000 [>.............................] - ETA: 3s
  672/10000 [=>............................] - ETA: 3s
  832/10000 [=>............................] - ETA: 3s
  992/10000 [=>............................] - ETA: 3s
 1152/10000 [==>...........................] - ETA: 3s
 1312/10000 [==>...........................] - ETA: 3s
 1472/10000 [===>..........................] - ETA: 3s
 1632/10000 [===>..........................] - ETA: 2s
 1792/10000 [====>.........................] - ETA: 2s
 1952/10000 [====>.........................] - ETA: 2s
 2112/10000 [=====>........................] - ETA: 2s
 2272/10000 [=====>........................] - ETA: 2s
 2432/10000 [======>.......................] - ETA: 2s
 2560/10000 [======>.......................] - ETA: 2s
 2720/10000 [=======>......................] - ETA: 2s
 2880/10000 [=======>......................] - ETA: 2s
 3040/10000 [========>.....................] - ETA: 2s
 3200/10000 [========>.....................] - ETA: 2s
 3360/10000 [=========>....................] - ETA: 2s
 3520/10000 [=========>....................] - ETA: 2s
 3680/10000 [==========>...................] - ETA: 2s
 3840/10000 [==========>...................] - ETA: 2s
 4000/10000 [===========>..................] - ETA: 2s
 4160/10000 [===========>..................] - ETA: 2s
 4320/10000 [===========>..................] - ETA: 1s
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
 7040/10000 [====================>.........] - ETA: 0s
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
10000/10000 [==============================] - 3s 334us/step
[[5.99646155e-09 2.25868568e-08 1.05749706e-07 ... 9.99998331e-01
  7.91176724e-09 5.78254685e-07]
 [2.82777842e-06 3.23802960e-05 9.99956012e-01 ... 4.26670361e-08
  4.87924069e-07 1.06992837e-09]
 [1.33254309e-06 9.99632001e-01 3.96026408e-05 ... 8.24421513e-05
  5.54349863e-05 7.34540299e-06]
 ...
 [2.19142340e-08 8.04460058e-07 8.64730083e-08 ... 1.06630123e-05
  1.08418853e-05 2.03662101e-04]
 [3.56831697e-06 6.38750748e-07 3.14628181e-08 ... 9.39162916e-08
  5.08830789e-03 1.05432787e-06]
 [6.44841691e-07 9.92901619e-08 3.32549498e-05 ... 2.09055551e-09
  7.05907155e-07 1.92584828e-08]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.05043689617647324, 'accuracy_test:': 0.9842000007629395}

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
   bb1eb78..6604896  master     -> origin/master
Updating bb1eb78..6604896
Fast-forward
 error_list/20200517/list_log_jupyter_20200517.md | 1749 +++++++++++-----------
 1 file changed, 874 insertions(+), 875 deletions(-)
[master dd9a284] ml_store
 1 file changed, 1870 insertions(+)
To github.com:arita37/mlmodels_store.git
   6604896..dd9a284  master -> master





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
{'loss': 0.47706859558820724, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-17 16:28:45.011493: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
[master 2fafecc] ml_store
 1 file changed, 234 insertions(+)
To github.com:arita37/mlmodels_store.git
   dd9a284..2fafecc  master -> master





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
[master 113f5cf] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   2fafecc..113f5cf  master -> master





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
 40%|████      | 2/5 [00:16<00:25,  8.37s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.8831178449707843, 'learning_rate': 0.011740880764918391, 'min_data_in_leaf': 30, 'num_leaves': 44} and reward: 0.3886
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xecB\x80Z\xd5;\x87X\r\x00\x00\x00learning_rateq\x02G?\x88\x0b\x9aWJ\xdf\xf2X\x10\x00\x00\x00min_data_in_leafq\x03K\x1eX\n\x00\x00\x00num_leavesq\x04K,u.' and reward: 0.3886
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xecB\x80Z\xd5;\x87X\r\x00\x00\x00learning_rateq\x02G?\x88\x0b\x9aWJ\xdf\xf2X\x10\x00\x00\x00min_data_in_leafq\x03K\x1eX\n\x00\x00\x00num_leavesq\x04K,u.' and reward: 0.3886
 60%|██████    | 3/5 [00:36<00:23, 11.68s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.8449881849304348, 'learning_rate': 0.015786656789035756, 'min_data_in_leaf': 29, 'num_leaves': 57} and reward: 0.3902
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeb\n$\xa9y\x0b4X\r\x00\x00\x00learning_rateq\x02G?\x90*`\x9a}\x03\xd6X\x10\x00\x00\x00min_data_in_leafq\x03K\x1dX\n\x00\x00\x00num_leavesq\x04K9u.' and reward: 0.3902
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeb\n$\xa9y\x0b4X\r\x00\x00\x00learning_rateq\x02G?\x90*`\x9a}\x03\xd6X\x10\x00\x00\x00min_data_in_leafq\x03K\x1dX\n\x00\x00\x00num_leavesq\x04K9u.' and reward: 0.3902
 80%|████████  | 4/5 [00:58<00:14, 14.78s/it] 80%|████████  | 4/5 [00:58<00:14, 14.54s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.7802558127471486, 'learning_rate': 0.022195862116898366, 'min_data_in_leaf': 27, 'num_leaves': 63} and reward: 0.3902
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8\xf7\xdb\t\xc8i\xb4X\r\x00\x00\x00learning_rateq\x02G?\x96\xba\x83\x17\x98+\x99X\x10\x00\x00\x00min_data_in_leafq\x03K\x1bX\n\x00\x00\x00num_leavesq\x04K?u.' and reward: 0.3902
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8\xf7\xdb\t\xc8i\xb4X\r\x00\x00\x00learning_rateq\x02G?\x96\xba\x83\x17\x98+\x99X\x10\x00\x00\x00min_data_in_leafq\x03K\x1bX\n\x00\x00\x00num_leavesq\x04K?u.' and reward: 0.3902
Time for Gradient Boosting hyperparameter optimization: 81.41807079315186
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
 40%|████      | 2/5 [00:56<01:25, 28.36s/it] 40%|████      | 2/5 [00:56<01:25, 28.37s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.24094107558705952, 'embedding_size_factor': 0.9161148503242805, 'layers.choice': 2, 'learning_rate': 0.0001271482750585144, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 9.367354223970784e-08} and reward: 0.2774
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xce\xd7(;\xf4i\xfeX\x15\x00\x00\x00embedding_size_factorq\x03G?\xedP\xd0\x170\xba\x1fX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G? \xaac]\xc1\xcauX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>y%2a\xec\xdbdu.' and reward: 0.2774
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xce\xd7(;\xf4i\xfeX\x15\x00\x00\x00embedding_size_factorq\x03G?\xedP\xd0\x170\xba\x1fX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G? \xaac]\xc1\xcauX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>y%2a\xec\xdbdu.' and reward: 0.2774
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 114.47123718261719
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.74s of the -79.98s of remaining time.
Ensemble size: 42
Ensemble weights: 
[0.35714286 0.02380952 0.         0.28571429 0.07142857 0.26190476]
	0.4006	 = Validation accuracy score
	1.55s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 201.58s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f907ec40c88>

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
   113f5cf..f6ef417  master     -> origin/master
Updating 113f5cf..f6ef417
Fast-forward
 error_list/20200517/list_log_testall_20200517.md | 386 ++++++++++-------------
 1 file changed, 173 insertions(+), 213 deletions(-)
[master 1f8eb7e] ml_store
 1 file changed, 204 insertions(+)
To github.com:arita37/mlmodels_store.git
   f6ef417..1f8eb7e  master -> master





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
[master 1bb60b5] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   1f8eb7e..1bb60b5  master -> master





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
100%|██████████| 10/10 [00:02<00:00,  3.47it/s, avg_epoch_loss=5.22]
INFO:root:Epoch[0] Elapsed time 2.883 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.216484
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.2164839744567875 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7ff5e9fc1518>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7ff5e9fc1518>

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
Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 88.07it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 1078.5835774739583,
    "abs_error": 372.80804443359375,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 2.470207395601272,
    "sMAPE": 0.5153156618862325,
    "MSIS": 98.80829258872382,
    "QuantileLoss[0.5]": 372.80802154541016,
    "Coverage[0.5]": 1.0,
    "RMSE": 32.84179619743656,
    "NRMSE": 0.6914062357355065,
    "ND": 0.6540492007606908,
    "wQuantileLoss[0.5]": 0.6540491606059827,
    "mean_wQuantileLoss": 0.6540491606059827,
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
100%|██████████| 10/10 [00:01<00:00,  7.31it/s, avg_epoch_loss=2.71e+3]
INFO:root:Epoch[0] Elapsed time 1.368 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=2713.411247
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 2713.4112467447917 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7ff5bf5239e8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7ff5bf5239e8>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 141.10it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
100%|██████████| 10/10 [00:02<00:00,  4.65it/s, avg_epoch_loss=5.15]
INFO:root:Epoch[0] Elapsed time 2.150 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.148806
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.148806381225586 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7ff5bc13c5f8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7ff5bc13c5f8>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 148.68it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 455.8248697916667,
    "abs_error": 232.7895965576172,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.5424521858408848,
    "sMAPE": 0.3599831436365695,
    "MSIS": 61.698094713121286,
    "QuantileLoss[0.5]": 232.78961181640625,
    "Coverage[0.5]": 0.9166666666666666,
    "RMSE": 21.350055498561748,
    "NRMSE": 0.44947485260129993,
    "ND": 0.40840280097827575,
    "wQuantileLoss[0.5]": 0.4084028277480811,
    "mean_wQuantileLoss": 0.4084028277480811,
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
 30%|███       | 3/10 [00:12<00:29,  4.22s/it, avg_epoch_loss=6.93] 60%|██████    | 6/10 [00:23<00:16,  4.07s/it, avg_epoch_loss=6.9]  90%|█████████ | 9/10 [00:35<00:03,  3.97s/it, avg_epoch_loss=6.87]100%|██████████| 10/10 [00:38<00:00,  3.87s/it, avg_epoch_loss=6.86]
INFO:root:Epoch[0] Elapsed time 38.735 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.855822
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.855821847915649 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7ff5e2663b00>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7ff5e2663b00>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 115.74it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 52824.770833333336,
    "abs_error": 2699.178955078125,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 17.884624316557495,
    "sMAPE": 1.4091646158187385,
    "MSIS": 715.3849079557585,
    "QuantileLoss[0.5]": 2699.1786193847656,
    "Coverage[0.5]": 1.0,
    "RMSE": 229.83640014874348,
    "NRMSE": 4.838661055763021,
    "ND": 4.735401675575658,
    "wQuantileLoss[0.5]": 4.73540108663994,
    "mean_wQuantileLoss": 4.73540108663994,
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
100%|██████████| 10/10 [00:00<00:00, 45.23it/s, avg_epoch_loss=5.23]
INFO:root:Epoch[0] Elapsed time 0.222 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.228130
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.228130292892456 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7ff5bc0e68d0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7ff5bc0e68d0>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 140.65it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 328.96010335286456,
    "abs_error": 180.833984375,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.1981969065550506,
    "sMAPE": 0.3129660600554028,
    "MSIS": 47.92787626220202,
    "QuantileLoss[0.5]": 180.83398056030273,
    "Coverage[0.5]": 0.6666666666666666,
    "RMSE": 18.137257327194334,
    "NRMSE": 0.381836996361986,
    "ND": 0.3172526041666667,
    "wQuantileLoss[0.5]": 0.3172525974742153,
    "mean_wQuantileLoss": 0.3172525974742153,
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
100%|██████████| 10/10 [00:01<00:00,  7.98it/s, avg_epoch_loss=123]
INFO:root:Epoch[0] Elapsed time 1.254 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=122.866774
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 122.86677375545166 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7ff5bc0eac50>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7ff5bc0eac50>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 146.92it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
 10%|█         | 1/10 [01:50<16:30, 110.01s/it, avg_epoch_loss=0.703] 20%|██        | 2/10 [04:42<17:10, 128.86s/it, avg_epoch_loss=0.686] 30%|███       | 3/10 [07:58<17:21, 148.85s/it, avg_epoch_loss=0.669] 40%|████      | 4/10 [10:45<15:25, 154.30s/it, avg_epoch_loss=0.652] 50%|█████     | 5/10 [14:13<14:12, 170.48s/it, avg_epoch_loss=0.634] 60%|██████    | 6/10 [17:13<11:33, 173.38s/it, avg_epoch_loss=0.617] 70%|███████   | 7/10 [20:26<08:57, 179.30s/it, avg_epoch_loss=0.599] 80%|████████  | 8/10 [23:52<06:14, 187.23s/it, avg_epoch_loss=0.581] 90%|█████████ | 9/10 [27:15<03:11, 191.98s/it, avg_epoch_loss=0.563]100%|██████████| 10/10 [30:21<00:00, 190.17s/it, avg_epoch_loss=0.546]100%|██████████| 10/10 [30:21<00:00, 182.17s/it, avg_epoch_loss=0.546]
INFO:root:Epoch[0] Elapsed time 1821.667 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.546187
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.5461866021156311 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7ff5b6797c50>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7ff5b6797c50>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 19.24it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
   1bb60b5..242659b  master     -> origin/master
Updating 1bb60b5..242659b
Fast-forward
 error_list/20200517/list_log_benchmark_20200517.md |  182 +-
 error_list/20200517/list_log_json_20200517.md      | 1146 ++++++-------
 error_list/20200517/list_log_jupyter_20200517.md   | 1749 ++++++++++----------
 error_list/20200517/list_log_testall_20200517.md   |  386 +++--
 4 files changed, 1747 insertions(+), 1716 deletions(-)
[master 9fd2c52] ml_store
 1 file changed, 508 insertions(+)
To github.com:arita37/mlmodels_store.git
   242659b..9fd2c52  master -> master





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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7f86c2e07860> 

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
[master f2b869d] ml_store
 1 file changed, 109 insertions(+)
To github.com:arita37/mlmodels_store.git
   9fd2c52..f2b869d  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 1.18468624e+00 -1.00016919e+00 -5.93843067e-01  1.04499441e+00
   9.65482331e-01  6.08514698e-01 -6.25342001e-01 -6.93286967e-02
  -1.08392067e-01 -3.43900709e-01]
 [ 9.80427414e-01  1.93752881e+00 -2.30839743e-01  3.66332015e-01
   1.10018476e+00 -1.04458938e+00 -3.44987210e-01  2.05117344e+00
   5.85662000e-01 -2.79308500e+00]
 [ 4.46895161e-01  3.86539145e-01  1.35010682e+00 -8.51455657e-01
   8.50637963e-01  1.00088142e+00 -1.16017010e+00 -3.84832249e-01
   1.45810824e+00 -3.31283170e-01]
 [ 4.41189807e-01  4.79852371e-01 -1.92003697e-01 -1.55269878e+00
  -1.88873982e+00  5.78464420e-01  3.98598388e-01 -9.61263599e-01
  -1.45832446e+00 -3.05376438e+00]
 [ 8.61462558e-01  7.43205537e-02 -1.34501002e+00 -1.99560718e-01
  -1.47533915e+00 -6.54603169e-01 -3.14563862e-01  3.18014296e-01
  -8.90271552e-01 -1.29525789e+00]
 [ 1.66752297e+00  1.22372221e+00 -4.59930104e-01 -5.93679025e-02
  -4.93856997e-01  1.44898940e+00 -1.18110317e+00 -4.77580855e-01
   2.59999942e-02 -7.90799954e-01]
 [ 6.23688521e-01  1.20660790e+00  9.03999174e-01 -2.82863552e-01
  -1.18913787e+00 -2.66326884e-01  1.42361443e+00  1.06897162e+00
   4.03714310e-02  1.57546791e+00]
 [ 1.02817479e+00 -5.08457134e-01  1.76533510e+00  7.77419205e-01
   6.17714185e-01 -1.18771172e-01  4.50155513e-01 -1.98998184e-01
   1.86647138e+00  8.70969803e-01]
 [ 5.69983848e-01 -5.33020326e-01 -1.75458969e-01 -1.42655542e+00
   6.06604307e-01  1.76795995e+00 -1.15985185e-01 -4.75372875e-01
   4.77610182e-01 -9.33914656e-01]
 [ 9.83799588e-01 -4.07240024e-01  9.32721414e-01  1.60564992e-01
  -1.27861800e+00 -1.20149976e-01  1.99759555e-01  3.85602292e-01
   7.18290736e-01 -5.30119800e-01]
 [ 6.81889336e-01 -1.15498263e+00  1.22895559e+00 -1.77632196e-01
   9.98545187e-01 -1.51045638e+00 -2.75846063e-01  1.01120706e+00
  -1.47656266e+00  1.30970591e+00]
 [ 6.13636707e-01  3.16658895e-01  1.34710546e+00 -1.89526695e+00
  -7.60458095e-01  8.97291174e-02 -3.29051549e-01  4.10265745e-01
   8.59870972e-01 -1.04906775e+00]
 [ 7.00175710e-01  5.56073510e-01  8.96864073e-02  1.69380911e+00
   8.82393314e-01  1.96869779e-01 -5.63788735e-01  1.69869255e-01
  -1.16400797e+00 -6.01156801e-01]
 [ 7.88018455e-01  3.01960045e-01  7.00982122e-01 -3.94689681e-01
  -1.20376927e+00 -1.17181338e+00  7.55392029e-01  9.84012237e-01
  -5.59681422e-01 -1.98937450e-01]
 [ 1.83829400e+00  5.02740882e-01  1.29101580e-01  1.55880554e+00
   1.32551412e+00  1.09402696e-01  1.40754000e+00 -1.21974440e+00
   2.44936865e+00  1.61694960e+00]
 [ 9.36211246e-01  2.04377395e-01 -1.49419377e+00  6.12232523e-01
  -9.84377246e-01  7.44884536e-01  4.94341651e-01 -3.62812886e-02
  -8.32395348e-01 -4.46699203e-01]
 [ 1.17867274e+00 -5.99804531e-01 -6.94693595e-01  1.12341216e+00
   1.17899425e+00  3.05267040e-01  1.33526763e-02  1.38877940e+00
  -6.61344243e-01  6.21803504e-01]
 [ 1.16777676e+00 -6.65754518e-01 -1.23312074e+00 -1.67419581e+00
   1.01313574e+00  8.25029824e-01 -1.20464572e-01 -4.98213564e-01
  -3.10984978e-01 -1.18231813e+00]
 [ 1.09488485e+00 -6.96245395e-02 -1.16444148e-01  3.53870427e-01
  -1.44189096e+00 -1.86955017e-01  1.29118890e+00 -1.53236162e-01
  -2.43250851e+00 -2.27729800e+00]
 [ 8.71225789e-01 -2.09752935e-01 -4.56987858e-01  9.35147780e-01
  -8.73535822e-01  1.81252782e+00  9.25501215e-01  1.40109881e-01
  -1.41914878e+00  1.06898597e+00]
 [ 8.88389445e-01  2.82995534e-01  1.79558917e-02  1.08030817e-01
  -8.49671873e-01  2.94176190e-02 -5.03973949e-01 -1.34793129e-01
   1.04921829e+00 -1.27046078e+00]
 [ 8.72267394e-01 -2.51630386e+00 -7.75070287e-01 -5.95667881e-01
   1.02600767e+00 -3.09121319e-01  1.74643509e+00  5.10937774e-01
   1.71066184e+00  1.41640538e-01]
 [ 7.75285326e-01  1.47016034e+00  1.03298378e+00 -8.70008223e-01
   7.86556511e-01  3.69190470e-01 -1.43195745e-01  8.53282186e-01
  -1.39711730e-01 -2.22414029e-01]
 [ 8.15836116e-01 -1.39169388e+00  2.50598029e+00  4.50217742e-01
  -8.82869820e-01  6.27437083e-01 -1.19586151e+00  7.51337235e-01
   1.40395436e-01  1.91979229e+00]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7f26beb9bef0>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7f26d8f0d748> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 0.6675918  -0.45252497 -0.60598132  1.16128569 -1.44620987  1.06996554
   1.92381543 -1.04553425  0.35528451  1.80358898]
 [ 0.46739791 -0.23787527 -0.15449119 -0.75566277 -0.54706224  1.85143789
  -1.46405357  0.20909668  1.55501599 -0.09243232]
 [ 0.72297801  0.18553562  0.91549927  0.39442803 -0.84983074  0.72552256
  -0.15050433  1.49588477  0.67545381 -0.43820027]
 [ 0.99785516 -0.6001388   0.45794708  0.14676526 -0.93355729  0.57180488
   0.57296273 -0.03681766  0.11236849 -0.01781755]
 [ 1.12062155 -0.7029204  -1.22957425  0.72555052 -1.18013412 -0.32420422
   1.10223673  0.81434313  0.78046993  1.10861676]
 [ 0.85982375  0.17195713 -0.34898419  0.49056104 -1.15649503 -1.39528303
   0.61472628 -0.52235647 -0.3692559  -0.977773  ]
 [ 1.16755486  0.0353601   0.7147896  -1.53879325  1.10863359 -0.44789518
  -1.75592564  0.61798553 -0.18417633  0.85270406]
 [ 0.6236295   0.98635218  1.45391758 -0.46615486  0.93640333  1.38499134
   0.03494359 -1.07296428  0.49515861  0.66168108]
 [ 1.12641981 -0.6294416   1.1010002  -1.1134361   0.94459507 -0.06741002
  -0.1834002   1.16143998 -0.02752939  0.78002714]
 [ 1.4468218   0.80745592  1.49810818  0.31223869 -0.68243019 -0.19332164
   0.28807817 -2.07680202  0.94750117 -0.30097615]
 [ 1.32720112 -0.16119832  0.6024509  -0.28638492 -0.5789623  -0.87088765
   1.37975819  0.50142959 -0.47861407 -0.89264667]
 [ 1.07258847 -0.58652394 -1.34267579 -1.23685338  1.24328724  0.87583893
  -0.3264995   0.62336218 -0.43495668  1.11438298]
 [ 0.92686981  0.39233491 -0.4234783   0.44838065 -1.09230828  1.1253235
  -0.94843966  0.10405339  0.52800342  1.00796648]
 [ 0.62567337  0.5924728   0.67457071  1.19783084  1.23187251  1.70459417
  -0.76730983  1.04008915 -0.91844004  1.46089238]
 [ 1.01195228 -1.88141087  1.70018815  0.4972691  -0.91766462  0.2373327
  -1.09033833 -2.14444405 -0.36956243  0.60878366]
 [ 0.81583612 -1.39169388  2.50598029  0.45021774 -0.88286982  0.62743708
  -1.19586151  0.75133724  0.14039544  1.91979229]
 [ 0.87874071 -0.01923163  0.31965694  0.15001628 -1.46662161  0.46353432
  -0.89868319  0.39788042 -0.99601089  0.3181542 ]
 [ 1.17867274 -0.59980453 -0.6946936   1.12341216  1.17899425  0.30526704
   0.01335268  1.3887794  -0.66134424  0.6218035 ]
 [ 1.13545112  0.8616231   0.04906169 -2.08639057 -1.1146902   0.36180164
  -0.80617821  0.42592018  0.0490804  -0.59608633]
 [ 0.85335555 -0.70435033 -0.67938378 -0.04586669 -1.29936179 -0.21873346
   0.59003946  1.53920701 -1.14870423 -0.95090925]
 [ 1.58463774  0.057121   -0.01771832 -0.79954749  1.32970299 -0.2915946
  -1.1077125  -0.25898285  0.1892932  -1.71939447]
 [ 0.10593645 -0.73728963  0.65032321  0.16466507 -1.53556118  0.77817418
   0.05031709  0.30981676  1.05132077  0.6065484 ]
 [ 0.62368852  1.2066079   0.90399917 -0.28286355 -1.18913787 -0.26632688
   1.42361443  1.06897162  0.04037143  1.57546791]
 [ 2.07582971 -1.40232915 -0.47918492  0.45112294  1.03436581 -0.6949209
  -0.4189379   0.5154138  -1.11487105 -1.95210529]
 [ 0.68188934 -1.15498263  1.22895559 -0.1776322   0.99854519 -1.51045638
  -0.27584606  1.01120706 -1.47656266  1.30970591]]
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
[[ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 6.91743730e-01  1.00978733e+00 -1.21333813e+00 -1.55694156e+00
  -1.20257258e+00 -6.12442128e-01 -2.69836174e+00 -1.39351805e-01
  -7.28537489e-01  7.22518992e-02]
 [ 8.58774962e-01  2.29371761e+00 -1.47023709e+00 -8.30010986e-01
  -6.72049816e-01 -1.01951985e+00  5.99213235e-01 -2.14653842e-01
   1.02124813e+00  6.06403944e-01]
 [ 1.16755486e+00  3.53600971e-02  7.14789597e-01 -1.53879325e+00
   1.10863359e+00 -4.47895185e-01 -1.75592564e+00  6.17985534e-01
  -1.84176326e-01  8.52704062e-01]
 [ 2.07582971e+00 -1.40232915e+00 -4.79184915e-01  4.51122939e-01
   1.03436581e+00 -6.94920901e-01 -4.18937898e-01  5.15413802e-01
  -1.11487105e+00 -1.95210529e+00]
 [ 9.64572049e-01 -1.06793987e-01  1.12232832e+00  1.45142926e+00
   1.21828168e+00 -6.18036848e-01  4.38166347e-01 -2.03720123e+00
  -1.94258918e+00 -9.97019796e-01]
 [ 6.21530991e-01 -1.50957268e+00 -1.01932039e-01 -1.08071069e+00
  -1.13742855e+00  7.25474004e-01  7.98063795e-01 -3.91782562e-02
  -2.28754171e-01  7.43356544e-01]
 [ 1.24549398e+00 -7.22391905e-01  1.11813340e+00  1.09899633e+00
   1.00277655e+00 -9.01634490e-01 -5.32234021e-01 -8.22467189e-01
   7.21711292e-01  6.74396105e-01]
 [ 6.10942600e-01 -2.79099641e+00 -1.33520272e+00 -4.56117555e-01
  -9.44959948e-01 -9.79890252e-01 -1.56993672e-01  6.92574348e-01
  -4.78672356e-01 -1.06460122e-01]
 [ 6.25673373e-01  5.92472801e-01  6.74570707e-01  1.19783084e+00
   1.23187251e+00  1.70459417e+00 -7.67309826e-01  1.04008915e+00
  -9.18440038e-01  1.46089238e+00]
 [ 1.01177337e+00  9.57467711e-02  7.31402517e-01  1.03345080e+00
  -1.42203164e+00 -1.46273275e-01 -1.74549518e-02 -8.57496825e-01
  -9.34181843e-01  9.54495667e-01]
 [ 1.17867274e+00 -5.99804531e-01 -6.94693595e-01  1.12341216e+00
   1.17899425e+00  3.05267040e-01  1.33526763e-02  1.38877940e+00
  -6.61344243e-01  6.21803504e-01]
 [ 4.46895161e-01  3.86539145e-01  1.35010682e+00 -8.51455657e-01
   8.50637963e-01  1.00088142e+00 -1.16017010e+00 -3.84832249e-01
   1.45810824e+00 -3.31283170e-01]
 [ 9.80427414e-01  1.93752881e+00 -2.30839743e-01  3.66332015e-01
   1.10018476e+00 -1.04458938e+00 -3.44987210e-01  2.05117344e+00
   5.85662000e-01 -2.79308500e+00]
 [ 1.25704434e+00 -1.82391985e+00 -6.12406973e-01  1.16707517e+00
  -6.23732812e-01 -3.96687001e-02  8.16043684e-01  8.85825799e-01
   1.89861649e-01  3.93109245e-01]
 [ 1.09488485e+00 -6.96245395e-02 -1.16444148e-01  3.53870427e-01
  -1.44189096e+00 -1.86955017e-01  1.29118890e+00 -1.53236162e-01
  -2.43250851e+00 -2.27729800e+00]
 [ 1.14809657e+00 -7.33271604e-01  2.62467445e-01  8.36004719e-01
   1.17353145e+00  1.54335911e+00  2.84748111e-01  7.58805660e-01
   8.84908814e-01  2.76499305e-01]
 [ 4.73307772e-01 -9.73267585e-01 -2.28140691e-01  1.75167729e-01
  -1.01366961e+00 -5.34836927e-02  3.93787731e-01 -1.83061987e-01
  -2.21028902e-01  5.80330113e-01]
 [ 1.07258847e+00 -5.86523939e-01 -1.34267579e+00 -1.23685338e+00
   1.24328724e+00  8.75838928e-01 -3.26499498e-01  6.23362177e-01
  -4.34956683e-01  1.11438298e+00]
 [ 1.98519313e+00  6.74711526e-01 -1.39662042e+00  6.18539131e-01
   1.22382712e+00 -4.43171931e-01 -1.89148284e-03  1.81053491e+00
  -1.30572692e+00 -8.61316361e-01]
 [ 7.75285326e-01  1.47016034e+00  1.03298378e+00 -8.70008223e-01
   7.86556511e-01  3.69190470e-01 -1.43195745e-01  8.53282186e-01
  -1.39711730e-01 -2.22414029e-01]
 [ 1.03967316e+00 -7.31530982e-01  3.61847316e-01 -1.56573815e+00
   9.59288190e-01  1.01382247e+00 -1.78791289e+00 -2.22711263e+00
  -1.69933360e+00 -4.24492791e-01]
 [ 1.01195228e+00 -1.88141087e+00  1.70018815e+00  4.97269099e-01
  -9.17664624e-01  2.37332699e-01 -1.09033833e+00 -2.14444405e+00
  -3.69562425e-01  6.08783659e-01]
 [ 8.98917161e-01  5.57439453e-01 -7.58067329e-01  1.81038744e-01
   8.41467206e-01  1.10717545e+00  6.93366226e-01  1.44287693e+00
  -5.39681562e-01 -8.08847196e-01]
 [ 9.71395338e-01  7.13049050e-01  1.76041518e+00  1.30620607e+00
   1.05765490e+00 -6.04602969e-01  1.28376990e-01  6.36583409e-01
   1.40925339e+00  9.66539250e-01]]
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
[master 06af829] ml_store
 1 file changed, 297 insertions(+)
To github.com:arita37/mlmodels_store.git
   f2b869d..06af829  master -> master





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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140497699385296
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140497699385072
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140497699383840
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140497699383392
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140497699382888
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140497699382552

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
grad_step = 000000, loss = 0.659649
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.527708
grad_step = 000002, loss = 0.442204
grad_step = 000003, loss = 0.359948
grad_step = 000004, loss = 0.272768
grad_step = 000005, loss = 0.187854
grad_step = 000006, loss = 0.121628
grad_step = 000007, loss = 0.108632
grad_step = 000008, loss = 0.109464
grad_step = 000009, loss = 0.077959
grad_step = 000010, loss = 0.040894
grad_step = 000011, loss = 0.022238
grad_step = 000012, loss = 0.019866
grad_step = 000013, loss = 0.021830
grad_step = 000014, loss = 0.021184
grad_step = 000015, loss = 0.017151
grad_step = 000016, loss = 0.011952
grad_step = 000017, loss = 0.009619
grad_step = 000018, loss = 0.011518
grad_step = 000019, loss = 0.014152
grad_step = 000020, loss = 0.014230
grad_step = 000021, loss = 0.013041
grad_step = 000022, loss = 0.011747
grad_step = 000023, loss = 0.011363
grad_step = 000024, loss = 0.011631
grad_step = 000025, loss = 0.011722
grad_step = 000026, loss = 0.010959
grad_step = 000027, loss = 0.009389
grad_step = 000028, loss = 0.007686
grad_step = 000029, loss = 0.006605
grad_step = 000030, loss = 0.006492
grad_step = 000031, loss = 0.006938
grad_step = 000032, loss = 0.007083
grad_step = 000033, loss = 0.006556
grad_step = 000034, loss = 0.005736
grad_step = 000035, loss = 0.005231
grad_step = 000036, loss = 0.005281
grad_step = 000037, loss = 0.005684
grad_step = 000038, loss = 0.006049
grad_step = 000039, loss = 0.006111
grad_step = 000040, loss = 0.005888
grad_step = 000041, loss = 0.005592
grad_step = 000042, loss = 0.005435
grad_step = 000043, loss = 0.005434
grad_step = 000044, loss = 0.005483
grad_step = 000045, loss = 0.005435
grad_step = 000046, loss = 0.005277
grad_step = 000047, loss = 0.005104
grad_step = 000048, loss = 0.005016
grad_step = 000049, loss = 0.005032
grad_step = 000050, loss = 0.005089
grad_step = 000051, loss = 0.005106
grad_step = 000052, loss = 0.005055
grad_step = 000053, loss = 0.004975
grad_step = 000054, loss = 0.004923
grad_step = 000055, loss = 0.004919
grad_step = 000056, loss = 0.004932
grad_step = 000057, loss = 0.004921
grad_step = 000058, loss = 0.004871
grad_step = 000059, loss = 0.004805
grad_step = 000060, loss = 0.004759
grad_step = 000061, loss = 0.004752
grad_step = 000062, loss = 0.004764
grad_step = 000063, loss = 0.004756
grad_step = 000064, loss = 0.004716
grad_step = 000065, loss = 0.004670
grad_step = 000066, loss = 0.004645
grad_step = 000067, loss = 0.004641
grad_step = 000068, loss = 0.004634
grad_step = 000069, loss = 0.004609
grad_step = 000070, loss = 0.004574
grad_step = 000071, loss = 0.004546
grad_step = 000072, loss = 0.004531
grad_step = 000073, loss = 0.004520
grad_step = 000074, loss = 0.004499
grad_step = 000075, loss = 0.004470
grad_step = 000076, loss = 0.004442
grad_step = 000077, loss = 0.004422
grad_step = 000078, loss = 0.004404
grad_step = 000079, loss = 0.004382
grad_step = 000080, loss = 0.004354
grad_step = 000081, loss = 0.004326
grad_step = 000082, loss = 0.004303
grad_step = 000083, loss = 0.004282
grad_step = 000084, loss = 0.004259
grad_step = 000085, loss = 0.004232
grad_step = 000086, loss = 0.004204
grad_step = 000087, loss = 0.004178
grad_step = 000088, loss = 0.004154
grad_step = 000089, loss = 0.004129
grad_step = 000090, loss = 0.004099
grad_step = 000091, loss = 0.004068
grad_step = 000092, loss = 0.004039
grad_step = 000093, loss = 0.004011
grad_step = 000094, loss = 0.003980
grad_step = 000095, loss = 0.003948
grad_step = 000096, loss = 0.003916
grad_step = 000097, loss = 0.003886
grad_step = 000098, loss = 0.003853
grad_step = 000099, loss = 0.003818
grad_step = 000100, loss = 0.003784
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.003748
grad_step = 000102, loss = 0.003713
grad_step = 000103, loss = 0.003674
grad_step = 000104, loss = 0.003637
grad_step = 000105, loss = 0.003598
grad_step = 000106, loss = 0.003558
grad_step = 000107, loss = 0.003518
grad_step = 000108, loss = 0.003476
grad_step = 000109, loss = 0.003434
grad_step = 000110, loss = 0.003390
grad_step = 000111, loss = 0.003347
grad_step = 000112, loss = 0.003300
grad_step = 000113, loss = 0.003252
grad_step = 000114, loss = 0.003202
grad_step = 000115, loss = 0.003150
grad_step = 000116, loss = 0.003097
grad_step = 000117, loss = 0.003041
grad_step = 000118, loss = 0.002983
grad_step = 000119, loss = 0.002924
grad_step = 000120, loss = 0.002864
grad_step = 000121, loss = 0.002801
grad_step = 000122, loss = 0.002739
grad_step = 000123, loss = 0.002679
grad_step = 000124, loss = 0.002615
grad_step = 000125, loss = 0.002550
grad_step = 000126, loss = 0.002482
grad_step = 000127, loss = 0.002418
grad_step = 000128, loss = 0.002355
grad_step = 000129, loss = 0.002295
grad_step = 000130, loss = 0.002238
grad_step = 000131, loss = 0.002197
grad_step = 000132, loss = 0.002190
grad_step = 000133, loss = 0.002167
grad_step = 000134, loss = 0.002069
grad_step = 000135, loss = 0.001999
grad_step = 000136, loss = 0.002010
grad_step = 000137, loss = 0.001995
grad_step = 000138, loss = 0.001911
grad_step = 000139, loss = 0.001853
grad_step = 000140, loss = 0.001852
grad_step = 000141, loss = 0.001846
grad_step = 000142, loss = 0.001785
grad_step = 000143, loss = 0.001713
grad_step = 000144, loss = 0.001680
grad_step = 000145, loss = 0.001678
grad_step = 000146, loss = 0.001663
grad_step = 000147, loss = 0.001610
grad_step = 000148, loss = 0.001541
grad_step = 000149, loss = 0.001494
grad_step = 000150, loss = 0.001472
grad_step = 000151, loss = 0.001467
grad_step = 000152, loss = 0.001464
grad_step = 000153, loss = 0.001405
grad_step = 000154, loss = 0.001341
grad_step = 000155, loss = 0.001301
grad_step = 000156, loss = 0.001283
grad_step = 000157, loss = 0.001272
grad_step = 000158, loss = 0.001243
grad_step = 000159, loss = 0.001195
grad_step = 000160, loss = 0.001146
grad_step = 000161, loss = 0.001112
grad_step = 000162, loss = 0.001091
grad_step = 000163, loss = 0.001073
grad_step = 000164, loss = 0.001065
grad_step = 000165, loss = 0.001060
grad_step = 000166, loss = 0.001057
grad_step = 000167, loss = 0.001051
grad_step = 000168, loss = 0.001020
grad_step = 000169, loss = 0.000960
grad_step = 000170, loss = 0.000904
grad_step = 000171, loss = 0.000876
grad_step = 000172, loss = 0.000879
grad_step = 000173, loss = 0.000904
grad_step = 000174, loss = 0.000920
grad_step = 000175, loss = 0.000910
grad_step = 000176, loss = 0.000843
grad_step = 000177, loss = 0.000794
grad_step = 000178, loss = 0.000793
grad_step = 000179, loss = 0.000809
grad_step = 000180, loss = 0.000797
grad_step = 000181, loss = 0.000754
grad_step = 000182, loss = 0.000725
grad_step = 000183, loss = 0.000722
grad_step = 000184, loss = 0.000725
grad_step = 000185, loss = 0.000724
grad_step = 000186, loss = 0.000705
grad_step = 000187, loss = 0.000724
grad_step = 000188, loss = 0.000850
grad_step = 000189, loss = 0.001029
grad_step = 000190, loss = 0.000834
grad_step = 000191, loss = 0.000642
grad_step = 000192, loss = 0.000754
grad_step = 000193, loss = 0.000762
grad_step = 000194, loss = 0.000622
grad_step = 000195, loss = 0.000648
grad_step = 000196, loss = 0.000709
grad_step = 000197, loss = 0.000630
grad_step = 000198, loss = 0.000590
grad_step = 000199, loss = 0.000651
grad_step = 000200, loss = 0.000621
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.000567
grad_step = 000202, loss = 0.000623
grad_step = 000203, loss = 0.000614
grad_step = 000204, loss = 0.000550
grad_step = 000205, loss = 0.000598
grad_step = 000206, loss = 0.000609
grad_step = 000207, loss = 0.000538
grad_step = 000208, loss = 0.000560
grad_step = 000209, loss = 0.000589
grad_step = 000210, loss = 0.000532
grad_step = 000211, loss = 0.000530
grad_step = 000212, loss = 0.000557
grad_step = 000213, loss = 0.000522
grad_step = 000214, loss = 0.000514
grad_step = 000215, loss = 0.000535
grad_step = 000216, loss = 0.000514
grad_step = 000217, loss = 0.000497
grad_step = 000218, loss = 0.000510
grad_step = 000219, loss = 0.000508
grad_step = 000220, loss = 0.000490
grad_step = 000221, loss = 0.000488
grad_step = 000222, loss = 0.000496
grad_step = 000223, loss = 0.000486
grad_step = 000224, loss = 0.000476
grad_step = 000225, loss = 0.000480
grad_step = 000226, loss = 0.000483
grad_step = 000227, loss = 0.000473
grad_step = 000228, loss = 0.000465
grad_step = 000229, loss = 0.000467
grad_step = 000230, loss = 0.000467
grad_step = 000231, loss = 0.000461
grad_step = 000232, loss = 0.000455
grad_step = 000233, loss = 0.000455
grad_step = 000234, loss = 0.000455
grad_step = 000235, loss = 0.000450
grad_step = 000236, loss = 0.000446
grad_step = 000237, loss = 0.000444
grad_step = 000238, loss = 0.000443
grad_step = 000239, loss = 0.000441
grad_step = 000240, loss = 0.000437
grad_step = 000241, loss = 0.000434
grad_step = 000242, loss = 0.000433
grad_step = 000243, loss = 0.000432
grad_step = 000244, loss = 0.000429
grad_step = 000245, loss = 0.000426
grad_step = 000246, loss = 0.000423
grad_step = 000247, loss = 0.000421
grad_step = 000248, loss = 0.000419
grad_step = 000249, loss = 0.000418
grad_step = 000250, loss = 0.000415
grad_step = 000251, loss = 0.000413
grad_step = 000252, loss = 0.000411
grad_step = 000253, loss = 0.000409
grad_step = 000254, loss = 0.000407
grad_step = 000255, loss = 0.000405
grad_step = 000256, loss = 0.000403
grad_step = 000257, loss = 0.000401
grad_step = 000258, loss = 0.000398
grad_step = 000259, loss = 0.000397
grad_step = 000260, loss = 0.000395
grad_step = 000261, loss = 0.000393
grad_step = 000262, loss = 0.000392
grad_step = 000263, loss = 0.000390
grad_step = 000264, loss = 0.000388
grad_step = 000265, loss = 0.000387
grad_step = 000266, loss = 0.000387
grad_step = 000267, loss = 0.000388
grad_step = 000268, loss = 0.000390
grad_step = 000269, loss = 0.000394
grad_step = 000270, loss = 0.000395
grad_step = 000271, loss = 0.000401
grad_step = 000272, loss = 0.000403
grad_step = 000273, loss = 0.000411
grad_step = 000274, loss = 0.000414
grad_step = 000275, loss = 0.000408
grad_step = 000276, loss = 0.000389
grad_step = 000277, loss = 0.000377
grad_step = 000278, loss = 0.000376
grad_step = 000279, loss = 0.000373
grad_step = 000280, loss = 0.000367
grad_step = 000281, loss = 0.000371
grad_step = 000282, loss = 0.000380
grad_step = 000283, loss = 0.000379
grad_step = 000284, loss = 0.000378
grad_step = 000285, loss = 0.000381
grad_step = 000286, loss = 0.000384
grad_step = 000287, loss = 0.000377
grad_step = 000288, loss = 0.000374
grad_step = 000289, loss = 0.000373
grad_step = 000290, loss = 0.000371
grad_step = 000291, loss = 0.000363
grad_step = 000292, loss = 0.000359
grad_step = 000293, loss = 0.000357
grad_step = 000294, loss = 0.000354
grad_step = 000295, loss = 0.000349
grad_step = 000296, loss = 0.000348
grad_step = 000297, loss = 0.000348
grad_step = 000298, loss = 0.000347
grad_step = 000299, loss = 0.000344
grad_step = 000300, loss = 0.000344
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000344
grad_step = 000302, loss = 0.000345
grad_step = 000303, loss = 0.000346
grad_step = 000304, loss = 0.000354
grad_step = 000305, loss = 0.000370
grad_step = 000306, loss = 0.000412
grad_step = 000307, loss = 0.000475
grad_step = 000308, loss = 0.000642
grad_step = 000309, loss = 0.000724
grad_step = 000310, loss = 0.000793
grad_step = 000311, loss = 0.000482
grad_step = 000312, loss = 0.000348
grad_step = 000313, loss = 0.000519
grad_step = 000314, loss = 0.000557
grad_step = 000315, loss = 0.000441
grad_step = 000316, loss = 0.000340
grad_step = 000317, loss = 0.000450
grad_step = 000318, loss = 0.000513
grad_step = 000319, loss = 0.000358
grad_step = 000320, loss = 0.000385
grad_step = 000321, loss = 0.000480
grad_step = 000322, loss = 0.000365
grad_step = 000323, loss = 0.000354
grad_step = 000324, loss = 0.000435
grad_step = 000325, loss = 0.000369
grad_step = 000326, loss = 0.000333
grad_step = 000327, loss = 0.000382
grad_step = 000328, loss = 0.000363
grad_step = 000329, loss = 0.000329
grad_step = 000330, loss = 0.000348
grad_step = 000331, loss = 0.000353
grad_step = 000332, loss = 0.000329
grad_step = 000333, loss = 0.000329
grad_step = 000334, loss = 0.000345
grad_step = 000335, loss = 0.000335
grad_step = 000336, loss = 0.000320
grad_step = 000337, loss = 0.000330
grad_step = 000338, loss = 0.000335
grad_step = 000339, loss = 0.000321
grad_step = 000340, loss = 0.000320
grad_step = 000341, loss = 0.000328
grad_step = 000342, loss = 0.000322
grad_step = 000343, loss = 0.000317
grad_step = 000344, loss = 0.000317
grad_step = 000345, loss = 0.000320
grad_step = 000346, loss = 0.000320
grad_step = 000347, loss = 0.000314
grad_step = 000348, loss = 0.000313
grad_step = 000349, loss = 0.000317
grad_step = 000350, loss = 0.000314
grad_step = 000351, loss = 0.000310
grad_step = 000352, loss = 0.000311
grad_step = 000353, loss = 0.000312
grad_step = 000354, loss = 0.000310
grad_step = 000355, loss = 0.000308
grad_step = 000356, loss = 0.000308
grad_step = 000357, loss = 0.000308
grad_step = 000358, loss = 0.000307
grad_step = 000359, loss = 0.000306
grad_step = 000360, loss = 0.000304
grad_step = 000361, loss = 0.000304
grad_step = 000362, loss = 0.000305
grad_step = 000363, loss = 0.000304
grad_step = 000364, loss = 0.000302
grad_step = 000365, loss = 0.000301
grad_step = 000366, loss = 0.000301
grad_step = 000367, loss = 0.000301
grad_step = 000368, loss = 0.000301
grad_step = 000369, loss = 0.000300
grad_step = 000370, loss = 0.000299
grad_step = 000371, loss = 0.000298
grad_step = 000372, loss = 0.000298
grad_step = 000373, loss = 0.000298
grad_step = 000374, loss = 0.000297
grad_step = 000375, loss = 0.000296
grad_step = 000376, loss = 0.000295
grad_step = 000377, loss = 0.000295
grad_step = 000378, loss = 0.000295
grad_step = 000379, loss = 0.000295
grad_step = 000380, loss = 0.000294
grad_step = 000381, loss = 0.000294
grad_step = 000382, loss = 0.000293
grad_step = 000383, loss = 0.000292
grad_step = 000384, loss = 0.000292
grad_step = 000385, loss = 0.000291
grad_step = 000386, loss = 0.000291
grad_step = 000387, loss = 0.000291
grad_step = 000388, loss = 0.000290
grad_step = 000389, loss = 0.000290
grad_step = 000390, loss = 0.000290
grad_step = 000391, loss = 0.000289
grad_step = 000392, loss = 0.000289
grad_step = 000393, loss = 0.000287
grad_step = 000394, loss = 0.000287
grad_step = 000395, loss = 0.000286
grad_step = 000396, loss = 0.000286
grad_step = 000397, loss = 0.000286
grad_step = 000398, loss = 0.000286
grad_step = 000399, loss = 0.000287
grad_step = 000400, loss = 0.000287
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000287
grad_step = 000402, loss = 0.000288
grad_step = 000403, loss = 0.000291
grad_step = 000404, loss = 0.000294
grad_step = 000405, loss = 0.000301
grad_step = 000406, loss = 0.000306
grad_step = 000407, loss = 0.000313
grad_step = 000408, loss = 0.000310
grad_step = 000409, loss = 0.000309
grad_step = 000410, loss = 0.000298
grad_step = 000411, loss = 0.000288
grad_step = 000412, loss = 0.000281
grad_step = 000413, loss = 0.000280
grad_step = 000414, loss = 0.000282
grad_step = 000415, loss = 0.000286
grad_step = 000416, loss = 0.000293
grad_step = 000417, loss = 0.000300
grad_step = 000418, loss = 0.000311
grad_step = 000419, loss = 0.000310
grad_step = 000420, loss = 0.000310
grad_step = 000421, loss = 0.000298
grad_step = 000422, loss = 0.000288
grad_step = 000423, loss = 0.000280
grad_step = 000424, loss = 0.000276
grad_step = 000425, loss = 0.000275
grad_step = 000426, loss = 0.000277
grad_step = 000427, loss = 0.000281
grad_step = 000428, loss = 0.000285
grad_step = 000429, loss = 0.000293
grad_step = 000430, loss = 0.000294
grad_step = 000431, loss = 0.000295
grad_step = 000432, loss = 0.000288
grad_step = 000433, loss = 0.000284
grad_step = 000434, loss = 0.000279
grad_step = 000435, loss = 0.000275
grad_step = 000436, loss = 0.000272
grad_step = 000437, loss = 0.000270
grad_step = 000438, loss = 0.000270
grad_step = 000439, loss = 0.000270
grad_step = 000440, loss = 0.000270
grad_step = 000441, loss = 0.000269
grad_step = 000442, loss = 0.000269
grad_step = 000443, loss = 0.000270
grad_step = 000444, loss = 0.000273
grad_step = 000445, loss = 0.000281
grad_step = 000446, loss = 0.000298
grad_step = 000447, loss = 0.000319
grad_step = 000448, loss = 0.000358
grad_step = 000449, loss = 0.000371
grad_step = 000450, loss = 0.000401
grad_step = 000451, loss = 0.000380
grad_step = 000452, loss = 0.000353
grad_step = 000453, loss = 0.000293
grad_step = 000454, loss = 0.000266
grad_step = 000455, loss = 0.000286
grad_step = 000456, loss = 0.000318
grad_step = 000457, loss = 0.000342
grad_step = 000458, loss = 0.000330
grad_step = 000459, loss = 0.000321
grad_step = 000460, loss = 0.000290
grad_step = 000461, loss = 0.000267
grad_step = 000462, loss = 0.000285
grad_step = 000463, loss = 0.000314
grad_step = 000464, loss = 0.000325
grad_step = 000465, loss = 0.000297
grad_step = 000466, loss = 0.000275
grad_step = 000467, loss = 0.000267
grad_step = 000468, loss = 0.000274
grad_step = 000469, loss = 0.000280
grad_step = 000470, loss = 0.000281
grad_step = 000471, loss = 0.000276
grad_step = 000472, loss = 0.000264
grad_step = 000473, loss = 0.000260
grad_step = 000474, loss = 0.000269
grad_step = 000475, loss = 0.000277
grad_step = 000476, loss = 0.000276
grad_step = 000477, loss = 0.000265
grad_step = 000478, loss = 0.000259
grad_step = 000479, loss = 0.000259
grad_step = 000480, loss = 0.000262
grad_step = 000481, loss = 0.000266
grad_step = 000482, loss = 0.000270
grad_step = 000483, loss = 0.000275
grad_step = 000484, loss = 0.000270
grad_step = 000485, loss = 0.000264
grad_step = 000486, loss = 0.000257
grad_step = 000487, loss = 0.000255
grad_step = 000488, loss = 0.000257
grad_step = 000489, loss = 0.000260
grad_step = 000490, loss = 0.000267
grad_step = 000491, loss = 0.000271
grad_step = 000492, loss = 0.000277
grad_step = 000493, loss = 0.000272
grad_step = 000494, loss = 0.000268
grad_step = 000495, loss = 0.000258
grad_step = 000496, loss = 0.000252
grad_step = 000497, loss = 0.000251
grad_step = 000498, loss = 0.000257
grad_step = 000499, loss = 0.000265
grad_step = 000500, loss = 0.000272
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000284
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
[[0.88156885 0.8613852  0.9075011  0.9422611  0.9960786 ]
 [0.8411987  0.90968794 0.9506947  1.0132096  0.9850843 ]
 [0.90547276 0.92457795 0.9993287  0.9912204  0.9551995 ]
 [0.9313562  1.0076474  1.0014691  0.9617479  0.91852146]
 [1.0058001  1.0090194  0.9565669  0.9270757  0.8767338 ]
 [0.99222016 0.96728057 0.92586684 0.8635098  0.8611491 ]
 [0.9581052  0.92055017 0.8618429  0.86974776 0.8337008 ]
 [0.9045886  0.82093894 0.84229195 0.82081866 0.83301157]
 [0.83965516 0.8371916  0.8259266  0.83727074 0.8626982 ]
 [0.8169845  0.81645274 0.8322726  0.8413422  0.8630543 ]
 [0.8151459  0.8226485  0.86651254 0.83151114 0.9221792 ]
 [0.8251338  0.84562683 0.8329536  0.92809284 0.95606816]
 [0.877424   0.8549799  0.90430367 0.9408393  0.9920269 ]
 [0.8412322  0.9241315  0.9620974  1.0197657  0.97851783]
 [0.9280691  0.9469541  1.0040979  0.9914518  0.93831563]
 [0.93753594 1.0168202  0.9889734  0.9442631  0.89807874]
 [1.0145931  1.007733   0.9354451  0.9071737  0.8594674 ]
 [0.9881009  0.94946563 0.9021164  0.8400121  0.84446347]
 [0.9405744  0.90263677 0.8412051  0.8526331  0.82312787]
 [0.9083265  0.82783735 0.83712095 0.81897676 0.83265424]
 [0.85373574 0.8515319  0.83145994 0.8445707  0.86788404]
 [0.8337156  0.8328943  0.83541167 0.8532171  0.8686021 ]
 [0.82202363 0.83411574 0.8829584  0.83779323 0.92665595]
 [0.83207583 0.8587513  0.8425789  0.93216926 0.9631368 ]
 [0.88772035 0.8679414  0.90940714 0.946627   1.0027977 ]
 [0.852407   0.9171294  0.9544441  1.021292   0.99595565]
 [0.91712433 0.93581986 1.0088124  1.0069792  0.9666821 ]
 [0.9433017  1.0244718  1.0127529  0.97640437 0.9287294 ]
 [1.0172356  1.027905   0.9672304  0.94110596 0.88657475]
 [1.0041114  0.9789597  0.9352988  0.86936784 0.86802626]
 [0.9653088  0.9262908  0.8677563  0.87367374 0.8410846 ]]

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
