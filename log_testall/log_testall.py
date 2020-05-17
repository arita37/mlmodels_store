
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
