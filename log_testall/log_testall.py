
  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_all 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '203a72830f23a80c3dd3ee4f0d2ce62ae396cb03', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03

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
[master ec7262d] ml_store
 2 files changed, 62 insertions(+), 10567 deletions(-)
 rewrite log_testall/log_testall.py (99%)
To github.com:arita37/mlmodels_store.git
   af6e3ce..ec7262d  master -> master





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
[master 61690a4] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   ec7262d..61690a4  master -> master





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
[master b9fb398] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   61690a4..b9fb398  master -> master





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
weighted_sequence_layer_1 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         4           sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         24          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-18 04:11:48.099123: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-18 04:11:48.104210: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-18 04:11:48.104416: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55bfb01dced0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-18 04:11:48.104433: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 218
Trainable params: 218
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.4900 - binary_crossentropy: 7.5582500/500 [==============================] - 1s 1ms/sample - loss: 0.5020 - binary_crossentropy: 7.7433 - val_loss: 0.5260 - val_binary_crossentropy: 8.1135

  #### metrics   #################################################### 
{'MSE': 0.514}

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
weighted_sequence_layer_1 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         4           sequence_max[0][0]               
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
Total params: 218
Trainable params: 218
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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_max[0][0]               
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
Total params: 418
Trainable params: 418
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.5300 - binary_crossentropy: 8.1752500/500 [==============================] - 1s 2ms/sample - loss: 0.5100 - binary_crossentropy: 7.8667 - val_loss: 0.5360 - val_binary_crossentropy: 8.2678

  #### metrics   #################################################### 
{'MSE': 0.523}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_max[0][0]               
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
Total params: 418
Trainable params: 418
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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
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
Total params: 567
Trainable params: 567
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2925 - binary_crossentropy: 1.8232500/500 [==============================] - 1s 2ms/sample - loss: 0.2670 - binary_crossentropy: 1.4488 - val_loss: 0.2893 - val_binary_crossentropy: 1.7891

  #### metrics   #################################################### 
{'MSE': 0.27803261967106585}

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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
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
Total params: 567
Trainable params: 567
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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
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
Total params: 463
Trainable params: 463
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2869 - binary_crossentropy: 0.7933500/500 [==============================] - 1s 2ms/sample - loss: 0.3034 - binary_crossentropy: 0.8366 - val_loss: 0.2966 - val_binary_crossentropy: 0.8117

  #### metrics   #################################################### 
{'MSE': 0.2980228885559387}

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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
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
Total params: 463
Trainable params: 463
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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
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
100/500 [=====>........................] - ETA: 2s - loss: 0.2494 - binary_crossentropy: 0.6920500/500 [==============================] - 1s 3ms/sample - loss: 0.2502 - binary_crossentropy: 0.6935 - val_loss: 0.2500 - val_binary_crossentropy: 0.6932

  #### metrics   #################################################### 
{'MSE': 0.249866014409752}

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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-18 04:13:02.620704: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 04:13:02.622505: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 04:13:02.627922: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-18 04:13:02.636703: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-18 04:13:02.638284: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-18 04:13:02.639637: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 04:13:02.641346: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2478 - val_binary_crossentropy: 0.6888
2020-05-18 04:13:03.865270: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 04:13:03.867069: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 04:13:03.871410: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-18 04:13:03.881978: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-18 04:13:03.883650: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-18 04:13:03.884957: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 04:13:03.886275: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2469452011312967}

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
2020-05-18 04:13:25.913451: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 04:13:25.914700: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 04:13:25.918520: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-18 04:13:25.925377: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-18 04:13:25.926455: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-18 04:13:25.927670: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 04:13:25.928807: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2520 - val_binary_crossentropy: 0.6971
2020-05-18 04:13:27.280954: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 04:13:27.282299: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 04:13:27.285127: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-18 04:13:27.291225: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-18 04:13:27.292219: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-18 04:13:27.293136: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 04:13:27.294060: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2524684991929744}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-18 04:14:00.336138: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 04:14:00.341818: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 04:14:00.356449: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-18 04:14:00.380482: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-18 04:14:00.384303: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-18 04:14:00.388003: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 04:14:00.391661: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.4857 - binary_crossentropy: 1.1937 - val_loss: 0.2670 - val_binary_crossentropy: 0.7283
2020-05-18 04:14:02.588686: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 04:14:02.592805: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 04:14:02.603247: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-18 04:14:02.624124: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-18 04:14:02.631494: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-18 04:14:02.634928: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 04:14:02.638430: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.3107769130772035}

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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
Total params: 675
Trainable params: 675
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2649 - binary_crossentropy: 0.7255500/500 [==============================] - 4s 8ms/sample - loss: 0.2676 - binary_crossentropy: 0.7311 - val_loss: 0.2659 - val_binary_crossentropy: 0.7273

  #### metrics   #################################################### 
{'MSE': 0.26582882160121774}

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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
Total params: 675
Trainable params: 675
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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 3, 2)         18          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 2)         10          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 2)         18          sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         2           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         18          sparse_feature_4[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_4[0][0]           
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
Total params: 287
Trainable params: 287
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2591 - binary_crossentropy: 0.7126500/500 [==============================] - 4s 9ms/sample - loss: 0.2685 - binary_crossentropy: 0.7322 - val_loss: 0.2683 - val_binary_crossentropy: 0.7320

  #### metrics   #################################################### 
{'MSE': 0.2662515048723308}

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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 3, 2)         18          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 2)         10          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 2)         18          sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         2           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         18          sparse_feature_4[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_4[0][0]           
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
Total params: 287
Trainable params: 287
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
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         4           sequence_mean[0][0]              
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_mean[0][0]              
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
Total params: 1,904
Trainable params: 1,904
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 5s - loss: 0.2629 - binary_crossentropy: 0.7193500/500 [==============================] - 4s 9ms/sample - loss: 0.2547 - binary_crossentropy: 0.7026 - val_loss: 0.2498 - val_binary_crossentropy: 0.6928

  #### metrics   #################################################### 
{'MSE': 0.25045693278418013}

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
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         4           sequence_mean[0][0]              
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_mean[0][0]              
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
region_10sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         2           regionsequence_max[0][0]         
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
Total params: 120
Trainable params: 120
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2520 - binary_crossentropy: 0.6971500/500 [==============================] - 6s 11ms/sample - loss: 0.2523 - binary_crossentropy: 0.6973 - val_loss: 0.2556 - val_binary_crossentropy: 0.7303

  #### metrics   #################################################### 
{'MSE': 0.25369291591088244}

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
region_10sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         2           regionsequence_max[0][0]         
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
Total params: 120
Trainable params: 120
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_max[0][0]               
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
Total params: 1,372
Trainable params: 1,372
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2802 - binary_crossentropy: 0.7624500/500 [==============================] - 6s 11ms/sample - loss: 0.2726 - binary_crossentropy: 0.7443 - val_loss: 0.2635 - val_binary_crossentropy: 0.7221

  #### metrics   #################################################### 
{'MSE': 0.26412961869539375}

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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_max[0][0]               
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         36          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         36          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 8, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 2, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 5, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 8, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 2, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 5, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 8, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 2, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 8, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 5, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 2, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 5, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           hash_10[0][0]                    
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
Total params: 3,188
Trainable params: 3,108
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2682 - binary_crossentropy: 0.7322500/500 [==============================] - 6s 12ms/sample - loss: 0.2733 - binary_crossentropy: 0.7436 - val_loss: 0.2667 - val_binary_crossentropy: 0.7290

  #### metrics   #################################################### 
{'MSE': 0.2669075422818304}

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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         36          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         36          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 8, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 2, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 5, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 8, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 2, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 5, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 8, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 2, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 8, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 5, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 2, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 5, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           hash_10[0][0]                    
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
Total params: 3,188
Trainable params: 3,108
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
   b9fb398..01c61ed  master     -> origin/master
Updating b9fb398..01c61ed
Fast-forward
 error_list/20200518/list_log_benchmark_20200518.md |  182 ++--
 error_list/20200518/list_log_json_20200518.md      | 1146 ++++++++++----------
 ...-10_203a72830f23a80c3dd3ee4f0d2ce62ae396cb03.py |  795 ++++++++++++++
 3 files changed, 1464 insertions(+), 659 deletions(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-18-04-10_203a72830f23a80c3dd3ee4f0d2ce62ae396cb03.py
[master ecbab8c] ml_store
 1 file changed, 4956 insertions(+)
To github.com:arita37/mlmodels_store.git
   01c61ed..ecbab8c  master -> master





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
[master 63dba53] ml_store
 1 file changed, 51 insertions(+)
To github.com:arita37/mlmodels_store.git
   ecbab8c..63dba53  master -> master





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
[master 79a0873] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   63dba53..79a0873  master -> master





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
[master 4516011] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   79a0873..4516011  master -> master





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

2020-05-18 04:22:44.859435: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-18 04:22:44.865180: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-18 04:22:44.865520: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5639e86680a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-18 04:22:44.865540: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 8s - loss: 1.3840
256/354 [====================>.........] - ETA: 3s - loss: 1.2320
354/354 [==============================] - 15s 43ms/step - loss: 1.3642 - val_loss: 2.9344

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
Already up to date.
[master c2b9ec8] ml_store
 1 file changed, 150 insertions(+)
To github.com:arita37/mlmodels_store.git
   4516011..c2b9ec8  master -> master





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
Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
Already up to date.
[master e224737] ml_store
 1 file changed, 49 insertions(+)
To github.com:arita37/mlmodels_store.git
   c2b9ec8..e224737  master -> master





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
[master fa489da] ml_store
 1 file changed, 45 insertions(+)
To github.com:arita37/mlmodels_store.git
   e224737..fa489da  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1744896/17464789 [=>............................] - ETA: 0s
 9289728/17464789 [==============>...............] - ETA: 0s
12926976/17464789 [=====================>........] - ETA: 0s
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
2020-05-18 04:23:47.945743: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-18 04:23:47.950349: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-18 04:23:47.950666: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b59e458e10 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-18 04:23:47.950685: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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
 2000/25000 [=>............................] - ETA: 9s - loss: 7.4596 - accuracy: 0.5135 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5848 - accuracy: 0.5053
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5861 - accuracy: 0.5052
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6513 - accuracy: 0.5010
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6257 - accuracy: 0.5027
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.5878 - accuracy: 0.5051
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6091 - accuracy: 0.5038
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6070 - accuracy: 0.5039
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5670 - accuracy: 0.5065
11000/25000 [============>.................] - ETA: 4s - loss: 7.5607 - accuracy: 0.5069
12000/25000 [=============>................] - ETA: 4s - loss: 7.5900 - accuracy: 0.5050
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6077 - accuracy: 0.5038
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6064 - accuracy: 0.5039
15000/25000 [=================>............] - ETA: 3s - loss: 7.6043 - accuracy: 0.5041
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6053 - accuracy: 0.5040
17000/25000 [===================>..........] - ETA: 2s - loss: 7.5999 - accuracy: 0.5044
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6300 - accuracy: 0.5024
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6343 - accuracy: 0.5021
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6329 - accuracy: 0.5022
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6287 - accuracy: 0.5025
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6513 - accuracy: 0.5010
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6520 - accuracy: 0.5010
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6564 - accuracy: 0.5007
25000/25000 [==============================] - 9s 377us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7f22330d8320>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f22331daef0> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.7433 - accuracy: 0.4950
 2000/25000 [=>............................] - ETA: 9s - loss: 7.7586 - accuracy: 0.4940 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.8046 - accuracy: 0.4910
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.8391 - accuracy: 0.4888
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7862 - accuracy: 0.4922
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.8200 - accuracy: 0.4900
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.8068 - accuracy: 0.4909
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7510 - accuracy: 0.4945
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7280 - accuracy: 0.4960
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7510 - accuracy: 0.4945
11000/25000 [============>.................] - ETA: 4s - loss: 7.7503 - accuracy: 0.4945
12000/25000 [=============>................] - ETA: 4s - loss: 7.7816 - accuracy: 0.4925
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7987 - accuracy: 0.4914
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7772 - accuracy: 0.4928
15000/25000 [=================>............] - ETA: 3s - loss: 7.7331 - accuracy: 0.4957
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7021 - accuracy: 0.4977
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6811 - accuracy: 0.4991
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6905 - accuracy: 0.4984
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6650 - accuracy: 0.5001
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6536 - accuracy: 0.5009
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6469 - accuracy: 0.5013
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6527 - accuracy: 0.5009
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6580 - accuracy: 0.5006
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6653 - accuracy: 0.5001
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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.6820 - accuracy: 0.4990
 2000/25000 [=>............................] - ETA: 9s - loss: 7.7280 - accuracy: 0.4960 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7024 - accuracy: 0.4977
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7663 - accuracy: 0.4935
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7034 - accuracy: 0.4976
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7663 - accuracy: 0.4935
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7017 - accuracy: 0.4977
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7280 - accuracy: 0.4960
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7705 - accuracy: 0.4932
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7724 - accuracy: 0.4931
11000/25000 [============>.................] - ETA: 4s - loss: 7.7363 - accuracy: 0.4955
12000/25000 [=============>................] - ETA: 4s - loss: 7.7165 - accuracy: 0.4967
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7020 - accuracy: 0.4977
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6852 - accuracy: 0.4988
15000/25000 [=================>............] - ETA: 3s - loss: 7.6584 - accuracy: 0.5005
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6810 - accuracy: 0.4991
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6901 - accuracy: 0.4985
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7007 - accuracy: 0.4978
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6924 - accuracy: 0.4983
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6919 - accuracy: 0.4983
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6827 - accuracy: 0.4990
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6868 - accuracy: 0.4987
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6800 - accuracy: 0.4991
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6762 - accuracy: 0.4994
25000/25000 [==============================] - 9s 367us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
   fa489da..f6309e9  master     -> origin/master
Updating fa489da..f6309e9
Fast-forward
 error_list/20200518/list_log_json_20200518.md    | 1146 +++++++++++-----------
 error_list/20200518/list_log_testall_20200518.md |  386 ++++----
 2 files changed, 746 insertions(+), 786 deletions(-)
[master 13ee1b2] ml_store
 1 file changed, 324 insertions(+)
To github.com:arita37/mlmodels_store.git
   f6309e9..13ee1b2  master -> master





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

13/13 [==============================] - 2s 116ms/step - loss: nan
Epoch 2/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 3/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 4/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 5/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 6/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 7/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 8/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 9/10

13/13 [==============================] - 0s 5ms/step - loss: nan
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
[master 71f9960] ml_store
 1 file changed, 126 insertions(+)
To github.com:arita37/mlmodels_store.git
   13ee1b2..71f9960  master -> master





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
 2629632/11490434 [=====>........................] - ETA: 0s
11067392/11490434 [===========================>..] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:37 - loss: 2.3073 - categorical_accuracy: 0.1562
   64/60000 [..............................] - ETA: 4:48 - loss: 2.2943 - categorical_accuracy: 0.1719
   96/60000 [..............................] - ETA: 3:48 - loss: 2.3325 - categorical_accuracy: 0.1667
  128/60000 [..............................] - ETA: 3:18 - loss: 2.3139 - categorical_accuracy: 0.1641
  160/60000 [..............................] - ETA: 3:02 - loss: 2.2617 - categorical_accuracy: 0.2250
  192/60000 [..............................] - ETA: 2:55 - loss: 2.2260 - categorical_accuracy: 0.2500
  224/60000 [..............................] - ETA: 2:46 - loss: 2.1885 - categorical_accuracy: 0.2634
  256/60000 [..............................] - ETA: 2:39 - loss: 2.1454 - categorical_accuracy: 0.2773
  288/60000 [..............................] - ETA: 2:33 - loss: 2.1152 - categorical_accuracy: 0.3056
  320/60000 [..............................] - ETA: 2:29 - loss: 2.0870 - categorical_accuracy: 0.3125
  352/60000 [..............................] - ETA: 2:25 - loss: 2.0705 - categorical_accuracy: 0.3182
  384/60000 [..............................] - ETA: 2:22 - loss: 2.0413 - categorical_accuracy: 0.3307
  416/60000 [..............................] - ETA: 2:20 - loss: 2.0071 - categorical_accuracy: 0.3438
  448/60000 [..............................] - ETA: 2:17 - loss: 1.9870 - categorical_accuracy: 0.3527
  480/60000 [..............................] - ETA: 2:17 - loss: 1.9532 - categorical_accuracy: 0.3625
  512/60000 [..............................] - ETA: 2:15 - loss: 1.9357 - categorical_accuracy: 0.3633
  544/60000 [..............................] - ETA: 2:13 - loss: 1.9061 - categorical_accuracy: 0.3713
  576/60000 [..............................] - ETA: 2:12 - loss: 1.8563 - categorical_accuracy: 0.3906
  608/60000 [..............................] - ETA: 2:11 - loss: 1.8280 - categorical_accuracy: 0.3997
  640/60000 [..............................] - ETA: 2:10 - loss: 1.8230 - categorical_accuracy: 0.4000
  672/60000 [..............................] - ETA: 2:09 - loss: 1.8314 - categorical_accuracy: 0.4003
  704/60000 [..............................] - ETA: 2:08 - loss: 1.8279 - categorical_accuracy: 0.4006
  736/60000 [..............................] - ETA: 2:07 - loss: 1.7972 - categorical_accuracy: 0.4130
  768/60000 [..............................] - ETA: 2:06 - loss: 1.7810 - categorical_accuracy: 0.4193
  800/60000 [..............................] - ETA: 2:05 - loss: 1.7609 - categorical_accuracy: 0.4238
  832/60000 [..............................] - ETA: 2:05 - loss: 1.7336 - categorical_accuracy: 0.4339
  864/60000 [..............................] - ETA: 2:06 - loss: 1.7013 - categorical_accuracy: 0.4479
  896/60000 [..............................] - ETA: 2:05 - loss: 1.6655 - categorical_accuracy: 0.4587
  928/60000 [..............................] - ETA: 2:04 - loss: 1.6487 - categorical_accuracy: 0.4612
  960/60000 [..............................] - ETA: 2:04 - loss: 1.6224 - categorical_accuracy: 0.4688
  992/60000 [..............................] - ETA: 2:03 - loss: 1.6112 - categorical_accuracy: 0.4708
 1024/60000 [..............................] - ETA: 2:03 - loss: 1.6023 - categorical_accuracy: 0.4746
 1056/60000 [..............................] - ETA: 2:02 - loss: 1.5845 - categorical_accuracy: 0.4811
 1088/60000 [..............................] - ETA: 2:02 - loss: 1.5655 - categorical_accuracy: 0.4871
 1120/60000 [..............................] - ETA: 2:01 - loss: 1.5446 - categorical_accuracy: 0.4946
 1152/60000 [..............................] - ETA: 2:01 - loss: 1.5224 - categorical_accuracy: 0.5009
 1184/60000 [..............................] - ETA: 2:00 - loss: 1.5138 - categorical_accuracy: 0.5025
 1216/60000 [..............................] - ETA: 2:00 - loss: 1.5002 - categorical_accuracy: 0.5082
 1248/60000 [..............................] - ETA: 2:00 - loss: 1.4782 - categorical_accuracy: 0.5152
 1280/60000 [..............................] - ETA: 1:59 - loss: 1.4589 - categorical_accuracy: 0.5203
 1312/60000 [..............................] - ETA: 1:59 - loss: 1.4464 - categorical_accuracy: 0.5244
 1344/60000 [..............................] - ETA: 1:58 - loss: 1.4349 - categorical_accuracy: 0.5268
 1376/60000 [..............................] - ETA: 1:58 - loss: 1.4239 - categorical_accuracy: 0.5291
 1408/60000 [..............................] - ETA: 1:58 - loss: 1.4115 - categorical_accuracy: 0.5348
 1440/60000 [..............................] - ETA: 1:58 - loss: 1.3991 - categorical_accuracy: 0.5396
 1472/60000 [..............................] - ETA: 1:57 - loss: 1.3824 - categorical_accuracy: 0.5435
 1504/60000 [..............................] - ETA: 1:57 - loss: 1.3686 - categorical_accuracy: 0.5479
 1536/60000 [..............................] - ETA: 1:57 - loss: 1.3539 - categorical_accuracy: 0.5521
 1568/60000 [..............................] - ETA: 1:57 - loss: 1.3406 - categorical_accuracy: 0.5561
 1600/60000 [..............................] - ETA: 1:56 - loss: 1.3256 - categorical_accuracy: 0.5606
 1632/60000 [..............................] - ETA: 1:56 - loss: 1.3120 - categorical_accuracy: 0.5662
 1664/60000 [..............................] - ETA: 1:56 - loss: 1.3030 - categorical_accuracy: 0.5685
 1696/60000 [..............................] - ETA: 1:56 - loss: 1.2967 - categorical_accuracy: 0.5702
 1728/60000 [..............................] - ETA: 1:56 - loss: 1.2861 - categorical_accuracy: 0.5735
 1760/60000 [..............................] - ETA: 1:56 - loss: 1.2746 - categorical_accuracy: 0.5767
 1792/60000 [..............................] - ETA: 1:56 - loss: 1.2682 - categorical_accuracy: 0.5792
 1824/60000 [..............................] - ETA: 1:56 - loss: 1.2625 - categorical_accuracy: 0.5806
 1856/60000 [..............................] - ETA: 1:56 - loss: 1.2535 - categorical_accuracy: 0.5835
 1888/60000 [..............................] - ETA: 1:56 - loss: 1.2398 - categorical_accuracy: 0.5885
 1920/60000 [..............................] - ETA: 1:56 - loss: 1.2277 - categorical_accuracy: 0.5927
 1952/60000 [..............................] - ETA: 1:55 - loss: 1.2148 - categorical_accuracy: 0.5973
 1984/60000 [..............................] - ETA: 1:55 - loss: 1.2050 - categorical_accuracy: 0.6003
 2016/60000 [>.............................] - ETA: 1:55 - loss: 1.1922 - categorical_accuracy: 0.6047
 2048/60000 [>.............................] - ETA: 1:55 - loss: 1.1922 - categorical_accuracy: 0.6050
 2080/60000 [>.............................] - ETA: 1:54 - loss: 1.1808 - categorical_accuracy: 0.6087
 2112/60000 [>.............................] - ETA: 1:54 - loss: 1.1691 - categorical_accuracy: 0.6117
 2144/60000 [>.............................] - ETA: 1:54 - loss: 1.1564 - categorical_accuracy: 0.6166
 2176/60000 [>.............................] - ETA: 1:54 - loss: 1.1491 - categorical_accuracy: 0.6195
 2208/60000 [>.............................] - ETA: 1:53 - loss: 1.1422 - categorical_accuracy: 0.6227
 2240/60000 [>.............................] - ETA: 1:53 - loss: 1.1331 - categorical_accuracy: 0.6268
 2272/60000 [>.............................] - ETA: 1:53 - loss: 1.1248 - categorical_accuracy: 0.6298
 2304/60000 [>.............................] - ETA: 1:53 - loss: 1.1164 - categorical_accuracy: 0.6319
 2336/60000 [>.............................] - ETA: 1:53 - loss: 1.1053 - categorical_accuracy: 0.6357
 2368/60000 [>.............................] - ETA: 1:52 - loss: 1.0969 - categorical_accuracy: 0.6377
 2400/60000 [>.............................] - ETA: 1:52 - loss: 1.0888 - categorical_accuracy: 0.6408
 2432/60000 [>.............................] - ETA: 1:52 - loss: 1.0807 - categorical_accuracy: 0.6439
 2464/60000 [>.............................] - ETA: 1:52 - loss: 1.0726 - categorical_accuracy: 0.6469
 2496/60000 [>.............................] - ETA: 1:52 - loss: 1.0634 - categorical_accuracy: 0.6494
 2528/60000 [>.............................] - ETA: 1:52 - loss: 1.0540 - categorical_accuracy: 0.6523
 2560/60000 [>.............................] - ETA: 1:52 - loss: 1.0493 - categorical_accuracy: 0.6539
 2592/60000 [>.............................] - ETA: 1:51 - loss: 1.0405 - categorical_accuracy: 0.6566
 2624/60000 [>.............................] - ETA: 1:52 - loss: 1.0291 - categorical_accuracy: 0.6608
 2656/60000 [>.............................] - ETA: 1:52 - loss: 1.0224 - categorical_accuracy: 0.6630
 2688/60000 [>.............................] - ETA: 1:52 - loss: 1.0139 - categorical_accuracy: 0.6659
 2720/60000 [>.............................] - ETA: 1:51 - loss: 1.0052 - categorical_accuracy: 0.6684
 2752/60000 [>.............................] - ETA: 1:51 - loss: 0.9969 - categorical_accuracy: 0.6719
 2784/60000 [>.............................] - ETA: 1:51 - loss: 0.9906 - categorical_accuracy: 0.6731
 2816/60000 [>.............................] - ETA: 1:51 - loss: 0.9831 - categorical_accuracy: 0.6754
 2848/60000 [>.............................] - ETA: 1:51 - loss: 0.9784 - categorical_accuracy: 0.6773
 2880/60000 [>.............................] - ETA: 1:51 - loss: 0.9697 - categorical_accuracy: 0.6799
 2912/60000 [>.............................] - ETA: 1:51 - loss: 0.9649 - categorical_accuracy: 0.6817
 2944/60000 [>.............................] - ETA: 1:51 - loss: 0.9626 - categorical_accuracy: 0.6824
 2976/60000 [>.............................] - ETA: 1:50 - loss: 0.9564 - categorical_accuracy: 0.6841
 3008/60000 [>.............................] - ETA: 1:50 - loss: 0.9511 - categorical_accuracy: 0.6862
 3040/60000 [>.............................] - ETA: 1:50 - loss: 0.9461 - categorical_accuracy: 0.6872
 3072/60000 [>.............................] - ETA: 1:50 - loss: 0.9400 - categorical_accuracy: 0.6891
 3104/60000 [>.............................] - ETA: 1:50 - loss: 0.9345 - categorical_accuracy: 0.6904
 3136/60000 [>.............................] - ETA: 1:50 - loss: 0.9264 - categorical_accuracy: 0.6932
 3168/60000 [>.............................] - ETA: 1:49 - loss: 0.9252 - categorical_accuracy: 0.6941
 3200/60000 [>.............................] - ETA: 1:49 - loss: 0.9254 - categorical_accuracy: 0.6944
 3232/60000 [>.............................] - ETA: 1:49 - loss: 0.9197 - categorical_accuracy: 0.6968
 3264/60000 [>.............................] - ETA: 1:49 - loss: 0.9133 - categorical_accuracy: 0.6991
 3296/60000 [>.............................] - ETA: 1:49 - loss: 0.9086 - categorical_accuracy: 0.7008
 3328/60000 [>.............................] - ETA: 1:49 - loss: 0.9054 - categorical_accuracy: 0.7016
 3360/60000 [>.............................] - ETA: 1:49 - loss: 0.9016 - categorical_accuracy: 0.7033
 3392/60000 [>.............................] - ETA: 1:48 - loss: 0.8973 - categorical_accuracy: 0.7043
 3424/60000 [>.............................] - ETA: 1:48 - loss: 0.8930 - categorical_accuracy: 0.7053
 3456/60000 [>.............................] - ETA: 1:48 - loss: 0.8901 - categorical_accuracy: 0.7063
 3488/60000 [>.............................] - ETA: 1:48 - loss: 0.8872 - categorical_accuracy: 0.7076
 3520/60000 [>.............................] - ETA: 1:48 - loss: 0.8816 - categorical_accuracy: 0.7097
 3552/60000 [>.............................] - ETA: 1:48 - loss: 0.8799 - categorical_accuracy: 0.7103
 3584/60000 [>.............................] - ETA: 1:48 - loss: 0.8741 - categorical_accuracy: 0.7121
 3616/60000 [>.............................] - ETA: 1:48 - loss: 0.8698 - categorical_accuracy: 0.7138
 3648/60000 [>.............................] - ETA: 1:48 - loss: 0.8649 - categorical_accuracy: 0.7155
 3680/60000 [>.............................] - ETA: 1:48 - loss: 0.8610 - categorical_accuracy: 0.7168
 3712/60000 [>.............................] - ETA: 1:48 - loss: 0.8579 - categorical_accuracy: 0.7182
 3744/60000 [>.............................] - ETA: 1:47 - loss: 0.8519 - categorical_accuracy: 0.7201
 3776/60000 [>.............................] - ETA: 1:47 - loss: 0.8518 - categorical_accuracy: 0.7209
 3808/60000 [>.............................] - ETA: 1:47 - loss: 0.8482 - categorical_accuracy: 0.7224
 3840/60000 [>.............................] - ETA: 1:47 - loss: 0.8453 - categorical_accuracy: 0.7234
 3872/60000 [>.............................] - ETA: 1:47 - loss: 0.8417 - categorical_accuracy: 0.7247
 3904/60000 [>.............................] - ETA: 1:47 - loss: 0.8387 - categorical_accuracy: 0.7259
 3936/60000 [>.............................] - ETA: 1:47 - loss: 0.8332 - categorical_accuracy: 0.7276
 3968/60000 [>.............................] - ETA: 1:47 - loss: 0.8284 - categorical_accuracy: 0.7288
 4000/60000 [=>............................] - ETA: 1:47 - loss: 0.8230 - categorical_accuracy: 0.7305
 4032/60000 [=>............................] - ETA: 1:47 - loss: 0.8195 - categorical_accuracy: 0.7316
 4064/60000 [=>............................] - ETA: 1:47 - loss: 0.8155 - categorical_accuracy: 0.7333
 4096/60000 [=>............................] - ETA: 1:46 - loss: 0.8115 - categorical_accuracy: 0.7344
 4128/60000 [=>............................] - ETA: 1:46 - loss: 0.8077 - categorical_accuracy: 0.7357
 4160/60000 [=>............................] - ETA: 1:46 - loss: 0.8040 - categorical_accuracy: 0.7365
 4192/60000 [=>............................] - ETA: 1:46 - loss: 0.8006 - categorical_accuracy: 0.7381
 4224/60000 [=>............................] - ETA: 1:46 - loss: 0.7986 - categorical_accuracy: 0.7393
 4256/60000 [=>............................] - ETA: 1:46 - loss: 0.7949 - categorical_accuracy: 0.7406
 4288/60000 [=>............................] - ETA: 1:46 - loss: 0.7917 - categorical_accuracy: 0.7418
 4320/60000 [=>............................] - ETA: 1:46 - loss: 0.7891 - categorical_accuracy: 0.7428
 4352/60000 [=>............................] - ETA: 1:46 - loss: 0.7845 - categorical_accuracy: 0.7445
 4384/60000 [=>............................] - ETA: 1:45 - loss: 0.7804 - categorical_accuracy: 0.7457
 4416/60000 [=>............................] - ETA: 1:45 - loss: 0.7755 - categorical_accuracy: 0.7473
 4448/60000 [=>............................] - ETA: 1:45 - loss: 0.7728 - categorical_accuracy: 0.7484
 4480/60000 [=>............................] - ETA: 1:45 - loss: 0.7687 - categorical_accuracy: 0.7498
 4512/60000 [=>............................] - ETA: 1:45 - loss: 0.7656 - categorical_accuracy: 0.7509
 4544/60000 [=>............................] - ETA: 1:45 - loss: 0.7638 - categorical_accuracy: 0.7520
 4576/60000 [=>............................] - ETA: 1:45 - loss: 0.7621 - categorical_accuracy: 0.7531
 4608/60000 [=>............................] - ETA: 1:45 - loss: 0.7590 - categorical_accuracy: 0.7541
 4640/60000 [=>............................] - ETA: 1:45 - loss: 0.7555 - categorical_accuracy: 0.7552
 4672/60000 [=>............................] - ETA: 1:45 - loss: 0.7524 - categorical_accuracy: 0.7558
 4704/60000 [=>............................] - ETA: 1:45 - loss: 0.7488 - categorical_accuracy: 0.7566
 4736/60000 [=>............................] - ETA: 1:45 - loss: 0.7451 - categorical_accuracy: 0.7578
 4768/60000 [=>............................] - ETA: 1:44 - loss: 0.7413 - categorical_accuracy: 0.7592
 4800/60000 [=>............................] - ETA: 1:44 - loss: 0.7378 - categorical_accuracy: 0.7606
 4832/60000 [=>............................] - ETA: 1:44 - loss: 0.7365 - categorical_accuracy: 0.7612
 4864/60000 [=>............................] - ETA: 1:44 - loss: 0.7331 - categorical_accuracy: 0.7623
 4896/60000 [=>............................] - ETA: 1:44 - loss: 0.7301 - categorical_accuracy: 0.7631
 4928/60000 [=>............................] - ETA: 1:44 - loss: 0.7266 - categorical_accuracy: 0.7642
 4960/60000 [=>............................] - ETA: 1:44 - loss: 0.7232 - categorical_accuracy: 0.7655
 4992/60000 [=>............................] - ETA: 1:44 - loss: 0.7217 - categorical_accuracy: 0.7660
 5024/60000 [=>............................] - ETA: 1:44 - loss: 0.7193 - categorical_accuracy: 0.7669
 5056/60000 [=>............................] - ETA: 1:43 - loss: 0.7159 - categorical_accuracy: 0.7680
 5088/60000 [=>............................] - ETA: 1:43 - loss: 0.7139 - categorical_accuracy: 0.7683
 5120/60000 [=>............................] - ETA: 1:43 - loss: 0.7103 - categorical_accuracy: 0.7695
 5152/60000 [=>............................] - ETA: 1:43 - loss: 0.7078 - categorical_accuracy: 0.7704
 5184/60000 [=>............................] - ETA: 1:43 - loss: 0.7052 - categorical_accuracy: 0.7712
 5216/60000 [=>............................] - ETA: 1:43 - loss: 0.7014 - categorical_accuracy: 0.7726
 5248/60000 [=>............................] - ETA: 1:43 - loss: 0.6986 - categorical_accuracy: 0.7736
 5280/60000 [=>............................] - ETA: 1:43 - loss: 0.6950 - categorical_accuracy: 0.7746
 5312/60000 [=>............................] - ETA: 1:43 - loss: 0.6935 - categorical_accuracy: 0.7752
 5344/60000 [=>............................] - ETA: 1:43 - loss: 0.6943 - categorical_accuracy: 0.7754
 5376/60000 [=>............................] - ETA: 1:43 - loss: 0.6921 - categorical_accuracy: 0.7762
 5408/60000 [=>............................] - ETA: 1:43 - loss: 0.6901 - categorical_accuracy: 0.7770
 5440/60000 [=>............................] - ETA: 1:43 - loss: 0.6908 - categorical_accuracy: 0.7772
 5472/60000 [=>............................] - ETA: 1:43 - loss: 0.6875 - categorical_accuracy: 0.7785
 5504/60000 [=>............................] - ETA: 1:42 - loss: 0.6846 - categorical_accuracy: 0.7796
 5536/60000 [=>............................] - ETA: 1:42 - loss: 0.6823 - categorical_accuracy: 0.7803
 5568/60000 [=>............................] - ETA: 1:42 - loss: 0.6819 - categorical_accuracy: 0.7805
 5600/60000 [=>............................] - ETA: 1:42 - loss: 0.6790 - categorical_accuracy: 0.7816
 5632/60000 [=>............................] - ETA: 1:42 - loss: 0.6778 - categorical_accuracy: 0.7825
 5664/60000 [=>............................] - ETA: 1:42 - loss: 0.6752 - categorical_accuracy: 0.7832
 5696/60000 [=>............................] - ETA: 1:42 - loss: 0.6729 - categorical_accuracy: 0.7841
 5728/60000 [=>............................] - ETA: 1:42 - loss: 0.6705 - categorical_accuracy: 0.7847
 5760/60000 [=>............................] - ETA: 1:42 - loss: 0.6681 - categorical_accuracy: 0.7854
 5792/60000 [=>............................] - ETA: 1:42 - loss: 0.6664 - categorical_accuracy: 0.7861
 5824/60000 [=>............................] - ETA: 1:42 - loss: 0.6646 - categorical_accuracy: 0.7866
 5856/60000 [=>............................] - ETA: 1:41 - loss: 0.6633 - categorical_accuracy: 0.7872
 5888/60000 [=>............................] - ETA: 1:41 - loss: 0.6606 - categorical_accuracy: 0.7880
 5920/60000 [=>............................] - ETA: 1:41 - loss: 0.6576 - categorical_accuracy: 0.7890
 5952/60000 [=>............................] - ETA: 1:41 - loss: 0.6557 - categorical_accuracy: 0.7897
 5984/60000 [=>............................] - ETA: 1:41 - loss: 0.6549 - categorical_accuracy: 0.7903
 6016/60000 [==>...........................] - ETA: 1:41 - loss: 0.6518 - categorical_accuracy: 0.7914
 6048/60000 [==>...........................] - ETA: 1:41 - loss: 0.6499 - categorical_accuracy: 0.7918
 6080/60000 [==>...........................] - ETA: 1:41 - loss: 0.6471 - categorical_accuracy: 0.7928
 6112/60000 [==>...........................] - ETA: 1:41 - loss: 0.6457 - categorical_accuracy: 0.7932
 6144/60000 [==>...........................] - ETA: 1:40 - loss: 0.6432 - categorical_accuracy: 0.7939
 6176/60000 [==>...........................] - ETA: 1:40 - loss: 0.6403 - categorical_accuracy: 0.7950
 6208/60000 [==>...........................] - ETA: 1:40 - loss: 0.6375 - categorical_accuracy: 0.7959
 6240/60000 [==>...........................] - ETA: 1:40 - loss: 0.6347 - categorical_accuracy: 0.7970
 6272/60000 [==>...........................] - ETA: 1:40 - loss: 0.6323 - categorical_accuracy: 0.7975
 6304/60000 [==>...........................] - ETA: 1:40 - loss: 0.6303 - categorical_accuracy: 0.7981
 6336/60000 [==>...........................] - ETA: 1:40 - loss: 0.6274 - categorical_accuracy: 0.7991
 6368/60000 [==>...........................] - ETA: 1:40 - loss: 0.6253 - categorical_accuracy: 0.7998
 6400/60000 [==>...........................] - ETA: 1:40 - loss: 0.6226 - categorical_accuracy: 0.8006
 6432/60000 [==>...........................] - ETA: 1:40 - loss: 0.6206 - categorical_accuracy: 0.8012
 6464/60000 [==>...........................] - ETA: 1:40 - loss: 0.6193 - categorical_accuracy: 0.8017
 6496/60000 [==>...........................] - ETA: 1:40 - loss: 0.6172 - categorical_accuracy: 0.8023
 6528/60000 [==>...........................] - ETA: 1:39 - loss: 0.6152 - categorical_accuracy: 0.8030
 6560/60000 [==>...........................] - ETA: 1:39 - loss: 0.6136 - categorical_accuracy: 0.8034
 6592/60000 [==>...........................] - ETA: 1:39 - loss: 0.6122 - categorical_accuracy: 0.8039
 6624/60000 [==>...........................] - ETA: 1:39 - loss: 0.6100 - categorical_accuracy: 0.8046
 6656/60000 [==>...........................] - ETA: 1:39 - loss: 0.6091 - categorical_accuracy: 0.8047
 6688/60000 [==>...........................] - ETA: 1:39 - loss: 0.6074 - categorical_accuracy: 0.8052
 6720/60000 [==>...........................] - ETA: 1:39 - loss: 0.6064 - categorical_accuracy: 0.8057
 6752/60000 [==>...........................] - ETA: 1:39 - loss: 0.6055 - categorical_accuracy: 0.8060
 6784/60000 [==>...........................] - ETA: 1:39 - loss: 0.6037 - categorical_accuracy: 0.8066
 6816/60000 [==>...........................] - ETA: 1:39 - loss: 0.6021 - categorical_accuracy: 0.8072
 6848/60000 [==>...........................] - ETA: 1:39 - loss: 0.6005 - categorical_accuracy: 0.8078
 6880/60000 [==>...........................] - ETA: 1:39 - loss: 0.5981 - categorical_accuracy: 0.8086
 6912/60000 [==>...........................] - ETA: 1:39 - loss: 0.5967 - categorical_accuracy: 0.8089
 6944/60000 [==>...........................] - ETA: 1:39 - loss: 0.5949 - categorical_accuracy: 0.8096
 6976/60000 [==>...........................] - ETA: 1:38 - loss: 0.5938 - categorical_accuracy: 0.8102
 7008/60000 [==>...........................] - ETA: 1:38 - loss: 0.5926 - categorical_accuracy: 0.8108
 7040/60000 [==>...........................] - ETA: 1:38 - loss: 0.5920 - categorical_accuracy: 0.8108
 7072/60000 [==>...........................] - ETA: 1:38 - loss: 0.5905 - categorical_accuracy: 0.8114
 7104/60000 [==>...........................] - ETA: 1:38 - loss: 0.5895 - categorical_accuracy: 0.8115
 7136/60000 [==>...........................] - ETA: 1:38 - loss: 0.5896 - categorical_accuracy: 0.8117
 7168/60000 [==>...........................] - ETA: 1:38 - loss: 0.5882 - categorical_accuracy: 0.8124
 7200/60000 [==>...........................] - ETA: 1:38 - loss: 0.5875 - categorical_accuracy: 0.8125
 7232/60000 [==>...........................] - ETA: 1:38 - loss: 0.5859 - categorical_accuracy: 0.8129
 7264/60000 [==>...........................] - ETA: 1:38 - loss: 0.5839 - categorical_accuracy: 0.8137
 7296/60000 [==>...........................] - ETA: 1:38 - loss: 0.5826 - categorical_accuracy: 0.8140
 7328/60000 [==>...........................] - ETA: 1:38 - loss: 0.5809 - categorical_accuracy: 0.8145
 7360/60000 [==>...........................] - ETA: 1:38 - loss: 0.5794 - categorical_accuracy: 0.8151
 7392/60000 [==>...........................] - ETA: 1:37 - loss: 0.5775 - categorical_accuracy: 0.8157
 7424/60000 [==>...........................] - ETA: 1:37 - loss: 0.5760 - categorical_accuracy: 0.8163
 7456/60000 [==>...........................] - ETA: 1:37 - loss: 0.5744 - categorical_accuracy: 0.8168
 7488/60000 [==>...........................] - ETA: 1:37 - loss: 0.5732 - categorical_accuracy: 0.8172
 7520/60000 [==>...........................] - ETA: 1:37 - loss: 0.5713 - categorical_accuracy: 0.8178
 7552/60000 [==>...........................] - ETA: 1:37 - loss: 0.5713 - categorical_accuracy: 0.8178
 7584/60000 [==>...........................] - ETA: 1:37 - loss: 0.5702 - categorical_accuracy: 0.8179
 7616/60000 [==>...........................] - ETA: 1:37 - loss: 0.5681 - categorical_accuracy: 0.8187
 7648/60000 [==>...........................] - ETA: 1:37 - loss: 0.5665 - categorical_accuracy: 0.8190
 7680/60000 [==>...........................] - ETA: 1:37 - loss: 0.5649 - categorical_accuracy: 0.8195
 7712/60000 [==>...........................] - ETA: 1:37 - loss: 0.5628 - categorical_accuracy: 0.8203
 7744/60000 [==>...........................] - ETA: 1:37 - loss: 0.5618 - categorical_accuracy: 0.8206
 7776/60000 [==>...........................] - ETA: 1:36 - loss: 0.5603 - categorical_accuracy: 0.8211
 7808/60000 [==>...........................] - ETA: 1:36 - loss: 0.5602 - categorical_accuracy: 0.8213
 7840/60000 [==>...........................] - ETA: 1:36 - loss: 0.5592 - categorical_accuracy: 0.8218
 7872/60000 [==>...........................] - ETA: 1:36 - loss: 0.5579 - categorical_accuracy: 0.8223
 7904/60000 [==>...........................] - ETA: 1:36 - loss: 0.5561 - categorical_accuracy: 0.8230
 7936/60000 [==>...........................] - ETA: 1:36 - loss: 0.5544 - categorical_accuracy: 0.8237
 7968/60000 [==>...........................] - ETA: 1:36 - loss: 0.5533 - categorical_accuracy: 0.8242
 8000/60000 [===>..........................] - ETA: 1:36 - loss: 0.5516 - categorical_accuracy: 0.8245
 8032/60000 [===>..........................] - ETA: 1:36 - loss: 0.5500 - categorical_accuracy: 0.8251
 8064/60000 [===>..........................] - ETA: 1:36 - loss: 0.5490 - categorical_accuracy: 0.8251
 8096/60000 [===>..........................] - ETA: 1:36 - loss: 0.5483 - categorical_accuracy: 0.8252
 8128/60000 [===>..........................] - ETA: 1:35 - loss: 0.5474 - categorical_accuracy: 0.8255
 8160/60000 [===>..........................] - ETA: 1:35 - loss: 0.5464 - categorical_accuracy: 0.8256
 8192/60000 [===>..........................] - ETA: 1:35 - loss: 0.5447 - categorical_accuracy: 0.8262
 8224/60000 [===>..........................] - ETA: 1:35 - loss: 0.5429 - categorical_accuracy: 0.8268
 8256/60000 [===>..........................] - ETA: 1:35 - loss: 0.5421 - categorical_accuracy: 0.8268
 8288/60000 [===>..........................] - ETA: 1:35 - loss: 0.5409 - categorical_accuracy: 0.8271
 8320/60000 [===>..........................] - ETA: 1:35 - loss: 0.5398 - categorical_accuracy: 0.8275
 8352/60000 [===>..........................] - ETA: 1:35 - loss: 0.5382 - categorical_accuracy: 0.8281
 8384/60000 [===>..........................] - ETA: 1:35 - loss: 0.5378 - categorical_accuracy: 0.8282
 8416/60000 [===>..........................] - ETA: 1:35 - loss: 0.5385 - categorical_accuracy: 0.8281
 8448/60000 [===>..........................] - ETA: 1:35 - loss: 0.5379 - categorical_accuracy: 0.8282
 8480/60000 [===>..........................] - ETA: 1:35 - loss: 0.5364 - categorical_accuracy: 0.8287
 8512/60000 [===>..........................] - ETA: 1:35 - loss: 0.5351 - categorical_accuracy: 0.8291
 8544/60000 [===>..........................] - ETA: 1:35 - loss: 0.5334 - categorical_accuracy: 0.8297
 8576/60000 [===>..........................] - ETA: 1:35 - loss: 0.5316 - categorical_accuracy: 0.8303
 8608/60000 [===>..........................] - ETA: 1:34 - loss: 0.5305 - categorical_accuracy: 0.8309
 8640/60000 [===>..........................] - ETA: 1:34 - loss: 0.5298 - categorical_accuracy: 0.8313
 8672/60000 [===>..........................] - ETA: 1:34 - loss: 0.5290 - categorical_accuracy: 0.8318
 8704/60000 [===>..........................] - ETA: 1:34 - loss: 0.5287 - categorical_accuracy: 0.8319
 8736/60000 [===>..........................] - ETA: 1:34 - loss: 0.5281 - categorical_accuracy: 0.8323
 8768/60000 [===>..........................] - ETA: 1:34 - loss: 0.5277 - categorical_accuracy: 0.8325
 8800/60000 [===>..........................] - ETA: 1:34 - loss: 0.5268 - categorical_accuracy: 0.8330
 8832/60000 [===>..........................] - ETA: 1:34 - loss: 0.5256 - categorical_accuracy: 0.8332
 8864/60000 [===>..........................] - ETA: 1:34 - loss: 0.5244 - categorical_accuracy: 0.8337
 8896/60000 [===>..........................] - ETA: 1:34 - loss: 0.5231 - categorical_accuracy: 0.8340
 8928/60000 [===>..........................] - ETA: 1:34 - loss: 0.5218 - categorical_accuracy: 0.8345
 8960/60000 [===>..........................] - ETA: 1:34 - loss: 0.5205 - categorical_accuracy: 0.8348
 8992/60000 [===>..........................] - ETA: 1:34 - loss: 0.5190 - categorical_accuracy: 0.8353
 9024/60000 [===>..........................] - ETA: 1:33 - loss: 0.5176 - categorical_accuracy: 0.8358
 9056/60000 [===>..........................] - ETA: 1:33 - loss: 0.5165 - categorical_accuracy: 0.8362
 9088/60000 [===>..........................] - ETA: 1:33 - loss: 0.5153 - categorical_accuracy: 0.8366
 9120/60000 [===>..........................] - ETA: 1:33 - loss: 0.5137 - categorical_accuracy: 0.8372
 9152/60000 [===>..........................] - ETA: 1:33 - loss: 0.5123 - categorical_accuracy: 0.8375
 9184/60000 [===>..........................] - ETA: 1:33 - loss: 0.5110 - categorical_accuracy: 0.8379
 9216/60000 [===>..........................] - ETA: 1:33 - loss: 0.5103 - categorical_accuracy: 0.8382
 9248/60000 [===>..........................] - ETA: 1:33 - loss: 0.5089 - categorical_accuracy: 0.8387
 9280/60000 [===>..........................] - ETA: 1:33 - loss: 0.5076 - categorical_accuracy: 0.8391
 9312/60000 [===>..........................] - ETA: 1:33 - loss: 0.5062 - categorical_accuracy: 0.8397
 9344/60000 [===>..........................] - ETA: 1:33 - loss: 0.5055 - categorical_accuracy: 0.8399
 9376/60000 [===>..........................] - ETA: 1:33 - loss: 0.5041 - categorical_accuracy: 0.8404
 9408/60000 [===>..........................] - ETA: 1:33 - loss: 0.5031 - categorical_accuracy: 0.8408
 9440/60000 [===>..........................] - ETA: 1:33 - loss: 0.5031 - categorical_accuracy: 0.8410
 9472/60000 [===>..........................] - ETA: 1:33 - loss: 0.5027 - categorical_accuracy: 0.8411
 9504/60000 [===>..........................] - ETA: 1:33 - loss: 0.5017 - categorical_accuracy: 0.8415
 9536/60000 [===>..........................] - ETA: 1:32 - loss: 0.5007 - categorical_accuracy: 0.8417
 9568/60000 [===>..........................] - ETA: 1:32 - loss: 0.4993 - categorical_accuracy: 0.8420
 9600/60000 [===>..........................] - ETA: 1:32 - loss: 0.4982 - categorical_accuracy: 0.8423
 9632/60000 [===>..........................] - ETA: 1:32 - loss: 0.4975 - categorical_accuracy: 0.8425
 9664/60000 [===>..........................] - ETA: 1:32 - loss: 0.4973 - categorical_accuracy: 0.8423
 9696/60000 [===>..........................] - ETA: 1:32 - loss: 0.4966 - categorical_accuracy: 0.8425
 9728/60000 [===>..........................] - ETA: 1:32 - loss: 0.4954 - categorical_accuracy: 0.8429
 9760/60000 [===>..........................] - ETA: 1:32 - loss: 0.4943 - categorical_accuracy: 0.8432
 9792/60000 [===>..........................] - ETA: 1:32 - loss: 0.4933 - categorical_accuracy: 0.8434
 9824/60000 [===>..........................] - ETA: 1:32 - loss: 0.4921 - categorical_accuracy: 0.8439
 9856/60000 [===>..........................] - ETA: 1:32 - loss: 0.4913 - categorical_accuracy: 0.8442
 9888/60000 [===>..........................] - ETA: 1:32 - loss: 0.4902 - categorical_accuracy: 0.8445
 9920/60000 [===>..........................] - ETA: 1:32 - loss: 0.4888 - categorical_accuracy: 0.8449
 9952/60000 [===>..........................] - ETA: 1:32 - loss: 0.4877 - categorical_accuracy: 0.8453
 9984/60000 [===>..........................] - ETA: 1:32 - loss: 0.4876 - categorical_accuracy: 0.8454
10016/60000 [====>.........................] - ETA: 1:32 - loss: 0.4865 - categorical_accuracy: 0.8455
10048/60000 [====>.........................] - ETA: 1:32 - loss: 0.4853 - categorical_accuracy: 0.8459
10080/60000 [====>.........................] - ETA: 1:31 - loss: 0.4841 - categorical_accuracy: 0.8463
10112/60000 [====>.........................] - ETA: 1:31 - loss: 0.4827 - categorical_accuracy: 0.8468
10144/60000 [====>.........................] - ETA: 1:31 - loss: 0.4818 - categorical_accuracy: 0.8471
10176/60000 [====>.........................] - ETA: 1:31 - loss: 0.4805 - categorical_accuracy: 0.8476
10208/60000 [====>.........................] - ETA: 1:31 - loss: 0.4805 - categorical_accuracy: 0.8477
10240/60000 [====>.........................] - ETA: 1:31 - loss: 0.4793 - categorical_accuracy: 0.8480
10272/60000 [====>.........................] - ETA: 1:31 - loss: 0.4787 - categorical_accuracy: 0.8484
10304/60000 [====>.........................] - ETA: 1:31 - loss: 0.4776 - categorical_accuracy: 0.8487
10336/60000 [====>.........................] - ETA: 1:31 - loss: 0.4764 - categorical_accuracy: 0.8491
10368/60000 [====>.........................] - ETA: 1:31 - loss: 0.4751 - categorical_accuracy: 0.8495
10400/60000 [====>.........................] - ETA: 1:31 - loss: 0.4742 - categorical_accuracy: 0.8497
10432/60000 [====>.........................] - ETA: 1:31 - loss: 0.4739 - categorical_accuracy: 0.8499
10464/60000 [====>.........................] - ETA: 1:31 - loss: 0.4731 - categorical_accuracy: 0.8499
10496/60000 [====>.........................] - ETA: 1:31 - loss: 0.4718 - categorical_accuracy: 0.8503
10528/60000 [====>.........................] - ETA: 1:31 - loss: 0.4711 - categorical_accuracy: 0.8505
10560/60000 [====>.........................] - ETA: 1:30 - loss: 0.4699 - categorical_accuracy: 0.8509
10592/60000 [====>.........................] - ETA: 1:30 - loss: 0.4686 - categorical_accuracy: 0.8513
10624/60000 [====>.........................] - ETA: 1:30 - loss: 0.4676 - categorical_accuracy: 0.8516
10656/60000 [====>.........................] - ETA: 1:30 - loss: 0.4666 - categorical_accuracy: 0.8519
10688/60000 [====>.........................] - ETA: 1:30 - loss: 0.4656 - categorical_accuracy: 0.8522
10720/60000 [====>.........................] - ETA: 1:30 - loss: 0.4648 - categorical_accuracy: 0.8524
10752/60000 [====>.........................] - ETA: 1:30 - loss: 0.4637 - categorical_accuracy: 0.8527
10784/60000 [====>.........................] - ETA: 1:30 - loss: 0.4631 - categorical_accuracy: 0.8529
10816/60000 [====>.........................] - ETA: 1:30 - loss: 0.4621 - categorical_accuracy: 0.8532
10848/60000 [====>.........................] - ETA: 1:30 - loss: 0.4613 - categorical_accuracy: 0.8534
10880/60000 [====>.........................] - ETA: 1:30 - loss: 0.4606 - categorical_accuracy: 0.8536
10912/60000 [====>.........................] - ETA: 1:30 - loss: 0.4597 - categorical_accuracy: 0.8538
10944/60000 [====>.........................] - ETA: 1:30 - loss: 0.4586 - categorical_accuracy: 0.8543
10976/60000 [====>.........................] - ETA: 1:30 - loss: 0.4577 - categorical_accuracy: 0.8545
11008/60000 [====>.........................] - ETA: 1:30 - loss: 0.4569 - categorical_accuracy: 0.8547
11040/60000 [====>.........................] - ETA: 1:30 - loss: 0.4565 - categorical_accuracy: 0.8550
11072/60000 [====>.........................] - ETA: 1:29 - loss: 0.4558 - categorical_accuracy: 0.8551
11104/60000 [====>.........................] - ETA: 1:29 - loss: 0.4550 - categorical_accuracy: 0.8553
11136/60000 [====>.........................] - ETA: 1:29 - loss: 0.4538 - categorical_accuracy: 0.8557
11168/60000 [====>.........................] - ETA: 1:29 - loss: 0.4526 - categorical_accuracy: 0.8561
11200/60000 [====>.........................] - ETA: 1:29 - loss: 0.4521 - categorical_accuracy: 0.8562
11232/60000 [====>.........................] - ETA: 1:29 - loss: 0.4513 - categorical_accuracy: 0.8564
11264/60000 [====>.........................] - ETA: 1:29 - loss: 0.4511 - categorical_accuracy: 0.8565
11296/60000 [====>.........................] - ETA: 1:29 - loss: 0.4507 - categorical_accuracy: 0.8566
11328/60000 [====>.........................] - ETA: 1:29 - loss: 0.4512 - categorical_accuracy: 0.8565
11360/60000 [====>.........................] - ETA: 1:29 - loss: 0.4506 - categorical_accuracy: 0.8567
11392/60000 [====>.........................] - ETA: 1:29 - loss: 0.4504 - categorical_accuracy: 0.8569
11424/60000 [====>.........................] - ETA: 1:29 - loss: 0.4494 - categorical_accuracy: 0.8572
11456/60000 [====>.........................] - ETA: 1:29 - loss: 0.4486 - categorical_accuracy: 0.8575
11488/60000 [====>.........................] - ETA: 1:29 - loss: 0.4480 - categorical_accuracy: 0.8576
11520/60000 [====>.........................] - ETA: 1:29 - loss: 0.4473 - categorical_accuracy: 0.8579
11552/60000 [====>.........................] - ETA: 1:29 - loss: 0.4467 - categorical_accuracy: 0.8581
11584/60000 [====>.........................] - ETA: 1:28 - loss: 0.4463 - categorical_accuracy: 0.8583
11616/60000 [====>.........................] - ETA: 1:28 - loss: 0.4454 - categorical_accuracy: 0.8586
11648/60000 [====>.........................] - ETA: 1:28 - loss: 0.4447 - categorical_accuracy: 0.8588
11680/60000 [====>.........................] - ETA: 1:28 - loss: 0.4439 - categorical_accuracy: 0.8589
11712/60000 [====>.........................] - ETA: 1:28 - loss: 0.4429 - categorical_accuracy: 0.8591
11744/60000 [====>.........................] - ETA: 1:28 - loss: 0.4420 - categorical_accuracy: 0.8594
11776/60000 [====>.........................] - ETA: 1:28 - loss: 0.4413 - categorical_accuracy: 0.8596
11808/60000 [====>.........................] - ETA: 1:28 - loss: 0.4414 - categorical_accuracy: 0.8596
11840/60000 [====>.........................] - ETA: 1:28 - loss: 0.4409 - categorical_accuracy: 0.8597
11872/60000 [====>.........................] - ETA: 1:28 - loss: 0.4401 - categorical_accuracy: 0.8600
11904/60000 [====>.........................] - ETA: 1:28 - loss: 0.4390 - categorical_accuracy: 0.8604
11936/60000 [====>.........................] - ETA: 1:28 - loss: 0.4385 - categorical_accuracy: 0.8606
11968/60000 [====>.........................] - ETA: 1:28 - loss: 0.4375 - categorical_accuracy: 0.8610
12000/60000 [=====>........................] - ETA: 1:28 - loss: 0.4366 - categorical_accuracy: 0.8612
12032/60000 [=====>........................] - ETA: 1:28 - loss: 0.4358 - categorical_accuracy: 0.8615
12064/60000 [=====>........................] - ETA: 1:27 - loss: 0.4359 - categorical_accuracy: 0.8613
12096/60000 [=====>........................] - ETA: 1:27 - loss: 0.4351 - categorical_accuracy: 0.8616
12128/60000 [=====>........................] - ETA: 1:27 - loss: 0.4346 - categorical_accuracy: 0.8616
12160/60000 [=====>........................] - ETA: 1:27 - loss: 0.4344 - categorical_accuracy: 0.8618
12192/60000 [=====>........................] - ETA: 1:27 - loss: 0.4335 - categorical_accuracy: 0.8621
12224/60000 [=====>........................] - ETA: 1:27 - loss: 0.4326 - categorical_accuracy: 0.8624
12256/60000 [=====>........................] - ETA: 1:27 - loss: 0.4317 - categorical_accuracy: 0.8627
12288/60000 [=====>........................] - ETA: 1:27 - loss: 0.4310 - categorical_accuracy: 0.8629
12320/60000 [=====>........................] - ETA: 1:27 - loss: 0.4302 - categorical_accuracy: 0.8631
12352/60000 [=====>........................] - ETA: 1:27 - loss: 0.4296 - categorical_accuracy: 0.8634
12384/60000 [=====>........................] - ETA: 1:27 - loss: 0.4287 - categorical_accuracy: 0.8637
12416/60000 [=====>........................] - ETA: 1:27 - loss: 0.4276 - categorical_accuracy: 0.8640
12448/60000 [=====>........................] - ETA: 1:27 - loss: 0.4272 - categorical_accuracy: 0.8642
12480/60000 [=====>........................] - ETA: 1:27 - loss: 0.4262 - categorical_accuracy: 0.8646
12512/60000 [=====>........................] - ETA: 1:27 - loss: 0.4257 - categorical_accuracy: 0.8646
12544/60000 [=====>........................] - ETA: 1:26 - loss: 0.4250 - categorical_accuracy: 0.8648
12576/60000 [=====>........................] - ETA: 1:26 - loss: 0.4245 - categorical_accuracy: 0.8649
12608/60000 [=====>........................] - ETA: 1:26 - loss: 0.4235 - categorical_accuracy: 0.8652
12640/60000 [=====>........................] - ETA: 1:26 - loss: 0.4227 - categorical_accuracy: 0.8655
12672/60000 [=====>........................] - ETA: 1:26 - loss: 0.4220 - categorical_accuracy: 0.8657
12704/60000 [=====>........................] - ETA: 1:26 - loss: 0.4216 - categorical_accuracy: 0.8658
12736/60000 [=====>........................] - ETA: 1:26 - loss: 0.4216 - categorical_accuracy: 0.8659
12768/60000 [=====>........................] - ETA: 1:26 - loss: 0.4206 - categorical_accuracy: 0.8662
12800/60000 [=====>........................] - ETA: 1:26 - loss: 0.4201 - categorical_accuracy: 0.8664
12832/60000 [=====>........................] - ETA: 1:26 - loss: 0.4194 - categorical_accuracy: 0.8665
12864/60000 [=====>........................] - ETA: 1:26 - loss: 0.4189 - categorical_accuracy: 0.8668
12896/60000 [=====>........................] - ETA: 1:26 - loss: 0.4186 - categorical_accuracy: 0.8669
12928/60000 [=====>........................] - ETA: 1:26 - loss: 0.4182 - categorical_accuracy: 0.8670
12960/60000 [=====>........................] - ETA: 1:26 - loss: 0.4175 - categorical_accuracy: 0.8672
12992/60000 [=====>........................] - ETA: 1:26 - loss: 0.4167 - categorical_accuracy: 0.8674
13024/60000 [=====>........................] - ETA: 1:26 - loss: 0.4160 - categorical_accuracy: 0.8676
13056/60000 [=====>........................] - ETA: 1:26 - loss: 0.4157 - categorical_accuracy: 0.8677
13088/60000 [=====>........................] - ETA: 1:25 - loss: 0.4153 - categorical_accuracy: 0.8677
13120/60000 [=====>........................] - ETA: 1:25 - loss: 0.4146 - categorical_accuracy: 0.8680
13152/60000 [=====>........................] - ETA: 1:25 - loss: 0.4136 - categorical_accuracy: 0.8683
13184/60000 [=====>........................] - ETA: 1:25 - loss: 0.4129 - categorical_accuracy: 0.8685
13216/60000 [=====>........................] - ETA: 1:25 - loss: 0.4124 - categorical_accuracy: 0.8686
13248/60000 [=====>........................] - ETA: 1:25 - loss: 0.4118 - categorical_accuracy: 0.8688
13280/60000 [=====>........................] - ETA: 1:25 - loss: 0.4111 - categorical_accuracy: 0.8691
13312/60000 [=====>........................] - ETA: 1:25 - loss: 0.4102 - categorical_accuracy: 0.8694
13344/60000 [=====>........................] - ETA: 1:25 - loss: 0.4094 - categorical_accuracy: 0.8696
13376/60000 [=====>........................] - ETA: 1:25 - loss: 0.4085 - categorical_accuracy: 0.8699
13408/60000 [=====>........................] - ETA: 1:25 - loss: 0.4078 - categorical_accuracy: 0.8702
13440/60000 [=====>........................] - ETA: 1:25 - loss: 0.4071 - categorical_accuracy: 0.8703
13472/60000 [=====>........................] - ETA: 1:25 - loss: 0.4065 - categorical_accuracy: 0.8705
13504/60000 [=====>........................] - ETA: 1:25 - loss: 0.4057 - categorical_accuracy: 0.8707
13536/60000 [=====>........................] - ETA: 1:25 - loss: 0.4058 - categorical_accuracy: 0.8707
13568/60000 [=====>........................] - ETA: 1:24 - loss: 0.4058 - categorical_accuracy: 0.8709
13600/60000 [=====>........................] - ETA: 1:24 - loss: 0.4051 - categorical_accuracy: 0.8710
13632/60000 [=====>........................] - ETA: 1:24 - loss: 0.4053 - categorical_accuracy: 0.8712
13664/60000 [=====>........................] - ETA: 1:24 - loss: 0.4047 - categorical_accuracy: 0.8713
13696/60000 [=====>........................] - ETA: 1:24 - loss: 0.4043 - categorical_accuracy: 0.8714
13728/60000 [=====>........................] - ETA: 1:24 - loss: 0.4038 - categorical_accuracy: 0.8716
13760/60000 [=====>........................] - ETA: 1:24 - loss: 0.4031 - categorical_accuracy: 0.8718
13792/60000 [=====>........................] - ETA: 1:24 - loss: 0.4025 - categorical_accuracy: 0.8720
13824/60000 [=====>........................] - ETA: 1:24 - loss: 0.4023 - categorical_accuracy: 0.8720
13856/60000 [=====>........................] - ETA: 1:24 - loss: 0.4017 - categorical_accuracy: 0.8723
13888/60000 [=====>........................] - ETA: 1:24 - loss: 0.4016 - categorical_accuracy: 0.8726
13920/60000 [=====>........................] - ETA: 1:24 - loss: 0.4011 - categorical_accuracy: 0.8727
13952/60000 [=====>........................] - ETA: 1:24 - loss: 0.4004 - categorical_accuracy: 0.8729
13984/60000 [=====>........................] - ETA: 1:24 - loss: 0.4001 - categorical_accuracy: 0.8730
14016/60000 [======>.......................] - ETA: 1:24 - loss: 0.3997 - categorical_accuracy: 0.8731
14048/60000 [======>.......................] - ETA: 1:24 - loss: 0.3990 - categorical_accuracy: 0.8734
14080/60000 [======>.......................] - ETA: 1:24 - loss: 0.3982 - categorical_accuracy: 0.8736
14112/60000 [======>.......................] - ETA: 1:23 - loss: 0.3974 - categorical_accuracy: 0.8739
14144/60000 [======>.......................] - ETA: 1:23 - loss: 0.3968 - categorical_accuracy: 0.8740
14176/60000 [======>.......................] - ETA: 1:23 - loss: 0.3961 - categorical_accuracy: 0.8742
14208/60000 [======>.......................] - ETA: 1:23 - loss: 0.3956 - categorical_accuracy: 0.8744
14240/60000 [======>.......................] - ETA: 1:23 - loss: 0.3950 - categorical_accuracy: 0.8746
14272/60000 [======>.......................] - ETA: 1:23 - loss: 0.3950 - categorical_accuracy: 0.8748
14304/60000 [======>.......................] - ETA: 1:23 - loss: 0.3943 - categorical_accuracy: 0.8751
14336/60000 [======>.......................] - ETA: 1:23 - loss: 0.3939 - categorical_accuracy: 0.8751
14368/60000 [======>.......................] - ETA: 1:23 - loss: 0.3934 - categorical_accuracy: 0.8753
14400/60000 [======>.......................] - ETA: 1:23 - loss: 0.3929 - categorical_accuracy: 0.8755
14432/60000 [======>.......................] - ETA: 1:23 - loss: 0.3923 - categorical_accuracy: 0.8757
14464/60000 [======>.......................] - ETA: 1:23 - loss: 0.3920 - categorical_accuracy: 0.8758
14496/60000 [======>.......................] - ETA: 1:23 - loss: 0.3918 - categorical_accuracy: 0.8760
14528/60000 [======>.......................] - ETA: 1:23 - loss: 0.3910 - categorical_accuracy: 0.8762
14560/60000 [======>.......................] - ETA: 1:23 - loss: 0.3904 - categorical_accuracy: 0.8764
14592/60000 [======>.......................] - ETA: 1:22 - loss: 0.3897 - categorical_accuracy: 0.8766
14624/60000 [======>.......................] - ETA: 1:22 - loss: 0.3893 - categorical_accuracy: 0.8768
14656/60000 [======>.......................] - ETA: 1:22 - loss: 0.3887 - categorical_accuracy: 0.8769
14688/60000 [======>.......................] - ETA: 1:22 - loss: 0.3882 - categorical_accuracy: 0.8770
14720/60000 [======>.......................] - ETA: 1:22 - loss: 0.3882 - categorical_accuracy: 0.8772
14752/60000 [======>.......................] - ETA: 1:22 - loss: 0.3878 - categorical_accuracy: 0.8773
14784/60000 [======>.......................] - ETA: 1:22 - loss: 0.3874 - categorical_accuracy: 0.8774
14816/60000 [======>.......................] - ETA: 1:22 - loss: 0.3871 - categorical_accuracy: 0.8774
14848/60000 [======>.......................] - ETA: 1:22 - loss: 0.3867 - categorical_accuracy: 0.8775
14880/60000 [======>.......................] - ETA: 1:22 - loss: 0.3859 - categorical_accuracy: 0.8778
14912/60000 [======>.......................] - ETA: 1:22 - loss: 0.3854 - categorical_accuracy: 0.8779
14944/60000 [======>.......................] - ETA: 1:22 - loss: 0.3847 - categorical_accuracy: 0.8781
14976/60000 [======>.......................] - ETA: 1:22 - loss: 0.3842 - categorical_accuracy: 0.8782
15008/60000 [======>.......................] - ETA: 1:22 - loss: 0.3834 - categorical_accuracy: 0.8785
15040/60000 [======>.......................] - ETA: 1:22 - loss: 0.3828 - categorical_accuracy: 0.8787
15072/60000 [======>.......................] - ETA: 1:22 - loss: 0.3821 - categorical_accuracy: 0.8789
15104/60000 [======>.......................] - ETA: 1:22 - loss: 0.3822 - categorical_accuracy: 0.8789
15136/60000 [======>.......................] - ETA: 1:22 - loss: 0.3815 - categorical_accuracy: 0.8792
15168/60000 [======>.......................] - ETA: 1:21 - loss: 0.3808 - categorical_accuracy: 0.8794
15200/60000 [======>.......................] - ETA: 1:21 - loss: 0.3807 - categorical_accuracy: 0.8795
15232/60000 [======>.......................] - ETA: 1:21 - loss: 0.3802 - categorical_accuracy: 0.8795
15264/60000 [======>.......................] - ETA: 1:21 - loss: 0.3798 - categorical_accuracy: 0.8796
15296/60000 [======>.......................] - ETA: 1:21 - loss: 0.3792 - categorical_accuracy: 0.8798
15328/60000 [======>.......................] - ETA: 1:21 - loss: 0.3787 - categorical_accuracy: 0.8800
15360/60000 [======>.......................] - ETA: 1:21 - loss: 0.3782 - categorical_accuracy: 0.8801
15392/60000 [======>.......................] - ETA: 1:21 - loss: 0.3777 - categorical_accuracy: 0.8803
15424/60000 [======>.......................] - ETA: 1:21 - loss: 0.3770 - categorical_accuracy: 0.8805
15456/60000 [======>.......................] - ETA: 1:21 - loss: 0.3763 - categorical_accuracy: 0.8808
15488/60000 [======>.......................] - ETA: 1:21 - loss: 0.3758 - categorical_accuracy: 0.8809
15520/60000 [======>.......................] - ETA: 1:21 - loss: 0.3752 - categorical_accuracy: 0.8811
15552/60000 [======>.......................] - ETA: 1:21 - loss: 0.3746 - categorical_accuracy: 0.8813
15584/60000 [======>.......................] - ETA: 1:21 - loss: 0.3739 - categorical_accuracy: 0.8815
15616/60000 [======>.......................] - ETA: 1:21 - loss: 0.3732 - categorical_accuracy: 0.8818
15648/60000 [======>.......................] - ETA: 1:21 - loss: 0.3727 - categorical_accuracy: 0.8820
15680/60000 [======>.......................] - ETA: 1:21 - loss: 0.3726 - categorical_accuracy: 0.8821
15712/60000 [======>.......................] - ETA: 1:21 - loss: 0.3720 - categorical_accuracy: 0.8823
15744/60000 [======>.......................] - ETA: 1:20 - loss: 0.3716 - categorical_accuracy: 0.8824
15776/60000 [======>.......................] - ETA: 1:20 - loss: 0.3710 - categorical_accuracy: 0.8826
15808/60000 [======>.......................] - ETA: 1:20 - loss: 0.3704 - categorical_accuracy: 0.8828
15840/60000 [======>.......................] - ETA: 1:20 - loss: 0.3701 - categorical_accuracy: 0.8829
15872/60000 [======>.......................] - ETA: 1:20 - loss: 0.3696 - categorical_accuracy: 0.8830
15904/60000 [======>.......................] - ETA: 1:20 - loss: 0.3697 - categorical_accuracy: 0.8829
15936/60000 [======>.......................] - ETA: 1:20 - loss: 0.3691 - categorical_accuracy: 0.8832
15968/60000 [======>.......................] - ETA: 1:20 - loss: 0.3686 - categorical_accuracy: 0.8833
16000/60000 [=======>......................] - ETA: 1:20 - loss: 0.3681 - categorical_accuracy: 0.8834
16032/60000 [=======>......................] - ETA: 1:20 - loss: 0.3681 - categorical_accuracy: 0.8836
16064/60000 [=======>......................] - ETA: 1:20 - loss: 0.3677 - categorical_accuracy: 0.8838
16096/60000 [=======>......................] - ETA: 1:20 - loss: 0.3671 - categorical_accuracy: 0.8839
16128/60000 [=======>......................] - ETA: 1:20 - loss: 0.3664 - categorical_accuracy: 0.8842
16160/60000 [=======>......................] - ETA: 1:20 - loss: 0.3663 - categorical_accuracy: 0.8842
16192/60000 [=======>......................] - ETA: 1:20 - loss: 0.3660 - categorical_accuracy: 0.8843
16224/60000 [=======>......................] - ETA: 1:20 - loss: 0.3656 - categorical_accuracy: 0.8844
16256/60000 [=======>......................] - ETA: 1:19 - loss: 0.3665 - categorical_accuracy: 0.8843
16288/60000 [=======>......................] - ETA: 1:19 - loss: 0.3659 - categorical_accuracy: 0.8845
16320/60000 [=======>......................] - ETA: 1:19 - loss: 0.3655 - categorical_accuracy: 0.8846
16352/60000 [=======>......................] - ETA: 1:19 - loss: 0.3649 - categorical_accuracy: 0.8847
16384/60000 [=======>......................] - ETA: 1:19 - loss: 0.3645 - categorical_accuracy: 0.8849
16416/60000 [=======>......................] - ETA: 1:19 - loss: 0.3643 - categorical_accuracy: 0.8850
16448/60000 [=======>......................] - ETA: 1:19 - loss: 0.3640 - categorical_accuracy: 0.8851
16480/60000 [=======>......................] - ETA: 1:19 - loss: 0.3637 - categorical_accuracy: 0.8851
16512/60000 [=======>......................] - ETA: 1:19 - loss: 0.3631 - categorical_accuracy: 0.8854
16544/60000 [=======>......................] - ETA: 1:19 - loss: 0.3627 - categorical_accuracy: 0.8855
16576/60000 [=======>......................] - ETA: 1:19 - loss: 0.3626 - categorical_accuracy: 0.8856
16608/60000 [=======>......................] - ETA: 1:19 - loss: 0.3624 - categorical_accuracy: 0.8856
16640/60000 [=======>......................] - ETA: 1:19 - loss: 0.3619 - categorical_accuracy: 0.8856
16672/60000 [=======>......................] - ETA: 1:19 - loss: 0.3613 - categorical_accuracy: 0.8858
16704/60000 [=======>......................] - ETA: 1:19 - loss: 0.3609 - categorical_accuracy: 0.8859
16736/60000 [=======>......................] - ETA: 1:19 - loss: 0.3604 - categorical_accuracy: 0.8861
16768/60000 [=======>......................] - ETA: 1:19 - loss: 0.3601 - categorical_accuracy: 0.8862
16800/60000 [=======>......................] - ETA: 1:18 - loss: 0.3596 - categorical_accuracy: 0.8864
16832/60000 [=======>......................] - ETA: 1:18 - loss: 0.3595 - categorical_accuracy: 0.8864
16864/60000 [=======>......................] - ETA: 1:18 - loss: 0.3595 - categorical_accuracy: 0.8865
16896/60000 [=======>......................] - ETA: 1:18 - loss: 0.3591 - categorical_accuracy: 0.8867
16928/60000 [=======>......................] - ETA: 1:18 - loss: 0.3588 - categorical_accuracy: 0.8867
16960/60000 [=======>......................] - ETA: 1:18 - loss: 0.3585 - categorical_accuracy: 0.8868
16992/60000 [=======>......................] - ETA: 1:18 - loss: 0.3586 - categorical_accuracy: 0.8868
17024/60000 [=======>......................] - ETA: 1:18 - loss: 0.3585 - categorical_accuracy: 0.8869
17056/60000 [=======>......................] - ETA: 1:18 - loss: 0.3581 - categorical_accuracy: 0.8871
17088/60000 [=======>......................] - ETA: 1:18 - loss: 0.3578 - categorical_accuracy: 0.8872
17120/60000 [=======>......................] - ETA: 1:18 - loss: 0.3578 - categorical_accuracy: 0.8873
17152/60000 [=======>......................] - ETA: 1:18 - loss: 0.3577 - categorical_accuracy: 0.8874
17184/60000 [=======>......................] - ETA: 1:18 - loss: 0.3572 - categorical_accuracy: 0.8875
17216/60000 [=======>......................] - ETA: 1:18 - loss: 0.3568 - categorical_accuracy: 0.8876
17248/60000 [=======>......................] - ETA: 1:18 - loss: 0.3566 - categorical_accuracy: 0.8876
17280/60000 [=======>......................] - ETA: 1:18 - loss: 0.3565 - categorical_accuracy: 0.8876
17312/60000 [=======>......................] - ETA: 1:17 - loss: 0.3561 - categorical_accuracy: 0.8877
17344/60000 [=======>......................] - ETA: 1:17 - loss: 0.3557 - categorical_accuracy: 0.8878
17376/60000 [=======>......................] - ETA: 1:17 - loss: 0.3558 - categorical_accuracy: 0.8878
17408/60000 [=======>......................] - ETA: 1:17 - loss: 0.3553 - categorical_accuracy: 0.8880
17440/60000 [=======>......................] - ETA: 1:17 - loss: 0.3548 - categorical_accuracy: 0.8881
17472/60000 [=======>......................] - ETA: 1:17 - loss: 0.3542 - categorical_accuracy: 0.8883
17504/60000 [=======>......................] - ETA: 1:17 - loss: 0.3539 - categorical_accuracy: 0.8885
17536/60000 [=======>......................] - ETA: 1:17 - loss: 0.3539 - categorical_accuracy: 0.8886
17568/60000 [=======>......................] - ETA: 1:17 - loss: 0.3534 - categorical_accuracy: 0.8888
17600/60000 [=======>......................] - ETA: 1:17 - loss: 0.3529 - categorical_accuracy: 0.8890
17632/60000 [=======>......................] - ETA: 1:17 - loss: 0.3527 - categorical_accuracy: 0.8890
17664/60000 [=======>......................] - ETA: 1:17 - loss: 0.3524 - categorical_accuracy: 0.8892
17696/60000 [=======>......................] - ETA: 1:17 - loss: 0.3524 - categorical_accuracy: 0.8892
17728/60000 [=======>......................] - ETA: 1:17 - loss: 0.3518 - categorical_accuracy: 0.8894
17760/60000 [=======>......................] - ETA: 1:17 - loss: 0.3516 - categorical_accuracy: 0.8895
17792/60000 [=======>......................] - ETA: 1:17 - loss: 0.3514 - categorical_accuracy: 0.8896
17824/60000 [=======>......................] - ETA: 1:17 - loss: 0.3512 - categorical_accuracy: 0.8897
17856/60000 [=======>......................] - ETA: 1:17 - loss: 0.3513 - categorical_accuracy: 0.8897
17888/60000 [=======>......................] - ETA: 1:16 - loss: 0.3514 - categorical_accuracy: 0.8898
17920/60000 [=======>......................] - ETA: 1:16 - loss: 0.3511 - categorical_accuracy: 0.8897
17952/60000 [=======>......................] - ETA: 1:16 - loss: 0.3516 - categorical_accuracy: 0.8897
17984/60000 [=======>......................] - ETA: 1:16 - loss: 0.3512 - categorical_accuracy: 0.8898
18016/60000 [========>.....................] - ETA: 1:16 - loss: 0.3507 - categorical_accuracy: 0.8900
18048/60000 [========>.....................] - ETA: 1:16 - loss: 0.3503 - categorical_accuracy: 0.8901
18080/60000 [========>.....................] - ETA: 1:16 - loss: 0.3502 - categorical_accuracy: 0.8902
18112/60000 [========>.....................] - ETA: 1:16 - loss: 0.3497 - categorical_accuracy: 0.8904
18144/60000 [========>.....................] - ETA: 1:16 - loss: 0.3494 - categorical_accuracy: 0.8904
18176/60000 [========>.....................] - ETA: 1:16 - loss: 0.3490 - categorical_accuracy: 0.8906
18208/60000 [========>.....................] - ETA: 1:16 - loss: 0.3485 - categorical_accuracy: 0.8907
18240/60000 [========>.....................] - ETA: 1:16 - loss: 0.3481 - categorical_accuracy: 0.8908
18272/60000 [========>.....................] - ETA: 1:16 - loss: 0.3477 - categorical_accuracy: 0.8910
18304/60000 [========>.....................] - ETA: 1:16 - loss: 0.3476 - categorical_accuracy: 0.8911
18336/60000 [========>.....................] - ETA: 1:16 - loss: 0.3472 - categorical_accuracy: 0.8911
18368/60000 [========>.....................] - ETA: 1:16 - loss: 0.3473 - categorical_accuracy: 0.8912
18400/60000 [========>.....................] - ETA: 1:16 - loss: 0.3472 - categorical_accuracy: 0.8913
18432/60000 [========>.....................] - ETA: 1:15 - loss: 0.3470 - categorical_accuracy: 0.8913
18464/60000 [========>.....................] - ETA: 1:15 - loss: 0.3466 - categorical_accuracy: 0.8914
18496/60000 [========>.....................] - ETA: 1:15 - loss: 0.3466 - categorical_accuracy: 0.8915
18528/60000 [========>.....................] - ETA: 1:15 - loss: 0.3464 - categorical_accuracy: 0.8916
18560/60000 [========>.....................] - ETA: 1:15 - loss: 0.3461 - categorical_accuracy: 0.8916
18592/60000 [========>.....................] - ETA: 1:15 - loss: 0.3458 - categorical_accuracy: 0.8917
18624/60000 [========>.....................] - ETA: 1:15 - loss: 0.3456 - categorical_accuracy: 0.8918
18656/60000 [========>.....................] - ETA: 1:15 - loss: 0.3452 - categorical_accuracy: 0.8919
18688/60000 [========>.....................] - ETA: 1:15 - loss: 0.3447 - categorical_accuracy: 0.8921
18720/60000 [========>.....................] - ETA: 1:15 - loss: 0.3445 - categorical_accuracy: 0.8922
18752/60000 [========>.....................] - ETA: 1:15 - loss: 0.3440 - categorical_accuracy: 0.8924
18784/60000 [========>.....................] - ETA: 1:15 - loss: 0.3440 - categorical_accuracy: 0.8923
18816/60000 [========>.....................] - ETA: 1:15 - loss: 0.3445 - categorical_accuracy: 0.8924
18848/60000 [========>.....................] - ETA: 1:15 - loss: 0.3440 - categorical_accuracy: 0.8925
18880/60000 [========>.....................] - ETA: 1:15 - loss: 0.3436 - categorical_accuracy: 0.8926
18912/60000 [========>.....................] - ETA: 1:15 - loss: 0.3432 - categorical_accuracy: 0.8927
18944/60000 [========>.....................] - ETA: 1:14 - loss: 0.3429 - categorical_accuracy: 0.8927
18976/60000 [========>.....................] - ETA: 1:14 - loss: 0.3424 - categorical_accuracy: 0.8929
19008/60000 [========>.....................] - ETA: 1:14 - loss: 0.3420 - categorical_accuracy: 0.8930
19040/60000 [========>.....................] - ETA: 1:14 - loss: 0.3417 - categorical_accuracy: 0.8931
19072/60000 [========>.....................] - ETA: 1:14 - loss: 0.3417 - categorical_accuracy: 0.8931
19104/60000 [========>.....................] - ETA: 1:14 - loss: 0.3416 - categorical_accuracy: 0.8931
19136/60000 [========>.....................] - ETA: 1:14 - loss: 0.3411 - categorical_accuracy: 0.8933
19168/60000 [========>.....................] - ETA: 1:14 - loss: 0.3406 - categorical_accuracy: 0.8935
19200/60000 [========>.....................] - ETA: 1:14 - loss: 0.3401 - categorical_accuracy: 0.8936
19232/60000 [========>.....................] - ETA: 1:14 - loss: 0.3397 - categorical_accuracy: 0.8937
19264/60000 [========>.....................] - ETA: 1:14 - loss: 0.3392 - categorical_accuracy: 0.8939
19296/60000 [========>.....................] - ETA: 1:14 - loss: 0.3388 - categorical_accuracy: 0.8940
19328/60000 [========>.....................] - ETA: 1:14 - loss: 0.3384 - categorical_accuracy: 0.8941
19360/60000 [========>.....................] - ETA: 1:14 - loss: 0.3379 - categorical_accuracy: 0.8943
19392/60000 [========>.....................] - ETA: 1:14 - loss: 0.3375 - categorical_accuracy: 0.8944
19424/60000 [========>.....................] - ETA: 1:14 - loss: 0.3370 - categorical_accuracy: 0.8946
19456/60000 [========>.....................] - ETA: 1:14 - loss: 0.3371 - categorical_accuracy: 0.8947
19488/60000 [========>.....................] - ETA: 1:13 - loss: 0.3368 - categorical_accuracy: 0.8948
19520/60000 [========>.....................] - ETA: 1:13 - loss: 0.3368 - categorical_accuracy: 0.8947
19552/60000 [========>.....................] - ETA: 1:13 - loss: 0.3365 - categorical_accuracy: 0.8948
19584/60000 [========>.....................] - ETA: 1:13 - loss: 0.3362 - categorical_accuracy: 0.8949
19616/60000 [========>.....................] - ETA: 1:13 - loss: 0.3358 - categorical_accuracy: 0.8951
19648/60000 [========>.....................] - ETA: 1:13 - loss: 0.3353 - categorical_accuracy: 0.8952
19680/60000 [========>.....................] - ETA: 1:13 - loss: 0.3350 - categorical_accuracy: 0.8953
19712/60000 [========>.....................] - ETA: 1:13 - loss: 0.3346 - categorical_accuracy: 0.8954
19744/60000 [========>.....................] - ETA: 1:13 - loss: 0.3345 - categorical_accuracy: 0.8955
19776/60000 [========>.....................] - ETA: 1:13 - loss: 0.3340 - categorical_accuracy: 0.8957
19808/60000 [========>.....................] - ETA: 1:13 - loss: 0.3335 - categorical_accuracy: 0.8958
19840/60000 [========>.....................] - ETA: 1:13 - loss: 0.3331 - categorical_accuracy: 0.8959
19872/60000 [========>.....................] - ETA: 1:13 - loss: 0.3329 - categorical_accuracy: 0.8960
19904/60000 [========>.....................] - ETA: 1:13 - loss: 0.3324 - categorical_accuracy: 0.8962
19936/60000 [========>.....................] - ETA: 1:13 - loss: 0.3320 - categorical_accuracy: 0.8963
19968/60000 [========>.....................] - ETA: 1:13 - loss: 0.3316 - categorical_accuracy: 0.8965
20000/60000 [=========>....................] - ETA: 1:13 - loss: 0.3314 - categorical_accuracy: 0.8965
20032/60000 [=========>....................] - ETA: 1:13 - loss: 0.3310 - categorical_accuracy: 0.8966
20064/60000 [=========>....................] - ETA: 1:12 - loss: 0.3306 - categorical_accuracy: 0.8967
20096/60000 [=========>....................] - ETA: 1:12 - loss: 0.3302 - categorical_accuracy: 0.8969
20128/60000 [=========>....................] - ETA: 1:12 - loss: 0.3298 - categorical_accuracy: 0.8970
20160/60000 [=========>....................] - ETA: 1:12 - loss: 0.3294 - categorical_accuracy: 0.8971
20192/60000 [=========>....................] - ETA: 1:12 - loss: 0.3292 - categorical_accuracy: 0.8972
20224/60000 [=========>....................] - ETA: 1:12 - loss: 0.3289 - categorical_accuracy: 0.8973
20256/60000 [=========>....................] - ETA: 1:12 - loss: 0.3285 - categorical_accuracy: 0.8974
20288/60000 [=========>....................] - ETA: 1:12 - loss: 0.3286 - categorical_accuracy: 0.8975
20320/60000 [=========>....................] - ETA: 1:12 - loss: 0.3282 - categorical_accuracy: 0.8976
20352/60000 [=========>....................] - ETA: 1:12 - loss: 0.3278 - categorical_accuracy: 0.8977
20384/60000 [=========>....................] - ETA: 1:12 - loss: 0.3274 - categorical_accuracy: 0.8979
20416/60000 [=========>....................] - ETA: 1:12 - loss: 0.3270 - categorical_accuracy: 0.8980
20448/60000 [=========>....................] - ETA: 1:12 - loss: 0.3267 - categorical_accuracy: 0.8981
20480/60000 [=========>....................] - ETA: 1:12 - loss: 0.3263 - categorical_accuracy: 0.8982
20512/60000 [=========>....................] - ETA: 1:12 - loss: 0.3259 - categorical_accuracy: 0.8984
20544/60000 [=========>....................] - ETA: 1:12 - loss: 0.3254 - categorical_accuracy: 0.8985
20576/60000 [=========>....................] - ETA: 1:12 - loss: 0.3251 - categorical_accuracy: 0.8986
20608/60000 [=========>....................] - ETA: 1:12 - loss: 0.3247 - categorical_accuracy: 0.8987
20640/60000 [=========>....................] - ETA: 1:11 - loss: 0.3243 - categorical_accuracy: 0.8988
20672/60000 [=========>....................] - ETA: 1:11 - loss: 0.3240 - categorical_accuracy: 0.8989
20704/60000 [=========>....................] - ETA: 1:11 - loss: 0.3244 - categorical_accuracy: 0.8989
20736/60000 [=========>....................] - ETA: 1:11 - loss: 0.3241 - categorical_accuracy: 0.8989
20768/60000 [=========>....................] - ETA: 1:11 - loss: 0.3237 - categorical_accuracy: 0.8990
20800/60000 [=========>....................] - ETA: 1:11 - loss: 0.3240 - categorical_accuracy: 0.8989
20832/60000 [=========>....................] - ETA: 1:11 - loss: 0.3236 - categorical_accuracy: 0.8991
20864/60000 [=========>....................] - ETA: 1:11 - loss: 0.3233 - categorical_accuracy: 0.8992
20896/60000 [=========>....................] - ETA: 1:11 - loss: 0.3229 - categorical_accuracy: 0.8993
20928/60000 [=========>....................] - ETA: 1:11 - loss: 0.3226 - categorical_accuracy: 0.8993
20960/60000 [=========>....................] - ETA: 1:11 - loss: 0.3224 - categorical_accuracy: 0.8994
20992/60000 [=========>....................] - ETA: 1:11 - loss: 0.3220 - categorical_accuracy: 0.8995
21024/60000 [=========>....................] - ETA: 1:11 - loss: 0.3217 - categorical_accuracy: 0.8997
21056/60000 [=========>....................] - ETA: 1:11 - loss: 0.3212 - categorical_accuracy: 0.8998
21088/60000 [=========>....................] - ETA: 1:11 - loss: 0.3210 - categorical_accuracy: 0.8999
21120/60000 [=========>....................] - ETA: 1:10 - loss: 0.3205 - categorical_accuracy: 0.9000
21152/60000 [=========>....................] - ETA: 1:10 - loss: 0.3203 - categorical_accuracy: 0.9002
21184/60000 [=========>....................] - ETA: 1:10 - loss: 0.3201 - categorical_accuracy: 0.9003
21216/60000 [=========>....................] - ETA: 1:10 - loss: 0.3198 - categorical_accuracy: 0.9004
21248/60000 [=========>....................] - ETA: 1:10 - loss: 0.3193 - categorical_accuracy: 0.9005
21280/60000 [=========>....................] - ETA: 1:10 - loss: 0.3193 - categorical_accuracy: 0.9005
21312/60000 [=========>....................] - ETA: 1:10 - loss: 0.3195 - categorical_accuracy: 0.9004
21344/60000 [=========>....................] - ETA: 1:10 - loss: 0.3195 - categorical_accuracy: 0.9005
21376/60000 [=========>....................] - ETA: 1:10 - loss: 0.3191 - categorical_accuracy: 0.9006
21408/60000 [=========>....................] - ETA: 1:10 - loss: 0.3188 - categorical_accuracy: 0.9006
21440/60000 [=========>....................] - ETA: 1:10 - loss: 0.3184 - categorical_accuracy: 0.9007
21472/60000 [=========>....................] - ETA: 1:10 - loss: 0.3187 - categorical_accuracy: 0.9007
21504/60000 [=========>....................] - ETA: 1:10 - loss: 0.3187 - categorical_accuracy: 0.9008
21536/60000 [=========>....................] - ETA: 1:10 - loss: 0.3183 - categorical_accuracy: 0.9009
21568/60000 [=========>....................] - ETA: 1:10 - loss: 0.3181 - categorical_accuracy: 0.9009
21600/60000 [=========>....................] - ETA: 1:10 - loss: 0.3181 - categorical_accuracy: 0.9010
21632/60000 [=========>....................] - ETA: 1:10 - loss: 0.3177 - categorical_accuracy: 0.9011
21664/60000 [=========>....................] - ETA: 1:10 - loss: 0.3176 - categorical_accuracy: 0.9011
21696/60000 [=========>....................] - ETA: 1:09 - loss: 0.3174 - categorical_accuracy: 0.9011
21728/60000 [=========>....................] - ETA: 1:09 - loss: 0.3171 - categorical_accuracy: 0.9012
21760/60000 [=========>....................] - ETA: 1:09 - loss: 0.3169 - categorical_accuracy: 0.9013
21792/60000 [=========>....................] - ETA: 1:09 - loss: 0.3166 - categorical_accuracy: 0.9014
21824/60000 [=========>....................] - ETA: 1:09 - loss: 0.3163 - categorical_accuracy: 0.9015
21856/60000 [=========>....................] - ETA: 1:09 - loss: 0.3159 - categorical_accuracy: 0.9017
21888/60000 [=========>....................] - ETA: 1:09 - loss: 0.3156 - categorical_accuracy: 0.9017
21920/60000 [=========>....................] - ETA: 1:09 - loss: 0.3154 - categorical_accuracy: 0.9017
21952/60000 [=========>....................] - ETA: 1:09 - loss: 0.3153 - categorical_accuracy: 0.9018
21984/60000 [=========>....................] - ETA: 1:09 - loss: 0.3152 - categorical_accuracy: 0.9018
22016/60000 [==========>...................] - ETA: 1:09 - loss: 0.3149 - categorical_accuracy: 0.9019
22048/60000 [==========>...................] - ETA: 1:09 - loss: 0.3147 - categorical_accuracy: 0.9020
22080/60000 [==========>...................] - ETA: 1:09 - loss: 0.3143 - categorical_accuracy: 0.9021
22112/60000 [==========>...................] - ETA: 1:09 - loss: 0.3142 - categorical_accuracy: 0.9022
22144/60000 [==========>...................] - ETA: 1:09 - loss: 0.3142 - categorical_accuracy: 0.9022
22176/60000 [==========>...................] - ETA: 1:09 - loss: 0.3138 - categorical_accuracy: 0.9023
22208/60000 [==========>...................] - ETA: 1:09 - loss: 0.3135 - categorical_accuracy: 0.9024
22240/60000 [==========>...................] - ETA: 1:08 - loss: 0.3131 - categorical_accuracy: 0.9025
22272/60000 [==========>...................] - ETA: 1:08 - loss: 0.3128 - categorical_accuracy: 0.9026
22304/60000 [==========>...................] - ETA: 1:08 - loss: 0.3125 - categorical_accuracy: 0.9026
22336/60000 [==========>...................] - ETA: 1:08 - loss: 0.3122 - categorical_accuracy: 0.9028
22368/60000 [==========>...................] - ETA: 1:08 - loss: 0.3119 - categorical_accuracy: 0.9029
22400/60000 [==========>...................] - ETA: 1:08 - loss: 0.3115 - categorical_accuracy: 0.9030
22432/60000 [==========>...................] - ETA: 1:08 - loss: 0.3112 - categorical_accuracy: 0.9031
22464/60000 [==========>...................] - ETA: 1:08 - loss: 0.3109 - categorical_accuracy: 0.9031
22496/60000 [==========>...................] - ETA: 1:08 - loss: 0.3107 - categorical_accuracy: 0.9032
22528/60000 [==========>...................] - ETA: 1:08 - loss: 0.3104 - categorical_accuracy: 0.9033
22560/60000 [==========>...................] - ETA: 1:08 - loss: 0.3100 - categorical_accuracy: 0.9035
22592/60000 [==========>...................] - ETA: 1:08 - loss: 0.3097 - categorical_accuracy: 0.9035
22624/60000 [==========>...................] - ETA: 1:08 - loss: 0.3094 - categorical_accuracy: 0.9036
22656/60000 [==========>...................] - ETA: 1:08 - loss: 0.3091 - categorical_accuracy: 0.9037
22688/60000 [==========>...................] - ETA: 1:08 - loss: 0.3088 - categorical_accuracy: 0.9038
22720/60000 [==========>...................] - ETA: 1:08 - loss: 0.3086 - categorical_accuracy: 0.9039
22752/60000 [==========>...................] - ETA: 1:07 - loss: 0.3082 - categorical_accuracy: 0.9040
22784/60000 [==========>...................] - ETA: 1:07 - loss: 0.3079 - categorical_accuracy: 0.9041
22816/60000 [==========>...................] - ETA: 1:07 - loss: 0.3078 - categorical_accuracy: 0.9041
22848/60000 [==========>...................] - ETA: 1:07 - loss: 0.3075 - categorical_accuracy: 0.9042
22880/60000 [==========>...................] - ETA: 1:07 - loss: 0.3072 - categorical_accuracy: 0.9043
22912/60000 [==========>...................] - ETA: 1:07 - loss: 0.3071 - categorical_accuracy: 0.9044
22944/60000 [==========>...................] - ETA: 1:07 - loss: 0.3067 - categorical_accuracy: 0.9045
22976/60000 [==========>...................] - ETA: 1:07 - loss: 0.3064 - categorical_accuracy: 0.9046
23008/60000 [==========>...................] - ETA: 1:07 - loss: 0.3061 - categorical_accuracy: 0.9046
23040/60000 [==========>...................] - ETA: 1:07 - loss: 0.3058 - categorical_accuracy: 0.9047
23072/60000 [==========>...................] - ETA: 1:07 - loss: 0.3059 - categorical_accuracy: 0.9047
23104/60000 [==========>...................] - ETA: 1:07 - loss: 0.3056 - categorical_accuracy: 0.9048
23136/60000 [==========>...................] - ETA: 1:07 - loss: 0.3052 - categorical_accuracy: 0.9049
23168/60000 [==========>...................] - ETA: 1:07 - loss: 0.3049 - categorical_accuracy: 0.9050
23200/60000 [==========>...................] - ETA: 1:07 - loss: 0.3048 - categorical_accuracy: 0.9050
23232/60000 [==========>...................] - ETA: 1:07 - loss: 0.3046 - categorical_accuracy: 0.9051
23264/60000 [==========>...................] - ETA: 1:07 - loss: 0.3045 - categorical_accuracy: 0.9051
23296/60000 [==========>...................] - ETA: 1:07 - loss: 0.3044 - categorical_accuracy: 0.9051
23328/60000 [==========>...................] - ETA: 1:06 - loss: 0.3040 - categorical_accuracy: 0.9052
23360/60000 [==========>...................] - ETA: 1:06 - loss: 0.3041 - categorical_accuracy: 0.9053
23392/60000 [==========>...................] - ETA: 1:06 - loss: 0.3037 - categorical_accuracy: 0.9054
23424/60000 [==========>...................] - ETA: 1:06 - loss: 0.3036 - categorical_accuracy: 0.9054
23456/60000 [==========>...................] - ETA: 1:06 - loss: 0.3033 - categorical_accuracy: 0.9054
23488/60000 [==========>...................] - ETA: 1:06 - loss: 0.3030 - categorical_accuracy: 0.9056
23520/60000 [==========>...................] - ETA: 1:06 - loss: 0.3027 - categorical_accuracy: 0.9056
23552/60000 [==========>...................] - ETA: 1:06 - loss: 0.3024 - categorical_accuracy: 0.9057
23584/60000 [==========>...................] - ETA: 1:06 - loss: 0.3020 - categorical_accuracy: 0.9058
23616/60000 [==========>...................] - ETA: 1:06 - loss: 0.3018 - categorical_accuracy: 0.9059
23648/60000 [==========>...................] - ETA: 1:06 - loss: 0.3020 - categorical_accuracy: 0.9058
23680/60000 [==========>...................] - ETA: 1:06 - loss: 0.3017 - categorical_accuracy: 0.9060
23712/60000 [==========>...................] - ETA: 1:06 - loss: 0.3014 - categorical_accuracy: 0.9060
23744/60000 [==========>...................] - ETA: 1:06 - loss: 0.3013 - categorical_accuracy: 0.9061
23776/60000 [==========>...................] - ETA: 1:06 - loss: 0.3010 - categorical_accuracy: 0.9062
23808/60000 [==========>...................] - ETA: 1:06 - loss: 0.3008 - categorical_accuracy: 0.9062
23840/60000 [==========>...................] - ETA: 1:05 - loss: 0.3006 - categorical_accuracy: 0.9062
23872/60000 [==========>...................] - ETA: 1:05 - loss: 0.3004 - categorical_accuracy: 0.9063
23904/60000 [==========>...................] - ETA: 1:05 - loss: 0.3001 - categorical_accuracy: 0.9065
23936/60000 [==========>...................] - ETA: 1:05 - loss: 0.2998 - categorical_accuracy: 0.9065
23968/60000 [==========>...................] - ETA: 1:05 - loss: 0.2996 - categorical_accuracy: 0.9066
24000/60000 [===========>..................] - ETA: 1:05 - loss: 0.2993 - categorical_accuracy: 0.9067
24032/60000 [===========>..................] - ETA: 1:05 - loss: 0.2990 - categorical_accuracy: 0.9068
24064/60000 [===========>..................] - ETA: 1:05 - loss: 0.2987 - categorical_accuracy: 0.9069
24096/60000 [===========>..................] - ETA: 1:05 - loss: 0.2983 - categorical_accuracy: 0.9070
24128/60000 [===========>..................] - ETA: 1:05 - loss: 0.2981 - categorical_accuracy: 0.9070
24160/60000 [===========>..................] - ETA: 1:05 - loss: 0.2982 - categorical_accuracy: 0.9070
24192/60000 [===========>..................] - ETA: 1:05 - loss: 0.2982 - categorical_accuracy: 0.9071
24224/60000 [===========>..................] - ETA: 1:05 - loss: 0.2980 - categorical_accuracy: 0.9072
24256/60000 [===========>..................] - ETA: 1:05 - loss: 0.2977 - categorical_accuracy: 0.9073
24288/60000 [===========>..................] - ETA: 1:05 - loss: 0.2975 - categorical_accuracy: 0.9073
24320/60000 [===========>..................] - ETA: 1:05 - loss: 0.2972 - categorical_accuracy: 0.9074
24352/60000 [===========>..................] - ETA: 1:05 - loss: 0.2969 - categorical_accuracy: 0.9075
24384/60000 [===========>..................] - ETA: 1:04 - loss: 0.2966 - categorical_accuracy: 0.9076
24416/60000 [===========>..................] - ETA: 1:04 - loss: 0.2962 - categorical_accuracy: 0.9077
24448/60000 [===========>..................] - ETA: 1:04 - loss: 0.2959 - categorical_accuracy: 0.9078
24480/60000 [===========>..................] - ETA: 1:04 - loss: 0.2957 - categorical_accuracy: 0.9078
24512/60000 [===========>..................] - ETA: 1:04 - loss: 0.2955 - categorical_accuracy: 0.9079
24544/60000 [===========>..................] - ETA: 1:04 - loss: 0.2956 - categorical_accuracy: 0.9080
24576/60000 [===========>..................] - ETA: 1:04 - loss: 0.2955 - categorical_accuracy: 0.9080
24608/60000 [===========>..................] - ETA: 1:04 - loss: 0.2953 - categorical_accuracy: 0.9081
24640/60000 [===========>..................] - ETA: 1:04 - loss: 0.2950 - categorical_accuracy: 0.9081
24672/60000 [===========>..................] - ETA: 1:04 - loss: 0.2947 - categorical_accuracy: 0.9082
24704/60000 [===========>..................] - ETA: 1:04 - loss: 0.2945 - categorical_accuracy: 0.9082
24736/60000 [===========>..................] - ETA: 1:04 - loss: 0.2943 - categorical_accuracy: 0.9084
24768/60000 [===========>..................] - ETA: 1:04 - loss: 0.2944 - categorical_accuracy: 0.9083
24800/60000 [===========>..................] - ETA: 1:04 - loss: 0.2941 - categorical_accuracy: 0.9084
24832/60000 [===========>..................] - ETA: 1:04 - loss: 0.2939 - categorical_accuracy: 0.9085
24864/60000 [===========>..................] - ETA: 1:04 - loss: 0.2936 - categorical_accuracy: 0.9086
24896/60000 [===========>..................] - ETA: 1:03 - loss: 0.2932 - categorical_accuracy: 0.9087
24928/60000 [===========>..................] - ETA: 1:03 - loss: 0.2929 - categorical_accuracy: 0.9088
24960/60000 [===========>..................] - ETA: 1:03 - loss: 0.2926 - categorical_accuracy: 0.9089
24992/60000 [===========>..................] - ETA: 1:03 - loss: 0.2924 - categorical_accuracy: 0.9090
25024/60000 [===========>..................] - ETA: 1:03 - loss: 0.2921 - categorical_accuracy: 0.9091
25056/60000 [===========>..................] - ETA: 1:03 - loss: 0.2919 - categorical_accuracy: 0.9091
25088/60000 [===========>..................] - ETA: 1:03 - loss: 0.2922 - categorical_accuracy: 0.9092
25120/60000 [===========>..................] - ETA: 1:03 - loss: 0.2922 - categorical_accuracy: 0.9092
25152/60000 [===========>..................] - ETA: 1:03 - loss: 0.2923 - categorical_accuracy: 0.9091
25184/60000 [===========>..................] - ETA: 1:03 - loss: 0.2920 - categorical_accuracy: 0.9091
25216/60000 [===========>..................] - ETA: 1:03 - loss: 0.2917 - categorical_accuracy: 0.9093
25248/60000 [===========>..................] - ETA: 1:03 - loss: 0.2920 - categorical_accuracy: 0.9093
25280/60000 [===========>..................] - ETA: 1:03 - loss: 0.2919 - categorical_accuracy: 0.9093
25312/60000 [===========>..................] - ETA: 1:03 - loss: 0.2916 - categorical_accuracy: 0.9094
25344/60000 [===========>..................] - ETA: 1:03 - loss: 0.2915 - categorical_accuracy: 0.9094
25376/60000 [===========>..................] - ETA: 1:03 - loss: 0.2914 - categorical_accuracy: 0.9094
25408/60000 [===========>..................] - ETA: 1:03 - loss: 0.2916 - categorical_accuracy: 0.9095
25440/60000 [===========>..................] - ETA: 1:03 - loss: 0.2913 - categorical_accuracy: 0.9095
25472/60000 [===========>..................] - ETA: 1:02 - loss: 0.2910 - categorical_accuracy: 0.9096
25504/60000 [===========>..................] - ETA: 1:02 - loss: 0.2908 - categorical_accuracy: 0.9097
25536/60000 [===========>..................] - ETA: 1:02 - loss: 0.2907 - categorical_accuracy: 0.9097
25568/60000 [===========>..................] - ETA: 1:02 - loss: 0.2904 - categorical_accuracy: 0.9098
25600/60000 [===========>..................] - ETA: 1:02 - loss: 0.2903 - categorical_accuracy: 0.9098
25632/60000 [===========>..................] - ETA: 1:02 - loss: 0.2900 - categorical_accuracy: 0.9098
25664/60000 [===========>..................] - ETA: 1:02 - loss: 0.2897 - categorical_accuracy: 0.9100
25696/60000 [===========>..................] - ETA: 1:02 - loss: 0.2894 - categorical_accuracy: 0.9101
25728/60000 [===========>..................] - ETA: 1:02 - loss: 0.2892 - categorical_accuracy: 0.9102
25760/60000 [===========>..................] - ETA: 1:02 - loss: 0.2890 - categorical_accuracy: 0.9102
25792/60000 [===========>..................] - ETA: 1:02 - loss: 0.2890 - categorical_accuracy: 0.9103
25824/60000 [===========>..................] - ETA: 1:02 - loss: 0.2887 - categorical_accuracy: 0.9104
25856/60000 [===========>..................] - ETA: 1:02 - loss: 0.2886 - categorical_accuracy: 0.9105
25888/60000 [===========>..................] - ETA: 1:02 - loss: 0.2883 - categorical_accuracy: 0.9106
25920/60000 [===========>..................] - ETA: 1:02 - loss: 0.2880 - categorical_accuracy: 0.9107
25952/60000 [===========>..................] - ETA: 1:02 - loss: 0.2877 - categorical_accuracy: 0.9107
25984/60000 [===========>..................] - ETA: 1:02 - loss: 0.2874 - categorical_accuracy: 0.9108
26016/60000 [============>.................] - ETA: 1:01 - loss: 0.2873 - categorical_accuracy: 0.9109
26048/60000 [============>.................] - ETA: 1:01 - loss: 0.2871 - categorical_accuracy: 0.9109
26080/60000 [============>.................] - ETA: 1:01 - loss: 0.2870 - categorical_accuracy: 0.9110
26112/60000 [============>.................] - ETA: 1:01 - loss: 0.2867 - categorical_accuracy: 0.9111
26144/60000 [============>.................] - ETA: 1:01 - loss: 0.2865 - categorical_accuracy: 0.9111
26176/60000 [============>.................] - ETA: 1:01 - loss: 0.2862 - categorical_accuracy: 0.9113
26208/60000 [============>.................] - ETA: 1:01 - loss: 0.2859 - categorical_accuracy: 0.9114
26240/60000 [============>.................] - ETA: 1:01 - loss: 0.2857 - categorical_accuracy: 0.9114
26272/60000 [============>.................] - ETA: 1:01 - loss: 0.2854 - categorical_accuracy: 0.9115
26304/60000 [============>.................] - ETA: 1:01 - loss: 0.2854 - categorical_accuracy: 0.9115
26336/60000 [============>.................] - ETA: 1:01 - loss: 0.2852 - categorical_accuracy: 0.9116
26368/60000 [============>.................] - ETA: 1:01 - loss: 0.2849 - categorical_accuracy: 0.9117
26400/60000 [============>.................] - ETA: 1:01 - loss: 0.2848 - categorical_accuracy: 0.9117
26432/60000 [============>.................] - ETA: 1:01 - loss: 0.2845 - categorical_accuracy: 0.9118
26464/60000 [============>.................] - ETA: 1:01 - loss: 0.2843 - categorical_accuracy: 0.9118
26496/60000 [============>.................] - ETA: 1:01 - loss: 0.2843 - categorical_accuracy: 0.9118
26528/60000 [============>.................] - ETA: 1:01 - loss: 0.2842 - categorical_accuracy: 0.9118
26560/60000 [============>.................] - ETA: 1:00 - loss: 0.2840 - categorical_accuracy: 0.9118
26592/60000 [============>.................] - ETA: 1:00 - loss: 0.2838 - categorical_accuracy: 0.9119
26624/60000 [============>.................] - ETA: 1:00 - loss: 0.2835 - categorical_accuracy: 0.9120
26656/60000 [============>.................] - ETA: 1:00 - loss: 0.2832 - categorical_accuracy: 0.9121
26688/60000 [============>.................] - ETA: 1:00 - loss: 0.2832 - categorical_accuracy: 0.9121
26720/60000 [============>.................] - ETA: 1:00 - loss: 0.2835 - categorical_accuracy: 0.9121
26752/60000 [============>.................] - ETA: 1:00 - loss: 0.2834 - categorical_accuracy: 0.9121
26784/60000 [============>.................] - ETA: 1:00 - loss: 0.2833 - categorical_accuracy: 0.9121
26816/60000 [============>.................] - ETA: 1:00 - loss: 0.2834 - categorical_accuracy: 0.9121
26848/60000 [============>.................] - ETA: 1:00 - loss: 0.2834 - categorical_accuracy: 0.9121
26880/60000 [============>.................] - ETA: 1:00 - loss: 0.2831 - categorical_accuracy: 0.9122
26912/60000 [============>.................] - ETA: 1:00 - loss: 0.2830 - categorical_accuracy: 0.9122
26944/60000 [============>.................] - ETA: 1:00 - loss: 0.2828 - categorical_accuracy: 0.9123
26976/60000 [============>.................] - ETA: 1:00 - loss: 0.2825 - categorical_accuracy: 0.9123
27008/60000 [============>.................] - ETA: 1:00 - loss: 0.2823 - categorical_accuracy: 0.9124
27040/60000 [============>.................] - ETA: 1:00 - loss: 0.2820 - categorical_accuracy: 0.9125
27072/60000 [============>.................] - ETA: 1:00 - loss: 0.2818 - categorical_accuracy: 0.9125
27104/60000 [============>.................] - ETA: 59s - loss: 0.2818 - categorical_accuracy: 0.9125 
27136/60000 [============>.................] - ETA: 59s - loss: 0.2819 - categorical_accuracy: 0.9126
27168/60000 [============>.................] - ETA: 59s - loss: 0.2817 - categorical_accuracy: 0.9125
27200/60000 [============>.................] - ETA: 59s - loss: 0.2814 - categorical_accuracy: 0.9126
27232/60000 [============>.................] - ETA: 59s - loss: 0.2813 - categorical_accuracy: 0.9127
27264/60000 [============>.................] - ETA: 59s - loss: 0.2811 - categorical_accuracy: 0.9128
27296/60000 [============>.................] - ETA: 59s - loss: 0.2808 - categorical_accuracy: 0.9129
27328/60000 [============>.................] - ETA: 59s - loss: 0.2806 - categorical_accuracy: 0.9130
27360/60000 [============>.................] - ETA: 59s - loss: 0.2804 - categorical_accuracy: 0.9130
27392/60000 [============>.................] - ETA: 59s - loss: 0.2801 - categorical_accuracy: 0.9131
27424/60000 [============>.................] - ETA: 59s - loss: 0.2798 - categorical_accuracy: 0.9133
27456/60000 [============>.................] - ETA: 59s - loss: 0.2795 - categorical_accuracy: 0.9134
27488/60000 [============>.................] - ETA: 59s - loss: 0.2792 - categorical_accuracy: 0.9134
27520/60000 [============>.................] - ETA: 59s - loss: 0.2789 - categorical_accuracy: 0.9135
27552/60000 [============>.................] - ETA: 59s - loss: 0.2788 - categorical_accuracy: 0.9136
27584/60000 [============>.................] - ETA: 59s - loss: 0.2788 - categorical_accuracy: 0.9136
27616/60000 [============>.................] - ETA: 59s - loss: 0.2788 - categorical_accuracy: 0.9136
27648/60000 [============>.................] - ETA: 58s - loss: 0.2786 - categorical_accuracy: 0.9137
27680/60000 [============>.................] - ETA: 58s - loss: 0.2782 - categorical_accuracy: 0.9138
27712/60000 [============>.................] - ETA: 58s - loss: 0.2780 - categorical_accuracy: 0.9138
27744/60000 [============>.................] - ETA: 58s - loss: 0.2781 - categorical_accuracy: 0.9138
27776/60000 [============>.................] - ETA: 58s - loss: 0.2778 - categorical_accuracy: 0.9139
27808/60000 [============>.................] - ETA: 58s - loss: 0.2775 - categorical_accuracy: 0.9140
27840/60000 [============>.................] - ETA: 58s - loss: 0.2774 - categorical_accuracy: 0.9140
27872/60000 [============>.................] - ETA: 58s - loss: 0.2773 - categorical_accuracy: 0.9140
27904/60000 [============>.................] - ETA: 58s - loss: 0.2772 - categorical_accuracy: 0.9140
27936/60000 [============>.................] - ETA: 58s - loss: 0.2770 - categorical_accuracy: 0.9141
27968/60000 [============>.................] - ETA: 58s - loss: 0.2770 - categorical_accuracy: 0.9140
28000/60000 [=============>................] - ETA: 58s - loss: 0.2768 - categorical_accuracy: 0.9141
28032/60000 [=============>................] - ETA: 58s - loss: 0.2765 - categorical_accuracy: 0.9142
28064/60000 [=============>................] - ETA: 58s - loss: 0.2764 - categorical_accuracy: 0.9143
28096/60000 [=============>................] - ETA: 58s - loss: 0.2762 - categorical_accuracy: 0.9143
28128/60000 [=============>................] - ETA: 58s - loss: 0.2762 - categorical_accuracy: 0.9143
28160/60000 [=============>................] - ETA: 57s - loss: 0.2760 - categorical_accuracy: 0.9144
28192/60000 [=============>................] - ETA: 57s - loss: 0.2758 - categorical_accuracy: 0.9144
28224/60000 [=============>................] - ETA: 57s - loss: 0.2755 - categorical_accuracy: 0.9145
28256/60000 [=============>................] - ETA: 57s - loss: 0.2752 - categorical_accuracy: 0.9146
28288/60000 [=============>................] - ETA: 57s - loss: 0.2750 - categorical_accuracy: 0.9147
28320/60000 [=============>................] - ETA: 57s - loss: 0.2747 - categorical_accuracy: 0.9148
28352/60000 [=============>................] - ETA: 57s - loss: 0.2747 - categorical_accuracy: 0.9148
28384/60000 [=============>................] - ETA: 57s - loss: 0.2745 - categorical_accuracy: 0.9149
28416/60000 [=============>................] - ETA: 57s - loss: 0.2743 - categorical_accuracy: 0.9149
28448/60000 [=============>................] - ETA: 57s - loss: 0.2746 - categorical_accuracy: 0.9149
28480/60000 [=============>................] - ETA: 57s - loss: 0.2744 - categorical_accuracy: 0.9150
28512/60000 [=============>................] - ETA: 57s - loss: 0.2741 - categorical_accuracy: 0.9150
28544/60000 [=============>................] - ETA: 57s - loss: 0.2739 - categorical_accuracy: 0.9151
28576/60000 [=============>................] - ETA: 57s - loss: 0.2737 - categorical_accuracy: 0.9152
28608/60000 [=============>................] - ETA: 57s - loss: 0.2735 - categorical_accuracy: 0.9152
28640/60000 [=============>................] - ETA: 57s - loss: 0.2733 - categorical_accuracy: 0.9153
28672/60000 [=============>................] - ETA: 57s - loss: 0.2731 - categorical_accuracy: 0.9153
28704/60000 [=============>................] - ETA: 57s - loss: 0.2730 - categorical_accuracy: 0.9154
28736/60000 [=============>................] - ETA: 56s - loss: 0.2727 - categorical_accuracy: 0.9154
28768/60000 [=============>................] - ETA: 56s - loss: 0.2725 - categorical_accuracy: 0.9155
28800/60000 [=============>................] - ETA: 56s - loss: 0.2726 - categorical_accuracy: 0.9155
28832/60000 [=============>................] - ETA: 56s - loss: 0.2727 - categorical_accuracy: 0.9155
28864/60000 [=============>................] - ETA: 56s - loss: 0.2726 - categorical_accuracy: 0.9155
28896/60000 [=============>................] - ETA: 56s - loss: 0.2727 - categorical_accuracy: 0.9155
28928/60000 [=============>................] - ETA: 56s - loss: 0.2724 - categorical_accuracy: 0.9156
28960/60000 [=============>................] - ETA: 56s - loss: 0.2724 - categorical_accuracy: 0.9156
28992/60000 [=============>................] - ETA: 56s - loss: 0.2722 - categorical_accuracy: 0.9157
29024/60000 [=============>................] - ETA: 56s - loss: 0.2720 - categorical_accuracy: 0.9158
29056/60000 [=============>................] - ETA: 56s - loss: 0.2717 - categorical_accuracy: 0.9159
29088/60000 [=============>................] - ETA: 56s - loss: 0.2715 - categorical_accuracy: 0.9159
29120/60000 [=============>................] - ETA: 56s - loss: 0.2715 - categorical_accuracy: 0.9159
29152/60000 [=============>................] - ETA: 56s - loss: 0.2713 - categorical_accuracy: 0.9160
29184/60000 [=============>................] - ETA: 56s - loss: 0.2711 - categorical_accuracy: 0.9160
29216/60000 [=============>................] - ETA: 56s - loss: 0.2710 - categorical_accuracy: 0.9161
29248/60000 [=============>................] - ETA: 56s - loss: 0.2707 - categorical_accuracy: 0.9162
29280/60000 [=============>................] - ETA: 55s - loss: 0.2705 - categorical_accuracy: 0.9163
29312/60000 [=============>................] - ETA: 55s - loss: 0.2702 - categorical_accuracy: 0.9164
29344/60000 [=============>................] - ETA: 55s - loss: 0.2700 - categorical_accuracy: 0.9165
29376/60000 [=============>................] - ETA: 55s - loss: 0.2697 - categorical_accuracy: 0.9166
29408/60000 [=============>................] - ETA: 55s - loss: 0.2695 - categorical_accuracy: 0.9166
29440/60000 [=============>................] - ETA: 55s - loss: 0.2692 - categorical_accuracy: 0.9167
29472/60000 [=============>................] - ETA: 55s - loss: 0.2690 - categorical_accuracy: 0.9168
29504/60000 [=============>................] - ETA: 55s - loss: 0.2688 - categorical_accuracy: 0.9168
29536/60000 [=============>................] - ETA: 55s - loss: 0.2687 - categorical_accuracy: 0.9168
29568/60000 [=============>................] - ETA: 55s - loss: 0.2686 - categorical_accuracy: 0.9169
29600/60000 [=============>................] - ETA: 55s - loss: 0.2684 - categorical_accuracy: 0.9170
29632/60000 [=============>................] - ETA: 55s - loss: 0.2681 - categorical_accuracy: 0.9170
29664/60000 [=============>................] - ETA: 55s - loss: 0.2680 - categorical_accuracy: 0.9171
29696/60000 [=============>................] - ETA: 55s - loss: 0.2679 - categorical_accuracy: 0.9171
29728/60000 [=============>................] - ETA: 55s - loss: 0.2678 - categorical_accuracy: 0.9171
29760/60000 [=============>................] - ETA: 55s - loss: 0.2675 - categorical_accuracy: 0.9172
29792/60000 [=============>................] - ETA: 54s - loss: 0.2673 - categorical_accuracy: 0.9173
29824/60000 [=============>................] - ETA: 54s - loss: 0.2671 - categorical_accuracy: 0.9174
29856/60000 [=============>................] - ETA: 54s - loss: 0.2670 - categorical_accuracy: 0.9174
29888/60000 [=============>................] - ETA: 54s - loss: 0.2668 - categorical_accuracy: 0.9175
29920/60000 [=============>................] - ETA: 54s - loss: 0.2667 - categorical_accuracy: 0.9175
29952/60000 [=============>................] - ETA: 54s - loss: 0.2665 - categorical_accuracy: 0.9175
29984/60000 [=============>................] - ETA: 54s - loss: 0.2669 - categorical_accuracy: 0.9175
30016/60000 [==============>...............] - ETA: 54s - loss: 0.2666 - categorical_accuracy: 0.9175
30048/60000 [==============>...............] - ETA: 54s - loss: 0.2664 - categorical_accuracy: 0.9176
30080/60000 [==============>...............] - ETA: 54s - loss: 0.2663 - categorical_accuracy: 0.9177
30112/60000 [==============>...............] - ETA: 54s - loss: 0.2663 - categorical_accuracy: 0.9177
30144/60000 [==============>...............] - ETA: 54s - loss: 0.2660 - categorical_accuracy: 0.9178
30176/60000 [==============>...............] - ETA: 54s - loss: 0.2659 - categorical_accuracy: 0.9178
30208/60000 [==============>...............] - ETA: 54s - loss: 0.2661 - categorical_accuracy: 0.9178
30240/60000 [==============>...............] - ETA: 54s - loss: 0.2660 - categorical_accuracy: 0.9178
30272/60000 [==============>...............] - ETA: 54s - loss: 0.2658 - categorical_accuracy: 0.9179
30304/60000 [==============>...............] - ETA: 54s - loss: 0.2656 - categorical_accuracy: 0.9179
30336/60000 [==============>...............] - ETA: 53s - loss: 0.2656 - categorical_accuracy: 0.9179
30368/60000 [==============>...............] - ETA: 53s - loss: 0.2654 - categorical_accuracy: 0.9180
30400/60000 [==============>...............] - ETA: 53s - loss: 0.2653 - categorical_accuracy: 0.9180
30432/60000 [==============>...............] - ETA: 53s - loss: 0.2651 - categorical_accuracy: 0.9181
30464/60000 [==============>...............] - ETA: 53s - loss: 0.2649 - categorical_accuracy: 0.9181
30496/60000 [==============>...............] - ETA: 53s - loss: 0.2648 - categorical_accuracy: 0.9182
30528/60000 [==============>...............] - ETA: 53s - loss: 0.2646 - categorical_accuracy: 0.9182
30560/60000 [==============>...............] - ETA: 53s - loss: 0.2644 - categorical_accuracy: 0.9183
30592/60000 [==============>...............] - ETA: 53s - loss: 0.2642 - categorical_accuracy: 0.9184
30624/60000 [==============>...............] - ETA: 53s - loss: 0.2640 - categorical_accuracy: 0.9185
30656/60000 [==============>...............] - ETA: 53s - loss: 0.2638 - categorical_accuracy: 0.9185
30688/60000 [==============>...............] - ETA: 53s - loss: 0.2636 - categorical_accuracy: 0.9186
30720/60000 [==============>...............] - ETA: 53s - loss: 0.2634 - categorical_accuracy: 0.9186
30752/60000 [==============>...............] - ETA: 53s - loss: 0.2633 - categorical_accuracy: 0.9186
30784/60000 [==============>...............] - ETA: 53s - loss: 0.2630 - categorical_accuracy: 0.9187
30816/60000 [==============>...............] - ETA: 53s - loss: 0.2630 - categorical_accuracy: 0.9187
30848/60000 [==============>...............] - ETA: 53s - loss: 0.2630 - categorical_accuracy: 0.9188
30880/60000 [==============>...............] - ETA: 52s - loss: 0.2634 - categorical_accuracy: 0.9187
30912/60000 [==============>...............] - ETA: 52s - loss: 0.2632 - categorical_accuracy: 0.9187
30944/60000 [==============>...............] - ETA: 52s - loss: 0.2632 - categorical_accuracy: 0.9187
30976/60000 [==============>...............] - ETA: 52s - loss: 0.2630 - categorical_accuracy: 0.9188
31008/60000 [==============>...............] - ETA: 52s - loss: 0.2629 - categorical_accuracy: 0.9188
31040/60000 [==============>...............] - ETA: 52s - loss: 0.2629 - categorical_accuracy: 0.9188
31072/60000 [==============>...............] - ETA: 52s - loss: 0.2626 - categorical_accuracy: 0.9189
31104/60000 [==============>...............] - ETA: 52s - loss: 0.2624 - categorical_accuracy: 0.9190
31136/60000 [==============>...............] - ETA: 52s - loss: 0.2622 - categorical_accuracy: 0.9191
31168/60000 [==============>...............] - ETA: 52s - loss: 0.2620 - categorical_accuracy: 0.9191
31200/60000 [==============>...............] - ETA: 52s - loss: 0.2618 - categorical_accuracy: 0.9192
31232/60000 [==============>...............] - ETA: 52s - loss: 0.2616 - categorical_accuracy: 0.9192
31264/60000 [==============>...............] - ETA: 52s - loss: 0.2616 - categorical_accuracy: 0.9193
31296/60000 [==============>...............] - ETA: 52s - loss: 0.2613 - categorical_accuracy: 0.9194
31328/60000 [==============>...............] - ETA: 52s - loss: 0.2613 - categorical_accuracy: 0.9193
31360/60000 [==============>...............] - ETA: 52s - loss: 0.2611 - categorical_accuracy: 0.9194
31392/60000 [==============>...............] - ETA: 52s - loss: 0.2609 - categorical_accuracy: 0.9195
31424/60000 [==============>...............] - ETA: 51s - loss: 0.2607 - categorical_accuracy: 0.9195
31456/60000 [==============>...............] - ETA: 51s - loss: 0.2605 - categorical_accuracy: 0.9196
31488/60000 [==============>...............] - ETA: 51s - loss: 0.2603 - categorical_accuracy: 0.9197
31520/60000 [==============>...............] - ETA: 51s - loss: 0.2601 - categorical_accuracy: 0.9197
31552/60000 [==============>...............] - ETA: 51s - loss: 0.2599 - categorical_accuracy: 0.9198
31584/60000 [==============>...............] - ETA: 51s - loss: 0.2596 - categorical_accuracy: 0.9199
31616/60000 [==============>...............] - ETA: 51s - loss: 0.2597 - categorical_accuracy: 0.9199
31648/60000 [==============>...............] - ETA: 51s - loss: 0.2597 - categorical_accuracy: 0.9199
31680/60000 [==============>...............] - ETA: 51s - loss: 0.2595 - categorical_accuracy: 0.9200
31712/60000 [==============>...............] - ETA: 51s - loss: 0.2592 - categorical_accuracy: 0.9201
31744/60000 [==============>...............] - ETA: 51s - loss: 0.2594 - categorical_accuracy: 0.9200
31776/60000 [==============>...............] - ETA: 51s - loss: 0.2592 - categorical_accuracy: 0.9201
31808/60000 [==============>...............] - ETA: 51s - loss: 0.2590 - categorical_accuracy: 0.9201
31840/60000 [==============>...............] - ETA: 51s - loss: 0.2590 - categorical_accuracy: 0.9202
31872/60000 [==============>...............] - ETA: 51s - loss: 0.2588 - categorical_accuracy: 0.9202
31904/60000 [==============>...............] - ETA: 51s - loss: 0.2586 - categorical_accuracy: 0.9203
31936/60000 [==============>...............] - ETA: 51s - loss: 0.2586 - categorical_accuracy: 0.9203
31968/60000 [==============>...............] - ETA: 50s - loss: 0.2584 - categorical_accuracy: 0.9204
32000/60000 [===============>..............] - ETA: 50s - loss: 0.2583 - categorical_accuracy: 0.9204
32032/60000 [===============>..............] - ETA: 50s - loss: 0.2580 - categorical_accuracy: 0.9205
32064/60000 [===============>..............] - ETA: 50s - loss: 0.2579 - categorical_accuracy: 0.9205
32096/60000 [===============>..............] - ETA: 50s - loss: 0.2578 - categorical_accuracy: 0.9205
32128/60000 [===============>..............] - ETA: 50s - loss: 0.2576 - categorical_accuracy: 0.9206
32160/60000 [===============>..............] - ETA: 50s - loss: 0.2574 - categorical_accuracy: 0.9206
32192/60000 [===============>..............] - ETA: 50s - loss: 0.2572 - categorical_accuracy: 0.9207
32224/60000 [===============>..............] - ETA: 50s - loss: 0.2570 - categorical_accuracy: 0.9207
32256/60000 [===============>..............] - ETA: 50s - loss: 0.2569 - categorical_accuracy: 0.9208
32288/60000 [===============>..............] - ETA: 50s - loss: 0.2567 - categorical_accuracy: 0.9208
32320/60000 [===============>..............] - ETA: 50s - loss: 0.2566 - categorical_accuracy: 0.9209
32352/60000 [===============>..............] - ETA: 50s - loss: 0.2565 - categorical_accuracy: 0.9209
32384/60000 [===============>..............] - ETA: 50s - loss: 0.2564 - categorical_accuracy: 0.9209
32416/60000 [===============>..............] - ETA: 50s - loss: 0.2563 - categorical_accuracy: 0.9209
32448/60000 [===============>..............] - ETA: 50s - loss: 0.2562 - categorical_accuracy: 0.9210
32480/60000 [===============>..............] - ETA: 50s - loss: 0.2560 - categorical_accuracy: 0.9210
32512/60000 [===============>..............] - ETA: 50s - loss: 0.2557 - categorical_accuracy: 0.9211
32544/60000 [===============>..............] - ETA: 49s - loss: 0.2556 - categorical_accuracy: 0.9212
32576/60000 [===============>..............] - ETA: 49s - loss: 0.2556 - categorical_accuracy: 0.9212
32608/60000 [===============>..............] - ETA: 49s - loss: 0.2554 - categorical_accuracy: 0.9212
32640/60000 [===============>..............] - ETA: 49s - loss: 0.2552 - categorical_accuracy: 0.9213
32672/60000 [===============>..............] - ETA: 49s - loss: 0.2553 - categorical_accuracy: 0.9213
32704/60000 [===============>..............] - ETA: 49s - loss: 0.2551 - categorical_accuracy: 0.9213
32736/60000 [===============>..............] - ETA: 49s - loss: 0.2551 - categorical_accuracy: 0.9213
32768/60000 [===============>..............] - ETA: 49s - loss: 0.2550 - categorical_accuracy: 0.9213
32800/60000 [===============>..............] - ETA: 49s - loss: 0.2549 - categorical_accuracy: 0.9214
32832/60000 [===============>..............] - ETA: 49s - loss: 0.2547 - categorical_accuracy: 0.9214
32864/60000 [===============>..............] - ETA: 49s - loss: 0.2551 - categorical_accuracy: 0.9213
32896/60000 [===============>..............] - ETA: 49s - loss: 0.2549 - categorical_accuracy: 0.9214
32928/60000 [===============>..............] - ETA: 49s - loss: 0.2547 - categorical_accuracy: 0.9214
32960/60000 [===============>..............] - ETA: 49s - loss: 0.2545 - categorical_accuracy: 0.9215
32992/60000 [===============>..............] - ETA: 49s - loss: 0.2547 - categorical_accuracy: 0.9214
33024/60000 [===============>..............] - ETA: 49s - loss: 0.2545 - categorical_accuracy: 0.9215
33056/60000 [===============>..............] - ETA: 49s - loss: 0.2544 - categorical_accuracy: 0.9215
33088/60000 [===============>..............] - ETA: 48s - loss: 0.2543 - categorical_accuracy: 0.9215
33120/60000 [===============>..............] - ETA: 48s - loss: 0.2541 - categorical_accuracy: 0.9216
33152/60000 [===============>..............] - ETA: 48s - loss: 0.2542 - categorical_accuracy: 0.9215
33184/60000 [===============>..............] - ETA: 48s - loss: 0.2540 - categorical_accuracy: 0.9216
33216/60000 [===============>..............] - ETA: 48s - loss: 0.2539 - categorical_accuracy: 0.9216
33248/60000 [===============>..............] - ETA: 48s - loss: 0.2538 - categorical_accuracy: 0.9216
33280/60000 [===============>..............] - ETA: 48s - loss: 0.2537 - categorical_accuracy: 0.9217
33312/60000 [===============>..............] - ETA: 48s - loss: 0.2536 - categorical_accuracy: 0.9217
33344/60000 [===============>..............] - ETA: 48s - loss: 0.2537 - categorical_accuracy: 0.9217
33376/60000 [===============>..............] - ETA: 48s - loss: 0.2536 - categorical_accuracy: 0.9217
33408/60000 [===============>..............] - ETA: 48s - loss: 0.2534 - categorical_accuracy: 0.9218
33440/60000 [===============>..............] - ETA: 48s - loss: 0.2533 - categorical_accuracy: 0.9219
33472/60000 [===============>..............] - ETA: 48s - loss: 0.2531 - categorical_accuracy: 0.9219
33504/60000 [===============>..............] - ETA: 48s - loss: 0.2530 - categorical_accuracy: 0.9219
33536/60000 [===============>..............] - ETA: 48s - loss: 0.2530 - categorical_accuracy: 0.9220
33568/60000 [===============>..............] - ETA: 48s - loss: 0.2528 - categorical_accuracy: 0.9221
33600/60000 [===============>..............] - ETA: 48s - loss: 0.2526 - categorical_accuracy: 0.9221
33632/60000 [===============>..............] - ETA: 47s - loss: 0.2524 - categorical_accuracy: 0.9222
33664/60000 [===============>..............] - ETA: 47s - loss: 0.2522 - categorical_accuracy: 0.9222
33696/60000 [===============>..............] - ETA: 47s - loss: 0.2520 - categorical_accuracy: 0.9223
33728/60000 [===============>..............] - ETA: 47s - loss: 0.2519 - categorical_accuracy: 0.9223
33760/60000 [===============>..............] - ETA: 47s - loss: 0.2518 - categorical_accuracy: 0.9223
33792/60000 [===============>..............] - ETA: 47s - loss: 0.2517 - categorical_accuracy: 0.9223
33824/60000 [===============>..............] - ETA: 47s - loss: 0.2515 - categorical_accuracy: 0.9224
33856/60000 [===============>..............] - ETA: 47s - loss: 0.2515 - categorical_accuracy: 0.9224
33888/60000 [===============>..............] - ETA: 47s - loss: 0.2514 - categorical_accuracy: 0.9224
33920/60000 [===============>..............] - ETA: 47s - loss: 0.2512 - categorical_accuracy: 0.9225
33952/60000 [===============>..............] - ETA: 47s - loss: 0.2509 - categorical_accuracy: 0.9226
33984/60000 [===============>..............] - ETA: 47s - loss: 0.2508 - categorical_accuracy: 0.9226
34016/60000 [================>.............] - ETA: 47s - loss: 0.2506 - categorical_accuracy: 0.9227
34048/60000 [================>.............] - ETA: 47s - loss: 0.2504 - categorical_accuracy: 0.9227
34080/60000 [================>.............] - ETA: 47s - loss: 0.2503 - categorical_accuracy: 0.9228
34112/60000 [================>.............] - ETA: 47s - loss: 0.2503 - categorical_accuracy: 0.9228
34144/60000 [================>.............] - ETA: 47s - loss: 0.2501 - categorical_accuracy: 0.9229
34176/60000 [================>.............] - ETA: 47s - loss: 0.2501 - categorical_accuracy: 0.9229
34208/60000 [================>.............] - ETA: 46s - loss: 0.2500 - categorical_accuracy: 0.9229
34240/60000 [================>.............] - ETA: 46s - loss: 0.2499 - categorical_accuracy: 0.9230
34272/60000 [================>.............] - ETA: 46s - loss: 0.2497 - categorical_accuracy: 0.9230
34304/60000 [================>.............] - ETA: 46s - loss: 0.2495 - categorical_accuracy: 0.9231
34336/60000 [================>.............] - ETA: 46s - loss: 0.2494 - categorical_accuracy: 0.9231
34368/60000 [================>.............] - ETA: 46s - loss: 0.2493 - categorical_accuracy: 0.9231
34400/60000 [================>.............] - ETA: 46s - loss: 0.2492 - categorical_accuracy: 0.9231
34432/60000 [================>.............] - ETA: 46s - loss: 0.2492 - categorical_accuracy: 0.9231
34464/60000 [================>.............] - ETA: 46s - loss: 0.2490 - categorical_accuracy: 0.9232
34496/60000 [================>.............] - ETA: 46s - loss: 0.2488 - categorical_accuracy: 0.9232
34528/60000 [================>.............] - ETA: 46s - loss: 0.2491 - categorical_accuracy: 0.9232
34560/60000 [================>.............] - ETA: 46s - loss: 0.2489 - categorical_accuracy: 0.9232
34592/60000 [================>.............] - ETA: 46s - loss: 0.2489 - categorical_accuracy: 0.9232
34624/60000 [================>.............] - ETA: 46s - loss: 0.2487 - categorical_accuracy: 0.9233
34656/60000 [================>.............] - ETA: 46s - loss: 0.2485 - categorical_accuracy: 0.9234
34688/60000 [================>.............] - ETA: 46s - loss: 0.2484 - categorical_accuracy: 0.9233
34720/60000 [================>.............] - ETA: 46s - loss: 0.2483 - categorical_accuracy: 0.9234
34752/60000 [================>.............] - ETA: 45s - loss: 0.2482 - categorical_accuracy: 0.9234
34784/60000 [================>.............] - ETA: 45s - loss: 0.2481 - categorical_accuracy: 0.9234
34816/60000 [================>.............] - ETA: 45s - loss: 0.2480 - categorical_accuracy: 0.9235
34848/60000 [================>.............] - ETA: 45s - loss: 0.2478 - categorical_accuracy: 0.9236
34880/60000 [================>.............] - ETA: 45s - loss: 0.2476 - categorical_accuracy: 0.9236
34912/60000 [================>.............] - ETA: 45s - loss: 0.2475 - categorical_accuracy: 0.9237
34944/60000 [================>.............] - ETA: 45s - loss: 0.2474 - categorical_accuracy: 0.9237
34976/60000 [================>.............] - ETA: 45s - loss: 0.2474 - categorical_accuracy: 0.9237
35008/60000 [================>.............] - ETA: 45s - loss: 0.2474 - categorical_accuracy: 0.9237
35040/60000 [================>.............] - ETA: 45s - loss: 0.2472 - categorical_accuracy: 0.9237
35072/60000 [================>.............] - ETA: 45s - loss: 0.2470 - categorical_accuracy: 0.9238
35104/60000 [================>.............] - ETA: 45s - loss: 0.2470 - categorical_accuracy: 0.9238
35136/60000 [================>.............] - ETA: 45s - loss: 0.2469 - categorical_accuracy: 0.9238
35168/60000 [================>.............] - ETA: 45s - loss: 0.2468 - categorical_accuracy: 0.9239
35200/60000 [================>.............] - ETA: 45s - loss: 0.2466 - categorical_accuracy: 0.9239
35232/60000 [================>.............] - ETA: 45s - loss: 0.2465 - categorical_accuracy: 0.9240
35264/60000 [================>.............] - ETA: 45s - loss: 0.2465 - categorical_accuracy: 0.9239
35296/60000 [================>.............] - ETA: 44s - loss: 0.2463 - categorical_accuracy: 0.9240
35328/60000 [================>.............] - ETA: 44s - loss: 0.2461 - categorical_accuracy: 0.9241
35360/60000 [================>.............] - ETA: 44s - loss: 0.2459 - categorical_accuracy: 0.9241
35392/60000 [================>.............] - ETA: 44s - loss: 0.2457 - categorical_accuracy: 0.9242
35424/60000 [================>.............] - ETA: 44s - loss: 0.2456 - categorical_accuracy: 0.9242
35456/60000 [================>.............] - ETA: 44s - loss: 0.2455 - categorical_accuracy: 0.9243
35488/60000 [================>.............] - ETA: 44s - loss: 0.2453 - categorical_accuracy: 0.9243
35520/60000 [================>.............] - ETA: 44s - loss: 0.2453 - categorical_accuracy: 0.9244
35552/60000 [================>.............] - ETA: 44s - loss: 0.2452 - categorical_accuracy: 0.9244
35584/60000 [================>.............] - ETA: 44s - loss: 0.2453 - categorical_accuracy: 0.9243
35616/60000 [================>.............] - ETA: 44s - loss: 0.2452 - categorical_accuracy: 0.9244
35648/60000 [================>.............] - ETA: 44s - loss: 0.2451 - categorical_accuracy: 0.9245
35680/60000 [================>.............] - ETA: 44s - loss: 0.2450 - categorical_accuracy: 0.9245
35712/60000 [================>.............] - ETA: 44s - loss: 0.2450 - categorical_accuracy: 0.9245
35744/60000 [================>.............] - ETA: 44s - loss: 0.2448 - categorical_accuracy: 0.9246
35776/60000 [================>.............] - ETA: 44s - loss: 0.2446 - categorical_accuracy: 0.9246
35808/60000 [================>.............] - ETA: 44s - loss: 0.2444 - categorical_accuracy: 0.9247
35840/60000 [================>.............] - ETA: 44s - loss: 0.2443 - categorical_accuracy: 0.9247
35872/60000 [================>.............] - ETA: 43s - loss: 0.2441 - categorical_accuracy: 0.9248
35904/60000 [================>.............] - ETA: 43s - loss: 0.2442 - categorical_accuracy: 0.9247
35936/60000 [================>.............] - ETA: 43s - loss: 0.2443 - categorical_accuracy: 0.9246
35968/60000 [================>.............] - ETA: 43s - loss: 0.2441 - categorical_accuracy: 0.9247
36000/60000 [=================>............] - ETA: 43s - loss: 0.2439 - categorical_accuracy: 0.9247
36032/60000 [=================>............] - ETA: 43s - loss: 0.2438 - categorical_accuracy: 0.9248
36064/60000 [=================>............] - ETA: 43s - loss: 0.2436 - categorical_accuracy: 0.9248
36096/60000 [=================>............] - ETA: 43s - loss: 0.2434 - categorical_accuracy: 0.9249
36128/60000 [=================>............] - ETA: 43s - loss: 0.2433 - categorical_accuracy: 0.9249
36160/60000 [=================>............] - ETA: 43s - loss: 0.2433 - categorical_accuracy: 0.9249
36192/60000 [=================>............] - ETA: 43s - loss: 0.2432 - categorical_accuracy: 0.9250
36224/60000 [=================>............] - ETA: 43s - loss: 0.2431 - categorical_accuracy: 0.9250
36256/60000 [=================>............] - ETA: 43s - loss: 0.2430 - categorical_accuracy: 0.9250
36288/60000 [=================>............] - ETA: 43s - loss: 0.2428 - categorical_accuracy: 0.9251
36320/60000 [=================>............] - ETA: 43s - loss: 0.2426 - categorical_accuracy: 0.9251
36352/60000 [=================>............] - ETA: 43s - loss: 0.2424 - categorical_accuracy: 0.9251
36384/60000 [=================>............] - ETA: 43s - loss: 0.2423 - categorical_accuracy: 0.9252
36416/60000 [=================>............] - ETA: 42s - loss: 0.2421 - categorical_accuracy: 0.9252
36448/60000 [=================>............] - ETA: 42s - loss: 0.2420 - categorical_accuracy: 0.9253
36480/60000 [=================>............] - ETA: 42s - loss: 0.2419 - categorical_accuracy: 0.9252
36512/60000 [=================>............] - ETA: 42s - loss: 0.2419 - categorical_accuracy: 0.9252
36544/60000 [=================>............] - ETA: 42s - loss: 0.2419 - categorical_accuracy: 0.9252
36576/60000 [=================>............] - ETA: 42s - loss: 0.2417 - categorical_accuracy: 0.9253
36608/60000 [=================>............] - ETA: 42s - loss: 0.2418 - categorical_accuracy: 0.9252
36640/60000 [=================>............] - ETA: 42s - loss: 0.2417 - categorical_accuracy: 0.9252
36672/60000 [=================>............] - ETA: 42s - loss: 0.2416 - categorical_accuracy: 0.9252
36704/60000 [=================>............] - ETA: 42s - loss: 0.2414 - categorical_accuracy: 0.9253
36736/60000 [=================>............] - ETA: 42s - loss: 0.2413 - categorical_accuracy: 0.9254
36768/60000 [=================>............] - ETA: 42s - loss: 0.2411 - categorical_accuracy: 0.9254
36800/60000 [=================>............] - ETA: 42s - loss: 0.2411 - categorical_accuracy: 0.9254
36832/60000 [=================>............] - ETA: 42s - loss: 0.2410 - categorical_accuracy: 0.9254
36864/60000 [=================>............] - ETA: 42s - loss: 0.2408 - categorical_accuracy: 0.9255
36896/60000 [=================>............] - ETA: 42s - loss: 0.2407 - categorical_accuracy: 0.9255
36928/60000 [=================>............] - ETA: 42s - loss: 0.2406 - categorical_accuracy: 0.9255
36960/60000 [=================>............] - ETA: 41s - loss: 0.2404 - categorical_accuracy: 0.9256
36992/60000 [=================>............] - ETA: 41s - loss: 0.2404 - categorical_accuracy: 0.9256
37024/60000 [=================>............] - ETA: 41s - loss: 0.2403 - categorical_accuracy: 0.9256
37056/60000 [=================>............] - ETA: 41s - loss: 0.2401 - categorical_accuracy: 0.9257
37088/60000 [=================>............] - ETA: 41s - loss: 0.2400 - categorical_accuracy: 0.9257
37120/60000 [=================>............] - ETA: 41s - loss: 0.2399 - categorical_accuracy: 0.9258
37152/60000 [=================>............] - ETA: 41s - loss: 0.2398 - categorical_accuracy: 0.9258
37184/60000 [=================>............] - ETA: 41s - loss: 0.2397 - categorical_accuracy: 0.9258
37216/60000 [=================>............] - ETA: 41s - loss: 0.2396 - categorical_accuracy: 0.9259
37248/60000 [=================>............] - ETA: 41s - loss: 0.2394 - categorical_accuracy: 0.9259
37280/60000 [=================>............] - ETA: 41s - loss: 0.2392 - categorical_accuracy: 0.9260
37312/60000 [=================>............] - ETA: 41s - loss: 0.2392 - categorical_accuracy: 0.9260
37344/60000 [=================>............] - ETA: 41s - loss: 0.2390 - categorical_accuracy: 0.9260
37376/60000 [=================>............] - ETA: 41s - loss: 0.2390 - categorical_accuracy: 0.9260
37408/60000 [=================>............] - ETA: 41s - loss: 0.2389 - categorical_accuracy: 0.9261
37440/60000 [=================>............] - ETA: 41s - loss: 0.2389 - categorical_accuracy: 0.9260
37472/60000 [=================>............] - ETA: 41s - loss: 0.2388 - categorical_accuracy: 0.9261
37504/60000 [=================>............] - ETA: 40s - loss: 0.2387 - categorical_accuracy: 0.9261
37536/60000 [=================>............] - ETA: 40s - loss: 0.2387 - categorical_accuracy: 0.9261
37568/60000 [=================>............] - ETA: 40s - loss: 0.2386 - categorical_accuracy: 0.9261
37600/60000 [=================>............] - ETA: 40s - loss: 0.2385 - categorical_accuracy: 0.9262
37632/60000 [=================>............] - ETA: 40s - loss: 0.2384 - categorical_accuracy: 0.9262
37664/60000 [=================>............] - ETA: 40s - loss: 0.2383 - categorical_accuracy: 0.9262
37696/60000 [=================>............] - ETA: 40s - loss: 0.2382 - categorical_accuracy: 0.9262
37728/60000 [=================>............] - ETA: 40s - loss: 0.2381 - categorical_accuracy: 0.9262
37760/60000 [=================>............] - ETA: 40s - loss: 0.2381 - categorical_accuracy: 0.9262
37792/60000 [=================>............] - ETA: 40s - loss: 0.2379 - categorical_accuracy: 0.9263
37824/60000 [=================>............] - ETA: 40s - loss: 0.2379 - categorical_accuracy: 0.9263
37856/60000 [=================>............] - ETA: 40s - loss: 0.2377 - categorical_accuracy: 0.9264
37888/60000 [=================>............] - ETA: 40s - loss: 0.2376 - categorical_accuracy: 0.9264
37920/60000 [=================>............] - ETA: 40s - loss: 0.2375 - categorical_accuracy: 0.9263
37952/60000 [=================>............] - ETA: 40s - loss: 0.2374 - categorical_accuracy: 0.9264
37984/60000 [=================>............] - ETA: 40s - loss: 0.2373 - categorical_accuracy: 0.9264
38016/60000 [==================>...........] - ETA: 40s - loss: 0.2373 - categorical_accuracy: 0.9265
38048/60000 [==================>...........] - ETA: 39s - loss: 0.2371 - categorical_accuracy: 0.9265
38080/60000 [==================>...........] - ETA: 39s - loss: 0.2369 - categorical_accuracy: 0.9266
38112/60000 [==================>...........] - ETA: 39s - loss: 0.2369 - categorical_accuracy: 0.9266
38144/60000 [==================>...........] - ETA: 39s - loss: 0.2367 - categorical_accuracy: 0.9266
38176/60000 [==================>...........] - ETA: 39s - loss: 0.2366 - categorical_accuracy: 0.9267
38208/60000 [==================>...........] - ETA: 39s - loss: 0.2364 - categorical_accuracy: 0.9268
38240/60000 [==================>...........] - ETA: 39s - loss: 0.2362 - categorical_accuracy: 0.9268
38272/60000 [==================>...........] - ETA: 39s - loss: 0.2360 - categorical_accuracy: 0.9269
38304/60000 [==================>...........] - ETA: 39s - loss: 0.2361 - categorical_accuracy: 0.9269
38336/60000 [==================>...........] - ETA: 39s - loss: 0.2360 - categorical_accuracy: 0.9269
38368/60000 [==================>...........] - ETA: 39s - loss: 0.2359 - categorical_accuracy: 0.9269
38400/60000 [==================>...........] - ETA: 39s - loss: 0.2358 - categorical_accuracy: 0.9270
38432/60000 [==================>...........] - ETA: 39s - loss: 0.2357 - categorical_accuracy: 0.9270
38464/60000 [==================>...........] - ETA: 39s - loss: 0.2356 - categorical_accuracy: 0.9270
38496/60000 [==================>...........] - ETA: 39s - loss: 0.2354 - categorical_accuracy: 0.9271
38528/60000 [==================>...........] - ETA: 39s - loss: 0.2353 - categorical_accuracy: 0.9272
38560/60000 [==================>...........] - ETA: 39s - loss: 0.2351 - categorical_accuracy: 0.9272
38592/60000 [==================>...........] - ETA: 38s - loss: 0.2349 - categorical_accuracy: 0.9273
38624/60000 [==================>...........] - ETA: 38s - loss: 0.2347 - categorical_accuracy: 0.9274
38656/60000 [==================>...........] - ETA: 38s - loss: 0.2347 - categorical_accuracy: 0.9274
38688/60000 [==================>...........] - ETA: 38s - loss: 0.2345 - categorical_accuracy: 0.9274
38720/60000 [==================>...........] - ETA: 38s - loss: 0.2343 - categorical_accuracy: 0.9275
38752/60000 [==================>...........] - ETA: 38s - loss: 0.2341 - categorical_accuracy: 0.9276
38784/60000 [==================>...........] - ETA: 38s - loss: 0.2344 - categorical_accuracy: 0.9275
38816/60000 [==================>...........] - ETA: 38s - loss: 0.2344 - categorical_accuracy: 0.9276
38848/60000 [==================>...........] - ETA: 38s - loss: 0.2343 - categorical_accuracy: 0.9276
38880/60000 [==================>...........] - ETA: 38s - loss: 0.2341 - categorical_accuracy: 0.9276
38912/60000 [==================>...........] - ETA: 38s - loss: 0.2342 - categorical_accuracy: 0.9276
38944/60000 [==================>...........] - ETA: 38s - loss: 0.2341 - categorical_accuracy: 0.9276
38976/60000 [==================>...........] - ETA: 38s - loss: 0.2339 - categorical_accuracy: 0.9277
39008/60000 [==================>...........] - ETA: 38s - loss: 0.2339 - categorical_accuracy: 0.9277
39040/60000 [==================>...........] - ETA: 38s - loss: 0.2338 - categorical_accuracy: 0.9277
39072/60000 [==================>...........] - ETA: 38s - loss: 0.2336 - categorical_accuracy: 0.9278
39104/60000 [==================>...........] - ETA: 38s - loss: 0.2335 - categorical_accuracy: 0.9278
39136/60000 [==================>...........] - ETA: 37s - loss: 0.2334 - categorical_accuracy: 0.9279
39168/60000 [==================>...........] - ETA: 37s - loss: 0.2332 - categorical_accuracy: 0.9279
39200/60000 [==================>...........] - ETA: 37s - loss: 0.2331 - categorical_accuracy: 0.9279
39232/60000 [==================>...........] - ETA: 37s - loss: 0.2330 - categorical_accuracy: 0.9279
39264/60000 [==================>...........] - ETA: 37s - loss: 0.2328 - categorical_accuracy: 0.9280
39296/60000 [==================>...........] - ETA: 37s - loss: 0.2327 - categorical_accuracy: 0.9281
39328/60000 [==================>...........] - ETA: 37s - loss: 0.2327 - categorical_accuracy: 0.9281
39360/60000 [==================>...........] - ETA: 37s - loss: 0.2325 - categorical_accuracy: 0.9281
39392/60000 [==================>...........] - ETA: 37s - loss: 0.2324 - categorical_accuracy: 0.9282
39424/60000 [==================>...........] - ETA: 37s - loss: 0.2326 - categorical_accuracy: 0.9282
39456/60000 [==================>...........] - ETA: 37s - loss: 0.2326 - categorical_accuracy: 0.9282
39488/60000 [==================>...........] - ETA: 37s - loss: 0.2326 - categorical_accuracy: 0.9282
39520/60000 [==================>...........] - ETA: 37s - loss: 0.2326 - categorical_accuracy: 0.9282
39552/60000 [==================>...........] - ETA: 37s - loss: 0.2324 - categorical_accuracy: 0.9282
39584/60000 [==================>...........] - ETA: 37s - loss: 0.2323 - categorical_accuracy: 0.9283
39616/60000 [==================>...........] - ETA: 37s - loss: 0.2324 - categorical_accuracy: 0.9282
39648/60000 [==================>...........] - ETA: 37s - loss: 0.2323 - categorical_accuracy: 0.9283
39680/60000 [==================>...........] - ETA: 36s - loss: 0.2321 - categorical_accuracy: 0.9283
39712/60000 [==================>...........] - ETA: 36s - loss: 0.2321 - categorical_accuracy: 0.9283
39744/60000 [==================>...........] - ETA: 36s - loss: 0.2320 - categorical_accuracy: 0.9283
39776/60000 [==================>...........] - ETA: 36s - loss: 0.2320 - categorical_accuracy: 0.9283
39808/60000 [==================>...........] - ETA: 36s - loss: 0.2319 - categorical_accuracy: 0.9283
39840/60000 [==================>...........] - ETA: 36s - loss: 0.2318 - categorical_accuracy: 0.9283
39872/60000 [==================>...........] - ETA: 36s - loss: 0.2317 - categorical_accuracy: 0.9284
39904/60000 [==================>...........] - ETA: 36s - loss: 0.2315 - categorical_accuracy: 0.9285
39936/60000 [==================>...........] - ETA: 36s - loss: 0.2313 - categorical_accuracy: 0.9285
39968/60000 [==================>...........] - ETA: 36s - loss: 0.2313 - categorical_accuracy: 0.9285
40000/60000 [===================>..........] - ETA: 36s - loss: 0.2312 - categorical_accuracy: 0.9286
40032/60000 [===================>..........] - ETA: 36s - loss: 0.2311 - categorical_accuracy: 0.9286
40064/60000 [===================>..........] - ETA: 36s - loss: 0.2312 - categorical_accuracy: 0.9285
40096/60000 [===================>..........] - ETA: 36s - loss: 0.2310 - categorical_accuracy: 0.9286
40128/60000 [===================>..........] - ETA: 36s - loss: 0.2309 - categorical_accuracy: 0.9286
40160/60000 [===================>..........] - ETA: 36s - loss: 0.2308 - categorical_accuracy: 0.9287
40192/60000 [===================>..........] - ETA: 36s - loss: 0.2306 - categorical_accuracy: 0.9287
40224/60000 [===================>..........] - ETA: 35s - loss: 0.2306 - categorical_accuracy: 0.9287
40256/60000 [===================>..........] - ETA: 35s - loss: 0.2305 - categorical_accuracy: 0.9287
40288/60000 [===================>..........] - ETA: 35s - loss: 0.2305 - categorical_accuracy: 0.9287
40320/60000 [===================>..........] - ETA: 35s - loss: 0.2304 - categorical_accuracy: 0.9287
40352/60000 [===================>..........] - ETA: 35s - loss: 0.2304 - categorical_accuracy: 0.9287
40384/60000 [===================>..........] - ETA: 35s - loss: 0.2302 - categorical_accuracy: 0.9288
40416/60000 [===================>..........] - ETA: 35s - loss: 0.2304 - categorical_accuracy: 0.9287
40448/60000 [===================>..........] - ETA: 35s - loss: 0.2303 - categorical_accuracy: 0.9287
40480/60000 [===================>..........] - ETA: 35s - loss: 0.2304 - categorical_accuracy: 0.9287
40512/60000 [===================>..........] - ETA: 35s - loss: 0.2302 - categorical_accuracy: 0.9287
40544/60000 [===================>..........] - ETA: 35s - loss: 0.2301 - categorical_accuracy: 0.9288
40576/60000 [===================>..........] - ETA: 35s - loss: 0.2300 - categorical_accuracy: 0.9288
40608/60000 [===================>..........] - ETA: 35s - loss: 0.2299 - categorical_accuracy: 0.9288
40640/60000 [===================>..........] - ETA: 35s - loss: 0.2298 - categorical_accuracy: 0.9289
40672/60000 [===================>..........] - ETA: 35s - loss: 0.2299 - categorical_accuracy: 0.9288
40704/60000 [===================>..........] - ETA: 35s - loss: 0.2298 - categorical_accuracy: 0.9289
40736/60000 [===================>..........] - ETA: 35s - loss: 0.2297 - categorical_accuracy: 0.9289
40768/60000 [===================>..........] - ETA: 34s - loss: 0.2295 - categorical_accuracy: 0.9289
40800/60000 [===================>..........] - ETA: 34s - loss: 0.2293 - categorical_accuracy: 0.9290
40832/60000 [===================>..........] - ETA: 34s - loss: 0.2292 - categorical_accuracy: 0.9290
40864/60000 [===================>..........] - ETA: 34s - loss: 0.2291 - categorical_accuracy: 0.9290
40896/60000 [===================>..........] - ETA: 34s - loss: 0.2289 - categorical_accuracy: 0.9291
40928/60000 [===================>..........] - ETA: 34s - loss: 0.2289 - categorical_accuracy: 0.9291
40960/60000 [===================>..........] - ETA: 34s - loss: 0.2288 - categorical_accuracy: 0.9291
40992/60000 [===================>..........] - ETA: 34s - loss: 0.2286 - categorical_accuracy: 0.9292
41024/60000 [===================>..........] - ETA: 34s - loss: 0.2284 - categorical_accuracy: 0.9292
41056/60000 [===================>..........] - ETA: 34s - loss: 0.2283 - categorical_accuracy: 0.9292
41088/60000 [===================>..........] - ETA: 34s - loss: 0.2282 - categorical_accuracy: 0.9292
41120/60000 [===================>..........] - ETA: 34s - loss: 0.2281 - categorical_accuracy: 0.9293
41152/60000 [===================>..........] - ETA: 34s - loss: 0.2280 - categorical_accuracy: 0.9293
41184/60000 [===================>..........] - ETA: 34s - loss: 0.2278 - categorical_accuracy: 0.9294
41216/60000 [===================>..........] - ETA: 34s - loss: 0.2277 - categorical_accuracy: 0.9294
41248/60000 [===================>..........] - ETA: 34s - loss: 0.2275 - categorical_accuracy: 0.9295
41280/60000 [===================>..........] - ETA: 34s - loss: 0.2276 - categorical_accuracy: 0.9295
41312/60000 [===================>..........] - ETA: 33s - loss: 0.2276 - categorical_accuracy: 0.9294
41344/60000 [===================>..........] - ETA: 33s - loss: 0.2275 - categorical_accuracy: 0.9294
41376/60000 [===================>..........] - ETA: 33s - loss: 0.2274 - categorical_accuracy: 0.9295
41408/60000 [===================>..........] - ETA: 33s - loss: 0.2272 - categorical_accuracy: 0.9295
41440/60000 [===================>..........] - ETA: 33s - loss: 0.2272 - categorical_accuracy: 0.9296
41472/60000 [===================>..........] - ETA: 33s - loss: 0.2271 - categorical_accuracy: 0.9296
41504/60000 [===================>..........] - ETA: 33s - loss: 0.2270 - categorical_accuracy: 0.9296
41536/60000 [===================>..........] - ETA: 33s - loss: 0.2268 - categorical_accuracy: 0.9297
41568/60000 [===================>..........] - ETA: 33s - loss: 0.2267 - categorical_accuracy: 0.9297
41600/60000 [===================>..........] - ETA: 33s - loss: 0.2267 - categorical_accuracy: 0.9297
41632/60000 [===================>..........] - ETA: 33s - loss: 0.2266 - categorical_accuracy: 0.9297
41664/60000 [===================>..........] - ETA: 33s - loss: 0.2264 - categorical_accuracy: 0.9297
41696/60000 [===================>..........] - ETA: 33s - loss: 0.2264 - categorical_accuracy: 0.9298
41728/60000 [===================>..........] - ETA: 33s - loss: 0.2263 - categorical_accuracy: 0.9298
41760/60000 [===================>..........] - ETA: 33s - loss: 0.2262 - categorical_accuracy: 0.9298
41792/60000 [===================>..........] - ETA: 33s - loss: 0.2262 - categorical_accuracy: 0.9298
41824/60000 [===================>..........] - ETA: 33s - loss: 0.2262 - categorical_accuracy: 0.9298
41856/60000 [===================>..........] - ETA: 32s - loss: 0.2262 - categorical_accuracy: 0.9298
41888/60000 [===================>..........] - ETA: 32s - loss: 0.2261 - categorical_accuracy: 0.9299
41920/60000 [===================>..........] - ETA: 32s - loss: 0.2259 - categorical_accuracy: 0.9299
41952/60000 [===================>..........] - ETA: 32s - loss: 0.2258 - categorical_accuracy: 0.9300
41984/60000 [===================>..........] - ETA: 32s - loss: 0.2257 - categorical_accuracy: 0.9300
42016/60000 [====================>.........] - ETA: 32s - loss: 0.2258 - categorical_accuracy: 0.9300
42048/60000 [====================>.........] - ETA: 32s - loss: 0.2257 - categorical_accuracy: 0.9300
42080/60000 [====================>.........] - ETA: 32s - loss: 0.2256 - categorical_accuracy: 0.9301
42112/60000 [====================>.........] - ETA: 32s - loss: 0.2254 - categorical_accuracy: 0.9301
42144/60000 [====================>.........] - ETA: 32s - loss: 0.2253 - categorical_accuracy: 0.9302
42176/60000 [====================>.........] - ETA: 32s - loss: 0.2252 - categorical_accuracy: 0.9302
42208/60000 [====================>.........] - ETA: 32s - loss: 0.2251 - categorical_accuracy: 0.9302
42240/60000 [====================>.........] - ETA: 32s - loss: 0.2250 - categorical_accuracy: 0.9303
42272/60000 [====================>.........] - ETA: 32s - loss: 0.2250 - categorical_accuracy: 0.9303
42304/60000 [====================>.........] - ETA: 32s - loss: 0.2249 - categorical_accuracy: 0.9303
42336/60000 [====================>.........] - ETA: 32s - loss: 0.2248 - categorical_accuracy: 0.9303
42368/60000 [====================>.........] - ETA: 32s - loss: 0.2247 - categorical_accuracy: 0.9303
42400/60000 [====================>.........] - ETA: 31s - loss: 0.2246 - categorical_accuracy: 0.9304
42432/60000 [====================>.........] - ETA: 31s - loss: 0.2247 - categorical_accuracy: 0.9303
42464/60000 [====================>.........] - ETA: 31s - loss: 0.2248 - categorical_accuracy: 0.9303
42496/60000 [====================>.........] - ETA: 31s - loss: 0.2246 - categorical_accuracy: 0.9303
42528/60000 [====================>.........] - ETA: 31s - loss: 0.2245 - categorical_accuracy: 0.9304
42560/60000 [====================>.........] - ETA: 31s - loss: 0.2244 - categorical_accuracy: 0.9304
42592/60000 [====================>.........] - ETA: 31s - loss: 0.2243 - categorical_accuracy: 0.9304
42624/60000 [====================>.........] - ETA: 31s - loss: 0.2242 - categorical_accuracy: 0.9305
42656/60000 [====================>.........] - ETA: 31s - loss: 0.2240 - categorical_accuracy: 0.9305
42688/60000 [====================>.........] - ETA: 31s - loss: 0.2239 - categorical_accuracy: 0.9305
42720/60000 [====================>.........] - ETA: 31s - loss: 0.2238 - categorical_accuracy: 0.9305
42752/60000 [====================>.........] - ETA: 31s - loss: 0.2237 - categorical_accuracy: 0.9306
42784/60000 [====================>.........] - ETA: 31s - loss: 0.2236 - categorical_accuracy: 0.9306
42816/60000 [====================>.........] - ETA: 31s - loss: 0.2236 - categorical_accuracy: 0.9306
42848/60000 [====================>.........] - ETA: 31s - loss: 0.2237 - categorical_accuracy: 0.9306
42880/60000 [====================>.........] - ETA: 31s - loss: 0.2237 - categorical_accuracy: 0.9306
42912/60000 [====================>.........] - ETA: 31s - loss: 0.2235 - categorical_accuracy: 0.9306
42944/60000 [====================>.........] - ETA: 30s - loss: 0.2234 - categorical_accuracy: 0.9307
42976/60000 [====================>.........] - ETA: 30s - loss: 0.2233 - categorical_accuracy: 0.9307
43008/60000 [====================>.........] - ETA: 30s - loss: 0.2233 - categorical_accuracy: 0.9307
43040/60000 [====================>.........] - ETA: 30s - loss: 0.2234 - categorical_accuracy: 0.9307
43072/60000 [====================>.........] - ETA: 30s - loss: 0.2234 - categorical_accuracy: 0.9307
43104/60000 [====================>.........] - ETA: 30s - loss: 0.2234 - categorical_accuracy: 0.9308
43136/60000 [====================>.........] - ETA: 30s - loss: 0.2233 - categorical_accuracy: 0.9308
43168/60000 [====================>.........] - ETA: 30s - loss: 0.2232 - categorical_accuracy: 0.9309
43200/60000 [====================>.........] - ETA: 30s - loss: 0.2231 - categorical_accuracy: 0.9309
43232/60000 [====================>.........] - ETA: 30s - loss: 0.2230 - categorical_accuracy: 0.9309
43264/60000 [====================>.........] - ETA: 30s - loss: 0.2231 - categorical_accuracy: 0.9309
43296/60000 [====================>.........] - ETA: 30s - loss: 0.2231 - categorical_accuracy: 0.9309
43328/60000 [====================>.........] - ETA: 30s - loss: 0.2230 - categorical_accuracy: 0.9309
43360/60000 [====================>.........] - ETA: 30s - loss: 0.2229 - categorical_accuracy: 0.9310
43392/60000 [====================>.........] - ETA: 30s - loss: 0.2228 - categorical_accuracy: 0.9310
43424/60000 [====================>.........] - ETA: 30s - loss: 0.2228 - categorical_accuracy: 0.9310
43456/60000 [====================>.........] - ETA: 30s - loss: 0.2229 - categorical_accuracy: 0.9310
43488/60000 [====================>.........] - ETA: 29s - loss: 0.2228 - categorical_accuracy: 0.9310
43520/60000 [====================>.........] - ETA: 29s - loss: 0.2227 - categorical_accuracy: 0.9310
43552/60000 [====================>.........] - ETA: 29s - loss: 0.2226 - categorical_accuracy: 0.9310
43584/60000 [====================>.........] - ETA: 29s - loss: 0.2225 - categorical_accuracy: 0.9311
43616/60000 [====================>.........] - ETA: 29s - loss: 0.2225 - categorical_accuracy: 0.9311
43648/60000 [====================>.........] - ETA: 29s - loss: 0.2224 - categorical_accuracy: 0.9311
43680/60000 [====================>.........] - ETA: 29s - loss: 0.2223 - categorical_accuracy: 0.9311
43712/60000 [====================>.........] - ETA: 29s - loss: 0.2223 - categorical_accuracy: 0.9311
43744/60000 [====================>.........] - ETA: 29s - loss: 0.2222 - categorical_accuracy: 0.9311
43776/60000 [====================>.........] - ETA: 29s - loss: 0.2221 - categorical_accuracy: 0.9312
43808/60000 [====================>.........] - ETA: 29s - loss: 0.2220 - categorical_accuracy: 0.9312
43840/60000 [====================>.........] - ETA: 29s - loss: 0.2219 - categorical_accuracy: 0.9312
43872/60000 [====================>.........] - ETA: 29s - loss: 0.2219 - categorical_accuracy: 0.9313
43904/60000 [====================>.........] - ETA: 29s - loss: 0.2218 - categorical_accuracy: 0.9313
43936/60000 [====================>.........] - ETA: 29s - loss: 0.2217 - categorical_accuracy: 0.9313
43968/60000 [====================>.........] - ETA: 29s - loss: 0.2215 - categorical_accuracy: 0.9314
44000/60000 [=====================>........] - ETA: 29s - loss: 0.2214 - categorical_accuracy: 0.9314
44032/60000 [=====================>........] - ETA: 28s - loss: 0.2213 - categorical_accuracy: 0.9315
44064/60000 [=====================>........] - ETA: 28s - loss: 0.2213 - categorical_accuracy: 0.9315
44096/60000 [=====================>........] - ETA: 28s - loss: 0.2212 - categorical_accuracy: 0.9315
44128/60000 [=====================>........] - ETA: 28s - loss: 0.2212 - categorical_accuracy: 0.9315
44160/60000 [=====================>........] - ETA: 28s - loss: 0.2211 - categorical_accuracy: 0.9315
44192/60000 [=====================>........] - ETA: 28s - loss: 0.2210 - categorical_accuracy: 0.9315
44224/60000 [=====================>........] - ETA: 28s - loss: 0.2209 - categorical_accuracy: 0.9315
44256/60000 [=====================>........] - ETA: 28s - loss: 0.2208 - categorical_accuracy: 0.9315
44288/60000 [=====================>........] - ETA: 28s - loss: 0.2210 - categorical_accuracy: 0.9315
44320/60000 [=====================>........] - ETA: 28s - loss: 0.2210 - categorical_accuracy: 0.9315
44352/60000 [=====================>........] - ETA: 28s - loss: 0.2210 - categorical_accuracy: 0.9315
44384/60000 [=====================>........] - ETA: 28s - loss: 0.2208 - categorical_accuracy: 0.9316
44416/60000 [=====================>........] - ETA: 28s - loss: 0.2208 - categorical_accuracy: 0.9316
44448/60000 [=====================>........] - ETA: 28s - loss: 0.2207 - categorical_accuracy: 0.9316
44480/60000 [=====================>........] - ETA: 28s - loss: 0.2206 - categorical_accuracy: 0.9317
44512/60000 [=====================>........] - ETA: 28s - loss: 0.2205 - categorical_accuracy: 0.9317
44544/60000 [=====================>........] - ETA: 28s - loss: 0.2205 - categorical_accuracy: 0.9317
44576/60000 [=====================>........] - ETA: 28s - loss: 0.2204 - categorical_accuracy: 0.9317
44608/60000 [=====================>........] - ETA: 27s - loss: 0.2202 - categorical_accuracy: 0.9318
44640/60000 [=====================>........] - ETA: 27s - loss: 0.2202 - categorical_accuracy: 0.9318
44672/60000 [=====================>........] - ETA: 27s - loss: 0.2201 - categorical_accuracy: 0.9318
44704/60000 [=====================>........] - ETA: 27s - loss: 0.2200 - categorical_accuracy: 0.9318
44736/60000 [=====================>........] - ETA: 27s - loss: 0.2200 - categorical_accuracy: 0.9318
44768/60000 [=====================>........] - ETA: 27s - loss: 0.2199 - categorical_accuracy: 0.9319
44800/60000 [=====================>........] - ETA: 27s - loss: 0.2199 - categorical_accuracy: 0.9319
44832/60000 [=====================>........] - ETA: 27s - loss: 0.2198 - categorical_accuracy: 0.9319
44864/60000 [=====================>........] - ETA: 27s - loss: 0.2197 - categorical_accuracy: 0.9320
44896/60000 [=====================>........] - ETA: 27s - loss: 0.2196 - categorical_accuracy: 0.9320
44928/60000 [=====================>........] - ETA: 27s - loss: 0.2195 - categorical_accuracy: 0.9320
44960/60000 [=====================>........] - ETA: 27s - loss: 0.2194 - categorical_accuracy: 0.9321
44992/60000 [=====================>........] - ETA: 27s - loss: 0.2195 - categorical_accuracy: 0.9321
45024/60000 [=====================>........] - ETA: 27s - loss: 0.2194 - categorical_accuracy: 0.9321
45056/60000 [=====================>........] - ETA: 27s - loss: 0.2193 - categorical_accuracy: 0.9321
45088/60000 [=====================>........] - ETA: 27s - loss: 0.2192 - categorical_accuracy: 0.9321
45120/60000 [=====================>........] - ETA: 27s - loss: 0.2191 - categorical_accuracy: 0.9322
45152/60000 [=====================>........] - ETA: 26s - loss: 0.2191 - categorical_accuracy: 0.9322
45184/60000 [=====================>........] - ETA: 26s - loss: 0.2191 - categorical_accuracy: 0.9322
45216/60000 [=====================>........] - ETA: 26s - loss: 0.2191 - categorical_accuracy: 0.9322
45248/60000 [=====================>........] - ETA: 26s - loss: 0.2191 - categorical_accuracy: 0.9322
45280/60000 [=====================>........] - ETA: 26s - loss: 0.2191 - categorical_accuracy: 0.9322
45312/60000 [=====================>........] - ETA: 26s - loss: 0.2190 - categorical_accuracy: 0.9322
45344/60000 [=====================>........] - ETA: 26s - loss: 0.2191 - categorical_accuracy: 0.9322
45376/60000 [=====================>........] - ETA: 26s - loss: 0.2190 - categorical_accuracy: 0.9323
45408/60000 [=====================>........] - ETA: 26s - loss: 0.2189 - categorical_accuracy: 0.9323
45440/60000 [=====================>........] - ETA: 26s - loss: 0.2188 - categorical_accuracy: 0.9323
45472/60000 [=====================>........] - ETA: 26s - loss: 0.2187 - categorical_accuracy: 0.9323
45504/60000 [=====================>........] - ETA: 26s - loss: 0.2186 - categorical_accuracy: 0.9324
45536/60000 [=====================>........] - ETA: 26s - loss: 0.2185 - categorical_accuracy: 0.9324
45568/60000 [=====================>........] - ETA: 26s - loss: 0.2184 - categorical_accuracy: 0.9324
45600/60000 [=====================>........] - ETA: 26s - loss: 0.2184 - categorical_accuracy: 0.9324
45632/60000 [=====================>........] - ETA: 26s - loss: 0.2183 - categorical_accuracy: 0.9324
45664/60000 [=====================>........] - ETA: 26s - loss: 0.2182 - categorical_accuracy: 0.9325
45696/60000 [=====================>........] - ETA: 25s - loss: 0.2181 - categorical_accuracy: 0.9325
45728/60000 [=====================>........] - ETA: 25s - loss: 0.2180 - categorical_accuracy: 0.9325
45760/60000 [=====================>........] - ETA: 25s - loss: 0.2180 - categorical_accuracy: 0.9325
45792/60000 [=====================>........] - ETA: 25s - loss: 0.2178 - categorical_accuracy: 0.9325
45824/60000 [=====================>........] - ETA: 25s - loss: 0.2177 - categorical_accuracy: 0.9326
45856/60000 [=====================>........] - ETA: 25s - loss: 0.2176 - categorical_accuracy: 0.9326
45888/60000 [=====================>........] - ETA: 25s - loss: 0.2177 - categorical_accuracy: 0.9326
45920/60000 [=====================>........] - ETA: 25s - loss: 0.2176 - categorical_accuracy: 0.9326
45952/60000 [=====================>........] - ETA: 25s - loss: 0.2175 - categorical_accuracy: 0.9326
45984/60000 [=====================>........] - ETA: 25s - loss: 0.2175 - categorical_accuracy: 0.9326
46016/60000 [======================>.......] - ETA: 25s - loss: 0.2173 - categorical_accuracy: 0.9327
46048/60000 [======================>.......] - ETA: 25s - loss: 0.2172 - categorical_accuracy: 0.9327
46080/60000 [======================>.......] - ETA: 25s - loss: 0.2171 - categorical_accuracy: 0.9327
46112/60000 [======================>.......] - ETA: 25s - loss: 0.2170 - categorical_accuracy: 0.9328
46144/60000 [======================>.......] - ETA: 25s - loss: 0.2169 - categorical_accuracy: 0.9328
46176/60000 [======================>.......] - ETA: 25s - loss: 0.2169 - categorical_accuracy: 0.9328
46208/60000 [======================>.......] - ETA: 25s - loss: 0.2168 - categorical_accuracy: 0.9328
46240/60000 [======================>.......] - ETA: 24s - loss: 0.2171 - categorical_accuracy: 0.9328
46272/60000 [======================>.......] - ETA: 24s - loss: 0.2170 - categorical_accuracy: 0.9328
46304/60000 [======================>.......] - ETA: 24s - loss: 0.2170 - categorical_accuracy: 0.9328
46336/60000 [======================>.......] - ETA: 24s - loss: 0.2168 - categorical_accuracy: 0.9329
46368/60000 [======================>.......] - ETA: 24s - loss: 0.2168 - categorical_accuracy: 0.9329
46400/60000 [======================>.......] - ETA: 24s - loss: 0.2168 - categorical_accuracy: 0.9329
46432/60000 [======================>.......] - ETA: 24s - loss: 0.2167 - categorical_accuracy: 0.9329
46464/60000 [======================>.......] - ETA: 24s - loss: 0.2166 - categorical_accuracy: 0.9329
46496/60000 [======================>.......] - ETA: 24s - loss: 0.2165 - categorical_accuracy: 0.9330
46528/60000 [======================>.......] - ETA: 24s - loss: 0.2164 - categorical_accuracy: 0.9330
46560/60000 [======================>.......] - ETA: 24s - loss: 0.2163 - categorical_accuracy: 0.9330
46592/60000 [======================>.......] - ETA: 24s - loss: 0.2163 - categorical_accuracy: 0.9330
46624/60000 [======================>.......] - ETA: 24s - loss: 0.2162 - categorical_accuracy: 0.9331
46656/60000 [======================>.......] - ETA: 24s - loss: 0.2162 - categorical_accuracy: 0.9331
46688/60000 [======================>.......] - ETA: 24s - loss: 0.2161 - categorical_accuracy: 0.9331
46720/60000 [======================>.......] - ETA: 24s - loss: 0.2160 - categorical_accuracy: 0.9331
46752/60000 [======================>.......] - ETA: 24s - loss: 0.2161 - categorical_accuracy: 0.9331
46784/60000 [======================>.......] - ETA: 23s - loss: 0.2159 - categorical_accuracy: 0.9332
46816/60000 [======================>.......] - ETA: 23s - loss: 0.2158 - categorical_accuracy: 0.9332
46848/60000 [======================>.......] - ETA: 23s - loss: 0.2157 - categorical_accuracy: 0.9333
46880/60000 [======================>.......] - ETA: 23s - loss: 0.2156 - categorical_accuracy: 0.9333
46912/60000 [======================>.......] - ETA: 23s - loss: 0.2155 - categorical_accuracy: 0.9333
46944/60000 [======================>.......] - ETA: 23s - loss: 0.2153 - categorical_accuracy: 0.9334
46976/60000 [======================>.......] - ETA: 23s - loss: 0.2155 - categorical_accuracy: 0.9334
47008/60000 [======================>.......] - ETA: 23s - loss: 0.2154 - categorical_accuracy: 0.9335
47040/60000 [======================>.......] - ETA: 23s - loss: 0.2153 - categorical_accuracy: 0.9335
47072/60000 [======================>.......] - ETA: 23s - loss: 0.2152 - categorical_accuracy: 0.9335
47104/60000 [======================>.......] - ETA: 23s - loss: 0.2150 - categorical_accuracy: 0.9336
47136/60000 [======================>.......] - ETA: 23s - loss: 0.2149 - categorical_accuracy: 0.9336
47168/60000 [======================>.......] - ETA: 23s - loss: 0.2148 - categorical_accuracy: 0.9337
47200/60000 [======================>.......] - ETA: 23s - loss: 0.2150 - categorical_accuracy: 0.9337
47232/60000 [======================>.......] - ETA: 23s - loss: 0.2149 - categorical_accuracy: 0.9337
47264/60000 [======================>.......] - ETA: 23s - loss: 0.2148 - categorical_accuracy: 0.9337
47296/60000 [======================>.......] - ETA: 23s - loss: 0.2147 - categorical_accuracy: 0.9338
47328/60000 [======================>.......] - ETA: 22s - loss: 0.2146 - categorical_accuracy: 0.9338
47360/60000 [======================>.......] - ETA: 22s - loss: 0.2145 - categorical_accuracy: 0.9338
47392/60000 [======================>.......] - ETA: 22s - loss: 0.2144 - categorical_accuracy: 0.9338
47424/60000 [======================>.......] - ETA: 22s - loss: 0.2143 - categorical_accuracy: 0.9339
47456/60000 [======================>.......] - ETA: 22s - loss: 0.2142 - categorical_accuracy: 0.9339
47488/60000 [======================>.......] - ETA: 22s - loss: 0.2141 - categorical_accuracy: 0.9339
47520/60000 [======================>.......] - ETA: 22s - loss: 0.2140 - categorical_accuracy: 0.9340
47552/60000 [======================>.......] - ETA: 22s - loss: 0.2139 - categorical_accuracy: 0.9340
47584/60000 [======================>.......] - ETA: 22s - loss: 0.2138 - categorical_accuracy: 0.9340
47616/60000 [======================>.......] - ETA: 22s - loss: 0.2137 - categorical_accuracy: 0.9340
47648/60000 [======================>.......] - ETA: 22s - loss: 0.2136 - categorical_accuracy: 0.9341
47680/60000 [======================>.......] - ETA: 22s - loss: 0.2135 - categorical_accuracy: 0.9341
47712/60000 [======================>.......] - ETA: 22s - loss: 0.2134 - categorical_accuracy: 0.9341
47744/60000 [======================>.......] - ETA: 22s - loss: 0.2132 - categorical_accuracy: 0.9342
47776/60000 [======================>.......] - ETA: 22s - loss: 0.2131 - categorical_accuracy: 0.9342
47808/60000 [======================>.......] - ETA: 22s - loss: 0.2130 - categorical_accuracy: 0.9343
47840/60000 [======================>.......] - ETA: 22s - loss: 0.2129 - categorical_accuracy: 0.9343
47872/60000 [======================>.......] - ETA: 22s - loss: 0.2130 - categorical_accuracy: 0.9343
47904/60000 [======================>.......] - ETA: 21s - loss: 0.2130 - categorical_accuracy: 0.9343
47936/60000 [======================>.......] - ETA: 21s - loss: 0.2129 - categorical_accuracy: 0.9343
47968/60000 [======================>.......] - ETA: 21s - loss: 0.2130 - categorical_accuracy: 0.9344
48000/60000 [=======================>......] - ETA: 21s - loss: 0.2129 - categorical_accuracy: 0.9344
48032/60000 [=======================>......] - ETA: 21s - loss: 0.2128 - categorical_accuracy: 0.9344
48064/60000 [=======================>......] - ETA: 21s - loss: 0.2127 - categorical_accuracy: 0.9344
48096/60000 [=======================>......] - ETA: 21s - loss: 0.2127 - categorical_accuracy: 0.9344
48128/60000 [=======================>......] - ETA: 21s - loss: 0.2125 - categorical_accuracy: 0.9345
48160/60000 [=======================>......] - ETA: 21s - loss: 0.2124 - categorical_accuracy: 0.9345
48192/60000 [=======================>......] - ETA: 21s - loss: 0.2124 - categorical_accuracy: 0.9345
48224/60000 [=======================>......] - ETA: 21s - loss: 0.2123 - categorical_accuracy: 0.9345
48256/60000 [=======================>......] - ETA: 21s - loss: 0.2123 - categorical_accuracy: 0.9345
48288/60000 [=======================>......] - ETA: 21s - loss: 0.2122 - categorical_accuracy: 0.9346
48320/60000 [=======================>......] - ETA: 21s - loss: 0.2121 - categorical_accuracy: 0.9346
48352/60000 [=======================>......] - ETA: 21s - loss: 0.2120 - categorical_accuracy: 0.9346
48384/60000 [=======================>......] - ETA: 21s - loss: 0.2122 - categorical_accuracy: 0.9346
48416/60000 [=======================>......] - ETA: 21s - loss: 0.2121 - categorical_accuracy: 0.9346
48448/60000 [=======================>......] - ETA: 20s - loss: 0.2120 - categorical_accuracy: 0.9347
48480/60000 [=======================>......] - ETA: 20s - loss: 0.2120 - categorical_accuracy: 0.9347
48512/60000 [=======================>......] - ETA: 20s - loss: 0.2121 - categorical_accuracy: 0.9346
48544/60000 [=======================>......] - ETA: 20s - loss: 0.2120 - categorical_accuracy: 0.9346
48576/60000 [=======================>......] - ETA: 20s - loss: 0.2119 - categorical_accuracy: 0.9347
48608/60000 [=======================>......] - ETA: 20s - loss: 0.2118 - categorical_accuracy: 0.9347
48640/60000 [=======================>......] - ETA: 20s - loss: 0.2117 - categorical_accuracy: 0.9347
48672/60000 [=======================>......] - ETA: 20s - loss: 0.2115 - categorical_accuracy: 0.9348
48704/60000 [=======================>......] - ETA: 20s - loss: 0.2114 - categorical_accuracy: 0.9348
48736/60000 [=======================>......] - ETA: 20s - loss: 0.2113 - categorical_accuracy: 0.9349
48768/60000 [=======================>......] - ETA: 20s - loss: 0.2112 - categorical_accuracy: 0.9349
48800/60000 [=======================>......] - ETA: 20s - loss: 0.2111 - categorical_accuracy: 0.9349
48832/60000 [=======================>......] - ETA: 20s - loss: 0.2110 - categorical_accuracy: 0.9350
48864/60000 [=======================>......] - ETA: 20s - loss: 0.2109 - categorical_accuracy: 0.9350
48896/60000 [=======================>......] - ETA: 20s - loss: 0.2109 - categorical_accuracy: 0.9350
48928/60000 [=======================>......] - ETA: 20s - loss: 0.2108 - categorical_accuracy: 0.9350
48960/60000 [=======================>......] - ETA: 20s - loss: 0.2107 - categorical_accuracy: 0.9350
48992/60000 [=======================>......] - ETA: 19s - loss: 0.2110 - categorical_accuracy: 0.9350
49024/60000 [=======================>......] - ETA: 19s - loss: 0.2109 - categorical_accuracy: 0.9350
49056/60000 [=======================>......] - ETA: 19s - loss: 0.2109 - categorical_accuracy: 0.9350
49088/60000 [=======================>......] - ETA: 19s - loss: 0.2108 - categorical_accuracy: 0.9350
49120/60000 [=======================>......] - ETA: 19s - loss: 0.2107 - categorical_accuracy: 0.9350
49152/60000 [=======================>......] - ETA: 19s - loss: 0.2106 - categorical_accuracy: 0.9350
49184/60000 [=======================>......] - ETA: 19s - loss: 0.2105 - categorical_accuracy: 0.9350
49216/60000 [=======================>......] - ETA: 19s - loss: 0.2105 - categorical_accuracy: 0.9350
49248/60000 [=======================>......] - ETA: 19s - loss: 0.2105 - categorical_accuracy: 0.9351
49280/60000 [=======================>......] - ETA: 19s - loss: 0.2104 - categorical_accuracy: 0.9351
49312/60000 [=======================>......] - ETA: 19s - loss: 0.2103 - categorical_accuracy: 0.9351
49344/60000 [=======================>......] - ETA: 19s - loss: 0.2102 - categorical_accuracy: 0.9351
49376/60000 [=======================>......] - ETA: 19s - loss: 0.2103 - categorical_accuracy: 0.9352
49408/60000 [=======================>......] - ETA: 19s - loss: 0.2102 - categorical_accuracy: 0.9352
49440/60000 [=======================>......] - ETA: 19s - loss: 0.2101 - categorical_accuracy: 0.9352
49472/60000 [=======================>......] - ETA: 19s - loss: 0.2102 - categorical_accuracy: 0.9352
49504/60000 [=======================>......] - ETA: 19s - loss: 0.2101 - categorical_accuracy: 0.9352
49536/60000 [=======================>......] - ETA: 18s - loss: 0.2101 - categorical_accuracy: 0.9352
49568/60000 [=======================>......] - ETA: 18s - loss: 0.2100 - categorical_accuracy: 0.9352
49600/60000 [=======================>......] - ETA: 18s - loss: 0.2098 - categorical_accuracy: 0.9353
49632/60000 [=======================>......] - ETA: 18s - loss: 0.2097 - categorical_accuracy: 0.9353
49664/60000 [=======================>......] - ETA: 18s - loss: 0.2097 - categorical_accuracy: 0.9353
49696/60000 [=======================>......] - ETA: 18s - loss: 0.2097 - categorical_accuracy: 0.9353
49728/60000 [=======================>......] - ETA: 18s - loss: 0.2096 - categorical_accuracy: 0.9353
49760/60000 [=======================>......] - ETA: 18s - loss: 0.2095 - categorical_accuracy: 0.9354
49792/60000 [=======================>......] - ETA: 18s - loss: 0.2094 - categorical_accuracy: 0.9354
49824/60000 [=======================>......] - ETA: 18s - loss: 0.2094 - categorical_accuracy: 0.9354
49856/60000 [=======================>......] - ETA: 18s - loss: 0.2094 - categorical_accuracy: 0.9354
49888/60000 [=======================>......] - ETA: 18s - loss: 0.2093 - categorical_accuracy: 0.9354
49920/60000 [=======================>......] - ETA: 18s - loss: 0.2093 - categorical_accuracy: 0.9355
49952/60000 [=======================>......] - ETA: 18s - loss: 0.2094 - categorical_accuracy: 0.9354
49984/60000 [=======================>......] - ETA: 18s - loss: 0.2092 - categorical_accuracy: 0.9355
50016/60000 [========================>.....] - ETA: 18s - loss: 0.2091 - categorical_accuracy: 0.9355
50048/60000 [========================>.....] - ETA: 18s - loss: 0.2091 - categorical_accuracy: 0.9355
50080/60000 [========================>.....] - ETA: 17s - loss: 0.2091 - categorical_accuracy: 0.9355
50112/60000 [========================>.....] - ETA: 17s - loss: 0.2090 - categorical_accuracy: 0.9355
50144/60000 [========================>.....] - ETA: 17s - loss: 0.2091 - categorical_accuracy: 0.9355
50176/60000 [========================>.....] - ETA: 17s - loss: 0.2091 - categorical_accuracy: 0.9355
50208/60000 [========================>.....] - ETA: 17s - loss: 0.2090 - categorical_accuracy: 0.9356
50240/60000 [========================>.....] - ETA: 17s - loss: 0.2088 - categorical_accuracy: 0.9356
50272/60000 [========================>.....] - ETA: 17s - loss: 0.2088 - categorical_accuracy: 0.9357
50304/60000 [========================>.....] - ETA: 17s - loss: 0.2087 - categorical_accuracy: 0.9357
50336/60000 [========================>.....] - ETA: 17s - loss: 0.2086 - categorical_accuracy: 0.9357
50368/60000 [========================>.....] - ETA: 17s - loss: 0.2086 - categorical_accuracy: 0.9357
50400/60000 [========================>.....] - ETA: 17s - loss: 0.2085 - categorical_accuracy: 0.9358
50432/60000 [========================>.....] - ETA: 17s - loss: 0.2085 - categorical_accuracy: 0.9358
50464/60000 [========================>.....] - ETA: 17s - loss: 0.2084 - categorical_accuracy: 0.9358
50496/60000 [========================>.....] - ETA: 17s - loss: 0.2084 - categorical_accuracy: 0.9358
50528/60000 [========================>.....] - ETA: 17s - loss: 0.2083 - categorical_accuracy: 0.9358
50560/60000 [========================>.....] - ETA: 17s - loss: 0.2082 - categorical_accuracy: 0.9359
50592/60000 [========================>.....] - ETA: 17s - loss: 0.2081 - categorical_accuracy: 0.9359
50624/60000 [========================>.....] - ETA: 16s - loss: 0.2081 - categorical_accuracy: 0.9359
50656/60000 [========================>.....] - ETA: 16s - loss: 0.2079 - categorical_accuracy: 0.9360
50688/60000 [========================>.....] - ETA: 16s - loss: 0.2078 - categorical_accuracy: 0.9360
50720/60000 [========================>.....] - ETA: 16s - loss: 0.2079 - categorical_accuracy: 0.9360
50752/60000 [========================>.....] - ETA: 16s - loss: 0.2078 - categorical_accuracy: 0.9360
50784/60000 [========================>.....] - ETA: 16s - loss: 0.2077 - categorical_accuracy: 0.9360
50816/60000 [========================>.....] - ETA: 16s - loss: 0.2077 - categorical_accuracy: 0.9360
50848/60000 [========================>.....] - ETA: 16s - loss: 0.2075 - categorical_accuracy: 0.9361
50880/60000 [========================>.....] - ETA: 16s - loss: 0.2074 - categorical_accuracy: 0.9361
50912/60000 [========================>.....] - ETA: 16s - loss: 0.2073 - categorical_accuracy: 0.9361
50944/60000 [========================>.....] - ETA: 16s - loss: 0.2073 - categorical_accuracy: 0.9361
50976/60000 [========================>.....] - ETA: 16s - loss: 0.2071 - categorical_accuracy: 0.9362
51008/60000 [========================>.....] - ETA: 16s - loss: 0.2071 - categorical_accuracy: 0.9362
51040/60000 [========================>.....] - ETA: 16s - loss: 0.2070 - categorical_accuracy: 0.9362
51072/60000 [========================>.....] - ETA: 16s - loss: 0.2069 - categorical_accuracy: 0.9363
51104/60000 [========================>.....] - ETA: 16s - loss: 0.2068 - categorical_accuracy: 0.9363
51136/60000 [========================>.....] - ETA: 16s - loss: 0.2067 - categorical_accuracy: 0.9363
51168/60000 [========================>.....] - ETA: 16s - loss: 0.2066 - categorical_accuracy: 0.9364
51200/60000 [========================>.....] - ETA: 15s - loss: 0.2065 - categorical_accuracy: 0.9364
51232/60000 [========================>.....] - ETA: 15s - loss: 0.2064 - categorical_accuracy: 0.9365
51264/60000 [========================>.....] - ETA: 15s - loss: 0.2063 - categorical_accuracy: 0.9365
51296/60000 [========================>.....] - ETA: 15s - loss: 0.2065 - categorical_accuracy: 0.9365
51328/60000 [========================>.....] - ETA: 15s - loss: 0.2064 - categorical_accuracy: 0.9365
51360/60000 [========================>.....] - ETA: 15s - loss: 0.2067 - categorical_accuracy: 0.9364
51392/60000 [========================>.....] - ETA: 15s - loss: 0.2066 - categorical_accuracy: 0.9365
51424/60000 [========================>.....] - ETA: 15s - loss: 0.2065 - categorical_accuracy: 0.9365
51456/60000 [========================>.....] - ETA: 15s - loss: 0.2064 - categorical_accuracy: 0.9365
51488/60000 [========================>.....] - ETA: 15s - loss: 0.2065 - categorical_accuracy: 0.9365
51520/60000 [========================>.....] - ETA: 15s - loss: 0.2064 - categorical_accuracy: 0.9366
51552/60000 [========================>.....] - ETA: 15s - loss: 0.2063 - categorical_accuracy: 0.9366
51584/60000 [========================>.....] - ETA: 15s - loss: 0.2062 - categorical_accuracy: 0.9366
51616/60000 [========================>.....] - ETA: 15s - loss: 0.2062 - categorical_accuracy: 0.9366
51648/60000 [========================>.....] - ETA: 15s - loss: 0.2061 - categorical_accuracy: 0.9366
51680/60000 [========================>.....] - ETA: 15s - loss: 0.2060 - categorical_accuracy: 0.9366
51712/60000 [========================>.....] - ETA: 15s - loss: 0.2059 - categorical_accuracy: 0.9367
51744/60000 [========================>.....] - ETA: 14s - loss: 0.2058 - categorical_accuracy: 0.9367
51776/60000 [========================>.....] - ETA: 14s - loss: 0.2057 - categorical_accuracy: 0.9367
51808/60000 [========================>.....] - ETA: 14s - loss: 0.2057 - categorical_accuracy: 0.9368
51840/60000 [========================>.....] - ETA: 14s - loss: 0.2056 - categorical_accuracy: 0.9368
51872/60000 [========================>.....] - ETA: 14s - loss: 0.2055 - categorical_accuracy: 0.9368
51904/60000 [========================>.....] - ETA: 14s - loss: 0.2054 - categorical_accuracy: 0.9368
51936/60000 [========================>.....] - ETA: 14s - loss: 0.2054 - categorical_accuracy: 0.9368
51968/60000 [========================>.....] - ETA: 14s - loss: 0.2053 - categorical_accuracy: 0.9369
52000/60000 [=========================>....] - ETA: 14s - loss: 0.2052 - categorical_accuracy: 0.9369
52032/60000 [=========================>....] - ETA: 14s - loss: 0.2051 - categorical_accuracy: 0.9369
52064/60000 [=========================>....] - ETA: 14s - loss: 0.2051 - categorical_accuracy: 0.9369
52096/60000 [=========================>....] - ETA: 14s - loss: 0.2050 - categorical_accuracy: 0.9370
52128/60000 [=========================>....] - ETA: 14s - loss: 0.2049 - categorical_accuracy: 0.9370
52160/60000 [=========================>....] - ETA: 14s - loss: 0.2049 - categorical_accuracy: 0.9370
52192/60000 [=========================>....] - ETA: 14s - loss: 0.2049 - categorical_accuracy: 0.9370
52224/60000 [=========================>....] - ETA: 14s - loss: 0.2047 - categorical_accuracy: 0.9371
52256/60000 [=========================>....] - ETA: 14s - loss: 0.2047 - categorical_accuracy: 0.9371
52288/60000 [=========================>....] - ETA: 13s - loss: 0.2046 - categorical_accuracy: 0.9371
52320/60000 [=========================>....] - ETA: 13s - loss: 0.2045 - categorical_accuracy: 0.9372
52352/60000 [=========================>....] - ETA: 13s - loss: 0.2044 - categorical_accuracy: 0.9372
52384/60000 [=========================>....] - ETA: 13s - loss: 0.2043 - categorical_accuracy: 0.9372
52416/60000 [=========================>....] - ETA: 13s - loss: 0.2042 - categorical_accuracy: 0.9372
52448/60000 [=========================>....] - ETA: 13s - loss: 0.2041 - categorical_accuracy: 0.9373
52480/60000 [=========================>....] - ETA: 13s - loss: 0.2041 - categorical_accuracy: 0.9373
52512/60000 [=========================>....] - ETA: 13s - loss: 0.2042 - categorical_accuracy: 0.9373
52544/60000 [=========================>....] - ETA: 13s - loss: 0.2041 - categorical_accuracy: 0.9373
52576/60000 [=========================>....] - ETA: 13s - loss: 0.2040 - categorical_accuracy: 0.9373
52608/60000 [=========================>....] - ETA: 13s - loss: 0.2039 - categorical_accuracy: 0.9373
52640/60000 [=========================>....] - ETA: 13s - loss: 0.2040 - categorical_accuracy: 0.9373
52672/60000 [=========================>....] - ETA: 13s - loss: 0.2040 - categorical_accuracy: 0.9373
52704/60000 [=========================>....] - ETA: 13s - loss: 0.2039 - categorical_accuracy: 0.9373
52736/60000 [=========================>....] - ETA: 13s - loss: 0.2038 - categorical_accuracy: 0.9374
52768/60000 [=========================>....] - ETA: 13s - loss: 0.2037 - categorical_accuracy: 0.9374
52800/60000 [=========================>....] - ETA: 13s - loss: 0.2036 - categorical_accuracy: 0.9374
52832/60000 [=========================>....] - ETA: 13s - loss: 0.2036 - categorical_accuracy: 0.9375
52864/60000 [=========================>....] - ETA: 12s - loss: 0.2035 - categorical_accuracy: 0.9375
52896/60000 [=========================>....] - ETA: 12s - loss: 0.2034 - categorical_accuracy: 0.9375
52928/60000 [=========================>....] - ETA: 12s - loss: 0.2034 - categorical_accuracy: 0.9375
52960/60000 [=========================>....] - ETA: 12s - loss: 0.2033 - categorical_accuracy: 0.9376
52992/60000 [=========================>....] - ETA: 12s - loss: 0.2031 - categorical_accuracy: 0.9376
53024/60000 [=========================>....] - ETA: 12s - loss: 0.2030 - categorical_accuracy: 0.9376
53056/60000 [=========================>....] - ETA: 12s - loss: 0.2030 - categorical_accuracy: 0.9376
53088/60000 [=========================>....] - ETA: 12s - loss: 0.2029 - categorical_accuracy: 0.9377
53120/60000 [=========================>....] - ETA: 12s - loss: 0.2028 - categorical_accuracy: 0.9377
53152/60000 [=========================>....] - ETA: 12s - loss: 0.2027 - categorical_accuracy: 0.9377
53184/60000 [=========================>....] - ETA: 12s - loss: 0.2026 - categorical_accuracy: 0.9377
53216/60000 [=========================>....] - ETA: 12s - loss: 0.2026 - categorical_accuracy: 0.9377
53248/60000 [=========================>....] - ETA: 12s - loss: 0.2025 - categorical_accuracy: 0.9377
53280/60000 [=========================>....] - ETA: 12s - loss: 0.2026 - categorical_accuracy: 0.9378
53312/60000 [=========================>....] - ETA: 12s - loss: 0.2025 - categorical_accuracy: 0.9378
53344/60000 [=========================>....] - ETA: 12s - loss: 0.2024 - categorical_accuracy: 0.9378
53376/60000 [=========================>....] - ETA: 12s - loss: 0.2023 - categorical_accuracy: 0.9379
53408/60000 [=========================>....] - ETA: 11s - loss: 0.2023 - categorical_accuracy: 0.9378
53440/60000 [=========================>....] - ETA: 11s - loss: 0.2023 - categorical_accuracy: 0.9378
53472/60000 [=========================>....] - ETA: 11s - loss: 0.2022 - categorical_accuracy: 0.9378
53504/60000 [=========================>....] - ETA: 11s - loss: 0.2022 - categorical_accuracy: 0.9378
53536/60000 [=========================>....] - ETA: 11s - loss: 0.2021 - categorical_accuracy: 0.9378
53568/60000 [=========================>....] - ETA: 11s - loss: 0.2021 - categorical_accuracy: 0.9379
53600/60000 [=========================>....] - ETA: 11s - loss: 0.2020 - categorical_accuracy: 0.9379
53632/60000 [=========================>....] - ETA: 11s - loss: 0.2019 - categorical_accuracy: 0.9379
53664/60000 [=========================>....] - ETA: 11s - loss: 0.2018 - categorical_accuracy: 0.9379
53696/60000 [=========================>....] - ETA: 11s - loss: 0.2017 - categorical_accuracy: 0.9380
53728/60000 [=========================>....] - ETA: 11s - loss: 0.2016 - categorical_accuracy: 0.9380
53760/60000 [=========================>....] - ETA: 11s - loss: 0.2015 - categorical_accuracy: 0.9380
53792/60000 [=========================>....] - ETA: 11s - loss: 0.2014 - categorical_accuracy: 0.9381
53824/60000 [=========================>....] - ETA: 11s - loss: 0.2014 - categorical_accuracy: 0.9381
53856/60000 [=========================>....] - ETA: 11s - loss: 0.2013 - categorical_accuracy: 0.9381
53888/60000 [=========================>....] - ETA: 11s - loss: 0.2012 - categorical_accuracy: 0.9381
53920/60000 [=========================>....] - ETA: 11s - loss: 0.2011 - categorical_accuracy: 0.9382
53952/60000 [=========================>....] - ETA: 10s - loss: 0.2010 - categorical_accuracy: 0.9382
53984/60000 [=========================>....] - ETA: 10s - loss: 0.2009 - categorical_accuracy: 0.9382
54016/60000 [==========================>...] - ETA: 10s - loss: 0.2008 - categorical_accuracy: 0.9383
54048/60000 [==========================>...] - ETA: 10s - loss: 0.2008 - categorical_accuracy: 0.9383
54080/60000 [==========================>...] - ETA: 10s - loss: 0.2007 - categorical_accuracy: 0.9383
54112/60000 [==========================>...] - ETA: 10s - loss: 0.2007 - categorical_accuracy: 0.9383
54144/60000 [==========================>...] - ETA: 10s - loss: 0.2006 - categorical_accuracy: 0.9383
54176/60000 [==========================>...] - ETA: 10s - loss: 0.2005 - categorical_accuracy: 0.9384
54208/60000 [==========================>...] - ETA: 10s - loss: 0.2004 - categorical_accuracy: 0.9384
54240/60000 [==========================>...] - ETA: 10s - loss: 0.2003 - categorical_accuracy: 0.9384
54272/60000 [==========================>...] - ETA: 10s - loss: 0.2002 - categorical_accuracy: 0.9384
54304/60000 [==========================>...] - ETA: 10s - loss: 0.2001 - categorical_accuracy: 0.9385
54336/60000 [==========================>...] - ETA: 10s - loss: 0.2000 - categorical_accuracy: 0.9385
54368/60000 [==========================>...] - ETA: 10s - loss: 0.1999 - categorical_accuracy: 0.9385
54400/60000 [==========================>...] - ETA: 10s - loss: 0.1998 - categorical_accuracy: 0.9386
54432/60000 [==========================>...] - ETA: 10s - loss: 0.1999 - categorical_accuracy: 0.9386
54464/60000 [==========================>...] - ETA: 10s - loss: 0.1998 - categorical_accuracy: 0.9386
54496/60000 [==========================>...] - ETA: 9s - loss: 0.1997 - categorical_accuracy: 0.9387 
54528/60000 [==========================>...] - ETA: 9s - loss: 0.1996 - categorical_accuracy: 0.9387
54560/60000 [==========================>...] - ETA: 9s - loss: 0.1995 - categorical_accuracy: 0.9387
54592/60000 [==========================>...] - ETA: 9s - loss: 0.1994 - categorical_accuracy: 0.9388
54624/60000 [==========================>...] - ETA: 9s - loss: 0.1993 - categorical_accuracy: 0.9388
54656/60000 [==========================>...] - ETA: 9s - loss: 0.1993 - categorical_accuracy: 0.9388
54688/60000 [==========================>...] - ETA: 9s - loss: 0.1992 - categorical_accuracy: 0.9388
54720/60000 [==========================>...] - ETA: 9s - loss: 0.1993 - categorical_accuracy: 0.9388
54752/60000 [==========================>...] - ETA: 9s - loss: 0.1992 - categorical_accuracy: 0.9389
54784/60000 [==========================>...] - ETA: 9s - loss: 0.1991 - categorical_accuracy: 0.9389
54816/60000 [==========================>...] - ETA: 9s - loss: 0.1991 - categorical_accuracy: 0.9389
54848/60000 [==========================>...] - ETA: 9s - loss: 0.1990 - categorical_accuracy: 0.9389
54880/60000 [==========================>...] - ETA: 9s - loss: 0.1990 - categorical_accuracy: 0.9389
54912/60000 [==========================>...] - ETA: 9s - loss: 0.1991 - categorical_accuracy: 0.9389
54944/60000 [==========================>...] - ETA: 9s - loss: 0.1990 - categorical_accuracy: 0.9389
54976/60000 [==========================>...] - ETA: 9s - loss: 0.1989 - categorical_accuracy: 0.9389
55008/60000 [==========================>...] - ETA: 9s - loss: 0.1990 - categorical_accuracy: 0.9389
55040/60000 [==========================>...] - ETA: 8s - loss: 0.1990 - categorical_accuracy: 0.9389
55072/60000 [==========================>...] - ETA: 8s - loss: 0.1989 - categorical_accuracy: 0.9390
55104/60000 [==========================>...] - ETA: 8s - loss: 0.1989 - categorical_accuracy: 0.9390
55136/60000 [==========================>...] - ETA: 8s - loss: 0.1988 - categorical_accuracy: 0.9390
55168/60000 [==========================>...] - ETA: 8s - loss: 0.1987 - categorical_accuracy: 0.9390
55200/60000 [==========================>...] - ETA: 8s - loss: 0.1986 - categorical_accuracy: 0.9391
55232/60000 [==========================>...] - ETA: 8s - loss: 0.1985 - categorical_accuracy: 0.9391
55264/60000 [==========================>...] - ETA: 8s - loss: 0.1984 - categorical_accuracy: 0.9391
55296/60000 [==========================>...] - ETA: 8s - loss: 0.1983 - categorical_accuracy: 0.9391
55328/60000 [==========================>...] - ETA: 8s - loss: 0.1982 - categorical_accuracy: 0.9392
55360/60000 [==========================>...] - ETA: 8s - loss: 0.1981 - categorical_accuracy: 0.9392
55392/60000 [==========================>...] - ETA: 8s - loss: 0.1981 - categorical_accuracy: 0.9392
55424/60000 [==========================>...] - ETA: 8s - loss: 0.1981 - categorical_accuracy: 0.9392
55456/60000 [==========================>...] - ETA: 8s - loss: 0.1980 - categorical_accuracy: 0.9392
55488/60000 [==========================>...] - ETA: 8s - loss: 0.1980 - categorical_accuracy: 0.9392
55520/60000 [==========================>...] - ETA: 8s - loss: 0.1979 - categorical_accuracy: 0.9392
55552/60000 [==========================>...] - ETA: 8s - loss: 0.1978 - categorical_accuracy: 0.9393
55584/60000 [==========================>...] - ETA: 8s - loss: 0.1978 - categorical_accuracy: 0.9393
55616/60000 [==========================>...] - ETA: 7s - loss: 0.1977 - categorical_accuracy: 0.9393
55648/60000 [==========================>...] - ETA: 7s - loss: 0.1976 - categorical_accuracy: 0.9393
55680/60000 [==========================>...] - ETA: 7s - loss: 0.1976 - categorical_accuracy: 0.9394
55712/60000 [==========================>...] - ETA: 7s - loss: 0.1978 - categorical_accuracy: 0.9394
55744/60000 [==========================>...] - ETA: 7s - loss: 0.1978 - categorical_accuracy: 0.9394
55776/60000 [==========================>...] - ETA: 7s - loss: 0.1977 - categorical_accuracy: 0.9394
55808/60000 [==========================>...] - ETA: 7s - loss: 0.1977 - categorical_accuracy: 0.9394
55840/60000 [==========================>...] - ETA: 7s - loss: 0.1976 - categorical_accuracy: 0.9394
55872/60000 [==========================>...] - ETA: 7s - loss: 0.1975 - categorical_accuracy: 0.9395
55904/60000 [==========================>...] - ETA: 7s - loss: 0.1975 - categorical_accuracy: 0.9394
55936/60000 [==========================>...] - ETA: 7s - loss: 0.1974 - categorical_accuracy: 0.9395
55968/60000 [==========================>...] - ETA: 7s - loss: 0.1973 - categorical_accuracy: 0.9395
56000/60000 [===========================>..] - ETA: 7s - loss: 0.1972 - categorical_accuracy: 0.9395
56032/60000 [===========================>..] - ETA: 7s - loss: 0.1971 - categorical_accuracy: 0.9396
56064/60000 [===========================>..] - ETA: 7s - loss: 0.1970 - categorical_accuracy: 0.9396
56096/60000 [===========================>..] - ETA: 7s - loss: 0.1970 - categorical_accuracy: 0.9396
56128/60000 [===========================>..] - ETA: 7s - loss: 0.1969 - categorical_accuracy: 0.9396
56160/60000 [===========================>..] - ETA: 6s - loss: 0.1968 - categorical_accuracy: 0.9396
56192/60000 [===========================>..] - ETA: 6s - loss: 0.1967 - categorical_accuracy: 0.9397
56224/60000 [===========================>..] - ETA: 6s - loss: 0.1967 - categorical_accuracy: 0.9397
56256/60000 [===========================>..] - ETA: 6s - loss: 0.1966 - categorical_accuracy: 0.9397
56288/60000 [===========================>..] - ETA: 6s - loss: 0.1965 - categorical_accuracy: 0.9397
56320/60000 [===========================>..] - ETA: 6s - loss: 0.1964 - categorical_accuracy: 0.9397
56352/60000 [===========================>..] - ETA: 6s - loss: 0.1963 - categorical_accuracy: 0.9398
56384/60000 [===========================>..] - ETA: 6s - loss: 0.1962 - categorical_accuracy: 0.9398
56416/60000 [===========================>..] - ETA: 6s - loss: 0.1962 - categorical_accuracy: 0.9398
56448/60000 [===========================>..] - ETA: 6s - loss: 0.1962 - categorical_accuracy: 0.9398
56480/60000 [===========================>..] - ETA: 6s - loss: 0.1961 - categorical_accuracy: 0.9399
56512/60000 [===========================>..] - ETA: 6s - loss: 0.1960 - categorical_accuracy: 0.9399
56544/60000 [===========================>..] - ETA: 6s - loss: 0.1960 - categorical_accuracy: 0.9399
56576/60000 [===========================>..] - ETA: 6s - loss: 0.1959 - categorical_accuracy: 0.9399
56608/60000 [===========================>..] - ETA: 6s - loss: 0.1958 - categorical_accuracy: 0.9400
56640/60000 [===========================>..] - ETA: 6s - loss: 0.1957 - categorical_accuracy: 0.9400
56672/60000 [===========================>..] - ETA: 6s - loss: 0.1957 - categorical_accuracy: 0.9400
56704/60000 [===========================>..] - ETA: 5s - loss: 0.1956 - categorical_accuracy: 0.9400
56736/60000 [===========================>..] - ETA: 5s - loss: 0.1957 - categorical_accuracy: 0.9400
56768/60000 [===========================>..] - ETA: 5s - loss: 0.1956 - categorical_accuracy: 0.9400
56800/60000 [===========================>..] - ETA: 5s - loss: 0.1956 - categorical_accuracy: 0.9400
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1955 - categorical_accuracy: 0.9400
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1954 - categorical_accuracy: 0.9401
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1954 - categorical_accuracy: 0.9401
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1953 - categorical_accuracy: 0.9401
56960/60000 [===========================>..] - ETA: 5s - loss: 0.1952 - categorical_accuracy: 0.9402
56992/60000 [===========================>..] - ETA: 5s - loss: 0.1951 - categorical_accuracy: 0.9402
57024/60000 [===========================>..] - ETA: 5s - loss: 0.1951 - categorical_accuracy: 0.9402
57056/60000 [===========================>..] - ETA: 5s - loss: 0.1950 - categorical_accuracy: 0.9402
57088/60000 [===========================>..] - ETA: 5s - loss: 0.1949 - categorical_accuracy: 0.9403
57120/60000 [===========================>..] - ETA: 5s - loss: 0.1949 - categorical_accuracy: 0.9402
57152/60000 [===========================>..] - ETA: 5s - loss: 0.1948 - categorical_accuracy: 0.9403
57184/60000 [===========================>..] - ETA: 5s - loss: 0.1947 - categorical_accuracy: 0.9403
57216/60000 [===========================>..] - ETA: 5s - loss: 0.1946 - categorical_accuracy: 0.9403
57248/60000 [===========================>..] - ETA: 4s - loss: 0.1945 - categorical_accuracy: 0.9403
57280/60000 [===========================>..] - ETA: 4s - loss: 0.1945 - categorical_accuracy: 0.9404
57312/60000 [===========================>..] - ETA: 4s - loss: 0.1944 - categorical_accuracy: 0.9404
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1943 - categorical_accuracy: 0.9404
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1942 - categorical_accuracy: 0.9405
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1941 - categorical_accuracy: 0.9405
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1941 - categorical_accuracy: 0.9405
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1941 - categorical_accuracy: 0.9405
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1940 - categorical_accuracy: 0.9405
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1939 - categorical_accuracy: 0.9406
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1938 - categorical_accuracy: 0.9406
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1937 - categorical_accuracy: 0.9406
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1936 - categorical_accuracy: 0.9406
57664/60000 [===========================>..] - ETA: 4s - loss: 0.1935 - categorical_accuracy: 0.9407
57696/60000 [===========================>..] - ETA: 4s - loss: 0.1934 - categorical_accuracy: 0.9407
57728/60000 [===========================>..] - ETA: 4s - loss: 0.1933 - categorical_accuracy: 0.9407
57760/60000 [===========================>..] - ETA: 4s - loss: 0.1933 - categorical_accuracy: 0.9407
57792/60000 [===========================>..] - ETA: 4s - loss: 0.1933 - categorical_accuracy: 0.9407
57824/60000 [===========================>..] - ETA: 3s - loss: 0.1933 - categorical_accuracy: 0.9407
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1933 - categorical_accuracy: 0.9407
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1933 - categorical_accuracy: 0.9407
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1932 - categorical_accuracy: 0.9408
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1932 - categorical_accuracy: 0.9408
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1932 - categorical_accuracy: 0.9408
58016/60000 [============================>.] - ETA: 3s - loss: 0.1931 - categorical_accuracy: 0.9408
58048/60000 [============================>.] - ETA: 3s - loss: 0.1930 - categorical_accuracy: 0.9408
58080/60000 [============================>.] - ETA: 3s - loss: 0.1930 - categorical_accuracy: 0.9408
58112/60000 [============================>.] - ETA: 3s - loss: 0.1929 - categorical_accuracy: 0.9408
58144/60000 [============================>.] - ETA: 3s - loss: 0.1929 - categorical_accuracy: 0.9408
58176/60000 [============================>.] - ETA: 3s - loss: 0.1928 - categorical_accuracy: 0.9409
58208/60000 [============================>.] - ETA: 3s - loss: 0.1927 - categorical_accuracy: 0.9409
58240/60000 [============================>.] - ETA: 3s - loss: 0.1927 - categorical_accuracy: 0.9409
58272/60000 [============================>.] - ETA: 3s - loss: 0.1926 - categorical_accuracy: 0.9409
58304/60000 [============================>.] - ETA: 3s - loss: 0.1926 - categorical_accuracy: 0.9409
58336/60000 [============================>.] - ETA: 3s - loss: 0.1926 - categorical_accuracy: 0.9409
58368/60000 [============================>.] - ETA: 2s - loss: 0.1925 - categorical_accuracy: 0.9409
58400/60000 [============================>.] - ETA: 2s - loss: 0.1924 - categorical_accuracy: 0.9409
58432/60000 [============================>.] - ETA: 2s - loss: 0.1923 - categorical_accuracy: 0.9410
58464/60000 [============================>.] - ETA: 2s - loss: 0.1922 - categorical_accuracy: 0.9410
58496/60000 [============================>.] - ETA: 2s - loss: 0.1922 - categorical_accuracy: 0.9410
58528/60000 [============================>.] - ETA: 2s - loss: 0.1921 - categorical_accuracy: 0.9410
58560/60000 [============================>.] - ETA: 2s - loss: 0.1920 - categorical_accuracy: 0.9411
58592/60000 [============================>.] - ETA: 2s - loss: 0.1920 - categorical_accuracy: 0.9411
58624/60000 [============================>.] - ETA: 2s - loss: 0.1919 - categorical_accuracy: 0.9411
58656/60000 [============================>.] - ETA: 2s - loss: 0.1918 - categorical_accuracy: 0.9411
58688/60000 [============================>.] - ETA: 2s - loss: 0.1918 - categorical_accuracy: 0.9411
58720/60000 [============================>.] - ETA: 2s - loss: 0.1917 - categorical_accuracy: 0.9411
58752/60000 [============================>.] - ETA: 2s - loss: 0.1917 - categorical_accuracy: 0.9411
58784/60000 [============================>.] - ETA: 2s - loss: 0.1916 - categorical_accuracy: 0.9412
58816/60000 [============================>.] - ETA: 2s - loss: 0.1915 - categorical_accuracy: 0.9412
58848/60000 [============================>.] - ETA: 2s - loss: 0.1915 - categorical_accuracy: 0.9412
58880/60000 [============================>.] - ETA: 2s - loss: 0.1915 - categorical_accuracy: 0.9412
58912/60000 [============================>.] - ETA: 1s - loss: 0.1915 - categorical_accuracy: 0.9412
58944/60000 [============================>.] - ETA: 1s - loss: 0.1916 - categorical_accuracy: 0.9412
58976/60000 [============================>.] - ETA: 1s - loss: 0.1915 - categorical_accuracy: 0.9412
59008/60000 [============================>.] - ETA: 1s - loss: 0.1914 - categorical_accuracy: 0.9412
59040/60000 [============================>.] - ETA: 1s - loss: 0.1914 - categorical_accuracy: 0.9413
59072/60000 [============================>.] - ETA: 1s - loss: 0.1913 - categorical_accuracy: 0.9413
59104/60000 [============================>.] - ETA: 1s - loss: 0.1913 - categorical_accuracy: 0.9413
59136/60000 [============================>.] - ETA: 1s - loss: 0.1912 - categorical_accuracy: 0.9413
59168/60000 [============================>.] - ETA: 1s - loss: 0.1911 - categorical_accuracy: 0.9413
59200/60000 [============================>.] - ETA: 1s - loss: 0.1910 - categorical_accuracy: 0.9414
59232/60000 [============================>.] - ETA: 1s - loss: 0.1910 - categorical_accuracy: 0.9414
59264/60000 [============================>.] - ETA: 1s - loss: 0.1910 - categorical_accuracy: 0.9414
59296/60000 [============================>.] - ETA: 1s - loss: 0.1909 - categorical_accuracy: 0.9414
59328/60000 [============================>.] - ETA: 1s - loss: 0.1908 - categorical_accuracy: 0.9414
59360/60000 [============================>.] - ETA: 1s - loss: 0.1908 - categorical_accuracy: 0.9414
59392/60000 [============================>.] - ETA: 1s - loss: 0.1907 - categorical_accuracy: 0.9414
59424/60000 [============================>.] - ETA: 1s - loss: 0.1907 - categorical_accuracy: 0.9415
59456/60000 [============================>.] - ETA: 0s - loss: 0.1907 - categorical_accuracy: 0.9415
59488/60000 [============================>.] - ETA: 0s - loss: 0.1907 - categorical_accuracy: 0.9415
59520/60000 [============================>.] - ETA: 0s - loss: 0.1907 - categorical_accuracy: 0.9415
59552/60000 [============================>.] - ETA: 0s - loss: 0.1907 - categorical_accuracy: 0.9415
59584/60000 [============================>.] - ETA: 0s - loss: 0.1906 - categorical_accuracy: 0.9415
59616/60000 [============================>.] - ETA: 0s - loss: 0.1905 - categorical_accuracy: 0.9415
59648/60000 [============================>.] - ETA: 0s - loss: 0.1905 - categorical_accuracy: 0.9416
59680/60000 [============================>.] - ETA: 0s - loss: 0.1904 - categorical_accuracy: 0.9416
59712/60000 [============================>.] - ETA: 0s - loss: 0.1903 - categorical_accuracy: 0.9416
59744/60000 [============================>.] - ETA: 0s - loss: 0.1902 - categorical_accuracy: 0.9417
59776/60000 [============================>.] - ETA: 0s - loss: 0.1901 - categorical_accuracy: 0.9417
59808/60000 [============================>.] - ETA: 0s - loss: 0.1900 - categorical_accuracy: 0.9417
59840/60000 [============================>.] - ETA: 0s - loss: 0.1900 - categorical_accuracy: 0.9417
59872/60000 [============================>.] - ETA: 0s - loss: 0.1899 - categorical_accuracy: 0.9418
59904/60000 [============================>.] - ETA: 0s - loss: 0.1898 - categorical_accuracy: 0.9418
59936/60000 [============================>.] - ETA: 0s - loss: 0.1899 - categorical_accuracy: 0.9418
59968/60000 [============================>.] - ETA: 0s - loss: 0.1899 - categorical_accuracy: 0.9418
60000/60000 [==============================] - 112s 2ms/step - loss: 0.1898 - categorical_accuracy: 0.9418 - val_loss: 0.0509 - val_categorical_accuracy: 0.9836

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
 1632/10000 [===>..........................] - ETA: 2s
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
 3360/10000 [=========>....................] - ETA: 2s
 3456/10000 [=========>....................] - ETA: 2s
 3616/10000 [=========>....................] - ETA: 2s
 3776/10000 [==========>...................] - ETA: 2s
 3936/10000 [==========>...................] - ETA: 2s
 4096/10000 [===========>..................] - ETA: 2s
 4256/10000 [===========>..................] - ETA: 2s
 4416/10000 [============>.................] - ETA: 1s
 4576/10000 [============>.................] - ETA: 1s
 4736/10000 [=============>................] - ETA: 1s
 4896/10000 [=============>................] - ETA: 1s
 5056/10000 [==============>...............] - ETA: 1s
 5216/10000 [==============>...............] - ETA: 1s
 5376/10000 [===============>..............] - ETA: 1s
 5536/10000 [===============>..............] - ETA: 1s
 5696/10000 [================>.............] - ETA: 1s
 5856/10000 [================>.............] - ETA: 1s
 6016/10000 [=================>............] - ETA: 1s
 6176/10000 [=================>............] - ETA: 1s
 6336/10000 [==================>...........] - ETA: 1s
 6496/10000 [==================>...........] - ETA: 1s
 6656/10000 [==================>...........] - ETA: 1s
 6816/10000 [===================>..........] - ETA: 1s
 6976/10000 [===================>..........] - ETA: 1s
 7136/10000 [====================>.........] - ETA: 1s
 7296/10000 [====================>.........] - ETA: 0s
 7456/10000 [=====================>........] - ETA: 0s
 7616/10000 [=====================>........] - ETA: 0s
 7776/10000 [======================>.......] - ETA: 0s
 7936/10000 [======================>.......] - ETA: 0s
 8096/10000 [=======================>......] - ETA: 0s
 8224/10000 [=======================>......] - ETA: 0s
 8384/10000 [========================>.....] - ETA: 0s
 8544/10000 [========================>.....] - ETA: 0s
 8704/10000 [=========================>....] - ETA: 0s
 8864/10000 [=========================>....] - ETA: 0s
 9024/10000 [==========================>...] - ETA: 0s
 9184/10000 [==========================>...] - ETA: 0s
 9344/10000 [===========================>..] - ETA: 0s
 9504/10000 [===========================>..] - ETA: 0s
 9664/10000 [===========================>..] - ETA: 0s
 9824/10000 [============================>.] - ETA: 0s
 9984/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 4s 353us/step
[[6.8103336e-09 5.1849856e-08 4.1723905e-07 ... 9.9999630e-01
  9.4768788e-09 2.5766944e-06]
 [6.0028365e-06 8.3364339e-06 9.9997687e-01 ... 4.2538356e-10
  4.0695854e-06 6.3332900e-10]
 [4.4399886e-08 9.9995422e-01 1.6249055e-06 ... 1.3290104e-05
  6.3744699e-07 4.0912692e-08]
 ...
 [9.1472363e-10 9.8750786e-07 1.4429879e-08 ... 2.6959851e-06
  3.2886257e-07 5.7440238e-06]
 [1.2162436e-06 5.1768421e-08 1.1308357e-08 ... 7.9235599e-07
  8.0204295e-04 3.6622433e-07]
 [1.5460929e-06 7.8825838e-07 3.2919095e-05 ... 2.1120513e-08
  1.3740014e-06 1.4533002e-07]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.05085584171635565, 'accuracy_test:': 0.9836000204086304}

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
   71f9960..5491182  master     -> origin/master
Updating 71f9960..5491182
Fast-forward
 error_list/20200518/list_log_json_20200518.md | 1146 ++++++++++++-------------
 1 file changed, 573 insertions(+), 573 deletions(-)
[master f9e7a70] ml_store
 1 file changed, 2044 insertions(+)
To github.com:arita37/mlmodels_store.git
   5491182..f9e7a70  master -> master





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
{'loss': 0.528468132019043, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-18 04:27:33.592438: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
[master e3d83c9] ml_store
 1 file changed, 234 insertions(+)
To github.com:arita37/mlmodels_store.git
   f9e7a70..e3d83c9  master -> master





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
