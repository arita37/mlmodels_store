
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
[master faaff36] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   e3d83c9..faaff36  master -> master





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
	Data preprocessing and feature engineering runtime = 0.22s ...
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
Finished Task with config: {'feature_fraction': 0.7632208205225469, 'learning_rate': 0.14759927035737502, 'min_data_in_leaf': 10, 'num_leaves': 30} and reward: 0.39
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8lN\x11\xf8\xa8\xf5X\r\x00\x00\x00learning_rateq\x02G?\xc2\xe4\x88k\x8c\x97\xfaX\x10\x00\x00\x00min_data_in_leafq\x03K\nX\n\x00\x00\x00num_leavesq\x04K\x1eu.' and reward: 0.39
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8lN\x11\xf8\xa8\xf5X\r\x00\x00\x00learning_rateq\x02G?\xc2\xe4\x88k\x8c\x97\xfaX\x10\x00\x00\x00min_data_in_leafq\x03K\nX\n\x00\x00\x00num_leavesq\x04K\x1eu.' and reward: 0.39
 60%|██████    | 3/5 [00:38<00:25, 12.65s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.7948120453733487, 'learning_rate': 0.008684561875617638, 'min_data_in_leaf': 9, 'num_leaves': 52} and reward: 0.3892
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9o\x19\xab\xab\r\x86X\r\x00\x00\x00learning_rateq\x02G?\x81\xc96)\xe3\r\xceX\x10\x00\x00\x00min_data_in_leafq\x03K\tX\n\x00\x00\x00num_leavesq\x04K4u.' and reward: 0.3892
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9o\x19\xab\xab\r\x86X\r\x00\x00\x00learning_rateq\x02G?\x81\xc96)\xe3\r\xceX\x10\x00\x00\x00min_data_in_leafq\x03K\tX\n\x00\x00\x00num_leavesq\x04K4u.' and reward: 0.3892
 80%|████████  | 4/5 [01:05<00:17, 17.06s/it] 80%|████████  | 4/5 [01:05<00:16, 16.50s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.8225277977411765, 'learning_rate': 0.023117512568337252, 'min_data_in_leaf': 26, 'num_leaves': 66} and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeaR%\xd0\xeb-\x1aX\r\x00\x00\x00learning_rateq\x02G?\x97\xac\x1e\x01\xc8\x94\x1aX\x10\x00\x00\x00min_data_in_leafq\x03K\x1aX\n\x00\x00\x00num_leavesq\x04KBu.' and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeaR%\xd0\xeb-\x1aX\r\x00\x00\x00learning_rateq\x02G?\x97\xac\x1e\x01\xc8\x94\x1aX\x10\x00\x00\x00min_data_in_leafq\x03K\x1aX\n\x00\x00\x00num_leavesq\x04KBu.' and reward: 0.3894
Time for Gradient Boosting hyperparameter optimization: 99.37409687042236
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
 40%|████      | 2/5 [00:49<01:13, 24.62s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.3776804088615886, 'embedding_size_factor': 0.8039953503019865, 'layers.choice': 1, 'learning_rate': 0.0056483252778193474, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 1.5442194883853775e-11} and reward: 0.3838
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd8+\xeas\x19\xa0\xacX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe9\xbaTt\xf5\xdb\xecX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?w"\xb2\xc5\x86\xc6\x91X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\xb0\xfa\x97h\xf5s\xc2u.' and reward: 0.3838
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd8+\xeas\x19\xa0\xacX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe9\xbaTt\xf5\xdb\xecX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?w"\xb2\xc5\x86\xc6\x91X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\xb0\xfa\x97h\xf5s\xc2u.' and reward: 0.3838
 60%|██████    | 3/5 [01:56<01:14, 37.48s/it] 60%|██████    | 3/5 [01:56<01:17, 38.91s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.03897939044069954, 'embedding_size_factor': 0.6653060079692634, 'layers.choice': 1, 'learning_rate': 0.0003323680409493642, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 0.0034308841194362377} and reward: 0.3482
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xa3\xf5\x1bNRV\xf1X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe5J/\xd3A\xed\xfeX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?5\xc85\xdd\xb9\x96\xb2X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?l\x1b\x15\xe2\xdb\xe8\xc9u.' and reward: 0.3482
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xa3\xf5\x1bNRV\xf1X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe5J/\xd3A\xed\xfeX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?5\xc85\xdd\xb9\x96\xb2X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?l\x1b\x15\xe2\xdb\xe8\xc9u.' and reward: 0.3482
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 209.84831023216248
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
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.78s of the -193.27s of remaining time.
Ensemble size: 36
Ensemble weights: 
[0.25       0.08333333 0.02777778 0.22222222 0.08333333 0.11111111
 0.22222222]
	0.4004	 = Validation accuracy score
	1.54s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 314.86s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f3ffa668358>

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
   faaff36..b6a7fc5  master     -> origin/master
Updating faaff36..b6a7fc5
Fast-forward
 error_list/20200518/list_log_json_20200518.md | 1146 ++++++++++++-------------
 1 file changed, 573 insertions(+), 573 deletions(-)
[master 707d6b2] ml_store
 1 file changed, 217 insertions(+)
To github.com:arita37/mlmodels_store.git
   b6a7fc5..707d6b2  master -> master





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
[master c18d108] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   707d6b2..c18d108  master -> master





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
100%|██████████| 10/10 [00:02<00:00,  3.71it/s, avg_epoch_loss=5.26]
INFO:root:Epoch[0] Elapsed time 2.696 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.255887
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.2558869361877445 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f4b3bf8f518>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f4b3bf8f518>

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
Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 86.02it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 1042.0297037760417,
    "abs_error": 365.4886779785156,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 2.421709640742148,
    "sMAPE": 0.5079984143681638,
    "MSIS": 96.8683840120224,
    "QuantileLoss[0.5]": 365.4886779785156,
    "Coverage[0.5]": 1.0,
    "RMSE": 32.28048487516942,
    "NRMSE": 0.6795891552667247,
    "ND": 0.6412082069798519,
    "wQuantileLoss[0.5]": 0.6412082069798519,
    "mean_wQuantileLoss": 0.6412082069798519,
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
100%|██████████| 10/10 [00:01<00:00,  7.57it/s, avg_epoch_loss=2.71e+3]
INFO:root:Epoch[0] Elapsed time 1.321 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=2713.411247
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 2713.4112467447917 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f4b34630ac8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f4b34630ac8>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 179.62it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
100%|██████████| 10/10 [00:01<00:00,  5.41it/s, avg_epoch_loss=5.3]
INFO:root:Epoch[0] Elapsed time 1.850 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.304522
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.304521703720093 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f4b1019dc18>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f4b1019dc18>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 137.78it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 249.4795939127604,
    "abs_error": 157.40029907226562,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.042926483597967,
    "sMAPE": 0.27591431748243406,
    "MSIS": 41.71706096158222,
    "QuantileLoss[0.5]": 157.40029525756836,
    "Coverage[0.5]": 0.5,
    "RMSE": 15.79492304231839,
    "NRMSE": 0.33252469562775555,
    "ND": 0.2761408755653783,
    "wQuantileLoss[0.5]": 0.27614086887292694,
    "mean_wQuantileLoss": 0.27614086887292694,
    "MAE_Coverage": 0.0
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
 30%|███       | 3/10 [00:12<00:29,  4.23s/it, avg_epoch_loss=6.93] 60%|██████    | 6/10 [00:23<00:16,  4.03s/it, avg_epoch_loss=6.91] 90%|█████████ | 9/10 [00:34<00:03,  3.92s/it, avg_epoch_loss=6.88]100%|██████████| 10/10 [00:37<00:00,  3.80s/it, avg_epoch_loss=6.87]
INFO:root:Epoch[0] Elapsed time 37.977 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.872491
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.872490739822387 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f4b1023af28>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f4b1023af28>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 116.63it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 53161.505208333336,
    "abs_error": 2709.94775390625,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 17.955977837234315,
    "sMAPE": 1.4109120595993538,
    "MSIS": 718.2389840762902,
    "QuantileLoss[0.5]": 2709.9474334716797,
    "Coverage[0.5]": 1.0,
    "RMSE": 230.56778874841416,
    "NRMSE": 4.85405871049293,
    "ND": 4.754294305098684,
    "wQuantileLoss[0.5]": 4.754293742932771,
    "mean_wQuantileLoss": 4.754293742932771,
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
100%|██████████| 10/10 [00:00<00:00, 47.02it/s, avg_epoch_loss=5.17]
INFO:root:Epoch[0] Elapsed time 0.213 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.167425
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.167425107955933 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f4b08810080>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f4b08810080>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 150.69it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 512.8293863932291,
    "abs_error": 187.63449096679688,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.243256721995498,
    "sMAPE": 0.3148806754715962,
    "MSIS": 49.73026726215639,
    "QuantileLoss[0.5]": 187.63447952270508,
    "Coverage[0.5]": 0.6666666666666666,
    "RMSE": 22.645736605225036,
    "NRMSE": 0.47675234958368495,
    "ND": 0.32918331748560853,
    "wQuantileLoss[0.5]": 0.3291832974082545,
    "mean_wQuantileLoss": 0.3291832974082545,
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
100%|██████████| 10/10 [00:01<00:00,  8.43it/s, avg_epoch_loss=161]
INFO:root:Epoch[0] Elapsed time 1.186 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=161.108291
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 161.1082910709855 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f4b10196c18>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f4b10196c18>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 152.20it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
 10%|█         | 1/10 [02:07<19:08, 127.64s/it, avg_epoch_loss=0.582] 20%|██        | 2/10 [05:25<19:50, 148.81s/it, avg_epoch_loss=0.565] 30%|███       | 3/10 [09:05<19:50, 170.11s/it, avg_epoch_loss=0.548] 40%|████      | 4/10 [12:01<17:10, 171.69s/it, avg_epoch_loss=0.531] 50%|█████     | 5/10 [15:38<15:27, 185.47s/it, avg_epoch_loss=0.515] 60%|██████    | 6/10 [18:38<12:14, 183.72s/it, avg_epoch_loss=0.499] 70%|███████   | 7/10 [21:49<09:17, 185.90s/it, avg_epoch_loss=0.484] 80%|████████  | 8/10 [25:07<06:19, 189.69s/it, avg_epoch_loss=0.47]  90%|█████████ | 9/10 [28:40<03:16, 196.45s/it, avg_epoch_loss=0.457]100%|██████████| 10/10 [32:13<00:00, 201.59s/it, avg_epoch_loss=0.447]100%|██████████| 10/10 [32:13<00:00, 193.37s/it, avg_epoch_loss=0.447]
INFO:root:Epoch[0] Elapsed time 1933.725 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.446552
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.4465524971485138 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f4b0877def0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f4b0877def0>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 15.50it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
   c18d108..aa77a65  master     -> origin/master
Updating c18d108..aa77a65
Fast-forward
 error_list/20200518/list_log_json_20200518.md     | 1146 ++++++++++-----------
 error_list/20200518/list_log_test_cli_20200518.md |  364 +++----
 2 files changed, 755 insertions(+), 755 deletions(-)
[master f1e7e93] ml_store
 1 file changed, 507 insertions(+)
To github.com:arita37/mlmodels_store.git
   aa77a65..f1e7e93  master -> master





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
{'roc_auc_score': 0.9615384615384616}

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_sklearn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_sklearn.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7f632a116710> 

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
[master f698701] ml_store
 1 file changed, 109 insertions(+)
To github.com:arita37/mlmodels_store.git
   f1e7e93..f698701  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 8.88611457e-01  8.49586845e-01 -3.09114176e-02 -1.22154015e-01
  -1.14722826e+00 -6.80851574e-01 -3.26061306e-01 -1.06787658e+00
  -7.66793627e-02  3.55717262e-01]
 [ 8.76994650e-01  1.23225307e+00 -8.67787223e-01 -2.54179868e-01
   8.91891405e-01  1.39984394e+00 -8.77281519e-01 -7.81911683e-01
  -4.37508983e-01 -1.44087602e+00]
 [ 1.24549398e+00 -7.22391905e-01  1.11813340e+00  1.09899633e+00
   1.00277655e+00 -9.01634490e-01 -5.32234021e-01 -8.22467189e-01
   7.21711292e-01  6.74396105e-01]
 [ 8.95510508e-01  9.20615118e-01  7.94528240e-01 -3.53679249e-02
   8.78099103e-01  2.11060505e+00 -1.02188594e+00 -1.30653407e+00
   7.63804802e-02 -1.87316098e+00]
 [ 1.39198128e+00 -1.90221025e-01 -5.37223024e-01 -4.48738033e-01
   7.04557071e-01 -6.72448039e-01 -7.01344426e-01 -5.57494722e-01
   9.39168744e-01  1.56263850e-01]
 [ 1.18468624e+00 -1.00016919e+00 -5.93843067e-01  1.04499441e+00
   9.65482331e-01  6.08514698e-01 -6.25342001e-01 -6.93286967e-02
  -1.08392067e-01 -3.43900709e-01]
 [ 1.18947778e+00 -6.80678141e-01 -5.68244809e-02 -8.45080274e-02
   8.21783210e-01 -2.97361883e-01 -1.86578994e-01  4.17302005e-01
   7.84770651e-01  4.92336556e-01]
 [ 1.14377130e+00  7.27813500e-01  3.52494364e-01  5.15073614e-01
   1.17718111e+00 -2.78253447e+00 -1.94332341e+00  5.84646610e-01
   3.24274243e-01 -2.36436952e-01]
 [ 9.97855163e-01 -6.00138799e-01  4.57947076e-01  1.46765263e-01
  -9.33557290e-01  5.71804879e-01  5.72962726e-01 -3.68176565e-02
   1.12368489e-01 -1.78175491e-02]
 [ 1.77547698e+00 -2.03394449e-01 -1.98837863e-01  2.42669441e-01
   9.64350564e-01  2.01830179e-01 -5.45774168e-01  6.61020288e-01
   1.79215821e+00 -7.00398505e-01]
 [ 1.16777676e+00 -6.65754518e-01 -1.23312074e+00 -1.67419581e+00
   1.01313574e+00  8.25029824e-01 -1.20464572e-01 -4.98213564e-01
  -3.10984978e-01 -1.18231813e+00]
 [ 7.61706684e-01 -1.48515645e+00  1.30253554e+00 -5.92461285e-01
  -1.64162479e+00 -2.30490794e+00 -1.34869645e+00 -3.18171727e-02
   1.12487742e-01 -3.62612088e-01]
 [ 1.02242019e+00  1.85300949e+00  6.44353666e-01  1.42251373e-01
   1.15080755e+00  5.13505480e-01 -4.59942831e-01  3.72456852e-01
  -1.48489803e-01  3.71670291e-01]
 [ 8.95623122e-01 -2.29820588e+00 -1.95225583e-02  1.45652739e+00
  -1.85064099e+00  3.16637236e-01  1.11337266e-01 -2.66412594e+00
  -4.26428618e-01 -8.39988915e-01]
 [ 6.10942600e-01 -2.79099641e+00 -1.33520272e+00 -4.56117555e-01
  -9.44959948e-01 -9.79890252e-01 -1.56993672e-01  6.92574348e-01
  -4.78672356e-01 -1.06460122e-01]
 [ 1.03967316e+00 -7.31530982e-01  3.61847316e-01 -1.56573815e+00
   9.59288190e-01  1.01382247e+00 -1.78791289e+00 -2.22711263e+00
  -1.69933360e+00 -4.24492791e-01]
 [ 9.29250600e-01 -1.10657307e+00 -1.95816909e+00 -3.59224096e-01
  -1.21258781e+00  5.05381903e-01  5.42645295e-01  1.21794090e+00
  -1.94068096e+00  6.77807571e-01]
 [ 1.06702918e+00 -4.29142278e-01  3.50167159e-01  1.20845633e+00
   7.51480619e-01  1.11570180e+00 -4.79157099e-01  8.40861558e-01
  -1.02887218e-01  1.71647264e-02]
 [ 8.57719529e-01  9.81122462e-02 -2.60466059e-01  1.06032751e+00
  -1.39003042e+00 -1.71116766e+00  2.65642403e-01  1.65712464e+00
   1.41767401e+00  4.45096710e-01]
 [ 7.83440538e-01 -5.11884476e-02  8.24584625e-01 -7.25597119e-01
   9.31717197e-01 -8.67768678e-01  3.03085711e+00 -1.35977326e-01
  -7.97269785e-01  6.54580153e-01]
 [ 6.92114488e-01 -6.06524918e-02  2.05635552e+00 -2.41350300e+00
   1.17456965e+00 -1.77756638e+00 -2.81736269e-01 -7.77858827e-01
   1.11584111e+00  1.76024923e+00]
 [ 8.98917161e-01  5.57439453e-01 -7.58067329e-01  1.81038744e-01
   8.41467206e-01  1.10717545e+00  6.93366226e-01  1.44287693e+00
  -5.39681562e-01 -8.08847196e-01]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 9.36211246e-01  2.04377395e-01 -1.49419377e+00  6.12232523e-01
  -9.84377246e-01  7.44884536e-01  4.94341651e-01 -3.62812886e-02
  -8.32395348e-01 -4.46699203e-01]
 [ 1.34740825e+00  7.33023232e-01  8.38634747e-01 -1.89881206e+00
  -5.42459922e-01 -1.11711069e+00 -1.09715436e+00 -5.08972278e-01
  -1.66485955e-01 -1.03918232e+00]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7fc10ad3feb8>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7fc1250b26d8> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 9.80427414e-01  1.93752881e+00 -2.30839743e-01  3.66332015e-01
   1.10018476e+00 -1.04458938e+00 -3.44987210e-01  2.05117344e+00
   5.85662000e-01 -2.79308500e+00]
 [ 8.88611457e-01  8.49586845e-01 -3.09114176e-02 -1.22154015e-01
  -1.14722826e+00 -6.80851574e-01 -3.26061306e-01 -1.06787658e+00
  -7.66793627e-02  3.55717262e-01]
 [ 1.44682180e+00  8.07455917e-01  1.49810818e+00  3.12238689e-01
  -6.82430193e-01 -1.93321640e-01  2.88078167e-01 -2.07680202e+00
   9.47501167e-01 -3.00976154e-01]
 [ 7.83440538e-01 -5.11884476e-02  8.24584625e-01 -7.25597119e-01
   9.31717197e-01 -8.67768678e-01  3.03085711e+00 -1.35977326e-01
  -7.97269785e-01  6.54580153e-01]
 [ 2.07582971e+00 -1.40232915e+00 -4.79184915e-01  4.51122939e-01
   1.03436581e+00 -6.94920901e-01 -4.18937898e-01  5.15413802e-01
  -1.11487105e+00 -1.95210529e+00]
 [ 9.29250600e-01 -1.10657307e+00 -1.95816909e+00 -3.59224096e-01
  -1.21258781e+00  5.05381903e-01  5.42645295e-01  1.21794090e+00
  -1.94068096e+00  6.77807571e-01]
 [ 1.98519313e+00  6.74711526e-01 -1.39662042e+00  6.18539131e-01
   1.22382712e+00 -4.43171931e-01 -1.89148284e-03  1.81053491e+00
  -1.30572692e+00 -8.61316361e-01]
 [ 1.02242019e+00  1.85300949e+00  6.44353666e-01  1.42251373e-01
   1.15080755e+00  5.13505480e-01 -4.59942831e-01  3.72456852e-01
  -1.48489803e-01  3.71670291e-01]
 [ 8.98917161e-01  5.57439453e-01 -7.58067329e-01  1.81038744e-01
   8.41467206e-01  1.10717545e+00  6.93366226e-01  1.44287693e+00
  -5.39681562e-01 -8.08847196e-01]
 [ 1.05936450e-01 -7.37289628e-01  6.50323214e-01  1.64665066e-01
  -1.53556118e+00  7.78174179e-01  5.03170861e-02  3.09816759e-01
   1.05132077e+00  6.06548400e-01]
 [ 1.18559003e+00  8.64644065e-02  1.23289919e+00 -2.14246673e+00
   1.03334100e+00 -8.30168864e-01  3.67231814e-01  4.51615951e-01
   1.10417433e+00 -4.22856961e-01]
 [ 6.23688521e-01  1.20660790e+00  9.03999174e-01 -2.82863552e-01
  -1.18913787e+00 -2.66326884e-01  1.42361443e+00  1.06897162e+00
   4.03714310e-02  1.57546791e+00]
 [ 1.21619061e+00 -1.90005215e-02  8.60891241e-01 -2.26760192e-01
  -1.36419132e+00 -1.56450785e+00  1.63169151e+00  9.31255679e-01
   9.49808815e-01 -8.80189065e-01]
 [ 8.61462558e-01  7.43205537e-02 -1.34501002e+00 -1.99560718e-01
  -1.47533915e+00 -6.54603169e-01 -3.14563862e-01  3.18014296e-01
  -8.90271552e-01 -1.29525789e+00]
 [ 1.01177337e+00  9.57467711e-02  7.31402517e-01  1.03345080e+00
  -1.42203164e+00 -1.46273275e-01 -1.74549518e-02 -8.57496825e-01
  -9.34181843e-01  9.54495667e-01]
 [ 8.48069266e-01  4.51946037e-01  6.30195671e-01 -1.57915629e+00
   8.27987371e-01 -8.28627979e-01 -1.05344713e-01  5.28879746e-01
  -2.23708651e+00 -4.14846901e-01]
 [ 1.14377130e+00  7.27813500e-01  3.52494364e-01  5.15073614e-01
   1.17718111e+00 -2.78253447e+00 -1.94332341e+00  5.84646610e-01
   3.24274243e-01 -2.36436952e-01]
 [ 5.63077902e-01 -1.17598267e+00 -1.74180344e-01  1.01012718e+00
   1.06796368e+00  9.20017933e-01 -1.68198840e-01 -1.95057341e-01
   8.05393424e-01  4.61164100e-01]
 [ 8.72267394e-01 -2.51630386e+00 -7.75070287e-01 -5.95667881e-01
   1.02600767e+00 -3.09121319e-01  1.74643509e+00  5.10937774e-01
   1.71066184e+00  1.41640538e-01]
 [ 3.45715997e-01 -4.13029310e-01 -4.68673816e-01  1.83471763e+00
   7.71514409e-01  5.64382855e-01  2.18628366e-02  2.13782807e+00
  -7.85533997e-01  8.53281222e-01]
 [ 1.16777676e+00 -6.65754518e-01 -1.23312074e+00 -1.67419581e+00
   1.01313574e+00  8.25029824e-01 -1.20464572e-01 -4.98213564e-01
  -3.10984978e-01 -1.18231813e+00]
 [ 4.73307772e-01 -9.73267585e-01 -2.28140691e-01  1.75167729e-01
  -1.01366961e+00 -5.34836927e-02  3.93787731e-01 -1.83061987e-01
  -2.21028902e-01  5.80330113e-01]
 [ 1.03967316e+00 -7.31530982e-01  3.61847316e-01 -1.56573815e+00
   9.59288190e-01  1.01382247e+00 -1.78791289e+00 -2.22711263e+00
  -1.69933360e+00 -4.24492791e-01]
 [ 7.75285326e-01  1.47016034e+00  1.03298378e+00 -8.70008223e-01
   7.86556511e-01  3.69190470e-01 -1.43195745e-01  8.53282186e-01
  -1.39711730e-01 -2.22414029e-01]
 [ 4.67397905e-01 -2.37875265e-01 -1.54491194e-01 -7.55662765e-01
  -5.47062239e-01  1.85143789e+00 -1.46405357e+00  2.09096677e-01
   1.55501599e+00 -9.24323185e-02]]
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
[[ 1.12641981e+00 -6.29441604e-01  1.10100020e+00 -1.11343610e+00
   9.44595066e-01 -6.74100249e-02 -1.83400197e-01  1.16143998e+00
  -2.75293863e-02  7.80027135e-01]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 5.58538729e-01 -5.16347909e-01 -5.18145552e-01  3.51116897e-01
   8.25506954e-01 -6.87704631e-02 -9.52062101e-01 -1.34776494e+00
   1.47073986e+00 -1.46140360e+00]
 [ 1.34728643e+00 -3.64538050e-01  8.07509886e-02 -4.59717681e-01
  -8.89487596e-01  1.70548352e+00  9.49961101e-02  2.40505552e-01
  -9.99426501e-01 -7.67803746e-01]
 [ 1.01177337e+00  9.57467711e-02  7.31402517e-01  1.03345080e+00
  -1.42203164e+00 -1.46273275e-01 -1.74549518e-02 -8.57496825e-01
  -9.34181843e-01  9.54495667e-01]
 [ 1.32857949e+00 -5.63236604e-01 -1.06179676e+00  2.39014596e+00
  -1.68450770e+00  2.45422849e-01 -5.69148654e-01  1.15259914e+00
  -2.24235772e-01  1.32247779e-01]
 [ 8.78643802e-01  1.03703898e+00 -4.77124206e-01  6.72619748e-01
  -1.04948638e+00  2.42887697e+00  5.24750492e-01  1.00568668e+00
   3.53567216e-01 -3.59901817e-02]
 [ 1.16777676e+00 -6.65754518e-01 -1.23312074e+00 -1.67419581e+00
   1.01313574e+00  8.25029824e-01 -1.20464572e-01 -4.98213564e-01
  -3.10984978e-01 -1.18231813e+00]
 [ 9.67037267e-01  3.82715174e-01 -8.06184817e-01 -2.88997343e-01
   9.08526041e-01 -3.91816240e-01  1.62091229e+00  6.84001328e-01
  -3.53409983e-01 -2.51674208e-01]
 [ 6.13636707e-01  3.16658895e-01  1.34710546e+00 -1.89526695e+00
  -7.60458095e-01  8.97291174e-02 -3.29051549e-01  4.10265745e-01
   8.59870972e-01 -1.04906775e+00]
 [ 4.73307772e-01 -9.73267585e-01 -2.28140691e-01  1.75167729e-01
  -1.01366961e+00 -5.34836927e-02  3.93787731e-01 -1.83061987e-01
  -2.21028902e-01  5.80330113e-01]
 [ 7.90323893e-01  1.61336137e+00 -2.09424782e+00 -3.74804687e-01
   9.15884042e-01 -7.49969617e-01  3.10272288e-01  2.05462410e+00
   5.34095368e-02 -2.28765829e-01]
 [ 8.95510508e-01  9.20615118e-01  7.94528240e-01 -3.53679249e-02
   8.78099103e-01  2.11060505e+00 -1.02188594e+00 -1.30653407e+00
   7.63804802e-02 -1.87316098e+00]
 [ 1.18468624e+00 -1.00016919e+00 -5.93843067e-01  1.04499441e+00
   9.65482331e-01  6.08514698e-01 -6.25342001e-01 -6.93286967e-02
  -1.08392067e-01 -3.43900709e-01]
 [ 7.83440538e-01 -5.11884476e-02  8.24584625e-01 -7.25597119e-01
   9.31717197e-01 -8.67768678e-01  3.03085711e+00 -1.35977326e-01
  -7.97269785e-01  6.54580153e-01]
 [ 6.10942600e-01 -2.79099641e+00 -1.33520272e+00 -4.56117555e-01
  -9.44959948e-01 -9.79890252e-01 -1.56993672e-01  6.92574348e-01
  -4.78672356e-01 -1.06460122e-01]
 [ 8.98917161e-01  5.57439453e-01 -7.58067329e-01  1.81038744e-01
   8.41467206e-01  1.10717545e+00  6.93366226e-01  1.44287693e+00
  -5.39681562e-01 -8.08847196e-01]
 [ 2.07582971e+00 -1.40232915e+00 -4.79184915e-01  4.51122939e-01
   1.03436581e+00 -6.94920901e-01 -4.18937898e-01  5.15413802e-01
  -1.11487105e+00 -1.95210529e+00]
 [ 6.91743730e-01  1.00978733e+00 -1.21333813e+00 -1.55694156e+00
  -1.20257258e+00 -6.12442128e-01 -2.69836174e+00 -1.39351805e-01
  -7.28537489e-01  7.22518992e-02]
 [ 3.45715997e-01 -4.13029310e-01 -4.68673816e-01  1.83471763e+00
   7.71514409e-01  5.64382855e-01  2.18628366e-02  2.13782807e+00
  -7.85533997e-01  8.53281222e-01]
 [ 7.73703613e-01  1.27852808e+00 -2.11416392e+00 -4.42229280e-01
   1.06821044e+00  3.23527354e-01 -2.50644065e+00 -1.09991490e-01
   8.54894544e-03 -4.11639163e-01]
 [ 8.58774962e-01  2.29371761e+00 -1.47023709e+00 -8.30010986e-01
  -6.72049816e-01 -1.01951985e+00  5.99213235e-01 -2.14653842e-01
   1.02124813e+00  6.06403944e-01]
 [ 6.25673373e-01  5.92472801e-01  6.74570707e-01  1.19783084e+00
   1.23187251e+00  1.70459417e+00 -7.67309826e-01  1.04008915e+00
  -9.18440038e-01  1.46089238e+00]
 [ 9.80427414e-01  1.93752881e+00 -2.30839743e-01  3.66332015e-01
   1.10018476e+00 -1.04458938e+00 -3.44987210e-01  2.05117344e+00
   5.85662000e-01 -2.79308500e+00]
 [ 4.41189807e-01  4.79852371e-01 -1.92003697e-01 -1.55269878e+00
  -1.88873982e+00  5.78464420e-01  3.98598388e-01 -9.61263599e-01
  -1.45832446e+00 -3.05376438e+00]]
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
[master cedfad6] ml_store
 1 file changed, 322 insertions(+)
To github.com:arita37/mlmodels_store.git
   f698701..cedfad6  master -> master





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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140682649550800
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140682649550576
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140682649549344
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140682649548896
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140682649548392
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140682649548056

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
grad_step = 000000, loss = 1.041672
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.926361
grad_step = 000002, loss = 0.846514
grad_step = 000003, loss = 0.758112
grad_step = 000004, loss = 0.649925
grad_step = 000005, loss = 0.521907
grad_step = 000006, loss = 0.375254
grad_step = 000007, loss = 0.225037
grad_step = 000008, loss = 0.115736
grad_step = 000009, loss = 0.110838
grad_step = 000010, loss = 0.160504
grad_step = 000011, loss = 0.133381
grad_step = 000012, loss = 0.063682
grad_step = 000013, loss = 0.021507
grad_step = 000014, loss = 0.019485
grad_step = 000015, loss = 0.030499
grad_step = 000016, loss = 0.039257
grad_step = 000017, loss = 0.044509
grad_step = 000018, loss = 0.045179
grad_step = 000019, loss = 0.039450
grad_step = 000020, loss = 0.029954
grad_step = 000021, loss = 0.022823
grad_step = 000022, loss = 0.022069
grad_step = 000023, loss = 0.025425
grad_step = 000024, loss = 0.028894
grad_step = 000025, loss = 0.029742
grad_step = 000026, loss = 0.025729
grad_step = 000027, loss = 0.018442
grad_step = 000028, loss = 0.012148
grad_step = 000029, loss = 0.008855
grad_step = 000030, loss = 0.007917
grad_step = 000031, loss = 0.008138
grad_step = 000032, loss = 0.008692
grad_step = 000033, loss = 0.008881
grad_step = 000034, loss = 0.008407
grad_step = 000035, loss = 0.007671
grad_step = 000036, loss = 0.007321
grad_step = 000037, loss = 0.007519
grad_step = 000038, loss = 0.007926
grad_step = 000039, loss = 0.008251
grad_step = 000040, loss = 0.008298
grad_step = 000041, loss = 0.007861
grad_step = 000042, loss = 0.007042
grad_step = 000043, loss = 0.006331
grad_step = 000044, loss = 0.006062
grad_step = 000045, loss = 0.006095
grad_step = 000046, loss = 0.006150
grad_step = 000047, loss = 0.006071
grad_step = 000048, loss = 0.005777
grad_step = 000049, loss = 0.005281
grad_step = 000050, loss = 0.004774
grad_step = 000051, loss = 0.004510
grad_step = 000052, loss = 0.004551
grad_step = 000053, loss = 0.004748
grad_step = 000054, loss = 0.004936
grad_step = 000055, loss = 0.004995
grad_step = 000056, loss = 0.004878
grad_step = 000057, loss = 0.004687
grad_step = 000058, loss = 0.004572
grad_step = 000059, loss = 0.004563
grad_step = 000060, loss = 0.004603
grad_step = 000061, loss = 0.004630
grad_step = 000062, loss = 0.004591
grad_step = 000063, loss = 0.004477
grad_step = 000064, loss = 0.004355
grad_step = 000065, loss = 0.004286
grad_step = 000066, loss = 0.004261
grad_step = 000067, loss = 0.004253
grad_step = 000068, loss = 0.004229
grad_step = 000069, loss = 0.004163
grad_step = 000070, loss = 0.004085
grad_step = 000071, loss = 0.004034
grad_step = 000072, loss = 0.004019
grad_step = 000073, loss = 0.004029
grad_step = 000074, loss = 0.004034
grad_step = 000075, loss = 0.004009
grad_step = 000076, loss = 0.003964
grad_step = 000077, loss = 0.003916
grad_step = 000078, loss = 0.003883
grad_step = 000079, loss = 0.003871
grad_step = 000080, loss = 0.003860
grad_step = 000081, loss = 0.003838
grad_step = 000082, loss = 0.003803
grad_step = 000083, loss = 0.003763
grad_step = 000084, loss = 0.003732
grad_step = 000085, loss = 0.003711
grad_step = 000086, loss = 0.003695
grad_step = 000087, loss = 0.003676
grad_step = 000088, loss = 0.003651
grad_step = 000089, loss = 0.003622
grad_step = 000090, loss = 0.003595
grad_step = 000091, loss = 0.003572
grad_step = 000092, loss = 0.003550
grad_step = 000093, loss = 0.003528
grad_step = 000094, loss = 0.003504
grad_step = 000095, loss = 0.003478
grad_step = 000096, loss = 0.003453
grad_step = 000097, loss = 0.003429
grad_step = 000098, loss = 0.003406
grad_step = 000099, loss = 0.003384
grad_step = 000100, loss = 0.003363
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.003339
grad_step = 000102, loss = 0.003315
grad_step = 000103, loss = 0.003290
grad_step = 000104, loss = 0.003267
grad_step = 000105, loss = 0.003245
grad_step = 000106, loss = 0.003223
grad_step = 000107, loss = 0.003200
grad_step = 000108, loss = 0.003176
grad_step = 000109, loss = 0.003152
grad_step = 000110, loss = 0.003129
grad_step = 000111, loss = 0.003106
grad_step = 000112, loss = 0.003082
grad_step = 000113, loss = 0.003057
grad_step = 000114, loss = 0.003033
grad_step = 000115, loss = 0.003010
grad_step = 000116, loss = 0.002985
grad_step = 000117, loss = 0.002961
grad_step = 000118, loss = 0.002937
grad_step = 000119, loss = 0.002913
grad_step = 000120, loss = 0.002889
grad_step = 000121, loss = 0.002864
grad_step = 000122, loss = 0.002840
grad_step = 000123, loss = 0.002817
grad_step = 000124, loss = 0.002793
grad_step = 000125, loss = 0.002769
grad_step = 000126, loss = 0.002745
grad_step = 000127, loss = 0.002721
grad_step = 000128, loss = 0.002697
grad_step = 000129, loss = 0.002673
grad_step = 000130, loss = 0.002649
grad_step = 000131, loss = 0.002625
grad_step = 000132, loss = 0.002602
grad_step = 000133, loss = 0.002578
grad_step = 000134, loss = 0.002555
grad_step = 000135, loss = 0.002532
grad_step = 000136, loss = 0.002509
grad_step = 000137, loss = 0.002486
grad_step = 000138, loss = 0.002463
grad_step = 000139, loss = 0.002440
grad_step = 000140, loss = 0.002417
grad_step = 000141, loss = 0.002395
grad_step = 000142, loss = 0.002373
grad_step = 000143, loss = 0.002350
grad_step = 000144, loss = 0.002328
grad_step = 000145, loss = 0.002306
grad_step = 000146, loss = 0.002284
grad_step = 000147, loss = 0.002262
grad_step = 000148, loss = 0.002239
grad_step = 000149, loss = 0.002217
grad_step = 000150, loss = 0.002195
grad_step = 000151, loss = 0.002172
grad_step = 000152, loss = 0.002148
grad_step = 000153, loss = 0.002124
grad_step = 000154, loss = 0.002101
grad_step = 000155, loss = 0.002077
grad_step = 000156, loss = 0.002053
grad_step = 000157, loss = 0.002028
grad_step = 000158, loss = 0.002003
grad_step = 000159, loss = 0.001978
grad_step = 000160, loss = 0.001952
grad_step = 000161, loss = 0.001927
grad_step = 000162, loss = 0.001902
grad_step = 000163, loss = 0.001879
grad_step = 000164, loss = 0.001858
grad_step = 000165, loss = 0.001830
grad_step = 000166, loss = 0.001796
grad_step = 000167, loss = 0.001768
grad_step = 000168, loss = 0.001747
grad_step = 000169, loss = 0.001720
grad_step = 000170, loss = 0.001687
grad_step = 000171, loss = 0.001660
grad_step = 000172, loss = 0.001637
grad_step = 000173, loss = 0.001609
grad_step = 000174, loss = 0.001577
grad_step = 000175, loss = 0.001550
grad_step = 000176, loss = 0.001525
grad_step = 000177, loss = 0.001496
grad_step = 000178, loss = 0.001466
grad_step = 000179, loss = 0.001439
grad_step = 000180, loss = 0.001412
grad_step = 000181, loss = 0.001383
grad_step = 000182, loss = 0.001355
grad_step = 000183, loss = 0.001330
grad_step = 000184, loss = 0.001304
grad_step = 000185, loss = 0.001276
grad_step = 000186, loss = 0.001251
grad_step = 000187, loss = 0.001227
grad_step = 000188, loss = 0.001202
grad_step = 000189, loss = 0.001177
grad_step = 000190, loss = 0.001154
grad_step = 000191, loss = 0.001131
grad_step = 000192, loss = 0.001108
grad_step = 000193, loss = 0.001087
grad_step = 000194, loss = 0.001066
grad_step = 000195, loss = 0.001045
grad_step = 000196, loss = 0.001026
grad_step = 000197, loss = 0.001008
grad_step = 000198, loss = 0.000991
grad_step = 000199, loss = 0.000975
grad_step = 000200, loss = 0.000958
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.000943
grad_step = 000202, loss = 0.000928
grad_step = 000203, loss = 0.000915
grad_step = 000204, loss = 0.000903
grad_step = 000205, loss = 0.000891
grad_step = 000206, loss = 0.000879
grad_step = 000207, loss = 0.000867
grad_step = 000208, loss = 0.000856
grad_step = 000209, loss = 0.000846
grad_step = 000210, loss = 0.000835
grad_step = 000211, loss = 0.000825
grad_step = 000212, loss = 0.000816
grad_step = 000213, loss = 0.000806
grad_step = 000214, loss = 0.000796
grad_step = 000215, loss = 0.000787
grad_step = 000216, loss = 0.000778
grad_step = 000217, loss = 0.000769
grad_step = 000218, loss = 0.000760
grad_step = 000219, loss = 0.000753
grad_step = 000220, loss = 0.000747
grad_step = 000221, loss = 0.000745
grad_step = 000222, loss = 0.000740
grad_step = 000223, loss = 0.000730
grad_step = 000224, loss = 0.000714
grad_step = 000225, loss = 0.000704
grad_step = 000226, loss = 0.000702
grad_step = 000227, loss = 0.000699
grad_step = 000228, loss = 0.000690
grad_step = 000229, loss = 0.000678
grad_step = 000230, loss = 0.000670
grad_step = 000231, loss = 0.000667
grad_step = 000232, loss = 0.000663
grad_step = 000233, loss = 0.000657
grad_step = 000234, loss = 0.000648
grad_step = 000235, loss = 0.000640
grad_step = 000236, loss = 0.000636
grad_step = 000237, loss = 0.000633
grad_step = 000238, loss = 0.000630
grad_step = 000239, loss = 0.000624
grad_step = 000240, loss = 0.000618
grad_step = 000241, loss = 0.000611
grad_step = 000242, loss = 0.000606
grad_step = 000243, loss = 0.000603
grad_step = 000244, loss = 0.000602
grad_step = 000245, loss = 0.000601
grad_step = 000246, loss = 0.000597
grad_step = 000247, loss = 0.000591
grad_step = 000248, loss = 0.000583
grad_step = 000249, loss = 0.000577
grad_step = 000250, loss = 0.000573
grad_step = 000251, loss = 0.000571
grad_step = 000252, loss = 0.000570
grad_step = 000253, loss = 0.000566
grad_step = 000254, loss = 0.000562
grad_step = 000255, loss = 0.000556
grad_step = 000256, loss = 0.000551
grad_step = 000257, loss = 0.000547
grad_step = 000258, loss = 0.000544
grad_step = 000259, loss = 0.000541
grad_step = 000260, loss = 0.000539
grad_step = 000261, loss = 0.000538
grad_step = 000262, loss = 0.000536
grad_step = 000263, loss = 0.000534
grad_step = 000264, loss = 0.000531
grad_step = 000265, loss = 0.000528
grad_step = 000266, loss = 0.000525
grad_step = 000267, loss = 0.000521
grad_step = 000268, loss = 0.000517
grad_step = 000269, loss = 0.000513
grad_step = 000270, loss = 0.000510
grad_step = 000271, loss = 0.000507
grad_step = 000272, loss = 0.000505
grad_step = 000273, loss = 0.000502
grad_step = 000274, loss = 0.000500
grad_step = 000275, loss = 0.000499
grad_step = 000276, loss = 0.000498
grad_step = 000277, loss = 0.000498
grad_step = 000278, loss = 0.000500
grad_step = 000279, loss = 0.000507
grad_step = 000280, loss = 0.000518
grad_step = 000281, loss = 0.000526
grad_step = 000282, loss = 0.000519
grad_step = 000283, loss = 0.000497
grad_step = 000284, loss = 0.000480
grad_step = 000285, loss = 0.000483
grad_step = 000286, loss = 0.000494
grad_step = 000287, loss = 0.000497
grad_step = 000288, loss = 0.000484
grad_step = 000289, loss = 0.000472
grad_step = 000290, loss = 0.000473
grad_step = 000291, loss = 0.000480
grad_step = 000292, loss = 0.000481
grad_step = 000293, loss = 0.000472
grad_step = 000294, loss = 0.000465
grad_step = 000295, loss = 0.000465
grad_step = 000296, loss = 0.000470
grad_step = 000297, loss = 0.000469
grad_step = 000298, loss = 0.000463
grad_step = 000299, loss = 0.000458
grad_step = 000300, loss = 0.000458
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000460
grad_step = 000302, loss = 0.000460
grad_step = 000303, loss = 0.000457
grad_step = 000304, loss = 0.000453
grad_step = 000305, loss = 0.000451
grad_step = 000306, loss = 0.000451
grad_step = 000307, loss = 0.000452
grad_step = 000308, loss = 0.000451
grad_step = 000309, loss = 0.000448
grad_step = 000310, loss = 0.000445
grad_step = 000311, loss = 0.000444
grad_step = 000312, loss = 0.000444
grad_step = 000313, loss = 0.000444
grad_step = 000314, loss = 0.000443
grad_step = 000315, loss = 0.000441
grad_step = 000316, loss = 0.000439
grad_step = 000317, loss = 0.000438
grad_step = 000318, loss = 0.000437
grad_step = 000319, loss = 0.000436
grad_step = 000320, loss = 0.000436
grad_step = 000321, loss = 0.000435
grad_step = 000322, loss = 0.000434
grad_step = 000323, loss = 0.000432
grad_step = 000324, loss = 0.000431
grad_step = 000325, loss = 0.000430
grad_step = 000326, loss = 0.000429
grad_step = 000327, loss = 0.000428
grad_step = 000328, loss = 0.000427
grad_step = 000329, loss = 0.000426
grad_step = 000330, loss = 0.000426
grad_step = 000331, loss = 0.000425
grad_step = 000332, loss = 0.000424
grad_step = 000333, loss = 0.000424
grad_step = 000334, loss = 0.000424
grad_step = 000335, loss = 0.000424
grad_step = 000336, loss = 0.000424
grad_step = 000337, loss = 0.000425
grad_step = 000338, loss = 0.000426
grad_step = 000339, loss = 0.000427
grad_step = 000340, loss = 0.000431
grad_step = 000341, loss = 0.000437
grad_step = 000342, loss = 0.000446
grad_step = 000343, loss = 0.000449
grad_step = 000344, loss = 0.000444
grad_step = 000345, loss = 0.000429
grad_step = 000346, loss = 0.000414
grad_step = 000347, loss = 0.000410
grad_step = 000348, loss = 0.000416
grad_step = 000349, loss = 0.000424
grad_step = 000350, loss = 0.000425
grad_step = 000351, loss = 0.000418
grad_step = 000352, loss = 0.000408
grad_step = 000353, loss = 0.000405
grad_step = 000354, loss = 0.000407
grad_step = 000355, loss = 0.000412
grad_step = 000356, loss = 0.000414
grad_step = 000357, loss = 0.000411
grad_step = 000358, loss = 0.000405
grad_step = 000359, loss = 0.000400
grad_step = 000360, loss = 0.000399
grad_step = 000361, loss = 0.000400
grad_step = 000362, loss = 0.000402
grad_step = 000363, loss = 0.000404
grad_step = 000364, loss = 0.000403
grad_step = 000365, loss = 0.000400
grad_step = 000366, loss = 0.000396
grad_step = 000367, loss = 0.000394
grad_step = 000368, loss = 0.000392
grad_step = 000369, loss = 0.000392
grad_step = 000370, loss = 0.000392
grad_step = 000371, loss = 0.000393
grad_step = 000372, loss = 0.000393
grad_step = 000373, loss = 0.000393
grad_step = 000374, loss = 0.000392
grad_step = 000375, loss = 0.000390
grad_step = 000376, loss = 0.000389
grad_step = 000377, loss = 0.000387
grad_step = 000378, loss = 0.000386
grad_step = 000379, loss = 0.000384
grad_step = 000380, loss = 0.000383
grad_step = 000381, loss = 0.000382
grad_step = 000382, loss = 0.000380
grad_step = 000383, loss = 0.000379
grad_step = 000384, loss = 0.000378
grad_step = 000385, loss = 0.000377
grad_step = 000386, loss = 0.000376
grad_step = 000387, loss = 0.000376
grad_step = 000388, loss = 0.000375
grad_step = 000389, loss = 0.000374
grad_step = 000390, loss = 0.000373
grad_step = 000391, loss = 0.000372
grad_step = 000392, loss = 0.000371
grad_step = 000393, loss = 0.000371
grad_step = 000394, loss = 0.000371
grad_step = 000395, loss = 0.000372
grad_step = 000396, loss = 0.000378
grad_step = 000397, loss = 0.000394
grad_step = 000398, loss = 0.000433
grad_step = 000399, loss = 0.000500
grad_step = 000400, loss = 0.000584
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000580
grad_step = 000402, loss = 0.000471
grad_step = 000403, loss = 0.000369
grad_step = 000404, loss = 0.000404
grad_step = 000405, loss = 0.000482
grad_step = 000406, loss = 0.000453
grad_step = 000407, loss = 0.000372
grad_step = 000408, loss = 0.000380
grad_step = 000409, loss = 0.000434
grad_step = 000410, loss = 0.000418
grad_step = 000411, loss = 0.000365
grad_step = 000412, loss = 0.000375
grad_step = 000413, loss = 0.000410
grad_step = 000414, loss = 0.000391
grad_step = 000415, loss = 0.000358
grad_step = 000416, loss = 0.000372
grad_step = 000417, loss = 0.000391
grad_step = 000418, loss = 0.000374
grad_step = 000419, loss = 0.000355
grad_step = 000420, loss = 0.000368
grad_step = 000421, loss = 0.000378
grad_step = 000422, loss = 0.000363
grad_step = 000423, loss = 0.000352
grad_step = 000424, loss = 0.000362
grad_step = 000425, loss = 0.000368
grad_step = 000426, loss = 0.000357
grad_step = 000427, loss = 0.000350
grad_step = 000428, loss = 0.000356
grad_step = 000429, loss = 0.000361
grad_step = 000430, loss = 0.000353
grad_step = 000431, loss = 0.000347
grad_step = 000432, loss = 0.000351
grad_step = 000433, loss = 0.000354
grad_step = 000434, loss = 0.000350
grad_step = 000435, loss = 0.000345
grad_step = 000436, loss = 0.000346
grad_step = 000437, loss = 0.000349
grad_step = 000438, loss = 0.000347
grad_step = 000439, loss = 0.000343
grad_step = 000440, loss = 0.000343
grad_step = 000441, loss = 0.000344
grad_step = 000442, loss = 0.000344
grad_step = 000443, loss = 0.000341
grad_step = 000444, loss = 0.000340
grad_step = 000445, loss = 0.000340
grad_step = 000446, loss = 0.000341
grad_step = 000447, loss = 0.000340
grad_step = 000448, loss = 0.000338
grad_step = 000449, loss = 0.000337
grad_step = 000450, loss = 0.000337
grad_step = 000451, loss = 0.000337
grad_step = 000452, loss = 0.000336
grad_step = 000453, loss = 0.000335
grad_step = 000454, loss = 0.000334
grad_step = 000455, loss = 0.000333
grad_step = 000456, loss = 0.000333
grad_step = 000457, loss = 0.000333
grad_step = 000458, loss = 0.000332
grad_step = 000459, loss = 0.000331
grad_step = 000460, loss = 0.000330
grad_step = 000461, loss = 0.000330
grad_step = 000462, loss = 0.000329
grad_step = 000463, loss = 0.000328
grad_step = 000464, loss = 0.000327
grad_step = 000465, loss = 0.000327
grad_step = 000466, loss = 0.000326
grad_step = 000467, loss = 0.000325
grad_step = 000468, loss = 0.000325
grad_step = 000469, loss = 0.000324
grad_step = 000470, loss = 0.000324
grad_step = 000471, loss = 0.000323
grad_step = 000472, loss = 0.000322
grad_step = 000473, loss = 0.000322
grad_step = 000474, loss = 0.000321
grad_step = 000475, loss = 0.000321
grad_step = 000476, loss = 0.000320
grad_step = 000477, loss = 0.000319
grad_step = 000478, loss = 0.000319
grad_step = 000479, loss = 0.000318
grad_step = 000480, loss = 0.000318
grad_step = 000481, loss = 0.000317
grad_step = 000482, loss = 0.000316
grad_step = 000483, loss = 0.000316
grad_step = 000484, loss = 0.000315
grad_step = 000485, loss = 0.000314
grad_step = 000486, loss = 0.000314
grad_step = 000487, loss = 0.000313
grad_step = 000488, loss = 0.000313
grad_step = 000489, loss = 0.000312
grad_step = 000490, loss = 0.000311
grad_step = 000491, loss = 0.000311
grad_step = 000492, loss = 0.000310
grad_step = 000493, loss = 0.000310
grad_step = 000494, loss = 0.000309
grad_step = 000495, loss = 0.000309
grad_step = 000496, loss = 0.000309
grad_step = 000497, loss = 0.000309
grad_step = 000498, loss = 0.000309
grad_step = 000499, loss = 0.000309
grad_step = 000500, loss = 0.000309
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000310
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
[[0.8566158  0.8761694  0.9459253  0.9498582  1.0091347 ]
 [0.84382164 0.9021661  0.94597155 1.0108639  1.0029191 ]
 [0.8989258  0.9230285  1.004299   0.9905002  0.96234775]
 [0.9286922  0.99295086 0.9819683  0.9450486  0.90902257]
 [0.9936392  0.99008375 0.9628942  0.90605843 0.86421824]
 [0.99664575 0.9652032  0.9287433  0.8594072  0.8701235 ]
 [0.93542373 0.9201868  0.85960925 0.84488106 0.8206903 ]
 [0.91367126 0.84922975 0.84876394 0.8106215  0.8463515 ]
 [0.8333055  0.829087   0.8046764  0.8323649  0.86298704]
 [0.82536936 0.80346984 0.81810474 0.8455665  0.8706118 ]
 [0.82507277 0.83295316 0.8464692  0.83024704 0.92070997]
 [0.8016344  0.855356   0.8532772  0.91111517 0.93227804]
 [0.84940267 0.86871547 0.94434625 0.9507328  1.0050361 ]
 [0.8512236  0.9154027  0.9550725  1.011571   0.9838363 ]
 [0.9111922  0.9354092  1.0088882  0.9853723  0.938156  ]
 [0.9414314  1.0027062  0.9779361  0.92960286 0.8843123 ]
 [0.9996822  0.9851727  0.94614965 0.8781402  0.8492295 ]
 [0.9846406  0.94526297 0.90331376 0.83430576 0.85179895]
 [0.9167384  0.90077555 0.83866155 0.8379866  0.81284046]
 [0.9113891  0.85045266 0.84387696 0.8163663  0.84127915]
 [0.8468741  0.8347293  0.810534   0.84015095 0.86560786]
 [0.83941895 0.81407446 0.8220774  0.8564366  0.87283695]
 [0.8462587  0.8393658  0.8565141  0.8342631  0.9259306 ]
 [0.81315464 0.8643176  0.85819584 0.9160309  0.93546796]
 [0.8618882  0.8831281  0.9483058  0.95334864 1.0130364 ]
 [0.8510742  0.90951616 0.9533004  1.0198247  1.0140035 ]
 [0.9086144  0.9343096  1.0166702  1.0051705  0.9718604 ]
 [0.94140553 1.0068148  0.9966322  0.9594945  0.9170962 ]
 [1.0048438  1.0034134  0.97601664 0.91537535 0.8722518 ]
 [1.0075145  0.97558063 0.9380369  0.8645787  0.8771452 ]
 [0.9416595  0.9258863  0.86465836 0.8500904  0.8289986 ]]

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
[master d832f45] ml_store
 1 file changed, 1123 insertions(+)
To github.com:arita37/mlmodels_store.git
   cedfad6..d832f45  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_classifier.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_classifier.py", line 522, in <module>
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_classifier.py", line 418, in get_params
    cf = json.load(open(data_path, mode='r'))
FileNotFoundError: [Errno 2] No such file or directory: 'model_tch/transformer_classifier.json'

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
[master 9851dc9] ml_store
 1 file changed, 38 insertions(+)
To github.com:arita37/mlmodels_store.git
   d832f45..9851dc9  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//matchzoo_models.py 

  #### Loading params   ############################################## 

  {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'dataset_pars': {'data_pack': '', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'resample': True, 'sort': False, 'callbacks': 'PADDING'}, 'dataloader': '', 'dataloader_pars': {'device': 'cpu', 'dataset': 'None', 'stage': 'train', 'callback': 'PADDING'}, 'preprocess': {'train': {'transform': True, 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'stage': 'train', 'resample': True, 'sort': False, 'dataloader_callback': 'PADDING'}, 'test': {'transform': True, 'batch_size': 20, 'stage': 'dev', 'dataloader_callback': 'PADDING'}}} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Loading dataset   ############################################# 

  #### Model init   ################################################## 
  0%|          | 0/231508 [00:00<?, ?B/s]100%|██████████| 231508/231508 [00:00<00:00, 10027830.99B/s]
  0%|          | 0/433 [00:00<?, ?B/s]100%|██████████| 433/433 [00:00<00:00, 411448.49B/s]
  0%|          | 0/440473133 [00:00<?, ?B/s]  1%|          | 4151296/440473133 [00:00<00:10, 41509238.89B/s]  2%|▏         | 10022912/440473133 [00:00<00:09, 45508758.90B/s]  3%|▎         | 15285248/440473133 [00:00<00:08, 47431485.27B/s]  5%|▍         | 20698112/440473133 [00:00<00:08, 49259895.13B/s]  6%|▌         | 26127360/440473133 [00:00<00:08, 50667363.30B/s]  7%|▋         | 31397888/440473133 [00:00<00:07, 51259657.11B/s]  8%|▊         | 36541440/440473133 [00:00<00:07, 51306629.76B/s]  9%|▉         | 41597952/440473133 [00:00<00:07, 51079666.68B/s] 11%|█         | 47130624/440473133 [00:00<00:07, 52282164.50B/s] 12%|█▏        | 53408768/440473133 [00:01<00:07, 55043514.37B/s] 14%|█▎        | 59590656/440473133 [00:01<00:06, 56913620.88B/s] 15%|█▍        | 65822720/440473133 [00:01<00:06, 58433653.85B/s] 16%|█▋        | 71658496/440473133 [00:01<00:06, 57547116.61B/s] 18%|█▊        | 77445120/440473133 [00:01<00:06, 57633876.51B/s] 19%|█▉        | 83207168/440473133 [00:01<00:06, 57617343.17B/s] 20%|██        | 88968192/440473133 [00:01<00:06, 56670225.00B/s] 21%|██▏       | 94638080/440473133 [00:01<00:06, 56335167.36B/s] 23%|██▎       | 100274176/440473133 [00:01<00:06, 55502336.88B/s] 24%|██▍       | 105829376/440473133 [00:01<00:06, 54692155.01B/s] 25%|██▌       | 111619072/440473133 [00:02<00:05, 55615652.82B/s] 27%|██▋       | 117188608/440473133 [00:02<00:05, 54633437.57B/s] 28%|██▊       | 122763264/440473133 [00:02<00:05, 54956687.06B/s] 29%|██▉       | 128266240/440473133 [00:02<00:05, 54607039.74B/s] 30%|███       | 133756928/440473133 [00:02<00:05, 54692610.17B/s] 32%|███▏      | 139295744/440473133 [00:02<00:05, 54898204.07B/s] 33%|███▎      | 145166336/440473133 [00:02<00:05, 55986762.47B/s] 34%|███▍      | 151078912/440473133 [00:02<00:05, 56892522.93B/s] 36%|███▌      | 156905472/440473133 [00:02<00:04, 57296691.75B/s] 37%|███▋      | 162954240/440473133 [00:02<00:04, 58217804.15B/s] 38%|███▊      | 168984576/440473133 [00:03<00:04, 58826498.79B/s] 40%|███▉      | 175151104/440473133 [00:03<00:04, 59646524.95B/s] 41%|████      | 181124096/440473133 [00:03<00:04, 58747093.82B/s] 42%|████▏     | 187008000/440473133 [00:03<00:04, 58086039.57B/s] 44%|████▍     | 192824320/440473133 [00:03<00:04, 56851137.96B/s] 45%|████▌     | 198522880/440473133 [00:03<00:04, 56884683.25B/s] 46%|████▋     | 204440576/440473133 [00:03<00:04, 57550794.08B/s] 48%|████▊     | 210203648/440473133 [00:03<00:04, 57269482.76B/s] 49%|████▉     | 215937024/440473133 [00:03<00:03, 56675551.36B/s] 50%|█████     | 222190592/440473133 [00:03<00:03, 58314366.08B/s] 52%|█████▏    | 228178944/440473133 [00:04<00:03, 58775105.34B/s] 53%|█████▎    | 234562560/440473133 [00:04<00:03, 60205579.11B/s] 55%|█████▍    | 240600064/440473133 [00:04<00:03, 57699461.88B/s] 56%|█████▌    | 246404096/440473133 [00:04<00:03, 56685517.80B/s] 57%|█████▋    | 252100608/440473133 [00:04<00:03, 56030980.93B/s] 59%|█████▊    | 257725440/440473133 [00:04<00:03, 55893522.79B/s] 60%|█████▉    | 263567360/440473133 [00:04<00:03, 56623416.28B/s] 61%|██████    | 269243392/440473133 [00:04<00:03, 54795815.88B/s] 62%|██████▏   | 274745344/440473133 [00:04<00:03, 54187807.99B/s] 64%|██████▎   | 280181760/440473133 [00:05<00:03, 52999922.42B/s] 65%|██████▍   | 285878272/440473133 [00:05<00:02, 54127760.28B/s] 66%|██████▌   | 291617792/440473133 [00:05<00:02, 55068190.61B/s] 68%|██████▊   | 297493504/440473133 [00:05<00:02, 56124739.76B/s] 69%|██████▉   | 303123456/440473133 [00:05<00:02, 55365334.63B/s] 70%|███████   | 309019648/440473133 [00:05<00:02, 56396624.13B/s] 72%|███████▏  | 315079680/440473133 [00:05<00:02, 57593371.00B/s] 73%|███████▎  | 321092608/440473133 [00:05<00:02, 58331129.24B/s] 74%|███████▍  | 327201792/440473133 [00:05<00:01, 59132268.00B/s] 76%|███████▌  | 333127680/440473133 [00:05<00:01, 59009433.51B/s] 77%|███████▋  | 339204096/440473133 [00:06<00:01, 59524800.29B/s] 78%|███████▊  | 345207808/440473133 [00:06<00:01, 59677126.02B/s] 80%|███████▉  | 351246336/440473133 [00:06<00:01, 59883930.95B/s] 81%|████████  | 357238784/440473133 [00:06<00:01, 59185088.94B/s] 82%|████████▏ | 363162624/440473133 [00:06<00:01, 57705994.27B/s] 84%|████████▍ | 368971776/440473133 [00:06<00:01, 57819689.37B/s] 85%|████████▌ | 374762496/440473133 [00:06<00:01, 57515038.91B/s] 86%|████████▋ | 380520448/440473133 [00:06<00:01, 55137447.11B/s] 88%|████████▊ | 386059264/440473133 [00:06<00:00, 54749010.90B/s] 89%|████████▉ | 391827456/440473133 [00:06<00:00, 55596522.18B/s] 90%|█████████ | 397403136/440473133 [00:07<00:00, 54512157.51B/s] 91%|█████████▏| 402955264/440473133 [00:07<00:00, 54810224.22B/s] 93%|█████████▎| 408622080/440473133 [00:07<00:00, 55354237.17B/s] 94%|█████████▍| 414167040/440473133 [00:07<00:00, 54739114.74B/s] 95%|█████████▌| 419746816/440473133 [00:07<00:00, 55051355.10B/s] 97%|█████████▋| 425403392/440473133 [00:07<00:00, 55496914.17B/s] 98%|█████████▊| 431125504/440473133 [00:07<00:00, 56000437.52B/s] 99%|█████████▉| 436807680/440473133 [00:07<00:00, 56242156.24B/s]100%|██████████| 440473133/440473133 [00:07<00:00, 56333316.89B/s]Downloading data from https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip

   8192/7094233 [..............................] - ETA: 0s
2727936/7094233 [==========>...................] - ETA: 0s
7094272/7094233 [==============================] - 0s 0us/step

Processing text_left with encode:   0%|          | 0/2118 [00:00<?, ?it/s]Processing text_left with encode:   0%|          | 2/2118 [00:00<02:11, 16.13it/s]Processing text_left with encode:  23%|██▎       | 490/2118 [00:00<01:10, 23.00it/s]Processing text_left with encode:  43%|████▎     | 917/2118 [00:00<00:36, 32.79it/s]Processing text_left with encode:  59%|█████▉    | 1246/2118 [00:00<00:18, 46.64it/s]Processing text_left with encode:  80%|████████  | 1695/2118 [00:00<00:06, 66.33it/s]Processing text_left with encode: 100%|██████████| 2118/2118 [00:00<00:00, 3438.55it/s]
Processing text_right with encode:   0%|          | 0/18841 [00:00<?, ?it/s]Processing text_right with encode:   1%|          | 174/18841 [00:00<00:10, 1717.02it/s]Processing text_right with encode:   2%|▏         | 359/18841 [00:00<00:10, 1753.84it/s]Processing text_right with encode:   3%|▎         | 542/18841 [00:00<00:10, 1775.45it/s]Processing text_right with encode:   4%|▍         | 717/18841 [00:00<00:10, 1767.44it/s]Processing text_right with encode:   5%|▍         | 878/18841 [00:00<00:10, 1715.33it/s]Processing text_right with encode:   6%|▌         | 1046/18841 [00:00<00:10, 1701.83it/s]Processing text_right with encode:   6%|▋         | 1204/18841 [00:00<00:10, 1663.32it/s]Processing text_right with encode:   7%|▋         | 1389/18841 [00:00<00:10, 1710.50it/s]Processing text_right with encode:   8%|▊         | 1566/18841 [00:00<00:10, 1726.28it/s]Processing text_right with encode:   9%|▉         | 1748/18841 [00:01<00:09, 1750.78it/s]Processing text_right with encode:  10%|█         | 1919/18841 [00:01<00:09, 1730.75it/s]Processing text_right with encode:  11%|█         | 2097/18841 [00:01<00:09, 1741.90it/s]Processing text_right with encode:  12%|█▏        | 2289/18841 [00:01<00:09, 1791.31it/s]Processing text_right with encode:  13%|█▎        | 2467/18841 [00:01<00:09, 1744.35it/s]Processing text_right with encode:  14%|█▍        | 2641/18841 [00:01<00:09, 1686.83it/s]Processing text_right with encode:  15%|█▌        | 2827/18841 [00:01<00:09, 1734.00it/s]Processing text_right with encode:  16%|█▌        | 3001/18841 [00:01<00:09, 1711.77it/s]Processing text_right with encode:  17%|█▋        | 3185/18841 [00:01<00:08, 1747.75it/s]Processing text_right with encode:  18%|█▊        | 3365/18841 [00:01<00:08, 1760.62it/s]Processing text_right with encode:  19%|█▉        | 3542/18841 [00:02<00:08, 1746.07it/s]Processing text_right with encode:  20%|█▉        | 3717/18841 [00:02<00:08, 1743.63it/s]Processing text_right with encode:  21%|██        | 3901/18841 [00:02<00:08, 1768.44it/s]Processing text_right with encode:  22%|██▏       | 4079/18841 [00:02<00:08, 1769.22it/s]Processing text_right with encode:  23%|██▎       | 4257/18841 [00:02<00:08, 1739.23it/s]Processing text_right with encode:  24%|██▎       | 4434/18841 [00:02<00:08, 1747.57it/s]Processing text_right with encode:  24%|██▍       | 4609/18841 [00:02<00:08, 1740.57it/s]Processing text_right with encode:  25%|██▌       | 4784/18841 [00:02<00:08, 1703.10it/s]Processing text_right with encode:  26%|██▋       | 4981/18841 [00:02<00:07, 1774.54it/s]Processing text_right with encode:  27%|██▋       | 5160/18841 [00:02<00:07, 1778.22it/s]Processing text_right with encode:  28%|██▊       | 5346/18841 [00:03<00:07, 1798.52it/s]Processing text_right with encode:  29%|██▉       | 5541/18841 [00:03<00:07, 1836.78it/s]Processing text_right with encode:  30%|███       | 5726/18841 [00:03<00:07, 1809.98it/s]Processing text_right with encode:  31%|███▏      | 5911/18841 [00:03<00:07, 1818.78it/s]Processing text_right with encode:  32%|███▏      | 6097/18841 [00:03<00:06, 1826.01it/s]Processing text_right with encode:  33%|███▎      | 6280/18841 [00:03<00:06, 1795.88it/s]Processing text_right with encode:  34%|███▍      | 6474/18841 [00:03<00:06, 1835.50it/s]Processing text_right with encode:  35%|███▌      | 6674/18841 [00:03<00:06, 1881.02it/s]Processing text_right with encode:  36%|███▋      | 6864/18841 [00:03<00:06, 1885.56it/s]Processing text_right with encode:  37%|███▋      | 7053/18841 [00:03<00:06, 1867.03it/s]Processing text_right with encode:  38%|███▊      | 7241/18841 [00:04<00:06, 1834.29it/s]Processing text_right with encode:  39%|███▉      | 7428/18841 [00:04<00:06, 1843.54it/s]Processing text_right with encode:  40%|████      | 7613/18841 [00:04<00:06, 1813.85it/s]Processing text_right with encode:  41%|████▏     | 7805/18841 [00:04<00:05, 1842.94it/s]Processing text_right with encode:  42%|████▏     | 7990/18841 [00:04<00:05, 1836.59it/s]Processing text_right with encode:  43%|████▎     | 8174/18841 [00:04<00:05, 1781.79it/s]Processing text_right with encode:  44%|████▍     | 8357/18841 [00:04<00:05, 1794.69it/s]Processing text_right with encode:  45%|████▌     | 8537/18841 [00:04<00:05, 1767.11it/s]Processing text_right with encode:  46%|████▋     | 8733/18841 [00:04<00:05, 1819.75it/s]Processing text_right with encode:  47%|████▋     | 8917/18841 [00:05<00:05, 1825.15it/s]Processing text_right with encode:  48%|████▊     | 9100/18841 [00:05<00:05, 1756.69it/s]Processing text_right with encode:  49%|████▉     | 9284/18841 [00:05<00:05, 1777.89it/s]Processing text_right with encode:  50%|█████     | 9463/18841 [00:05<00:05, 1757.45it/s]Processing text_right with encode:  51%|█████     | 9653/18841 [00:05<00:05, 1796.74it/s]Processing text_right with encode:  52%|█████▏    | 9834/18841 [00:05<00:05, 1778.52it/s]Processing text_right with encode:  53%|█████▎    | 10042/18841 [00:05<00:04, 1857.55it/s]Processing text_right with encode:  54%|█████▍    | 10229/18841 [00:05<00:04, 1826.79it/s]Processing text_right with encode:  55%|█████▌    | 10442/18841 [00:05<00:04, 1906.44it/s]Processing text_right with encode:  56%|█████▋    | 10635/18841 [00:05<00:04, 1890.40it/s]Processing text_right with encode:  57%|█████▋    | 10826/18841 [00:06<00:04, 1883.70it/s]Processing text_right with encode:  58%|█████▊    | 11016/18841 [00:06<00:04, 1824.16it/s]Processing text_right with encode:  59%|█████▉    | 11200/18841 [00:06<00:04, 1826.60it/s]Processing text_right with encode:  60%|██████    | 11384/18841 [00:06<00:04, 1745.49it/s]Processing text_right with encode:  61%|██████▏   | 11567/18841 [00:06<00:04, 1767.79it/s]Processing text_right with encode:  62%|██████▏   | 11756/18841 [00:06<00:03, 1799.52it/s]Processing text_right with encode:  63%|██████▎   | 11937/18841 [00:06<00:03, 1787.40it/s]Processing text_right with encode:  64%|██████▍   | 12117/18841 [00:06<00:03, 1721.82it/s]Processing text_right with encode:  65%|██████▌   | 12308/18841 [00:06<00:03, 1773.21it/s]Processing text_right with encode:  66%|██████▋   | 12493/18841 [00:06<00:03, 1792.41it/s]Processing text_right with encode:  67%|██████▋   | 12674/18841 [00:07<00:03, 1793.21it/s]Processing text_right with encode:  68%|██████▊   | 12854/18841 [00:07<00:03, 1788.97it/s]Processing text_right with encode:  69%|██████▉   | 13034/18841 [00:07<00:03, 1788.57it/s]Processing text_right with encode:  70%|███████   | 13214/18841 [00:07<00:03, 1786.34it/s]Processing text_right with encode:  71%|███████   | 13417/18841 [00:07<00:02, 1852.92it/s]Processing text_right with encode:  72%|███████▏  | 13610/18841 [00:07<00:02, 1874.96it/s]Processing text_right with encode:  73%|███████▎  | 13799/18841 [00:07<00:02, 1876.67it/s]Processing text_right with encode:  74%|███████▍  | 13989/18841 [00:07<00:02, 1876.62it/s]Processing text_right with encode:  75%|███████▌  | 14177/18841 [00:07<00:02, 1859.37it/s]Processing text_right with encode:  76%|███████▌  | 14364/18841 [00:07<00:02, 1845.48it/s]Processing text_right with encode:  77%|███████▋  | 14549/18841 [00:08<00:02, 1818.45it/s]Processing text_right with encode:  78%|███████▊  | 14762/18841 [00:08<00:02, 1901.18it/s]Processing text_right with encode:  79%|███████▉  | 14954/18841 [00:08<00:02, 1894.54it/s]Processing text_right with encode:  80%|████████  | 15145/18841 [00:08<00:01, 1883.35it/s]Processing text_right with encode:  81%|████████▏ | 15335/18841 [00:08<00:01, 1887.06it/s]Processing text_right with encode:  82%|████████▏ | 15530/18841 [00:08<00:01, 1903.61it/s]Processing text_right with encode:  83%|████████▎ | 15726/18841 [00:08<00:01, 1918.67it/s]Processing text_right with encode:  84%|████████▍ | 15919/18841 [00:08<00:01, 1903.27it/s]Processing text_right with encode:  86%|████████▌ | 16110/18841 [00:08<00:01, 1892.07it/s]Processing text_right with encode:  87%|████████▋ | 16305/18841 [00:09<00:01, 1907.93it/s]Processing text_right with encode:  88%|████████▊ | 16508/18841 [00:09<00:01, 1939.49it/s]Processing text_right with encode:  89%|████████▊ | 16703/18841 [00:09<00:01, 1878.65it/s]Processing text_right with encode:  90%|████████▉ | 16892/18841 [00:09<00:01, 1793.67it/s]Processing text_right with encode:  91%|█████████ | 17073/18841 [00:09<00:00, 1780.31it/s]Processing text_right with encode:  92%|█████████▏| 17260/18841 [00:09<00:00, 1805.77it/s]Processing text_right with encode:  93%|█████████▎| 17446/18841 [00:09<00:00, 1820.62it/s]Processing text_right with encode:  94%|█████████▎| 17636/18841 [00:09<00:00, 1843.64it/s]Processing text_right with encode:  95%|█████████▍| 17832/18841 [00:09<00:00, 1876.73it/s]Processing text_right with encode:  96%|█████████▌| 18035/18841 [00:09<00:00, 1918.40it/s]Processing text_right with encode:  97%|█████████▋| 18230/18841 [00:10<00:00, 1927.26it/s]Processing text_right with encode:  98%|█████████▊| 18441/18841 [00:10<00:00, 1976.54it/s]Processing text_right with encode:  99%|█████████▉| 18640/18841 [00:10<00:00, 1977.15it/s]Processing text_right with encode: 100%|█████████▉| 18839/18841 [00:10<00:00, 1925.13it/s]Processing text_right with encode: 100%|██████████| 18841/18841 [00:10<00:00, 1818.46it/s]
Processing length_left with len:   0%|          | 0/2118 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 2118/2118 [00:00<00:00, 594057.50it/s]
Processing length_right with len:   0%|          | 0/18841 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 18841/18841 [00:00<00:00, 936524.59it/s]
Processing text_left with encode:   0%|          | 0/633 [00:00<?, ?it/s]Processing text_left with encode:  76%|███████▌  | 481/633 [00:00<00:00, 4797.71it/s]Processing text_left with encode: 100%|██████████| 633/633 [00:00<00:00, 4563.94it/s]
Processing text_right with encode:   0%|          | 0/5961 [00:00<?, ?it/s]Processing text_right with encode:   3%|▎         | 178/5961 [00:00<00:03, 1776.57it/s]Processing text_right with encode:   6%|▌         | 354/5961 [00:00<00:03, 1770.42it/s]Processing text_right with encode:   9%|▊         | 521/5961 [00:00<00:03, 1737.45it/s]Processing text_right with encode:  12%|█▏        | 700/5961 [00:00<00:03, 1751.64it/s]Processing text_right with encode:  15%|█▍        | 884/5961 [00:00<00:02, 1776.86it/s]Processing text_right with encode:  18%|█▊        | 1060/5961 [00:00<00:02, 1768.11it/s]Processing text_right with encode:  21%|██        | 1241/5961 [00:00<00:02, 1778.61it/s]Processing text_right with encode:  24%|██▎       | 1414/5961 [00:00<00:02, 1761.15it/s]Processing text_right with encode:  27%|██▋       | 1587/5961 [00:00<00:02, 1749.67it/s]Processing text_right with encode:  29%|██▉       | 1755/5961 [00:01<00:02, 1706.29it/s]Processing text_right with encode:  32%|███▏      | 1923/5961 [00:01<00:02, 1696.70it/s]Processing text_right with encode:  35%|███▌      | 2101/5961 [00:01<00:02, 1720.08it/s]Processing text_right with encode:  39%|███▊      | 2298/5961 [00:01<00:02, 1785.26it/s]Processing text_right with encode:  42%|████▏     | 2476/5961 [00:01<00:01, 1779.74it/s]Processing text_right with encode:  45%|████▍     | 2668/5961 [00:01<00:01, 1818.94it/s]Processing text_right with encode:  48%|████▊     | 2873/5961 [00:01<00:01, 1881.52it/s]Processing text_right with encode:  51%|█████▏    | 3069/5961 [00:01<00:01, 1903.97it/s]Processing text_right with encode:  55%|█████▍    | 3261/5961 [00:01<00:01, 1906.75it/s]Processing text_right with encode:  58%|█████▊    | 3452/5961 [00:01<00:01, 1869.90it/s]Processing text_right with encode:  61%|██████    | 3640/5961 [00:02<00:01, 1868.76it/s]Processing text_right with encode:  64%|██████▍   | 3828/5961 [00:02<00:01, 1822.22it/s]Processing text_right with encode:  68%|██████▊   | 4040/5961 [00:02<00:01, 1902.18it/s]Processing text_right with encode:  71%|███████   | 4241/5961 [00:02<00:00, 1932.43it/s]Processing text_right with encode:  74%|███████▍  | 4436/5961 [00:02<00:00, 1934.12it/s]Processing text_right with encode:  78%|███████▊  | 4632/5961 [00:02<00:00, 1939.47it/s]Processing text_right with encode:  81%|████████  | 4827/5961 [00:02<00:00, 1843.85it/s]Processing text_right with encode:  84%|████████▍ | 5013/5961 [00:02<00:00, 1811.55it/s]Processing text_right with encode:  87%|████████▋ | 5202/5961 [00:02<00:00, 1834.15it/s]Processing text_right with encode:  90%|█████████ | 5387/5961 [00:02<00:00, 1779.23it/s]Processing text_right with encode:  93%|█████████▎| 5566/5961 [00:03<00:00, 1712.11it/s]Processing text_right with encode:  96%|█████████▋| 5739/5961 [00:03<00:00, 1670.62it/s]Processing text_right with encode:  99%|█████████▉| 5920/5961 [00:03<00:00, 1707.72it/s]Processing text_right with encode: 100%|██████████| 5961/5961 [00:03<00:00, 1803.36it/s]
Processing length_left with len:   0%|          | 0/633 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 633/633 [00:00<00:00, 404354.92it/s]
Processing length_right with len:   0%|          | 0/5961 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 5961/5961 [00:00<00:00, 734603.97it/s]
  #### Model  fit   ############################################# 

  0%|          | 0/102 [00:00<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:27<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:27<?, ?it/s, loss=0.899]Epoch 1/1:   1%|          | 1/102 [00:27<46:26, 27.59s/it, loss=0.899]Epoch 1/1:   1%|          | 1/102 [01:18<46:26, 27.59s/it, loss=0.899]Epoch 1/1:   1%|          | 1/102 [01:18<46:26, 27.59s/it, loss=1.034]Epoch 1/1:   2%|▏         | 2/102 [01:18<57:49, 34.69s/it, loss=1.034]Epoch 1/1:   2%|▏         | 2/102 [02:51<57:49, 34.69s/it, loss=1.034]Epoch 1/1:   2%|▏         | 2/102 [02:51<57:49, 34.69s/it, loss=0.906]Epoch 1/1:   3%|▎         | 3/102 [02:51<1:25:46, 51.99s/it, loss=0.906]Epoch 1/1:   3%|▎         | 3/102 [03:55<1:25:46, 51.99s/it, loss=0.906]Epoch 1/1:   3%|▎         | 3/102 [03:55<1:25:46, 51.99s/it, loss=0.823]Epoch 1/1:   4%|▍         | 4/102 [03:55<1:30:53, 55.65s/it, loss=0.823]Killed

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
   9851dc9..ae27348  master     -> origin/master
Updating 9851dc9..ae27348
Fast-forward
 deps.txt                                           |    6 +-
 error_list/20200518/list_log_json_20200518.md      | 1146 ++++++++++----------
 .../20200518/list_log_pullrequest_20200518.md      |    2 +-
 error_list/20200518/list_log_test_cli_20200518.md  |  364 +++----
 ...-09_203a72830f23a80c3dd3ee4f0d2ce62ae396cb03.py |  661 +++++++++++
 5 files changed, 1422 insertions(+), 757 deletions(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-18-05-09_203a72830f23a80c3dd3ee4f0d2ce62ae396cb03.py
[master ec13ceb] ml_store
 1 file changed, 69 insertions(+)
To github.com:arita37/mlmodels_store.git
   ae27348..ec13ceb  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py 

  #### Loading params   ############################################## 

  {'data_info': {'data_path': 'mlmodels/dataset/vision/MNIST', 'dataset': 'MNIST', 'data_type': 'tch_dataset', 'batch_size': 10, 'train': True}, 'preprocessors': [{'name': 'tch_dataset_start', 'uri': 'mlmodels/preprocess/generic.py::get_dataset_torch', 'args': {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True}}]} {'checkpointdir': 'ztest/model_tch/torchhub/restnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18/'} 

  #### Loading dataset   ############################################# 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f67f2f0dd90>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f67f2f0dd90>

  function with postional parmater data_info <function get_dataset_torch at 0x7f67f2f0dd90> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:09, 143410.87it/s] 81%|████████  | 7995392/9912422 [00:00<00:09, 204712.74it/s]9920512it [00:00, 42251957.42it/s]                           
0it [00:00, ?it/s]32768it [00:00, 524204.01it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 158311.72it/s]1654784it [00:00, 11017946.34it/s]                         
0it [00:00, ?it/s]8192it [00:00, 242048.40it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting mlmodels/dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting mlmodels/dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting mlmodels/dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting mlmodels/dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw
Processing...
Done!

  #### Model init, fit   ############################################# 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f67f2243ae8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f67f2243ae8>

  function with postional parmater data_info <function get_dataset_torch at 0x7f67f2243ae8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.001967287043730418 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.01093752646446228 	 Accuracy: 1
model saves at 1 accuracy
Train Epoch: 2 	 Loss: 0.0015582449436187743 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.009677712798118592 	 Accuracy: 1

  #### Predict   ##################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f67f22438c8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f67f22438c8>

  function with postional parmater data_info <function get_dataset_torch at 0x7f67f22438c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  #### metrics   ##################################################### 
None

  #### Plot   ######################################################## 

  #### Save  ######################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18//torch_model/ ['model.pb', 'torch_model_pars.pkl'] 

  #### Load   ######################################################## 
<__main__.Model object at 0x7f67f28b7898>

  #### Loading params   ############################################## 

  {'data_info': {'data_path': 'mlmodels/dataset/vision/MNIST', 'dataset': 'MNIST', 'data_type': 'tch_dataset', 'batch_size': 10, 'train': True}, 'preprocessors': [{'name': 'tch_dataset_start', 'uri': 'mlmodels.preprocess.generic::get_dataset_torch', 'args': {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}}, 'shuffle': True, 'download': True}}]} {'checkpointdir': 'ztest/model_tch/torchhub/restnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18/'} 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### Predict   ##################################################### 
img_01.png
torch_model

  #### metrics   ##################################################### 

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18//torch_model/ ['model.pb', 'torch_model_pars.pkl'] 
img_01.png
torch_model

Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Downloading: "https://github.com/facebookresearch/pytorch_GAN_zoo/archive/hub.zip" to /home/runner/.cache/torch/hub/hub.zip
Using cache found in /home/runner/.cache/torch/hub/facebookresearch_pytorch_GAN_zoo_hub
<__main__.Model object at 0x7f67f059b668>

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
[master ac800a9] ml_store
 2 files changed, 149 insertions(+), 5 deletions(-)
To github.com:arita37/mlmodels_store.git
   ec13ceb..ac800a9  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//03_nbeats_dataloader.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//03_nbeats_dataloader.py", line 9, in <module>
    from dataloader import DataLoader
ModuleNotFoundError: No module named 'dataloader'

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
[master 5a796dc] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   ac800a9..5a796dc  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_sentence.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Epoch:   0%|          | 0/1 [00:00<?, ?it/s]
Iteration:   0%|          | 0/29440 [00:00<?, ?it/s][A
Iteration:   0%|          | 1/29440 [00:11<94:53:47, 11.60s/it][A
Iteration:   0%|          | 2/29440 [00:26<102:25:10, 12.52s/it][A
Iteration:   0%|          | 3/29440 [01:12<184:32:43, 22.57s/it][A
Iteration:   0%|          | 4/29440 [01:48<218:41:55, 26.75s/it][A
Iteration:   0%|          | 5/29440 [02:21<232:57:56, 28.49s/it][A
Iteration:   0%|          | 6/29440 [03:33<339:52:32, 41.57s/it][A
Iteration:   0%|          | 7/29440 [04:35<390:05:27, 47.71s/it][A
Iteration:   0%|          | 8/29440 [06:26<544:39:51, 66.62s/it][A
Iteration:   0%|          | 9/29440 [07:24<523:12:43, 64.00s/it][A
Iteration:   0%|          | 10/29440 [11:31<973:53:08, 119.13s/it][A
Iteration:   0%|          | 11/29440 [12:33<831:46:26, 101.75s/it][A
Iteration:   0%|          | 12/29440 [13:47<764:55:58, 93.58s/it] [A
Iteration:   0%|          | 13/29440 [15:26<777:41:55, 95.14s/it][A
Iteration:   0%|          | 14/29440 [16:05<641:05:48, 78.43s/it][A
Iteration:   0%|          | 15/29440 [17:40<680:48:36, 83.29s/it][A
Iteration:   0%|          | 16/29440 [19:40<770:51:57, 94.31s/it][A
Iteration:   0%|          | 17/29440 [21:05<748:38:37, 91.60s/it][A
Iteration:   0%|          | 18/29440 [23:36<893:55:40, 109.38s/it][A
Iteration:   0%|          | 19/29440 [24:45<795:48:38, 97.38s/it] [A
Iteration:   0%|          | 20/29440 [30:21<1380:42:37, 168.95s/it][A
Iteration:   0%|          | 21/29440 [32:48<1324:44:29, 162.11s/it][A
Iteration:   0%|          | 22/29440 [34:20<1153:11:57, 141.12s/it][A
Iteration:   0%|          | 23/29440 [36:49<1173:46:30, 143.64s/it][AKilled

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
   5a796dc..211aa2b  master     -> origin/master
Updating 5a796dc..211aa2b
Fast-forward
 error_list/20200518/list_log_benchmark_20200518.md |  182 +-
 error_list/20200518/list_log_jupyter_20200518.md   | 1750 ++++++++++---------
 .../20200518/list_log_pullrequest_20200518.md      |    2 +-
 log_dataloader/log_dataloader.py                   |   38 +-
 log_jupyter/log_jupyter.py                         | 1792 ++++++++++----------
 5 files changed, 1863 insertions(+), 1901 deletions(-)
[master 632538e] ml_store
 1 file changed, 74 insertions(+)
To github.com:arita37/mlmodels_store.git
   211aa2b..632538e  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//pytorch_vae.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//pytorch_vae.py", line 34, in <module>
    "beta_vae": md.model.beta_vae,
AttributeError: module 'mlmodels.model_tch.raw.pytorch_vae' has no attribute 'model'

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
[master 46b0c0d] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   632538e..46b0c0d  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//pplm.py 
 Generating text ... 
= Prefix of sentence =
<|endoftext|>The potato

 Unperturbed generated text :

<|endoftext|>The potato-shaped, potato-eating insect of modern times (Ophiocordyceps elegans) has a unique ability to adapt quickly to a wide range of environments. It is able to survive in many different environments, including the Arctic, deserts

 Perturbed generated text :

<|endoftext|>The potato bomb is nothing new. It's been on the news a lot since 9/11. In fact, since the bombing in Paris last November, a bomb has been detonated in every major European country in the European Union.

The bomb in Brussels


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
   46b0c0d..f6b8958  master     -> origin/master
Updating 46b0c0d..f6b8958
Fast-forward
 error_list/20200518/list_log_benchmark_20200518.md |  182 +-
 error_list/20200518/list_log_jupyter_20200518.md   | 1747 ++++++++++----------
 .../20200518/list_log_pullrequest_20200518.md      |    2 +-
 3 files changed, 971 insertions(+), 960 deletions(-)
[master 9259c26] ml_store
 1 file changed, 53 insertions(+)
To github.com:arita37/mlmodels_store.git
   f6b8958..9259c26  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/textcnn/model', 'checkpointdir': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/textcnn//checkpoint/'}

  #### Loading dataset   ############################################# 
>>>>> load:  {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64}
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=533c6788c38d9384b8f4ae225dd982f31e647237d1bc9f6cc582561d7fc19b4a
  Stored in directory: /tmp/pip-ephem-wheel-cache-cogpirfa/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.2.5
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//textcnn.py", line 153, in create_tabular_dataset
    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/__init__.py", line 30, in load
    return util.load_model(name, **overrides)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/util.py", line 169, in load_model
    raise IOError(Errors.E050.format(name=name))
OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//textcnn.py", line 477, in <module>
    test( data_path="model_tch/textcnn.json", pars_choice = "test01" )
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//textcnn.py", line 442, in test
    Xtuple = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//textcnn.py", line 334, in get_dataset
    trainset, validset, vocab = create_tabular_dataset( data_pars['train_path'], data_pars['valid_path'], lang, pretrained_emb)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//textcnn.py", line 159, in create_tabular_dataset
    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)  
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/__init__.py", line 30, in load
    return util.load_model(name, **overrides)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/util.py", line 169, in load_model
    raise IOError(Errors.E050.format(name=name))
OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.

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
   9259c26..96c1eab  master     -> origin/master
Updating 9259c26..96c1eab
Fast-forward
 error_list/20200518/list_log_jupyter_20200518.md   | 1747 ++++++++++----------
 .../20200518/list_log_pullrequest_20200518.md      |    2 +-
 2 files changed, 874 insertions(+), 875 deletions(-)
[master 2573b23] ml_store
 2 files changed, 112 insertions(+)
To github.com:arita37/mlmodels_store.git
   96c1eab..2573b23  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//mlp.py 

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
[master 8715484] ml_store
 1 file changed, 32 insertions(+)
To github.com:arita37/mlmodels_store.git
   2573b23..8715484  master -> master
