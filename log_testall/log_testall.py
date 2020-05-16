
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
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
Already up to date.
[master 64de111] ml_store
 2 files changed, 62 insertions(+), 11298 deletions(-)
 rewrite log_testall/log_testall.py (99%)
To github.com:arita37/mlmodels_store.git
   1f2bff8..64de111  master -> master





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
From github.com:arita37/mlmodels_store
   64de111..e50d349  master     -> origin/master
Updating 64de111..e50d349
Fast-forward
 ...-10_76b7a81be9b27c2e92c4951280c0a8da664b997c.py | 623 +++++++++++++++++++++
 1 file changed, 623 insertions(+)
 create mode 100644 log_pullrequest/log_pr_2020-05-16-20-10_76b7a81be9b27c2e92c4951280c0a8da664b997c.py
[master dfe6d0a] ml_store
 1 file changed, 54 insertions(+)
To github.com:arita37/mlmodels_store.git
   e50d349..dfe6d0a  master -> master





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
[master 6dc4b70] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   dfe6d0a..6dc4b70  master -> master





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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         36          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-16 20:12:40.400731: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-16 20:12:40.405353: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-16 20:12:40.405508: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c7457eae50 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-16 20:12:40.405563: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
100/500 [=====>........................] - ETA: 1s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 1ms/sample - loss: 0.2500 - binary_crossentropy: 0.6930 - val_loss: 0.2499 - val_binary_crossentropy: 0.6928

  #### metrics   #################################################### 
{'MSE': 0.2497129679191662}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         36          sparse_feature_2[0][0]           
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
100/500 [=====>........................] - ETA: 1s - loss: 0.2688 - binary_crossentropy: 1.1244500/500 [==============================] - 1s 2ms/sample - loss: 0.2683 - binary_crossentropy: 1.0452 - val_loss: 0.2678 - val_binary_crossentropy: 0.9380

  #### metrics   #################################################### 
{'MSE': 0.26764240120831895}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         16          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_2[0][0]           
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
Total params: 602
Trainable params: 602
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 1s 2ms/sample - loss: 0.2498 - binary_crossentropy: 0.6927 - val_loss: 0.2504 - val_binary_crossentropy: 0.6939

  #### metrics   #################################################### 
{'MSE': 0.24971267994424562}

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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         16          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_2[0][0]           
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
Total params: 602
Trainable params: 602
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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         24          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_2[0][0]           
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
100/500 [=====>........................] - ETA: 2s - loss: 0.2524 - binary_crossentropy: 0.6987500/500 [==============================] - 1s 3ms/sample - loss: 0.2469 - binary_crossentropy: 0.6869 - val_loss: 0.2588 - val_binary_crossentropy: 0.7110

  #### metrics   #################################################### 
{'MSE': 0.2538572418732552}

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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         24          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_2[0][0]           
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_max[0][0]               
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
Total params: 188
Trainable params: 188
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2619 - binary_crossentropy: 0.9797500/500 [==============================] - 2s 3ms/sample - loss: 0.2612 - binary_crossentropy: 0.9501 - val_loss: 0.2657 - val_binary_crossentropy: 1.0390

  #### metrics   #################################################### 
{'MSE': 0.2633955924402059}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_max[0][0]               
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
Total params: 188
Trainable params: 188
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-16 20:13:58.838336: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 20:13:58.840679: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 20:13:58.846821: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-16 20:13:58.857484: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-16 20:13:58.859252: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-16 20:13:58.860955: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 20:13:58.862624: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2461 - val_binary_crossentropy: 0.6853
2020-05-16 20:14:00.062685: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 20:14:00.064655: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 20:14:00.068901: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-16 20:14:00.077740: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-16 20:14:00.079212: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-16 20:14:00.080560: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 20:14:00.081864: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24433190541419592}

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
2020-05-16 20:14:23.237500: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 20:14:23.238924: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 20:14:23.242986: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-16 20:14:23.249882: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-16 20:14:23.251046: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-16 20:14:23.252172: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 20:14:23.253201: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2484 - val_binary_crossentropy: 0.6900
2020-05-16 20:14:24.717677: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 20:14:24.719034: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 20:14:24.722000: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-16 20:14:24.728027: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-16 20:14:24.729036: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-16 20:14:24.729987: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 20:14:24.730930: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2477923343749661}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-16 20:14:58.157418: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 20:14:58.162567: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 20:14:58.177911: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-16 20:14:58.211622: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-16 20:14:58.217733: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-16 20:14:58.222661: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 20:14:58.228207: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.4232 - binary_crossentropy: 1.0514 - val_loss: 0.3123 - val_binary_crossentropy: 0.8366
2020-05-16 20:15:00.499941: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 20:15:00.504895: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 20:15:00.517002: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-16 20:15:00.542729: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-16 20:15:00.547134: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-16 20:15:00.551404: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 20:15:00.555614: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.39579831416632544}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
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
Total params: 730
Trainable params: 730
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2531 - binary_crossentropy: 0.6993500/500 [==============================] - 4s 8ms/sample - loss: 0.2533 - binary_crossentropy: 0.6998 - val_loss: 0.2555 - val_binary_crossentropy: 0.7043

  #### metrics   #################################################### 
{'MSE': 0.2544575212373744}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
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
Total params: 730
Trainable params: 730
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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 3, 2)         18          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 2)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 2)         4           sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         4           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         14          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         6           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         12          sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         2           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_5[0][0]           
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
Total params: 251
Trainable params: 251
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.4468 - binary_crossentropy: 4.9221500/500 [==============================] - 4s 9ms/sample - loss: 0.4410 - binary_crossentropy: 4.5313 - val_loss: 0.4387 - val_binary_crossentropy: 4.5380

  #### metrics   #################################################### 
{'MSE': 0.43798238693985386}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 3, 2)         18          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 2)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 2)         4           sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         4           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         14          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         6           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         12          sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         2           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_5[0][0]           
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
Total params: 251
Trainable params: 251
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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 1,844
Trainable params: 1,844
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.3123 - binary_crossentropy: 1.7357500/500 [==============================] - 5s 9ms/sample - loss: 0.2769 - binary_crossentropy: 1.4245 - val_loss: 0.2834 - val_binary_crossentropy: 1.4079

  #### metrics   #################################################### 
{'MSE': 0.28007788045271287}

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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 1,844
Trainable params: 1,844
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
regionsequence_mean (InputLayer [(None, 9)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 5)]          0                                            
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
region_10sparse_seq_emb_regions (None, 9, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 5, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         6           regionsequence_max[0][0]         
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
100/500 [=====>........................] - ETA: 8s - loss: 0.2641 - binary_crossentropy: 0.7216500/500 [==============================] - 6s 12ms/sample - loss: 0.2568 - binary_crossentropy: 0.7068 - val_loss: 0.2583 - val_binary_crossentropy: 0.7098

  #### metrics   #################################################### 
{'MSE': 0.2567794356744431}

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
regionsequence_mean (InputLayer [(None, 9)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 5)]          0                                            
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
region_10sparse_seq_emb_regions (None, 9, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 5, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         6           regionsequence_max[0][0]         
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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
Total params: 1,397
Trainable params: 1,397
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2705 - binary_crossentropy: 0.9991500/500 [==============================] - 6s 11ms/sample - loss: 0.2544 - binary_crossentropy: 0.8072 - val_loss: 0.2510 - val_binary_crossentropy: 0.7473

  #### metrics   #################################################### 
{'MSE': 0.2530161828777749}

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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
Total params: 1,397
Trainable params: 1,397
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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         36          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         8           hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 9, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 6, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 8, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 9, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 6, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 8, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 9, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 6, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 9, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 8, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 6, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 8, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           hash_11[0][0]                    
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
100/500 [=====>........................] - ETA: 8s - loss: 0.2587 - binary_crossentropy: 0.7108500/500 [==============================] - 6s 13ms/sample - loss: 0.2559 - binary_crossentropy: 0.7051 - val_loss: 0.2520 - val_binary_crossentropy: 0.6971

  #### metrics   #################################################### 
{'MSE': 0.2520898512712263}

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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         36          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         8           hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 9, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 6, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 8, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 9, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 6, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 8, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 9, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 6, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 9, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 8, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 6, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 8, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           hash_11[0][0]                    
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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
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
Total params: 485
Trainable params: 485
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2497 - binary_crossentropy: 0.6926500/500 [==============================] - 6s 12ms/sample - loss: 0.2501 - binary_crossentropy: 0.6933 - val_loss: 0.2499 - val_binary_crossentropy: 0.6929

  #### metrics   #################################################### 
{'MSE': 0.2498670614867302}

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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
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
Total params: 485
Trainable params: 485
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_125 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_194[0][0] 
                                                                 sequence_pooling_layer_195[0][0] 
                                                                 sequence_pooling_layer_196[0][0] 
                                                                 sequence_pooling_layer_197[0][0] 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
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
100/500 [=====>........................] - ETA: 8s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 7s 13ms/sample - loss: 0.2499 - binary_crossentropy: 0.6929 - val_loss: 0.2499 - val_binary_crossentropy: 0.6930

  #### metrics   #################################################### 
{'MSE': 0.24963563371091407}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_125 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_194[0][0] 
                                                                 sequence_pooling_layer_195[0][0] 
                                                                 sequence_pooling_layer_196[0][0] 
                                                                 sequence_pooling_layer_197[0][0] 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_47 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_90 (Concatenate)    (None, 1, 20)        0           no_mask_130[0][0]                
                                                                 no_mask_130[1][0]                
                                                                 no_mask_130[2][0]                
                                                                 no_mask_130[3][0]                
                                                                 no_mask_130[4][0]                
__________________________________________________________________________________________________
no_mask_131 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
100/500 [=====>........................] - ETA: 8s - loss: 0.2486 - binary_crossentropy: 0.6902500/500 [==============================] - 7s 13ms/sample - loss: 0.2539 - binary_crossentropy: 0.7534 - val_loss: 0.2523 - val_binary_crossentropy: 0.7499

  #### metrics   #################################################### 
{'MSE': 0.25261672001223945}

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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_47 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_90 (Concatenate)    (None, 1, 20)        0           no_mask_130[0][0]                
                                                                 no_mask_130[1][0]                
                                                                 no_mask_130[2][0]                
                                                                 no_mask_130[3][0]                
                                                                 no_mask_130[4][0]                
__________________________________________________________________________________________________
no_mask_131 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
   6dc4b70..201dcb8  master     -> origin/master
Updating 6dc4b70..201dcb8
Fast-forward
 error_list/20200516/list_log_json_20200516.md | 1146 ++++++++++++-------------
 1 file changed, 573 insertions(+), 573 deletions(-)
[master 9d0f7d5] ml_store
 1 file changed, 5668 insertions(+)
To github.com:arita37/mlmodels_store.git
   201dcb8..9d0f7d5  master -> master





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
[master f8b3a24] ml_store
 1 file changed, 51 insertions(+)
To github.com:arita37/mlmodels_store.git
   9d0f7d5..f8b3a24  master -> master





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
[master 5882418] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   f8b3a24..5882418  master -> master





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
[master a7d3cd7] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   5882418..a7d3cd7  master -> master





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

2020-05-16 20:27:55.143939: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-16 20:27:55.149133: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-16 20:27:55.149310: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a29826f2e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-16 20:27:55.149323: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 8s - loss: 1.3841
256/354 [====================>.........] - ETA: 3s - loss: 1.2460
354/354 [==============================] - 15s 43ms/step - loss: 1.2913 - val_loss: 2.1865

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
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
Already up to date.
[master 6acf1a6] ml_store
 1 file changed, 151 insertions(+)
To github.com:arita37/mlmodels_store.git
   a7d3cd7..6acf1a6  master -> master





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
[master a606da8] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   6acf1a6..a606da8  master -> master





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
[master ae0c86c] ml_store
 1 file changed, 45 insertions(+)
To github.com:arita37/mlmodels_store.git
   a606da8..ae0c86c  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2514944/17464789 [===>..........................] - ETA: 0s
 7479296/17464789 [===========>..................] - ETA: 0s
12419072/17464789 [====================>.........] - ETA: 0s
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
2020-05-16 20:28:59.207728: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-16 20:28:59.212182: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-16 20:28:59.212364: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a1345818e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-16 20:28:59.212381: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 13s - loss: 7.1760 - accuracy: 0.5320
 2000/25000 [=>............................] - ETA: 9s - loss: 7.5056 - accuracy: 0.5105 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5644 - accuracy: 0.5067
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6015 - accuracy: 0.5042
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7188 - accuracy: 0.4966
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6385 - accuracy: 0.5018
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6579 - accuracy: 0.5006
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6398 - accuracy: 0.5017
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6683 - accuracy: 0.4999
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6896 - accuracy: 0.4985
11000/25000 [============>.................] - ETA: 4s - loss: 7.7112 - accuracy: 0.4971
12000/25000 [=============>................] - ETA: 4s - loss: 7.7407 - accuracy: 0.4952
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7079 - accuracy: 0.4973
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7082 - accuracy: 0.4973
15000/25000 [=================>............] - ETA: 3s - loss: 7.7167 - accuracy: 0.4967
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6973 - accuracy: 0.4980
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6657 - accuracy: 0.5001
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6726 - accuracy: 0.4996
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6884 - accuracy: 0.4986
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6743 - accuracy: 0.4995
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6725 - accuracy: 0.4996
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6799 - accuracy: 0.4991
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6692 - accuracy: 0.4998
25000/25000 [==============================] - 10s 383us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7f6623becf28>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f65f636a438> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.9273 - accuracy: 0.4830
 2000/25000 [=>............................] - ETA: 9s - loss: 7.9733 - accuracy: 0.4800 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.8097 - accuracy: 0.4907
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.8046 - accuracy: 0.4910
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.8322 - accuracy: 0.4892
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7688 - accuracy: 0.4933
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7280 - accuracy: 0.4960
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6858 - accuracy: 0.4988
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6973 - accuracy: 0.4980
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7203 - accuracy: 0.4965
11000/25000 [============>.................] - ETA: 4s - loss: 7.7057 - accuracy: 0.4975
12000/25000 [=============>................] - ETA: 4s - loss: 7.7062 - accuracy: 0.4974
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7232 - accuracy: 0.4963
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7269 - accuracy: 0.4961
15000/25000 [=================>............] - ETA: 3s - loss: 7.6932 - accuracy: 0.4983
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6839 - accuracy: 0.4989
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6774 - accuracy: 0.4993
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6820 - accuracy: 0.4990
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6941 - accuracy: 0.4982
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6827 - accuracy: 0.4990
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6790 - accuracy: 0.4992
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6799 - accuracy: 0.4991
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6753 - accuracy: 0.4994
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6596 - accuracy: 0.5005
25000/25000 [==============================] - 10s 383us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 13s - loss: 7.3600 - accuracy: 0.5200
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8046 - accuracy: 0.4910 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7382 - accuracy: 0.4953
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6666 - accuracy: 0.5000
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5746 - accuracy: 0.5060
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5644 - accuracy: 0.5067
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.5330 - accuracy: 0.5087
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5861 - accuracy: 0.5052
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5610 - accuracy: 0.5069
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5716 - accuracy: 0.5062
11000/25000 [============>.................] - ETA: 4s - loss: 7.5997 - accuracy: 0.5044
12000/25000 [=============>................] - ETA: 4s - loss: 7.6513 - accuracy: 0.5010
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6501 - accuracy: 0.5011
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6633 - accuracy: 0.5002
15000/25000 [=================>............] - ETA: 3s - loss: 7.6625 - accuracy: 0.5003
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6532 - accuracy: 0.5009
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6702 - accuracy: 0.4998
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6513 - accuracy: 0.5010
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6577 - accuracy: 0.5006
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6559 - accuracy: 0.5007
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6520 - accuracy: 0.5010
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6576 - accuracy: 0.5006
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6580 - accuracy: 0.5006
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6647 - accuracy: 0.5001
25000/25000 [==============================] - 10s 383us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
   ae0c86c..5e38c5f  master     -> origin/master
Updating ae0c86c..5e38c5f
Fast-forward
 error_list/20200516/list_log_benchmark_20200516.md | 182 ++++++++++-----------
 1 file changed, 86 insertions(+), 96 deletions(-)
[master 2da8211] ml_store
 1 file changed, 323 insertions(+)
To github.com:arita37/mlmodels_store.git
   5e38c5f..2da8211  master -> master





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

13/13 [==============================] - 2s 123ms/step - loss: nan
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
[master 68ba0e0] ml_store
 1 file changed, 126 insertions(+)
To github.com:arita37/mlmodels_store.git
   2da8211..68ba0e0  master -> master





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
 3366912/11490434 [=======>......................] - ETA: 0s
 8265728/11490434 [====================>.........] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:37 - loss: 2.3113 - categorical_accuracy: 0.1875
   64/60000 [..............................] - ETA: 4:50 - loss: 2.2786 - categorical_accuracy: 0.1875
   96/60000 [..............................] - ETA: 3:49 - loss: 2.2382 - categorical_accuracy: 0.2188
  128/60000 [..............................] - ETA: 3:20 - loss: 2.2443 - categorical_accuracy: 0.2109
  160/60000 [..............................] - ETA: 3:02 - loss: 2.1986 - categorical_accuracy: 0.2313
  192/60000 [..............................] - ETA: 2:49 - loss: 2.2135 - categorical_accuracy: 0.2240
  224/60000 [..............................] - ETA: 2:40 - loss: 2.1817 - categorical_accuracy: 0.2411
  256/60000 [..............................] - ETA: 2:34 - loss: 2.1372 - categorical_accuracy: 0.2695
  288/60000 [..............................] - ETA: 2:28 - loss: 2.0867 - categorical_accuracy: 0.2847
  320/60000 [..............................] - ETA: 2:25 - loss: 2.0291 - categorical_accuracy: 0.3094
  352/60000 [..............................] - ETA: 2:21 - loss: 2.0235 - categorical_accuracy: 0.3153
  384/60000 [..............................] - ETA: 2:19 - loss: 2.0100 - categorical_accuracy: 0.3203
  416/60000 [..............................] - ETA: 2:17 - loss: 1.9871 - categorical_accuracy: 0.3293
  448/60000 [..............................] - ETA: 2:15 - loss: 1.9740 - categorical_accuracy: 0.3393
  480/60000 [..............................] - ETA: 2:13 - loss: 1.9517 - categorical_accuracy: 0.3396
  512/60000 [..............................] - ETA: 2:12 - loss: 1.9133 - categorical_accuracy: 0.3516
  544/60000 [..............................] - ETA: 2:10 - loss: 1.8766 - categorical_accuracy: 0.3621
  576/60000 [..............................] - ETA: 2:09 - loss: 1.8546 - categorical_accuracy: 0.3628
  608/60000 [..............................] - ETA: 2:08 - loss: 1.8423 - categorical_accuracy: 0.3651
  640/60000 [..............................] - ETA: 2:07 - loss: 1.8019 - categorical_accuracy: 0.3844
  672/60000 [..............................] - ETA: 2:06 - loss: 1.7754 - categorical_accuracy: 0.3958
  704/60000 [..............................] - ETA: 2:05 - loss: 1.7404 - categorical_accuracy: 0.4105
  736/60000 [..............................] - ETA: 2:04 - loss: 1.7138 - categorical_accuracy: 0.4226
  768/60000 [..............................] - ETA: 2:03 - loss: 1.6848 - categorical_accuracy: 0.4323
  800/60000 [..............................] - ETA: 2:02 - loss: 1.6616 - categorical_accuracy: 0.4425
  832/60000 [..............................] - ETA: 2:02 - loss: 1.6477 - categorical_accuracy: 0.4471
  864/60000 [..............................] - ETA: 2:01 - loss: 1.6225 - categorical_accuracy: 0.4537
  896/60000 [..............................] - ETA: 2:00 - loss: 1.5936 - categorical_accuracy: 0.4609
  928/60000 [..............................] - ETA: 2:00 - loss: 1.5621 - categorical_accuracy: 0.4752
  960/60000 [..............................] - ETA: 2:00 - loss: 1.5388 - categorical_accuracy: 0.4844
  992/60000 [..............................] - ETA: 2:00 - loss: 1.5076 - categorical_accuracy: 0.4950
 1024/60000 [..............................] - ETA: 1:59 - loss: 1.4917 - categorical_accuracy: 0.4990
 1056/60000 [..............................] - ETA: 1:59 - loss: 1.4798 - categorical_accuracy: 0.5009
 1088/60000 [..............................] - ETA: 1:58 - loss: 1.4642 - categorical_accuracy: 0.5092
 1120/60000 [..............................] - ETA: 1:58 - loss: 1.4385 - categorical_accuracy: 0.5188
 1152/60000 [..............................] - ETA: 1:58 - loss: 1.4193 - categorical_accuracy: 0.5269
 1184/60000 [..............................] - ETA: 1:58 - loss: 1.4085 - categorical_accuracy: 0.5287
 1216/60000 [..............................] - ETA: 1:57 - loss: 1.3895 - categorical_accuracy: 0.5354
 1248/60000 [..............................] - ETA: 1:57 - loss: 1.3702 - categorical_accuracy: 0.5425
 1280/60000 [..............................] - ETA: 1:57 - loss: 1.3488 - categorical_accuracy: 0.5477
 1312/60000 [..............................] - ETA: 1:56 - loss: 1.3346 - categorical_accuracy: 0.5526
 1344/60000 [..............................] - ETA: 1:56 - loss: 1.3189 - categorical_accuracy: 0.5573
 1376/60000 [..............................] - ETA: 1:56 - loss: 1.2993 - categorical_accuracy: 0.5632
 1408/60000 [..............................] - ETA: 1:55 - loss: 1.2807 - categorical_accuracy: 0.5689
 1440/60000 [..............................] - ETA: 1:55 - loss: 1.2658 - categorical_accuracy: 0.5736
 1472/60000 [..............................] - ETA: 1:55 - loss: 1.2569 - categorical_accuracy: 0.5754
 1504/60000 [..............................] - ETA: 1:54 - loss: 1.2574 - categorical_accuracy: 0.5751
 1536/60000 [..............................] - ETA: 1:54 - loss: 1.2498 - categorical_accuracy: 0.5775
 1568/60000 [..............................] - ETA: 1:54 - loss: 1.2362 - categorical_accuracy: 0.5810
 1600/60000 [..............................] - ETA: 1:54 - loss: 1.2276 - categorical_accuracy: 0.5856
 1632/60000 [..............................] - ETA: 1:53 - loss: 1.2210 - categorical_accuracy: 0.5870
 1664/60000 [..............................] - ETA: 1:53 - loss: 1.2074 - categorical_accuracy: 0.5907
 1696/60000 [..............................] - ETA: 1:53 - loss: 1.2027 - categorical_accuracy: 0.5908
 1728/60000 [..............................] - ETA: 1:53 - loss: 1.1950 - categorical_accuracy: 0.5943
 1760/60000 [..............................] - ETA: 1:52 - loss: 1.1859 - categorical_accuracy: 0.5983
 1792/60000 [..............................] - ETA: 1:52 - loss: 1.1730 - categorical_accuracy: 0.6038
 1824/60000 [..............................] - ETA: 1:52 - loss: 1.1602 - categorical_accuracy: 0.6086
 1856/60000 [..............................] - ETA: 1:52 - loss: 1.1547 - categorical_accuracy: 0.6115
 1888/60000 [..............................] - ETA: 1:52 - loss: 1.1452 - categorical_accuracy: 0.6160
 1920/60000 [..............................] - ETA: 1:52 - loss: 1.1352 - categorical_accuracy: 0.6187
 1952/60000 [..............................] - ETA: 1:51 - loss: 1.1258 - categorical_accuracy: 0.6219
 1984/60000 [..............................] - ETA: 1:51 - loss: 1.1188 - categorical_accuracy: 0.6250
 2016/60000 [>.............................] - ETA: 1:51 - loss: 1.1139 - categorical_accuracy: 0.6270
 2048/60000 [>.............................] - ETA: 1:51 - loss: 1.1079 - categorical_accuracy: 0.6289
 2080/60000 [>.............................] - ETA: 1:51 - loss: 1.0953 - categorical_accuracy: 0.6332
 2112/60000 [>.............................] - ETA: 1:51 - loss: 1.0871 - categorical_accuracy: 0.6354
 2144/60000 [>.............................] - ETA: 1:51 - loss: 1.0817 - categorical_accuracy: 0.6367
 2176/60000 [>.............................] - ETA: 1:51 - loss: 1.0744 - categorical_accuracy: 0.6392
 2208/60000 [>.............................] - ETA: 1:50 - loss: 1.0697 - categorical_accuracy: 0.6404
 2240/60000 [>.............................] - ETA: 1:50 - loss: 1.0631 - categorical_accuracy: 0.6433
 2272/60000 [>.............................] - ETA: 1:50 - loss: 1.0570 - categorical_accuracy: 0.6461
 2304/60000 [>.............................] - ETA: 1:50 - loss: 1.0469 - categorical_accuracy: 0.6506
 2336/60000 [>.............................] - ETA: 1:50 - loss: 1.0369 - categorical_accuracy: 0.6533
 2368/60000 [>.............................] - ETA: 1:50 - loss: 1.0295 - categorical_accuracy: 0.6562
 2400/60000 [>.............................] - ETA: 1:49 - loss: 1.0206 - categorical_accuracy: 0.6592
 2432/60000 [>.............................] - ETA: 1:49 - loss: 1.0141 - categorical_accuracy: 0.6608
 2464/60000 [>.............................] - ETA: 1:49 - loss: 1.0106 - categorical_accuracy: 0.6611
 2496/60000 [>.............................] - ETA: 1:49 - loss: 1.0031 - categorical_accuracy: 0.6639
 2528/60000 [>.............................] - ETA: 1:49 - loss: 0.9967 - categorical_accuracy: 0.6665
 2560/60000 [>.............................] - ETA: 1:49 - loss: 0.9899 - categorical_accuracy: 0.6691
 2592/60000 [>.............................] - ETA: 1:49 - loss: 0.9820 - categorical_accuracy: 0.6717
 2624/60000 [>.............................] - ETA: 1:49 - loss: 0.9750 - categorical_accuracy: 0.6734
 2656/60000 [>.............................] - ETA: 1:49 - loss: 0.9723 - categorical_accuracy: 0.6751
 2688/60000 [>.............................] - ETA: 1:49 - loss: 0.9679 - categorical_accuracy: 0.6763
 2720/60000 [>.............................] - ETA: 1:49 - loss: 0.9643 - categorical_accuracy: 0.6779
 2752/60000 [>.............................] - ETA: 1:48 - loss: 0.9567 - categorical_accuracy: 0.6806
 2784/60000 [>.............................] - ETA: 1:48 - loss: 0.9488 - categorical_accuracy: 0.6835
 2816/60000 [>.............................] - ETA: 1:48 - loss: 0.9406 - categorical_accuracy: 0.6864
 2848/60000 [>.............................] - ETA: 1:48 - loss: 0.9348 - categorical_accuracy: 0.6875
 2880/60000 [>.............................] - ETA: 1:48 - loss: 0.9288 - categorical_accuracy: 0.6889
 2912/60000 [>.............................] - ETA: 1:48 - loss: 0.9251 - categorical_accuracy: 0.6902
 2944/60000 [>.............................] - ETA: 1:48 - loss: 0.9181 - categorical_accuracy: 0.6926
 2976/60000 [>.............................] - ETA: 1:48 - loss: 0.9138 - categorical_accuracy: 0.6939
 3008/60000 [>.............................] - ETA: 1:48 - loss: 0.9083 - categorical_accuracy: 0.6965
 3040/60000 [>.............................] - ETA: 1:47 - loss: 0.9057 - categorical_accuracy: 0.6961
 3072/60000 [>.............................] - ETA: 1:47 - loss: 0.9029 - categorical_accuracy: 0.6973
 3104/60000 [>.............................] - ETA: 1:47 - loss: 0.8972 - categorical_accuracy: 0.6991
 3136/60000 [>.............................] - ETA: 1:47 - loss: 0.8919 - categorical_accuracy: 0.7012
 3168/60000 [>.............................] - ETA: 1:47 - loss: 0.8865 - categorical_accuracy: 0.7027
 3200/60000 [>.............................] - ETA: 1:47 - loss: 0.8817 - categorical_accuracy: 0.7041
 3232/60000 [>.............................] - ETA: 1:47 - loss: 0.8773 - categorical_accuracy: 0.7054
 3264/60000 [>.............................] - ETA: 1:47 - loss: 0.8724 - categorical_accuracy: 0.7074
 3296/60000 [>.............................] - ETA: 1:47 - loss: 0.8676 - categorical_accuracy: 0.7096
 3328/60000 [>.............................] - ETA: 1:47 - loss: 0.8631 - categorical_accuracy: 0.7109
 3360/60000 [>.............................] - ETA: 1:47 - loss: 0.8583 - categorical_accuracy: 0.7125
 3392/60000 [>.............................] - ETA: 1:47 - loss: 0.8547 - categorical_accuracy: 0.7134
 3424/60000 [>.............................] - ETA: 1:46 - loss: 0.8514 - categorical_accuracy: 0.7150
 3456/60000 [>.............................] - ETA: 1:46 - loss: 0.8468 - categorical_accuracy: 0.7164
 3488/60000 [>.............................] - ETA: 1:46 - loss: 0.8422 - categorical_accuracy: 0.7185
 3520/60000 [>.............................] - ETA: 1:46 - loss: 0.8406 - categorical_accuracy: 0.7193
 3552/60000 [>.............................] - ETA: 1:46 - loss: 0.8369 - categorical_accuracy: 0.7210
 3584/60000 [>.............................] - ETA: 1:46 - loss: 0.8338 - categorical_accuracy: 0.7221
 3616/60000 [>.............................] - ETA: 1:46 - loss: 0.8296 - categorical_accuracy: 0.7232
 3648/60000 [>.............................] - ETA: 1:46 - loss: 0.8266 - categorical_accuracy: 0.7245
 3680/60000 [>.............................] - ETA: 1:46 - loss: 0.8223 - categorical_accuracy: 0.7258
 3712/60000 [>.............................] - ETA: 1:46 - loss: 0.8185 - categorical_accuracy: 0.7266
 3744/60000 [>.............................] - ETA: 1:46 - loss: 0.8152 - categorical_accuracy: 0.7276
 3776/60000 [>.............................] - ETA: 1:46 - loss: 0.8139 - categorical_accuracy: 0.7283
 3808/60000 [>.............................] - ETA: 1:45 - loss: 0.8081 - categorical_accuracy: 0.7306
 3840/60000 [>.............................] - ETA: 1:45 - loss: 0.8059 - categorical_accuracy: 0.7310
 3872/60000 [>.............................] - ETA: 1:45 - loss: 0.8032 - categorical_accuracy: 0.7324
 3904/60000 [>.............................] - ETA: 1:45 - loss: 0.7994 - categorical_accuracy: 0.7336
 3936/60000 [>.............................] - ETA: 1:45 - loss: 0.7974 - categorical_accuracy: 0.7350
 3968/60000 [>.............................] - ETA: 1:45 - loss: 0.7936 - categorical_accuracy: 0.7361
 4000/60000 [=>............................] - ETA: 1:45 - loss: 0.7892 - categorical_accuracy: 0.7375
 4032/60000 [=>............................] - ETA: 1:45 - loss: 0.7859 - categorical_accuracy: 0.7383
 4064/60000 [=>............................] - ETA: 1:45 - loss: 0.7819 - categorical_accuracy: 0.7397
 4096/60000 [=>............................] - ETA: 1:45 - loss: 0.7794 - categorical_accuracy: 0.7410
 4128/60000 [=>............................] - ETA: 1:44 - loss: 0.7743 - categorical_accuracy: 0.7430
 4160/60000 [=>............................] - ETA: 1:44 - loss: 0.7724 - categorical_accuracy: 0.7442
 4192/60000 [=>............................] - ETA: 1:44 - loss: 0.7677 - categorical_accuracy: 0.7459
 4224/60000 [=>............................] - ETA: 1:44 - loss: 0.7656 - categorical_accuracy: 0.7467
 4256/60000 [=>............................] - ETA: 1:44 - loss: 0.7650 - categorical_accuracy: 0.7474
 4288/60000 [=>............................] - ETA: 1:44 - loss: 0.7624 - categorical_accuracy: 0.7481
 4320/60000 [=>............................] - ETA: 1:44 - loss: 0.7584 - categorical_accuracy: 0.7498
 4352/60000 [=>............................] - ETA: 1:44 - loss: 0.7539 - categorical_accuracy: 0.7516
 4384/60000 [=>............................] - ETA: 1:44 - loss: 0.7506 - categorical_accuracy: 0.7527
 4416/60000 [=>............................] - ETA: 1:44 - loss: 0.7460 - categorical_accuracy: 0.7543
 4448/60000 [=>............................] - ETA: 1:44 - loss: 0.7444 - categorical_accuracy: 0.7552
 4480/60000 [=>............................] - ETA: 1:44 - loss: 0.7410 - categorical_accuracy: 0.7560
 4512/60000 [=>............................] - ETA: 1:44 - loss: 0.7389 - categorical_accuracy: 0.7569
 4544/60000 [=>............................] - ETA: 1:43 - loss: 0.7351 - categorical_accuracy: 0.7581
 4576/60000 [=>............................] - ETA: 1:43 - loss: 0.7332 - categorical_accuracy: 0.7585
 4608/60000 [=>............................] - ETA: 1:43 - loss: 0.7320 - categorical_accuracy: 0.7593
 4640/60000 [=>............................] - ETA: 1:43 - loss: 0.7308 - categorical_accuracy: 0.7599
 4672/60000 [=>............................] - ETA: 1:43 - loss: 0.7297 - categorical_accuracy: 0.7601
 4704/60000 [=>............................] - ETA: 1:43 - loss: 0.7272 - categorical_accuracy: 0.7608
 4736/60000 [=>............................] - ETA: 1:43 - loss: 0.7244 - categorical_accuracy: 0.7620
 4768/60000 [=>............................] - ETA: 1:43 - loss: 0.7206 - categorical_accuracy: 0.7634
 4800/60000 [=>............................] - ETA: 1:43 - loss: 0.7175 - categorical_accuracy: 0.7646
 4832/60000 [=>............................] - ETA: 1:43 - loss: 0.7156 - categorical_accuracy: 0.7651
 4864/60000 [=>............................] - ETA: 1:43 - loss: 0.7123 - categorical_accuracy: 0.7660
 4896/60000 [=>............................] - ETA: 1:42 - loss: 0.7085 - categorical_accuracy: 0.7674
 4928/60000 [=>............................] - ETA: 1:42 - loss: 0.7048 - categorical_accuracy: 0.7687
 4960/60000 [=>............................] - ETA: 1:42 - loss: 0.7035 - categorical_accuracy: 0.7694
 4992/60000 [=>............................] - ETA: 1:42 - loss: 0.6997 - categorical_accuracy: 0.7706
 5024/60000 [=>............................] - ETA: 1:42 - loss: 0.6967 - categorical_accuracy: 0.7715
 5056/60000 [=>............................] - ETA: 1:42 - loss: 0.6936 - categorical_accuracy: 0.7725
 5088/60000 [=>............................] - ETA: 1:42 - loss: 0.6905 - categorical_accuracy: 0.7738
 5120/60000 [=>............................] - ETA: 1:42 - loss: 0.6875 - categorical_accuracy: 0.7750
 5152/60000 [=>............................] - ETA: 1:42 - loss: 0.6856 - categorical_accuracy: 0.7758
 5184/60000 [=>............................] - ETA: 1:42 - loss: 0.6862 - categorical_accuracy: 0.7758
 5216/60000 [=>............................] - ETA: 1:42 - loss: 0.6843 - categorical_accuracy: 0.7765
 5248/60000 [=>............................] - ETA: 1:42 - loss: 0.6817 - categorical_accuracy: 0.7774
 5280/60000 [=>............................] - ETA: 1:41 - loss: 0.6797 - categorical_accuracy: 0.7778
 5312/60000 [=>............................] - ETA: 1:41 - loss: 0.6764 - categorical_accuracy: 0.7790
 5344/60000 [=>............................] - ETA: 1:41 - loss: 0.6740 - categorical_accuracy: 0.7799
 5376/60000 [=>............................] - ETA: 1:41 - loss: 0.6715 - categorical_accuracy: 0.7807
 5408/60000 [=>............................] - ETA: 1:41 - loss: 0.6712 - categorical_accuracy: 0.7814
 5440/60000 [=>............................] - ETA: 1:41 - loss: 0.6690 - categorical_accuracy: 0.7824
 5472/60000 [=>............................] - ETA: 1:41 - loss: 0.6669 - categorical_accuracy: 0.7831
 5504/60000 [=>............................] - ETA: 1:41 - loss: 0.6642 - categorical_accuracy: 0.7840
 5536/60000 [=>............................] - ETA: 1:41 - loss: 0.6619 - categorical_accuracy: 0.7850
 5568/60000 [=>............................] - ETA: 1:41 - loss: 0.6601 - categorical_accuracy: 0.7857
 5600/60000 [=>............................] - ETA: 1:41 - loss: 0.6609 - categorical_accuracy: 0.7857
 5632/60000 [=>............................] - ETA: 1:41 - loss: 0.6585 - categorical_accuracy: 0.7866
 5664/60000 [=>............................] - ETA: 1:41 - loss: 0.6561 - categorical_accuracy: 0.7874
 5696/60000 [=>............................] - ETA: 1:41 - loss: 0.6552 - categorical_accuracy: 0.7879
 5728/60000 [=>............................] - ETA: 1:40 - loss: 0.6524 - categorical_accuracy: 0.7888
 5760/60000 [=>............................] - ETA: 1:40 - loss: 0.6498 - categorical_accuracy: 0.7892
 5792/60000 [=>............................] - ETA: 1:40 - loss: 0.6489 - categorical_accuracy: 0.7895
 5824/60000 [=>............................] - ETA: 1:40 - loss: 0.6460 - categorical_accuracy: 0.7905
 5856/60000 [=>............................] - ETA: 1:40 - loss: 0.6439 - categorical_accuracy: 0.7913
 5888/60000 [=>............................] - ETA: 1:40 - loss: 0.6423 - categorical_accuracy: 0.7919
 5920/60000 [=>............................] - ETA: 1:40 - loss: 0.6400 - categorical_accuracy: 0.7927
 5952/60000 [=>............................] - ETA: 1:40 - loss: 0.6374 - categorical_accuracy: 0.7937
 5984/60000 [=>............................] - ETA: 1:40 - loss: 0.6365 - categorical_accuracy: 0.7941
 6016/60000 [==>...........................] - ETA: 1:40 - loss: 0.6360 - categorical_accuracy: 0.7942
 6048/60000 [==>...........................] - ETA: 1:40 - loss: 0.6356 - categorical_accuracy: 0.7943
 6080/60000 [==>...........................] - ETA: 1:40 - loss: 0.6337 - categorical_accuracy: 0.7949
 6112/60000 [==>...........................] - ETA: 1:40 - loss: 0.6313 - categorical_accuracy: 0.7958
 6144/60000 [==>...........................] - ETA: 1:40 - loss: 0.6295 - categorical_accuracy: 0.7965
 6176/60000 [==>...........................] - ETA: 1:39 - loss: 0.6276 - categorical_accuracy: 0.7970
 6208/60000 [==>...........................] - ETA: 1:39 - loss: 0.6267 - categorical_accuracy: 0.7974
 6240/60000 [==>...........................] - ETA: 1:39 - loss: 0.6260 - categorical_accuracy: 0.7978
 6272/60000 [==>...........................] - ETA: 1:39 - loss: 0.6254 - categorical_accuracy: 0.7982
 6304/60000 [==>...........................] - ETA: 1:39 - loss: 0.6230 - categorical_accuracy: 0.7989
 6336/60000 [==>...........................] - ETA: 1:39 - loss: 0.6226 - categorical_accuracy: 0.7992
 6368/60000 [==>...........................] - ETA: 1:39 - loss: 0.6208 - categorical_accuracy: 0.7998
 6400/60000 [==>...........................] - ETA: 1:39 - loss: 0.6184 - categorical_accuracy: 0.8005
 6432/60000 [==>...........................] - ETA: 1:39 - loss: 0.6171 - categorical_accuracy: 0.8010
 6464/60000 [==>...........................] - ETA: 1:39 - loss: 0.6165 - categorical_accuracy: 0.8009
 6496/60000 [==>...........................] - ETA: 1:39 - loss: 0.6142 - categorical_accuracy: 0.8017
 6528/60000 [==>...........................] - ETA: 1:39 - loss: 0.6128 - categorical_accuracy: 0.8024
 6560/60000 [==>...........................] - ETA: 1:38 - loss: 0.6105 - categorical_accuracy: 0.8030
 6592/60000 [==>...........................] - ETA: 1:38 - loss: 0.6084 - categorical_accuracy: 0.8039
 6624/60000 [==>...........................] - ETA: 1:38 - loss: 0.6076 - categorical_accuracy: 0.8042
 6656/60000 [==>...........................] - ETA: 1:38 - loss: 0.6066 - categorical_accuracy: 0.8045
 6688/60000 [==>...........................] - ETA: 1:38 - loss: 0.6042 - categorical_accuracy: 0.8053
 6720/60000 [==>...........................] - ETA: 1:38 - loss: 0.6016 - categorical_accuracy: 0.8061
 6752/60000 [==>...........................] - ETA: 1:38 - loss: 0.6010 - categorical_accuracy: 0.8066
 6784/60000 [==>...........................] - ETA: 1:38 - loss: 0.5992 - categorical_accuracy: 0.8072
 6816/60000 [==>...........................] - ETA: 1:38 - loss: 0.5971 - categorical_accuracy: 0.8080
 6848/60000 [==>...........................] - ETA: 1:38 - loss: 0.5953 - categorical_accuracy: 0.8087
 6880/60000 [==>...........................] - ETA: 1:38 - loss: 0.5936 - categorical_accuracy: 0.8090
 6912/60000 [==>...........................] - ETA: 1:38 - loss: 0.5917 - categorical_accuracy: 0.8096
 6944/60000 [==>...........................] - ETA: 1:38 - loss: 0.5896 - categorical_accuracy: 0.8103
 6976/60000 [==>...........................] - ETA: 1:38 - loss: 0.5879 - categorical_accuracy: 0.8108
 7008/60000 [==>...........................] - ETA: 1:37 - loss: 0.5864 - categorical_accuracy: 0.8111
 7040/60000 [==>...........................] - ETA: 1:37 - loss: 0.5851 - categorical_accuracy: 0.8115
 7072/60000 [==>...........................] - ETA: 1:37 - loss: 0.5832 - categorical_accuracy: 0.8119
 7104/60000 [==>...........................] - ETA: 1:37 - loss: 0.5814 - categorical_accuracy: 0.8125
 7136/60000 [==>...........................] - ETA: 1:37 - loss: 0.5802 - categorical_accuracy: 0.8128
 7168/60000 [==>...........................] - ETA: 1:37 - loss: 0.5790 - categorical_accuracy: 0.8131
 7200/60000 [==>...........................] - ETA: 1:37 - loss: 0.5774 - categorical_accuracy: 0.8135
 7232/60000 [==>...........................] - ETA: 1:37 - loss: 0.5763 - categorical_accuracy: 0.8139
 7264/60000 [==>...........................] - ETA: 1:37 - loss: 0.5751 - categorical_accuracy: 0.8142
 7296/60000 [==>...........................] - ETA: 1:37 - loss: 0.5748 - categorical_accuracy: 0.8144
 7328/60000 [==>...........................] - ETA: 1:37 - loss: 0.5736 - categorical_accuracy: 0.8147
 7360/60000 [==>...........................] - ETA: 1:37 - loss: 0.5725 - categorical_accuracy: 0.8148
 7392/60000 [==>...........................] - ETA: 1:37 - loss: 0.5709 - categorical_accuracy: 0.8155
 7424/60000 [==>...........................] - ETA: 1:37 - loss: 0.5693 - categorical_accuracy: 0.8159
 7456/60000 [==>...........................] - ETA: 1:37 - loss: 0.5681 - categorical_accuracy: 0.8161
 7488/60000 [==>...........................] - ETA: 1:37 - loss: 0.5664 - categorical_accuracy: 0.8166
 7520/60000 [==>...........................] - ETA: 1:36 - loss: 0.5655 - categorical_accuracy: 0.8169
 7552/60000 [==>...........................] - ETA: 1:36 - loss: 0.5641 - categorical_accuracy: 0.8174
 7584/60000 [==>...........................] - ETA: 1:36 - loss: 0.5637 - categorical_accuracy: 0.8176
 7616/60000 [==>...........................] - ETA: 1:36 - loss: 0.5625 - categorical_accuracy: 0.8180
 7648/60000 [==>...........................] - ETA: 1:36 - loss: 0.5609 - categorical_accuracy: 0.8184
 7680/60000 [==>...........................] - ETA: 1:36 - loss: 0.5606 - categorical_accuracy: 0.8186
 7712/60000 [==>...........................] - ETA: 1:36 - loss: 0.5593 - categorical_accuracy: 0.8190
 7744/60000 [==>...........................] - ETA: 1:36 - loss: 0.5585 - categorical_accuracy: 0.8192
 7776/60000 [==>...........................] - ETA: 1:36 - loss: 0.5573 - categorical_accuracy: 0.8196
 7808/60000 [==>...........................] - ETA: 1:36 - loss: 0.5564 - categorical_accuracy: 0.8199
 7840/60000 [==>...........................] - ETA: 1:36 - loss: 0.5547 - categorical_accuracy: 0.8205
 7872/60000 [==>...........................] - ETA: 1:36 - loss: 0.5527 - categorical_accuracy: 0.8213
 7904/60000 [==>...........................] - ETA: 1:36 - loss: 0.5521 - categorical_accuracy: 0.8214
 7936/60000 [==>...........................] - ETA: 1:36 - loss: 0.5508 - categorical_accuracy: 0.8218
 7968/60000 [==>...........................] - ETA: 1:36 - loss: 0.5489 - categorical_accuracy: 0.8224
 8000/60000 [===>..........................] - ETA: 1:36 - loss: 0.5472 - categorical_accuracy: 0.8230
 8032/60000 [===>..........................] - ETA: 1:36 - loss: 0.5457 - categorical_accuracy: 0.8235
 8064/60000 [===>..........................] - ETA: 1:35 - loss: 0.5440 - categorical_accuracy: 0.8242
 8096/60000 [===>..........................] - ETA: 1:35 - loss: 0.5430 - categorical_accuracy: 0.8244
 8128/60000 [===>..........................] - ETA: 1:35 - loss: 0.5419 - categorical_accuracy: 0.8248
 8160/60000 [===>..........................] - ETA: 1:35 - loss: 0.5403 - categorical_accuracy: 0.8251
 8192/60000 [===>..........................] - ETA: 1:35 - loss: 0.5401 - categorical_accuracy: 0.8251
 8224/60000 [===>..........................] - ETA: 1:35 - loss: 0.5391 - categorical_accuracy: 0.8253
 8256/60000 [===>..........................] - ETA: 1:35 - loss: 0.5376 - categorical_accuracy: 0.8258
 8288/60000 [===>..........................] - ETA: 1:35 - loss: 0.5365 - categorical_accuracy: 0.8261
 8320/60000 [===>..........................] - ETA: 1:35 - loss: 0.5349 - categorical_accuracy: 0.8267
 8352/60000 [===>..........................] - ETA: 1:35 - loss: 0.5343 - categorical_accuracy: 0.8271
 8384/60000 [===>..........................] - ETA: 1:35 - loss: 0.5333 - categorical_accuracy: 0.8274
 8416/60000 [===>..........................] - ETA: 1:35 - loss: 0.5318 - categorical_accuracy: 0.8278
 8448/60000 [===>..........................] - ETA: 1:35 - loss: 0.5304 - categorical_accuracy: 0.8282
 8480/60000 [===>..........................] - ETA: 1:35 - loss: 0.5293 - categorical_accuracy: 0.8285
 8512/60000 [===>..........................] - ETA: 1:34 - loss: 0.5276 - categorical_accuracy: 0.8291
 8544/60000 [===>..........................] - ETA: 1:34 - loss: 0.5262 - categorical_accuracy: 0.8295
 8576/60000 [===>..........................] - ETA: 1:34 - loss: 0.5244 - categorical_accuracy: 0.8301
 8608/60000 [===>..........................] - ETA: 1:34 - loss: 0.5227 - categorical_accuracy: 0.8307
 8640/60000 [===>..........................] - ETA: 1:34 - loss: 0.5211 - categorical_accuracy: 0.8313
 8672/60000 [===>..........................] - ETA: 1:34 - loss: 0.5193 - categorical_accuracy: 0.8319
 8704/60000 [===>..........................] - ETA: 1:34 - loss: 0.5181 - categorical_accuracy: 0.8321
 8736/60000 [===>..........................] - ETA: 1:34 - loss: 0.5166 - categorical_accuracy: 0.8326
 8768/60000 [===>..........................] - ETA: 1:34 - loss: 0.5153 - categorical_accuracy: 0.8330
 8800/60000 [===>..........................] - ETA: 1:34 - loss: 0.5154 - categorical_accuracy: 0.8328
 8832/60000 [===>..........................] - ETA: 1:34 - loss: 0.5139 - categorical_accuracy: 0.8332
 8864/60000 [===>..........................] - ETA: 1:34 - loss: 0.5122 - categorical_accuracy: 0.8338
 8896/60000 [===>..........................] - ETA: 1:34 - loss: 0.5111 - categorical_accuracy: 0.8340
 8928/60000 [===>..........................] - ETA: 1:34 - loss: 0.5094 - categorical_accuracy: 0.8346
 8960/60000 [===>..........................] - ETA: 1:34 - loss: 0.5088 - categorical_accuracy: 0.8348
 8992/60000 [===>..........................] - ETA: 1:33 - loss: 0.5079 - categorical_accuracy: 0.8351
 9024/60000 [===>..........................] - ETA: 1:33 - loss: 0.5064 - categorical_accuracy: 0.8355
 9056/60000 [===>..........................] - ETA: 1:33 - loss: 0.5053 - categorical_accuracy: 0.8359
 9088/60000 [===>..........................] - ETA: 1:33 - loss: 0.5043 - categorical_accuracy: 0.8363
 9120/60000 [===>..........................] - ETA: 1:33 - loss: 0.5033 - categorical_accuracy: 0.8367
 9152/60000 [===>..........................] - ETA: 1:33 - loss: 0.5018 - categorical_accuracy: 0.8372
 9184/60000 [===>..........................] - ETA: 1:33 - loss: 0.5004 - categorical_accuracy: 0.8377
 9216/60000 [===>..........................] - ETA: 1:33 - loss: 0.4988 - categorical_accuracy: 0.8382
 9248/60000 [===>..........................] - ETA: 1:33 - loss: 0.4978 - categorical_accuracy: 0.8383
 9280/60000 [===>..........................] - ETA: 1:33 - loss: 0.4968 - categorical_accuracy: 0.8387
 9312/60000 [===>..........................] - ETA: 1:33 - loss: 0.4960 - categorical_accuracy: 0.8388
 9344/60000 [===>..........................] - ETA: 1:33 - loss: 0.4951 - categorical_accuracy: 0.8390
 9376/60000 [===>..........................] - ETA: 1:33 - loss: 0.4938 - categorical_accuracy: 0.8395
 9408/60000 [===>..........................] - ETA: 1:33 - loss: 0.4927 - categorical_accuracy: 0.8398
 9440/60000 [===>..........................] - ETA: 1:33 - loss: 0.4918 - categorical_accuracy: 0.8400
 9472/60000 [===>..........................] - ETA: 1:33 - loss: 0.4918 - categorical_accuracy: 0.8402
 9504/60000 [===>..........................] - ETA: 1:32 - loss: 0.4906 - categorical_accuracy: 0.8405
 9536/60000 [===>..........................] - ETA: 1:32 - loss: 0.4915 - categorical_accuracy: 0.8406
 9568/60000 [===>..........................] - ETA: 1:32 - loss: 0.4906 - categorical_accuracy: 0.8409
 9600/60000 [===>..........................] - ETA: 1:32 - loss: 0.4896 - categorical_accuracy: 0.8414
 9632/60000 [===>..........................] - ETA: 1:32 - loss: 0.4891 - categorical_accuracy: 0.8415
 9664/60000 [===>..........................] - ETA: 1:32 - loss: 0.4886 - categorical_accuracy: 0.8417
 9696/60000 [===>..........................] - ETA: 1:32 - loss: 0.4874 - categorical_accuracy: 0.8421
 9728/60000 [===>..........................] - ETA: 1:32 - loss: 0.4866 - categorical_accuracy: 0.8422
 9760/60000 [===>..........................] - ETA: 1:32 - loss: 0.4856 - categorical_accuracy: 0.8426
 9792/60000 [===>..........................] - ETA: 1:32 - loss: 0.4849 - categorical_accuracy: 0.8429
 9824/60000 [===>..........................] - ETA: 1:32 - loss: 0.4842 - categorical_accuracy: 0.8431
 9856/60000 [===>..........................] - ETA: 1:32 - loss: 0.4829 - categorical_accuracy: 0.8436
 9888/60000 [===>..........................] - ETA: 1:32 - loss: 0.4820 - categorical_accuracy: 0.8441
 9920/60000 [===>..........................] - ETA: 1:32 - loss: 0.4812 - categorical_accuracy: 0.8444
 9952/60000 [===>..........................] - ETA: 1:32 - loss: 0.4807 - categorical_accuracy: 0.8445
 9984/60000 [===>..........................] - ETA: 1:31 - loss: 0.4802 - categorical_accuracy: 0.8448
10016/60000 [====>.........................] - ETA: 1:31 - loss: 0.4790 - categorical_accuracy: 0.8451
10048/60000 [====>.........................] - ETA: 1:31 - loss: 0.4792 - categorical_accuracy: 0.8451
10080/60000 [====>.........................] - ETA: 1:31 - loss: 0.4781 - categorical_accuracy: 0.8453
10112/60000 [====>.........................] - ETA: 1:31 - loss: 0.4772 - categorical_accuracy: 0.8456
10144/60000 [====>.........................] - ETA: 1:31 - loss: 0.4764 - categorical_accuracy: 0.8459
10176/60000 [====>.........................] - ETA: 1:31 - loss: 0.4757 - categorical_accuracy: 0.8462
10208/60000 [====>.........................] - ETA: 1:31 - loss: 0.4750 - categorical_accuracy: 0.8464
10240/60000 [====>.........................] - ETA: 1:31 - loss: 0.4740 - categorical_accuracy: 0.8467
10272/60000 [====>.........................] - ETA: 1:31 - loss: 0.4729 - categorical_accuracy: 0.8470
10304/60000 [====>.........................] - ETA: 1:31 - loss: 0.4724 - categorical_accuracy: 0.8471
10336/60000 [====>.........................] - ETA: 1:31 - loss: 0.4716 - categorical_accuracy: 0.8474
10368/60000 [====>.........................] - ETA: 1:31 - loss: 0.4717 - categorical_accuracy: 0.8475
10400/60000 [====>.........................] - ETA: 1:31 - loss: 0.4720 - categorical_accuracy: 0.8477
10432/60000 [====>.........................] - ETA: 1:31 - loss: 0.4708 - categorical_accuracy: 0.8481
10464/60000 [====>.........................] - ETA: 1:30 - loss: 0.4696 - categorical_accuracy: 0.8485
10496/60000 [====>.........................] - ETA: 1:30 - loss: 0.4685 - categorical_accuracy: 0.8489
10528/60000 [====>.........................] - ETA: 1:30 - loss: 0.4686 - categorical_accuracy: 0.8490
10560/60000 [====>.........................] - ETA: 1:30 - loss: 0.4676 - categorical_accuracy: 0.8492
10592/60000 [====>.........................] - ETA: 1:30 - loss: 0.4679 - categorical_accuracy: 0.8493
10624/60000 [====>.........................] - ETA: 1:30 - loss: 0.4673 - categorical_accuracy: 0.8496
10656/60000 [====>.........................] - ETA: 1:30 - loss: 0.4668 - categorical_accuracy: 0.8498
10688/60000 [====>.........................] - ETA: 1:30 - loss: 0.4660 - categorical_accuracy: 0.8501
10720/60000 [====>.........................] - ETA: 1:30 - loss: 0.4654 - categorical_accuracy: 0.8503
10752/60000 [====>.........................] - ETA: 1:30 - loss: 0.4642 - categorical_accuracy: 0.8507
10784/60000 [====>.........................] - ETA: 1:30 - loss: 0.4629 - categorical_accuracy: 0.8512
10816/60000 [====>.........................] - ETA: 1:30 - loss: 0.4626 - categorical_accuracy: 0.8512
10848/60000 [====>.........................] - ETA: 1:30 - loss: 0.4616 - categorical_accuracy: 0.8516
10880/60000 [====>.........................] - ETA: 1:30 - loss: 0.4611 - categorical_accuracy: 0.8518
10912/60000 [====>.........................] - ETA: 1:30 - loss: 0.4600 - categorical_accuracy: 0.8522
10944/60000 [====>.........................] - ETA: 1:29 - loss: 0.4589 - categorical_accuracy: 0.8526
10976/60000 [====>.........................] - ETA: 1:29 - loss: 0.4582 - categorical_accuracy: 0.8528
11008/60000 [====>.........................] - ETA: 1:29 - loss: 0.4581 - categorical_accuracy: 0.8527
11040/60000 [====>.........................] - ETA: 1:29 - loss: 0.4573 - categorical_accuracy: 0.8530
11072/60000 [====>.........................] - ETA: 1:29 - loss: 0.4563 - categorical_accuracy: 0.8532
11104/60000 [====>.........................] - ETA: 1:29 - loss: 0.4553 - categorical_accuracy: 0.8536
11136/60000 [====>.........................] - ETA: 1:29 - loss: 0.4543 - categorical_accuracy: 0.8538
11168/60000 [====>.........................] - ETA: 1:29 - loss: 0.4537 - categorical_accuracy: 0.8540
11200/60000 [====>.........................] - ETA: 1:29 - loss: 0.4540 - categorical_accuracy: 0.8540
11232/60000 [====>.........................] - ETA: 1:29 - loss: 0.4529 - categorical_accuracy: 0.8543
11264/60000 [====>.........................] - ETA: 1:29 - loss: 0.4521 - categorical_accuracy: 0.8546
11296/60000 [====>.........................] - ETA: 1:29 - loss: 0.4510 - categorical_accuracy: 0.8549
11328/60000 [====>.........................] - ETA: 1:29 - loss: 0.4507 - categorical_accuracy: 0.8550
11360/60000 [====>.........................] - ETA: 1:29 - loss: 0.4509 - categorical_accuracy: 0.8553
11392/60000 [====>.........................] - ETA: 1:29 - loss: 0.4498 - categorical_accuracy: 0.8557
11424/60000 [====>.........................] - ETA: 1:29 - loss: 0.4489 - categorical_accuracy: 0.8559
11456/60000 [====>.........................] - ETA: 1:29 - loss: 0.4478 - categorical_accuracy: 0.8562
11488/60000 [====>.........................] - ETA: 1:28 - loss: 0.4468 - categorical_accuracy: 0.8566
11520/60000 [====>.........................] - ETA: 1:28 - loss: 0.4457 - categorical_accuracy: 0.8570
11552/60000 [====>.........................] - ETA: 1:28 - loss: 0.4446 - categorical_accuracy: 0.8573
11584/60000 [====>.........................] - ETA: 1:28 - loss: 0.4439 - categorical_accuracy: 0.8575
11616/60000 [====>.........................] - ETA: 1:28 - loss: 0.4431 - categorical_accuracy: 0.8578
11648/60000 [====>.........................] - ETA: 1:28 - loss: 0.4426 - categorical_accuracy: 0.8579
11680/60000 [====>.........................] - ETA: 1:28 - loss: 0.4416 - categorical_accuracy: 0.8582
11712/60000 [====>.........................] - ETA: 1:28 - loss: 0.4410 - categorical_accuracy: 0.8584
11744/60000 [====>.........................] - ETA: 1:28 - loss: 0.4405 - categorical_accuracy: 0.8587
11776/60000 [====>.........................] - ETA: 1:28 - loss: 0.4400 - categorical_accuracy: 0.8588
11808/60000 [====>.........................] - ETA: 1:28 - loss: 0.4396 - categorical_accuracy: 0.8588
11840/60000 [====>.........................] - ETA: 1:28 - loss: 0.4388 - categorical_accuracy: 0.8590
11872/60000 [====>.........................] - ETA: 1:28 - loss: 0.4379 - categorical_accuracy: 0.8592
11904/60000 [====>.........................] - ETA: 1:28 - loss: 0.4368 - categorical_accuracy: 0.8596
11936/60000 [====>.........................] - ETA: 1:28 - loss: 0.4381 - categorical_accuracy: 0.8597
11968/60000 [====>.........................] - ETA: 1:28 - loss: 0.4373 - categorical_accuracy: 0.8600
12000/60000 [=====>........................] - ETA: 1:27 - loss: 0.4363 - categorical_accuracy: 0.8602
12032/60000 [=====>........................] - ETA: 1:27 - loss: 0.4354 - categorical_accuracy: 0.8605
12064/60000 [=====>........................] - ETA: 1:27 - loss: 0.4344 - categorical_accuracy: 0.8609
12096/60000 [=====>........................] - ETA: 1:27 - loss: 0.4337 - categorical_accuracy: 0.8611
12128/60000 [=====>........................] - ETA: 1:27 - loss: 0.4327 - categorical_accuracy: 0.8614
12160/60000 [=====>........................] - ETA: 1:27 - loss: 0.4318 - categorical_accuracy: 0.8617
12192/60000 [=====>........................] - ETA: 1:27 - loss: 0.4319 - categorical_accuracy: 0.8616
12224/60000 [=====>........................] - ETA: 1:27 - loss: 0.4309 - categorical_accuracy: 0.8619
12256/60000 [=====>........................] - ETA: 1:27 - loss: 0.4303 - categorical_accuracy: 0.8622
12288/60000 [=====>........................] - ETA: 1:27 - loss: 0.4293 - categorical_accuracy: 0.8625
12320/60000 [=====>........................] - ETA: 1:27 - loss: 0.4293 - categorical_accuracy: 0.8627
12352/60000 [=====>........................] - ETA: 1:27 - loss: 0.4287 - categorical_accuracy: 0.8629
12384/60000 [=====>........................] - ETA: 1:27 - loss: 0.4280 - categorical_accuracy: 0.8630
12416/60000 [=====>........................] - ETA: 1:27 - loss: 0.4271 - categorical_accuracy: 0.8633
12448/60000 [=====>........................] - ETA: 1:27 - loss: 0.4265 - categorical_accuracy: 0.8634
12480/60000 [=====>........................] - ETA: 1:26 - loss: 0.4259 - categorical_accuracy: 0.8636
12512/60000 [=====>........................] - ETA: 1:26 - loss: 0.4252 - categorical_accuracy: 0.8639
12544/60000 [=====>........................] - ETA: 1:26 - loss: 0.4243 - categorical_accuracy: 0.8642
12576/60000 [=====>........................] - ETA: 1:26 - loss: 0.4233 - categorical_accuracy: 0.8645
12608/60000 [=====>........................] - ETA: 1:26 - loss: 0.4228 - categorical_accuracy: 0.8648
12640/60000 [=====>........................] - ETA: 1:26 - loss: 0.4223 - categorical_accuracy: 0.8650
12672/60000 [=====>........................] - ETA: 1:26 - loss: 0.4215 - categorical_accuracy: 0.8653
12704/60000 [=====>........................] - ETA: 1:26 - loss: 0.4208 - categorical_accuracy: 0.8656
12736/60000 [=====>........................] - ETA: 1:26 - loss: 0.4206 - categorical_accuracy: 0.8657
12768/60000 [=====>........................] - ETA: 1:26 - loss: 0.4202 - categorical_accuracy: 0.8659
12800/60000 [=====>........................] - ETA: 1:26 - loss: 0.4200 - categorical_accuracy: 0.8660
12832/60000 [=====>........................] - ETA: 1:26 - loss: 0.4191 - categorical_accuracy: 0.8663
12864/60000 [=====>........................] - ETA: 1:26 - loss: 0.4185 - categorical_accuracy: 0.8664
12896/60000 [=====>........................] - ETA: 1:26 - loss: 0.4177 - categorical_accuracy: 0.8668
12928/60000 [=====>........................] - ETA: 1:26 - loss: 0.4172 - categorical_accuracy: 0.8669
12960/60000 [=====>........................] - ETA: 1:26 - loss: 0.4168 - categorical_accuracy: 0.8669
12992/60000 [=====>........................] - ETA: 1:26 - loss: 0.4159 - categorical_accuracy: 0.8672
13024/60000 [=====>........................] - ETA: 1:25 - loss: 0.4151 - categorical_accuracy: 0.8674
13056/60000 [=====>........................] - ETA: 1:25 - loss: 0.4146 - categorical_accuracy: 0.8676
13088/60000 [=====>........................] - ETA: 1:25 - loss: 0.4140 - categorical_accuracy: 0.8677
13120/60000 [=====>........................] - ETA: 1:25 - loss: 0.4135 - categorical_accuracy: 0.8679
13152/60000 [=====>........................] - ETA: 1:25 - loss: 0.4138 - categorical_accuracy: 0.8680
13184/60000 [=====>........................] - ETA: 1:25 - loss: 0.4131 - categorical_accuracy: 0.8682
13216/60000 [=====>........................] - ETA: 1:25 - loss: 0.4132 - categorical_accuracy: 0.8682
13248/60000 [=====>........................] - ETA: 1:25 - loss: 0.4129 - categorical_accuracy: 0.8682
13280/60000 [=====>........................] - ETA: 1:25 - loss: 0.4123 - categorical_accuracy: 0.8683
13312/60000 [=====>........................] - ETA: 1:25 - loss: 0.4119 - categorical_accuracy: 0.8685
13344/60000 [=====>........................] - ETA: 1:25 - loss: 0.4115 - categorical_accuracy: 0.8687
13376/60000 [=====>........................] - ETA: 1:25 - loss: 0.4106 - categorical_accuracy: 0.8690
13408/60000 [=====>........................] - ETA: 1:25 - loss: 0.4106 - categorical_accuracy: 0.8691
13440/60000 [=====>........................] - ETA: 1:25 - loss: 0.4100 - categorical_accuracy: 0.8693
13472/60000 [=====>........................] - ETA: 1:25 - loss: 0.4093 - categorical_accuracy: 0.8696
13504/60000 [=====>........................] - ETA: 1:25 - loss: 0.4085 - categorical_accuracy: 0.8698
13536/60000 [=====>........................] - ETA: 1:24 - loss: 0.4080 - categorical_accuracy: 0.8700
13568/60000 [=====>........................] - ETA: 1:24 - loss: 0.4075 - categorical_accuracy: 0.8701
13600/60000 [=====>........................] - ETA: 1:24 - loss: 0.4073 - categorical_accuracy: 0.8703
13632/60000 [=====>........................] - ETA: 1:24 - loss: 0.4066 - categorical_accuracy: 0.8706
13664/60000 [=====>........................] - ETA: 1:24 - loss: 0.4061 - categorical_accuracy: 0.8708
13696/60000 [=====>........................] - ETA: 1:24 - loss: 0.4055 - categorical_accuracy: 0.8711
13728/60000 [=====>........................] - ETA: 1:24 - loss: 0.4047 - categorical_accuracy: 0.8714
13760/60000 [=====>........................] - ETA: 1:24 - loss: 0.4041 - categorical_accuracy: 0.8716
13792/60000 [=====>........................] - ETA: 1:24 - loss: 0.4040 - categorical_accuracy: 0.8717
13824/60000 [=====>........................] - ETA: 1:24 - loss: 0.4034 - categorical_accuracy: 0.8718
13856/60000 [=====>........................] - ETA: 1:24 - loss: 0.4030 - categorical_accuracy: 0.8720
13888/60000 [=====>........................] - ETA: 1:24 - loss: 0.4029 - categorical_accuracy: 0.8720
13920/60000 [=====>........................] - ETA: 1:24 - loss: 0.4020 - categorical_accuracy: 0.8723
13952/60000 [=====>........................] - ETA: 1:24 - loss: 0.4017 - categorical_accuracy: 0.8723
13984/60000 [=====>........................] - ETA: 1:24 - loss: 0.4012 - categorical_accuracy: 0.8726
14016/60000 [======>.......................] - ETA: 1:24 - loss: 0.4005 - categorical_accuracy: 0.8728
14048/60000 [======>.......................] - ETA: 1:24 - loss: 0.4003 - categorical_accuracy: 0.8727
14080/60000 [======>.......................] - ETA: 1:23 - loss: 0.4000 - categorical_accuracy: 0.8729
14112/60000 [======>.......................] - ETA: 1:23 - loss: 0.3997 - categorical_accuracy: 0.8729
14144/60000 [======>.......................] - ETA: 1:23 - loss: 0.3995 - categorical_accuracy: 0.8729
14176/60000 [======>.......................] - ETA: 1:23 - loss: 0.3989 - categorical_accuracy: 0.8731
14208/60000 [======>.......................] - ETA: 1:23 - loss: 0.3982 - categorical_accuracy: 0.8733
14240/60000 [======>.......................] - ETA: 1:23 - loss: 0.3980 - categorical_accuracy: 0.8735
14272/60000 [======>.......................] - ETA: 1:23 - loss: 0.3974 - categorical_accuracy: 0.8737
14304/60000 [======>.......................] - ETA: 1:23 - loss: 0.3966 - categorical_accuracy: 0.8740
14336/60000 [======>.......................] - ETA: 1:23 - loss: 0.3961 - categorical_accuracy: 0.8742
14368/60000 [======>.......................] - ETA: 1:23 - loss: 0.3954 - categorical_accuracy: 0.8744
14400/60000 [======>.......................] - ETA: 1:23 - loss: 0.3949 - categorical_accuracy: 0.8746
14432/60000 [======>.......................] - ETA: 1:23 - loss: 0.3948 - categorical_accuracy: 0.8747
14464/60000 [======>.......................] - ETA: 1:23 - loss: 0.3942 - categorical_accuracy: 0.8749
14496/60000 [======>.......................] - ETA: 1:23 - loss: 0.3939 - categorical_accuracy: 0.8749
14528/60000 [======>.......................] - ETA: 1:23 - loss: 0.3943 - categorical_accuracy: 0.8749
14560/60000 [======>.......................] - ETA: 1:23 - loss: 0.3936 - categorical_accuracy: 0.8751
14592/60000 [======>.......................] - ETA: 1:23 - loss: 0.3932 - categorical_accuracy: 0.8752
14624/60000 [======>.......................] - ETA: 1:22 - loss: 0.3926 - categorical_accuracy: 0.8753
14656/60000 [======>.......................] - ETA: 1:22 - loss: 0.3920 - categorical_accuracy: 0.8756
14688/60000 [======>.......................] - ETA: 1:22 - loss: 0.3912 - categorical_accuracy: 0.8759
14720/60000 [======>.......................] - ETA: 1:22 - loss: 0.3908 - categorical_accuracy: 0.8761
14752/60000 [======>.......................] - ETA: 1:22 - loss: 0.3904 - categorical_accuracy: 0.8762
14784/60000 [======>.......................] - ETA: 1:22 - loss: 0.3899 - categorical_accuracy: 0.8764
14816/60000 [======>.......................] - ETA: 1:22 - loss: 0.3897 - categorical_accuracy: 0.8764
14848/60000 [======>.......................] - ETA: 1:22 - loss: 0.3892 - categorical_accuracy: 0.8765
14880/60000 [======>.......................] - ETA: 1:22 - loss: 0.3886 - categorical_accuracy: 0.8767
14912/60000 [======>.......................] - ETA: 1:22 - loss: 0.3878 - categorical_accuracy: 0.8770
14944/60000 [======>.......................] - ETA: 1:22 - loss: 0.3882 - categorical_accuracy: 0.8770
14976/60000 [======>.......................] - ETA: 1:22 - loss: 0.3878 - categorical_accuracy: 0.8771
15008/60000 [======>.......................] - ETA: 1:22 - loss: 0.3873 - categorical_accuracy: 0.8773
15040/60000 [======>.......................] - ETA: 1:22 - loss: 0.3871 - categorical_accuracy: 0.8773
15072/60000 [======>.......................] - ETA: 1:22 - loss: 0.3864 - categorical_accuracy: 0.8775
15104/60000 [======>.......................] - ETA: 1:22 - loss: 0.3856 - categorical_accuracy: 0.8778
15136/60000 [======>.......................] - ETA: 1:21 - loss: 0.3849 - categorical_accuracy: 0.8780
15168/60000 [======>.......................] - ETA: 1:21 - loss: 0.3850 - categorical_accuracy: 0.8780
15200/60000 [======>.......................] - ETA: 1:21 - loss: 0.3844 - categorical_accuracy: 0.8782
15232/60000 [======>.......................] - ETA: 1:21 - loss: 0.3839 - categorical_accuracy: 0.8783
15264/60000 [======>.......................] - ETA: 1:21 - loss: 0.3832 - categorical_accuracy: 0.8785
15296/60000 [======>.......................] - ETA: 1:21 - loss: 0.3831 - categorical_accuracy: 0.8786
15328/60000 [======>.......................] - ETA: 1:21 - loss: 0.3831 - categorical_accuracy: 0.8786
15360/60000 [======>.......................] - ETA: 1:21 - loss: 0.3830 - categorical_accuracy: 0.8787
15392/60000 [======>.......................] - ETA: 1:21 - loss: 0.3825 - categorical_accuracy: 0.8789
15424/60000 [======>.......................] - ETA: 1:21 - loss: 0.3822 - categorical_accuracy: 0.8791
15456/60000 [======>.......................] - ETA: 1:21 - loss: 0.3817 - categorical_accuracy: 0.8792
15488/60000 [======>.......................] - ETA: 1:21 - loss: 0.3811 - categorical_accuracy: 0.8794
15520/60000 [======>.......................] - ETA: 1:21 - loss: 0.3808 - categorical_accuracy: 0.8795
15552/60000 [======>.......................] - ETA: 1:21 - loss: 0.3802 - categorical_accuracy: 0.8797
15584/60000 [======>.......................] - ETA: 1:21 - loss: 0.3802 - categorical_accuracy: 0.8797
15616/60000 [======>.......................] - ETA: 1:21 - loss: 0.3795 - categorical_accuracy: 0.8800
15648/60000 [======>.......................] - ETA: 1:20 - loss: 0.3790 - categorical_accuracy: 0.8802
15680/60000 [======>.......................] - ETA: 1:20 - loss: 0.3790 - categorical_accuracy: 0.8803
15712/60000 [======>.......................] - ETA: 1:20 - loss: 0.3787 - categorical_accuracy: 0.8803
15744/60000 [======>.......................] - ETA: 1:20 - loss: 0.3785 - categorical_accuracy: 0.8804
15776/60000 [======>.......................] - ETA: 1:20 - loss: 0.3781 - categorical_accuracy: 0.8805
15808/60000 [======>.......................] - ETA: 1:20 - loss: 0.3777 - categorical_accuracy: 0.8805
15840/60000 [======>.......................] - ETA: 1:20 - loss: 0.3770 - categorical_accuracy: 0.8807
15872/60000 [======>.......................] - ETA: 1:20 - loss: 0.3769 - categorical_accuracy: 0.8809
15904/60000 [======>.......................] - ETA: 1:20 - loss: 0.3770 - categorical_accuracy: 0.8808
15936/60000 [======>.......................] - ETA: 1:20 - loss: 0.3771 - categorical_accuracy: 0.8809
15968/60000 [======>.......................] - ETA: 1:20 - loss: 0.3770 - categorical_accuracy: 0.8810
16000/60000 [=======>......................] - ETA: 1:20 - loss: 0.3768 - categorical_accuracy: 0.8811
16032/60000 [=======>......................] - ETA: 1:20 - loss: 0.3764 - categorical_accuracy: 0.8812
16064/60000 [=======>......................] - ETA: 1:20 - loss: 0.3760 - categorical_accuracy: 0.8813
16096/60000 [=======>......................] - ETA: 1:20 - loss: 0.3756 - categorical_accuracy: 0.8814
16128/60000 [=======>......................] - ETA: 1:20 - loss: 0.3753 - categorical_accuracy: 0.8815
16160/60000 [=======>......................] - ETA: 1:20 - loss: 0.3753 - categorical_accuracy: 0.8815
16192/60000 [=======>......................] - ETA: 1:19 - loss: 0.3749 - categorical_accuracy: 0.8816
16224/60000 [=======>......................] - ETA: 1:19 - loss: 0.3742 - categorical_accuracy: 0.8818
16256/60000 [=======>......................] - ETA: 1:19 - loss: 0.3736 - categorical_accuracy: 0.8821
16288/60000 [=======>......................] - ETA: 1:19 - loss: 0.3729 - categorical_accuracy: 0.8823
16320/60000 [=======>......................] - ETA: 1:19 - loss: 0.3726 - categorical_accuracy: 0.8825
16352/60000 [=======>......................] - ETA: 1:19 - loss: 0.3722 - categorical_accuracy: 0.8826
16384/60000 [=======>......................] - ETA: 1:19 - loss: 0.3715 - categorical_accuracy: 0.8828
16416/60000 [=======>......................] - ETA: 1:19 - loss: 0.3711 - categorical_accuracy: 0.8829
16448/60000 [=======>......................] - ETA: 1:19 - loss: 0.3705 - categorical_accuracy: 0.8831
16480/60000 [=======>......................] - ETA: 1:19 - loss: 0.3700 - categorical_accuracy: 0.8833
16512/60000 [=======>......................] - ETA: 1:19 - loss: 0.3696 - categorical_accuracy: 0.8834
16544/60000 [=======>......................] - ETA: 1:19 - loss: 0.3690 - categorical_accuracy: 0.8836
16576/60000 [=======>......................] - ETA: 1:19 - loss: 0.3685 - categorical_accuracy: 0.8837
16608/60000 [=======>......................] - ETA: 1:19 - loss: 0.3683 - categorical_accuracy: 0.8837
16640/60000 [=======>......................] - ETA: 1:19 - loss: 0.3680 - categorical_accuracy: 0.8837
16672/60000 [=======>......................] - ETA: 1:19 - loss: 0.3680 - categorical_accuracy: 0.8835
16704/60000 [=======>......................] - ETA: 1:19 - loss: 0.3682 - categorical_accuracy: 0.8836
16736/60000 [=======>......................] - ETA: 1:18 - loss: 0.3677 - categorical_accuracy: 0.8837
16768/60000 [=======>......................] - ETA: 1:18 - loss: 0.3671 - categorical_accuracy: 0.8838
16800/60000 [=======>......................] - ETA: 1:18 - loss: 0.3669 - categorical_accuracy: 0.8838
16832/60000 [=======>......................] - ETA: 1:18 - loss: 0.3667 - categorical_accuracy: 0.8840
16864/60000 [=======>......................] - ETA: 1:18 - loss: 0.3663 - categorical_accuracy: 0.8841
16896/60000 [=======>......................] - ETA: 1:18 - loss: 0.3658 - categorical_accuracy: 0.8843
16928/60000 [=======>......................] - ETA: 1:18 - loss: 0.3657 - categorical_accuracy: 0.8843
16960/60000 [=======>......................] - ETA: 1:18 - loss: 0.3658 - categorical_accuracy: 0.8841
16992/60000 [=======>......................] - ETA: 1:18 - loss: 0.3657 - categorical_accuracy: 0.8842
17024/60000 [=======>......................] - ETA: 1:18 - loss: 0.3652 - categorical_accuracy: 0.8843
17056/60000 [=======>......................] - ETA: 1:18 - loss: 0.3647 - categorical_accuracy: 0.8845
17088/60000 [=======>......................] - ETA: 1:18 - loss: 0.3642 - categorical_accuracy: 0.8847
17120/60000 [=======>......................] - ETA: 1:18 - loss: 0.3635 - categorical_accuracy: 0.8849
17152/60000 [=======>......................] - ETA: 1:18 - loss: 0.3632 - categorical_accuracy: 0.8850
17184/60000 [=======>......................] - ETA: 1:18 - loss: 0.3627 - categorical_accuracy: 0.8852
17216/60000 [=======>......................] - ETA: 1:18 - loss: 0.3621 - categorical_accuracy: 0.8854
17248/60000 [=======>......................] - ETA: 1:17 - loss: 0.3616 - categorical_accuracy: 0.8856
17280/60000 [=======>......................] - ETA: 1:17 - loss: 0.3615 - categorical_accuracy: 0.8855
17312/60000 [=======>......................] - ETA: 1:17 - loss: 0.3610 - categorical_accuracy: 0.8856
17344/60000 [=======>......................] - ETA: 1:17 - loss: 0.3605 - categorical_accuracy: 0.8858
17376/60000 [=======>......................] - ETA: 1:17 - loss: 0.3602 - categorical_accuracy: 0.8859
17408/60000 [=======>......................] - ETA: 1:17 - loss: 0.3597 - categorical_accuracy: 0.8861
17440/60000 [=======>......................] - ETA: 1:17 - loss: 0.3594 - categorical_accuracy: 0.8862
17472/60000 [=======>......................] - ETA: 1:17 - loss: 0.3588 - categorical_accuracy: 0.8864
17504/60000 [=======>......................] - ETA: 1:17 - loss: 0.3590 - categorical_accuracy: 0.8864
17536/60000 [=======>......................] - ETA: 1:17 - loss: 0.3587 - categorical_accuracy: 0.8865
17568/60000 [=======>......................] - ETA: 1:17 - loss: 0.3581 - categorical_accuracy: 0.8867
17600/60000 [=======>......................] - ETA: 1:17 - loss: 0.3585 - categorical_accuracy: 0.8866
17632/60000 [=======>......................] - ETA: 1:17 - loss: 0.3581 - categorical_accuracy: 0.8867
17664/60000 [=======>......................] - ETA: 1:17 - loss: 0.3579 - categorical_accuracy: 0.8868
17696/60000 [=======>......................] - ETA: 1:17 - loss: 0.3573 - categorical_accuracy: 0.8870
17728/60000 [=======>......................] - ETA: 1:17 - loss: 0.3569 - categorical_accuracy: 0.8871
17760/60000 [=======>......................] - ETA: 1:17 - loss: 0.3569 - categorical_accuracy: 0.8870
17792/60000 [=======>......................] - ETA: 1:16 - loss: 0.3564 - categorical_accuracy: 0.8871
17824/60000 [=======>......................] - ETA: 1:16 - loss: 0.3561 - categorical_accuracy: 0.8872
17856/60000 [=======>......................] - ETA: 1:16 - loss: 0.3561 - categorical_accuracy: 0.8873
17888/60000 [=======>......................] - ETA: 1:16 - loss: 0.3555 - categorical_accuracy: 0.8875
17920/60000 [=======>......................] - ETA: 1:16 - loss: 0.3549 - categorical_accuracy: 0.8877
17952/60000 [=======>......................] - ETA: 1:16 - loss: 0.3545 - categorical_accuracy: 0.8879
17984/60000 [=======>......................] - ETA: 1:16 - loss: 0.3547 - categorical_accuracy: 0.8878
18016/60000 [========>.....................] - ETA: 1:16 - loss: 0.3549 - categorical_accuracy: 0.8879
18048/60000 [========>.....................] - ETA: 1:16 - loss: 0.3547 - categorical_accuracy: 0.8880
18080/60000 [========>.....................] - ETA: 1:16 - loss: 0.3544 - categorical_accuracy: 0.8881
18112/60000 [========>.....................] - ETA: 1:16 - loss: 0.3540 - categorical_accuracy: 0.8881
18144/60000 [========>.....................] - ETA: 1:16 - loss: 0.3538 - categorical_accuracy: 0.8882
18176/60000 [========>.....................] - ETA: 1:16 - loss: 0.3533 - categorical_accuracy: 0.8884
18208/60000 [========>.....................] - ETA: 1:16 - loss: 0.3529 - categorical_accuracy: 0.8886
18240/60000 [========>.....................] - ETA: 1:16 - loss: 0.3523 - categorical_accuracy: 0.8887
18272/60000 [========>.....................] - ETA: 1:16 - loss: 0.3519 - categorical_accuracy: 0.8889
18304/60000 [========>.....................] - ETA: 1:16 - loss: 0.3518 - categorical_accuracy: 0.8889
18336/60000 [========>.....................] - ETA: 1:15 - loss: 0.3514 - categorical_accuracy: 0.8890
18368/60000 [========>.....................] - ETA: 1:15 - loss: 0.3511 - categorical_accuracy: 0.8892
18400/60000 [========>.....................] - ETA: 1:15 - loss: 0.3506 - categorical_accuracy: 0.8893
18432/60000 [========>.....................] - ETA: 1:15 - loss: 0.3500 - categorical_accuracy: 0.8895
18464/60000 [========>.....................] - ETA: 1:15 - loss: 0.3496 - categorical_accuracy: 0.8897
18496/60000 [========>.....................] - ETA: 1:15 - loss: 0.3491 - categorical_accuracy: 0.8899
18528/60000 [========>.....................] - ETA: 1:15 - loss: 0.3496 - categorical_accuracy: 0.8899
18560/60000 [========>.....................] - ETA: 1:15 - loss: 0.3492 - categorical_accuracy: 0.8900
18592/60000 [========>.....................] - ETA: 1:15 - loss: 0.3488 - categorical_accuracy: 0.8901
18624/60000 [========>.....................] - ETA: 1:15 - loss: 0.3485 - categorical_accuracy: 0.8901
18656/60000 [========>.....................] - ETA: 1:15 - loss: 0.3479 - categorical_accuracy: 0.8903
18688/60000 [========>.....................] - ETA: 1:15 - loss: 0.3475 - categorical_accuracy: 0.8905
18720/60000 [========>.....................] - ETA: 1:15 - loss: 0.3473 - categorical_accuracy: 0.8905
18752/60000 [========>.....................] - ETA: 1:15 - loss: 0.3470 - categorical_accuracy: 0.8907
18784/60000 [========>.....................] - ETA: 1:15 - loss: 0.3466 - categorical_accuracy: 0.8908
18816/60000 [========>.....................] - ETA: 1:15 - loss: 0.3461 - categorical_accuracy: 0.8910
18848/60000 [========>.....................] - ETA: 1:15 - loss: 0.3456 - categorical_accuracy: 0.8911
18880/60000 [========>.....................] - ETA: 1:14 - loss: 0.3451 - categorical_accuracy: 0.8913
18912/60000 [========>.....................] - ETA: 1:14 - loss: 0.3445 - categorical_accuracy: 0.8915
18944/60000 [========>.....................] - ETA: 1:14 - loss: 0.3442 - categorical_accuracy: 0.8915
18976/60000 [========>.....................] - ETA: 1:14 - loss: 0.3439 - categorical_accuracy: 0.8916
19008/60000 [========>.....................] - ETA: 1:14 - loss: 0.3435 - categorical_accuracy: 0.8917
19040/60000 [========>.....................] - ETA: 1:14 - loss: 0.3432 - categorical_accuracy: 0.8918
19072/60000 [========>.....................] - ETA: 1:14 - loss: 0.3428 - categorical_accuracy: 0.8919
19104/60000 [========>.....................] - ETA: 1:14 - loss: 0.3423 - categorical_accuracy: 0.8921
19136/60000 [========>.....................] - ETA: 1:14 - loss: 0.3423 - categorical_accuracy: 0.8921
19168/60000 [========>.....................] - ETA: 1:14 - loss: 0.3419 - categorical_accuracy: 0.8923
19200/60000 [========>.....................] - ETA: 1:14 - loss: 0.3419 - categorical_accuracy: 0.8923
19232/60000 [========>.....................] - ETA: 1:14 - loss: 0.3417 - categorical_accuracy: 0.8924
19264/60000 [========>.....................] - ETA: 1:14 - loss: 0.3414 - categorical_accuracy: 0.8925
19296/60000 [========>.....................] - ETA: 1:14 - loss: 0.3411 - categorical_accuracy: 0.8926
19328/60000 [========>.....................] - ETA: 1:14 - loss: 0.3408 - categorical_accuracy: 0.8927
19360/60000 [========>.....................] - ETA: 1:14 - loss: 0.3406 - categorical_accuracy: 0.8928
19392/60000 [========>.....................] - ETA: 1:14 - loss: 0.3411 - categorical_accuracy: 0.8928
19424/60000 [========>.....................] - ETA: 1:13 - loss: 0.3408 - categorical_accuracy: 0.8930
19456/60000 [========>.....................] - ETA: 1:13 - loss: 0.3404 - categorical_accuracy: 0.8931
19488/60000 [========>.....................] - ETA: 1:13 - loss: 0.3403 - categorical_accuracy: 0.8932
19520/60000 [========>.....................] - ETA: 1:13 - loss: 0.3403 - categorical_accuracy: 0.8932
19552/60000 [========>.....................] - ETA: 1:13 - loss: 0.3399 - categorical_accuracy: 0.8934
19584/60000 [========>.....................] - ETA: 1:13 - loss: 0.3396 - categorical_accuracy: 0.8934
19616/60000 [========>.....................] - ETA: 1:13 - loss: 0.3392 - categorical_accuracy: 0.8936
19648/60000 [========>.....................] - ETA: 1:13 - loss: 0.3389 - categorical_accuracy: 0.8936
19680/60000 [========>.....................] - ETA: 1:13 - loss: 0.3384 - categorical_accuracy: 0.8938
19712/60000 [========>.....................] - ETA: 1:13 - loss: 0.3381 - categorical_accuracy: 0.8939
19744/60000 [========>.....................] - ETA: 1:13 - loss: 0.3378 - categorical_accuracy: 0.8939
19776/60000 [========>.....................] - ETA: 1:13 - loss: 0.3374 - categorical_accuracy: 0.8941
19808/60000 [========>.....................] - ETA: 1:13 - loss: 0.3370 - categorical_accuracy: 0.8942
19840/60000 [========>.....................] - ETA: 1:13 - loss: 0.3368 - categorical_accuracy: 0.8943
19872/60000 [========>.....................] - ETA: 1:13 - loss: 0.3363 - categorical_accuracy: 0.8944
19904/60000 [========>.....................] - ETA: 1:13 - loss: 0.3362 - categorical_accuracy: 0.8943
19936/60000 [========>.....................] - ETA: 1:13 - loss: 0.3359 - categorical_accuracy: 0.8944
19968/60000 [========>.....................] - ETA: 1:13 - loss: 0.3356 - categorical_accuracy: 0.8945
20000/60000 [=========>....................] - ETA: 1:12 - loss: 0.3353 - categorical_accuracy: 0.8946
20032/60000 [=========>....................] - ETA: 1:12 - loss: 0.3349 - categorical_accuracy: 0.8947
20064/60000 [=========>....................] - ETA: 1:12 - loss: 0.3348 - categorical_accuracy: 0.8947
20096/60000 [=========>....................] - ETA: 1:12 - loss: 0.3343 - categorical_accuracy: 0.8949
20128/60000 [=========>....................] - ETA: 1:12 - loss: 0.3342 - categorical_accuracy: 0.8950
20160/60000 [=========>....................] - ETA: 1:12 - loss: 0.3341 - categorical_accuracy: 0.8950
20192/60000 [=========>....................] - ETA: 1:12 - loss: 0.3338 - categorical_accuracy: 0.8951
20224/60000 [=========>....................] - ETA: 1:12 - loss: 0.3336 - categorical_accuracy: 0.8952
20256/60000 [=========>....................] - ETA: 1:12 - loss: 0.3335 - categorical_accuracy: 0.8952
20288/60000 [=========>....................] - ETA: 1:12 - loss: 0.3331 - categorical_accuracy: 0.8954
20320/60000 [=========>....................] - ETA: 1:12 - loss: 0.3327 - categorical_accuracy: 0.8955
20352/60000 [=========>....................] - ETA: 1:12 - loss: 0.3325 - categorical_accuracy: 0.8954
20384/60000 [=========>....................] - ETA: 1:12 - loss: 0.3323 - categorical_accuracy: 0.8955
20416/60000 [=========>....................] - ETA: 1:12 - loss: 0.3322 - categorical_accuracy: 0.8955
20448/60000 [=========>....................] - ETA: 1:12 - loss: 0.3317 - categorical_accuracy: 0.8956
20480/60000 [=========>....................] - ETA: 1:12 - loss: 0.3315 - categorical_accuracy: 0.8957
20512/60000 [=========>....................] - ETA: 1:12 - loss: 0.3312 - categorical_accuracy: 0.8958
20544/60000 [=========>....................] - ETA: 1:11 - loss: 0.3308 - categorical_accuracy: 0.8959
20576/60000 [=========>....................] - ETA: 1:11 - loss: 0.3305 - categorical_accuracy: 0.8959
20608/60000 [=========>....................] - ETA: 1:11 - loss: 0.3302 - categorical_accuracy: 0.8960
20640/60000 [=========>....................] - ETA: 1:11 - loss: 0.3298 - categorical_accuracy: 0.8961
20672/60000 [=========>....................] - ETA: 1:11 - loss: 0.3296 - categorical_accuracy: 0.8962
20704/60000 [=========>....................] - ETA: 1:11 - loss: 0.3294 - categorical_accuracy: 0.8963
20736/60000 [=========>....................] - ETA: 1:11 - loss: 0.3292 - categorical_accuracy: 0.8963
20768/60000 [=========>....................] - ETA: 1:11 - loss: 0.3291 - categorical_accuracy: 0.8964
20800/60000 [=========>....................] - ETA: 1:11 - loss: 0.3288 - categorical_accuracy: 0.8965
20832/60000 [=========>....................] - ETA: 1:11 - loss: 0.3284 - categorical_accuracy: 0.8966
20864/60000 [=========>....................] - ETA: 1:11 - loss: 0.3279 - categorical_accuracy: 0.8968
20896/60000 [=========>....................] - ETA: 1:11 - loss: 0.3278 - categorical_accuracy: 0.8968
20928/60000 [=========>....................] - ETA: 1:11 - loss: 0.3275 - categorical_accuracy: 0.8968
20960/60000 [=========>....................] - ETA: 1:11 - loss: 0.3273 - categorical_accuracy: 0.8969
20992/60000 [=========>....................] - ETA: 1:11 - loss: 0.3270 - categorical_accuracy: 0.8970
21024/60000 [=========>....................] - ETA: 1:11 - loss: 0.3266 - categorical_accuracy: 0.8971
21056/60000 [=========>....................] - ETA: 1:10 - loss: 0.3264 - categorical_accuracy: 0.8972
21088/60000 [=========>....................] - ETA: 1:10 - loss: 0.3259 - categorical_accuracy: 0.8973
21120/60000 [=========>....................] - ETA: 1:10 - loss: 0.3255 - categorical_accuracy: 0.8975
21152/60000 [=========>....................] - ETA: 1:10 - loss: 0.3251 - categorical_accuracy: 0.8976
21184/60000 [=========>....................] - ETA: 1:10 - loss: 0.3246 - categorical_accuracy: 0.8978
21216/60000 [=========>....................] - ETA: 1:10 - loss: 0.3243 - categorical_accuracy: 0.8979
21248/60000 [=========>....................] - ETA: 1:10 - loss: 0.3239 - categorical_accuracy: 0.8980
21280/60000 [=========>....................] - ETA: 1:10 - loss: 0.3235 - categorical_accuracy: 0.8981
21312/60000 [=========>....................] - ETA: 1:10 - loss: 0.3235 - categorical_accuracy: 0.8982
21344/60000 [=========>....................] - ETA: 1:10 - loss: 0.3231 - categorical_accuracy: 0.8983
21376/60000 [=========>....................] - ETA: 1:10 - loss: 0.3228 - categorical_accuracy: 0.8983
21408/60000 [=========>....................] - ETA: 1:10 - loss: 0.3226 - categorical_accuracy: 0.8984
21440/60000 [=========>....................] - ETA: 1:10 - loss: 0.3222 - categorical_accuracy: 0.8986
21472/60000 [=========>....................] - ETA: 1:10 - loss: 0.3217 - categorical_accuracy: 0.8987
21504/60000 [=========>....................] - ETA: 1:10 - loss: 0.3215 - categorical_accuracy: 0.8988
21536/60000 [=========>....................] - ETA: 1:10 - loss: 0.3212 - categorical_accuracy: 0.8989
21568/60000 [=========>....................] - ETA: 1:10 - loss: 0.3211 - categorical_accuracy: 0.8989
21600/60000 [=========>....................] - ETA: 1:09 - loss: 0.3207 - categorical_accuracy: 0.8990
21632/60000 [=========>....................] - ETA: 1:09 - loss: 0.3205 - categorical_accuracy: 0.8990
21664/60000 [=========>....................] - ETA: 1:09 - loss: 0.3203 - categorical_accuracy: 0.8991
21696/60000 [=========>....................] - ETA: 1:09 - loss: 0.3202 - categorical_accuracy: 0.8991
21728/60000 [=========>....................] - ETA: 1:09 - loss: 0.3202 - categorical_accuracy: 0.8992
21760/60000 [=========>....................] - ETA: 1:09 - loss: 0.3199 - categorical_accuracy: 0.8992
21792/60000 [=========>....................] - ETA: 1:09 - loss: 0.3195 - categorical_accuracy: 0.8994
21824/60000 [=========>....................] - ETA: 1:09 - loss: 0.3193 - categorical_accuracy: 0.8994
21856/60000 [=========>....................] - ETA: 1:09 - loss: 0.3190 - categorical_accuracy: 0.8995
21888/60000 [=========>....................] - ETA: 1:09 - loss: 0.3187 - categorical_accuracy: 0.8996
21920/60000 [=========>....................] - ETA: 1:09 - loss: 0.3187 - categorical_accuracy: 0.8995
21952/60000 [=========>....................] - ETA: 1:09 - loss: 0.3195 - categorical_accuracy: 0.8995
21984/60000 [=========>....................] - ETA: 1:09 - loss: 0.3192 - categorical_accuracy: 0.8996
22016/60000 [==========>...................] - ETA: 1:09 - loss: 0.3188 - categorical_accuracy: 0.8996
22048/60000 [==========>...................] - ETA: 1:09 - loss: 0.3184 - categorical_accuracy: 0.8997
22080/60000 [==========>...................] - ETA: 1:09 - loss: 0.3180 - categorical_accuracy: 0.8999
22112/60000 [==========>...................] - ETA: 1:09 - loss: 0.3178 - categorical_accuracy: 0.8999
22144/60000 [==========>...................] - ETA: 1:09 - loss: 0.3177 - categorical_accuracy: 0.8999
22176/60000 [==========>...................] - ETA: 1:08 - loss: 0.3176 - categorical_accuracy: 0.9000
22208/60000 [==========>...................] - ETA: 1:08 - loss: 0.3172 - categorical_accuracy: 0.9001
22240/60000 [==========>...................] - ETA: 1:08 - loss: 0.3175 - categorical_accuracy: 0.9001
22272/60000 [==========>...................] - ETA: 1:08 - loss: 0.3172 - categorical_accuracy: 0.9001
22304/60000 [==========>...................] - ETA: 1:08 - loss: 0.3171 - categorical_accuracy: 0.9002
22336/60000 [==========>...................] - ETA: 1:08 - loss: 0.3167 - categorical_accuracy: 0.9003
22368/60000 [==========>...................] - ETA: 1:08 - loss: 0.3165 - categorical_accuracy: 0.9004
22400/60000 [==========>...................] - ETA: 1:08 - loss: 0.3161 - categorical_accuracy: 0.9005
22432/60000 [==========>...................] - ETA: 1:08 - loss: 0.3158 - categorical_accuracy: 0.9007
22464/60000 [==========>...................] - ETA: 1:08 - loss: 0.3154 - categorical_accuracy: 0.9008
22496/60000 [==========>...................] - ETA: 1:08 - loss: 0.3151 - categorical_accuracy: 0.9009
22528/60000 [==========>...................] - ETA: 1:08 - loss: 0.3147 - categorical_accuracy: 0.9010
22560/60000 [==========>...................] - ETA: 1:08 - loss: 0.3143 - categorical_accuracy: 0.9012
22592/60000 [==========>...................] - ETA: 1:08 - loss: 0.3141 - categorical_accuracy: 0.9012
22624/60000 [==========>...................] - ETA: 1:08 - loss: 0.3137 - categorical_accuracy: 0.9013
22656/60000 [==========>...................] - ETA: 1:08 - loss: 0.3134 - categorical_accuracy: 0.9014
22688/60000 [==========>...................] - ETA: 1:07 - loss: 0.3132 - categorical_accuracy: 0.9015
22720/60000 [==========>...................] - ETA: 1:07 - loss: 0.3131 - categorical_accuracy: 0.9015
22752/60000 [==========>...................] - ETA: 1:07 - loss: 0.3129 - categorical_accuracy: 0.9016
22784/60000 [==========>...................] - ETA: 1:07 - loss: 0.3126 - categorical_accuracy: 0.9017
22816/60000 [==========>...................] - ETA: 1:07 - loss: 0.3124 - categorical_accuracy: 0.9018
22848/60000 [==========>...................] - ETA: 1:07 - loss: 0.3121 - categorical_accuracy: 0.9018
22880/60000 [==========>...................] - ETA: 1:07 - loss: 0.3118 - categorical_accuracy: 0.9019
22912/60000 [==========>...................] - ETA: 1:07 - loss: 0.3119 - categorical_accuracy: 0.9019
22944/60000 [==========>...................] - ETA: 1:07 - loss: 0.3116 - categorical_accuracy: 0.9020
22976/60000 [==========>...................] - ETA: 1:07 - loss: 0.3117 - categorical_accuracy: 0.9021
23008/60000 [==========>...................] - ETA: 1:07 - loss: 0.3117 - categorical_accuracy: 0.9021
23040/60000 [==========>...................] - ETA: 1:07 - loss: 0.3114 - categorical_accuracy: 0.9022
23072/60000 [==========>...................] - ETA: 1:07 - loss: 0.3111 - categorical_accuracy: 0.9023
23104/60000 [==========>...................] - ETA: 1:07 - loss: 0.3107 - categorical_accuracy: 0.9024
23136/60000 [==========>...................] - ETA: 1:07 - loss: 0.3104 - categorical_accuracy: 0.9025
23168/60000 [==========>...................] - ETA: 1:07 - loss: 0.3107 - categorical_accuracy: 0.9025
23200/60000 [==========>...................] - ETA: 1:07 - loss: 0.3104 - categorical_accuracy: 0.9026
23232/60000 [==========>...................] - ETA: 1:07 - loss: 0.3101 - categorical_accuracy: 0.9027
23264/60000 [==========>...................] - ETA: 1:06 - loss: 0.3098 - categorical_accuracy: 0.9028
23296/60000 [==========>...................] - ETA: 1:06 - loss: 0.3098 - categorical_accuracy: 0.9028
23328/60000 [==========>...................] - ETA: 1:06 - loss: 0.3095 - categorical_accuracy: 0.9029
23360/60000 [==========>...................] - ETA: 1:06 - loss: 0.3093 - categorical_accuracy: 0.9030
23392/60000 [==========>...................] - ETA: 1:06 - loss: 0.3090 - categorical_accuracy: 0.9030
23424/60000 [==========>...................] - ETA: 1:06 - loss: 0.3086 - categorical_accuracy: 0.9031
23456/60000 [==========>...................] - ETA: 1:06 - loss: 0.3086 - categorical_accuracy: 0.9032
23488/60000 [==========>...................] - ETA: 1:06 - loss: 0.3082 - categorical_accuracy: 0.9034
23520/60000 [==========>...................] - ETA: 1:06 - loss: 0.3080 - categorical_accuracy: 0.9034
23552/60000 [==========>...................] - ETA: 1:06 - loss: 0.3076 - categorical_accuracy: 0.9036
23584/60000 [==========>...................] - ETA: 1:06 - loss: 0.3072 - categorical_accuracy: 0.9037
23616/60000 [==========>...................] - ETA: 1:06 - loss: 0.3072 - categorical_accuracy: 0.9038
23648/60000 [==========>...................] - ETA: 1:06 - loss: 0.3069 - categorical_accuracy: 0.9039
23680/60000 [==========>...................] - ETA: 1:06 - loss: 0.3067 - categorical_accuracy: 0.9039
23712/60000 [==========>...................] - ETA: 1:06 - loss: 0.3064 - categorical_accuracy: 0.9040
23744/60000 [==========>...................] - ETA: 1:06 - loss: 0.3061 - categorical_accuracy: 0.9041
23776/60000 [==========>...................] - ETA: 1:05 - loss: 0.3061 - categorical_accuracy: 0.9041
23808/60000 [==========>...................] - ETA: 1:05 - loss: 0.3057 - categorical_accuracy: 0.9042
23840/60000 [==========>...................] - ETA: 1:05 - loss: 0.3054 - categorical_accuracy: 0.9043
23872/60000 [==========>...................] - ETA: 1:05 - loss: 0.3050 - categorical_accuracy: 0.9044
23904/60000 [==========>...................] - ETA: 1:05 - loss: 0.3047 - categorical_accuracy: 0.9045
23936/60000 [==========>...................] - ETA: 1:05 - loss: 0.3045 - categorical_accuracy: 0.9046
23968/60000 [==========>...................] - ETA: 1:05 - loss: 0.3042 - categorical_accuracy: 0.9047
24000/60000 [===========>..................] - ETA: 1:05 - loss: 0.3041 - categorical_accuracy: 0.9048
24032/60000 [===========>..................] - ETA: 1:05 - loss: 0.3044 - categorical_accuracy: 0.9048
24064/60000 [===========>..................] - ETA: 1:05 - loss: 0.3044 - categorical_accuracy: 0.9047
24096/60000 [===========>..................] - ETA: 1:05 - loss: 0.3040 - categorical_accuracy: 0.9048
24128/60000 [===========>..................] - ETA: 1:05 - loss: 0.3039 - categorical_accuracy: 0.9049
24160/60000 [===========>..................] - ETA: 1:05 - loss: 0.3036 - categorical_accuracy: 0.9050
24192/60000 [===========>..................] - ETA: 1:05 - loss: 0.3035 - categorical_accuracy: 0.9051
24224/60000 [===========>..................] - ETA: 1:05 - loss: 0.3031 - categorical_accuracy: 0.9051
24256/60000 [===========>..................] - ETA: 1:05 - loss: 0.3028 - categorical_accuracy: 0.9052
24288/60000 [===========>..................] - ETA: 1:05 - loss: 0.3025 - categorical_accuracy: 0.9053
24320/60000 [===========>..................] - ETA: 1:04 - loss: 0.3024 - categorical_accuracy: 0.9053
24352/60000 [===========>..................] - ETA: 1:04 - loss: 0.3021 - categorical_accuracy: 0.9054
24384/60000 [===========>..................] - ETA: 1:04 - loss: 0.3017 - categorical_accuracy: 0.9055
24416/60000 [===========>..................] - ETA: 1:04 - loss: 0.3015 - categorical_accuracy: 0.9056
24448/60000 [===========>..................] - ETA: 1:04 - loss: 0.3011 - categorical_accuracy: 0.9057
24480/60000 [===========>..................] - ETA: 1:04 - loss: 0.3008 - categorical_accuracy: 0.9058
24512/60000 [===========>..................] - ETA: 1:04 - loss: 0.3006 - categorical_accuracy: 0.9059
24544/60000 [===========>..................] - ETA: 1:04 - loss: 0.3004 - categorical_accuracy: 0.9060
24576/60000 [===========>..................] - ETA: 1:04 - loss: 0.3000 - categorical_accuracy: 0.9061
24608/60000 [===========>..................] - ETA: 1:04 - loss: 0.2997 - categorical_accuracy: 0.9062
24640/60000 [===========>..................] - ETA: 1:04 - loss: 0.2997 - categorical_accuracy: 0.9060
24672/60000 [===========>..................] - ETA: 1:04 - loss: 0.2994 - categorical_accuracy: 0.9061
24704/60000 [===========>..................] - ETA: 1:04 - loss: 0.2994 - categorical_accuracy: 0.9061
24736/60000 [===========>..................] - ETA: 1:04 - loss: 0.2993 - categorical_accuracy: 0.9062
24768/60000 [===========>..................] - ETA: 1:04 - loss: 0.2990 - categorical_accuracy: 0.9062
24800/60000 [===========>..................] - ETA: 1:04 - loss: 0.2988 - categorical_accuracy: 0.9063
24832/60000 [===========>..................] - ETA: 1:04 - loss: 0.2986 - categorical_accuracy: 0.9064
24864/60000 [===========>..................] - ETA: 1:04 - loss: 0.2982 - categorical_accuracy: 0.9065
24896/60000 [===========>..................] - ETA: 1:03 - loss: 0.2979 - categorical_accuracy: 0.9066
24928/60000 [===========>..................] - ETA: 1:03 - loss: 0.2978 - categorical_accuracy: 0.9066
24960/60000 [===========>..................] - ETA: 1:03 - loss: 0.2977 - categorical_accuracy: 0.9067
24992/60000 [===========>..................] - ETA: 1:03 - loss: 0.2973 - categorical_accuracy: 0.9068
25024/60000 [===========>..................] - ETA: 1:03 - loss: 0.2973 - categorical_accuracy: 0.9068
25056/60000 [===========>..................] - ETA: 1:03 - loss: 0.2971 - categorical_accuracy: 0.9069
25088/60000 [===========>..................] - ETA: 1:03 - loss: 0.2970 - categorical_accuracy: 0.9069
25120/60000 [===========>..................] - ETA: 1:03 - loss: 0.2969 - categorical_accuracy: 0.9069
25152/60000 [===========>..................] - ETA: 1:03 - loss: 0.2967 - categorical_accuracy: 0.9070
25184/60000 [===========>..................] - ETA: 1:03 - loss: 0.2965 - categorical_accuracy: 0.9070
25216/60000 [===========>..................] - ETA: 1:03 - loss: 0.2963 - categorical_accuracy: 0.9071
25248/60000 [===========>..................] - ETA: 1:03 - loss: 0.2960 - categorical_accuracy: 0.9072
25280/60000 [===========>..................] - ETA: 1:03 - loss: 0.2957 - categorical_accuracy: 0.9073
25312/60000 [===========>..................] - ETA: 1:03 - loss: 0.2958 - categorical_accuracy: 0.9073
25344/60000 [===========>..................] - ETA: 1:03 - loss: 0.2954 - categorical_accuracy: 0.9074
25376/60000 [===========>..................] - ETA: 1:03 - loss: 0.2953 - categorical_accuracy: 0.9074
25408/60000 [===========>..................] - ETA: 1:03 - loss: 0.2950 - categorical_accuracy: 0.9074
25440/60000 [===========>..................] - ETA: 1:02 - loss: 0.2947 - categorical_accuracy: 0.9075
25472/60000 [===========>..................] - ETA: 1:02 - loss: 0.2945 - categorical_accuracy: 0.9076
25504/60000 [===========>..................] - ETA: 1:02 - loss: 0.2942 - categorical_accuracy: 0.9077
25536/60000 [===========>..................] - ETA: 1:02 - loss: 0.2939 - categorical_accuracy: 0.9078
25568/60000 [===========>..................] - ETA: 1:02 - loss: 0.2936 - categorical_accuracy: 0.9079
25600/60000 [===========>..................] - ETA: 1:02 - loss: 0.2938 - categorical_accuracy: 0.9079
25632/60000 [===========>..................] - ETA: 1:02 - loss: 0.2936 - categorical_accuracy: 0.9079
25664/60000 [===========>..................] - ETA: 1:02 - loss: 0.2933 - categorical_accuracy: 0.9080
25696/60000 [===========>..................] - ETA: 1:02 - loss: 0.2932 - categorical_accuracy: 0.9081
25728/60000 [===========>..................] - ETA: 1:02 - loss: 0.2930 - categorical_accuracy: 0.9081
25760/60000 [===========>..................] - ETA: 1:02 - loss: 0.2927 - categorical_accuracy: 0.9082
25792/60000 [===========>..................] - ETA: 1:02 - loss: 0.2924 - categorical_accuracy: 0.9082
25824/60000 [===========>..................] - ETA: 1:02 - loss: 0.2921 - categorical_accuracy: 0.9083
25856/60000 [===========>..................] - ETA: 1:02 - loss: 0.2918 - categorical_accuracy: 0.9085
25888/60000 [===========>..................] - ETA: 1:02 - loss: 0.2921 - categorical_accuracy: 0.9085
25920/60000 [===========>..................] - ETA: 1:02 - loss: 0.2917 - categorical_accuracy: 0.9086
25952/60000 [===========>..................] - ETA: 1:02 - loss: 0.2916 - categorical_accuracy: 0.9086
25984/60000 [===========>..................] - ETA: 1:01 - loss: 0.2915 - categorical_accuracy: 0.9087
26016/60000 [============>.................] - ETA: 1:01 - loss: 0.2912 - categorical_accuracy: 0.9087
26048/60000 [============>.................] - ETA: 1:01 - loss: 0.2910 - categorical_accuracy: 0.9088
26080/60000 [============>.................] - ETA: 1:01 - loss: 0.2909 - categorical_accuracy: 0.9089
26112/60000 [============>.................] - ETA: 1:01 - loss: 0.2906 - categorical_accuracy: 0.9090
26144/60000 [============>.................] - ETA: 1:01 - loss: 0.2908 - categorical_accuracy: 0.9090
26176/60000 [============>.................] - ETA: 1:01 - loss: 0.2905 - categorical_accuracy: 0.9091
26208/60000 [============>.................] - ETA: 1:01 - loss: 0.2903 - categorical_accuracy: 0.9091
26240/60000 [============>.................] - ETA: 1:01 - loss: 0.2900 - categorical_accuracy: 0.9093
26272/60000 [============>.................] - ETA: 1:01 - loss: 0.2897 - categorical_accuracy: 0.9093
26304/60000 [============>.................] - ETA: 1:01 - loss: 0.2894 - categorical_accuracy: 0.9094
26336/60000 [============>.................] - ETA: 1:01 - loss: 0.2891 - categorical_accuracy: 0.9095
26368/60000 [============>.................] - ETA: 1:01 - loss: 0.2888 - categorical_accuracy: 0.9096
26400/60000 [============>.................] - ETA: 1:01 - loss: 0.2885 - categorical_accuracy: 0.9097
26432/60000 [============>.................] - ETA: 1:01 - loss: 0.2884 - categorical_accuracy: 0.9097
26464/60000 [============>.................] - ETA: 1:01 - loss: 0.2881 - categorical_accuracy: 0.9098
26496/60000 [============>.................] - ETA: 1:01 - loss: 0.2880 - categorical_accuracy: 0.9098
26528/60000 [============>.................] - ETA: 1:00 - loss: 0.2880 - categorical_accuracy: 0.9099
26560/60000 [============>.................] - ETA: 1:00 - loss: 0.2877 - categorical_accuracy: 0.9100
26592/60000 [============>.................] - ETA: 1:00 - loss: 0.2876 - categorical_accuracy: 0.9100
26624/60000 [============>.................] - ETA: 1:00 - loss: 0.2875 - categorical_accuracy: 0.9100
26656/60000 [============>.................] - ETA: 1:00 - loss: 0.2872 - categorical_accuracy: 0.9102
26688/60000 [============>.................] - ETA: 1:00 - loss: 0.2870 - categorical_accuracy: 0.9102
26720/60000 [============>.................] - ETA: 1:00 - loss: 0.2867 - categorical_accuracy: 0.9103
26752/60000 [============>.................] - ETA: 1:00 - loss: 0.2864 - categorical_accuracy: 0.9104
26784/60000 [============>.................] - ETA: 1:00 - loss: 0.2862 - categorical_accuracy: 0.9105
26816/60000 [============>.................] - ETA: 1:00 - loss: 0.2859 - categorical_accuracy: 0.9106
26848/60000 [============>.................] - ETA: 1:00 - loss: 0.2857 - categorical_accuracy: 0.9106
26880/60000 [============>.................] - ETA: 1:00 - loss: 0.2857 - categorical_accuracy: 0.9106
26912/60000 [============>.................] - ETA: 1:00 - loss: 0.2859 - categorical_accuracy: 0.9106
26944/60000 [============>.................] - ETA: 1:00 - loss: 0.2856 - categorical_accuracy: 0.9107
26976/60000 [============>.................] - ETA: 1:00 - loss: 0.2852 - categorical_accuracy: 0.9108
27008/60000 [============>.................] - ETA: 1:00 - loss: 0.2850 - categorical_accuracy: 0.9109
27040/60000 [============>.................] - ETA: 1:00 - loss: 0.2847 - categorical_accuracy: 0.9110
27072/60000 [============>.................] - ETA: 1:00 - loss: 0.2845 - categorical_accuracy: 0.9111
27104/60000 [============>.................] - ETA: 59s - loss: 0.2844 - categorical_accuracy: 0.9111 
27136/60000 [============>.................] - ETA: 59s - loss: 0.2842 - categorical_accuracy: 0.9112
27168/60000 [============>.................] - ETA: 59s - loss: 0.2839 - categorical_accuracy: 0.9113
27200/60000 [============>.................] - ETA: 59s - loss: 0.2837 - categorical_accuracy: 0.9113
27232/60000 [============>.................] - ETA: 59s - loss: 0.2836 - categorical_accuracy: 0.9114
27264/60000 [============>.................] - ETA: 59s - loss: 0.2833 - categorical_accuracy: 0.9114
27296/60000 [============>.................] - ETA: 59s - loss: 0.2831 - categorical_accuracy: 0.9115
27328/60000 [============>.................] - ETA: 59s - loss: 0.2828 - categorical_accuracy: 0.9116
27360/60000 [============>.................] - ETA: 59s - loss: 0.2826 - categorical_accuracy: 0.9116
27392/60000 [============>.................] - ETA: 59s - loss: 0.2825 - categorical_accuracy: 0.9117
27424/60000 [============>.................] - ETA: 59s - loss: 0.2822 - categorical_accuracy: 0.9118
27456/60000 [============>.................] - ETA: 59s - loss: 0.2819 - categorical_accuracy: 0.9119
27488/60000 [============>.................] - ETA: 59s - loss: 0.2816 - categorical_accuracy: 0.9120
27520/60000 [============>.................] - ETA: 59s - loss: 0.2817 - categorical_accuracy: 0.9120
27552/60000 [============>.................] - ETA: 59s - loss: 0.2815 - categorical_accuracy: 0.9121
27584/60000 [============>.................] - ETA: 59s - loss: 0.2812 - categorical_accuracy: 0.9122
27616/60000 [============>.................] - ETA: 59s - loss: 0.2811 - categorical_accuracy: 0.9122
27648/60000 [============>.................] - ETA: 58s - loss: 0.2808 - categorical_accuracy: 0.9123
27680/60000 [============>.................] - ETA: 58s - loss: 0.2806 - categorical_accuracy: 0.9123
27712/60000 [============>.................] - ETA: 58s - loss: 0.2805 - categorical_accuracy: 0.9123
27744/60000 [============>.................] - ETA: 58s - loss: 0.2806 - categorical_accuracy: 0.9123
27776/60000 [============>.................] - ETA: 58s - loss: 0.2804 - categorical_accuracy: 0.9124
27808/60000 [============>.................] - ETA: 58s - loss: 0.2802 - categorical_accuracy: 0.9124
27840/60000 [============>.................] - ETA: 58s - loss: 0.2801 - categorical_accuracy: 0.9125
27872/60000 [============>.................] - ETA: 58s - loss: 0.2800 - categorical_accuracy: 0.9125
27904/60000 [============>.................] - ETA: 58s - loss: 0.2797 - categorical_accuracy: 0.9126
27936/60000 [============>.................] - ETA: 58s - loss: 0.2794 - categorical_accuracy: 0.9127
27968/60000 [============>.................] - ETA: 58s - loss: 0.2793 - categorical_accuracy: 0.9127
28000/60000 [=============>................] - ETA: 58s - loss: 0.2792 - categorical_accuracy: 0.9127
28032/60000 [=============>................] - ETA: 58s - loss: 0.2789 - categorical_accuracy: 0.9128
28064/60000 [=============>................] - ETA: 58s - loss: 0.2786 - categorical_accuracy: 0.9129
28096/60000 [=============>................] - ETA: 58s - loss: 0.2783 - categorical_accuracy: 0.9130
28128/60000 [=============>................] - ETA: 58s - loss: 0.2782 - categorical_accuracy: 0.9130
28160/60000 [=============>................] - ETA: 58s - loss: 0.2780 - categorical_accuracy: 0.9131
28192/60000 [=============>................] - ETA: 58s - loss: 0.2781 - categorical_accuracy: 0.9131
28224/60000 [=============>................] - ETA: 57s - loss: 0.2778 - categorical_accuracy: 0.9132
28256/60000 [=============>................] - ETA: 57s - loss: 0.2777 - categorical_accuracy: 0.9132
28288/60000 [=============>................] - ETA: 57s - loss: 0.2775 - categorical_accuracy: 0.9132
28320/60000 [=============>................] - ETA: 57s - loss: 0.2773 - categorical_accuracy: 0.9133
28352/60000 [=============>................] - ETA: 57s - loss: 0.2774 - categorical_accuracy: 0.9133
28384/60000 [=============>................] - ETA: 57s - loss: 0.2772 - categorical_accuracy: 0.9133
28416/60000 [=============>................] - ETA: 57s - loss: 0.2769 - categorical_accuracy: 0.9134
28448/60000 [=============>................] - ETA: 57s - loss: 0.2766 - categorical_accuracy: 0.9135
28480/60000 [=============>................] - ETA: 57s - loss: 0.2763 - categorical_accuracy: 0.9136
28512/60000 [=============>................] - ETA: 57s - loss: 0.2763 - categorical_accuracy: 0.9136
28544/60000 [=============>................] - ETA: 57s - loss: 0.2762 - categorical_accuracy: 0.9136
28576/60000 [=============>................] - ETA: 57s - loss: 0.2760 - categorical_accuracy: 0.9136
28608/60000 [=============>................] - ETA: 57s - loss: 0.2758 - categorical_accuracy: 0.9137
28640/60000 [=============>................] - ETA: 57s - loss: 0.2760 - categorical_accuracy: 0.9137
28672/60000 [=============>................] - ETA: 57s - loss: 0.2758 - categorical_accuracy: 0.9137
28704/60000 [=============>................] - ETA: 57s - loss: 0.2756 - categorical_accuracy: 0.9138
28736/60000 [=============>................] - ETA: 57s - loss: 0.2754 - categorical_accuracy: 0.9138
28768/60000 [=============>................] - ETA: 56s - loss: 0.2753 - categorical_accuracy: 0.9139
28800/60000 [=============>................] - ETA: 56s - loss: 0.2751 - categorical_accuracy: 0.9139
28832/60000 [=============>................] - ETA: 56s - loss: 0.2748 - categorical_accuracy: 0.9140
28864/60000 [=============>................] - ETA: 56s - loss: 0.2747 - categorical_accuracy: 0.9140
28896/60000 [=============>................] - ETA: 56s - loss: 0.2744 - categorical_accuracy: 0.9141
28928/60000 [=============>................] - ETA: 56s - loss: 0.2744 - categorical_accuracy: 0.9142
28960/60000 [=============>................] - ETA: 56s - loss: 0.2742 - categorical_accuracy: 0.9142
28992/60000 [=============>................] - ETA: 56s - loss: 0.2741 - categorical_accuracy: 0.9142
29024/60000 [=============>................] - ETA: 56s - loss: 0.2738 - categorical_accuracy: 0.9143
29056/60000 [=============>................] - ETA: 56s - loss: 0.2737 - categorical_accuracy: 0.9143
29088/60000 [=============>................] - ETA: 56s - loss: 0.2736 - categorical_accuracy: 0.9144
29120/60000 [=============>................] - ETA: 56s - loss: 0.2733 - categorical_accuracy: 0.9145
29152/60000 [=============>................] - ETA: 56s - loss: 0.2730 - categorical_accuracy: 0.9146
29184/60000 [=============>................] - ETA: 56s - loss: 0.2728 - categorical_accuracy: 0.9146
29216/60000 [=============>................] - ETA: 56s - loss: 0.2726 - categorical_accuracy: 0.9147
29248/60000 [=============>................] - ETA: 56s - loss: 0.2724 - categorical_accuracy: 0.9147
29280/60000 [=============>................] - ETA: 55s - loss: 0.2723 - categorical_accuracy: 0.9147
29312/60000 [=============>................] - ETA: 55s - loss: 0.2722 - categorical_accuracy: 0.9147
29344/60000 [=============>................] - ETA: 55s - loss: 0.2720 - categorical_accuracy: 0.9148
29376/60000 [=============>................] - ETA: 55s - loss: 0.2718 - categorical_accuracy: 0.9149
29408/60000 [=============>................] - ETA: 55s - loss: 0.2716 - categorical_accuracy: 0.9149
29440/60000 [=============>................] - ETA: 55s - loss: 0.2714 - categorical_accuracy: 0.9149
29472/60000 [=============>................] - ETA: 55s - loss: 0.2714 - categorical_accuracy: 0.9149
29504/60000 [=============>................] - ETA: 55s - loss: 0.2711 - categorical_accuracy: 0.9150
29536/60000 [=============>................] - ETA: 55s - loss: 0.2709 - categorical_accuracy: 0.9151
29568/60000 [=============>................] - ETA: 55s - loss: 0.2707 - categorical_accuracy: 0.9151
29600/60000 [=============>................] - ETA: 55s - loss: 0.2704 - categorical_accuracy: 0.9152
29632/60000 [=============>................] - ETA: 55s - loss: 0.2703 - categorical_accuracy: 0.9152
29664/60000 [=============>................] - ETA: 55s - loss: 0.2701 - categorical_accuracy: 0.9153
29696/60000 [=============>................] - ETA: 55s - loss: 0.2704 - categorical_accuracy: 0.9153
29728/60000 [=============>................] - ETA: 55s - loss: 0.2701 - categorical_accuracy: 0.9153
29760/60000 [=============>................] - ETA: 55s - loss: 0.2700 - categorical_accuracy: 0.9154
29792/60000 [=============>................] - ETA: 55s - loss: 0.2699 - categorical_accuracy: 0.9154
29824/60000 [=============>................] - ETA: 54s - loss: 0.2698 - categorical_accuracy: 0.9154
29856/60000 [=============>................] - ETA: 54s - loss: 0.2696 - categorical_accuracy: 0.9155
29888/60000 [=============>................] - ETA: 54s - loss: 0.2696 - categorical_accuracy: 0.9156
29920/60000 [=============>................] - ETA: 54s - loss: 0.2693 - categorical_accuracy: 0.9156
29952/60000 [=============>................] - ETA: 54s - loss: 0.2693 - categorical_accuracy: 0.9157
29984/60000 [=============>................] - ETA: 54s - loss: 0.2691 - categorical_accuracy: 0.9157
30016/60000 [==============>...............] - ETA: 54s - loss: 0.2688 - categorical_accuracy: 0.9158
30048/60000 [==============>...............] - ETA: 54s - loss: 0.2688 - categorical_accuracy: 0.9159
30080/60000 [==============>...............] - ETA: 54s - loss: 0.2685 - categorical_accuracy: 0.9159
30112/60000 [==============>...............] - ETA: 54s - loss: 0.2683 - categorical_accuracy: 0.9160
30144/60000 [==============>...............] - ETA: 54s - loss: 0.2681 - categorical_accuracy: 0.9161
30176/60000 [==============>...............] - ETA: 54s - loss: 0.2682 - categorical_accuracy: 0.9161
30208/60000 [==============>...............] - ETA: 54s - loss: 0.2681 - categorical_accuracy: 0.9161
30240/60000 [==============>...............] - ETA: 54s - loss: 0.2679 - categorical_accuracy: 0.9162
30272/60000 [==============>...............] - ETA: 54s - loss: 0.2678 - categorical_accuracy: 0.9162
30304/60000 [==============>...............] - ETA: 54s - loss: 0.2675 - categorical_accuracy: 0.9163
30336/60000 [==============>...............] - ETA: 54s - loss: 0.2673 - categorical_accuracy: 0.9163
30368/60000 [==============>...............] - ETA: 53s - loss: 0.2672 - categorical_accuracy: 0.9164
30400/60000 [==============>...............] - ETA: 53s - loss: 0.2671 - categorical_accuracy: 0.9164
30432/60000 [==============>...............] - ETA: 53s - loss: 0.2669 - categorical_accuracy: 0.9165
30464/60000 [==============>...............] - ETA: 53s - loss: 0.2666 - categorical_accuracy: 0.9166
30496/60000 [==============>...............] - ETA: 53s - loss: 0.2664 - categorical_accuracy: 0.9167
30528/60000 [==============>...............] - ETA: 53s - loss: 0.2664 - categorical_accuracy: 0.9167
30560/60000 [==============>...............] - ETA: 53s - loss: 0.2662 - categorical_accuracy: 0.9167
30592/60000 [==============>...............] - ETA: 53s - loss: 0.2660 - categorical_accuracy: 0.9168
30624/60000 [==============>...............] - ETA: 53s - loss: 0.2658 - categorical_accuracy: 0.9168
30656/60000 [==============>...............] - ETA: 53s - loss: 0.2657 - categorical_accuracy: 0.9169
30688/60000 [==============>...............] - ETA: 53s - loss: 0.2655 - categorical_accuracy: 0.9169
30720/60000 [==============>...............] - ETA: 53s - loss: 0.2653 - categorical_accuracy: 0.9169
30752/60000 [==============>...............] - ETA: 53s - loss: 0.2652 - categorical_accuracy: 0.9169
30784/60000 [==============>...............] - ETA: 53s - loss: 0.2650 - categorical_accuracy: 0.9170
30816/60000 [==============>...............] - ETA: 53s - loss: 0.2650 - categorical_accuracy: 0.9170
30848/60000 [==============>...............] - ETA: 53s - loss: 0.2647 - categorical_accuracy: 0.9170
30880/60000 [==============>...............] - ETA: 53s - loss: 0.2646 - categorical_accuracy: 0.9171
30912/60000 [==============>...............] - ETA: 52s - loss: 0.2645 - categorical_accuracy: 0.9171
30944/60000 [==============>...............] - ETA: 52s - loss: 0.2642 - categorical_accuracy: 0.9172
30976/60000 [==============>...............] - ETA: 52s - loss: 0.2642 - categorical_accuracy: 0.9172
31008/60000 [==============>...............] - ETA: 52s - loss: 0.2640 - categorical_accuracy: 0.9173
31040/60000 [==============>...............] - ETA: 52s - loss: 0.2639 - categorical_accuracy: 0.9173
31072/60000 [==============>...............] - ETA: 52s - loss: 0.2636 - categorical_accuracy: 0.9174
31104/60000 [==============>...............] - ETA: 52s - loss: 0.2634 - categorical_accuracy: 0.9175
31136/60000 [==============>...............] - ETA: 52s - loss: 0.2632 - categorical_accuracy: 0.9175
31168/60000 [==============>...............] - ETA: 52s - loss: 0.2631 - categorical_accuracy: 0.9176
31200/60000 [==============>...............] - ETA: 52s - loss: 0.2629 - categorical_accuracy: 0.9176
31232/60000 [==============>...............] - ETA: 52s - loss: 0.2629 - categorical_accuracy: 0.9176
31264/60000 [==============>...............] - ETA: 52s - loss: 0.2627 - categorical_accuracy: 0.9177
31296/60000 [==============>...............] - ETA: 52s - loss: 0.2626 - categorical_accuracy: 0.9178
31328/60000 [==============>...............] - ETA: 52s - loss: 0.2625 - categorical_accuracy: 0.9178
31360/60000 [==============>...............] - ETA: 52s - loss: 0.2623 - categorical_accuracy: 0.9178
31392/60000 [==============>...............] - ETA: 52s - loss: 0.2621 - categorical_accuracy: 0.9179
31424/60000 [==============>...............] - ETA: 52s - loss: 0.2619 - categorical_accuracy: 0.9179
31456/60000 [==============>...............] - ETA: 52s - loss: 0.2616 - categorical_accuracy: 0.9180
31488/60000 [==============>...............] - ETA: 51s - loss: 0.2615 - categorical_accuracy: 0.9180
31520/60000 [==============>...............] - ETA: 51s - loss: 0.2613 - categorical_accuracy: 0.9181
31552/60000 [==============>...............] - ETA: 51s - loss: 0.2611 - categorical_accuracy: 0.9182
31584/60000 [==============>...............] - ETA: 51s - loss: 0.2612 - categorical_accuracy: 0.9182
31616/60000 [==============>...............] - ETA: 51s - loss: 0.2611 - categorical_accuracy: 0.9182
31648/60000 [==============>...............] - ETA: 51s - loss: 0.2610 - categorical_accuracy: 0.9183
31680/60000 [==============>...............] - ETA: 51s - loss: 0.2607 - categorical_accuracy: 0.9183
31712/60000 [==============>...............] - ETA: 51s - loss: 0.2605 - categorical_accuracy: 0.9184
31744/60000 [==============>...............] - ETA: 51s - loss: 0.2603 - categorical_accuracy: 0.9185
31776/60000 [==============>...............] - ETA: 51s - loss: 0.2601 - categorical_accuracy: 0.9185
31808/60000 [==============>...............] - ETA: 51s - loss: 0.2599 - categorical_accuracy: 0.9186
31840/60000 [==============>...............] - ETA: 51s - loss: 0.2597 - categorical_accuracy: 0.9187
31872/60000 [==============>...............] - ETA: 51s - loss: 0.2595 - categorical_accuracy: 0.9187
31904/60000 [==============>...............] - ETA: 51s - loss: 0.2593 - categorical_accuracy: 0.9188
31936/60000 [==============>...............] - ETA: 51s - loss: 0.2592 - categorical_accuracy: 0.9188
31968/60000 [==============>...............] - ETA: 51s - loss: 0.2593 - categorical_accuracy: 0.9188
32000/60000 [===============>..............] - ETA: 51s - loss: 0.2594 - categorical_accuracy: 0.9188
32032/60000 [===============>..............] - ETA: 50s - loss: 0.2594 - categorical_accuracy: 0.9188
32064/60000 [===============>..............] - ETA: 50s - loss: 0.2592 - categorical_accuracy: 0.9189
32096/60000 [===============>..............] - ETA: 50s - loss: 0.2590 - categorical_accuracy: 0.9189
32128/60000 [===============>..............] - ETA: 50s - loss: 0.2589 - categorical_accuracy: 0.9190
32160/60000 [===============>..............] - ETA: 50s - loss: 0.2586 - categorical_accuracy: 0.9191
32192/60000 [===============>..............] - ETA: 50s - loss: 0.2586 - categorical_accuracy: 0.9190
32224/60000 [===============>..............] - ETA: 50s - loss: 0.2584 - categorical_accuracy: 0.9191
32256/60000 [===============>..............] - ETA: 50s - loss: 0.2582 - categorical_accuracy: 0.9191
32288/60000 [===============>..............] - ETA: 50s - loss: 0.2582 - categorical_accuracy: 0.9192
32320/60000 [===============>..............] - ETA: 50s - loss: 0.2581 - categorical_accuracy: 0.9192
32352/60000 [===============>..............] - ETA: 50s - loss: 0.2579 - categorical_accuracy: 0.9193
32384/60000 [===============>..............] - ETA: 50s - loss: 0.2576 - categorical_accuracy: 0.9193
32416/60000 [===============>..............] - ETA: 50s - loss: 0.2578 - categorical_accuracy: 0.9193
32448/60000 [===============>..............] - ETA: 50s - loss: 0.2577 - categorical_accuracy: 0.9193
32480/60000 [===============>..............] - ETA: 50s - loss: 0.2578 - categorical_accuracy: 0.9193
32512/60000 [===============>..............] - ETA: 50s - loss: 0.2578 - categorical_accuracy: 0.9194
32544/60000 [===============>..............] - ETA: 50s - loss: 0.2576 - categorical_accuracy: 0.9194
32576/60000 [===============>..............] - ETA: 49s - loss: 0.2574 - categorical_accuracy: 0.9195
32608/60000 [===============>..............] - ETA: 49s - loss: 0.2574 - categorical_accuracy: 0.9195
32640/60000 [===============>..............] - ETA: 49s - loss: 0.2573 - categorical_accuracy: 0.9195
32672/60000 [===============>..............] - ETA: 49s - loss: 0.2571 - categorical_accuracy: 0.9196
32704/60000 [===============>..............] - ETA: 49s - loss: 0.2570 - categorical_accuracy: 0.9196
32736/60000 [===============>..............] - ETA: 49s - loss: 0.2569 - categorical_accuracy: 0.9196
32768/60000 [===============>..............] - ETA: 49s - loss: 0.2567 - categorical_accuracy: 0.9197
32800/60000 [===============>..............] - ETA: 49s - loss: 0.2565 - categorical_accuracy: 0.9198
32832/60000 [===============>..............] - ETA: 49s - loss: 0.2563 - categorical_accuracy: 0.9198
32864/60000 [===============>..............] - ETA: 49s - loss: 0.2561 - categorical_accuracy: 0.9199
32896/60000 [===============>..............] - ETA: 49s - loss: 0.2560 - categorical_accuracy: 0.9199
32928/60000 [===============>..............] - ETA: 49s - loss: 0.2558 - categorical_accuracy: 0.9200
32960/60000 [===============>..............] - ETA: 49s - loss: 0.2556 - categorical_accuracy: 0.9201
32992/60000 [===============>..............] - ETA: 49s - loss: 0.2555 - categorical_accuracy: 0.9201
33024/60000 [===============>..............] - ETA: 49s - loss: 0.2553 - categorical_accuracy: 0.9202
33056/60000 [===============>..............] - ETA: 49s - loss: 0.2551 - categorical_accuracy: 0.9203
33088/60000 [===============>..............] - ETA: 49s - loss: 0.2549 - categorical_accuracy: 0.9203
33120/60000 [===============>..............] - ETA: 48s - loss: 0.2548 - categorical_accuracy: 0.9203
33152/60000 [===============>..............] - ETA: 48s - loss: 0.2546 - categorical_accuracy: 0.9204
33184/60000 [===============>..............] - ETA: 48s - loss: 0.2546 - categorical_accuracy: 0.9204
33216/60000 [===============>..............] - ETA: 48s - loss: 0.2544 - categorical_accuracy: 0.9205
33248/60000 [===============>..............] - ETA: 48s - loss: 0.2542 - categorical_accuracy: 0.9205
33280/60000 [===============>..............] - ETA: 48s - loss: 0.2540 - categorical_accuracy: 0.9206
33312/60000 [===============>..............] - ETA: 48s - loss: 0.2538 - categorical_accuracy: 0.9207
33344/60000 [===============>..............] - ETA: 48s - loss: 0.2536 - categorical_accuracy: 0.9208
33376/60000 [===============>..............] - ETA: 48s - loss: 0.2534 - categorical_accuracy: 0.9208
33408/60000 [===============>..............] - ETA: 48s - loss: 0.2533 - categorical_accuracy: 0.9209
33440/60000 [===============>..............] - ETA: 48s - loss: 0.2531 - categorical_accuracy: 0.9209
33472/60000 [===============>..............] - ETA: 48s - loss: 0.2529 - categorical_accuracy: 0.9210
33504/60000 [===============>..............] - ETA: 48s - loss: 0.2527 - categorical_accuracy: 0.9211
33536/60000 [===============>..............] - ETA: 48s - loss: 0.2525 - categorical_accuracy: 0.9211
33568/60000 [===============>..............] - ETA: 48s - loss: 0.2523 - categorical_accuracy: 0.9212
33600/60000 [===============>..............] - ETA: 48s - loss: 0.2524 - categorical_accuracy: 0.9212
33632/60000 [===============>..............] - ETA: 48s - loss: 0.2522 - categorical_accuracy: 0.9213
33664/60000 [===============>..............] - ETA: 47s - loss: 0.2519 - categorical_accuracy: 0.9214
33696/60000 [===============>..............] - ETA: 47s - loss: 0.2518 - categorical_accuracy: 0.9214
33728/60000 [===============>..............] - ETA: 47s - loss: 0.2518 - categorical_accuracy: 0.9214
33760/60000 [===============>..............] - ETA: 47s - loss: 0.2519 - categorical_accuracy: 0.9214
33792/60000 [===============>..............] - ETA: 47s - loss: 0.2517 - categorical_accuracy: 0.9215
33824/60000 [===============>..............] - ETA: 47s - loss: 0.2518 - categorical_accuracy: 0.9215
33856/60000 [===============>..............] - ETA: 47s - loss: 0.2516 - categorical_accuracy: 0.9216
33888/60000 [===============>..............] - ETA: 47s - loss: 0.2515 - categorical_accuracy: 0.9216
33920/60000 [===============>..............] - ETA: 47s - loss: 0.2514 - categorical_accuracy: 0.9216
33952/60000 [===============>..............] - ETA: 47s - loss: 0.2518 - categorical_accuracy: 0.9215
33984/60000 [===============>..............] - ETA: 47s - loss: 0.2517 - categorical_accuracy: 0.9216
34016/60000 [================>.............] - ETA: 47s - loss: 0.2516 - categorical_accuracy: 0.9216
34048/60000 [================>.............] - ETA: 47s - loss: 0.2514 - categorical_accuracy: 0.9216
34080/60000 [================>.............] - ETA: 47s - loss: 0.2513 - categorical_accuracy: 0.9217
34112/60000 [================>.............] - ETA: 47s - loss: 0.2511 - categorical_accuracy: 0.9217
34144/60000 [================>.............] - ETA: 47s - loss: 0.2511 - categorical_accuracy: 0.9217
34176/60000 [================>.............] - ETA: 47s - loss: 0.2509 - categorical_accuracy: 0.9218
34208/60000 [================>.............] - ETA: 46s - loss: 0.2507 - categorical_accuracy: 0.9218
34240/60000 [================>.............] - ETA: 46s - loss: 0.2505 - categorical_accuracy: 0.9219
34272/60000 [================>.............] - ETA: 46s - loss: 0.2504 - categorical_accuracy: 0.9219
34304/60000 [================>.............] - ETA: 46s - loss: 0.2503 - categorical_accuracy: 0.9220
34336/60000 [================>.............] - ETA: 46s - loss: 0.2501 - categorical_accuracy: 0.9220
34368/60000 [================>.............] - ETA: 46s - loss: 0.2500 - categorical_accuracy: 0.9220
34400/60000 [================>.............] - ETA: 46s - loss: 0.2499 - categorical_accuracy: 0.9220
34432/60000 [================>.............] - ETA: 46s - loss: 0.2498 - categorical_accuracy: 0.9221
34464/60000 [================>.............] - ETA: 46s - loss: 0.2496 - categorical_accuracy: 0.9222
34496/60000 [================>.............] - ETA: 46s - loss: 0.2494 - categorical_accuracy: 0.9222
34528/60000 [================>.............] - ETA: 46s - loss: 0.2492 - categorical_accuracy: 0.9223
34560/60000 [================>.............] - ETA: 46s - loss: 0.2492 - categorical_accuracy: 0.9223
34592/60000 [================>.............] - ETA: 46s - loss: 0.2491 - categorical_accuracy: 0.9223
34624/60000 [================>.............] - ETA: 46s - loss: 0.2490 - categorical_accuracy: 0.9224
34656/60000 [================>.............] - ETA: 46s - loss: 0.2488 - categorical_accuracy: 0.9224
34688/60000 [================>.............] - ETA: 46s - loss: 0.2487 - categorical_accuracy: 0.9225
34720/60000 [================>.............] - ETA: 46s - loss: 0.2485 - categorical_accuracy: 0.9226
34752/60000 [================>.............] - ETA: 45s - loss: 0.2490 - categorical_accuracy: 0.9225
34784/60000 [================>.............] - ETA: 45s - loss: 0.2488 - categorical_accuracy: 0.9226
34816/60000 [================>.............] - ETA: 45s - loss: 0.2486 - categorical_accuracy: 0.9226
34848/60000 [================>.............] - ETA: 45s - loss: 0.2485 - categorical_accuracy: 0.9226
34880/60000 [================>.............] - ETA: 45s - loss: 0.2483 - categorical_accuracy: 0.9226
34912/60000 [================>.............] - ETA: 45s - loss: 0.2482 - categorical_accuracy: 0.9227
34944/60000 [================>.............] - ETA: 45s - loss: 0.2481 - categorical_accuracy: 0.9227
34976/60000 [================>.............] - ETA: 45s - loss: 0.2480 - categorical_accuracy: 0.9227
35008/60000 [================>.............] - ETA: 45s - loss: 0.2478 - categorical_accuracy: 0.9228
35040/60000 [================>.............] - ETA: 45s - loss: 0.2480 - categorical_accuracy: 0.9227
35072/60000 [================>.............] - ETA: 45s - loss: 0.2479 - categorical_accuracy: 0.9228
35104/60000 [================>.............] - ETA: 45s - loss: 0.2480 - categorical_accuracy: 0.9228
35136/60000 [================>.............] - ETA: 45s - loss: 0.2478 - categorical_accuracy: 0.9228
35168/60000 [================>.............] - ETA: 45s - loss: 0.2477 - categorical_accuracy: 0.9229
35200/60000 [================>.............] - ETA: 45s - loss: 0.2476 - categorical_accuracy: 0.9229
35232/60000 [================>.............] - ETA: 45s - loss: 0.2474 - categorical_accuracy: 0.9230
35264/60000 [================>.............] - ETA: 45s - loss: 0.2472 - categorical_accuracy: 0.9230
35296/60000 [================>.............] - ETA: 44s - loss: 0.2471 - categorical_accuracy: 0.9230
35328/60000 [================>.............] - ETA: 44s - loss: 0.2471 - categorical_accuracy: 0.9231
35360/60000 [================>.............] - ETA: 44s - loss: 0.2468 - categorical_accuracy: 0.9231
35392/60000 [================>.............] - ETA: 44s - loss: 0.2467 - categorical_accuracy: 0.9232
35424/60000 [================>.............] - ETA: 44s - loss: 0.2468 - categorical_accuracy: 0.9232
35456/60000 [================>.............] - ETA: 44s - loss: 0.2468 - categorical_accuracy: 0.9233
35488/60000 [================>.............] - ETA: 44s - loss: 0.2466 - categorical_accuracy: 0.9233
35520/60000 [================>.............] - ETA: 44s - loss: 0.2465 - categorical_accuracy: 0.9234
35552/60000 [================>.............] - ETA: 44s - loss: 0.2463 - categorical_accuracy: 0.9234
35584/60000 [================>.............] - ETA: 44s - loss: 0.2462 - categorical_accuracy: 0.9235
35616/60000 [================>.............] - ETA: 44s - loss: 0.2460 - categorical_accuracy: 0.9235
35648/60000 [================>.............] - ETA: 44s - loss: 0.2458 - categorical_accuracy: 0.9236
35680/60000 [================>.............] - ETA: 44s - loss: 0.2456 - categorical_accuracy: 0.9237
35712/60000 [================>.............] - ETA: 44s - loss: 0.2455 - categorical_accuracy: 0.9237
35744/60000 [================>.............] - ETA: 44s - loss: 0.2453 - categorical_accuracy: 0.9238
35776/60000 [================>.............] - ETA: 44s - loss: 0.2451 - categorical_accuracy: 0.9238
35808/60000 [================>.............] - ETA: 44s - loss: 0.2449 - categorical_accuracy: 0.9239
35840/60000 [================>.............] - ETA: 43s - loss: 0.2447 - categorical_accuracy: 0.9240
35872/60000 [================>.............] - ETA: 43s - loss: 0.2447 - categorical_accuracy: 0.9240
35904/60000 [================>.............] - ETA: 43s - loss: 0.2446 - categorical_accuracy: 0.9240
35936/60000 [================>.............] - ETA: 43s - loss: 0.2444 - categorical_accuracy: 0.9241
35968/60000 [================>.............] - ETA: 43s - loss: 0.2444 - categorical_accuracy: 0.9240
36000/60000 [=================>............] - ETA: 43s - loss: 0.2444 - categorical_accuracy: 0.9241
36032/60000 [=================>............] - ETA: 43s - loss: 0.2443 - categorical_accuracy: 0.9241
36064/60000 [=================>............] - ETA: 43s - loss: 0.2441 - categorical_accuracy: 0.9242
36096/60000 [=================>............] - ETA: 43s - loss: 0.2440 - categorical_accuracy: 0.9242
36128/60000 [=================>............] - ETA: 43s - loss: 0.2438 - categorical_accuracy: 0.9242
36160/60000 [=================>............] - ETA: 43s - loss: 0.2437 - categorical_accuracy: 0.9243
36192/60000 [=================>............] - ETA: 43s - loss: 0.2435 - categorical_accuracy: 0.9243
36224/60000 [=================>............] - ETA: 43s - loss: 0.2433 - categorical_accuracy: 0.9244
36256/60000 [=================>............] - ETA: 43s - loss: 0.2432 - categorical_accuracy: 0.9244
36288/60000 [=================>............] - ETA: 43s - loss: 0.2430 - categorical_accuracy: 0.9245
36320/60000 [=================>............] - ETA: 43s - loss: 0.2430 - categorical_accuracy: 0.9245
36352/60000 [=================>............] - ETA: 43s - loss: 0.2432 - categorical_accuracy: 0.9245
36384/60000 [=================>............] - ETA: 42s - loss: 0.2430 - categorical_accuracy: 0.9246
36416/60000 [=================>............] - ETA: 42s - loss: 0.2429 - categorical_accuracy: 0.9246
36448/60000 [=================>............] - ETA: 42s - loss: 0.2427 - categorical_accuracy: 0.9246
36480/60000 [=================>............] - ETA: 42s - loss: 0.2425 - categorical_accuracy: 0.9247
36512/60000 [=================>............] - ETA: 42s - loss: 0.2425 - categorical_accuracy: 0.9247
36544/60000 [=================>............] - ETA: 42s - loss: 0.2425 - categorical_accuracy: 0.9247
36576/60000 [=================>............] - ETA: 42s - loss: 0.2425 - categorical_accuracy: 0.9247
36608/60000 [=================>............] - ETA: 42s - loss: 0.2423 - categorical_accuracy: 0.9247
36640/60000 [=================>............] - ETA: 42s - loss: 0.2422 - categorical_accuracy: 0.9248
36672/60000 [=================>............] - ETA: 42s - loss: 0.2421 - categorical_accuracy: 0.9248
36704/60000 [=================>............] - ETA: 42s - loss: 0.2423 - categorical_accuracy: 0.9248
36736/60000 [=================>............] - ETA: 42s - loss: 0.2421 - categorical_accuracy: 0.9248
36768/60000 [=================>............] - ETA: 42s - loss: 0.2420 - categorical_accuracy: 0.9248
36800/60000 [=================>............] - ETA: 42s - loss: 0.2418 - categorical_accuracy: 0.9248
36832/60000 [=================>............] - ETA: 42s - loss: 0.2417 - categorical_accuracy: 0.9249
36864/60000 [=================>............] - ETA: 42s - loss: 0.2415 - categorical_accuracy: 0.9249
36896/60000 [=================>............] - ETA: 42s - loss: 0.2415 - categorical_accuracy: 0.9250
36928/60000 [=================>............] - ETA: 41s - loss: 0.2413 - categorical_accuracy: 0.9250
36960/60000 [=================>............] - ETA: 41s - loss: 0.2412 - categorical_accuracy: 0.9251
36992/60000 [=================>............] - ETA: 41s - loss: 0.2410 - categorical_accuracy: 0.9251
37024/60000 [=================>............] - ETA: 41s - loss: 0.2410 - categorical_accuracy: 0.9251
37056/60000 [=================>............] - ETA: 41s - loss: 0.2409 - categorical_accuracy: 0.9251
37088/60000 [=================>............] - ETA: 41s - loss: 0.2407 - categorical_accuracy: 0.9252
37120/60000 [=================>............] - ETA: 41s - loss: 0.2405 - categorical_accuracy: 0.9253
37152/60000 [=================>............] - ETA: 41s - loss: 0.2404 - categorical_accuracy: 0.9253
37184/60000 [=================>............] - ETA: 41s - loss: 0.2402 - categorical_accuracy: 0.9254
37216/60000 [=================>............] - ETA: 41s - loss: 0.2400 - categorical_accuracy: 0.9254
37248/60000 [=================>............] - ETA: 41s - loss: 0.2398 - categorical_accuracy: 0.9255
37280/60000 [=================>............] - ETA: 41s - loss: 0.2397 - categorical_accuracy: 0.9255
37312/60000 [=================>............] - ETA: 41s - loss: 0.2396 - categorical_accuracy: 0.9255
37344/60000 [=================>............] - ETA: 41s - loss: 0.2395 - categorical_accuracy: 0.9255
37376/60000 [=================>............] - ETA: 41s - loss: 0.2394 - categorical_accuracy: 0.9255
37408/60000 [=================>............] - ETA: 41s - loss: 0.2392 - categorical_accuracy: 0.9256
37440/60000 [=================>............] - ETA: 41s - loss: 0.2391 - categorical_accuracy: 0.9256
37472/60000 [=================>............] - ETA: 40s - loss: 0.2390 - categorical_accuracy: 0.9257
37504/60000 [=================>............] - ETA: 40s - loss: 0.2389 - categorical_accuracy: 0.9257
37536/60000 [=================>............] - ETA: 40s - loss: 0.2387 - categorical_accuracy: 0.9258
37568/60000 [=================>............] - ETA: 40s - loss: 0.2386 - categorical_accuracy: 0.9258
37600/60000 [=================>............] - ETA: 40s - loss: 0.2385 - categorical_accuracy: 0.9258
37632/60000 [=================>............] - ETA: 40s - loss: 0.2383 - categorical_accuracy: 0.9259
37664/60000 [=================>............] - ETA: 40s - loss: 0.2383 - categorical_accuracy: 0.9259
37696/60000 [=================>............] - ETA: 40s - loss: 0.2381 - categorical_accuracy: 0.9260
37728/60000 [=================>............] - ETA: 40s - loss: 0.2380 - categorical_accuracy: 0.9260
37760/60000 [=================>............] - ETA: 40s - loss: 0.2378 - categorical_accuracy: 0.9260
37792/60000 [=================>............] - ETA: 40s - loss: 0.2376 - categorical_accuracy: 0.9261
37824/60000 [=================>............] - ETA: 40s - loss: 0.2375 - categorical_accuracy: 0.9261
37856/60000 [=================>............] - ETA: 40s - loss: 0.2374 - categorical_accuracy: 0.9262
37888/60000 [=================>............] - ETA: 40s - loss: 0.2372 - categorical_accuracy: 0.9262
37920/60000 [=================>............] - ETA: 40s - loss: 0.2370 - categorical_accuracy: 0.9263
37952/60000 [=================>............] - ETA: 40s - loss: 0.2370 - categorical_accuracy: 0.9263
37984/60000 [=================>............] - ETA: 40s - loss: 0.2368 - categorical_accuracy: 0.9263
38016/60000 [==================>...........] - ETA: 39s - loss: 0.2367 - categorical_accuracy: 0.9263
38048/60000 [==================>...........] - ETA: 39s - loss: 0.2365 - categorical_accuracy: 0.9264
38080/60000 [==================>...........] - ETA: 39s - loss: 0.2364 - categorical_accuracy: 0.9265
38112/60000 [==================>...........] - ETA: 39s - loss: 0.2366 - categorical_accuracy: 0.9265
38144/60000 [==================>...........] - ETA: 39s - loss: 0.2366 - categorical_accuracy: 0.9265
38176/60000 [==================>...........] - ETA: 39s - loss: 0.2364 - categorical_accuracy: 0.9266
38208/60000 [==================>...........] - ETA: 39s - loss: 0.2363 - categorical_accuracy: 0.9266
38240/60000 [==================>...........] - ETA: 39s - loss: 0.2362 - categorical_accuracy: 0.9266
38272/60000 [==================>...........] - ETA: 39s - loss: 0.2360 - categorical_accuracy: 0.9267
38304/60000 [==================>...........] - ETA: 39s - loss: 0.2358 - categorical_accuracy: 0.9267
38336/60000 [==================>...........] - ETA: 39s - loss: 0.2357 - categorical_accuracy: 0.9268
38368/60000 [==================>...........] - ETA: 39s - loss: 0.2359 - categorical_accuracy: 0.9267
38400/60000 [==================>...........] - ETA: 39s - loss: 0.2357 - categorical_accuracy: 0.9267
38432/60000 [==================>...........] - ETA: 39s - loss: 0.2356 - categorical_accuracy: 0.9268
38464/60000 [==================>...........] - ETA: 39s - loss: 0.2355 - categorical_accuracy: 0.9268
38496/60000 [==================>...........] - ETA: 39s - loss: 0.2353 - categorical_accuracy: 0.9269
38528/60000 [==================>...........] - ETA: 39s - loss: 0.2353 - categorical_accuracy: 0.9269
38560/60000 [==================>...........] - ETA: 38s - loss: 0.2351 - categorical_accuracy: 0.9269
38592/60000 [==================>...........] - ETA: 38s - loss: 0.2350 - categorical_accuracy: 0.9269
38624/60000 [==================>...........] - ETA: 38s - loss: 0.2348 - categorical_accuracy: 0.9270
38656/60000 [==================>...........] - ETA: 38s - loss: 0.2351 - categorical_accuracy: 0.9270
38688/60000 [==================>...........] - ETA: 38s - loss: 0.2350 - categorical_accuracy: 0.9270
38720/60000 [==================>...........] - ETA: 38s - loss: 0.2350 - categorical_accuracy: 0.9270
38752/60000 [==================>...........] - ETA: 38s - loss: 0.2348 - categorical_accuracy: 0.9270
38784/60000 [==================>...........] - ETA: 38s - loss: 0.2348 - categorical_accuracy: 0.9271
38816/60000 [==================>...........] - ETA: 38s - loss: 0.2346 - categorical_accuracy: 0.9271
38848/60000 [==================>...........] - ETA: 38s - loss: 0.2345 - categorical_accuracy: 0.9272
38880/60000 [==================>...........] - ETA: 38s - loss: 0.2347 - categorical_accuracy: 0.9272
38912/60000 [==================>...........] - ETA: 38s - loss: 0.2345 - categorical_accuracy: 0.9272
38944/60000 [==================>...........] - ETA: 38s - loss: 0.2346 - categorical_accuracy: 0.9272
38976/60000 [==================>...........] - ETA: 38s - loss: 0.2346 - categorical_accuracy: 0.9272
39008/60000 [==================>...........] - ETA: 38s - loss: 0.2348 - categorical_accuracy: 0.9272
39040/60000 [==================>...........] - ETA: 38s - loss: 0.2347 - categorical_accuracy: 0.9272
39072/60000 [==================>...........] - ETA: 38s - loss: 0.2345 - categorical_accuracy: 0.9273
39104/60000 [==================>...........] - ETA: 37s - loss: 0.2344 - categorical_accuracy: 0.9273
39136/60000 [==================>...........] - ETA: 37s - loss: 0.2343 - categorical_accuracy: 0.9274
39168/60000 [==================>...........] - ETA: 37s - loss: 0.2341 - categorical_accuracy: 0.9274
39200/60000 [==================>...........] - ETA: 37s - loss: 0.2342 - categorical_accuracy: 0.9274
39232/60000 [==================>...........] - ETA: 37s - loss: 0.2341 - categorical_accuracy: 0.9275
39264/60000 [==================>...........] - ETA: 37s - loss: 0.2341 - categorical_accuracy: 0.9275
39296/60000 [==================>...........] - ETA: 37s - loss: 0.2339 - categorical_accuracy: 0.9275
39328/60000 [==================>...........] - ETA: 37s - loss: 0.2339 - categorical_accuracy: 0.9275
39360/60000 [==================>...........] - ETA: 37s - loss: 0.2338 - categorical_accuracy: 0.9276
39392/60000 [==================>...........] - ETA: 37s - loss: 0.2337 - categorical_accuracy: 0.9276
39424/60000 [==================>...........] - ETA: 37s - loss: 0.2338 - categorical_accuracy: 0.9276
39456/60000 [==================>...........] - ETA: 37s - loss: 0.2337 - categorical_accuracy: 0.9276
39488/60000 [==================>...........] - ETA: 37s - loss: 0.2335 - categorical_accuracy: 0.9277
39520/60000 [==================>...........] - ETA: 37s - loss: 0.2334 - categorical_accuracy: 0.9277
39552/60000 [==================>...........] - ETA: 37s - loss: 0.2334 - categorical_accuracy: 0.9277
39584/60000 [==================>...........] - ETA: 37s - loss: 0.2333 - categorical_accuracy: 0.9277
39616/60000 [==================>...........] - ETA: 37s - loss: 0.2331 - categorical_accuracy: 0.9278
39648/60000 [==================>...........] - ETA: 37s - loss: 0.2331 - categorical_accuracy: 0.9278
39680/60000 [==================>...........] - ETA: 36s - loss: 0.2330 - categorical_accuracy: 0.9278
39712/60000 [==================>...........] - ETA: 36s - loss: 0.2329 - categorical_accuracy: 0.9279
39744/60000 [==================>...........] - ETA: 36s - loss: 0.2329 - categorical_accuracy: 0.9279
39776/60000 [==================>...........] - ETA: 36s - loss: 0.2328 - categorical_accuracy: 0.9279
39808/60000 [==================>...........] - ETA: 36s - loss: 0.2328 - categorical_accuracy: 0.9279
39840/60000 [==================>...........] - ETA: 36s - loss: 0.2326 - categorical_accuracy: 0.9279
39872/60000 [==================>...........] - ETA: 36s - loss: 0.2325 - categorical_accuracy: 0.9280
39904/60000 [==================>...........] - ETA: 36s - loss: 0.2323 - categorical_accuracy: 0.9280
39936/60000 [==================>...........] - ETA: 36s - loss: 0.2322 - categorical_accuracy: 0.9281
39968/60000 [==================>...........] - ETA: 36s - loss: 0.2321 - categorical_accuracy: 0.9281
40000/60000 [===================>..........] - ETA: 36s - loss: 0.2321 - categorical_accuracy: 0.9281
40032/60000 [===================>..........] - ETA: 36s - loss: 0.2320 - categorical_accuracy: 0.9282
40064/60000 [===================>..........] - ETA: 36s - loss: 0.2320 - categorical_accuracy: 0.9281
40096/60000 [===================>..........] - ETA: 36s - loss: 0.2319 - categorical_accuracy: 0.9282
40128/60000 [===================>..........] - ETA: 36s - loss: 0.2318 - categorical_accuracy: 0.9282
40160/60000 [===================>..........] - ETA: 36s - loss: 0.2317 - categorical_accuracy: 0.9282
40192/60000 [===================>..........] - ETA: 36s - loss: 0.2316 - categorical_accuracy: 0.9282
40224/60000 [===================>..........] - ETA: 35s - loss: 0.2315 - categorical_accuracy: 0.9283
40256/60000 [===================>..........] - ETA: 35s - loss: 0.2313 - categorical_accuracy: 0.9283
40288/60000 [===================>..........] - ETA: 35s - loss: 0.2311 - categorical_accuracy: 0.9284
40320/60000 [===================>..........] - ETA: 35s - loss: 0.2310 - categorical_accuracy: 0.9284
40352/60000 [===================>..........] - ETA: 35s - loss: 0.2309 - categorical_accuracy: 0.9285
40384/60000 [===================>..........] - ETA: 35s - loss: 0.2309 - categorical_accuracy: 0.9285
40416/60000 [===================>..........] - ETA: 35s - loss: 0.2308 - categorical_accuracy: 0.9285
40448/60000 [===================>..........] - ETA: 35s - loss: 0.2308 - categorical_accuracy: 0.9285
40480/60000 [===================>..........] - ETA: 35s - loss: 0.2309 - categorical_accuracy: 0.9285
40512/60000 [===================>..........] - ETA: 35s - loss: 0.2308 - categorical_accuracy: 0.9285
40544/60000 [===================>..........] - ETA: 35s - loss: 0.2307 - categorical_accuracy: 0.9286
40576/60000 [===================>..........] - ETA: 35s - loss: 0.2306 - categorical_accuracy: 0.9286
40608/60000 [===================>..........] - ETA: 35s - loss: 0.2305 - categorical_accuracy: 0.9286
40640/60000 [===================>..........] - ETA: 35s - loss: 0.2303 - categorical_accuracy: 0.9287
40672/60000 [===================>..........] - ETA: 35s - loss: 0.2302 - categorical_accuracy: 0.9287
40704/60000 [===================>..........] - ETA: 35s - loss: 0.2301 - categorical_accuracy: 0.9287
40736/60000 [===================>..........] - ETA: 35s - loss: 0.2299 - categorical_accuracy: 0.9288
40768/60000 [===================>..........] - ETA: 34s - loss: 0.2299 - categorical_accuracy: 0.9288
40800/60000 [===================>..........] - ETA: 34s - loss: 0.2297 - categorical_accuracy: 0.9288
40832/60000 [===================>..........] - ETA: 34s - loss: 0.2295 - categorical_accuracy: 0.9289
40864/60000 [===================>..........] - ETA: 34s - loss: 0.2294 - categorical_accuracy: 0.9289
40896/60000 [===================>..........] - ETA: 34s - loss: 0.2293 - categorical_accuracy: 0.9289
40928/60000 [===================>..........] - ETA: 34s - loss: 0.2292 - categorical_accuracy: 0.9290
40960/60000 [===================>..........] - ETA: 34s - loss: 0.2291 - categorical_accuracy: 0.9290
40992/60000 [===================>..........] - ETA: 34s - loss: 0.2290 - categorical_accuracy: 0.9290
41024/60000 [===================>..........] - ETA: 34s - loss: 0.2289 - categorical_accuracy: 0.9290
41056/60000 [===================>..........] - ETA: 34s - loss: 0.2288 - categorical_accuracy: 0.9291
41088/60000 [===================>..........] - ETA: 34s - loss: 0.2287 - categorical_accuracy: 0.9291
41120/60000 [===================>..........] - ETA: 34s - loss: 0.2285 - categorical_accuracy: 0.9292
41152/60000 [===================>..........] - ETA: 34s - loss: 0.2284 - categorical_accuracy: 0.9292
41184/60000 [===================>..........] - ETA: 34s - loss: 0.2282 - categorical_accuracy: 0.9292
41216/60000 [===================>..........] - ETA: 34s - loss: 0.2282 - categorical_accuracy: 0.9293
41248/60000 [===================>..........] - ETA: 34s - loss: 0.2280 - categorical_accuracy: 0.9293
41280/60000 [===================>..........] - ETA: 34s - loss: 0.2279 - categorical_accuracy: 0.9294
41312/60000 [===================>..........] - ETA: 33s - loss: 0.2277 - categorical_accuracy: 0.9294
41344/60000 [===================>..........] - ETA: 33s - loss: 0.2275 - categorical_accuracy: 0.9295
41376/60000 [===================>..........] - ETA: 33s - loss: 0.2274 - categorical_accuracy: 0.9295
41408/60000 [===================>..........] - ETA: 33s - loss: 0.2272 - categorical_accuracy: 0.9296
41440/60000 [===================>..........] - ETA: 33s - loss: 0.2272 - categorical_accuracy: 0.9296
41472/60000 [===================>..........] - ETA: 33s - loss: 0.2270 - categorical_accuracy: 0.9297
41504/60000 [===================>..........] - ETA: 33s - loss: 0.2270 - categorical_accuracy: 0.9297
41536/60000 [===================>..........] - ETA: 33s - loss: 0.2271 - categorical_accuracy: 0.9296
41568/60000 [===================>..........] - ETA: 33s - loss: 0.2269 - categorical_accuracy: 0.9296
41600/60000 [===================>..........] - ETA: 33s - loss: 0.2268 - categorical_accuracy: 0.9297
41632/60000 [===================>..........] - ETA: 33s - loss: 0.2267 - categorical_accuracy: 0.9297
41664/60000 [===================>..........] - ETA: 33s - loss: 0.2265 - categorical_accuracy: 0.9297
41696/60000 [===================>..........] - ETA: 33s - loss: 0.2264 - categorical_accuracy: 0.9298
41728/60000 [===================>..........] - ETA: 33s - loss: 0.2263 - categorical_accuracy: 0.9298
41760/60000 [===================>..........] - ETA: 33s - loss: 0.2263 - categorical_accuracy: 0.9298
41792/60000 [===================>..........] - ETA: 33s - loss: 0.2261 - categorical_accuracy: 0.9299
41824/60000 [===================>..........] - ETA: 33s - loss: 0.2261 - categorical_accuracy: 0.9299
41856/60000 [===================>..........] - ETA: 32s - loss: 0.2261 - categorical_accuracy: 0.9299
41888/60000 [===================>..........] - ETA: 32s - loss: 0.2260 - categorical_accuracy: 0.9299
41920/60000 [===================>..........] - ETA: 32s - loss: 0.2259 - categorical_accuracy: 0.9299
41952/60000 [===================>..........] - ETA: 32s - loss: 0.2257 - categorical_accuracy: 0.9300
41984/60000 [===================>..........] - ETA: 32s - loss: 0.2257 - categorical_accuracy: 0.9300
42016/60000 [====================>.........] - ETA: 32s - loss: 0.2258 - categorical_accuracy: 0.9300
42048/60000 [====================>.........] - ETA: 32s - loss: 0.2259 - categorical_accuracy: 0.9300
42080/60000 [====================>.........] - ETA: 32s - loss: 0.2258 - categorical_accuracy: 0.9300
42112/60000 [====================>.........] - ETA: 32s - loss: 0.2257 - categorical_accuracy: 0.9300
42144/60000 [====================>.........] - ETA: 32s - loss: 0.2256 - categorical_accuracy: 0.9300
42176/60000 [====================>.........] - ETA: 32s - loss: 0.2255 - categorical_accuracy: 0.9301
42208/60000 [====================>.........] - ETA: 32s - loss: 0.2255 - categorical_accuracy: 0.9301
42240/60000 [====================>.........] - ETA: 32s - loss: 0.2254 - categorical_accuracy: 0.9301
42272/60000 [====================>.........] - ETA: 32s - loss: 0.2253 - categorical_accuracy: 0.9302
42304/60000 [====================>.........] - ETA: 32s - loss: 0.2251 - categorical_accuracy: 0.9302
42336/60000 [====================>.........] - ETA: 32s - loss: 0.2250 - categorical_accuracy: 0.9302
42368/60000 [====================>.........] - ETA: 32s - loss: 0.2249 - categorical_accuracy: 0.9303
42400/60000 [====================>.........] - ETA: 31s - loss: 0.2247 - categorical_accuracy: 0.9303
42432/60000 [====================>.........] - ETA: 31s - loss: 0.2246 - categorical_accuracy: 0.9304
42464/60000 [====================>.........] - ETA: 31s - loss: 0.2244 - categorical_accuracy: 0.9304
42496/60000 [====================>.........] - ETA: 31s - loss: 0.2243 - categorical_accuracy: 0.9305
42528/60000 [====================>.........] - ETA: 31s - loss: 0.2241 - categorical_accuracy: 0.9305
42560/60000 [====================>.........] - ETA: 31s - loss: 0.2241 - categorical_accuracy: 0.9305
42592/60000 [====================>.........] - ETA: 31s - loss: 0.2240 - categorical_accuracy: 0.9306
42624/60000 [====================>.........] - ETA: 31s - loss: 0.2240 - categorical_accuracy: 0.9306
42656/60000 [====================>.........] - ETA: 31s - loss: 0.2239 - categorical_accuracy: 0.9306
42688/60000 [====================>.........] - ETA: 31s - loss: 0.2238 - categorical_accuracy: 0.9306
42720/60000 [====================>.........] - ETA: 31s - loss: 0.2237 - categorical_accuracy: 0.9306
42752/60000 [====================>.........] - ETA: 31s - loss: 0.2235 - categorical_accuracy: 0.9307
42784/60000 [====================>.........] - ETA: 31s - loss: 0.2235 - categorical_accuracy: 0.9307
42816/60000 [====================>.........] - ETA: 31s - loss: 0.2234 - categorical_accuracy: 0.9307
42848/60000 [====================>.........] - ETA: 31s - loss: 0.2232 - categorical_accuracy: 0.9308
42880/60000 [====================>.........] - ETA: 31s - loss: 0.2231 - categorical_accuracy: 0.9308
42912/60000 [====================>.........] - ETA: 31s - loss: 0.2230 - categorical_accuracy: 0.9308
42944/60000 [====================>.........] - ETA: 30s - loss: 0.2228 - categorical_accuracy: 0.9309
42976/60000 [====================>.........] - ETA: 30s - loss: 0.2227 - categorical_accuracy: 0.9309
43008/60000 [====================>.........] - ETA: 30s - loss: 0.2226 - categorical_accuracy: 0.9310
43040/60000 [====================>.........] - ETA: 30s - loss: 0.2224 - categorical_accuracy: 0.9310
43072/60000 [====================>.........] - ETA: 30s - loss: 0.2225 - categorical_accuracy: 0.9310
43104/60000 [====================>.........] - ETA: 30s - loss: 0.2224 - categorical_accuracy: 0.9310
43136/60000 [====================>.........] - ETA: 30s - loss: 0.2224 - categorical_accuracy: 0.9310
43168/60000 [====================>.........] - ETA: 30s - loss: 0.2224 - categorical_accuracy: 0.9311
43200/60000 [====================>.........] - ETA: 30s - loss: 0.2222 - categorical_accuracy: 0.9311
43232/60000 [====================>.........] - ETA: 30s - loss: 0.2223 - categorical_accuracy: 0.9311
43264/60000 [====================>.........] - ETA: 30s - loss: 0.2221 - categorical_accuracy: 0.9312
43296/60000 [====================>.........] - ETA: 30s - loss: 0.2220 - categorical_accuracy: 0.9312
43328/60000 [====================>.........] - ETA: 30s - loss: 0.2218 - categorical_accuracy: 0.9313
43360/60000 [====================>.........] - ETA: 30s - loss: 0.2217 - categorical_accuracy: 0.9313
43392/60000 [====================>.........] - ETA: 30s - loss: 0.2215 - categorical_accuracy: 0.9314
43424/60000 [====================>.........] - ETA: 30s - loss: 0.2215 - categorical_accuracy: 0.9314
43456/60000 [====================>.........] - ETA: 30s - loss: 0.2214 - categorical_accuracy: 0.9314
43488/60000 [====================>.........] - ETA: 29s - loss: 0.2213 - categorical_accuracy: 0.9314
43520/60000 [====================>.........] - ETA: 29s - loss: 0.2212 - categorical_accuracy: 0.9314
43552/60000 [====================>.........] - ETA: 29s - loss: 0.2213 - categorical_accuracy: 0.9314
43584/60000 [====================>.........] - ETA: 29s - loss: 0.2213 - categorical_accuracy: 0.9314
43616/60000 [====================>.........] - ETA: 29s - loss: 0.2212 - categorical_accuracy: 0.9314
43648/60000 [====================>.........] - ETA: 29s - loss: 0.2212 - categorical_accuracy: 0.9314
43680/60000 [====================>.........] - ETA: 29s - loss: 0.2212 - categorical_accuracy: 0.9314
43712/60000 [====================>.........] - ETA: 29s - loss: 0.2210 - categorical_accuracy: 0.9315
43744/60000 [====================>.........] - ETA: 29s - loss: 0.2209 - categorical_accuracy: 0.9315
43776/60000 [====================>.........] - ETA: 29s - loss: 0.2208 - categorical_accuracy: 0.9316
43808/60000 [====================>.........] - ETA: 29s - loss: 0.2207 - categorical_accuracy: 0.9316
43840/60000 [====================>.........] - ETA: 29s - loss: 0.2205 - categorical_accuracy: 0.9317
43872/60000 [====================>.........] - ETA: 29s - loss: 0.2205 - categorical_accuracy: 0.9317
43904/60000 [====================>.........] - ETA: 29s - loss: 0.2204 - categorical_accuracy: 0.9317
43936/60000 [====================>.........] - ETA: 29s - loss: 0.2203 - categorical_accuracy: 0.9317
43968/60000 [====================>.........] - ETA: 29s - loss: 0.2201 - categorical_accuracy: 0.9318
44000/60000 [=====================>........] - ETA: 29s - loss: 0.2200 - categorical_accuracy: 0.9318
44032/60000 [=====================>........] - ETA: 29s - loss: 0.2201 - categorical_accuracy: 0.9318
44064/60000 [=====================>........] - ETA: 28s - loss: 0.2201 - categorical_accuracy: 0.9318
44096/60000 [=====================>........] - ETA: 28s - loss: 0.2199 - categorical_accuracy: 0.9319
44128/60000 [=====================>........] - ETA: 28s - loss: 0.2198 - categorical_accuracy: 0.9319
44160/60000 [=====================>........] - ETA: 28s - loss: 0.2196 - categorical_accuracy: 0.9320
44192/60000 [=====================>........] - ETA: 28s - loss: 0.2195 - categorical_accuracy: 0.9320
44224/60000 [=====================>........] - ETA: 28s - loss: 0.2194 - categorical_accuracy: 0.9320
44256/60000 [=====================>........] - ETA: 28s - loss: 0.2193 - categorical_accuracy: 0.9321
44288/60000 [=====================>........] - ETA: 28s - loss: 0.2193 - categorical_accuracy: 0.9321
44320/60000 [=====================>........] - ETA: 28s - loss: 0.2191 - categorical_accuracy: 0.9321
44352/60000 [=====================>........] - ETA: 28s - loss: 0.2191 - categorical_accuracy: 0.9321
44384/60000 [=====================>........] - ETA: 28s - loss: 0.2190 - categorical_accuracy: 0.9321
44416/60000 [=====================>........] - ETA: 28s - loss: 0.2189 - categorical_accuracy: 0.9321
44448/60000 [=====================>........] - ETA: 28s - loss: 0.2188 - categorical_accuracy: 0.9322
44480/60000 [=====================>........] - ETA: 28s - loss: 0.2187 - categorical_accuracy: 0.9322
44512/60000 [=====================>........] - ETA: 28s - loss: 0.2185 - categorical_accuracy: 0.9323
44544/60000 [=====================>........] - ETA: 28s - loss: 0.2186 - categorical_accuracy: 0.9323
44576/60000 [=====================>........] - ETA: 28s - loss: 0.2186 - categorical_accuracy: 0.9323
44608/60000 [=====================>........] - ETA: 27s - loss: 0.2185 - categorical_accuracy: 0.9323
44640/60000 [=====================>........] - ETA: 27s - loss: 0.2184 - categorical_accuracy: 0.9323
44672/60000 [=====================>........] - ETA: 27s - loss: 0.2183 - categorical_accuracy: 0.9323
44704/60000 [=====================>........] - ETA: 27s - loss: 0.2182 - categorical_accuracy: 0.9324
44736/60000 [=====================>........] - ETA: 27s - loss: 0.2187 - categorical_accuracy: 0.9323
44768/60000 [=====================>........] - ETA: 27s - loss: 0.2186 - categorical_accuracy: 0.9324
44800/60000 [=====================>........] - ETA: 27s - loss: 0.2185 - categorical_accuracy: 0.9324
44832/60000 [=====================>........] - ETA: 27s - loss: 0.2185 - categorical_accuracy: 0.9324
44864/60000 [=====================>........] - ETA: 27s - loss: 0.2184 - categorical_accuracy: 0.9324
44896/60000 [=====================>........] - ETA: 27s - loss: 0.2183 - categorical_accuracy: 0.9324
44928/60000 [=====================>........] - ETA: 27s - loss: 0.2182 - categorical_accuracy: 0.9325
44960/60000 [=====================>........] - ETA: 27s - loss: 0.2181 - categorical_accuracy: 0.9325
44992/60000 [=====================>........] - ETA: 27s - loss: 0.2181 - categorical_accuracy: 0.9325
45024/60000 [=====================>........] - ETA: 27s - loss: 0.2180 - categorical_accuracy: 0.9325
45056/60000 [=====================>........] - ETA: 27s - loss: 0.2179 - categorical_accuracy: 0.9325
45088/60000 [=====================>........] - ETA: 27s - loss: 0.2179 - categorical_accuracy: 0.9326
45120/60000 [=====================>........] - ETA: 27s - loss: 0.2178 - categorical_accuracy: 0.9326
45152/60000 [=====================>........] - ETA: 26s - loss: 0.2178 - categorical_accuracy: 0.9326
45184/60000 [=====================>........] - ETA: 26s - loss: 0.2177 - categorical_accuracy: 0.9326
45216/60000 [=====================>........] - ETA: 26s - loss: 0.2176 - categorical_accuracy: 0.9327
45248/60000 [=====================>........] - ETA: 26s - loss: 0.2174 - categorical_accuracy: 0.9327
45280/60000 [=====================>........] - ETA: 26s - loss: 0.2173 - categorical_accuracy: 0.9328
45312/60000 [=====================>........] - ETA: 26s - loss: 0.2172 - categorical_accuracy: 0.9328
45344/60000 [=====================>........] - ETA: 26s - loss: 0.2171 - categorical_accuracy: 0.9328
45376/60000 [=====================>........] - ETA: 26s - loss: 0.2170 - categorical_accuracy: 0.9328
45408/60000 [=====================>........] - ETA: 26s - loss: 0.2170 - categorical_accuracy: 0.9329
45440/60000 [=====================>........] - ETA: 26s - loss: 0.2169 - categorical_accuracy: 0.9329
45472/60000 [=====================>........] - ETA: 26s - loss: 0.2168 - categorical_accuracy: 0.9329
45504/60000 [=====================>........] - ETA: 26s - loss: 0.2168 - categorical_accuracy: 0.9330
45536/60000 [=====================>........] - ETA: 26s - loss: 0.2168 - categorical_accuracy: 0.9330
45568/60000 [=====================>........] - ETA: 26s - loss: 0.2168 - categorical_accuracy: 0.9330
45600/60000 [=====================>........] - ETA: 26s - loss: 0.2167 - categorical_accuracy: 0.9330
45632/60000 [=====================>........] - ETA: 26s - loss: 0.2167 - categorical_accuracy: 0.9330
45664/60000 [=====================>........] - ETA: 26s - loss: 0.2166 - categorical_accuracy: 0.9331
45696/60000 [=====================>........] - ETA: 25s - loss: 0.2164 - categorical_accuracy: 0.9331
45728/60000 [=====================>........] - ETA: 25s - loss: 0.2163 - categorical_accuracy: 0.9331
45760/60000 [=====================>........] - ETA: 25s - loss: 0.2162 - categorical_accuracy: 0.9332
45792/60000 [=====================>........] - ETA: 25s - loss: 0.2162 - categorical_accuracy: 0.9332
45824/60000 [=====================>........] - ETA: 25s - loss: 0.2163 - categorical_accuracy: 0.9332
45856/60000 [=====================>........] - ETA: 25s - loss: 0.2161 - categorical_accuracy: 0.9332
45888/60000 [=====================>........] - ETA: 25s - loss: 0.2160 - categorical_accuracy: 0.9333
45920/60000 [=====================>........] - ETA: 25s - loss: 0.2160 - categorical_accuracy: 0.9333
45952/60000 [=====================>........] - ETA: 25s - loss: 0.2159 - categorical_accuracy: 0.9333
45984/60000 [=====================>........] - ETA: 25s - loss: 0.2158 - categorical_accuracy: 0.9333
46016/60000 [======================>.......] - ETA: 25s - loss: 0.2158 - categorical_accuracy: 0.9333
46048/60000 [======================>.......] - ETA: 25s - loss: 0.2157 - categorical_accuracy: 0.9333
46080/60000 [======================>.......] - ETA: 25s - loss: 0.2156 - categorical_accuracy: 0.9333
46112/60000 [======================>.......] - ETA: 25s - loss: 0.2155 - categorical_accuracy: 0.9333
46144/60000 [======================>.......] - ETA: 25s - loss: 0.2154 - categorical_accuracy: 0.9334
46176/60000 [======================>.......] - ETA: 25s - loss: 0.2152 - categorical_accuracy: 0.9334
46208/60000 [======================>.......] - ETA: 25s - loss: 0.2151 - categorical_accuracy: 0.9334
46240/60000 [======================>.......] - ETA: 24s - loss: 0.2151 - categorical_accuracy: 0.9335
46272/60000 [======================>.......] - ETA: 24s - loss: 0.2150 - categorical_accuracy: 0.9335
46304/60000 [======================>.......] - ETA: 24s - loss: 0.2149 - categorical_accuracy: 0.9335
46336/60000 [======================>.......] - ETA: 24s - loss: 0.2147 - categorical_accuracy: 0.9336
46368/60000 [======================>.......] - ETA: 24s - loss: 0.2146 - categorical_accuracy: 0.9336
46400/60000 [======================>.......] - ETA: 24s - loss: 0.2145 - categorical_accuracy: 0.9336
46432/60000 [======================>.......] - ETA: 24s - loss: 0.2144 - categorical_accuracy: 0.9337
46464/60000 [======================>.......] - ETA: 24s - loss: 0.2144 - categorical_accuracy: 0.9336
46496/60000 [======================>.......] - ETA: 24s - loss: 0.2143 - categorical_accuracy: 0.9337
46528/60000 [======================>.......] - ETA: 24s - loss: 0.2142 - categorical_accuracy: 0.9337
46560/60000 [======================>.......] - ETA: 24s - loss: 0.2141 - categorical_accuracy: 0.9338
46592/60000 [======================>.......] - ETA: 24s - loss: 0.2141 - categorical_accuracy: 0.9338
46624/60000 [======================>.......] - ETA: 24s - loss: 0.2140 - categorical_accuracy: 0.9338
46656/60000 [======================>.......] - ETA: 24s - loss: 0.2140 - categorical_accuracy: 0.9338
46688/60000 [======================>.......] - ETA: 24s - loss: 0.2139 - categorical_accuracy: 0.9338
46720/60000 [======================>.......] - ETA: 24s - loss: 0.2139 - categorical_accuracy: 0.9338
46752/60000 [======================>.......] - ETA: 24s - loss: 0.2138 - categorical_accuracy: 0.9339
46784/60000 [======================>.......] - ETA: 24s - loss: 0.2137 - categorical_accuracy: 0.9339
46816/60000 [======================>.......] - ETA: 23s - loss: 0.2136 - categorical_accuracy: 0.9339
46848/60000 [======================>.......] - ETA: 23s - loss: 0.2135 - categorical_accuracy: 0.9339
46880/60000 [======================>.......] - ETA: 23s - loss: 0.2134 - categorical_accuracy: 0.9340
46912/60000 [======================>.......] - ETA: 23s - loss: 0.2134 - categorical_accuracy: 0.9340
46944/60000 [======================>.......] - ETA: 23s - loss: 0.2133 - categorical_accuracy: 0.9340
46976/60000 [======================>.......] - ETA: 23s - loss: 0.2132 - categorical_accuracy: 0.9340
47008/60000 [======================>.......] - ETA: 23s - loss: 0.2131 - categorical_accuracy: 0.9340
47040/60000 [======================>.......] - ETA: 23s - loss: 0.2131 - categorical_accuracy: 0.9340
47072/60000 [======================>.......] - ETA: 23s - loss: 0.2131 - categorical_accuracy: 0.9340
47104/60000 [======================>.......] - ETA: 23s - loss: 0.2130 - categorical_accuracy: 0.9341
47136/60000 [======================>.......] - ETA: 23s - loss: 0.2129 - categorical_accuracy: 0.9341
47168/60000 [======================>.......] - ETA: 23s - loss: 0.2128 - categorical_accuracy: 0.9341
47200/60000 [======================>.......] - ETA: 23s - loss: 0.2127 - categorical_accuracy: 0.9342
47232/60000 [======================>.......] - ETA: 23s - loss: 0.2127 - categorical_accuracy: 0.9342
47264/60000 [======================>.......] - ETA: 23s - loss: 0.2126 - categorical_accuracy: 0.9342
47296/60000 [======================>.......] - ETA: 23s - loss: 0.2124 - categorical_accuracy: 0.9342
47328/60000 [======================>.......] - ETA: 23s - loss: 0.2124 - categorical_accuracy: 0.9342
47360/60000 [======================>.......] - ETA: 22s - loss: 0.2123 - categorical_accuracy: 0.9343
47392/60000 [======================>.......] - ETA: 22s - loss: 0.2121 - categorical_accuracy: 0.9343
47424/60000 [======================>.......] - ETA: 22s - loss: 0.2121 - categorical_accuracy: 0.9343
47456/60000 [======================>.......] - ETA: 22s - loss: 0.2120 - categorical_accuracy: 0.9343
47488/60000 [======================>.......] - ETA: 22s - loss: 0.2121 - categorical_accuracy: 0.9343
47520/60000 [======================>.......] - ETA: 22s - loss: 0.2120 - categorical_accuracy: 0.9344
47552/60000 [======================>.......] - ETA: 22s - loss: 0.2119 - categorical_accuracy: 0.9344
47584/60000 [======================>.......] - ETA: 22s - loss: 0.2118 - categorical_accuracy: 0.9344
47616/60000 [======================>.......] - ETA: 22s - loss: 0.2117 - categorical_accuracy: 0.9345
47648/60000 [======================>.......] - ETA: 22s - loss: 0.2116 - categorical_accuracy: 0.9345
47680/60000 [======================>.......] - ETA: 22s - loss: 0.2115 - categorical_accuracy: 0.9345
47712/60000 [======================>.......] - ETA: 22s - loss: 0.2115 - categorical_accuracy: 0.9345
47744/60000 [======================>.......] - ETA: 22s - loss: 0.2114 - categorical_accuracy: 0.9346
47776/60000 [======================>.......] - ETA: 22s - loss: 0.2112 - categorical_accuracy: 0.9346
47808/60000 [======================>.......] - ETA: 22s - loss: 0.2113 - categorical_accuracy: 0.9346
47840/60000 [======================>.......] - ETA: 22s - loss: 0.2111 - categorical_accuracy: 0.9347
47872/60000 [======================>.......] - ETA: 22s - loss: 0.2111 - categorical_accuracy: 0.9347
47904/60000 [======================>.......] - ETA: 21s - loss: 0.2112 - categorical_accuracy: 0.9347
47936/60000 [======================>.......] - ETA: 21s - loss: 0.2112 - categorical_accuracy: 0.9347
47968/60000 [======================>.......] - ETA: 21s - loss: 0.2111 - categorical_accuracy: 0.9347
48000/60000 [=======================>......] - ETA: 21s - loss: 0.2110 - categorical_accuracy: 0.9347
48032/60000 [=======================>......] - ETA: 21s - loss: 0.2109 - categorical_accuracy: 0.9347
48064/60000 [=======================>......] - ETA: 21s - loss: 0.2108 - categorical_accuracy: 0.9348
48096/60000 [=======================>......] - ETA: 21s - loss: 0.2108 - categorical_accuracy: 0.9348
48128/60000 [=======================>......] - ETA: 21s - loss: 0.2107 - categorical_accuracy: 0.9348
48160/60000 [=======================>......] - ETA: 21s - loss: 0.2106 - categorical_accuracy: 0.9348
48192/60000 [=======================>......] - ETA: 21s - loss: 0.2105 - categorical_accuracy: 0.9349
48224/60000 [=======================>......] - ETA: 21s - loss: 0.2106 - categorical_accuracy: 0.9349
48256/60000 [=======================>......] - ETA: 21s - loss: 0.2106 - categorical_accuracy: 0.9349
48288/60000 [=======================>......] - ETA: 21s - loss: 0.2105 - categorical_accuracy: 0.9349
48320/60000 [=======================>......] - ETA: 21s - loss: 0.2104 - categorical_accuracy: 0.9350
48352/60000 [=======================>......] - ETA: 21s - loss: 0.2103 - categorical_accuracy: 0.9350
48384/60000 [=======================>......] - ETA: 21s - loss: 0.2101 - categorical_accuracy: 0.9350
48416/60000 [=======================>......] - ETA: 21s - loss: 0.2100 - categorical_accuracy: 0.9351
48448/60000 [=======================>......] - ETA: 20s - loss: 0.2099 - categorical_accuracy: 0.9351
48480/60000 [=======================>......] - ETA: 20s - loss: 0.2098 - categorical_accuracy: 0.9351
48512/60000 [=======================>......] - ETA: 20s - loss: 0.2096 - categorical_accuracy: 0.9352
48544/60000 [=======================>......] - ETA: 20s - loss: 0.2095 - categorical_accuracy: 0.9352
48576/60000 [=======================>......] - ETA: 20s - loss: 0.2094 - categorical_accuracy: 0.9353
48608/60000 [=======================>......] - ETA: 20s - loss: 0.2093 - categorical_accuracy: 0.9353
48640/60000 [=======================>......] - ETA: 20s - loss: 0.2092 - categorical_accuracy: 0.9353
48672/60000 [=======================>......] - ETA: 20s - loss: 0.2091 - categorical_accuracy: 0.9353
48704/60000 [=======================>......] - ETA: 20s - loss: 0.2090 - categorical_accuracy: 0.9354
48736/60000 [=======================>......] - ETA: 20s - loss: 0.2088 - categorical_accuracy: 0.9354
48768/60000 [=======================>......] - ETA: 20s - loss: 0.2087 - categorical_accuracy: 0.9355
48800/60000 [=======================>......] - ETA: 20s - loss: 0.2086 - categorical_accuracy: 0.9355
48832/60000 [=======================>......] - ETA: 20s - loss: 0.2085 - categorical_accuracy: 0.9355
48864/60000 [=======================>......] - ETA: 20s - loss: 0.2084 - categorical_accuracy: 0.9356
48896/60000 [=======================>......] - ETA: 20s - loss: 0.2083 - categorical_accuracy: 0.9356
48928/60000 [=======================>......] - ETA: 20s - loss: 0.2082 - categorical_accuracy: 0.9356
48960/60000 [=======================>......] - ETA: 20s - loss: 0.2081 - categorical_accuracy: 0.9357
48992/60000 [=======================>......] - ETA: 19s - loss: 0.2083 - categorical_accuracy: 0.9356
49024/60000 [=======================>......] - ETA: 19s - loss: 0.2081 - categorical_accuracy: 0.9357
49056/60000 [=======================>......] - ETA: 19s - loss: 0.2080 - categorical_accuracy: 0.9357
49088/60000 [=======================>......] - ETA: 19s - loss: 0.2079 - categorical_accuracy: 0.9357
49120/60000 [=======================>......] - ETA: 19s - loss: 0.2078 - categorical_accuracy: 0.9358
49152/60000 [=======================>......] - ETA: 19s - loss: 0.2077 - categorical_accuracy: 0.9358
49184/60000 [=======================>......] - ETA: 19s - loss: 0.2076 - categorical_accuracy: 0.9359
49216/60000 [=======================>......] - ETA: 19s - loss: 0.2075 - categorical_accuracy: 0.9359
49248/60000 [=======================>......] - ETA: 19s - loss: 0.2075 - categorical_accuracy: 0.9359
49280/60000 [=======================>......] - ETA: 19s - loss: 0.2075 - categorical_accuracy: 0.9359
49312/60000 [=======================>......] - ETA: 19s - loss: 0.2074 - categorical_accuracy: 0.9359
49344/60000 [=======================>......] - ETA: 19s - loss: 0.2075 - categorical_accuracy: 0.9359
49376/60000 [=======================>......] - ETA: 19s - loss: 0.2074 - categorical_accuracy: 0.9359
49408/60000 [=======================>......] - ETA: 19s - loss: 0.2073 - categorical_accuracy: 0.9359
49440/60000 [=======================>......] - ETA: 19s - loss: 0.2072 - categorical_accuracy: 0.9360
49472/60000 [=======================>......] - ETA: 19s - loss: 0.2071 - categorical_accuracy: 0.9360
49504/60000 [=======================>......] - ETA: 19s - loss: 0.2070 - categorical_accuracy: 0.9360
49536/60000 [=======================>......] - ETA: 18s - loss: 0.2069 - categorical_accuracy: 0.9361
49568/60000 [=======================>......] - ETA: 18s - loss: 0.2068 - categorical_accuracy: 0.9361
49600/60000 [=======================>......] - ETA: 18s - loss: 0.2067 - categorical_accuracy: 0.9361
49632/60000 [=======================>......] - ETA: 18s - loss: 0.2066 - categorical_accuracy: 0.9362
49664/60000 [=======================>......] - ETA: 18s - loss: 0.2065 - categorical_accuracy: 0.9362
49696/60000 [=======================>......] - ETA: 18s - loss: 0.2066 - categorical_accuracy: 0.9362
49728/60000 [=======================>......] - ETA: 18s - loss: 0.2066 - categorical_accuracy: 0.9362
49760/60000 [=======================>......] - ETA: 18s - loss: 0.2066 - categorical_accuracy: 0.9362
49792/60000 [=======================>......] - ETA: 18s - loss: 0.2067 - categorical_accuracy: 0.9362
49824/60000 [=======================>......] - ETA: 18s - loss: 0.2067 - categorical_accuracy: 0.9362
49856/60000 [=======================>......] - ETA: 18s - loss: 0.2067 - categorical_accuracy: 0.9362
49888/60000 [=======================>......] - ETA: 18s - loss: 0.2066 - categorical_accuracy: 0.9362
49920/60000 [=======================>......] - ETA: 18s - loss: 0.2065 - categorical_accuracy: 0.9363
49952/60000 [=======================>......] - ETA: 18s - loss: 0.2064 - categorical_accuracy: 0.9363
49984/60000 [=======================>......] - ETA: 18s - loss: 0.2063 - categorical_accuracy: 0.9363
50016/60000 [========================>.....] - ETA: 18s - loss: 0.2062 - categorical_accuracy: 0.9364
50048/60000 [========================>.....] - ETA: 18s - loss: 0.2063 - categorical_accuracy: 0.9363
50080/60000 [========================>.....] - ETA: 18s - loss: 0.2064 - categorical_accuracy: 0.9363
50112/60000 [========================>.....] - ETA: 17s - loss: 0.2064 - categorical_accuracy: 0.9363
50144/60000 [========================>.....] - ETA: 17s - loss: 0.2063 - categorical_accuracy: 0.9363
50176/60000 [========================>.....] - ETA: 17s - loss: 0.2062 - categorical_accuracy: 0.9363
50208/60000 [========================>.....] - ETA: 17s - loss: 0.2061 - categorical_accuracy: 0.9363
50240/60000 [========================>.....] - ETA: 17s - loss: 0.2060 - categorical_accuracy: 0.9364
50272/60000 [========================>.....] - ETA: 17s - loss: 0.2059 - categorical_accuracy: 0.9364
50304/60000 [========================>.....] - ETA: 17s - loss: 0.2058 - categorical_accuracy: 0.9364
50336/60000 [========================>.....] - ETA: 17s - loss: 0.2057 - categorical_accuracy: 0.9364
50368/60000 [========================>.....] - ETA: 17s - loss: 0.2057 - categorical_accuracy: 0.9364
50400/60000 [========================>.....] - ETA: 17s - loss: 0.2056 - categorical_accuracy: 0.9365
50432/60000 [========================>.....] - ETA: 17s - loss: 0.2055 - categorical_accuracy: 0.9365
50464/60000 [========================>.....] - ETA: 17s - loss: 0.2054 - categorical_accuracy: 0.9365
50496/60000 [========================>.....] - ETA: 17s - loss: 0.2053 - categorical_accuracy: 0.9366
50528/60000 [========================>.....] - ETA: 17s - loss: 0.2053 - categorical_accuracy: 0.9366
50560/60000 [========================>.....] - ETA: 17s - loss: 0.2052 - categorical_accuracy: 0.9366
50592/60000 [========================>.....] - ETA: 17s - loss: 0.2051 - categorical_accuracy: 0.9367
50624/60000 [========================>.....] - ETA: 17s - loss: 0.2050 - categorical_accuracy: 0.9367
50656/60000 [========================>.....] - ETA: 16s - loss: 0.2049 - categorical_accuracy: 0.9367
50688/60000 [========================>.....] - ETA: 16s - loss: 0.2048 - categorical_accuracy: 0.9368
50720/60000 [========================>.....] - ETA: 16s - loss: 0.2048 - categorical_accuracy: 0.9368
50752/60000 [========================>.....] - ETA: 16s - loss: 0.2047 - categorical_accuracy: 0.9368
50784/60000 [========================>.....] - ETA: 16s - loss: 0.2047 - categorical_accuracy: 0.9368
50816/60000 [========================>.....] - ETA: 16s - loss: 0.2045 - categorical_accuracy: 0.9368
50848/60000 [========================>.....] - ETA: 16s - loss: 0.2044 - categorical_accuracy: 0.9369
50880/60000 [========================>.....] - ETA: 16s - loss: 0.2044 - categorical_accuracy: 0.9368
50912/60000 [========================>.....] - ETA: 16s - loss: 0.2044 - categorical_accuracy: 0.9368
50944/60000 [========================>.....] - ETA: 16s - loss: 0.2044 - categorical_accuracy: 0.9368
50976/60000 [========================>.....] - ETA: 16s - loss: 0.2043 - categorical_accuracy: 0.9368
51008/60000 [========================>.....] - ETA: 16s - loss: 0.2043 - categorical_accuracy: 0.9369
51040/60000 [========================>.....] - ETA: 16s - loss: 0.2042 - categorical_accuracy: 0.9369
51072/60000 [========================>.....] - ETA: 16s - loss: 0.2041 - categorical_accuracy: 0.9369
51104/60000 [========================>.....] - ETA: 16s - loss: 0.2041 - categorical_accuracy: 0.9369
51136/60000 [========================>.....] - ETA: 16s - loss: 0.2040 - categorical_accuracy: 0.9370
51168/60000 [========================>.....] - ETA: 16s - loss: 0.2039 - categorical_accuracy: 0.9370
51200/60000 [========================>.....] - ETA: 15s - loss: 0.2038 - categorical_accuracy: 0.9370
51232/60000 [========================>.....] - ETA: 15s - loss: 0.2037 - categorical_accuracy: 0.9371
51264/60000 [========================>.....] - ETA: 15s - loss: 0.2037 - categorical_accuracy: 0.9371
51296/60000 [========================>.....] - ETA: 15s - loss: 0.2036 - categorical_accuracy: 0.9371
51328/60000 [========================>.....] - ETA: 15s - loss: 0.2034 - categorical_accuracy: 0.9372
51360/60000 [========================>.....] - ETA: 15s - loss: 0.2034 - categorical_accuracy: 0.9372
51392/60000 [========================>.....] - ETA: 15s - loss: 0.2033 - categorical_accuracy: 0.9372
51424/60000 [========================>.....] - ETA: 15s - loss: 0.2033 - categorical_accuracy: 0.9372
51456/60000 [========================>.....] - ETA: 15s - loss: 0.2033 - categorical_accuracy: 0.9372
51488/60000 [========================>.....] - ETA: 15s - loss: 0.2032 - categorical_accuracy: 0.9372
51520/60000 [========================>.....] - ETA: 15s - loss: 0.2032 - categorical_accuracy: 0.9372
51552/60000 [========================>.....] - ETA: 15s - loss: 0.2030 - categorical_accuracy: 0.9373
51584/60000 [========================>.....] - ETA: 15s - loss: 0.2032 - categorical_accuracy: 0.9373
51616/60000 [========================>.....] - ETA: 15s - loss: 0.2032 - categorical_accuracy: 0.9372
51648/60000 [========================>.....] - ETA: 15s - loss: 0.2031 - categorical_accuracy: 0.9373
51680/60000 [========================>.....] - ETA: 15s - loss: 0.2030 - categorical_accuracy: 0.9373
51712/60000 [========================>.....] - ETA: 15s - loss: 0.2029 - categorical_accuracy: 0.9373
51744/60000 [========================>.....] - ETA: 14s - loss: 0.2028 - categorical_accuracy: 0.9374
51776/60000 [========================>.....] - ETA: 14s - loss: 0.2028 - categorical_accuracy: 0.9374
51808/60000 [========================>.....] - ETA: 14s - loss: 0.2027 - categorical_accuracy: 0.9374
51840/60000 [========================>.....] - ETA: 14s - loss: 0.2026 - categorical_accuracy: 0.9374
51872/60000 [========================>.....] - ETA: 14s - loss: 0.2025 - categorical_accuracy: 0.9375
51904/60000 [========================>.....] - ETA: 14s - loss: 0.2025 - categorical_accuracy: 0.9375
51936/60000 [========================>.....] - ETA: 14s - loss: 0.2025 - categorical_accuracy: 0.9375
51968/60000 [========================>.....] - ETA: 14s - loss: 0.2025 - categorical_accuracy: 0.9375
52000/60000 [=========================>....] - ETA: 14s - loss: 0.2024 - categorical_accuracy: 0.9375
52032/60000 [=========================>....] - ETA: 14s - loss: 0.2023 - categorical_accuracy: 0.9375
52064/60000 [=========================>....] - ETA: 14s - loss: 0.2026 - categorical_accuracy: 0.9375
52096/60000 [=========================>....] - ETA: 14s - loss: 0.2027 - categorical_accuracy: 0.9374
52128/60000 [=========================>....] - ETA: 14s - loss: 0.2026 - categorical_accuracy: 0.9375
52160/60000 [=========================>....] - ETA: 14s - loss: 0.2025 - categorical_accuracy: 0.9375
52192/60000 [=========================>....] - ETA: 14s - loss: 0.2025 - categorical_accuracy: 0.9375
52224/60000 [=========================>....] - ETA: 14s - loss: 0.2024 - categorical_accuracy: 0.9375
52256/60000 [=========================>....] - ETA: 14s - loss: 0.2024 - categorical_accuracy: 0.9375
52288/60000 [=========================>....] - ETA: 13s - loss: 0.2023 - categorical_accuracy: 0.9376
52320/60000 [=========================>....] - ETA: 13s - loss: 0.2022 - categorical_accuracy: 0.9376
52352/60000 [=========================>....] - ETA: 13s - loss: 0.2021 - categorical_accuracy: 0.9376
52384/60000 [=========================>....] - ETA: 13s - loss: 0.2020 - categorical_accuracy: 0.9376
52416/60000 [=========================>....] - ETA: 13s - loss: 0.2021 - categorical_accuracy: 0.9376
52448/60000 [=========================>....] - ETA: 13s - loss: 0.2021 - categorical_accuracy: 0.9376
52480/60000 [=========================>....] - ETA: 13s - loss: 0.2022 - categorical_accuracy: 0.9376
52512/60000 [=========================>....] - ETA: 13s - loss: 0.2021 - categorical_accuracy: 0.9376
52544/60000 [=========================>....] - ETA: 13s - loss: 0.2021 - categorical_accuracy: 0.9376
52576/60000 [=========================>....] - ETA: 13s - loss: 0.2020 - categorical_accuracy: 0.9377
52608/60000 [=========================>....] - ETA: 13s - loss: 0.2019 - categorical_accuracy: 0.9377
52640/60000 [=========================>....] - ETA: 13s - loss: 0.2018 - categorical_accuracy: 0.9377
52672/60000 [=========================>....] - ETA: 13s - loss: 0.2017 - categorical_accuracy: 0.9377
52704/60000 [=========================>....] - ETA: 13s - loss: 0.2016 - categorical_accuracy: 0.9378
52736/60000 [=========================>....] - ETA: 13s - loss: 0.2015 - categorical_accuracy: 0.9378
52768/60000 [=========================>....] - ETA: 13s - loss: 0.2016 - categorical_accuracy: 0.9378
52800/60000 [=========================>....] - ETA: 13s - loss: 0.2015 - categorical_accuracy: 0.9378
52832/60000 [=========================>....] - ETA: 13s - loss: 0.2014 - categorical_accuracy: 0.9378
52864/60000 [=========================>....] - ETA: 12s - loss: 0.2014 - categorical_accuracy: 0.9379
52896/60000 [=========================>....] - ETA: 12s - loss: 0.2013 - categorical_accuracy: 0.9379
52928/60000 [=========================>....] - ETA: 12s - loss: 0.2012 - categorical_accuracy: 0.9379
52960/60000 [=========================>....] - ETA: 12s - loss: 0.2011 - categorical_accuracy: 0.9380
52992/60000 [=========================>....] - ETA: 12s - loss: 0.2010 - categorical_accuracy: 0.9380
53024/60000 [=========================>....] - ETA: 12s - loss: 0.2009 - categorical_accuracy: 0.9380
53056/60000 [=========================>....] - ETA: 12s - loss: 0.2008 - categorical_accuracy: 0.9380
53088/60000 [=========================>....] - ETA: 12s - loss: 0.2008 - categorical_accuracy: 0.9380
53120/60000 [=========================>....] - ETA: 12s - loss: 0.2007 - categorical_accuracy: 0.9380
53152/60000 [=========================>....] - ETA: 12s - loss: 0.2006 - categorical_accuracy: 0.9380
53184/60000 [=========================>....] - ETA: 12s - loss: 0.2006 - categorical_accuracy: 0.9381
53216/60000 [=========================>....] - ETA: 12s - loss: 0.2005 - categorical_accuracy: 0.9381
53248/60000 [=========================>....] - ETA: 12s - loss: 0.2005 - categorical_accuracy: 0.9381
53280/60000 [=========================>....] - ETA: 12s - loss: 0.2004 - categorical_accuracy: 0.9381
53312/60000 [=========================>....] - ETA: 12s - loss: 0.2003 - categorical_accuracy: 0.9381
53344/60000 [=========================>....] - ETA: 12s - loss: 0.2003 - categorical_accuracy: 0.9382
53376/60000 [=========================>....] - ETA: 12s - loss: 0.2003 - categorical_accuracy: 0.9382
53408/60000 [=========================>....] - ETA: 11s - loss: 0.2002 - categorical_accuracy: 0.9382
53440/60000 [=========================>....] - ETA: 11s - loss: 0.2001 - categorical_accuracy: 0.9382
53472/60000 [=========================>....] - ETA: 11s - loss: 0.2000 - categorical_accuracy: 0.9382
53504/60000 [=========================>....] - ETA: 11s - loss: 0.1999 - categorical_accuracy: 0.9383
53536/60000 [=========================>....] - ETA: 11s - loss: 0.1998 - categorical_accuracy: 0.9383
53568/60000 [=========================>....] - ETA: 11s - loss: 0.1998 - categorical_accuracy: 0.9383
53600/60000 [=========================>....] - ETA: 11s - loss: 0.1997 - categorical_accuracy: 0.9383
53632/60000 [=========================>....] - ETA: 11s - loss: 0.1997 - categorical_accuracy: 0.9384
53664/60000 [=========================>....] - ETA: 11s - loss: 0.1995 - categorical_accuracy: 0.9384
53696/60000 [=========================>....] - ETA: 11s - loss: 0.1994 - categorical_accuracy: 0.9384
53728/60000 [=========================>....] - ETA: 11s - loss: 0.1994 - categorical_accuracy: 0.9385
53760/60000 [=========================>....] - ETA: 11s - loss: 0.1993 - categorical_accuracy: 0.9385
53792/60000 [=========================>....] - ETA: 11s - loss: 0.1993 - categorical_accuracy: 0.9385
53824/60000 [=========================>....] - ETA: 11s - loss: 0.1991 - categorical_accuracy: 0.9385
53856/60000 [=========================>....] - ETA: 11s - loss: 0.1991 - categorical_accuracy: 0.9385
53888/60000 [=========================>....] - ETA: 11s - loss: 0.1990 - categorical_accuracy: 0.9386
53920/60000 [=========================>....] - ETA: 11s - loss: 0.1989 - categorical_accuracy: 0.9386
53952/60000 [=========================>....] - ETA: 10s - loss: 0.1988 - categorical_accuracy: 0.9386
53984/60000 [=========================>....] - ETA: 10s - loss: 0.1987 - categorical_accuracy: 0.9387
54016/60000 [==========================>...] - ETA: 10s - loss: 0.1986 - categorical_accuracy: 0.9387
54048/60000 [==========================>...] - ETA: 10s - loss: 0.1985 - categorical_accuracy: 0.9387
54080/60000 [==========================>...] - ETA: 10s - loss: 0.1984 - categorical_accuracy: 0.9387
54112/60000 [==========================>...] - ETA: 10s - loss: 0.1984 - categorical_accuracy: 0.9388
54144/60000 [==========================>...] - ETA: 10s - loss: 0.1983 - categorical_accuracy: 0.9388
54176/60000 [==========================>...] - ETA: 10s - loss: 0.1982 - categorical_accuracy: 0.9388
54208/60000 [==========================>...] - ETA: 10s - loss: 0.1981 - categorical_accuracy: 0.9389
54240/60000 [==========================>...] - ETA: 10s - loss: 0.1980 - categorical_accuracy: 0.9389
54272/60000 [==========================>...] - ETA: 10s - loss: 0.1980 - categorical_accuracy: 0.9389
54304/60000 [==========================>...] - ETA: 10s - loss: 0.1979 - categorical_accuracy: 0.9389
54336/60000 [==========================>...] - ETA: 10s - loss: 0.1978 - categorical_accuracy: 0.9390
54368/60000 [==========================>...] - ETA: 10s - loss: 0.1977 - categorical_accuracy: 0.9390
54400/60000 [==========================>...] - ETA: 10s - loss: 0.1977 - categorical_accuracy: 0.9390
54432/60000 [==========================>...] - ETA: 10s - loss: 0.1977 - categorical_accuracy: 0.9390
54464/60000 [==========================>...] - ETA: 10s - loss: 0.1977 - categorical_accuracy: 0.9390
54496/60000 [==========================>...] - ETA: 9s - loss: 0.1976 - categorical_accuracy: 0.9390 
54528/60000 [==========================>...] - ETA: 9s - loss: 0.1976 - categorical_accuracy: 0.9390
54560/60000 [==========================>...] - ETA: 9s - loss: 0.1976 - categorical_accuracy: 0.9390
54592/60000 [==========================>...] - ETA: 9s - loss: 0.1975 - categorical_accuracy: 0.9391
54624/60000 [==========================>...] - ETA: 9s - loss: 0.1974 - categorical_accuracy: 0.9391
54656/60000 [==========================>...] - ETA: 9s - loss: 0.1974 - categorical_accuracy: 0.9391
54688/60000 [==========================>...] - ETA: 9s - loss: 0.1973 - categorical_accuracy: 0.9391
54720/60000 [==========================>...] - ETA: 9s - loss: 0.1972 - categorical_accuracy: 0.9391
54752/60000 [==========================>...] - ETA: 9s - loss: 0.1971 - categorical_accuracy: 0.9392
54784/60000 [==========================>...] - ETA: 9s - loss: 0.1971 - categorical_accuracy: 0.9392
54816/60000 [==========================>...] - ETA: 9s - loss: 0.1970 - categorical_accuracy: 0.9392
54848/60000 [==========================>...] - ETA: 9s - loss: 0.1969 - categorical_accuracy: 0.9392
54880/60000 [==========================>...] - ETA: 9s - loss: 0.1969 - categorical_accuracy: 0.9392
54912/60000 [==========================>...] - ETA: 9s - loss: 0.1968 - categorical_accuracy: 0.9392
54944/60000 [==========================>...] - ETA: 9s - loss: 0.1967 - categorical_accuracy: 0.9393
54976/60000 [==========================>...] - ETA: 9s - loss: 0.1966 - categorical_accuracy: 0.9393
55008/60000 [==========================>...] - ETA: 9s - loss: 0.1965 - categorical_accuracy: 0.9393
55040/60000 [==========================>...] - ETA: 9s - loss: 0.1964 - categorical_accuracy: 0.9393
55072/60000 [==========================>...] - ETA: 8s - loss: 0.1964 - categorical_accuracy: 0.9394
55104/60000 [==========================>...] - ETA: 8s - loss: 0.1963 - categorical_accuracy: 0.9394
55136/60000 [==========================>...] - ETA: 8s - loss: 0.1963 - categorical_accuracy: 0.9394
55168/60000 [==========================>...] - ETA: 8s - loss: 0.1963 - categorical_accuracy: 0.9394
55200/60000 [==========================>...] - ETA: 8s - loss: 0.1962 - categorical_accuracy: 0.9394
55232/60000 [==========================>...] - ETA: 8s - loss: 0.1961 - categorical_accuracy: 0.9395
55264/60000 [==========================>...] - ETA: 8s - loss: 0.1960 - categorical_accuracy: 0.9395
55296/60000 [==========================>...] - ETA: 8s - loss: 0.1961 - categorical_accuracy: 0.9395
55328/60000 [==========================>...] - ETA: 8s - loss: 0.1960 - categorical_accuracy: 0.9395
55360/60000 [==========================>...] - ETA: 8s - loss: 0.1960 - categorical_accuracy: 0.9395
55392/60000 [==========================>...] - ETA: 8s - loss: 0.1960 - categorical_accuracy: 0.9395
55424/60000 [==========================>...] - ETA: 8s - loss: 0.1960 - categorical_accuracy: 0.9395
55456/60000 [==========================>...] - ETA: 8s - loss: 0.1960 - categorical_accuracy: 0.9395
55488/60000 [==========================>...] - ETA: 8s - loss: 0.1959 - categorical_accuracy: 0.9395
55520/60000 [==========================>...] - ETA: 8s - loss: 0.1958 - categorical_accuracy: 0.9396
55552/60000 [==========================>...] - ETA: 8s - loss: 0.1957 - categorical_accuracy: 0.9396
55584/60000 [==========================>...] - ETA: 8s - loss: 0.1956 - categorical_accuracy: 0.9396
55616/60000 [==========================>...] - ETA: 7s - loss: 0.1955 - categorical_accuracy: 0.9397
55648/60000 [==========================>...] - ETA: 7s - loss: 0.1954 - categorical_accuracy: 0.9397
55680/60000 [==========================>...] - ETA: 7s - loss: 0.1953 - categorical_accuracy: 0.9397
55712/60000 [==========================>...] - ETA: 7s - loss: 0.1952 - categorical_accuracy: 0.9397
55744/60000 [==========================>...] - ETA: 7s - loss: 0.1952 - categorical_accuracy: 0.9398
55776/60000 [==========================>...] - ETA: 7s - loss: 0.1952 - categorical_accuracy: 0.9398
55808/60000 [==========================>...] - ETA: 7s - loss: 0.1951 - categorical_accuracy: 0.9398
55840/60000 [==========================>...] - ETA: 7s - loss: 0.1950 - categorical_accuracy: 0.9398
55872/60000 [==========================>...] - ETA: 7s - loss: 0.1949 - categorical_accuracy: 0.9399
55904/60000 [==========================>...] - ETA: 7s - loss: 0.1948 - categorical_accuracy: 0.9399
55936/60000 [==========================>...] - ETA: 7s - loss: 0.1947 - categorical_accuracy: 0.9399
55968/60000 [==========================>...] - ETA: 7s - loss: 0.1947 - categorical_accuracy: 0.9399
56000/60000 [===========================>..] - ETA: 7s - loss: 0.1946 - categorical_accuracy: 0.9400
56032/60000 [===========================>..] - ETA: 7s - loss: 0.1946 - categorical_accuracy: 0.9400
56064/60000 [===========================>..] - ETA: 7s - loss: 0.1946 - categorical_accuracy: 0.9400
56096/60000 [===========================>..] - ETA: 7s - loss: 0.1945 - categorical_accuracy: 0.9400
56128/60000 [===========================>..] - ETA: 7s - loss: 0.1944 - categorical_accuracy: 0.9401
56160/60000 [===========================>..] - ETA: 6s - loss: 0.1943 - categorical_accuracy: 0.9401
56192/60000 [===========================>..] - ETA: 6s - loss: 0.1942 - categorical_accuracy: 0.9401
56224/60000 [===========================>..] - ETA: 6s - loss: 0.1941 - categorical_accuracy: 0.9401
56256/60000 [===========================>..] - ETA: 6s - loss: 0.1940 - categorical_accuracy: 0.9401
56288/60000 [===========================>..] - ETA: 6s - loss: 0.1940 - categorical_accuracy: 0.9402
56320/60000 [===========================>..] - ETA: 6s - loss: 0.1939 - categorical_accuracy: 0.9402
56352/60000 [===========================>..] - ETA: 6s - loss: 0.1939 - categorical_accuracy: 0.9402
56384/60000 [===========================>..] - ETA: 6s - loss: 0.1938 - categorical_accuracy: 0.9402
56416/60000 [===========================>..] - ETA: 6s - loss: 0.1939 - categorical_accuracy: 0.9402
56448/60000 [===========================>..] - ETA: 6s - loss: 0.1938 - categorical_accuracy: 0.9403
56480/60000 [===========================>..] - ETA: 6s - loss: 0.1939 - categorical_accuracy: 0.9403
56512/60000 [===========================>..] - ETA: 6s - loss: 0.1938 - categorical_accuracy: 0.9403
56544/60000 [===========================>..] - ETA: 6s - loss: 0.1937 - categorical_accuracy: 0.9403
56576/60000 [===========================>..] - ETA: 6s - loss: 0.1937 - categorical_accuracy: 0.9403
56608/60000 [===========================>..] - ETA: 6s - loss: 0.1936 - categorical_accuracy: 0.9404
56640/60000 [===========================>..] - ETA: 6s - loss: 0.1935 - categorical_accuracy: 0.9404
56672/60000 [===========================>..] - ETA: 6s - loss: 0.1935 - categorical_accuracy: 0.9404
56704/60000 [===========================>..] - ETA: 5s - loss: 0.1935 - categorical_accuracy: 0.9404
56736/60000 [===========================>..] - ETA: 5s - loss: 0.1935 - categorical_accuracy: 0.9404
56768/60000 [===========================>..] - ETA: 5s - loss: 0.1934 - categorical_accuracy: 0.9404
56800/60000 [===========================>..] - ETA: 5s - loss: 0.1934 - categorical_accuracy: 0.9404
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1933 - categorical_accuracy: 0.9404
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1933 - categorical_accuracy: 0.9404
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1933 - categorical_accuracy: 0.9404
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1932 - categorical_accuracy: 0.9404
56960/60000 [===========================>..] - ETA: 5s - loss: 0.1932 - categorical_accuracy: 0.9404
56992/60000 [===========================>..] - ETA: 5s - loss: 0.1933 - categorical_accuracy: 0.9404
57024/60000 [===========================>..] - ETA: 5s - loss: 0.1932 - categorical_accuracy: 0.9404
57056/60000 [===========================>..] - ETA: 5s - loss: 0.1932 - categorical_accuracy: 0.9404
57088/60000 [===========================>..] - ETA: 5s - loss: 0.1931 - categorical_accuracy: 0.9405
57120/60000 [===========================>..] - ETA: 5s - loss: 0.1931 - categorical_accuracy: 0.9405
57152/60000 [===========================>..] - ETA: 5s - loss: 0.1931 - categorical_accuracy: 0.9405
57184/60000 [===========================>..] - ETA: 5s - loss: 0.1930 - categorical_accuracy: 0.9405
57216/60000 [===========================>..] - ETA: 5s - loss: 0.1930 - categorical_accuracy: 0.9405
57248/60000 [===========================>..] - ETA: 4s - loss: 0.1930 - categorical_accuracy: 0.9405
57280/60000 [===========================>..] - ETA: 4s - loss: 0.1930 - categorical_accuracy: 0.9405
57312/60000 [===========================>..] - ETA: 4s - loss: 0.1929 - categorical_accuracy: 0.9406
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1929 - categorical_accuracy: 0.9405
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1929 - categorical_accuracy: 0.9406
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1928 - categorical_accuracy: 0.9406
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1927 - categorical_accuracy: 0.9406
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1927 - categorical_accuracy: 0.9406
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1927 - categorical_accuracy: 0.9406
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1926 - categorical_accuracy: 0.9406
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1925 - categorical_accuracy: 0.9407
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1925 - categorical_accuracy: 0.9407
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1925 - categorical_accuracy: 0.9407
57664/60000 [===========================>..] - ETA: 4s - loss: 0.1924 - categorical_accuracy: 0.9407
57696/60000 [===========================>..] - ETA: 4s - loss: 0.1924 - categorical_accuracy: 0.9407
57728/60000 [===========================>..] - ETA: 4s - loss: 0.1923 - categorical_accuracy: 0.9407
57760/60000 [===========================>..] - ETA: 4s - loss: 0.1922 - categorical_accuracy: 0.9408
57792/60000 [===========================>..] - ETA: 4s - loss: 0.1923 - categorical_accuracy: 0.9408
57824/60000 [===========================>..] - ETA: 3s - loss: 0.1923 - categorical_accuracy: 0.9408
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1923 - categorical_accuracy: 0.9408
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1923 - categorical_accuracy: 0.9408
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1923 - categorical_accuracy: 0.9408
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1922 - categorical_accuracy: 0.9408
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1921 - categorical_accuracy: 0.9408
58016/60000 [============================>.] - ETA: 3s - loss: 0.1920 - categorical_accuracy: 0.9409
58048/60000 [============================>.] - ETA: 3s - loss: 0.1920 - categorical_accuracy: 0.9409
58080/60000 [============================>.] - ETA: 3s - loss: 0.1919 - categorical_accuracy: 0.9409
58112/60000 [============================>.] - ETA: 3s - loss: 0.1918 - categorical_accuracy: 0.9409
58144/60000 [============================>.] - ETA: 3s - loss: 0.1919 - categorical_accuracy: 0.9409
58176/60000 [============================>.] - ETA: 3s - loss: 0.1918 - categorical_accuracy: 0.9409
58208/60000 [============================>.] - ETA: 3s - loss: 0.1917 - categorical_accuracy: 0.9409
58240/60000 [============================>.] - ETA: 3s - loss: 0.1917 - categorical_accuracy: 0.9410
58272/60000 [============================>.] - ETA: 3s - loss: 0.1916 - categorical_accuracy: 0.9410
58304/60000 [============================>.] - ETA: 3s - loss: 0.1916 - categorical_accuracy: 0.9410
58336/60000 [============================>.] - ETA: 3s - loss: 0.1915 - categorical_accuracy: 0.9410
58368/60000 [============================>.] - ETA: 2s - loss: 0.1914 - categorical_accuracy: 0.9410
58400/60000 [============================>.] - ETA: 2s - loss: 0.1913 - categorical_accuracy: 0.9410
58432/60000 [============================>.] - ETA: 2s - loss: 0.1913 - categorical_accuracy: 0.9411
58464/60000 [============================>.] - ETA: 2s - loss: 0.1913 - categorical_accuracy: 0.9411
58496/60000 [============================>.] - ETA: 2s - loss: 0.1913 - categorical_accuracy: 0.9411
58528/60000 [============================>.] - ETA: 2s - loss: 0.1913 - categorical_accuracy: 0.9411
58560/60000 [============================>.] - ETA: 2s - loss: 0.1912 - categorical_accuracy: 0.9411
58592/60000 [============================>.] - ETA: 2s - loss: 0.1912 - categorical_accuracy: 0.9412
58624/60000 [============================>.] - ETA: 2s - loss: 0.1911 - categorical_accuracy: 0.9412
58656/60000 [============================>.] - ETA: 2s - loss: 0.1910 - categorical_accuracy: 0.9412
58688/60000 [============================>.] - ETA: 2s - loss: 0.1909 - categorical_accuracy: 0.9412
58720/60000 [============================>.] - ETA: 2s - loss: 0.1909 - categorical_accuracy: 0.9412
58752/60000 [============================>.] - ETA: 2s - loss: 0.1908 - categorical_accuracy: 0.9412
58784/60000 [============================>.] - ETA: 2s - loss: 0.1907 - categorical_accuracy: 0.9413
58816/60000 [============================>.] - ETA: 2s - loss: 0.1908 - categorical_accuracy: 0.9412
58848/60000 [============================>.] - ETA: 2s - loss: 0.1908 - categorical_accuracy: 0.9412
58880/60000 [============================>.] - ETA: 2s - loss: 0.1910 - categorical_accuracy: 0.9412
58912/60000 [============================>.] - ETA: 1s - loss: 0.1909 - categorical_accuracy: 0.9413
58944/60000 [============================>.] - ETA: 1s - loss: 0.1908 - categorical_accuracy: 0.9413
58976/60000 [============================>.] - ETA: 1s - loss: 0.1907 - categorical_accuracy: 0.9413
59008/60000 [============================>.] - ETA: 1s - loss: 0.1906 - categorical_accuracy: 0.9413
59040/60000 [============================>.] - ETA: 1s - loss: 0.1906 - categorical_accuracy: 0.9413
59072/60000 [============================>.] - ETA: 1s - loss: 0.1906 - categorical_accuracy: 0.9414
59104/60000 [============================>.] - ETA: 1s - loss: 0.1905 - categorical_accuracy: 0.9414
59136/60000 [============================>.] - ETA: 1s - loss: 0.1904 - categorical_accuracy: 0.9414
59168/60000 [============================>.] - ETA: 1s - loss: 0.1904 - categorical_accuracy: 0.9414
59200/60000 [============================>.] - ETA: 1s - loss: 0.1904 - categorical_accuracy: 0.9414
59232/60000 [============================>.] - ETA: 1s - loss: 0.1905 - categorical_accuracy: 0.9414
59264/60000 [============================>.] - ETA: 1s - loss: 0.1904 - categorical_accuracy: 0.9414
59296/60000 [============================>.] - ETA: 1s - loss: 0.1903 - categorical_accuracy: 0.9414
59328/60000 [============================>.] - ETA: 1s - loss: 0.1902 - categorical_accuracy: 0.9415
59360/60000 [============================>.] - ETA: 1s - loss: 0.1903 - categorical_accuracy: 0.9415
59392/60000 [============================>.] - ETA: 1s - loss: 0.1902 - categorical_accuracy: 0.9415
59424/60000 [============================>.] - ETA: 1s - loss: 0.1901 - categorical_accuracy: 0.9415
59456/60000 [============================>.] - ETA: 0s - loss: 0.1900 - categorical_accuracy: 0.9415
59488/60000 [============================>.] - ETA: 0s - loss: 0.1900 - categorical_accuracy: 0.9416
59520/60000 [============================>.] - ETA: 0s - loss: 0.1899 - categorical_accuracy: 0.9416
59552/60000 [============================>.] - ETA: 0s - loss: 0.1900 - categorical_accuracy: 0.9416
59584/60000 [============================>.] - ETA: 0s - loss: 0.1899 - categorical_accuracy: 0.9416
59616/60000 [============================>.] - ETA: 0s - loss: 0.1899 - categorical_accuracy: 0.9416
59648/60000 [============================>.] - ETA: 0s - loss: 0.1898 - categorical_accuracy: 0.9417
59680/60000 [============================>.] - ETA: 0s - loss: 0.1897 - categorical_accuracy: 0.9417
59712/60000 [============================>.] - ETA: 0s - loss: 0.1897 - categorical_accuracy: 0.9417
59744/60000 [============================>.] - ETA: 0s - loss: 0.1896 - categorical_accuracy: 0.9417
59776/60000 [============================>.] - ETA: 0s - loss: 0.1895 - categorical_accuracy: 0.9417
59808/60000 [============================>.] - ETA: 0s - loss: 0.1895 - categorical_accuracy: 0.9417
59840/60000 [============================>.] - ETA: 0s - loss: 0.1894 - categorical_accuracy: 0.9418
59872/60000 [============================>.] - ETA: 0s - loss: 0.1894 - categorical_accuracy: 0.9418
59904/60000 [============================>.] - ETA: 0s - loss: 0.1893 - categorical_accuracy: 0.9418
59936/60000 [============================>.] - ETA: 0s - loss: 0.1893 - categorical_accuracy: 0.9418
59968/60000 [============================>.] - ETA: 0s - loss: 0.1892 - categorical_accuracy: 0.9418
60000/60000 [==============================] - 113s 2ms/step - loss: 0.1891 - categorical_accuracy: 0.9419 - val_loss: 0.0516 - val_categorical_accuracy: 0.9832

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 17s
  192/10000 [..............................] - ETA: 5s 
  352/10000 [>.............................] - ETA: 4s
  512/10000 [>.............................] - ETA: 4s
  672/10000 [=>............................] - ETA: 3s
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
 4832/10000 [=============>................] - ETA: 1s
 4992/10000 [=============>................] - ETA: 1s
 5152/10000 [==============>...............] - ETA: 1s
 5312/10000 [==============>...............] - ETA: 1s
 5472/10000 [===============>..............] - ETA: 1s
 5632/10000 [===============>..............] - ETA: 1s
 5792/10000 [================>.............] - ETA: 1s
 5952/10000 [================>.............] - ETA: 1s
 6112/10000 [=================>............] - ETA: 1s
 6272/10000 [=================>............] - ETA: 1s
 6432/10000 [==================>...........] - ETA: 1s
 6592/10000 [==================>...........] - ETA: 1s
 6752/10000 [===================>..........] - ETA: 1s
 6912/10000 [===================>..........] - ETA: 1s
 7072/10000 [====================>.........] - ETA: 1s
 7232/10000 [====================>.........] - ETA: 1s
 7392/10000 [=====================>........] - ETA: 0s
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
10000/10000 [==============================] - 4s 360us/step
[[7.94123522e-09 2.53359089e-09 6.42053010e-07 ... 9.99998569e-01
  5.30095212e-09 5.92989295e-07]
 [3.58776515e-06 2.72030138e-06 9.99984741e-01 ... 5.56324977e-08
  1.04302637e-06 8.90337637e-10]
 [1.00569787e-06 9.99705255e-01 4.83359909e-05 ... 1.68642422e-04
  9.28105783e-06 1.77202537e-06]
 ...
 [5.51519896e-09 3.24574216e-07 3.22067848e-08 ... 6.86053409e-06
  8.67484005e-07 1.76397643e-05]
 [1.83499310e-07 4.33020347e-07 3.03411056e-08 ... 1.37420201e-07
  4.83671436e-04 1.84911084e-06]
 [7.74758882e-06 2.12846771e-07 5.35801701e-06 ... 1.38808876e-08
  1.24348946e-06 4.46349837e-08]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.051574891280476, 'accuracy_test:': 0.9832000136375427}

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
   68ba0e0..7474c00  master     -> origin/master
Updating 68ba0e0..7474c00
Fast-forward
 error_list/20200516/list_log_benchmark_20200516.md | 182 +++++-----
 error_list/20200516/list_log_testall_20200516.md   | 386 +++++++++------------
 2 files changed, 269 insertions(+), 299 deletions(-)
[master 8332382] ml_store
 1 file changed, 2044 insertions(+)
To github.com:arita37/mlmodels_store.git
   7474c00..8332382  master -> master





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
{'loss': 0.5916592851281166, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-16 20:32:50.393813: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
[master 54aafec] ml_store
 1 file changed, 234 insertions(+)
To github.com:arita37/mlmodels_store.git
   8332382..54aafec  master -> master





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
[master eadeb6e] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   54aafec..eadeb6e  master -> master





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
	Data preprocessing and feature engineering runtime = 0.25s ...
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
 40%|████      | 2/5 [00:21<00:32, 10.87s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.9094940446122824, 'learning_rate': 0.0061658309170420185, 'min_data_in_leaf': 29, 'num_leaves': 53} and reward: 0.3828
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\x1a\x93A0\x87QX\r\x00\x00\x00learning_rateq\x02G?yAW\xa2C\x03\xebX\x10\x00\x00\x00min_data_in_leafq\x03K\x1dX\n\x00\x00\x00num_leavesq\x04K5u.' and reward: 0.3828
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\x1a\x93A0\x87QX\r\x00\x00\x00learning_rateq\x02G?yAW\xa2C\x03\xebX\x10\x00\x00\x00min_data_in_leafq\x03K\x1dX\n\x00\x00\x00num_leavesq\x04K5u.' and reward: 0.3828
 60%|██████    | 3/5 [00:50<00:32, 16.38s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.9505887238143308, 'learning_rate': 0.016602722115870553, 'min_data_in_leaf': 14, 'num_leaves': 40} and reward: 0.3906
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeek9\x0b\x17S_X\r\x00\x00\x00learning_rateq\x02G?\x91\x00M\xd2\x0c\x88\xa1X\x10\x00\x00\x00min_data_in_leafq\x03K\x0eX\n\x00\x00\x00num_leavesq\x04K(u.' and reward: 0.3906
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeek9\x0b\x17S_X\r\x00\x00\x00learning_rateq\x02G?\x91\x00M\xd2\x0c\x88\xa1X\x10\x00\x00\x00min_data_in_leafq\x03K\x0eX\n\x00\x00\x00num_leavesq\x04K(u.' and reward: 0.3906
 80%|████████  | 4/5 [01:15<00:18, 18.81s/it] 80%|████████  | 4/5 [01:15<00:18, 18.86s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.8176248908421629, 'learning_rate': 0.013633530818772626, 'min_data_in_leaf': 24, 'num_leaves': 34} and reward: 0.389
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xea)\xfb\xac\xd2\x01OX\r\x00\x00\x00learning_rateq\x02G?\x8b\xeb\xe5\x87\xf7\x10IX\x10\x00\x00\x00min_data_in_leafq\x03K\x18X\n\x00\x00\x00num_leavesq\x04K"u.' and reward: 0.389
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xea)\xfb\xac\xd2\x01OX\r\x00\x00\x00learning_rateq\x02G?\x8b\xeb\xe5\x87\xf7\x10IX\x10\x00\x00\x00min_data_in_leafq\x03K\x18X\n\x00\x00\x00num_leavesq\x04K"u.' and reward: 0.389
Time for Gradient Boosting hyperparameter optimization: 96.90743327140808
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
 40%|████      | 2/5 [00:54<01:21, 27.03s/it] 40%|████      | 2/5 [00:54<01:21, 27.03s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.10282939337550231, 'embedding_size_factor': 1.2860560195436066, 'layers.choice': 0, 'learning_rate': 0.0013033312424268776, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 8.461800813505108e-12} and reward: 0.3808
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xbaS\x06\xf1\x9d\x84fX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf4\x93\xafz\x0c8:X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?UZ\x91C\xf9\x18\x0cX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\xa2\x9b\x92\x04\x04\xd4\xccu.' and reward: 0.3808
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xbaS\x06\xf1\x9d\x84fX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf4\x93\xafz\x0c8:X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?UZ\x91C\xf9\x18\x0cX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\xa2\x9b\x92\x04\x04\xd4\xccu.' and reward: 0.3808
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 109.00662302970886
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.75s of the -89.88s of remaining time.
Ensemble size: 49
Ensemble weights: 
[0.16326531 0.10204082 0.24489796 0.08163265 0.16326531 0.24489796]
	0.3982	 = Validation accuracy score
	1.55s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 211.5s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f4272ea5e10>

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
   eadeb6e..2b4bbbf  master     -> origin/master
Updating eadeb6e..2b4bbbf
Fast-forward
 error_list/20200516/list_log_jupyter_20200516.md | 1749 +++++++++++-----------
 1 file changed, 874 insertions(+), 875 deletions(-)
[master 465005e] ml_store
 1 file changed, 205 insertions(+)
To github.com:arita37/mlmodels_store.git
   2b4bbbf..465005e  master -> master





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
[master f1ff8b3] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   465005e..f1ff8b3  master -> master





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
100%|██████████| 10/10 [00:02<00:00,  3.66it/s, avg_epoch_loss=5.26]
INFO:root:Epoch[0] Elapsed time 2.731 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.264621
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.264620637893676 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f2018eef4e0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f2018eef4e0>

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
Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 98.42it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 1119.45947265625,
    "abs_error": 381.2309875488281,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 2.526017394035163,
    "sMAPE": 0.523471428968189,
    "MSIS": 101.04070061439711,
    "QuantileLoss[0.5]": 381.23099517822266,
    "Coverage[0.5]": 1.0,
    "RMSE": 33.45832441495315,
    "NRMSE": 0.7043857771569083,
    "ND": 0.6688262939453125,
    "wQuantileLoss[0.5]": 0.6688263073302152,
    "mean_wQuantileLoss": 0.6688263073302152,
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
100%|██████████| 10/10 [00:01<00:00,  7.27it/s, avg_epoch_loss=2.71e+3]
INFO:root:Epoch[0] Elapsed time 1.375 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=2713.411247
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 2713.4112467447917 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f1fec999cf8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f1fec999cf8>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 145.67it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
100%|██████████| 10/10 [00:01<00:00,  5.11it/s, avg_epoch_loss=5.21]
INFO:root:Epoch[0] Elapsed time 1.958 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.212549
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.212549018859863 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f1feca01860>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f1feca01860>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 147.93it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 244.89591471354166,
    "abs_error": 161.8299102783203,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.0722768651796168,
    "sMAPE": 0.2717786747439779,
    "MSIS": 42.89107056302584,
    "QuantileLoss[0.5]": 161.82989883422852,
    "Coverage[0.5]": 0.75,
    "RMSE": 15.649150606775489,
    "NRMSE": 0.32945580224790505,
    "ND": 0.2839121232952988,
    "wQuantileLoss[0.5]": 0.28391210321794474,
    "mean_wQuantileLoss": 0.28391210321794474,
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
 30%|███       | 3/10 [00:13<00:31,  4.47s/it, avg_epoch_loss=6.95] 60%|██████    | 6/10 [00:25<00:17,  4.32s/it, avg_epoch_loss=6.91] 90%|█████████ | 9/10 [00:37<00:04,  4.21s/it, avg_epoch_loss=6.89]100%|██████████| 10/10 [00:41<00:00,  4.12s/it, avg_epoch_loss=6.88]
INFO:root:Epoch[0] Elapsed time 41.209 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.879690
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.879690074920655 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f1fec080780>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f1fec080780>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 142.55it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 54709.354166666664,
    "abs_error": 2770.5927734375,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 18.357808693593856,
    "sMAPE": 1.420020158429592,
    "MSIS": 734.3123995089874,
    "QuantileLoss[0.5]": 2770.592758178711,
    "Coverage[0.5]": 1.0,
    "RMSE": 233.90030817993093,
    "NRMSE": 4.924217014314335,
    "ND": 4.860689076206141,
    "wQuantileLoss[0.5]": 4.860689049436335,
    "mean_wQuantileLoss": 4.860689049436335,
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
100%|██████████| 10/10 [00:00<00:00, 45.90it/s, avg_epoch_loss=5.06]
INFO:root:Epoch[0] Elapsed time 0.219 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.062227
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.062226867675781 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f1fd56e5f60>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f1fd56e5f60>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 140.32it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 378.4283854166667,
    "abs_error": 210.19497680664062,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.3927413691269965,
    "sMAPE": 0.33903607287085835,
    "MSIS": 55.70965961807046,
    "QuantileLoss[0.5]": 210.1949806213379,
    "Coverage[0.5]": 0.75,
    "RMSE": 19.45323585979121,
    "NRMSE": 0.4095418075745518,
    "ND": 0.3687631172046327,
    "wQuantileLoss[0.5]": 0.36876312389708404,
    "mean_wQuantileLoss": 0.36876312389708404,
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
100%|██████████| 10/10 [00:01<00:00,  7.84it/s, avg_epoch_loss=144]
INFO:root:Epoch[0] Elapsed time 1.276 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=144.412806
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 144.41280616614463 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f1fd581ef60>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f1fd581ef60>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 145.18it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
 10%|█         | 1/10 [02:09<19:21, 129.03s/it, avg_epoch_loss=0.488] 20%|██        | 2/10 [05:17<19:35, 146.97s/it, avg_epoch_loss=0.47]  30%|███       | 3/10 [08:30<18:45, 160.78s/it, avg_epoch_loss=0.453] 40%|████      | 4/10 [12:05<17:40, 176.79s/it, avg_epoch_loss=0.437] 50%|█████     | 5/10 [15:54<16:02, 192.50s/it, avg_epoch_loss=0.424] 60%|██████    | 6/10 [19:21<13:07, 196.82s/it, avg_epoch_loss=0.414] 70%|███████   | 7/10 [22:59<10:09, 203.29s/it, avg_epoch_loss=0.407] 80%|████████  | 8/10 [26:28<06:49, 204.97s/it, avg_epoch_loss=0.404] 90%|█████████ | 9/10 [29:43<03:22, 202.09s/it, avg_epoch_loss=0.401]100%|██████████| 10/10 [33:20<00:00, 206.56s/it, avg_epoch_loss=0.399]100%|██████████| 10/10 [33:20<00:00, 200.08s/it, avg_epoch_loss=0.399]
INFO:root:Epoch[0] Elapsed time 2000.765 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.398938
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.3989381194114685 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f1fd580fcc0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f1fd580fcc0>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 19.76it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
From github.com:arita37/mlmodels_store
   f1ff8b3..243417a  master     -> origin/master
Updating f1ff8b3..243417a
Fast-forward
 error_list/20200516/list_log_jupyter_20200516.md | 1749 +++++++++++-----------
 1 file changed, 875 insertions(+), 874 deletions(-)
[master 5279d1f] ml_store
 1 file changed, 506 insertions(+)
To github.com:arita37/mlmodels_store.git
   243417a..5279d1f  master -> master





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
{'roc_auc_score': 0.9642857142857143}

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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7f3b1b4300f0> 

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
[master d9b2114] ml_store
 1 file changed, 109 insertions(+)
To github.com:arita37/mlmodels_store.git
   5279d1f..d9b2114  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 9.83799588e-01 -4.07240024e-01  9.32721414e-01  1.60564992e-01
  -1.27861800e+00 -1.20149976e-01  1.99759555e-01  3.85602292e-01
   7.18290736e-01 -5.30119800e-01]
 [ 1.01195228e+00 -1.88141087e+00  1.70018815e+00  4.97269099e-01
  -9.17664624e-01  2.37332699e-01 -1.09033833e+00 -2.14444405e+00
  -3.69562425e-01  6.08783659e-01]
 [ 4.67397905e-01 -2.37875265e-01 -1.54491194e-01 -7.55662765e-01
  -5.47062239e-01  1.85143789e+00 -1.46405357e+00  2.09096677e-01
   1.55501599e+00 -9.24323185e-02]
 [ 9.26869810e-01  3.92334911e-01 -4.23478297e-01  4.48380651e-01
  -1.09230828e+00  1.12532350e+00 -9.48439656e-01  1.04053390e-01
   5.28003422e-01  1.00796648e+00]
 [ 9.67037267e-01  3.82715174e-01 -8.06184817e-01 -2.88997343e-01
   9.08526041e-01 -3.91816240e-01  1.62091229e+00  6.84001328e-01
  -3.53409983e-01 -2.51674208e-01]
 [ 1.24549398e+00 -7.22391905e-01  1.11813340e+00  1.09899633e+00
   1.00277655e+00 -9.01634490e-01 -5.32234021e-01 -8.22467189e-01
   7.21711292e-01  6.74396105e-01]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 1.01177337e+00  9.57467711e-02  7.31402517e-01  1.03345080e+00
  -1.42203164e+00 -1.46273275e-01 -1.74549518e-02 -8.57496825e-01
  -9.34181843e-01  9.54495667e-01]
 [ 1.39198128e+00 -1.90221025e-01 -5.37223024e-01 -4.48738033e-01
   7.04557071e-01 -6.72448039e-01 -7.01344426e-01 -5.57494722e-01
   9.39168744e-01  1.56263850e-01]
 [ 4.46895161e-01  3.86539145e-01  1.35010682e+00 -8.51455657e-01
   8.50637963e-01  1.00088142e+00 -1.16017010e+00 -3.84832249e-01
   1.45810824e+00 -3.31283170e-01]
 [ 8.76994650e-01  1.23225307e+00 -8.67787223e-01 -2.54179868e-01
   8.91891405e-01  1.39984394e+00 -8.77281519e-01 -7.81911683e-01
  -4.37508983e-01 -1.44087602e+00]
 [ 1.37661405e+00 -6.00225330e-01  7.25916853e-01 -3.79517516e-01
  -6.27546260e-01 -1.01480369e+00  9.66220863e-01  4.35986196e-01
  -6.87487393e-01  3.32107876e+00]
 [ 1.66752297e+00  1.22372221e+00 -4.59930104e-01 -5.93679025e-02
  -4.93856997e-01  1.44898940e+00 -1.18110317e+00 -4.77580855e-01
   2.59999942e-02 -7.90799954e-01]
 [ 9.36211246e-01  2.04377395e-01 -1.49419377e+00  6.12232523e-01
  -9.84377246e-01  7.44884536e-01  4.94341651e-01 -3.62812886e-02
  -8.32395348e-01 -4.46699203e-01]
 [ 4.73307772e-01 -9.73267585e-01 -2.28140691e-01  1.75167729e-01
  -1.01366961e+00 -5.34836927e-02  3.93787731e-01 -1.83061987e-01
  -2.21028902e-01  5.80330113e-01]
 [ 1.12062155e+00 -7.02920403e-01 -1.22957425e+00  7.25550518e-01
  -1.18013412e+00 -3.24204219e-01  1.10223673e+00  8.14343129e-01
   7.80469930e-01  1.10861676e+00]
 [ 1.06040861e+00  5.10307597e-01  5.01725109e-01 -9.15791849e-01
  -9.07318361e-01 -4.07252043e-01 -1.79612295e-01  9.84951672e-01
   1.07125243e+00 -5.93343754e-01]
 [ 1.09488485e+00 -6.96245395e-02 -1.16444148e-01  3.53870427e-01
  -1.44189096e+00 -1.86955017e-01  1.29118890e+00 -1.53236162e-01
  -2.43250851e+00 -2.27729800e+00]
 [ 4.41189807e-01  4.79852371e-01 -1.92003697e-01 -1.55269878e+00
  -1.88873982e+00  5.78464420e-01  3.98598388e-01 -9.61263599e-01
  -1.45832446e+00 -3.05376438e+00]
 [ 9.64572049e-01 -1.06793987e-01  1.12232832e+00  1.45142926e+00
   1.21828168e+00 -6.18036848e-01  4.38166347e-01 -2.03720123e+00
  -1.94258918e+00 -9.97019796e-01]
 [ 1.18468624e+00 -1.00016919e+00 -5.93843067e-01  1.04499441e+00
   9.65482331e-01  6.08514698e-01 -6.25342001e-01 -6.93286967e-02
  -1.08392067e-01 -3.43900709e-01]
 [ 8.88611457e-01  8.49586845e-01 -3.09114176e-02 -1.22154015e-01
  -1.14722826e+00 -6.80851574e-01 -3.26061306e-01 -1.06787658e+00
  -7.66793627e-02  3.55717262e-01]
 [ 5.63077902e-01 -1.17598267e+00 -1.74180344e-01  1.01012718e+00
   1.06796368e+00  9.20017933e-01 -1.68198840e-01 -1.95057341e-01
   8.05393424e-01  4.61164100e-01]
 [ 8.78643802e-01  1.03703898e+00 -4.77124206e-01  6.72619748e-01
  -1.04948638e+00  2.42887697e+00  5.24750492e-01  1.00568668e+00
   3.53567216e-01 -3.59901817e-02]
 [ 1.25704434e+00 -1.82391985e+00 -6.12406973e-01  1.16707517e+00
  -6.23732812e-01 -3.96687001e-02  8.16043684e-01  8.85825799e-01
   1.89861649e-01  3.93109245e-01]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7fb2718bbe48>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7fb28bc2d668> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 1.34728643 -0.36453805  0.08075099 -0.45971768 -0.8894876   1.70548352
   0.09499611  0.24050555 -0.9994265  -0.76780375]
 [ 1.17867274 -0.59980453 -0.6946936   1.12341216  1.17899425  0.30526704
   0.01335268  1.3887794  -0.66134424  0.6218035 ]
 [ 0.98042741  1.93752881 -0.23083974  0.36633201  1.10018476 -1.04458938
  -0.34498721  2.05117344  0.585662   -2.793085  ]
 [ 0.5630779  -1.17598267 -0.17418034  1.01012718  1.06796368  0.92001793
  -0.16819884 -0.19505734  0.80539342  0.4611641 ]
 [ 1.03967316 -0.73153098  0.36184732 -1.56573815  0.95928819  1.01382247
  -1.78791289 -2.22711263 -1.6993336  -0.42449279]
 [ 0.99785516 -0.6001388   0.45794708  0.14676526 -0.93355729  0.57180488
   0.57296273 -0.03681766  0.11236849 -0.01781755]
 [ 2.07582971 -1.40232915 -0.47918492  0.45112294  1.03436581 -0.6949209
  -0.4189379   0.5154138  -1.11487105 -1.95210529]
 [ 0.88861146  0.84958685 -0.03091142 -0.12215402 -1.14722826 -0.68085157
  -0.32606131 -1.06787658 -0.07667936  0.35571726]
 [ 1.24549398 -0.72239191  1.1181334   1.09899633  1.00277655 -0.90163449
  -0.53223402 -0.82246719  0.72171129  0.6743961 ]
 [ 0.9292506  -1.10657307 -1.95816909 -0.3592241  -1.21258781  0.5053819
   0.54264529  1.2179409  -1.94068096  0.67780757]
 [ 0.56998385 -0.53302033 -0.17545897 -1.42655542  0.60660431  1.76795995
  -0.11598519 -0.47537288  0.47761018 -0.93391466]
 [ 0.98379959 -0.40724002  0.93272141  0.16056499 -1.278618   -0.12014998
   0.19975956  0.38560229  0.71829074 -0.5301198 ]
 [ 0.55853873 -0.51634791 -0.51814555  0.3511169   0.82550695 -0.06877046
  -0.9520621  -1.34776494  1.47073986 -1.4614036 ]
 [ 0.92686981  0.39233491 -0.4234783   0.44838065 -1.09230828  1.1253235
  -0.94843966  0.10405339  0.52800342  1.00796648]
 [ 1.01195228 -1.88141087  1.70018815  0.4972691  -0.91766462  0.2373327
  -1.09033833 -2.14444405 -0.36956243  0.60878366]
 [ 1.4468218   0.80745592  1.49810818  0.31223869 -0.68243019 -0.19332164
   0.28807817 -2.07680202  0.94750117 -0.30097615]
 [ 1.18559003  0.08646441  1.23289919 -2.14246673  1.033341   -0.83016886
   0.36723181  0.45161595  1.10417433 -0.42285696]
 [ 1.32857949 -0.5632366  -1.06179676  2.39014596 -1.6845077   0.24542285
  -0.56914865  1.15259914 -0.22423577  0.13224778]
 [ 1.58463774  0.057121   -0.01771832 -0.79954749  1.32970299 -0.2915946
  -1.1077125  -0.25898285  0.1892932  -1.71939447]
 [ 0.62153099 -1.50957268 -0.10193204 -1.08071069 -1.13742855  0.725474
   0.7980638  -0.03917826 -0.22875417  0.74335654]
 [ 1.02242019  1.85300949  0.64435367  0.14225137  1.15080755  0.51350548
  -0.45994283  0.37245685 -0.1484898   0.37167029]
 [ 1.32720112 -0.16119832  0.6024509  -0.28638492 -0.5789623  -0.87088765
   1.37975819  0.50142959 -0.47861407 -0.89264667]
 [ 1.06702918 -0.42914228  0.35016716  1.20845633  0.75148062  1.1157018
  -0.4791571   0.84086156 -0.10288722  0.01716473]
 [ 0.78344054 -0.05118845  0.82458463 -0.72559712  0.9317172  -0.86776868
   3.03085711 -0.13597733 -0.79726979  0.65458015]
 [ 1.64661853 -1.52568032 -0.6069984   0.79502609  1.08480038 -0.37443832
   0.42952614  0.1340482   1.20205486  0.10622272]]
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
[[ 0.68188934 -1.15498263  1.22895559 -0.1776322   0.99854519 -1.51045638
  -0.27584606  1.01120706 -1.47656266  1.30970591]
 [ 0.77370361  1.27852808 -2.11416392 -0.44222928  1.06821044  0.32352735
  -2.50644065 -0.10999149  0.00854895 -0.41163916]
 [ 0.92686981  0.39233491 -0.4234783   0.44838065 -1.09230828  1.1253235
  -0.94843966  0.10405339  0.52800342  1.00796648]
 [ 0.89562312 -2.29820588 -0.01952256  1.45652739 -1.85064099  0.31663724
   0.11133727 -2.66412594 -0.42642862 -0.83998891]
 [ 0.81583612 -1.39169388  2.50598029  0.45021774 -0.88286982  0.62743708
  -1.19586151  0.75133724  0.14039544  1.91979229]
 [ 1.24549398 -0.72239191  1.1181334   1.09899633  1.00277655 -0.90163449
  -0.53223402 -0.82246719  0.72171129  0.6743961 ]
 [ 0.5630779  -1.17598267 -0.17418034  1.01012718  1.06796368  0.92001793
  -0.16819884 -0.19505734  0.80539342  0.4611641 ]
 [ 1.18468624 -1.00016919 -0.59384307  1.04499441  0.96548233  0.6085147
  -0.625342   -0.0693287  -0.10839207 -0.34390071]
 [ 1.58463774  0.057121   -0.01771832 -0.79954749  1.32970299 -0.2915946
  -1.1077125  -0.25898285  0.1892932  -1.71939447]
 [ 0.99785516 -0.6001388   0.45794708  0.14676526 -0.93355729  0.57180488
   0.57296273 -0.03681766  0.11236849 -0.01781755]
 [ 0.56998385 -0.53302033 -0.17545897 -1.42655542  0.60660431  1.76795995
  -0.11598519 -0.47537288  0.47761018 -0.93391466]
 [ 1.21619061 -0.01900052  0.86089124 -0.22676019 -1.36419132 -1.56450785
   1.63169151  0.93125568  0.94980882 -0.88018906]
 [ 1.16777676 -0.66575452 -1.23312074 -1.67419581  1.01313574  0.82502982
  -0.12046457 -0.49821356 -0.31098498 -1.18231813]
 [ 0.345716   -0.41302931 -0.46867382  1.83471763  0.77151441  0.56438286
   0.02186284  2.13782807 -0.785534    0.85328122]
 [ 0.6109426  -2.79099641 -1.33520272 -0.45611756 -0.94495995 -0.97989025
  -0.15699367  0.69257435 -0.47867236 -0.10646012]
 [ 1.25704434 -1.82391985 -0.61240697  1.16707517 -0.62373281 -0.0396687
   0.81604368  0.8858258   0.18986165  0.39310924]
 [ 1.34740825  0.73302323  0.83863475 -1.89881206 -0.54245992 -1.11711069
  -1.09715436 -0.50897228 -0.16648595 -1.03918232]
 [ 1.01177337  0.09574677  0.73140252  1.0334508  -1.42203164 -0.14627327
  -0.01745495 -0.85749682 -0.93418184  0.95449567]
 [ 1.02242019  1.85300949  0.64435367  0.14225137  1.15080755  0.51350548
  -0.45994283  0.37245685 -0.1484898   0.37167029]
 [ 1.03967316 -0.73153098  0.36184732 -1.56573815  0.95928819  1.01382247
  -1.78791289 -2.22711263 -1.6993336  -0.42449279]
 [ 0.70017571  0.55607351  0.08968641  1.69380911  0.88239331  0.19686978
  -0.56378873  0.16986926 -1.16400797 -0.6011568 ]
 [ 0.10593645 -0.73728963  0.65032321  0.16466507 -1.53556118  0.77817418
   0.05031709  0.30981676  1.05132077  0.6065484 ]
 [ 0.76170668 -1.48515645  1.30253554 -0.59246129 -1.64162479 -2.30490794
  -1.34869645 -0.03181717  0.11248774 -0.36261209]
 [ 0.44689516  0.38653915  1.35010682 -0.85145566  0.85063796  1.00088142
  -1.1601701  -0.38483225  1.45810824 -0.33128317]
 [ 1.18559003  0.08646441  1.23289919 -2.14246673  1.033341   -0.83016886
   0.36723181  0.45161595  1.10417433 -0.42285696]]
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
[master f05cb6c] ml_store
 1 file changed, 272 insertions(+)
To github.com:arita37/mlmodels_store.git
   d9b2114..f05cb6c  master -> master





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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @139800038846416
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @139800038846192
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @139800038844960
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @139800038844512
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @139800038844008
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @139800038843672

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
grad_step = 000000, loss = 0.759507
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.620523
grad_step = 000002, loss = 0.511739
grad_step = 000003, loss = 0.392546
grad_step = 000004, loss = 0.262004
grad_step = 000005, loss = 0.134054
grad_step = 000006, loss = 0.041496
grad_step = 000007, loss = 0.049646
grad_step = 000008, loss = 0.116534
grad_step = 000009, loss = 0.113930
grad_step = 000010, loss = 0.065568
grad_step = 000011, loss = 0.028118
grad_step = 000012, loss = 0.015509
grad_step = 000013, loss = 0.020805
grad_step = 000014, loss = 0.031587
grad_step = 000015, loss = 0.038977
grad_step = 000016, loss = 0.040005
grad_step = 000017, loss = 0.035192
grad_step = 000018, loss = 0.027050
grad_step = 000019, loss = 0.018794
grad_step = 000020, loss = 0.013069
grad_step = 000021, loss = 0.011433
grad_step = 000022, loss = 0.013343
grad_step = 000023, loss = 0.016093
grad_step = 000024, loss = 0.017037
grad_step = 000025, loss = 0.015447
grad_step = 000026, loss = 0.012230
grad_step = 000027, loss = 0.009353
grad_step = 000028, loss = 0.008116
grad_step = 000029, loss = 0.008482
grad_step = 000030, loss = 0.009590
grad_step = 000031, loss = 0.010448
grad_step = 000032, loss = 0.010456
grad_step = 000033, loss = 0.009555
grad_step = 000034, loss = 0.008137
grad_step = 000035, loss = 0.006816
grad_step = 000036, loss = 0.006116
grad_step = 000037, loss = 0.006219
grad_step = 000038, loss = 0.006779
grad_step = 000039, loss = 0.007244
grad_step = 000040, loss = 0.007289
grad_step = 000041, loss = 0.006898
grad_step = 000042, loss = 0.006361
grad_step = 000043, loss = 0.005991
grad_step = 000044, loss = 0.005917
grad_step = 000045, loss = 0.006056
grad_step = 000046, loss = 0.006215
grad_step = 000047, loss = 0.006232
grad_step = 000048, loss = 0.006056
grad_step = 000049, loss = 0.005762
grad_step = 000050, loss = 0.005490
grad_step = 000051, loss = 0.005359
grad_step = 000052, loss = 0.005394
grad_step = 000053, loss = 0.005512
grad_step = 000054, loss = 0.005586
grad_step = 000055, loss = 0.005535
grad_step = 000056, loss = 0.005375
grad_step = 000057, loss = 0.005196
grad_step = 000058, loss = 0.005088
grad_step = 000059, loss = 0.005080
grad_step = 000060, loss = 0.005128
grad_step = 000061, loss = 0.005158
grad_step = 000062, loss = 0.005126
grad_step = 000063, loss = 0.005039
grad_step = 000064, loss = 0.004943
grad_step = 000065, loss = 0.004882
grad_step = 000066, loss = 0.004866
grad_step = 000067, loss = 0.004871
grad_step = 000068, loss = 0.004857
grad_step = 000069, loss = 0.004806
grad_step = 000070, loss = 0.004730
grad_step = 000071, loss = 0.004662
grad_step = 000072, loss = 0.004626
grad_step = 000073, loss = 0.004614
grad_step = 000074, loss = 0.004597
grad_step = 000075, loss = 0.004557
grad_step = 000076, loss = 0.004498
grad_step = 000077, loss = 0.004438
grad_step = 000078, loss = 0.004390
grad_step = 000079, loss = 0.004352
grad_step = 000080, loss = 0.004315
grad_step = 000081, loss = 0.004270
grad_step = 000082, loss = 0.004214
grad_step = 000083, loss = 0.004154
grad_step = 000084, loss = 0.004096
grad_step = 000085, loss = 0.004040
grad_step = 000086, loss = 0.003981
grad_step = 000087, loss = 0.003916
grad_step = 000088, loss = 0.003848
grad_step = 000089, loss = 0.003779
grad_step = 000090, loss = 0.003711
grad_step = 000091, loss = 0.003636
grad_step = 000092, loss = 0.003558
grad_step = 000093, loss = 0.003477
grad_step = 000094, loss = 0.003391
grad_step = 000095, loss = 0.003302
grad_step = 000096, loss = 0.003212
grad_step = 000097, loss = 0.003117
grad_step = 000098, loss = 0.003016
grad_step = 000099, loss = 0.002919
grad_step = 000100, loss = 0.002824
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002722
grad_step = 000102, loss = 0.002620
grad_step = 000103, loss = 0.002520
grad_step = 000104, loss = 0.002417
grad_step = 000105, loss = 0.002318
grad_step = 000106, loss = 0.002222
grad_step = 000107, loss = 0.002126
grad_step = 000108, loss = 0.002036
grad_step = 000109, loss = 0.001950
grad_step = 000110, loss = 0.001871
grad_step = 000111, loss = 0.001796
grad_step = 000112, loss = 0.001728
grad_step = 000113, loss = 0.001665
grad_step = 000114, loss = 0.001610
grad_step = 000115, loss = 0.001561
grad_step = 000116, loss = 0.001518
grad_step = 000117, loss = 0.001481
grad_step = 000118, loss = 0.001446
grad_step = 000119, loss = 0.001417
grad_step = 000120, loss = 0.001395
grad_step = 000121, loss = 0.001379
grad_step = 000122, loss = 0.001364
grad_step = 000123, loss = 0.001352
grad_step = 000124, loss = 0.001338
grad_step = 000125, loss = 0.001322
grad_step = 000126, loss = 0.001297
grad_step = 000127, loss = 0.001268
grad_step = 000128, loss = 0.001247
grad_step = 000129, loss = 0.001225
grad_step = 000130, loss = 0.001198
grad_step = 000131, loss = 0.001180
grad_step = 000132, loss = 0.001160
grad_step = 000133, loss = 0.001138
grad_step = 000134, loss = 0.001120
grad_step = 000135, loss = 0.001103
grad_step = 000136, loss = 0.001085
grad_step = 000137, loss = 0.001067
grad_step = 000138, loss = 0.001051
grad_step = 000139, loss = 0.001035
grad_step = 000140, loss = 0.001020
grad_step = 000141, loss = 0.001006
grad_step = 000142, loss = 0.000994
grad_step = 000143, loss = 0.000986
grad_step = 000144, loss = 0.000979
grad_step = 000145, loss = 0.000962
grad_step = 000146, loss = 0.000943
grad_step = 000147, loss = 0.000931
grad_step = 000148, loss = 0.000918
grad_step = 000149, loss = 0.000909
grad_step = 000150, loss = 0.000899
grad_step = 000151, loss = 0.000882
grad_step = 000152, loss = 0.000863
grad_step = 000153, loss = 0.000852
grad_step = 000154, loss = 0.000850
grad_step = 000155, loss = 0.000835
grad_step = 000156, loss = 0.000821
grad_step = 000157, loss = 0.000811
grad_step = 000158, loss = 0.000794
grad_step = 000159, loss = 0.000782
grad_step = 000160, loss = 0.000774
grad_step = 000161, loss = 0.000767
grad_step = 000162, loss = 0.000765
grad_step = 000163, loss = 0.000785
grad_step = 000164, loss = 0.000774
grad_step = 000165, loss = 0.000751
grad_step = 000166, loss = 0.000713
grad_step = 000167, loss = 0.000718
grad_step = 000168, loss = 0.000742
grad_step = 000169, loss = 0.000706
grad_step = 000170, loss = 0.000677
grad_step = 000171, loss = 0.000678
grad_step = 000172, loss = 0.000679
grad_step = 000173, loss = 0.000667
grad_step = 000174, loss = 0.000644
grad_step = 000175, loss = 0.000646
grad_step = 000176, loss = 0.000656
grad_step = 000177, loss = 0.000636
grad_step = 000178, loss = 0.000616
grad_step = 000179, loss = 0.000613
grad_step = 000180, loss = 0.000612
grad_step = 000181, loss = 0.000605
grad_step = 000182, loss = 0.000591
grad_step = 000183, loss = 0.000582
grad_step = 000184, loss = 0.000582
grad_step = 000185, loss = 0.000581
grad_step = 000186, loss = 0.000574
grad_step = 000187, loss = 0.000562
grad_step = 000188, loss = 0.000553
grad_step = 000189, loss = 0.000550
grad_step = 000190, loss = 0.000548
grad_step = 000191, loss = 0.000547
grad_step = 000192, loss = 0.000543
grad_step = 000193, loss = 0.000537
grad_step = 000194, loss = 0.000529
grad_step = 000195, loss = 0.000523
grad_step = 000196, loss = 0.000520
grad_step = 000197, loss = 0.000518
grad_step = 000198, loss = 0.000517
grad_step = 000199, loss = 0.000516
grad_step = 000200, loss = 0.000517
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.000519
grad_step = 000202, loss = 0.000522
grad_step = 000203, loss = 0.000522
grad_step = 000204, loss = 0.000521
grad_step = 000205, loss = 0.000513
grad_step = 000206, loss = 0.000504
grad_step = 000207, loss = 0.000493
grad_step = 000208, loss = 0.000487
grad_step = 000209, loss = 0.000484
grad_step = 000210, loss = 0.000486
grad_step = 000211, loss = 0.000488
grad_step = 000212, loss = 0.000489
grad_step = 000213, loss = 0.000489
grad_step = 000214, loss = 0.000486
grad_step = 000215, loss = 0.000480
grad_step = 000216, loss = 0.000472
grad_step = 000217, loss = 0.000466
grad_step = 000218, loss = 0.000462
grad_step = 000219, loss = 0.000460
grad_step = 000220, loss = 0.000459
grad_step = 000221, loss = 0.000457
grad_step = 000222, loss = 0.000457
grad_step = 000223, loss = 0.000458
grad_step = 000224, loss = 0.000460
grad_step = 000225, loss = 0.000460
grad_step = 000226, loss = 0.000460
grad_step = 000227, loss = 0.000459
grad_step = 000228, loss = 0.000458
grad_step = 000229, loss = 0.000454
grad_step = 000230, loss = 0.000448
grad_step = 000231, loss = 0.000440
grad_step = 000232, loss = 0.000435
grad_step = 000233, loss = 0.000431
grad_step = 000234, loss = 0.000429
grad_step = 000235, loss = 0.000428
grad_step = 000236, loss = 0.000428
grad_step = 000237, loss = 0.000429
grad_step = 000238, loss = 0.000431
grad_step = 000239, loss = 0.000434
grad_step = 000240, loss = 0.000436
grad_step = 000241, loss = 0.000438
grad_step = 000242, loss = 0.000438
grad_step = 000243, loss = 0.000436
grad_step = 000244, loss = 0.000429
grad_step = 000245, loss = 0.000420
grad_step = 000246, loss = 0.000412
grad_step = 000247, loss = 0.000407
grad_step = 000248, loss = 0.000406
grad_step = 000249, loss = 0.000408
grad_step = 000250, loss = 0.000411
grad_step = 000251, loss = 0.000413
grad_step = 000252, loss = 0.000414
grad_step = 000253, loss = 0.000414
grad_step = 000254, loss = 0.000412
grad_step = 000255, loss = 0.000407
grad_step = 000256, loss = 0.000402
grad_step = 000257, loss = 0.000397
grad_step = 000258, loss = 0.000393
grad_step = 000259, loss = 0.000391
grad_step = 000260, loss = 0.000389
grad_step = 000261, loss = 0.000388
grad_step = 000262, loss = 0.000387
grad_step = 000263, loss = 0.000387
grad_step = 000264, loss = 0.000388
grad_step = 000265, loss = 0.000389
grad_step = 000266, loss = 0.000391
grad_step = 000267, loss = 0.000393
grad_step = 000268, loss = 0.000395
grad_step = 000269, loss = 0.000397
grad_step = 000270, loss = 0.000399
grad_step = 000271, loss = 0.000401
grad_step = 000272, loss = 0.000398
grad_step = 000273, loss = 0.000392
grad_step = 000274, loss = 0.000382
grad_step = 000275, loss = 0.000373
grad_step = 000276, loss = 0.000367
grad_step = 000277, loss = 0.000367
grad_step = 000278, loss = 0.000370
grad_step = 000279, loss = 0.000374
grad_step = 000280, loss = 0.000378
grad_step = 000281, loss = 0.000377
grad_step = 000282, loss = 0.000375
grad_step = 000283, loss = 0.000372
grad_step = 000284, loss = 0.000371
grad_step = 000285, loss = 0.000373
grad_step = 000286, loss = 0.000375
grad_step = 000287, loss = 0.000373
grad_step = 000288, loss = 0.000366
grad_step = 000289, loss = 0.000357
grad_step = 000290, loss = 0.000351
grad_step = 000291, loss = 0.000351
grad_step = 000292, loss = 0.000354
grad_step = 000293, loss = 0.000359
grad_step = 000294, loss = 0.000362
grad_step = 000295, loss = 0.000361
grad_step = 000296, loss = 0.000358
grad_step = 000297, loss = 0.000353
grad_step = 000298, loss = 0.000349
grad_step = 000299, loss = 0.000346
grad_step = 000300, loss = 0.000344
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000345
grad_step = 000302, loss = 0.000347
grad_step = 000303, loss = 0.000349
grad_step = 000304, loss = 0.000350
grad_step = 000305, loss = 0.000351
grad_step = 000306, loss = 0.000350
grad_step = 000307, loss = 0.000350
grad_step = 000308, loss = 0.000348
grad_step = 000309, loss = 0.000345
grad_step = 000310, loss = 0.000341
grad_step = 000311, loss = 0.000339
grad_step = 000312, loss = 0.000340
grad_step = 000313, loss = 0.000346
grad_step = 000314, loss = 0.000356
grad_step = 000315, loss = 0.000371
grad_step = 000316, loss = 0.000383
grad_step = 000317, loss = 0.000388
grad_step = 000318, loss = 0.000378
grad_step = 000319, loss = 0.000355
grad_step = 000320, loss = 0.000335
grad_step = 000321, loss = 0.000327
grad_step = 000322, loss = 0.000334
grad_step = 000323, loss = 0.000343
grad_step = 000324, loss = 0.000345
grad_step = 000325, loss = 0.000334
grad_step = 000326, loss = 0.000322
grad_step = 000327, loss = 0.000319
grad_step = 000328, loss = 0.000325
grad_step = 000329, loss = 0.000331
grad_step = 000330, loss = 0.000329
grad_step = 000331, loss = 0.000322
grad_step = 000332, loss = 0.000314
grad_step = 000333, loss = 0.000311
grad_step = 000334, loss = 0.000314
grad_step = 000335, loss = 0.000318
grad_step = 000336, loss = 0.000319
grad_step = 000337, loss = 0.000315
grad_step = 000338, loss = 0.000312
grad_step = 000339, loss = 0.000310
grad_step = 000340, loss = 0.000313
grad_step = 000341, loss = 0.000316
grad_step = 000342, loss = 0.000319
grad_step = 000343, loss = 0.000320
grad_step = 000344, loss = 0.000322
grad_step = 000345, loss = 0.000321
grad_step = 000346, loss = 0.000321
grad_step = 000347, loss = 0.000318
grad_step = 000348, loss = 0.000321
grad_step = 000349, loss = 0.000323
grad_step = 000350, loss = 0.000329
grad_step = 000351, loss = 0.000325
grad_step = 000352, loss = 0.000319
grad_step = 000353, loss = 0.000305
grad_step = 000354, loss = 0.000295
grad_step = 000355, loss = 0.000292
grad_step = 000356, loss = 0.000296
grad_step = 000357, loss = 0.000304
grad_step = 000358, loss = 0.000309
grad_step = 000359, loss = 0.000311
grad_step = 000360, loss = 0.000304
grad_step = 000361, loss = 0.000296
grad_step = 000362, loss = 0.000289
grad_step = 000363, loss = 0.000287
grad_step = 000364, loss = 0.000289
grad_step = 000365, loss = 0.000292
grad_step = 000366, loss = 0.000294
grad_step = 000367, loss = 0.000292
grad_step = 000368, loss = 0.000287
grad_step = 000369, loss = 0.000282
grad_step = 000370, loss = 0.000279
grad_step = 000371, loss = 0.000277
grad_step = 000372, loss = 0.000277
grad_step = 000373, loss = 0.000279
grad_step = 000374, loss = 0.000280
grad_step = 000375, loss = 0.000281
grad_step = 000376, loss = 0.000281
grad_step = 000377, loss = 0.000281
grad_step = 000378, loss = 0.000282
grad_step = 000379, loss = 0.000284
grad_step = 000380, loss = 0.000295
grad_step = 000381, loss = 0.000318
grad_step = 000382, loss = 0.000382
grad_step = 000383, loss = 0.000451
grad_step = 000384, loss = 0.000563
grad_step = 000385, loss = 0.000445
grad_step = 000386, loss = 0.000317
grad_step = 000387, loss = 0.000294
grad_step = 000388, loss = 0.000378
grad_step = 000389, loss = 0.000381
grad_step = 000390, loss = 0.000274
grad_step = 000391, loss = 0.000325
grad_step = 000392, loss = 0.000382
grad_step = 000393, loss = 0.000284
grad_step = 000394, loss = 0.000304
grad_step = 000395, loss = 0.000356
grad_step = 000396, loss = 0.000283
grad_step = 000397, loss = 0.000295
grad_step = 000398, loss = 0.000331
grad_step = 000399, loss = 0.000280
grad_step = 000400, loss = 0.000286
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000312
grad_step = 000402, loss = 0.000277
grad_step = 000403, loss = 0.000277
grad_step = 000404, loss = 0.000298
grad_step = 000405, loss = 0.000271
grad_step = 000406, loss = 0.000268
grad_step = 000407, loss = 0.000287
grad_step = 000408, loss = 0.000264
grad_step = 000409, loss = 0.000263
grad_step = 000410, loss = 0.000278
grad_step = 000411, loss = 0.000257
grad_step = 000412, loss = 0.000260
grad_step = 000413, loss = 0.000269
grad_step = 000414, loss = 0.000253
grad_step = 000415, loss = 0.000257
grad_step = 000416, loss = 0.000261
grad_step = 000417, loss = 0.000251
grad_step = 000418, loss = 0.000253
grad_step = 000419, loss = 0.000256
grad_step = 000420, loss = 0.000252
grad_step = 000421, loss = 0.000250
grad_step = 000422, loss = 0.000254
grad_step = 000423, loss = 0.000256
grad_step = 000424, loss = 0.000258
grad_step = 000425, loss = 0.000272
grad_step = 000426, loss = 0.000295
grad_step = 000427, loss = 0.000312
grad_step = 000428, loss = 0.000347
grad_step = 000429, loss = 0.000331
grad_step = 000430, loss = 0.000292
grad_step = 000431, loss = 0.000249
grad_step = 000432, loss = 0.000251
grad_step = 000433, loss = 0.000276
grad_step = 000434, loss = 0.000282
grad_step = 000435, loss = 0.000266
grad_step = 000436, loss = 0.000243
grad_step = 000437, loss = 0.000244
grad_step = 000438, loss = 0.000261
grad_step = 000439, loss = 0.000262
grad_step = 000440, loss = 0.000246
grad_step = 000441, loss = 0.000235
grad_step = 000442, loss = 0.000243
grad_step = 000443, loss = 0.000253
grad_step = 000444, loss = 0.000246
grad_step = 000445, loss = 0.000236
grad_step = 000446, loss = 0.000234
grad_step = 000447, loss = 0.000239
grad_step = 000448, loss = 0.000241
grad_step = 000449, loss = 0.000237
grad_step = 000450, loss = 0.000233
grad_step = 000451, loss = 0.000231
grad_step = 000452, loss = 0.000232
grad_step = 000453, loss = 0.000234
grad_step = 000454, loss = 0.000234
grad_step = 000455, loss = 0.000231
grad_step = 000456, loss = 0.000227
grad_step = 000457, loss = 0.000226
grad_step = 000458, loss = 0.000228
grad_step = 000459, loss = 0.000229
grad_step = 000460, loss = 0.000227
grad_step = 000461, loss = 0.000224
grad_step = 000462, loss = 0.000223
grad_step = 000463, loss = 0.000223
grad_step = 000464, loss = 0.000224
grad_step = 000465, loss = 0.000224
grad_step = 000466, loss = 0.000223
grad_step = 000467, loss = 0.000221
grad_step = 000468, loss = 0.000220
grad_step = 000469, loss = 0.000219
grad_step = 000470, loss = 0.000219
grad_step = 000471, loss = 0.000219
grad_step = 000472, loss = 0.000219
grad_step = 000473, loss = 0.000219
grad_step = 000474, loss = 0.000218
grad_step = 000475, loss = 0.000217
grad_step = 000476, loss = 0.000216
grad_step = 000477, loss = 0.000215
grad_step = 000478, loss = 0.000214
grad_step = 000479, loss = 0.000214
grad_step = 000480, loss = 0.000213
grad_step = 000481, loss = 0.000213
grad_step = 000482, loss = 0.000213
grad_step = 000483, loss = 0.000213
grad_step = 000484, loss = 0.000213
grad_step = 000485, loss = 0.000214
grad_step = 000486, loss = 0.000215
grad_step = 000487, loss = 0.000216
grad_step = 000488, loss = 0.000216
grad_step = 000489, loss = 0.000217
grad_step = 000490, loss = 0.000216
grad_step = 000491, loss = 0.000216
grad_step = 000492, loss = 0.000214
grad_step = 000493, loss = 0.000213
grad_step = 000494, loss = 0.000211
grad_step = 000495, loss = 0.000209
grad_step = 000496, loss = 0.000207
grad_step = 000497, loss = 0.000205
grad_step = 000498, loss = 0.000204
grad_step = 000499, loss = 0.000203
grad_step = 000500, loss = 0.000203
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000203
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
[[0.86932784 0.8463576  0.9332684  0.9424215  0.9975705 ]
 [0.8438508  0.92739904 0.9290447  0.9909675  0.9857776 ]
 [0.897816   0.9137627  1.0082173  0.9713358  0.9365926 ]
 [0.93239087 1.0026234  0.9937619  0.9422736  0.90711045]
 [0.9841754  0.9847388  0.95595473 0.9116502  0.8550742 ]
 [0.9877639  0.9346089  0.9219391  0.8546271  0.849853  ]
 [0.93650764 0.8957352  0.8602542  0.846938   0.7999213 ]
 [0.8964084  0.83613163 0.8513592  0.8114299  0.8399083 ]
 [0.8276229  0.83924145 0.8066352  0.84236616 0.85598874]
 [0.83789897 0.8156443  0.83558816 0.83235097 0.8222074 ]
 [0.8014699  0.8254368  0.85338753 0.8399474  0.90114105]
 [0.8076166  0.841041   0.8140585  0.912982   0.937359  ]
 [0.86200655 0.842332   0.9325524  0.9463792  1.0015873 ]
 [0.84107333 0.93684316 0.93581915 0.99315906 0.97363687]
 [0.90939915 0.9301044  1.0093725  0.9655128  0.9287571 ]
 [0.94080794 1.0138448  0.98072267 0.929903   0.89495313]
 [0.9867939  0.985905   0.9432267  0.8947922  0.83431077]
 [0.9748037  0.9301293  0.89749265 0.8345185  0.83558536]
 [0.92698824 0.8857962  0.83294445 0.82827336 0.7916552 ]
 [0.9008765  0.8344965  0.8321459  0.8027008  0.8346899 ]
 [0.838456   0.84580654 0.79605544 0.8370532  0.8581039 ]
 [0.85148895 0.8254515  0.8286804  0.83343947 0.81610066]
 [0.81368077 0.8313141  0.8481583  0.8364296  0.8996143 ]
 [0.8129063  0.8530013  0.8077017  0.9057207  0.9305707 ]
 [0.872007   0.8522748  0.92879677 0.9399033  0.9947413 ]
 [0.8498473  0.93000686 0.92636704 0.99618256 0.9962594 ]
 [0.90478903 0.92212176 1.0095599  0.97953844 0.94853103]
 [0.94484144 1.0153131  1.0065423  0.9501105  0.9171455 ]
 [0.9914189  0.99699795 0.96511185 0.9150635  0.85418856]
 [0.99349976 0.9482275  0.9276172  0.8579471  0.85019016]
 [0.9405131  0.90196353 0.8600852  0.84598273 0.8006231 ]]

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
From github.com:arita37/mlmodels_store
   f05cb6c..202032b  master     -> origin/master
Updating f05cb6c..202032b
Fast-forward
 deps.txt                                           |   6 +-
 ...-10_76b7a81be9b27c2e92c4951280c0a8da664b997c.py | 632 +++++++++++++++++++++
 2 files changed, 637 insertions(+), 1 deletion(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-16-21-10_76b7a81be9b27c2e92c4951280c0a8da664b997c.py
[master 67bd9e3] ml_store
 1 file changed, 1130 insertions(+)
To github.com:arita37/mlmodels_store.git
   202032b..67bd9e3  master -> master





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
[master 9e8bcbb] ml_store
 2 files changed, 39 insertions(+), 5 deletions(-)
To github.com:arita37/mlmodels_store.git
   67bd9e3..9e8bcbb  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//matchzoo_models.py 

  #### Loading params   ############################################## 

  {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'dataset_pars': {'data_pack': '', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'resample': True, 'sort': False, 'callbacks': 'PADDING'}, 'dataloader': '', 'dataloader_pars': {'device': 'cpu', 'dataset': 'None', 'stage': 'train', 'callback': 'PADDING'}, 'preprocess': {'train': {'transform': True, 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'stage': 'train', 'resample': True, 'sort': False, 'dataloader_callback': 'PADDING'}, 'test': {'transform': True, 'batch_size': 20, 'stage': 'dev', 'dataloader_callback': 'PADDING'}}} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Loading dataset   ############################################# 

  #### Model init   ################################################## 
  0%|          | 0/231508 [00:00<?, ?B/s]100%|██████████| 231508/231508 [00:00<00:00, 10135855.22B/s]
  0%|          | 0/433 [00:00<?, ?B/s]100%|██████████| 433/433 [00:00<00:00, 408670.93B/s]
  0%|          | 0/440473133 [00:00<?, ?B/s]  1%|          | 3705856/440473133 [00:00<00:11, 37056828.33B/s]  2%|▏         | 8632320/440473133 [00:00<00:10, 40032448.18B/s]  3%|▎         | 13209600/440473133 [00:00<00:10, 41596557.04B/s]  4%|▍         | 17185792/440473133 [00:00<00:10, 41028462.32B/s]  5%|▍         | 21875712/440473133 [00:00<00:09, 42626354.09B/s]  6%|▌         | 26793984/440473133 [00:00<00:09, 44401973.70B/s]  7%|▋         | 31671296/440473133 [00:00<00:08, 45627140.39B/s]  8%|▊         | 36115456/440473133 [00:00<00:08, 45263287.00B/s]  9%|▉         | 41006080/440473133 [00:00<00:08, 46296863.21B/s] 10%|█         | 45705216/440473133 [00:01<00:08, 46489930.59B/s] 11%|█▏        | 50270208/440473133 [00:01<00:08, 45656767.01B/s] 13%|█▎        | 55114752/440473133 [00:01<00:08, 46458505.45B/s] 14%|█▎        | 60112896/440473133 [00:01<00:08, 47457678.19B/s] 15%|█▍        | 65017856/440473133 [00:01<00:07, 47920728.16B/s] 16%|█▌        | 69797888/440473133 [00:01<00:07, 47315471.76B/s] 17%|█▋        | 74522624/440473133 [00:01<00:07, 47289850.82B/s] 18%|█▊        | 79443968/440473133 [00:01<00:07, 47846904.75B/s] 19%|█▉        | 84473856/440473133 [00:01<00:07, 48556098.31B/s] 20%|██        | 89520128/440473133 [00:01<00:07, 49110335.74B/s] 21%|██▏       | 94434304/440473133 [00:02<00:07, 47335300.18B/s] 23%|██▎       | 99183616/440473133 [00:02<00:07, 46427926.37B/s] 24%|██▎       | 104284160/440473133 [00:02<00:07, 47710524.69B/s] 25%|██▍       | 109074432/440473133 [00:02<00:07, 46718663.43B/s] 26%|██▌       | 113764352/440473133 [00:02<00:07, 45507465.39B/s] 27%|██▋       | 118874112/440473133 [00:02<00:06, 47051576.04B/s] 28%|██▊       | 123971584/440473133 [00:02<00:06, 48161121.11B/s] 29%|██▉       | 128898048/440473133 [00:02<00:06, 48486486.54B/s] 30%|███       | 133866496/440473133 [00:02<00:06, 48834637.73B/s] 32%|███▏      | 138764288/440473133 [00:02<00:06, 48679110.09B/s] 33%|███▎      | 143685632/440473133 [00:03<00:06, 48835184.94B/s] 34%|███▎      | 148576256/440473133 [00:03<00:06, 47566196.51B/s] 35%|███▍      | 153345024/440473133 [00:03<00:06, 45739316.10B/s] 36%|███▌      | 158350336/440473133 [00:03<00:06, 46952346.60B/s] 37%|███▋      | 163230720/440473133 [00:03<00:05, 47491862.69B/s] 38%|███▊      | 168130560/440473133 [00:03<00:05, 47931571.01B/s] 39%|███▉      | 173033472/440473133 [00:03<00:05, 48253650.66B/s] 40%|████      | 177869824/440473133 [00:03<00:05, 46321260.03B/s] 41%|████▏     | 182525952/440473133 [00:03<00:05, 46275316.70B/s] 43%|████▎     | 187309056/440473133 [00:03<00:05, 46730075.03B/s] 44%|████▎     | 192022528/440473133 [00:04<00:05, 46845654.83B/s] 45%|████▍     | 196929536/440473133 [00:04<00:05, 47485545.21B/s] 46%|████▌     | 201896960/440473133 [00:04<00:04, 48117776.62B/s] 47%|████▋     | 206716928/440473133 [00:04<00:04, 47022112.40B/s] 48%|████▊     | 211430400/440473133 [00:04<00:05, 45768879.81B/s] 49%|████▉     | 216386560/440473133 [00:04<00:04, 46843091.07B/s] 50%|█████     | 221367296/440473133 [00:04<00:04, 47692311.75B/s] 51%|█████▏    | 226314240/440473133 [00:04<00:04, 48206433.61B/s] 52%|█████▏    | 231233536/440473133 [00:04<00:04, 48495284.51B/s] 54%|█████▎    | 236092416/440473133 [00:05<00:04, 48492210.65B/s] 55%|█████▍    | 241085440/440473133 [00:05<00:04, 48914825.65B/s] 56%|█████▌    | 246207488/440473133 [00:05<00:03, 49582965.70B/s] 57%|█████▋    | 251250688/440473133 [00:05<00:03, 49832242.95B/s] 58%|█████▊    | 256411648/440473133 [00:05<00:03, 50345695.28B/s] 59%|█████▉    | 261450752/440473133 [00:05<00:03, 50121591.20B/s] 61%|██████    | 266586112/440473133 [00:05<00:03, 50484849.63B/s] 62%|██████▏   | 271681536/440473133 [00:05<00:03, 50619266.91B/s] 63%|██████▎   | 276746240/440473133 [00:05<00:03, 49996363.26B/s] 64%|██████▍   | 281749504/440473133 [00:05<00:03, 49586158.88B/s] 65%|██████▌   | 286800896/440473133 [00:06<00:03, 49859495.09B/s] 66%|██████▌   | 291789824/440473133 [00:06<00:03, 48663988.85B/s] 67%|██████▋   | 296665088/440473133 [00:06<00:03, 47339747.65B/s] 68%|██████▊   | 301413376/440473133 [00:06<00:02, 46965669.44B/s] 70%|██████▉   | 306306048/440473133 [00:06<00:02, 47534912.14B/s] 71%|███████   | 311411712/440473133 [00:06<00:02, 48538098.83B/s] 72%|███████▏  | 316321792/440473133 [00:06<00:02, 48704527.45B/s] 73%|███████▎  | 321270784/440473133 [00:06<00:02, 48936975.31B/s] 74%|███████▍  | 326170624/440473133 [00:06<00:02, 48949208.99B/s] 75%|███████▌  | 331070464/440473133 [00:06<00:02, 47148475.62B/s] 76%|███████▌  | 335803392/440473133 [00:07<00:02, 46651954.89B/s] 77%|███████▋  | 340568064/440473133 [00:07<00:02, 46930893.97B/s] 78%|███████▊  | 345631744/440473133 [00:07<00:01, 47982329.95B/s] 80%|███████▉  | 350697472/440473133 [00:07<00:01, 48751902.84B/s] 81%|████████  | 355739648/440473133 [00:07<00:01, 49235159.73B/s] 82%|████████▏ | 360673280/440473133 [00:07<00:01, 49232073.49B/s] 83%|████████▎ | 365603840/440473133 [00:07<00:01, 49042089.31B/s] 84%|████████▍ | 370536448/440473133 [00:07<00:01, 49125416.77B/s] 85%|████████▌ | 375452672/440473133 [00:07<00:01, 49028134.69B/s] 86%|████████▋ | 380358656/440473133 [00:07<00:01, 48299761.77B/s] 87%|████████▋ | 385388544/440473133 [00:08<00:01, 48872228.64B/s] 89%|████████▊ | 390281216/440473133 [00:08<00:01, 47392185.66B/s] 90%|████████▉ | 395033600/440473133 [00:08<00:00, 46433777.34B/s] 91%|█████████ | 399796224/440473133 [00:08<00:00, 46779821.44B/s] 92%|█████████▏| 404799488/440473133 [00:08<00:00, 47709117.04B/s] 93%|█████████▎| 409659392/440473133 [00:08<00:00, 47971819.70B/s] 94%|█████████▍| 414465024/440473133 [00:08<00:00, 46331012.95B/s] 95%|█████████▌| 419117056/440473133 [00:08<00:00, 45711198.29B/s] 96%|█████████▌| 423703552/440473133 [00:08<00:00, 45500183.72B/s] 97%|█████████▋| 428783616/440473133 [00:08<00:00, 46964354.10B/s] 98%|█████████▊| 433630208/440473133 [00:09<00:00, 47403317.83B/s]100%|█████████▉| 438385664/440473133 [00:09<00:00, 46514380.59B/s]100%|██████████| 440473133/440473133 [00:09<00:00, 47686182.96B/s]Downloading data from https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip

   8192/7094233 [..............................] - ETA: 0s
1064960/7094233 [===>..........................] - ETA: 0s
2105344/7094233 [=======>......................] - ETA: 0s
3145728/7094233 [============>.................] - ETA: 0s
4186112/7094233 [================>.............] - ETA: 0s
5005312/7094233 [====================>.........] - ETA: 0s
6012928/7094233 [========================>.....] - ETA: 0s
7053312/7094233 [============================>.] - ETA: 0s
7094272/7094233 [==============================] - 0s 0us/step

Processing text_left with encode:   0%|          | 0/2118 [00:00<?, ?it/s]Processing text_left with encode:   0%|          | 2/2118 [00:00<05:33,  6.35it/s]Processing text_left with encode:  21%|██        | 438/2118 [00:00<03:05,  9.07it/s]Processing text_left with encode:  41%|████▏     | 874/2118 [00:00<01:36, 12.95it/s]Processing text_left with encode:  58%|█████▊    | 1233/2118 [00:00<00:47, 18.47it/s]Processing text_left with encode:  79%|███████▉  | 1676/2118 [00:00<00:16, 26.33it/s]Processing text_left with encode: 100%|██████████| 2118/2118 [00:00<00:00, 2601.12it/s]
Processing text_right with encode:   0%|          | 0/18841 [00:00<?, ?it/s]Processing text_right with encode:   1%|          | 150/18841 [00:00<00:18, 1007.46it/s]Processing text_right with encode:   2%|▏         | 324/18841 [00:00<00:16, 1152.87it/s]Processing text_right with encode:   3%|▎         | 496/18841 [00:00<00:14, 1279.25it/s]Processing text_right with encode:   4%|▎         | 668/18841 [00:00<00:13, 1383.91it/s]Processing text_right with encode:   4%|▍         | 831/18841 [00:00<00:12, 1446.88it/s]Processing text_right with encode:   5%|▌         | 993/18841 [00:00<00:11, 1494.23it/s]Processing text_right with encode:   6%|▌         | 1146/18841 [00:00<00:11, 1503.92it/s]Processing text_right with encode:   7%|▋         | 1313/18841 [00:00<00:11, 1548.69it/s]Processing text_right with encode:   8%|▊         | 1466/18841 [00:00<00:11, 1530.51it/s]Processing text_right with encode:   9%|▊         | 1625/18841 [00:01<00:11, 1545.80it/s]Processing text_right with encode:  10%|▉         | 1796/18841 [00:01<00:10, 1590.98it/s]Processing text_right with encode:  10%|█         | 1955/18841 [00:01<00:11, 1522.27it/s]Processing text_right with encode:  11%|█         | 2108/18841 [00:01<00:11, 1456.82it/s]Processing text_right with encode:  12%|█▏        | 2281/18841 [00:01<00:10, 1528.18it/s]Processing text_right with encode:  13%|█▎        | 2448/18841 [00:01<00:10, 1564.86it/s]Processing text_right with encode:  14%|█▍        | 2613/18841 [00:01<00:10, 1587.63it/s]Processing text_right with encode:  15%|█▍        | 2802/18841 [00:01<00:09, 1667.38it/s]Processing text_right with encode:  16%|█▌        | 2971/18841 [00:01<00:09, 1615.77it/s]Processing text_right with encode:  17%|█▋        | 3135/18841 [00:02<00:09, 1609.12it/s]Processing text_right with encode:  18%|█▊        | 3298/18841 [00:02<00:09, 1581.59it/s]Processing text_right with encode:  18%|█▊        | 3458/18841 [00:02<00:09, 1582.71it/s]Processing text_right with encode:  19%|█▉        | 3630/18841 [00:02<00:09, 1619.41it/s]Processing text_right with encode:  20%|██        | 3793/18841 [00:02<00:09, 1609.09it/s]Processing text_right with encode:  21%|██        | 3961/18841 [00:02<00:09, 1627.41it/s]Processing text_right with encode:  22%|██▏       | 4125/18841 [00:02<00:09, 1574.22it/s]Processing text_right with encode:  23%|██▎       | 4284/18841 [00:02<00:09, 1552.50it/s]Processing text_right with encode:  24%|██▎       | 4440/18841 [00:02<00:09, 1548.04it/s]Processing text_right with encode:  24%|██▍       | 4603/18841 [00:02<00:09, 1570.13it/s]Processing text_right with encode:  25%|██▌       | 4764/18841 [00:03<00:08, 1578.64it/s]Processing text_right with encode:  26%|██▌       | 4939/18841 [00:03<00:08, 1625.17it/s]Processing text_right with encode:  27%|██▋       | 5103/18841 [00:03<00:08, 1613.67it/s]Processing text_right with encode:  28%|██▊       | 5282/18841 [00:03<00:08, 1662.42it/s]Processing text_right with encode:  29%|██▉       | 5449/18841 [00:03<00:08, 1622.45it/s]Processing text_right with encode:  30%|██▉       | 5612/18841 [00:03<00:08, 1589.29it/s]Processing text_right with encode:  31%|███       | 5774/18841 [00:03<00:08, 1595.09it/s]Processing text_right with encode:  32%|███▏      | 5944/18841 [00:03<00:07, 1623.56it/s]Processing text_right with encode:  32%|███▏      | 6107/18841 [00:03<00:07, 1620.76it/s]Processing text_right with encode:  33%|███▎      | 6270/18841 [00:03<00:07, 1605.25it/s]Processing text_right with encode:  34%|███▍      | 6450/18841 [00:04<00:07, 1657.08it/s]Processing text_right with encode:  35%|███▌      | 6633/18841 [00:04<00:07, 1704.63it/s]Processing text_right with encode:  36%|███▌      | 6805/18841 [00:04<00:07, 1667.23it/s]Processing text_right with encode:  37%|███▋      | 6973/18841 [00:04<00:07, 1637.53it/s]Processing text_right with encode:  38%|███▊      | 7138/18841 [00:04<00:07, 1610.79it/s]Processing text_right with encode:  39%|███▉      | 7309/18841 [00:04<00:07, 1635.54it/s]Processing text_right with encode:  40%|███▉      | 7484/18841 [00:04<00:06, 1664.48it/s]Processing text_right with encode:  41%|████      | 7651/18841 [00:04<00:06, 1663.35it/s]Processing text_right with encode:  41%|████▏     | 7819/18841 [00:04<00:06, 1665.21it/s]Processing text_right with encode:  42%|████▏     | 7986/18841 [00:04<00:06, 1639.08it/s]Processing text_right with encode:  43%|████▎     | 8151/18841 [00:05<00:06, 1634.18it/s]Processing text_right with encode:  44%|████▍     | 8316/18841 [00:05<00:06, 1637.60it/s]Processing text_right with encode:  45%|████▌     | 8482/18841 [00:05<00:06, 1642.76it/s]Processing text_right with encode:  46%|████▌     | 8647/18841 [00:05<00:06, 1642.01it/s]Processing text_right with encode:  47%|████▋     | 8816/18841 [00:05<00:06, 1655.12it/s]Processing text_right with encode:  48%|████▊     | 8982/18841 [00:05<00:06, 1625.86it/s]Processing text_right with encode:  49%|████▊     | 9145/18841 [00:05<00:06, 1602.88it/s]Processing text_right with encode:  49%|████▉     | 9312/18841 [00:05<00:05, 1622.42it/s]Processing text_right with encode:  50%|█████     | 9475/18841 [00:05<00:05, 1624.65it/s]Processing text_right with encode:  51%|█████     | 9638/18841 [00:06<00:05, 1612.23it/s]Processing text_right with encode:  52%|█████▏    | 9800/18841 [00:06<00:05, 1592.12it/s]Processing text_right with encode:  53%|█████▎    | 9970/18841 [00:06<00:05, 1620.82it/s]Processing text_right with encode:  54%|█████▍    | 10144/18841 [00:06<00:05, 1653.80it/s]Processing text_right with encode:  55%|█████▍    | 10310/18841 [00:06<00:05, 1613.94it/s]Processing text_right with encode:  56%|█████▌    | 10500/18841 [00:06<00:04, 1688.43it/s]Processing text_right with encode:  57%|█████▋    | 10671/18841 [00:06<00:04, 1673.44it/s]Processing text_right with encode:  58%|█████▊    | 10840/18841 [00:06<00:04, 1666.95it/s]Processing text_right with encode:  58%|█████▊    | 11008/18841 [00:06<00:04, 1634.37it/s]Processing text_right with encode:  59%|█████▉    | 11173/18841 [00:06<00:04, 1620.44it/s]Processing text_right with encode:  60%|██████    | 11336/18841 [00:07<00:04, 1596.10it/s]Processing text_right with encode:  61%|██████    | 11503/18841 [00:07<00:04, 1614.43it/s]Processing text_right with encode:  62%|██████▏   | 11674/18841 [00:07<00:04, 1639.91it/s]Processing text_right with encode:  63%|██████▎   | 11839/18841 [00:07<00:04, 1632.03it/s]Processing text_right with encode:  64%|██████▎   | 12010/18841 [00:07<00:04, 1654.32it/s]Processing text_right with encode:  65%|██████▍   | 12176/18841 [00:07<00:04, 1635.06it/s]Processing text_right with encode:  66%|██████▌   | 12344/18841 [00:07<00:03, 1647.25it/s]Processing text_right with encode:  66%|██████▋   | 12509/18841 [00:07<00:03, 1643.90it/s]Processing text_right with encode:  67%|██████▋   | 12674/18841 [00:07<00:03, 1641.15it/s]Processing text_right with encode:  68%|██████▊   | 12839/18841 [00:07<00:03, 1640.97it/s]Processing text_right with encode:  69%|██████▉   | 13004/18841 [00:08<00:03, 1635.35it/s]Processing text_right with encode:  70%|██████▉   | 13171/18841 [00:08<00:03, 1643.27it/s]Processing text_right with encode:  71%|███████   | 13341/18841 [00:08<00:03, 1659.29it/s]Processing text_right with encode:  72%|███████▏  | 13507/18841 [00:08<00:03, 1580.45it/s]Processing text_right with encode:  73%|███████▎  | 13677/18841 [00:08<00:03, 1613.07it/s]Processing text_right with encode:  73%|███████▎  | 13844/18841 [00:08<00:03, 1627.90it/s]Processing text_right with encode:  74%|███████▍  | 14021/18841 [00:08<00:02, 1667.48it/s]Processing text_right with encode:  75%|███████▌  | 14189/18841 [00:08<00:02, 1632.67it/s]Processing text_right with encode:  76%|███████▌  | 14353/18841 [00:08<00:02, 1618.06it/s]Processing text_right with encode:  77%|███████▋  | 14521/18841 [00:08<00:02, 1635.54it/s]Processing text_right with encode:  78%|███████▊  | 14692/18841 [00:09<00:02, 1656.64it/s]Processing text_right with encode:  79%|███████▉  | 14862/18841 [00:09<00:02, 1667.79it/s]Processing text_right with encode:  80%|███████▉  | 15030/18841 [00:09<00:02, 1658.73it/s]Processing text_right with encode:  81%|████████  | 15200/18841 [00:09<00:02, 1669.40it/s]Processing text_right with encode:  82%|████████▏ | 15368/18841 [00:09<00:02, 1620.26it/s]Processing text_right with encode:  82%|████████▏ | 15531/18841 [00:09<00:02, 1537.88it/s]Processing text_right with encode:  83%|████████▎ | 15692/18841 [00:09<00:02, 1556.39it/s]Processing text_right with encode:  84%|████████▍ | 15867/18841 [00:09<00:01, 1606.68it/s]Processing text_right with encode:  85%|████████▌ | 16035/18841 [00:09<00:01, 1627.57it/s]Processing text_right with encode:  86%|████████▌ | 16199/18841 [00:10<00:01, 1619.52it/s]Processing text_right with encode:  87%|████████▋ | 16362/18841 [00:10<00:01, 1609.91it/s]Processing text_right with encode:  88%|████████▊ | 16528/18841 [00:10<00:01, 1621.83it/s]Processing text_right with encode:  89%|████████▊ | 16691/18841 [00:10<00:01, 1620.48it/s]Processing text_right with encode:  89%|████████▉ | 16854/18841 [00:10<00:01, 1613.14it/s]Processing text_right with encode:  90%|█████████ | 17019/18841 [00:10<00:01, 1622.26it/s]Processing text_right with encode:  91%|█████████ | 17182/18841 [00:10<00:01, 1599.16it/s]Processing text_right with encode:  92%|█████████▏| 17352/18841 [00:10<00:00, 1625.84it/s]Processing text_right with encode:  93%|█████████▎| 17515/18841 [00:10<00:00, 1611.67it/s]Processing text_right with encode:  94%|█████████▍| 17690/18841 [00:10<00:00, 1650.58it/s]Processing text_right with encode:  95%|█████████▍| 17856/18841 [00:11<00:00, 1648.65it/s]Processing text_right with encode:  96%|█████████▌| 18031/18841 [00:11<00:00, 1677.12it/s]Processing text_right with encode:  97%|█████████▋| 18210/18841 [00:11<00:00, 1707.84it/s]Processing text_right with encode:  98%|█████████▊| 18382/18841 [00:11<00:00, 1682.81it/s]Processing text_right with encode:  99%|█████████▊| 18562/18841 [00:11<00:00, 1710.94it/s]Processing text_right with encode:  99%|█████████▉| 18734/18841 [00:11<00:00, 1661.85it/s]Processing text_right with encode: 100%|██████████| 18841/18841 [00:11<00:00, 1620.24it/s]
Processing length_left with len:   0%|          | 0/2118 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 2118/2118 [00:00<00:00, 513528.87it/s]
Processing length_right with len:   0%|          | 0/18841 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 18841/18841 [00:00<00:00, 742338.30it/s]
Processing text_left with encode:   0%|          | 0/633 [00:00<?, ?it/s]Processing text_left with encode:  69%|██████▊   | 435/633 [00:00<00:00, 4348.46it/s]Processing text_left with encode: 100%|██████████| 633/633 [00:00<00:00, 4221.78it/s]
Processing text_right with encode:   0%|          | 0/5961 [00:00<?, ?it/s]Processing text_right with encode:   3%|▎         | 172/5961 [00:00<00:03, 1716.81it/s]Processing text_right with encode:   6%|▌         | 343/5961 [00:00<00:03, 1713.85it/s]Processing text_right with encode:   8%|▊         | 495/5961 [00:00<00:03, 1647.11it/s]Processing text_right with encode:  11%|█         | 664/5961 [00:00<00:03, 1657.78it/s]Processing text_right with encode:  14%|█▍        | 827/5961 [00:00<00:03, 1646.10it/s]Processing text_right with encode:  16%|█▋        | 975/5961 [00:00<00:03, 1591.68it/s]Processing text_right with encode:  19%|█▉        | 1139/5961 [00:00<00:03, 1604.83it/s]Processing text_right with encode:  22%|██▏       | 1312/5961 [00:00<00:02, 1638.71it/s]Processing text_right with encode:  25%|██▍       | 1476/5961 [00:00<00:02, 1634.70it/s]Processing text_right with encode:  27%|██▋       | 1633/5961 [00:01<00:02, 1609.98it/s]Processing text_right with encode:  30%|███       | 1790/5961 [00:01<00:02, 1569.82it/s]Processing text_right with encode:  33%|███▎      | 1957/5961 [00:01<00:02, 1598.08it/s]Processing text_right with encode:  36%|███▌      | 2126/5961 [00:01<00:02, 1624.24it/s]Processing text_right with encode:  38%|███▊      | 2290/5961 [00:01<00:02, 1628.46it/s]Processing text_right with encode:  41%|████      | 2452/5961 [00:01<00:02, 1580.90it/s]Processing text_right with encode:  44%|████▍     | 2622/5961 [00:01<00:02, 1608.88it/s]Processing text_right with encode:  47%|████▋     | 2810/5961 [00:01<00:01, 1679.23it/s]Processing text_right with encode:  50%|████▉     | 2980/5961 [00:01<00:01, 1684.04it/s]Processing text_right with encode:  53%|█████▎    | 3162/5961 [00:01<00:01, 1721.02it/s]Processing text_right with encode:  56%|█████▌    | 3335/5961 [00:02<00:01, 1677.33it/s]Processing text_right with encode:  59%|█████▉    | 3504/5961 [00:02<00:01, 1650.86it/s]Processing text_right with encode:  62%|██████▏   | 3677/5961 [00:02<00:01, 1672.78it/s]Processing text_right with encode:  65%|██████▍   | 3845/5961 [00:02<00:01, 1627.32it/s]Processing text_right with encode:  68%|██████▊   | 4031/5961 [00:02<00:01, 1686.98it/s]Processing text_right with encode:  70%|███████   | 4201/5961 [00:02<00:01, 1679.02it/s]Processing text_right with encode:  73%|███████▎  | 4370/5961 [00:02<00:00, 1668.73it/s]Processing text_right with encode:  76%|███████▌  | 4542/5961 [00:02<00:00, 1680.74it/s]Processing text_right with encode:  79%|███████▉  | 4711/5961 [00:02<00:00, 1675.02it/s]Processing text_right with encode:  82%|████████▏ | 4879/5961 [00:02<00:00, 1612.86it/s]Processing text_right with encode:  85%|████████▍ | 5041/5961 [00:03<00:00, 1545.67it/s]Processing text_right with encode:  87%|████████▋ | 5215/5961 [00:03<00:00, 1598.07it/s]Processing text_right with encode:  90%|█████████ | 5377/5961 [00:03<00:00, 1580.79it/s]Processing text_right with encode:  93%|█████████▎| 5536/5961 [00:03<00:00, 1553.40it/s]Processing text_right with encode:  96%|█████████▌| 5693/5961 [00:03<00:00, 1540.10it/s]Processing text_right with encode:  98%|█████████▊| 5854/5961 [00:03<00:00, 1560.40it/s]Processing text_right with encode: 100%|██████████| 5961/5961 [00:03<00:00, 1630.25it/s]
Processing length_left with len:   0%|          | 0/633 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 633/633 [00:00<00:00, 478118.93it/s]
Processing length_right with len:   0%|          | 0/5961 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 5961/5961 [00:00<00:00, 685631.72it/s]
  #### Model  fit   ############################################# 

  0%|          | 0/102 [00:00<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [01:10<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [01:10<?, ?it/s, loss=1.038]Epoch 1/1:   1%|          | 1/102 [01:10<1:58:03, 70.13s/it, loss=1.038]Epoch 1/1:   1%|          | 1/102 [01:55<1:58:03, 70.13s/it, loss=1.038]Epoch 1/1:   1%|          | 1/102 [01:55<1:58:03, 70.13s/it, loss=1.058]Epoch 1/1:   2%|▏         | 2/102 [01:55<1:44:21, 62.61s/it, loss=1.058]Killed

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
From github.com:arita37/mlmodels_store
   9e8bcbb..4914890  master     -> origin/master
Updating 9e8bcbb..4914890
Fast-forward
 error_list/20200516/list_log_json_20200516.md      | 1146 ++++++-------
 error_list/20200516/list_log_jupyter_20200516.md   | 1750 ++++++++++----------
 .../20200516/list_log_pullrequest_20200516.md      |    2 +-
 error_list/20200516/list_log_test_cli_20200516.md  |  138 +-
 4 files changed, 1517 insertions(+), 1519 deletions(-)
[master c46ce5f] ml_store
 1 file changed, 74 insertions(+)
To github.com:arita37/mlmodels_store.git
   4914890..c46ce5f  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py 

  #### Loading params   ############################################## 

  {'data_info': {'data_path': 'mlmodels/dataset/vision/MNIST', 'dataset': 'MNIST', 'data_type': 'tch_dataset', 'batch_size': 10, 'train': True}, 'preprocessors': [{'name': 'tch_dataset_start', 'uri': 'mlmodels/preprocess/generic.py::get_dataset_torch', 'args': {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True}}]} {'checkpointdir': 'ztest/model_tch/torchhub/restnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18/'} 

  #### Loading dataset   ############################################# 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2f14baad90>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2f14baad90>

  function with postional parmater data_info <function get_dataset_torch at 0x7f2f14baad90> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:08, 145016.65it/s] 76%|███████▌  | 7528448/9912422 [00:00<00:11, 206995.39it/s]9920512it [00:00, 42417613.87it/s]                           
0it [00:00, ?it/s]32768it [00:00, 630037.74it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 477494.92it/s]1654784it [00:00, 11345241.67it/s]                         
0it [00:00, ?it/s]8192it [00:00, 184541.27it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2f140fbae8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2f140fbae8>

  function with postional parmater data_info <function get_dataset_torch at 0x7f2f140fbae8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.0019427751302719116 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.011238692045211792 	 Accuracy: 0
model saves at 0 accuracy
Train Epoch: 2 	 Loss: 0.0013638022541999817 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.010076085329055786 	 Accuracy: 1
model saves at 1 accuracy

  #### Predict   ##################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2f140fb8c8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2f140fb8c8>

  function with postional parmater data_info <function get_dataset_torch at 0x7f2f140fb8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  #### metrics   ##################################################### 
None

  #### Plot   ######################################################## 

  #### Save  ######################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18//torch_model/ ['model.pb', 'torch_model_pars.pkl'] 

  #### Load   ######################################################## 
<__main__.Model object at 0x7f2f14554b00>

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
<__main__.Model object at 0x7f2f0c6d6438>

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
[master 2dd0d49] ml_store
 1 file changed, 149 insertions(+)
To github.com:arita37/mlmodels_store.git
   c46ce5f..2dd0d49  master -> master





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
