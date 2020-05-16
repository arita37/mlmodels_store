
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
