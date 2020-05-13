
  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_all 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '6672e19fe4cfa7df885e45d91d645534b8989485', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/6672e19fe4cfa7df885e45d91d645534b8989485

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
From github.com:arita37/mlmodels_store
   38c01ce..109f5b5  master     -> origin/master
Updating 38c01ce..109f5b5
Fast-forward
 error_list/20200513/list_log_benchmark_20200513.md |  186 +--
 error_list/20200513/list_log_jupyter_20200513.md   | 1661 ++++++++++----------
 error_list/20200513/list_log_test_cli_20200513.md  |  152 +-
 ...-08_6672e19fe4cfa7df885e45d91d645534b8989485.py |  373 +++++
 ...-13_6672e19fe4cfa7df885e45d91d645534b8989485.py |  621 ++++++++
 5 files changed, 1995 insertions(+), 998 deletions(-)
 create mode 100644 log_dataloader/log_2020-05-13-00-08_6672e19fe4cfa7df885e45d91d645534b8989485.py
 create mode 100644 log_pullrequest/log_pr_2020-05-13-00-13_6672e19fe4cfa7df885e45d91d645534b8989485.py
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
[master 15cc613] ml_store
 1 file changed, 71 insertions(+)
 create mode 100644 log_testall/log_testall_2020-05-13-00-15_6672e19fe4cfa7df885e45d91d645534b8989485.py
To github.com:arita37/mlmodels_store.git
   109f5b5..15cc613  master -> master





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
[master a3b1fe0] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   15cc613..a3b1fe0  master -> master





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
[master 094f91d] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   a3b1fe0..094f91d  master -> master





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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         12          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         20          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-13 00:16:24.689096: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 00:16:24.694717: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-05-13 00:16:24.694874: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e6d4c47cf0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 00:16:24.694890: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 198
Trainable params: 198
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 1s 1ms/sample - loss: 0.2547 - binary_crossentropy: 0.7818 - val_loss: 0.2499 - val_binary_crossentropy: 0.6929

  #### metrics   #################################################### 
{'MSE': 0.25213336951268484}

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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         12          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
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
Total params: 198
Trainable params: 198
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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
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
Total params: 403
Trainable params: 403
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.4600 - binary_crossentropy: 7.0955500/500 [==============================] - 1s 2ms/sample - loss: 0.4620 - binary_crossentropy: 7.1263 - val_loss: 0.5060 - val_binary_crossentropy: 7.8050

  #### metrics   #################################################### 
{'MSE': 0.484}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
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
Total params: 403
Trainable params: 403
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         28          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_2[0][0]           
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
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 2ms/sample - loss: 0.2501 - binary_crossentropy: 0.6933 - val_loss: 0.2501 - val_binary_crossentropy: 0.6933

  #### metrics   #################################################### 
{'MSE': 0.24988218095295558}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         28          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_2[0][0]           
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
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
weighted_sequence_layer_9 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
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
Total params: 388
Trainable params: 388
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2623 - binary_crossentropy: 0.9839500/500 [==============================] - 1s 3ms/sample - loss: 0.2704 - binary_crossentropy: 0.8443 - val_loss: 0.2658 - val_binary_crossentropy: 0.7772

  #### metrics   #################################################### 
{'MSE': 0.2632275362151099}

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
weighted_sequence_layer_9 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
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
Total params: 388
Trainable params: 388
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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_max[0][0]               
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
Total params: 178
Trainable params: 178
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2718 - binary_crossentropy: 1.1318500/500 [==============================] - 2s 3ms/sample - loss: 0.2585 - binary_crossentropy: 0.8677 - val_loss: 0.2525 - val_binary_crossentropy: 0.7506

  #### metrics   #################################################### 
{'MSE': 0.25492013289398535}

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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_max[0][0]               
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
Total params: 178
Trainable params: 178
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-13 00:17:48.203839: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 00:17:48.206247: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 00:17:48.212380: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-13 00:17:48.222582: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-13 00:17:48.224441: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 00:17:48.226036: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 00:17:48.227619: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2496 - val_binary_crossentropy: 0.6923
2020-05-13 00:17:49.499101: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 00:17:49.501663: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 00:17:49.506895: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-13 00:17:49.517802: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-13 00:17:49.519665: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 00:17:49.521394: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 00:17:49.523029: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24934984877197644}

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
2020-05-13 00:18:14.816117: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 00:18:14.817444: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 00:18:14.820937: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-13 00:18:14.826994: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-13 00:18:14.828009: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 00:18:14.828997: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 00:18:14.829893: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2508 - val_binary_crossentropy: 0.6948
2020-05-13 00:18:16.511734: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 00:18:16.513202: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 00:18:16.517262: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-13 00:18:16.523758: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-13 00:18:16.524897: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 00:18:16.525954: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 00:18:16.526938: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.25100918841449094}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-13 00:18:53.314322: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 00:18:53.320036: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 00:18:53.336989: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-13 00:18:53.367859: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-13 00:18:53.372979: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 00:18:53.378092: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 00:18:53.383044: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.3964 - binary_crossentropy: 0.9932 - val_loss: 0.2514 - val_binary_crossentropy: 0.6959
2020-05-13 00:18:55.791169: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 00:18:55.795470: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 00:18:55.806790: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-13 00:18:55.831536: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-13 00:18:55.835304: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 00:18:55.838862: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 00:18:55.842315: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.23888547737313814}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
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
Total params: 650
Trainable params: 650
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.3045 - binary_crossentropy: 1.7230500/500 [==============================] - 5s 9ms/sample - loss: 0.2916 - binary_crossentropy: 1.6652 - val_loss: 0.2895 - val_binary_crossentropy: 1.6628

  #### metrics   #################################################### 
{'MSE': 0.29021207912335445}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
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
Total params: 650
Trainable params: 650
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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 2)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 2)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 2)         2           sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         14          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         14          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         14          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         6           sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_2[0][0]           
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
Total params: 212
Trainable params: 212
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.5300 - binary_crossentropy: 8.1752500/500 [==============================] - 5s 10ms/sample - loss: 0.5320 - binary_crossentropy: 8.2061 - val_loss: 0.5060 - val_binary_crossentropy: 7.8050

  #### metrics   #################################################### 
{'MSE': 0.519}

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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 2)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 2)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 2)         2           sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         14          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         14          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         14          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         6           sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_2[0][0]           
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
Total params: 212
Trainable params: 212
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
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
Total params: 1,889
Trainable params: 1,889
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2923 - binary_crossentropy: 2.0771500/500 [==============================] - 5s 10ms/sample - loss: 0.3213 - binary_crossentropy: 2.1689 - val_loss: 0.2943 - val_binary_crossentropy: 1.9259

  #### metrics   #################################################### 
{'MSE': 0.30681065024102383}

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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
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
Total params: 1,889
Trainable params: 1,889
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
regionsequence_mean (InputLayer [(None, 7)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 6)]          0                                            
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
region_10sparse_seq_emb_regions (None, 6, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 6, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 6, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 6, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 6, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 6, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 6, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 6, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 6, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 6, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 6, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 6, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 6, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 6, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 6, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 6, 1)         6           regionsequence_max[0][0]         
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
Total params: 160
Trainable params: 160
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2515 - binary_crossentropy: 0.6960500/500 [==============================] - 6s 13ms/sample - loss: 0.2533 - binary_crossentropy: 0.6997 - val_loss: 0.2523 - val_binary_crossentropy: 0.6976

  #### metrics   #################################################### 
{'MSE': 0.25240205802405585}

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
regionsequence_mean (InputLayer [(None, 7)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 6)]          0                                            
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
region_10sparse_seq_emb_regions (None, 6, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 6, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 6, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 6, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 6, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 6, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 6, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 6, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 6, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 6, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 6, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 6, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 6, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 6, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 6, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 6, 1)         6           regionsequence_max[0][0]         
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
Total params: 160
Trainable params: 160
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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_max[0][0]               
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
Total params: 1,437
Trainable params: 1,437
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2713 - binary_crossentropy: 0.7381500/500 [==============================] - 6s 13ms/sample - loss: 0.2555 - binary_crossentropy: 0.7047 - val_loss: 0.2519 - val_binary_crossentropy: 0.6970

  #### metrics   #################################################### 
{'MSE': 0.25283761294310864}

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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_max[0][0]               
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
Total params: 1,437
Trainable params: 1,437
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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         4           hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         12          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 7, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 4, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 1, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         12          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 7, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         12          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 4, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         12          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 1, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 7, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 4, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 7, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 1, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 4, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 1, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           hash_11[0][0]                    
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
Total params: 3,035
Trainable params: 2,955
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2870 - binary_crossentropy: 1.8028500/500 [==============================] - 7s 14ms/sample - loss: 0.3620 - binary_crossentropy: 2.7888 - val_loss: 0.3474 - val_binary_crossentropy: 2.8815

  #### metrics   #################################################### 
{'MSE': 0.34710980325214974}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         4           hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         12          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 7, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 4, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 1, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         12          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 7, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         12          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 4, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         12          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 1, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 7, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 4, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 7, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 1, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 4, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 1, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           hash_11[0][0]                    
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
Total params: 3,035
Trainable params: 2,955
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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         36          sequence_max[0][0]               
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
Total params: 409
Trainable params: 409
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2580 - binary_crossentropy: 0.7096500/500 [==============================] - 7s 15ms/sample - loss: 0.2510 - binary_crossentropy: 0.6952 - val_loss: 0.2500 - val_binary_crossentropy: 0.6931

  #### metrics   #################################################### 
{'MSE': 0.25042817433132736}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         36          sequence_max[0][0]               
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
Total params: 409
Trainable params: 409
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_125 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_194[0][0] 
                                                                 sequence_pooling_layer_195[0][0] 
                                                                 sequence_pooling_layer_196[0][0] 
                                                                 sequence_pooling_layer_197[0][0] 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
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
Total params: 2,064
Trainable params: 2,064
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.3000 - binary_crossentropy: 1.8440500/500 [==============================] - 7s 15ms/sample - loss: 0.2875 - binary_crossentropy: 1.5811 - val_loss: 0.2825 - val_binary_crossentropy: 1.5676

  #### metrics   #################################################### 
{'MSE': 0.28378431392190784}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_125 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_194[0][0] 
                                                                 sequence_pooling_layer_195[0][0] 
                                                                 sequence_pooling_layer_196[0][0] 
                                                                 sequence_pooling_layer_197[0][0] 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
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
Total params: 2,064
Trainable params: 2,064
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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_47 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_90 (Concatenate)    (None, 1, 20)        0           no_mask_130[0][0]                
                                                                 no_mask_130[1][0]                
                                                                 no_mask_130[2][0]                
                                                                 no_mask_130[3][0]                
                                                                 no_mask_130[4][0]                
__________________________________________________________________________________________________
no_mask_131 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
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
100/500 [=====>........................] - ETA: 9s - loss: 0.2863 - binary_crossentropy: 1.0465500/500 [==============================] - 8s 16ms/sample - loss: 0.3007 - binary_crossentropy: 1.0514 - val_loss: 0.2982 - val_binary_crossentropy: 0.9910

  #### metrics   #################################################### 
{'MSE': 0.2976960705019532}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_47 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_90 (Concatenate)    (None, 1, 20)        0           no_mask_130[0][0]                
                                                                 no_mask_130[1][0]                
                                                                 no_mask_130[2][0]                
                                                                 no_mask_130[3][0]                
                                                                 no_mask_130[4][0]                
__________________________________________________________________________________________________
no_mask_131 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
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
   094f91d..f1ec777  master     -> origin/master
Updating 094f91d..f1ec777
Fast-forward
 error_list/20200513/list_log_test_cli_20200513.md  |  138 +-
 ...-20_6672e19fe4cfa7df885e45d91d645534b8989485.py | 3977 ++++++++++++++++++++
 ...-20_6672e19fe4cfa7df885e45d91d645534b8989485.py | 2010 ++++++++++
 ...-24_6672e19fe4cfa7df885e45d91d645534b8989485.py | 3483 +++++++++++++++++
 4 files changed, 9539 insertions(+), 69 deletions(-)
 create mode 100644 log_json/log_json_2020-05-13-00-20_6672e19fe4cfa7df885e45d91d645534b8989485.py
 create mode 100644 log_jupyter/log_jupyter_2020-05-13-00-20_6672e19fe4cfa7df885e45d91d645534b8989485.py
 create mode 100644 log_test_cli/log_cli_2020-05-13-00-24_6672e19fe4cfa7df885e45d91d645534b8989485.py
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
[master 276fe57] ml_store
 1 file changed, 5672 insertions(+)
To github.com:arita37/mlmodels_store.git
   f1ec777..276fe57  master -> master





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
[master d53ccc7] ml_store
 1 file changed, 50 insertions(+)
To github.com:arita37/mlmodels_store.git
   276fe57..d53ccc7  master -> master





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
[master 638e646] ml_store
 1 file changed, 46 insertions(+)
To github.com:arita37/mlmodels_store.git
   d53ccc7..638e646  master -> master





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
[master cbab89e] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   638e646..cbab89e  master -> master





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

2020-05-13 00:33:34.981012: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 00:33:34.987115: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-05-13 00:33:34.987314: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5630aa40adb0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 00:33:34.987333: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 7s - loss: 1.3861
256/354 [====================>.........] - ETA: 3s - loss: 1.2567
354/354 [==============================] - 14s 38ms/step - loss: 1.3482 - val_loss: 1.8973

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
[master 959f744] ml_store
 1 file changed, 149 insertions(+)
To github.com:arita37/mlmodels_store.git
   cbab89e..959f744  master -> master





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
[master 9e4cfeb] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   959f744..9e4cfeb  master -> master





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
[master ae90786] ml_store
 1 file changed, 44 insertions(+)
To github.com:arita37/mlmodels_store.git
   9e4cfeb..ae90786  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 46s
   57344/17464789 [..............................] - ETA: 39s
   90112/17464789 [..............................] - ETA: 37s
  180224/17464789 [..............................] - ETA: 25s
  335872/17464789 [..............................] - ETA: 16s
  663552/17464789 [>.............................] - ETA: 10s
 1310720/17464789 [=>............................] - ETA: 5s 
 2621440/17464789 [===>..........................] - ETA: 2s
 5160960/17464789 [=======>......................] - ETA: 1s
 7847936/17464789 [============>.................] - ETA: 0s
10567680/17464789 [=================>............] - ETA: 0s
13189120/17464789 [=====================>........] - ETA: 0s
16089088/17464789 [==========================>...] - ETA: 0s
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
2020-05-13 00:34:39.124993: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 00:34:39.129453: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-05-13 00:34:39.129604: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56242f2c9040 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 00:34:39.129620: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 13s - loss: 7.9120 - accuracy: 0.4840
 2000/25000 [=>............................] - ETA: 9s - loss: 7.9580 - accuracy: 0.4810 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.9017 - accuracy: 0.4847
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6360 - accuracy: 0.5020
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7341 - accuracy: 0.4956
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7075 - accuracy: 0.4973
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7104 - accuracy: 0.4971
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6781 - accuracy: 0.4992
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6581 - accuracy: 0.5006
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6712 - accuracy: 0.4997
11000/25000 [============>.................] - ETA: 4s - loss: 7.6499 - accuracy: 0.5011
12000/25000 [=============>................] - ETA: 4s - loss: 7.6462 - accuracy: 0.5013
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6690 - accuracy: 0.4998
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6852 - accuracy: 0.4988
15000/25000 [=================>............] - ETA: 3s - loss: 7.6789 - accuracy: 0.4992
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7059 - accuracy: 0.4974
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6829 - accuracy: 0.4989
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6854 - accuracy: 0.4988
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6731 - accuracy: 0.4996
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6697 - accuracy: 0.4998
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6790 - accuracy: 0.4992
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6638 - accuracy: 0.5002
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6580 - accuracy: 0.5006
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
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
(<mlmodels.util.Model_empty object at 0x7f8623e3bd68>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f86277d3c88> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 8.0500 - accuracy: 0.4750
 2000/25000 [=>............................] - ETA: 9s - loss: 7.9196 - accuracy: 0.4835 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.8302 - accuracy: 0.4893
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7816 - accuracy: 0.4925
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.8169 - accuracy: 0.4902
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.8353 - accuracy: 0.4890
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.8221 - accuracy: 0.4899
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7874 - accuracy: 0.4921
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.8200 - accuracy: 0.4900
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7832 - accuracy: 0.4924
11000/25000 [============>.................] - ETA: 4s - loss: 7.8004 - accuracy: 0.4913
12000/25000 [=============>................] - ETA: 3s - loss: 7.7803 - accuracy: 0.4926
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7398 - accuracy: 0.4952
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7488 - accuracy: 0.4946
15000/25000 [=================>............] - ETA: 3s - loss: 7.7402 - accuracy: 0.4952
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7165 - accuracy: 0.4967
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7117 - accuracy: 0.4971
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7177 - accuracy: 0.4967
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7344 - accuracy: 0.4956
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7073 - accuracy: 0.4974
21000/25000 [========================>.....] - ETA: 1s - loss: 7.7017 - accuracy: 0.4977
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6834 - accuracy: 0.4989
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6500 - accuracy: 0.5011
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6513 - accuracy: 0.5010
25000/25000 [==============================] - 9s 356us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 13s - loss: 7.6360 - accuracy: 0.5020
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6743 - accuracy: 0.4995 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6206 - accuracy: 0.5030
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5861 - accuracy: 0.5052
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7310 - accuracy: 0.4958
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7126 - accuracy: 0.4970
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7455 - accuracy: 0.4949
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7241 - accuracy: 0.4963
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6717 - accuracy: 0.4997
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6482 - accuracy: 0.5012
11000/25000 [============>.................] - ETA: 4s - loss: 7.6583 - accuracy: 0.5005
12000/25000 [=============>................] - ETA: 4s - loss: 7.6487 - accuracy: 0.5012
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6784 - accuracy: 0.4992
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6776 - accuracy: 0.4993
15000/25000 [=================>............] - ETA: 3s - loss: 7.6452 - accuracy: 0.5014
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6475 - accuracy: 0.5013
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6802 - accuracy: 0.4991
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6930 - accuracy: 0.4983
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6650 - accuracy: 0.5001
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6758 - accuracy: 0.4994
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6703 - accuracy: 0.4998
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6854 - accuracy: 0.4988
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6913 - accuracy: 0.4984
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6903 - accuracy: 0.4985
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
   ae90786..258f5a3  master     -> origin/master
Updating ae90786..258f5a3
Fast-forward
 error_list/20200513/list_log_benchmark_20200513.md |  180 +--
 error_list/20200513/list_log_jupyter_20200513.md   | 1661 ++++++++++----------
 error_list/20200513/list_log_test_cli_20200513.md  |  138 +-
 3 files changed, 983 insertions(+), 996 deletions(-)
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
[master ac27993] ml_store
 1 file changed, 334 insertions(+)
To github.com:arita37/mlmodels_store.git
   258f5a3..ac27993  master -> master





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

13/13 [==============================] - 2s 119ms/step - loss: nan
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
[master f85da80] ml_store
 1 file changed, 125 insertions(+)
To github.com:arita37/mlmodels_store.git
   ac27993..f85da80  master -> master





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
   24576/11490434 [..............................] - ETA: 27s
   49152/11490434 [..............................] - ETA: 28s
   98304/11490434 [..............................] - ETA: 21s
  204800/11490434 [..............................] - ETA: 13s
  434176/11490434 [>.............................] - ETA: 7s 
  884736/11490434 [=>............................] - ETA: 4s
 1810432/11490434 [===>..........................] - ETA: 2s
 3629056/11490434 [========>.....................] - ETA: 1s
 6414336/11490434 [===============>..............] - ETA: 0s
 8953856/11490434 [======================>.......] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:47 - loss: 2.3138 - categorical_accuracy: 0.0625
   64/60000 [..............................] - ETA: 4:44 - loss: 2.2619 - categorical_accuracy: 0.0625
   96/60000 [..............................] - ETA: 3:42 - loss: 2.3068 - categorical_accuracy: 0.0729
  128/60000 [..............................] - ETA: 3:10 - loss: 2.2760 - categorical_accuracy: 0.1250
  160/60000 [..............................] - ETA: 2:51 - loss: 2.2172 - categorical_accuracy: 0.1937
  192/60000 [..............................] - ETA: 2:38 - loss: 2.2122 - categorical_accuracy: 0.2083
  224/60000 [..............................] - ETA: 2:30 - loss: 2.1819 - categorical_accuracy: 0.2277
  256/60000 [..............................] - ETA: 2:24 - loss: 2.1295 - categorical_accuracy: 0.2500
  288/60000 [..............................] - ETA: 2:18 - loss: 2.1491 - categorical_accuracy: 0.2500
  320/60000 [..............................] - ETA: 2:15 - loss: 2.1210 - categorical_accuracy: 0.2594
  352/60000 [..............................] - ETA: 2:11 - loss: 2.0832 - categorical_accuracy: 0.2784
  384/60000 [..............................] - ETA: 2:09 - loss: 2.0583 - categorical_accuracy: 0.2786
  416/60000 [..............................] - ETA: 2:07 - loss: 2.0213 - categorical_accuracy: 0.2885
  448/60000 [..............................] - ETA: 2:05 - loss: 1.9800 - categorical_accuracy: 0.3013
  480/60000 [..............................] - ETA: 2:04 - loss: 1.9612 - categorical_accuracy: 0.3083
  512/60000 [..............................] - ETA: 2:03 - loss: 1.9381 - categorical_accuracy: 0.3145
  544/60000 [..............................] - ETA: 2:02 - loss: 1.9016 - categorical_accuracy: 0.3272
  576/60000 [..............................] - ETA: 2:00 - loss: 1.8852 - categorical_accuracy: 0.3299
  608/60000 [..............................] - ETA: 1:59 - loss: 1.8568 - categorical_accuracy: 0.3438
  640/60000 [..............................] - ETA: 1:58 - loss: 1.8314 - categorical_accuracy: 0.3609
  672/60000 [..............................] - ETA: 1:57 - loss: 1.7981 - categorical_accuracy: 0.3720
  704/60000 [..............................] - ETA: 1:56 - loss: 1.7852 - categorical_accuracy: 0.3793
  736/60000 [..............................] - ETA: 1:56 - loss: 1.7799 - categorical_accuracy: 0.3832
  768/60000 [..............................] - ETA: 1:56 - loss: 1.7597 - categorical_accuracy: 0.3919
  800/60000 [..............................] - ETA: 1:55 - loss: 1.7340 - categorical_accuracy: 0.4013
  864/60000 [..............................] - ETA: 1:53 - loss: 1.6652 - categorical_accuracy: 0.4282
  896/60000 [..............................] - ETA: 1:52 - loss: 1.6378 - categorical_accuracy: 0.4375
  928/60000 [..............................] - ETA: 1:52 - loss: 1.6191 - categorical_accuracy: 0.4429
  960/60000 [..............................] - ETA: 1:51 - loss: 1.5943 - categorical_accuracy: 0.4521
  992/60000 [..............................] - ETA: 1:51 - loss: 1.5664 - categorical_accuracy: 0.4627
 1024/60000 [..............................] - ETA: 1:50 - loss: 1.5473 - categorical_accuracy: 0.4717
 1056/60000 [..............................] - ETA: 1:50 - loss: 1.5333 - categorical_accuracy: 0.4754
 1088/60000 [..............................] - ETA: 1:49 - loss: 1.5180 - categorical_accuracy: 0.4807
 1120/60000 [..............................] - ETA: 1:49 - loss: 1.4936 - categorical_accuracy: 0.4866
 1152/60000 [..............................] - ETA: 1:49 - loss: 1.4710 - categorical_accuracy: 0.4948
 1184/60000 [..............................] - ETA: 1:48 - loss: 1.4702 - categorical_accuracy: 0.4975
 1216/60000 [..............................] - ETA: 1:48 - loss: 1.4551 - categorical_accuracy: 0.5016
 1248/60000 [..............................] - ETA: 1:48 - loss: 1.4410 - categorical_accuracy: 0.5048
 1280/60000 [..............................] - ETA: 1:47 - loss: 1.4378 - categorical_accuracy: 0.5055
 1312/60000 [..............................] - ETA: 1:47 - loss: 1.4277 - categorical_accuracy: 0.5084
 1344/60000 [..............................] - ETA: 1:47 - loss: 1.4176 - categorical_accuracy: 0.5126
 1376/60000 [..............................] - ETA: 1:47 - loss: 1.3988 - categorical_accuracy: 0.5196
 1440/60000 [..............................] - ETA: 1:46 - loss: 1.3686 - categorical_accuracy: 0.5306
 1472/60000 [..............................] - ETA: 1:46 - loss: 1.3513 - categorical_accuracy: 0.5367
 1504/60000 [..............................] - ETA: 1:46 - loss: 1.3390 - categorical_accuracy: 0.5392
 1536/60000 [..............................] - ETA: 1:47 - loss: 1.3227 - categorical_accuracy: 0.5430
 1568/60000 [..............................] - ETA: 1:47 - loss: 1.3045 - categorical_accuracy: 0.5491
 1600/60000 [..............................] - ETA: 1:47 - loss: 1.2969 - categorical_accuracy: 0.5525
 1632/60000 [..............................] - ETA: 1:46 - loss: 1.2877 - categorical_accuracy: 0.5558
 1664/60000 [..............................] - ETA: 1:46 - loss: 1.2750 - categorical_accuracy: 0.5613
 1696/60000 [..............................] - ETA: 1:46 - loss: 1.2676 - categorical_accuracy: 0.5625
 1760/60000 [..............................] - ETA: 1:46 - loss: 1.2496 - categorical_accuracy: 0.5710
 1792/60000 [..............................] - ETA: 1:45 - loss: 1.2385 - categorical_accuracy: 0.5748
 1856/60000 [..............................] - ETA: 1:45 - loss: 1.2160 - categorical_accuracy: 0.5814
 1920/60000 [..............................] - ETA: 1:44 - loss: 1.1957 - categorical_accuracy: 0.5891
 1952/60000 [..............................] - ETA: 1:44 - loss: 1.1865 - categorical_accuracy: 0.5932
 1984/60000 [..............................] - ETA: 1:44 - loss: 1.1786 - categorical_accuracy: 0.5963
 2016/60000 [>.............................] - ETA: 1:43 - loss: 1.1695 - categorical_accuracy: 0.5992
 2048/60000 [>.............................] - ETA: 1:43 - loss: 1.1659 - categorical_accuracy: 0.6006
 2080/60000 [>.............................] - ETA: 1:43 - loss: 1.1595 - categorical_accuracy: 0.6024
 2112/60000 [>.............................] - ETA: 1:43 - loss: 1.1543 - categorical_accuracy: 0.6056
 2144/60000 [>.............................] - ETA: 1:43 - loss: 1.1435 - categorical_accuracy: 0.6091
 2176/60000 [>.............................] - ETA: 1:43 - loss: 1.1360 - categorical_accuracy: 0.6126
 2208/60000 [>.............................] - ETA: 1:43 - loss: 1.1277 - categorical_accuracy: 0.6155
 2240/60000 [>.............................] - ETA: 1:43 - loss: 1.1184 - categorical_accuracy: 0.6196
 2272/60000 [>.............................] - ETA: 1:43 - loss: 1.1064 - categorical_accuracy: 0.6241
 2304/60000 [>.............................] - ETA: 1:43 - loss: 1.0998 - categorical_accuracy: 0.6267
 2336/60000 [>.............................] - ETA: 1:42 - loss: 1.1049 - categorical_accuracy: 0.6276
 2368/60000 [>.............................] - ETA: 1:42 - loss: 1.1012 - categorical_accuracy: 0.6292
 2432/60000 [>.............................] - ETA: 1:42 - loss: 1.0873 - categorical_accuracy: 0.6340
 2496/60000 [>.............................] - ETA: 1:41 - loss: 1.0714 - categorical_accuracy: 0.6398
 2528/60000 [>.............................] - ETA: 1:41 - loss: 1.0642 - categorical_accuracy: 0.6420
 2560/60000 [>.............................] - ETA: 1:41 - loss: 1.0560 - categorical_accuracy: 0.6441
 2592/60000 [>.............................] - ETA: 1:41 - loss: 1.0487 - categorical_accuracy: 0.6470
 2624/60000 [>.............................] - ETA: 1:41 - loss: 1.0434 - categorical_accuracy: 0.6482
 2656/60000 [>.............................] - ETA: 1:41 - loss: 1.0356 - categorical_accuracy: 0.6514
 2688/60000 [>.............................] - ETA: 1:41 - loss: 1.0261 - categorical_accuracy: 0.6551
 2720/60000 [>.............................] - ETA: 1:40 - loss: 1.0220 - categorical_accuracy: 0.6570
 2752/60000 [>.............................] - ETA: 1:40 - loss: 1.0140 - categorical_accuracy: 0.6595
 2784/60000 [>.............................] - ETA: 1:40 - loss: 1.0088 - categorical_accuracy: 0.6613
 2816/60000 [>.............................] - ETA: 1:40 - loss: 1.0003 - categorical_accuracy: 0.6641
 2848/60000 [>.............................] - ETA: 1:40 - loss: 0.9981 - categorical_accuracy: 0.6643
 2880/60000 [>.............................] - ETA: 1:40 - loss: 0.9927 - categorical_accuracy: 0.6663
 2912/60000 [>.............................] - ETA: 1:40 - loss: 0.9887 - categorical_accuracy: 0.6672
 2944/60000 [>.............................] - ETA: 1:40 - loss: 0.9854 - categorical_accuracy: 0.6685
 2976/60000 [>.............................] - ETA: 1:40 - loss: 0.9811 - categorical_accuracy: 0.6700
 3008/60000 [>.............................] - ETA: 1:40 - loss: 0.9766 - categorical_accuracy: 0.6715
 3040/60000 [>.............................] - ETA: 1:40 - loss: 0.9700 - categorical_accuracy: 0.6737
 3072/60000 [>.............................] - ETA: 1:39 - loss: 0.9648 - categorical_accuracy: 0.6751
 3104/60000 [>.............................] - ETA: 1:39 - loss: 0.9596 - categorical_accuracy: 0.6775
 3136/60000 [>.............................] - ETA: 1:39 - loss: 0.9520 - categorical_accuracy: 0.6798
 3168/60000 [>.............................] - ETA: 1:40 - loss: 0.9453 - categorical_accuracy: 0.6821
 3200/60000 [>.............................] - ETA: 1:39 - loss: 0.9398 - categorical_accuracy: 0.6837
 3264/60000 [>.............................] - ETA: 1:39 - loss: 0.9289 - categorical_accuracy: 0.6875
 3296/60000 [>.............................] - ETA: 1:39 - loss: 0.9236 - categorical_accuracy: 0.6887
 3328/60000 [>.............................] - ETA: 1:39 - loss: 0.9213 - categorical_accuracy: 0.6905
 3360/60000 [>.............................] - ETA: 1:39 - loss: 0.9162 - categorical_accuracy: 0.6920
 3392/60000 [>.............................] - ETA: 1:39 - loss: 0.9136 - categorical_accuracy: 0.6934
 3424/60000 [>.............................] - ETA: 1:39 - loss: 0.9102 - categorical_accuracy: 0.6948
 3456/60000 [>.............................] - ETA: 1:39 - loss: 0.9046 - categorical_accuracy: 0.6970
 3488/60000 [>.............................] - ETA: 1:39 - loss: 0.9019 - categorical_accuracy: 0.6981
 3520/60000 [>.............................] - ETA: 1:39 - loss: 0.8971 - categorical_accuracy: 0.6991
 3552/60000 [>.............................] - ETA: 1:39 - loss: 0.8932 - categorical_accuracy: 0.7002
 3584/60000 [>.............................] - ETA: 1:38 - loss: 0.8905 - categorical_accuracy: 0.7012
 3648/60000 [>.............................] - ETA: 1:38 - loss: 0.8800 - categorical_accuracy: 0.7053
 3712/60000 [>.............................] - ETA: 1:38 - loss: 0.8711 - categorical_accuracy: 0.7082
 3744/60000 [>.............................] - ETA: 1:38 - loss: 0.8663 - categorical_accuracy: 0.7099
 3776/60000 [>.............................] - ETA: 1:38 - loss: 0.8618 - categorical_accuracy: 0.7116
 3808/60000 [>.............................] - ETA: 1:38 - loss: 0.8576 - categorical_accuracy: 0.7135
 3840/60000 [>.............................] - ETA: 1:38 - loss: 0.8532 - categorical_accuracy: 0.7151
 3872/60000 [>.............................] - ETA: 1:37 - loss: 0.8500 - categorical_accuracy: 0.7164
 3904/60000 [>.............................] - ETA: 1:37 - loss: 0.8451 - categorical_accuracy: 0.7182
 3936/60000 [>.............................] - ETA: 1:37 - loss: 0.8420 - categorical_accuracy: 0.7190
 3968/60000 [>.............................] - ETA: 1:37 - loss: 0.8380 - categorical_accuracy: 0.7205
 4000/60000 [=>............................] - ETA: 1:37 - loss: 0.8350 - categorical_accuracy: 0.7220
 4032/60000 [=>............................] - ETA: 1:37 - loss: 0.8325 - categorical_accuracy: 0.7230
 4064/60000 [=>............................] - ETA: 1:37 - loss: 0.8277 - categorical_accuracy: 0.7244
 4096/60000 [=>............................] - ETA: 1:37 - loss: 0.8250 - categorical_accuracy: 0.7253
 4128/60000 [=>............................] - ETA: 1:37 - loss: 0.8215 - categorical_accuracy: 0.7263
 4160/60000 [=>............................] - ETA: 1:37 - loss: 0.8165 - categorical_accuracy: 0.7284
 4192/60000 [=>............................] - ETA: 1:37 - loss: 0.8112 - categorical_accuracy: 0.7302
 4224/60000 [=>............................] - ETA: 1:37 - loss: 0.8073 - categorical_accuracy: 0.7315
 4256/60000 [=>............................] - ETA: 1:37 - loss: 0.8035 - categorical_accuracy: 0.7331
 4288/60000 [=>............................] - ETA: 1:36 - loss: 0.8000 - categorical_accuracy: 0.7346
 4320/60000 [=>............................] - ETA: 1:36 - loss: 0.7951 - categorical_accuracy: 0.7363
 4352/60000 [=>............................] - ETA: 1:36 - loss: 0.7915 - categorical_accuracy: 0.7381
 4384/60000 [=>............................] - ETA: 1:36 - loss: 0.7909 - categorical_accuracy: 0.7388
 4448/60000 [=>............................] - ETA: 1:36 - loss: 0.7835 - categorical_accuracy: 0.7415
 4480/60000 [=>............................] - ETA: 1:36 - loss: 0.7800 - categorical_accuracy: 0.7426
 4544/60000 [=>............................] - ETA: 1:36 - loss: 0.7723 - categorical_accuracy: 0.7452
 4576/60000 [=>............................] - ETA: 1:36 - loss: 0.7696 - categorical_accuracy: 0.7458
 4608/60000 [=>............................] - ETA: 1:35 - loss: 0.7657 - categorical_accuracy: 0.7472
 4640/60000 [=>............................] - ETA: 1:35 - loss: 0.7628 - categorical_accuracy: 0.7481
 4672/60000 [=>............................] - ETA: 1:35 - loss: 0.7584 - categorical_accuracy: 0.7498
 4704/60000 [=>............................] - ETA: 1:35 - loss: 0.7552 - categorical_accuracy: 0.7511
 4736/60000 [=>............................] - ETA: 1:35 - loss: 0.7528 - categorical_accuracy: 0.7519
 4768/60000 [=>............................] - ETA: 1:35 - loss: 0.7514 - categorical_accuracy: 0.7519
 4800/60000 [=>............................] - ETA: 1:35 - loss: 0.7470 - categorical_accuracy: 0.7533
 4832/60000 [=>............................] - ETA: 1:35 - loss: 0.7435 - categorical_accuracy: 0.7541
 4896/60000 [=>............................] - ETA: 1:35 - loss: 0.7360 - categorical_accuracy: 0.7565
 4928/60000 [=>............................] - ETA: 1:35 - loss: 0.7322 - categorical_accuracy: 0.7577
 4960/60000 [=>............................] - ETA: 1:35 - loss: 0.7279 - categorical_accuracy: 0.7591
 4992/60000 [=>............................] - ETA: 1:35 - loss: 0.7253 - categorical_accuracy: 0.7598
 5024/60000 [=>............................] - ETA: 1:34 - loss: 0.7241 - categorical_accuracy: 0.7607
 5056/60000 [=>............................] - ETA: 1:34 - loss: 0.7215 - categorical_accuracy: 0.7611
 5088/60000 [=>............................] - ETA: 1:34 - loss: 0.7178 - categorical_accuracy: 0.7624
 5120/60000 [=>............................] - ETA: 1:34 - loss: 0.7153 - categorical_accuracy: 0.7635
 5152/60000 [=>............................] - ETA: 1:34 - loss: 0.7128 - categorical_accuracy: 0.7642
 5184/60000 [=>............................] - ETA: 1:34 - loss: 0.7099 - categorical_accuracy: 0.7652
 5216/60000 [=>............................] - ETA: 1:34 - loss: 0.7060 - categorical_accuracy: 0.7667
 5248/60000 [=>............................] - ETA: 1:34 - loss: 0.7039 - categorical_accuracy: 0.7677
 5280/60000 [=>............................] - ETA: 1:34 - loss: 0.7015 - categorical_accuracy: 0.7686
 5312/60000 [=>............................] - ETA: 1:34 - loss: 0.6992 - categorical_accuracy: 0.7696
 5344/60000 [=>............................] - ETA: 1:34 - loss: 0.6964 - categorical_accuracy: 0.7704
 5376/60000 [=>............................] - ETA: 1:34 - loss: 0.6940 - categorical_accuracy: 0.7712
 5408/60000 [=>............................] - ETA: 1:34 - loss: 0.6921 - categorical_accuracy: 0.7722
 5440/60000 [=>............................] - ETA: 1:34 - loss: 0.6897 - categorical_accuracy: 0.7732
 5472/60000 [=>............................] - ETA: 1:34 - loss: 0.6875 - categorical_accuracy: 0.7739
 5504/60000 [=>............................] - ETA: 1:34 - loss: 0.6846 - categorical_accuracy: 0.7749
 5536/60000 [=>............................] - ETA: 1:34 - loss: 0.6817 - categorical_accuracy: 0.7758
 5568/60000 [=>............................] - ETA: 1:34 - loss: 0.6815 - categorical_accuracy: 0.7762
 5600/60000 [=>............................] - ETA: 1:34 - loss: 0.6804 - categorical_accuracy: 0.7764
 5632/60000 [=>............................] - ETA: 1:34 - loss: 0.6784 - categorical_accuracy: 0.7773
 5664/60000 [=>............................] - ETA: 1:34 - loss: 0.6778 - categorical_accuracy: 0.7777
 5696/60000 [=>............................] - ETA: 1:34 - loss: 0.6753 - categorical_accuracy: 0.7784
 5728/60000 [=>............................] - ETA: 1:34 - loss: 0.6747 - categorical_accuracy: 0.7792
 5760/60000 [=>............................] - ETA: 1:34 - loss: 0.6736 - categorical_accuracy: 0.7797
 5792/60000 [=>............................] - ETA: 1:34 - loss: 0.6706 - categorical_accuracy: 0.7807
 5824/60000 [=>............................] - ETA: 1:33 - loss: 0.6688 - categorical_accuracy: 0.7812
 5888/60000 [=>............................] - ETA: 1:33 - loss: 0.6636 - categorical_accuracy: 0.7829
 5920/60000 [=>............................] - ETA: 1:33 - loss: 0.6607 - categorical_accuracy: 0.7838
 5984/60000 [=>............................] - ETA: 1:33 - loss: 0.6551 - categorical_accuracy: 0.7858
 6016/60000 [==>...........................] - ETA: 1:33 - loss: 0.6527 - categorical_accuracy: 0.7866
 6048/60000 [==>...........................] - ETA: 1:33 - loss: 0.6500 - categorical_accuracy: 0.7872
 6080/60000 [==>...........................] - ETA: 1:33 - loss: 0.6475 - categorical_accuracy: 0.7880
 6112/60000 [==>...........................] - ETA: 1:33 - loss: 0.6462 - categorical_accuracy: 0.7884
 6144/60000 [==>...........................] - ETA: 1:33 - loss: 0.6438 - categorical_accuracy: 0.7892
 6176/60000 [==>...........................] - ETA: 1:33 - loss: 0.6433 - categorical_accuracy: 0.7897
 6208/60000 [==>...........................] - ETA: 1:33 - loss: 0.6416 - categorical_accuracy: 0.7904
 6240/60000 [==>...........................] - ETA: 1:33 - loss: 0.6395 - categorical_accuracy: 0.7910
 6272/60000 [==>...........................] - ETA: 1:33 - loss: 0.6375 - categorical_accuracy: 0.7916
 6304/60000 [==>...........................] - ETA: 1:33 - loss: 0.6353 - categorical_accuracy: 0.7924
 6336/60000 [==>...........................] - ETA: 1:32 - loss: 0.6334 - categorical_accuracy: 0.7931
 6400/60000 [==>...........................] - ETA: 1:32 - loss: 0.6296 - categorical_accuracy: 0.7944
 6432/60000 [==>...........................] - ETA: 1:32 - loss: 0.6276 - categorical_accuracy: 0.7949
 6464/60000 [==>...........................] - ETA: 1:32 - loss: 0.6273 - categorical_accuracy: 0.7952
 6496/60000 [==>...........................] - ETA: 1:32 - loss: 0.6257 - categorical_accuracy: 0.7956
 6528/60000 [==>...........................] - ETA: 1:32 - loss: 0.6258 - categorical_accuracy: 0.7960
 6560/60000 [==>...........................] - ETA: 1:32 - loss: 0.6243 - categorical_accuracy: 0.7966
 6592/60000 [==>...........................] - ETA: 1:32 - loss: 0.6221 - categorical_accuracy: 0.7975
 6624/60000 [==>...........................] - ETA: 1:32 - loss: 0.6207 - categorical_accuracy: 0.7980
 6656/60000 [==>...........................] - ETA: 1:32 - loss: 0.6195 - categorical_accuracy: 0.7984
 6688/60000 [==>...........................] - ETA: 1:32 - loss: 0.6214 - categorical_accuracy: 0.7983
 6720/60000 [==>...........................] - ETA: 1:32 - loss: 0.6190 - categorical_accuracy: 0.7991
 6752/60000 [==>...........................] - ETA: 1:32 - loss: 0.6176 - categorical_accuracy: 0.7995
 6784/60000 [==>...........................] - ETA: 1:32 - loss: 0.6159 - categorical_accuracy: 0.8000
 6816/60000 [==>...........................] - ETA: 1:31 - loss: 0.6139 - categorical_accuracy: 0.8005
 6848/60000 [==>...........................] - ETA: 1:31 - loss: 0.6123 - categorical_accuracy: 0.8010
 6880/60000 [==>...........................] - ETA: 1:31 - loss: 0.6099 - categorical_accuracy: 0.8017
 6912/60000 [==>...........................] - ETA: 1:31 - loss: 0.6086 - categorical_accuracy: 0.8021
 6944/60000 [==>...........................] - ETA: 1:31 - loss: 0.6067 - categorical_accuracy: 0.8029
 6976/60000 [==>...........................] - ETA: 1:31 - loss: 0.6052 - categorical_accuracy: 0.8030
 7008/60000 [==>...........................] - ETA: 1:31 - loss: 0.6041 - categorical_accuracy: 0.8032
 7072/60000 [==>...........................] - ETA: 1:31 - loss: 0.6001 - categorical_accuracy: 0.8047
 7104/60000 [==>...........................] - ETA: 1:31 - loss: 0.6004 - categorical_accuracy: 0.8046
 7136/60000 [==>...........................] - ETA: 1:31 - loss: 0.5989 - categorical_accuracy: 0.8051
 7168/60000 [==>...........................] - ETA: 1:31 - loss: 0.5986 - categorical_accuracy: 0.8052
 7200/60000 [==>...........................] - ETA: 1:31 - loss: 0.5975 - categorical_accuracy: 0.8056
 7232/60000 [==>...........................] - ETA: 1:31 - loss: 0.5973 - categorical_accuracy: 0.8060
 7264/60000 [==>...........................] - ETA: 1:31 - loss: 0.5953 - categorical_accuracy: 0.8067
 7296/60000 [==>...........................] - ETA: 1:31 - loss: 0.5944 - categorical_accuracy: 0.8072
 7328/60000 [==>...........................] - ETA: 1:30 - loss: 0.5928 - categorical_accuracy: 0.8076
 7360/60000 [==>...........................] - ETA: 1:30 - loss: 0.5909 - categorical_accuracy: 0.8083
 7392/60000 [==>...........................] - ETA: 1:30 - loss: 0.5902 - categorical_accuracy: 0.8084
 7424/60000 [==>...........................] - ETA: 1:30 - loss: 0.5882 - categorical_accuracy: 0.8093
 7456/60000 [==>...........................] - ETA: 1:30 - loss: 0.5869 - categorical_accuracy: 0.8097
 7488/60000 [==>...........................] - ETA: 1:30 - loss: 0.5848 - categorical_accuracy: 0.8105
 7520/60000 [==>...........................] - ETA: 1:30 - loss: 0.5833 - categorical_accuracy: 0.8110
 7584/60000 [==>...........................] - ETA: 1:30 - loss: 0.5819 - categorical_accuracy: 0.8116
 7616/60000 [==>...........................] - ETA: 1:30 - loss: 0.5800 - categorical_accuracy: 0.8122
 7680/60000 [==>...........................] - ETA: 1:30 - loss: 0.5761 - categorical_accuracy: 0.8137
 7744/60000 [==>...........................] - ETA: 1:29 - loss: 0.5735 - categorical_accuracy: 0.8147
 7776/60000 [==>...........................] - ETA: 1:29 - loss: 0.5725 - categorical_accuracy: 0.8149
 7808/60000 [==>...........................] - ETA: 1:29 - loss: 0.5714 - categorical_accuracy: 0.8152
 7840/60000 [==>...........................] - ETA: 1:29 - loss: 0.5705 - categorical_accuracy: 0.8156
 7872/60000 [==>...........................] - ETA: 1:29 - loss: 0.5687 - categorical_accuracy: 0.8162
 7904/60000 [==>...........................] - ETA: 1:29 - loss: 0.5672 - categorical_accuracy: 0.8168
 7936/60000 [==>...........................] - ETA: 1:29 - loss: 0.5654 - categorical_accuracy: 0.8174
 7968/60000 [==>...........................] - ETA: 1:29 - loss: 0.5635 - categorical_accuracy: 0.8180
 8000/60000 [===>..........................] - ETA: 1:29 - loss: 0.5621 - categorical_accuracy: 0.8185
 8032/60000 [===>..........................] - ETA: 1:29 - loss: 0.5605 - categorical_accuracy: 0.8191
 8064/60000 [===>..........................] - ETA: 1:29 - loss: 0.5588 - categorical_accuracy: 0.8198
 8128/60000 [===>..........................] - ETA: 1:29 - loss: 0.5564 - categorical_accuracy: 0.8206
 8192/60000 [===>..........................] - ETA: 1:28 - loss: 0.5536 - categorical_accuracy: 0.8214
 8224/60000 [===>..........................] - ETA: 1:28 - loss: 0.5522 - categorical_accuracy: 0.8219
 8256/60000 [===>..........................] - ETA: 1:28 - loss: 0.5513 - categorical_accuracy: 0.8222
 8320/60000 [===>..........................] - ETA: 1:28 - loss: 0.5492 - categorical_accuracy: 0.8228
 8384/60000 [===>..........................] - ETA: 1:28 - loss: 0.5463 - categorical_accuracy: 0.8238
 8416/60000 [===>..........................] - ETA: 1:28 - loss: 0.5451 - categorical_accuracy: 0.8243
 8448/60000 [===>..........................] - ETA: 1:28 - loss: 0.5432 - categorical_accuracy: 0.8249
 8480/60000 [===>..........................] - ETA: 1:28 - loss: 0.5420 - categorical_accuracy: 0.8254
 8512/60000 [===>..........................] - ETA: 1:28 - loss: 0.5409 - categorical_accuracy: 0.8257
 8544/60000 [===>..........................] - ETA: 1:28 - loss: 0.5393 - categorical_accuracy: 0.8262
 8576/60000 [===>..........................] - ETA: 1:28 - loss: 0.5391 - categorical_accuracy: 0.8264
 8608/60000 [===>..........................] - ETA: 1:28 - loss: 0.5379 - categorical_accuracy: 0.8267
 8640/60000 [===>..........................] - ETA: 1:28 - loss: 0.5366 - categorical_accuracy: 0.8271
 8672/60000 [===>..........................] - ETA: 1:27 - loss: 0.5353 - categorical_accuracy: 0.8274
 8704/60000 [===>..........................] - ETA: 1:27 - loss: 0.5340 - categorical_accuracy: 0.8277
 8736/60000 [===>..........................] - ETA: 1:27 - loss: 0.5331 - categorical_accuracy: 0.8280
 8768/60000 [===>..........................] - ETA: 1:27 - loss: 0.5322 - categorical_accuracy: 0.8282
 8800/60000 [===>..........................] - ETA: 1:27 - loss: 0.5309 - categorical_accuracy: 0.8286
 8832/60000 [===>..........................] - ETA: 1:27 - loss: 0.5295 - categorical_accuracy: 0.8290
 8864/60000 [===>..........................] - ETA: 1:27 - loss: 0.5279 - categorical_accuracy: 0.8295
 8896/60000 [===>..........................] - ETA: 1:27 - loss: 0.5265 - categorical_accuracy: 0.8300
 8928/60000 [===>..........................] - ETA: 1:27 - loss: 0.5253 - categorical_accuracy: 0.8304
 8960/60000 [===>..........................] - ETA: 1:27 - loss: 0.5250 - categorical_accuracy: 0.8308
 8992/60000 [===>..........................] - ETA: 1:27 - loss: 0.5238 - categorical_accuracy: 0.8312
 9024/60000 [===>..........................] - ETA: 1:27 - loss: 0.5238 - categorical_accuracy: 0.8313
 9056/60000 [===>..........................] - ETA: 1:27 - loss: 0.5239 - categorical_accuracy: 0.8313
 9088/60000 [===>..........................] - ETA: 1:27 - loss: 0.5233 - categorical_accuracy: 0.8315
 9120/60000 [===>..........................] - ETA: 1:27 - loss: 0.5231 - categorical_accuracy: 0.8314
 9152/60000 [===>..........................] - ETA: 1:27 - loss: 0.5228 - categorical_accuracy: 0.8316
 9184/60000 [===>..........................] - ETA: 1:27 - loss: 0.5223 - categorical_accuracy: 0.8319
 9216/60000 [===>..........................] - ETA: 1:27 - loss: 0.5211 - categorical_accuracy: 0.8322
 9248/60000 [===>..........................] - ETA: 1:27 - loss: 0.5204 - categorical_accuracy: 0.8323
 9280/60000 [===>..........................] - ETA: 1:27 - loss: 0.5190 - categorical_accuracy: 0.8328
 9312/60000 [===>..........................] - ETA: 1:26 - loss: 0.5180 - categorical_accuracy: 0.8330
 9344/60000 [===>..........................] - ETA: 1:26 - loss: 0.5181 - categorical_accuracy: 0.8332
 9408/60000 [===>..........................] - ETA: 1:26 - loss: 0.5158 - categorical_accuracy: 0.8339
 9440/60000 [===>..........................] - ETA: 1:26 - loss: 0.5150 - categorical_accuracy: 0.8342
 9472/60000 [===>..........................] - ETA: 1:26 - loss: 0.5138 - categorical_accuracy: 0.8347
 9536/60000 [===>..........................] - ETA: 1:26 - loss: 0.5124 - categorical_accuracy: 0.8352
 9600/60000 [===>..........................] - ETA: 1:26 - loss: 0.5100 - categorical_accuracy: 0.8359
 9664/60000 [===>..........................] - ETA: 1:26 - loss: 0.5086 - categorical_accuracy: 0.8364
 9728/60000 [===>..........................] - ETA: 1:25 - loss: 0.5073 - categorical_accuracy: 0.8368
 9760/60000 [===>..........................] - ETA: 1:25 - loss: 0.5067 - categorical_accuracy: 0.8370
 9792/60000 [===>..........................] - ETA: 1:25 - loss: 0.5055 - categorical_accuracy: 0.8374
 9824/60000 [===>..........................] - ETA: 1:25 - loss: 0.5045 - categorical_accuracy: 0.8376
 9856/60000 [===>..........................] - ETA: 1:25 - loss: 0.5036 - categorical_accuracy: 0.8380
 9888/60000 [===>..........................] - ETA: 1:25 - loss: 0.5027 - categorical_accuracy: 0.8383
 9920/60000 [===>..........................] - ETA: 1:25 - loss: 0.5021 - categorical_accuracy: 0.8386
 9952/60000 [===>..........................] - ETA: 1:25 - loss: 0.5015 - categorical_accuracy: 0.8387
 9984/60000 [===>..........................] - ETA: 1:25 - loss: 0.5012 - categorical_accuracy: 0.8387
10016/60000 [====>.........................] - ETA: 1:25 - loss: 0.5002 - categorical_accuracy: 0.8392
10048/60000 [====>.........................] - ETA: 1:25 - loss: 0.4994 - categorical_accuracy: 0.8394
10080/60000 [====>.........................] - ETA: 1:25 - loss: 0.4985 - categorical_accuracy: 0.8396
10112/60000 [====>.........................] - ETA: 1:25 - loss: 0.4977 - categorical_accuracy: 0.8399
10144/60000 [====>.........................] - ETA: 1:25 - loss: 0.4967 - categorical_accuracy: 0.8402
10176/60000 [====>.........................] - ETA: 1:25 - loss: 0.4957 - categorical_accuracy: 0.8406
10208/60000 [====>.........................] - ETA: 1:25 - loss: 0.4943 - categorical_accuracy: 0.8410
10272/60000 [====>.........................] - ETA: 1:24 - loss: 0.4922 - categorical_accuracy: 0.8417
10304/60000 [====>.........................] - ETA: 1:24 - loss: 0.4924 - categorical_accuracy: 0.8417
10336/60000 [====>.........................] - ETA: 1:24 - loss: 0.4915 - categorical_accuracy: 0.8420
10368/60000 [====>.........................] - ETA: 1:24 - loss: 0.4909 - categorical_accuracy: 0.8422
10400/60000 [====>.........................] - ETA: 1:24 - loss: 0.4908 - categorical_accuracy: 0.8424
10432/60000 [====>.........................] - ETA: 1:24 - loss: 0.4906 - categorical_accuracy: 0.8424
10464/60000 [====>.........................] - ETA: 1:24 - loss: 0.4895 - categorical_accuracy: 0.8428
10496/60000 [====>.........................] - ETA: 1:24 - loss: 0.4886 - categorical_accuracy: 0.8430
10528/60000 [====>.........................] - ETA: 1:24 - loss: 0.4885 - categorical_accuracy: 0.8432
10560/60000 [====>.........................] - ETA: 1:24 - loss: 0.4875 - categorical_accuracy: 0.8435
10624/60000 [====>.........................] - ETA: 1:24 - loss: 0.4852 - categorical_accuracy: 0.8443
10656/60000 [====>.........................] - ETA: 1:24 - loss: 0.4839 - categorical_accuracy: 0.8448
10688/60000 [====>.........................] - ETA: 1:24 - loss: 0.4831 - categorical_accuracy: 0.8452
10720/60000 [====>.........................] - ETA: 1:24 - loss: 0.4826 - categorical_accuracy: 0.8455
10752/60000 [====>.........................] - ETA: 1:24 - loss: 0.4819 - categorical_accuracy: 0.8456
10784/60000 [====>.........................] - ETA: 1:24 - loss: 0.4809 - categorical_accuracy: 0.8459
10816/60000 [====>.........................] - ETA: 1:23 - loss: 0.4798 - categorical_accuracy: 0.8462
10848/60000 [====>.........................] - ETA: 1:23 - loss: 0.4788 - categorical_accuracy: 0.8464
10880/60000 [====>.........................] - ETA: 1:23 - loss: 0.4777 - categorical_accuracy: 0.8468
10944/60000 [====>.........................] - ETA: 1:23 - loss: 0.4758 - categorical_accuracy: 0.8471
10976/60000 [====>.........................] - ETA: 1:23 - loss: 0.4747 - categorical_accuracy: 0.8475
11008/60000 [====>.........................] - ETA: 1:23 - loss: 0.4734 - categorical_accuracy: 0.8479
11040/60000 [====>.........................] - ETA: 1:23 - loss: 0.4722 - categorical_accuracy: 0.8484
11072/60000 [====>.........................] - ETA: 1:23 - loss: 0.4711 - categorical_accuracy: 0.8488
11104/60000 [====>.........................] - ETA: 1:23 - loss: 0.4703 - categorical_accuracy: 0.8491
11136/60000 [====>.........................] - ETA: 1:23 - loss: 0.4694 - categorical_accuracy: 0.8492
11168/60000 [====>.........................] - ETA: 1:23 - loss: 0.4684 - categorical_accuracy: 0.8495
11200/60000 [====>.........................] - ETA: 1:23 - loss: 0.4675 - categorical_accuracy: 0.8497
11232/60000 [====>.........................] - ETA: 1:23 - loss: 0.4671 - categorical_accuracy: 0.8501
11264/60000 [====>.........................] - ETA: 1:23 - loss: 0.4661 - categorical_accuracy: 0.8504
11328/60000 [====>.........................] - ETA: 1:22 - loss: 0.4653 - categorical_accuracy: 0.8507
11360/60000 [====>.........................] - ETA: 1:22 - loss: 0.4643 - categorical_accuracy: 0.8511
11424/60000 [====>.........................] - ETA: 1:22 - loss: 0.4623 - categorical_accuracy: 0.8517
11456/60000 [====>.........................] - ETA: 1:22 - loss: 0.4616 - categorical_accuracy: 0.8520
11488/60000 [====>.........................] - ETA: 1:22 - loss: 0.4606 - categorical_accuracy: 0.8524
11520/60000 [====>.........................] - ETA: 1:22 - loss: 0.4596 - categorical_accuracy: 0.8527
11552/60000 [====>.........................] - ETA: 1:22 - loss: 0.4593 - categorical_accuracy: 0.8528
11584/60000 [====>.........................] - ETA: 1:22 - loss: 0.4581 - categorical_accuracy: 0.8532
11616/60000 [====>.........................] - ETA: 1:22 - loss: 0.4571 - categorical_accuracy: 0.8537
11648/60000 [====>.........................] - ETA: 1:22 - loss: 0.4565 - categorical_accuracy: 0.8539
11680/60000 [====>.........................] - ETA: 1:22 - loss: 0.4555 - categorical_accuracy: 0.8542
11712/60000 [====>.........................] - ETA: 1:22 - loss: 0.4547 - categorical_accuracy: 0.8544
11744/60000 [====>.........................] - ETA: 1:22 - loss: 0.4537 - categorical_accuracy: 0.8547
11776/60000 [====>.........................] - ETA: 1:22 - loss: 0.4527 - categorical_accuracy: 0.8550
11808/60000 [====>.........................] - ETA: 1:22 - loss: 0.4522 - categorical_accuracy: 0.8551
11872/60000 [====>.........................] - ETA: 1:21 - loss: 0.4504 - categorical_accuracy: 0.8558
11936/60000 [====>.........................] - ETA: 1:21 - loss: 0.4491 - categorical_accuracy: 0.8563
12000/60000 [=====>........................] - ETA: 1:21 - loss: 0.4471 - categorical_accuracy: 0.8568
12064/60000 [=====>........................] - ETA: 1:21 - loss: 0.4455 - categorical_accuracy: 0.8573
12128/60000 [=====>........................] - ETA: 1:21 - loss: 0.4444 - categorical_accuracy: 0.8576
12160/60000 [=====>........................] - ETA: 1:21 - loss: 0.4437 - categorical_accuracy: 0.8579
12192/60000 [=====>........................] - ETA: 1:21 - loss: 0.4433 - categorical_accuracy: 0.8580
12224/60000 [=====>........................] - ETA: 1:21 - loss: 0.4439 - categorical_accuracy: 0.8581
12256/60000 [=====>........................] - ETA: 1:21 - loss: 0.4431 - categorical_accuracy: 0.8584
12320/60000 [=====>........................] - ETA: 1:20 - loss: 0.4417 - categorical_accuracy: 0.8588
12384/60000 [=====>........................] - ETA: 1:20 - loss: 0.4401 - categorical_accuracy: 0.8594
12416/60000 [=====>........................] - ETA: 1:20 - loss: 0.4393 - categorical_accuracy: 0.8596
12480/60000 [=====>........................] - ETA: 1:20 - loss: 0.4376 - categorical_accuracy: 0.8602
12512/60000 [=====>........................] - ETA: 1:20 - loss: 0.4378 - categorical_accuracy: 0.8601
12544/60000 [=====>........................] - ETA: 1:20 - loss: 0.4374 - categorical_accuracy: 0.8602
12608/60000 [=====>........................] - ETA: 1:20 - loss: 0.4359 - categorical_accuracy: 0.8606
12672/60000 [=====>........................] - ETA: 1:20 - loss: 0.4342 - categorical_accuracy: 0.8612
12736/60000 [=====>........................] - ETA: 1:20 - loss: 0.4330 - categorical_accuracy: 0.8618
12768/60000 [=====>........................] - ETA: 1:19 - loss: 0.4323 - categorical_accuracy: 0.8620
12800/60000 [=====>........................] - ETA: 1:19 - loss: 0.4315 - categorical_accuracy: 0.8623
12832/60000 [=====>........................] - ETA: 1:19 - loss: 0.4313 - categorical_accuracy: 0.8625
12864/60000 [=====>........................] - ETA: 1:19 - loss: 0.4306 - categorical_accuracy: 0.8627
12896/60000 [=====>........................] - ETA: 1:19 - loss: 0.4296 - categorical_accuracy: 0.8631
12928/60000 [=====>........................] - ETA: 1:19 - loss: 0.4290 - categorical_accuracy: 0.8632
12960/60000 [=====>........................] - ETA: 1:19 - loss: 0.4285 - categorical_accuracy: 0.8633
12992/60000 [=====>........................] - ETA: 1:19 - loss: 0.4275 - categorical_accuracy: 0.8637
13024/60000 [=====>........................] - ETA: 1:19 - loss: 0.4268 - categorical_accuracy: 0.8639
13056/60000 [=====>........................] - ETA: 1:19 - loss: 0.4265 - categorical_accuracy: 0.8641
13088/60000 [=====>........................] - ETA: 1:19 - loss: 0.4261 - categorical_accuracy: 0.8642
13120/60000 [=====>........................] - ETA: 1:19 - loss: 0.4252 - categorical_accuracy: 0.8645
13152/60000 [=====>........................] - ETA: 1:19 - loss: 0.4245 - categorical_accuracy: 0.8647
13184/60000 [=====>........................] - ETA: 1:19 - loss: 0.4238 - categorical_accuracy: 0.8648
13216/60000 [=====>........................] - ETA: 1:19 - loss: 0.4232 - categorical_accuracy: 0.8649
13248/60000 [=====>........................] - ETA: 1:19 - loss: 0.4227 - categorical_accuracy: 0.8650
13280/60000 [=====>........................] - ETA: 1:19 - loss: 0.4221 - categorical_accuracy: 0.8651
13312/60000 [=====>........................] - ETA: 1:19 - loss: 0.4215 - categorical_accuracy: 0.8654
13344/60000 [=====>........................] - ETA: 1:18 - loss: 0.4210 - categorical_accuracy: 0.8656
13376/60000 [=====>........................] - ETA: 1:18 - loss: 0.4201 - categorical_accuracy: 0.8659
13408/60000 [=====>........................] - ETA: 1:18 - loss: 0.4208 - categorical_accuracy: 0.8659
13440/60000 [=====>........................] - ETA: 1:18 - loss: 0.4202 - categorical_accuracy: 0.8661
13504/60000 [=====>........................] - ETA: 1:18 - loss: 0.4196 - categorical_accuracy: 0.8664
13568/60000 [=====>........................] - ETA: 1:18 - loss: 0.4182 - categorical_accuracy: 0.8668
13600/60000 [=====>........................] - ETA: 1:18 - loss: 0.4178 - categorical_accuracy: 0.8669
13632/60000 [=====>........................] - ETA: 1:18 - loss: 0.4171 - categorical_accuracy: 0.8671
13664/60000 [=====>........................] - ETA: 1:18 - loss: 0.4163 - categorical_accuracy: 0.8673
13696/60000 [=====>........................] - ETA: 1:18 - loss: 0.4154 - categorical_accuracy: 0.8676
13728/60000 [=====>........................] - ETA: 1:18 - loss: 0.4146 - categorical_accuracy: 0.8679
13760/60000 [=====>........................] - ETA: 1:18 - loss: 0.4146 - categorical_accuracy: 0.8679
13792/60000 [=====>........................] - ETA: 1:18 - loss: 0.4138 - categorical_accuracy: 0.8682
13824/60000 [=====>........................] - ETA: 1:18 - loss: 0.4130 - categorical_accuracy: 0.8685
13856/60000 [=====>........................] - ETA: 1:18 - loss: 0.4123 - categorical_accuracy: 0.8686
13888/60000 [=====>........................] - ETA: 1:18 - loss: 0.4119 - categorical_accuracy: 0.8687
13920/60000 [=====>........................] - ETA: 1:18 - loss: 0.4111 - categorical_accuracy: 0.8690
13952/60000 [=====>........................] - ETA: 1:17 - loss: 0.4106 - categorical_accuracy: 0.8691
13984/60000 [=====>........................] - ETA: 1:17 - loss: 0.4103 - categorical_accuracy: 0.8694
14048/60000 [======>.......................] - ETA: 1:17 - loss: 0.4095 - categorical_accuracy: 0.8697
14112/60000 [======>.......................] - ETA: 1:17 - loss: 0.4086 - categorical_accuracy: 0.8700
14176/60000 [======>.......................] - ETA: 1:17 - loss: 0.4074 - categorical_accuracy: 0.8704
14208/60000 [======>.......................] - ETA: 1:17 - loss: 0.4068 - categorical_accuracy: 0.8706
14272/60000 [======>.......................] - ETA: 1:17 - loss: 0.4056 - categorical_accuracy: 0.8710
14304/60000 [======>.......................] - ETA: 1:17 - loss: 0.4050 - categorical_accuracy: 0.8712
14368/60000 [======>.......................] - ETA: 1:17 - loss: 0.4036 - categorical_accuracy: 0.8717
14400/60000 [======>.......................] - ETA: 1:17 - loss: 0.4029 - categorical_accuracy: 0.8720
14464/60000 [======>.......................] - ETA: 1:16 - loss: 0.4015 - categorical_accuracy: 0.8725
14528/60000 [======>.......................] - ETA: 1:16 - loss: 0.4007 - categorical_accuracy: 0.8728
14560/60000 [======>.......................] - ETA: 1:16 - loss: 0.4004 - categorical_accuracy: 0.8727
14592/60000 [======>.......................] - ETA: 1:16 - loss: 0.3998 - categorical_accuracy: 0.8729
14624/60000 [======>.......................] - ETA: 1:16 - loss: 0.3992 - categorical_accuracy: 0.8730
14656/60000 [======>.......................] - ETA: 1:16 - loss: 0.3988 - categorical_accuracy: 0.8732
14688/60000 [======>.......................] - ETA: 1:16 - loss: 0.3982 - categorical_accuracy: 0.8734
14720/60000 [======>.......................] - ETA: 1:16 - loss: 0.3975 - categorical_accuracy: 0.8736
14784/60000 [======>.......................] - ETA: 1:16 - loss: 0.3965 - categorical_accuracy: 0.8740
14848/60000 [======>.......................] - ETA: 1:16 - loss: 0.3959 - categorical_accuracy: 0.8742
14880/60000 [======>.......................] - ETA: 1:16 - loss: 0.3956 - categorical_accuracy: 0.8743
14944/60000 [======>.......................] - ETA: 1:16 - loss: 0.3948 - categorical_accuracy: 0.8746
14976/60000 [======>.......................] - ETA: 1:15 - loss: 0.3942 - categorical_accuracy: 0.8749
15008/60000 [======>.......................] - ETA: 1:15 - loss: 0.3937 - categorical_accuracy: 0.8751
15072/60000 [======>.......................] - ETA: 1:15 - loss: 0.3922 - categorical_accuracy: 0.8755
15136/60000 [======>.......................] - ETA: 1:15 - loss: 0.3912 - categorical_accuracy: 0.8759
15168/60000 [======>.......................] - ETA: 1:15 - loss: 0.3906 - categorical_accuracy: 0.8761
15200/60000 [======>.......................] - ETA: 1:15 - loss: 0.3902 - categorical_accuracy: 0.8763
15232/60000 [======>.......................] - ETA: 1:15 - loss: 0.3895 - categorical_accuracy: 0.8764
15264/60000 [======>.......................] - ETA: 1:15 - loss: 0.3888 - categorical_accuracy: 0.8767
15296/60000 [======>.......................] - ETA: 1:15 - loss: 0.3889 - categorical_accuracy: 0.8767
15328/60000 [======>.......................] - ETA: 1:15 - loss: 0.3887 - categorical_accuracy: 0.8768
15360/60000 [======>.......................] - ETA: 1:15 - loss: 0.3887 - categorical_accuracy: 0.8770
15392/60000 [======>.......................] - ETA: 1:15 - loss: 0.3883 - categorical_accuracy: 0.8771
15456/60000 [======>.......................] - ETA: 1:15 - loss: 0.3874 - categorical_accuracy: 0.8774
15488/60000 [======>.......................] - ETA: 1:14 - loss: 0.3870 - categorical_accuracy: 0.8776
15520/60000 [======>.......................] - ETA: 1:14 - loss: 0.3866 - categorical_accuracy: 0.8778
15552/60000 [======>.......................] - ETA: 1:14 - loss: 0.3860 - categorical_accuracy: 0.8780
15584/60000 [======>.......................] - ETA: 1:14 - loss: 0.3855 - categorical_accuracy: 0.8781
15616/60000 [======>.......................] - ETA: 1:14 - loss: 0.3850 - categorical_accuracy: 0.8782
15648/60000 [======>.......................] - ETA: 1:14 - loss: 0.3848 - categorical_accuracy: 0.8784
15712/60000 [======>.......................] - ETA: 1:14 - loss: 0.3839 - categorical_accuracy: 0.8787
15744/60000 [======>.......................] - ETA: 1:14 - loss: 0.3836 - categorical_accuracy: 0.8788
15808/60000 [======>.......................] - ETA: 1:14 - loss: 0.3831 - categorical_accuracy: 0.8788
15840/60000 [======>.......................] - ETA: 1:14 - loss: 0.3830 - categorical_accuracy: 0.8789
15872/60000 [======>.......................] - ETA: 1:14 - loss: 0.3826 - categorical_accuracy: 0.8790
15936/60000 [======>.......................] - ETA: 1:14 - loss: 0.3820 - categorical_accuracy: 0.8793
15968/60000 [======>.......................] - ETA: 1:14 - loss: 0.3817 - categorical_accuracy: 0.8794
16000/60000 [=======>......................] - ETA: 1:14 - loss: 0.3820 - categorical_accuracy: 0.8792
16064/60000 [=======>......................] - ETA: 1:13 - loss: 0.3809 - categorical_accuracy: 0.8796
16096/60000 [=======>......................] - ETA: 1:13 - loss: 0.3804 - categorical_accuracy: 0.8798
16128/60000 [=======>......................] - ETA: 1:13 - loss: 0.3799 - categorical_accuracy: 0.8800
16192/60000 [=======>......................] - ETA: 1:13 - loss: 0.3796 - categorical_accuracy: 0.8801
16224/60000 [=======>......................] - ETA: 1:13 - loss: 0.3790 - categorical_accuracy: 0.8803
16288/60000 [=======>......................] - ETA: 1:13 - loss: 0.3781 - categorical_accuracy: 0.8805
16352/60000 [=======>......................] - ETA: 1:13 - loss: 0.3772 - categorical_accuracy: 0.8807
16416/60000 [=======>......................] - ETA: 1:13 - loss: 0.3761 - categorical_accuracy: 0.8811
16448/60000 [=======>......................] - ETA: 1:13 - loss: 0.3755 - categorical_accuracy: 0.8813
16480/60000 [=======>......................] - ETA: 1:13 - loss: 0.3755 - categorical_accuracy: 0.8813
16512/60000 [=======>......................] - ETA: 1:13 - loss: 0.3752 - categorical_accuracy: 0.8814
16544/60000 [=======>......................] - ETA: 1:13 - loss: 0.3746 - categorical_accuracy: 0.8816
16576/60000 [=======>......................] - ETA: 1:12 - loss: 0.3745 - categorical_accuracy: 0.8816
16608/60000 [=======>......................] - ETA: 1:12 - loss: 0.3741 - categorical_accuracy: 0.8818
16640/60000 [=======>......................] - ETA: 1:12 - loss: 0.3735 - categorical_accuracy: 0.8820
16672/60000 [=======>......................] - ETA: 1:12 - loss: 0.3728 - categorical_accuracy: 0.8823
16704/60000 [=======>......................] - ETA: 1:12 - loss: 0.3724 - categorical_accuracy: 0.8824
16768/60000 [=======>......................] - ETA: 1:12 - loss: 0.3715 - categorical_accuracy: 0.8827
16832/60000 [=======>......................] - ETA: 1:12 - loss: 0.3705 - categorical_accuracy: 0.8830
16896/60000 [=======>......................] - ETA: 1:12 - loss: 0.3694 - categorical_accuracy: 0.8833
16960/60000 [=======>......................] - ETA: 1:12 - loss: 0.3685 - categorical_accuracy: 0.8835
17024/60000 [=======>......................] - ETA: 1:12 - loss: 0.3674 - categorical_accuracy: 0.8839
17056/60000 [=======>......................] - ETA: 1:12 - loss: 0.3673 - categorical_accuracy: 0.8839
17088/60000 [=======>......................] - ETA: 1:12 - loss: 0.3670 - categorical_accuracy: 0.8838
17120/60000 [=======>......................] - ETA: 1:11 - loss: 0.3666 - categorical_accuracy: 0.8840
17152/60000 [=======>......................] - ETA: 1:11 - loss: 0.3661 - categorical_accuracy: 0.8841
17184/60000 [=======>......................] - ETA: 1:11 - loss: 0.3655 - categorical_accuracy: 0.8843
17216/60000 [=======>......................] - ETA: 1:11 - loss: 0.3653 - categorical_accuracy: 0.8844
17248/60000 [=======>......................] - ETA: 1:11 - loss: 0.3648 - categorical_accuracy: 0.8846
17280/60000 [=======>......................] - ETA: 1:11 - loss: 0.3649 - categorical_accuracy: 0.8847
17312/60000 [=======>......................] - ETA: 1:11 - loss: 0.3644 - categorical_accuracy: 0.8848
17344/60000 [=======>......................] - ETA: 1:11 - loss: 0.3641 - categorical_accuracy: 0.8849
17376/60000 [=======>......................] - ETA: 1:11 - loss: 0.3636 - categorical_accuracy: 0.8850
17408/60000 [=======>......................] - ETA: 1:11 - loss: 0.3638 - categorical_accuracy: 0.8849
17440/60000 [=======>......................] - ETA: 1:11 - loss: 0.3636 - categorical_accuracy: 0.8849
17472/60000 [=======>......................] - ETA: 1:11 - loss: 0.3633 - categorical_accuracy: 0.8850
17504/60000 [=======>......................] - ETA: 1:11 - loss: 0.3628 - categorical_accuracy: 0.8852
17536/60000 [=======>......................] - ETA: 1:11 - loss: 0.3626 - categorical_accuracy: 0.8853
17568/60000 [=======>......................] - ETA: 1:11 - loss: 0.3621 - categorical_accuracy: 0.8854
17600/60000 [=======>......................] - ETA: 1:11 - loss: 0.3622 - categorical_accuracy: 0.8855
17664/60000 [=======>......................] - ETA: 1:11 - loss: 0.3622 - categorical_accuracy: 0.8854
17728/60000 [=======>......................] - ETA: 1:10 - loss: 0.3621 - categorical_accuracy: 0.8854
17792/60000 [=======>......................] - ETA: 1:10 - loss: 0.3614 - categorical_accuracy: 0.8857
17856/60000 [=======>......................] - ETA: 1:10 - loss: 0.3606 - categorical_accuracy: 0.8859
17888/60000 [=======>......................] - ETA: 1:10 - loss: 0.3607 - categorical_accuracy: 0.8859
17920/60000 [=======>......................] - ETA: 1:10 - loss: 0.3606 - categorical_accuracy: 0.8860
17952/60000 [=======>......................] - ETA: 1:10 - loss: 0.3602 - categorical_accuracy: 0.8861
17984/60000 [=======>......................] - ETA: 1:10 - loss: 0.3601 - categorical_accuracy: 0.8861
18016/60000 [========>.....................] - ETA: 1:10 - loss: 0.3595 - categorical_accuracy: 0.8862
18048/60000 [========>.....................] - ETA: 1:10 - loss: 0.3592 - categorical_accuracy: 0.8863
18080/60000 [========>.....................] - ETA: 1:10 - loss: 0.3588 - categorical_accuracy: 0.8864
18112/60000 [========>.....................] - ETA: 1:10 - loss: 0.3584 - categorical_accuracy: 0.8866
18144/60000 [========>.....................] - ETA: 1:10 - loss: 0.3580 - categorical_accuracy: 0.8867
18176/60000 [========>.....................] - ETA: 1:10 - loss: 0.3579 - categorical_accuracy: 0.8868
18208/60000 [========>.....................] - ETA: 1:10 - loss: 0.3575 - categorical_accuracy: 0.8869
18240/60000 [========>.....................] - ETA: 1:10 - loss: 0.3572 - categorical_accuracy: 0.8870
18272/60000 [========>.....................] - ETA: 1:10 - loss: 0.3568 - categorical_accuracy: 0.8871
18304/60000 [========>.....................] - ETA: 1:09 - loss: 0.3563 - categorical_accuracy: 0.8873
18336/60000 [========>.....................] - ETA: 1:09 - loss: 0.3557 - categorical_accuracy: 0.8875
18368/60000 [========>.....................] - ETA: 1:09 - loss: 0.3558 - categorical_accuracy: 0.8876
18400/60000 [========>.....................] - ETA: 1:09 - loss: 0.3554 - categorical_accuracy: 0.8878
18432/60000 [========>.....................] - ETA: 1:09 - loss: 0.3548 - categorical_accuracy: 0.8880
18464/60000 [========>.....................] - ETA: 1:09 - loss: 0.3544 - categorical_accuracy: 0.8881
18496/60000 [========>.....................] - ETA: 1:09 - loss: 0.3539 - categorical_accuracy: 0.8882
18528/60000 [========>.....................] - ETA: 1:09 - loss: 0.3536 - categorical_accuracy: 0.8883
18560/60000 [========>.....................] - ETA: 1:09 - loss: 0.3533 - categorical_accuracy: 0.8884
18592/60000 [========>.....................] - ETA: 1:09 - loss: 0.3533 - categorical_accuracy: 0.8884
18624/60000 [========>.....................] - ETA: 1:09 - loss: 0.3531 - categorical_accuracy: 0.8885
18656/60000 [========>.....................] - ETA: 1:09 - loss: 0.3527 - categorical_accuracy: 0.8887
18688/60000 [========>.....................] - ETA: 1:09 - loss: 0.3523 - categorical_accuracy: 0.8888
18720/60000 [========>.....................] - ETA: 1:09 - loss: 0.3527 - categorical_accuracy: 0.8887
18752/60000 [========>.....................] - ETA: 1:09 - loss: 0.3529 - categorical_accuracy: 0.8888
18784/60000 [========>.....................] - ETA: 1:09 - loss: 0.3524 - categorical_accuracy: 0.8889
18816/60000 [========>.....................] - ETA: 1:09 - loss: 0.3520 - categorical_accuracy: 0.8891
18848/60000 [========>.....................] - ETA: 1:09 - loss: 0.3519 - categorical_accuracy: 0.8892
18880/60000 [========>.....................] - ETA: 1:09 - loss: 0.3516 - categorical_accuracy: 0.8893
18912/60000 [========>.....................] - ETA: 1:09 - loss: 0.3517 - categorical_accuracy: 0.8892
18944/60000 [========>.....................] - ETA: 1:08 - loss: 0.3514 - categorical_accuracy: 0.8894
18976/60000 [========>.....................] - ETA: 1:08 - loss: 0.3510 - categorical_accuracy: 0.8895
19008/60000 [========>.....................] - ETA: 1:08 - loss: 0.3508 - categorical_accuracy: 0.8896
19040/60000 [========>.....................] - ETA: 1:08 - loss: 0.3507 - categorical_accuracy: 0.8896
19072/60000 [========>.....................] - ETA: 1:08 - loss: 0.3504 - categorical_accuracy: 0.8897
19104/60000 [========>.....................] - ETA: 1:08 - loss: 0.3500 - categorical_accuracy: 0.8899
19136/60000 [========>.....................] - ETA: 1:08 - loss: 0.3497 - categorical_accuracy: 0.8899
19200/60000 [========>.....................] - ETA: 1:08 - loss: 0.3490 - categorical_accuracy: 0.8901
19232/60000 [========>.....................] - ETA: 1:08 - loss: 0.3486 - categorical_accuracy: 0.8902
19296/60000 [========>.....................] - ETA: 1:08 - loss: 0.3480 - categorical_accuracy: 0.8905
19360/60000 [========>.....................] - ETA: 1:08 - loss: 0.3472 - categorical_accuracy: 0.8908
19392/60000 [========>.....................] - ETA: 1:08 - loss: 0.3467 - categorical_accuracy: 0.8909
19424/60000 [========>.....................] - ETA: 1:08 - loss: 0.3467 - categorical_accuracy: 0.8910
19456/60000 [========>.....................] - ETA: 1:08 - loss: 0.3466 - categorical_accuracy: 0.8910
19488/60000 [========>.....................] - ETA: 1:07 - loss: 0.3462 - categorical_accuracy: 0.8912
19520/60000 [========>.....................] - ETA: 1:07 - loss: 0.3468 - categorical_accuracy: 0.8911
19552/60000 [========>.....................] - ETA: 1:07 - loss: 0.3464 - categorical_accuracy: 0.8912
19584/60000 [========>.....................] - ETA: 1:07 - loss: 0.3461 - categorical_accuracy: 0.8913
19616/60000 [========>.....................] - ETA: 1:07 - loss: 0.3461 - categorical_accuracy: 0.8914
19648/60000 [========>.....................] - ETA: 1:07 - loss: 0.3456 - categorical_accuracy: 0.8915
19680/60000 [========>.....................] - ETA: 1:07 - loss: 0.3452 - categorical_accuracy: 0.8917
19712/60000 [========>.....................] - ETA: 1:07 - loss: 0.3449 - categorical_accuracy: 0.8918
19744/60000 [========>.....................] - ETA: 1:07 - loss: 0.3448 - categorical_accuracy: 0.8918
19776/60000 [========>.....................] - ETA: 1:07 - loss: 0.3445 - categorical_accuracy: 0.8919
19808/60000 [========>.....................] - ETA: 1:07 - loss: 0.3440 - categorical_accuracy: 0.8921
19840/60000 [========>.....................] - ETA: 1:07 - loss: 0.3437 - categorical_accuracy: 0.8921
19904/60000 [========>.....................] - ETA: 1:07 - loss: 0.3428 - categorical_accuracy: 0.8924
19936/60000 [========>.....................] - ETA: 1:07 - loss: 0.3424 - categorical_accuracy: 0.8926
19968/60000 [========>.....................] - ETA: 1:07 - loss: 0.3422 - categorical_accuracy: 0.8925
20000/60000 [=========>....................] - ETA: 1:07 - loss: 0.3429 - categorical_accuracy: 0.8925
20032/60000 [=========>....................] - ETA: 1:07 - loss: 0.3427 - categorical_accuracy: 0.8926
20064/60000 [=========>....................] - ETA: 1:06 - loss: 0.3424 - categorical_accuracy: 0.8926
20096/60000 [=========>....................] - ETA: 1:06 - loss: 0.3421 - categorical_accuracy: 0.8927
20128/60000 [=========>....................] - ETA: 1:06 - loss: 0.3417 - categorical_accuracy: 0.8928
20160/60000 [=========>....................] - ETA: 1:06 - loss: 0.3413 - categorical_accuracy: 0.8930
20192/60000 [=========>....................] - ETA: 1:06 - loss: 0.3410 - categorical_accuracy: 0.8930
20224/60000 [=========>....................] - ETA: 1:06 - loss: 0.3407 - categorical_accuracy: 0.8931
20256/60000 [=========>....................] - ETA: 1:06 - loss: 0.3403 - categorical_accuracy: 0.8932
20288/60000 [=========>....................] - ETA: 1:06 - loss: 0.3399 - categorical_accuracy: 0.8934
20320/60000 [=========>....................] - ETA: 1:06 - loss: 0.3397 - categorical_accuracy: 0.8934
20352/60000 [=========>....................] - ETA: 1:06 - loss: 0.3394 - categorical_accuracy: 0.8934
20384/60000 [=========>....................] - ETA: 1:06 - loss: 0.3390 - categorical_accuracy: 0.8935
20416/60000 [=========>....................] - ETA: 1:06 - loss: 0.3386 - categorical_accuracy: 0.8936
20448/60000 [=========>....................] - ETA: 1:06 - loss: 0.3383 - categorical_accuracy: 0.8936
20480/60000 [=========>....................] - ETA: 1:06 - loss: 0.3379 - categorical_accuracy: 0.8937
20512/60000 [=========>....................] - ETA: 1:06 - loss: 0.3383 - categorical_accuracy: 0.8937
20544/60000 [=========>....................] - ETA: 1:06 - loss: 0.3383 - categorical_accuracy: 0.8937
20608/60000 [=========>....................] - ETA: 1:06 - loss: 0.3376 - categorical_accuracy: 0.8940
20640/60000 [=========>....................] - ETA: 1:05 - loss: 0.3371 - categorical_accuracy: 0.8941
20672/60000 [=========>....................] - ETA: 1:05 - loss: 0.3367 - categorical_accuracy: 0.8943
20736/60000 [=========>....................] - ETA: 1:05 - loss: 0.3360 - categorical_accuracy: 0.8945
20768/60000 [=========>....................] - ETA: 1:05 - loss: 0.3362 - categorical_accuracy: 0.8945
20800/60000 [=========>....................] - ETA: 1:05 - loss: 0.3358 - categorical_accuracy: 0.8947
20832/60000 [=========>....................] - ETA: 1:05 - loss: 0.3355 - categorical_accuracy: 0.8948
20864/60000 [=========>....................] - ETA: 1:05 - loss: 0.3350 - categorical_accuracy: 0.8949
20896/60000 [=========>....................] - ETA: 1:05 - loss: 0.3348 - categorical_accuracy: 0.8950
20928/60000 [=========>....................] - ETA: 1:05 - loss: 0.3346 - categorical_accuracy: 0.8951
20960/60000 [=========>....................] - ETA: 1:05 - loss: 0.3342 - categorical_accuracy: 0.8952
20992/60000 [=========>....................] - ETA: 1:05 - loss: 0.3337 - categorical_accuracy: 0.8953
21056/60000 [=========>....................] - ETA: 1:05 - loss: 0.3336 - categorical_accuracy: 0.8953
21120/60000 [=========>....................] - ETA: 1:05 - loss: 0.3335 - categorical_accuracy: 0.8953
21152/60000 [=========>....................] - ETA: 1:05 - loss: 0.3335 - categorical_accuracy: 0.8952
21216/60000 [=========>....................] - ETA: 1:04 - loss: 0.3331 - categorical_accuracy: 0.8954
21248/60000 [=========>....................] - ETA: 1:04 - loss: 0.3329 - categorical_accuracy: 0.8954
21312/60000 [=========>....................] - ETA: 1:04 - loss: 0.3323 - categorical_accuracy: 0.8956
21376/60000 [=========>....................] - ETA: 1:04 - loss: 0.3317 - categorical_accuracy: 0.8958
21408/60000 [=========>....................] - ETA: 1:04 - loss: 0.3313 - categorical_accuracy: 0.8959
21440/60000 [=========>....................] - ETA: 1:04 - loss: 0.3309 - categorical_accuracy: 0.8960
21472/60000 [=========>....................] - ETA: 1:04 - loss: 0.3308 - categorical_accuracy: 0.8960
21536/60000 [=========>....................] - ETA: 1:04 - loss: 0.3305 - categorical_accuracy: 0.8961
21568/60000 [=========>....................] - ETA: 1:04 - loss: 0.3307 - categorical_accuracy: 0.8961
21600/60000 [=========>....................] - ETA: 1:04 - loss: 0.3303 - categorical_accuracy: 0.8963
21632/60000 [=========>....................] - ETA: 1:04 - loss: 0.3299 - categorical_accuracy: 0.8964
21696/60000 [=========>....................] - ETA: 1:04 - loss: 0.3294 - categorical_accuracy: 0.8966
21728/60000 [=========>....................] - ETA: 1:04 - loss: 0.3291 - categorical_accuracy: 0.8968
21760/60000 [=========>....................] - ETA: 1:03 - loss: 0.3286 - categorical_accuracy: 0.8969
21792/60000 [=========>....................] - ETA: 1:03 - loss: 0.3283 - categorical_accuracy: 0.8970
21824/60000 [=========>....................] - ETA: 1:03 - loss: 0.3279 - categorical_accuracy: 0.8972
21856/60000 [=========>....................] - ETA: 1:03 - loss: 0.3276 - categorical_accuracy: 0.8972
21888/60000 [=========>....................] - ETA: 1:03 - loss: 0.3273 - categorical_accuracy: 0.8973
21920/60000 [=========>....................] - ETA: 1:03 - loss: 0.3276 - categorical_accuracy: 0.8973
21984/60000 [=========>....................] - ETA: 1:03 - loss: 0.3268 - categorical_accuracy: 0.8976
22016/60000 [==========>...................] - ETA: 1:03 - loss: 0.3267 - categorical_accuracy: 0.8975
22080/60000 [==========>...................] - ETA: 1:03 - loss: 0.3262 - categorical_accuracy: 0.8977
22112/60000 [==========>...................] - ETA: 1:03 - loss: 0.3257 - categorical_accuracy: 0.8979
22176/60000 [==========>...................] - ETA: 1:03 - loss: 0.3249 - categorical_accuracy: 0.8982
22208/60000 [==========>...................] - ETA: 1:03 - loss: 0.3250 - categorical_accuracy: 0.8983
22240/60000 [==========>...................] - ETA: 1:03 - loss: 0.3245 - categorical_accuracy: 0.8984
22272/60000 [==========>...................] - ETA: 1:03 - loss: 0.3243 - categorical_accuracy: 0.8985
22304/60000 [==========>...................] - ETA: 1:03 - loss: 0.3240 - categorical_accuracy: 0.8986
22336/60000 [==========>...................] - ETA: 1:02 - loss: 0.3236 - categorical_accuracy: 0.8987
22368/60000 [==========>...................] - ETA: 1:02 - loss: 0.3233 - categorical_accuracy: 0.8988
22400/60000 [==========>...................] - ETA: 1:02 - loss: 0.3230 - categorical_accuracy: 0.8989
22432/60000 [==========>...................] - ETA: 1:02 - loss: 0.3226 - categorical_accuracy: 0.8990
22464/60000 [==========>...................] - ETA: 1:02 - loss: 0.3226 - categorical_accuracy: 0.8990
22496/60000 [==========>...................] - ETA: 1:02 - loss: 0.3223 - categorical_accuracy: 0.8991
22560/60000 [==========>...................] - ETA: 1:02 - loss: 0.3218 - categorical_accuracy: 0.8992
22592/60000 [==========>...................] - ETA: 1:02 - loss: 0.3215 - categorical_accuracy: 0.8993
22656/60000 [==========>...................] - ETA: 1:02 - loss: 0.3212 - categorical_accuracy: 0.8994
22688/60000 [==========>...................] - ETA: 1:02 - loss: 0.3208 - categorical_accuracy: 0.8995
22720/60000 [==========>...................] - ETA: 1:02 - loss: 0.3204 - categorical_accuracy: 0.8996
22784/60000 [==========>...................] - ETA: 1:02 - loss: 0.3199 - categorical_accuracy: 0.8998
22816/60000 [==========>...................] - ETA: 1:02 - loss: 0.3199 - categorical_accuracy: 0.8997
22848/60000 [==========>...................] - ETA: 1:02 - loss: 0.3196 - categorical_accuracy: 0.8998
22880/60000 [==========>...................] - ETA: 1:01 - loss: 0.3192 - categorical_accuracy: 0.9000
22912/60000 [==========>...................] - ETA: 1:01 - loss: 0.3189 - categorical_accuracy: 0.9000
22944/60000 [==========>...................] - ETA: 1:01 - loss: 0.3189 - categorical_accuracy: 0.9000
22976/60000 [==========>...................] - ETA: 1:01 - loss: 0.3190 - categorical_accuracy: 0.9001
23008/60000 [==========>...................] - ETA: 1:01 - loss: 0.3186 - categorical_accuracy: 0.9002
23072/60000 [==========>...................] - ETA: 1:01 - loss: 0.3183 - categorical_accuracy: 0.9003
23104/60000 [==========>...................] - ETA: 1:01 - loss: 0.3179 - categorical_accuracy: 0.9004
23136/60000 [==========>...................] - ETA: 1:01 - loss: 0.3175 - categorical_accuracy: 0.9005
23168/60000 [==========>...................] - ETA: 1:01 - loss: 0.3173 - categorical_accuracy: 0.9006
23232/60000 [==========>...................] - ETA: 1:01 - loss: 0.3165 - categorical_accuracy: 0.9009
23264/60000 [==========>...................] - ETA: 1:01 - loss: 0.3163 - categorical_accuracy: 0.9009
23296/60000 [==========>...................] - ETA: 1:01 - loss: 0.3161 - categorical_accuracy: 0.9010
23328/60000 [==========>...................] - ETA: 1:01 - loss: 0.3158 - categorical_accuracy: 0.9011
23360/60000 [==========>...................] - ETA: 1:01 - loss: 0.3156 - categorical_accuracy: 0.9011
23392/60000 [==========>...................] - ETA: 1:01 - loss: 0.3152 - categorical_accuracy: 0.9012
23424/60000 [==========>...................] - ETA: 1:01 - loss: 0.3149 - categorical_accuracy: 0.9014
23456/60000 [==========>...................] - ETA: 1:01 - loss: 0.3146 - categorical_accuracy: 0.9015
23488/60000 [==========>...................] - ETA: 1:00 - loss: 0.3144 - categorical_accuracy: 0.9016
23552/60000 [==========>...................] - ETA: 1:00 - loss: 0.3140 - categorical_accuracy: 0.9016
23584/60000 [==========>...................] - ETA: 1:00 - loss: 0.3137 - categorical_accuracy: 0.9017
23616/60000 [==========>...................] - ETA: 1:00 - loss: 0.3136 - categorical_accuracy: 0.9018
23680/60000 [==========>...................] - ETA: 1:00 - loss: 0.3131 - categorical_accuracy: 0.9019
23744/60000 [==========>...................] - ETA: 1:00 - loss: 0.3127 - categorical_accuracy: 0.9020
23776/60000 [==========>...................] - ETA: 1:00 - loss: 0.3126 - categorical_accuracy: 0.9021
23840/60000 [==========>...................] - ETA: 1:00 - loss: 0.3120 - categorical_accuracy: 0.9023
23904/60000 [==========>...................] - ETA: 1:00 - loss: 0.3118 - categorical_accuracy: 0.9024
23936/60000 [==========>...................] - ETA: 1:00 - loss: 0.3116 - categorical_accuracy: 0.9024
24000/60000 [===========>..................] - ETA: 1:00 - loss: 0.3112 - categorical_accuracy: 0.9025
24064/60000 [===========>..................] - ETA: 59s - loss: 0.3110 - categorical_accuracy: 0.9026 
24096/60000 [===========>..................] - ETA: 59s - loss: 0.3106 - categorical_accuracy: 0.9027
24128/60000 [===========>..................] - ETA: 59s - loss: 0.3105 - categorical_accuracy: 0.9027
24160/60000 [===========>..................] - ETA: 59s - loss: 0.3104 - categorical_accuracy: 0.9028
24192/60000 [===========>..................] - ETA: 59s - loss: 0.3104 - categorical_accuracy: 0.9028
24224/60000 [===========>..................] - ETA: 59s - loss: 0.3100 - categorical_accuracy: 0.9029
24256/60000 [===========>..................] - ETA: 59s - loss: 0.3098 - categorical_accuracy: 0.9030
24288/60000 [===========>..................] - ETA: 59s - loss: 0.3097 - categorical_accuracy: 0.9031
24320/60000 [===========>..................] - ETA: 59s - loss: 0.3094 - categorical_accuracy: 0.9032
24352/60000 [===========>..................] - ETA: 59s - loss: 0.3091 - categorical_accuracy: 0.9033
24384/60000 [===========>..................] - ETA: 59s - loss: 0.3088 - categorical_accuracy: 0.9034
24416/60000 [===========>..................] - ETA: 59s - loss: 0.3085 - categorical_accuracy: 0.9035
24480/60000 [===========>..................] - ETA: 59s - loss: 0.3082 - categorical_accuracy: 0.9036
24512/60000 [===========>..................] - ETA: 59s - loss: 0.3079 - categorical_accuracy: 0.9037
24544/60000 [===========>..................] - ETA: 59s - loss: 0.3076 - categorical_accuracy: 0.9038
24576/60000 [===========>..................] - ETA: 59s - loss: 0.3073 - categorical_accuracy: 0.9039
24608/60000 [===========>..................] - ETA: 59s - loss: 0.3070 - categorical_accuracy: 0.9040
24640/60000 [===========>..................] - ETA: 58s - loss: 0.3068 - categorical_accuracy: 0.9041
24672/60000 [===========>..................] - ETA: 58s - loss: 0.3066 - categorical_accuracy: 0.9041
24704/60000 [===========>..................] - ETA: 58s - loss: 0.3062 - categorical_accuracy: 0.9042
24736/60000 [===========>..................] - ETA: 58s - loss: 0.3062 - categorical_accuracy: 0.9042
24768/60000 [===========>..................] - ETA: 58s - loss: 0.3062 - categorical_accuracy: 0.9042
24800/60000 [===========>..................] - ETA: 58s - loss: 0.3060 - categorical_accuracy: 0.9043
24832/60000 [===========>..................] - ETA: 58s - loss: 0.3062 - categorical_accuracy: 0.9044
24864/60000 [===========>..................] - ETA: 58s - loss: 0.3058 - categorical_accuracy: 0.9045
24896/60000 [===========>..................] - ETA: 58s - loss: 0.3054 - categorical_accuracy: 0.9046
24928/60000 [===========>..................] - ETA: 58s - loss: 0.3052 - categorical_accuracy: 0.9047
24992/60000 [===========>..................] - ETA: 58s - loss: 0.3050 - categorical_accuracy: 0.9048
25024/60000 [===========>..................] - ETA: 58s - loss: 0.3047 - categorical_accuracy: 0.9049
25056/60000 [===========>..................] - ETA: 58s - loss: 0.3044 - categorical_accuracy: 0.9050
25088/60000 [===========>..................] - ETA: 58s - loss: 0.3043 - categorical_accuracy: 0.9050
25152/60000 [===========>..................] - ETA: 58s - loss: 0.3047 - categorical_accuracy: 0.9049
25184/60000 [===========>..................] - ETA: 58s - loss: 0.3044 - categorical_accuracy: 0.9050
25248/60000 [===========>..................] - ETA: 57s - loss: 0.3039 - categorical_accuracy: 0.9051
25312/60000 [===========>..................] - ETA: 57s - loss: 0.3034 - categorical_accuracy: 0.9053
25376/60000 [===========>..................] - ETA: 57s - loss: 0.3032 - categorical_accuracy: 0.9055
25440/60000 [===========>..................] - ETA: 57s - loss: 0.3030 - categorical_accuracy: 0.9055
25504/60000 [===========>..................] - ETA: 57s - loss: 0.3025 - categorical_accuracy: 0.9057
25536/60000 [===========>..................] - ETA: 57s - loss: 0.3025 - categorical_accuracy: 0.9056
25568/60000 [===========>..................] - ETA: 57s - loss: 0.3027 - categorical_accuracy: 0.9055
25600/60000 [===========>..................] - ETA: 57s - loss: 0.3025 - categorical_accuracy: 0.9056
25632/60000 [===========>..................] - ETA: 57s - loss: 0.3023 - categorical_accuracy: 0.9057
25664/60000 [===========>..................] - ETA: 57s - loss: 0.3022 - categorical_accuracy: 0.9057
25696/60000 [===========>..................] - ETA: 57s - loss: 0.3020 - categorical_accuracy: 0.9057
25728/60000 [===========>..................] - ETA: 57s - loss: 0.3017 - categorical_accuracy: 0.9058
25760/60000 [===========>..................] - ETA: 57s - loss: 0.3015 - categorical_accuracy: 0.9059
25792/60000 [===========>..................] - ETA: 56s - loss: 0.3012 - categorical_accuracy: 0.9060
25824/60000 [===========>..................] - ETA: 56s - loss: 0.3010 - categorical_accuracy: 0.9061
25856/60000 [===========>..................] - ETA: 56s - loss: 0.3007 - categorical_accuracy: 0.9062
25888/60000 [===========>..................] - ETA: 56s - loss: 0.3004 - categorical_accuracy: 0.9063
25920/60000 [===========>..................] - ETA: 56s - loss: 0.3001 - categorical_accuracy: 0.9064
25952/60000 [===========>..................] - ETA: 56s - loss: 0.2999 - categorical_accuracy: 0.9064
25984/60000 [===========>..................] - ETA: 56s - loss: 0.2997 - categorical_accuracy: 0.9065
26016/60000 [============>.................] - ETA: 56s - loss: 0.2996 - categorical_accuracy: 0.9065
26048/60000 [============>.................] - ETA: 56s - loss: 0.2995 - categorical_accuracy: 0.9065
26080/60000 [============>.................] - ETA: 56s - loss: 0.2993 - categorical_accuracy: 0.9066
26144/60000 [============>.................] - ETA: 56s - loss: 0.2989 - categorical_accuracy: 0.9067
26176/60000 [============>.................] - ETA: 56s - loss: 0.2986 - categorical_accuracy: 0.9068
26208/60000 [============>.................] - ETA: 56s - loss: 0.2985 - categorical_accuracy: 0.9069
26240/60000 [============>.................] - ETA: 56s - loss: 0.2982 - categorical_accuracy: 0.9070
26272/60000 [============>.................] - ETA: 56s - loss: 0.2979 - categorical_accuracy: 0.9071
26336/60000 [============>.................] - ETA: 56s - loss: 0.2973 - categorical_accuracy: 0.9073
26400/60000 [============>.................] - ETA: 55s - loss: 0.2970 - categorical_accuracy: 0.9074
26464/60000 [============>.................] - ETA: 55s - loss: 0.2967 - categorical_accuracy: 0.9076
26496/60000 [============>.................] - ETA: 55s - loss: 0.2964 - categorical_accuracy: 0.9077
26528/60000 [============>.................] - ETA: 55s - loss: 0.2962 - categorical_accuracy: 0.9078
26560/60000 [============>.................] - ETA: 55s - loss: 0.2959 - categorical_accuracy: 0.9079
26592/60000 [============>.................] - ETA: 55s - loss: 0.2956 - categorical_accuracy: 0.9080
26624/60000 [============>.................] - ETA: 55s - loss: 0.2954 - categorical_accuracy: 0.9080
26688/60000 [============>.................] - ETA: 55s - loss: 0.2950 - categorical_accuracy: 0.9081
26720/60000 [============>.................] - ETA: 55s - loss: 0.2948 - categorical_accuracy: 0.9081
26752/60000 [============>.................] - ETA: 55s - loss: 0.2946 - categorical_accuracy: 0.9082
26784/60000 [============>.................] - ETA: 55s - loss: 0.2942 - categorical_accuracy: 0.9083
26816/60000 [============>.................] - ETA: 55s - loss: 0.2940 - categorical_accuracy: 0.9083
26848/60000 [============>.................] - ETA: 55s - loss: 0.2942 - categorical_accuracy: 0.9083
26880/60000 [============>.................] - ETA: 55s - loss: 0.2943 - categorical_accuracy: 0.9083
26944/60000 [============>.................] - ETA: 55s - loss: 0.2941 - categorical_accuracy: 0.9084
27008/60000 [============>.................] - ETA: 54s - loss: 0.2939 - categorical_accuracy: 0.9085
27072/60000 [============>.................] - ETA: 54s - loss: 0.2933 - categorical_accuracy: 0.9087
27104/60000 [============>.................] - ETA: 54s - loss: 0.2930 - categorical_accuracy: 0.9088
27136/60000 [============>.................] - ETA: 54s - loss: 0.2928 - categorical_accuracy: 0.9089
27168/60000 [============>.................] - ETA: 54s - loss: 0.2925 - categorical_accuracy: 0.9090
27232/60000 [============>.................] - ETA: 54s - loss: 0.2922 - categorical_accuracy: 0.9090
27264/60000 [============>.................] - ETA: 54s - loss: 0.2921 - categorical_accuracy: 0.9090
27296/60000 [============>.................] - ETA: 54s - loss: 0.2918 - categorical_accuracy: 0.9091
27360/60000 [============>.................] - ETA: 54s - loss: 0.2915 - categorical_accuracy: 0.9092
27392/60000 [============>.................] - ETA: 54s - loss: 0.2914 - categorical_accuracy: 0.9092
27424/60000 [============>.................] - ETA: 54s - loss: 0.2911 - categorical_accuracy: 0.9093
27456/60000 [============>.................] - ETA: 54s - loss: 0.2910 - categorical_accuracy: 0.9093
27488/60000 [============>.................] - ETA: 54s - loss: 0.2908 - categorical_accuracy: 0.9093
27520/60000 [============>.................] - ETA: 54s - loss: 0.2905 - categorical_accuracy: 0.9094
27552/60000 [============>.................] - ETA: 54s - loss: 0.2908 - categorical_accuracy: 0.9094
27616/60000 [============>.................] - ETA: 53s - loss: 0.2905 - categorical_accuracy: 0.9095
27648/60000 [============>.................] - ETA: 53s - loss: 0.2902 - categorical_accuracy: 0.9096
27680/60000 [============>.................] - ETA: 53s - loss: 0.2899 - categorical_accuracy: 0.9097
27712/60000 [============>.................] - ETA: 53s - loss: 0.2896 - categorical_accuracy: 0.9098
27744/60000 [============>.................] - ETA: 53s - loss: 0.2897 - categorical_accuracy: 0.9097
27776/60000 [============>.................] - ETA: 53s - loss: 0.2895 - categorical_accuracy: 0.9098
27808/60000 [============>.................] - ETA: 53s - loss: 0.2892 - categorical_accuracy: 0.9099
27840/60000 [============>.................] - ETA: 53s - loss: 0.2889 - categorical_accuracy: 0.9100
27872/60000 [============>.................] - ETA: 53s - loss: 0.2887 - categorical_accuracy: 0.9101
27904/60000 [============>.................] - ETA: 53s - loss: 0.2885 - categorical_accuracy: 0.9101
27936/60000 [============>.................] - ETA: 53s - loss: 0.2884 - categorical_accuracy: 0.9102
27968/60000 [============>.................] - ETA: 53s - loss: 0.2883 - categorical_accuracy: 0.9102
28000/60000 [=============>................] - ETA: 53s - loss: 0.2882 - categorical_accuracy: 0.9102
28032/60000 [=============>................] - ETA: 53s - loss: 0.2881 - categorical_accuracy: 0.9103
28064/60000 [=============>................] - ETA: 53s - loss: 0.2879 - categorical_accuracy: 0.9103
28096/60000 [=============>................] - ETA: 53s - loss: 0.2876 - categorical_accuracy: 0.9104
28128/60000 [=============>................] - ETA: 53s - loss: 0.2873 - categorical_accuracy: 0.9106
28192/60000 [=============>................] - ETA: 52s - loss: 0.2869 - categorical_accuracy: 0.9106
28256/60000 [=============>................] - ETA: 52s - loss: 0.2867 - categorical_accuracy: 0.9107
28320/60000 [=============>................] - ETA: 52s - loss: 0.2863 - categorical_accuracy: 0.9108
28352/60000 [=============>................] - ETA: 52s - loss: 0.2861 - categorical_accuracy: 0.9109
28384/60000 [=============>................] - ETA: 52s - loss: 0.2859 - categorical_accuracy: 0.9110
28416/60000 [=============>................] - ETA: 52s - loss: 0.2858 - categorical_accuracy: 0.9110
28448/60000 [=============>................] - ETA: 52s - loss: 0.2856 - categorical_accuracy: 0.9110
28480/60000 [=============>................] - ETA: 52s - loss: 0.2854 - categorical_accuracy: 0.9111
28544/60000 [=============>................] - ETA: 52s - loss: 0.2851 - categorical_accuracy: 0.9112
28608/60000 [=============>................] - ETA: 52s - loss: 0.2846 - categorical_accuracy: 0.9114
28640/60000 [=============>................] - ETA: 52s - loss: 0.2845 - categorical_accuracy: 0.9114
28672/60000 [=============>................] - ETA: 52s - loss: 0.2845 - categorical_accuracy: 0.9114
28704/60000 [=============>................] - ETA: 52s - loss: 0.2844 - categorical_accuracy: 0.9114
28736/60000 [=============>................] - ETA: 52s - loss: 0.2841 - categorical_accuracy: 0.9115
28768/60000 [=============>................] - ETA: 51s - loss: 0.2839 - categorical_accuracy: 0.9116
28800/60000 [=============>................] - ETA: 51s - loss: 0.2837 - categorical_accuracy: 0.9117
28832/60000 [=============>................] - ETA: 51s - loss: 0.2834 - categorical_accuracy: 0.9117
28864/60000 [=============>................] - ETA: 51s - loss: 0.2832 - categorical_accuracy: 0.9118
28896/60000 [=============>................] - ETA: 51s - loss: 0.2831 - categorical_accuracy: 0.9119
28928/60000 [=============>................] - ETA: 51s - loss: 0.2829 - categorical_accuracy: 0.9120
28960/60000 [=============>................] - ETA: 51s - loss: 0.2829 - categorical_accuracy: 0.9121
28992/60000 [=============>................] - ETA: 51s - loss: 0.2827 - categorical_accuracy: 0.9121
29024/60000 [=============>................] - ETA: 51s - loss: 0.2824 - categorical_accuracy: 0.9122
29088/60000 [=============>................] - ETA: 51s - loss: 0.2819 - categorical_accuracy: 0.9124
29120/60000 [=============>................] - ETA: 51s - loss: 0.2816 - categorical_accuracy: 0.9125
29152/60000 [=============>................] - ETA: 51s - loss: 0.2818 - categorical_accuracy: 0.9125
29184/60000 [=============>................] - ETA: 51s - loss: 0.2816 - categorical_accuracy: 0.9126
29216/60000 [=============>................] - ETA: 51s - loss: 0.2815 - categorical_accuracy: 0.9126
29248/60000 [=============>................] - ETA: 51s - loss: 0.2814 - categorical_accuracy: 0.9126
29280/60000 [=============>................] - ETA: 51s - loss: 0.2812 - categorical_accuracy: 0.9126
29312/60000 [=============>................] - ETA: 51s - loss: 0.2811 - categorical_accuracy: 0.9127
29344/60000 [=============>................] - ETA: 51s - loss: 0.2810 - categorical_accuracy: 0.9127
29408/60000 [=============>................] - ETA: 50s - loss: 0.2805 - categorical_accuracy: 0.9129
29440/60000 [=============>................] - ETA: 50s - loss: 0.2803 - categorical_accuracy: 0.9129
29504/60000 [=============>................] - ETA: 50s - loss: 0.2800 - categorical_accuracy: 0.9130
29536/60000 [=============>................] - ETA: 50s - loss: 0.2799 - categorical_accuracy: 0.9131
29600/60000 [=============>................] - ETA: 50s - loss: 0.2794 - categorical_accuracy: 0.9132
29632/60000 [=============>................] - ETA: 50s - loss: 0.2792 - categorical_accuracy: 0.9133
29664/60000 [=============>................] - ETA: 50s - loss: 0.2790 - categorical_accuracy: 0.9133
29696/60000 [=============>................] - ETA: 50s - loss: 0.2789 - categorical_accuracy: 0.9133
29728/60000 [=============>................] - ETA: 50s - loss: 0.2789 - categorical_accuracy: 0.9133
29760/60000 [=============>................] - ETA: 50s - loss: 0.2788 - categorical_accuracy: 0.9133
29792/60000 [=============>................] - ETA: 50s - loss: 0.2786 - categorical_accuracy: 0.9134
29824/60000 [=============>................] - ETA: 50s - loss: 0.2783 - categorical_accuracy: 0.9135
29856/60000 [=============>................] - ETA: 50s - loss: 0.2782 - categorical_accuracy: 0.9136
29920/60000 [=============>................] - ETA: 50s - loss: 0.2778 - categorical_accuracy: 0.9137
29952/60000 [=============>................] - ETA: 49s - loss: 0.2775 - categorical_accuracy: 0.9138
29984/60000 [=============>................] - ETA: 49s - loss: 0.2774 - categorical_accuracy: 0.9139
30048/60000 [==============>...............] - ETA: 49s - loss: 0.2769 - categorical_accuracy: 0.9140
30080/60000 [==============>...............] - ETA: 49s - loss: 0.2767 - categorical_accuracy: 0.9141
30112/60000 [==============>...............] - ETA: 49s - loss: 0.2764 - categorical_accuracy: 0.9142
30144/60000 [==============>...............] - ETA: 49s - loss: 0.2763 - categorical_accuracy: 0.9142
30176/60000 [==============>...............] - ETA: 49s - loss: 0.2760 - categorical_accuracy: 0.9143
30208/60000 [==============>...............] - ETA: 49s - loss: 0.2758 - categorical_accuracy: 0.9143
30240/60000 [==============>...............] - ETA: 49s - loss: 0.2756 - categorical_accuracy: 0.9144
30272/60000 [==============>...............] - ETA: 49s - loss: 0.2754 - categorical_accuracy: 0.9145
30304/60000 [==============>...............] - ETA: 49s - loss: 0.2752 - categorical_accuracy: 0.9145
30336/60000 [==============>...............] - ETA: 49s - loss: 0.2750 - categorical_accuracy: 0.9146
30368/60000 [==============>...............] - ETA: 49s - loss: 0.2753 - categorical_accuracy: 0.9146
30400/60000 [==============>...............] - ETA: 49s - loss: 0.2750 - categorical_accuracy: 0.9147
30432/60000 [==============>...............] - ETA: 49s - loss: 0.2747 - categorical_accuracy: 0.9148
30464/60000 [==============>...............] - ETA: 49s - loss: 0.2751 - categorical_accuracy: 0.9148
30496/60000 [==============>...............] - ETA: 49s - loss: 0.2751 - categorical_accuracy: 0.9148
30528/60000 [==============>...............] - ETA: 49s - loss: 0.2748 - categorical_accuracy: 0.9149
30560/60000 [==============>...............] - ETA: 48s - loss: 0.2747 - categorical_accuracy: 0.9149
30592/60000 [==============>...............] - ETA: 48s - loss: 0.2745 - categorical_accuracy: 0.9150
30656/60000 [==============>...............] - ETA: 48s - loss: 0.2745 - categorical_accuracy: 0.9150
30688/60000 [==============>...............] - ETA: 48s - loss: 0.2746 - categorical_accuracy: 0.9150
30720/60000 [==============>...............] - ETA: 48s - loss: 0.2746 - categorical_accuracy: 0.9151
30752/60000 [==============>...............] - ETA: 48s - loss: 0.2743 - categorical_accuracy: 0.9152
30784/60000 [==============>...............] - ETA: 48s - loss: 0.2741 - categorical_accuracy: 0.9152
30816/60000 [==============>...............] - ETA: 48s - loss: 0.2739 - categorical_accuracy: 0.9153
30848/60000 [==============>...............] - ETA: 48s - loss: 0.2737 - categorical_accuracy: 0.9154
30880/60000 [==============>...............] - ETA: 48s - loss: 0.2736 - categorical_accuracy: 0.9154
30912/60000 [==============>...............] - ETA: 48s - loss: 0.2733 - categorical_accuracy: 0.9155
30944/60000 [==============>...............] - ETA: 48s - loss: 0.2732 - categorical_accuracy: 0.9156
30976/60000 [==============>...............] - ETA: 48s - loss: 0.2731 - categorical_accuracy: 0.9156
31008/60000 [==============>...............] - ETA: 48s - loss: 0.2729 - categorical_accuracy: 0.9157
31040/60000 [==============>...............] - ETA: 48s - loss: 0.2726 - categorical_accuracy: 0.9158
31072/60000 [==============>...............] - ETA: 48s - loss: 0.2725 - categorical_accuracy: 0.9158
31136/60000 [==============>...............] - ETA: 47s - loss: 0.2721 - categorical_accuracy: 0.9159
31168/60000 [==============>...............] - ETA: 47s - loss: 0.2720 - categorical_accuracy: 0.9159
31200/60000 [==============>...............] - ETA: 47s - loss: 0.2719 - categorical_accuracy: 0.9160
31232/60000 [==============>...............] - ETA: 47s - loss: 0.2718 - categorical_accuracy: 0.9160
31264/60000 [==============>...............] - ETA: 47s - loss: 0.2717 - categorical_accuracy: 0.9160
31296/60000 [==============>...............] - ETA: 47s - loss: 0.2716 - categorical_accuracy: 0.9160
31328/60000 [==============>...............] - ETA: 47s - loss: 0.2714 - categorical_accuracy: 0.9161
31360/60000 [==============>...............] - ETA: 47s - loss: 0.2712 - categorical_accuracy: 0.9161
31424/60000 [==============>...............] - ETA: 47s - loss: 0.2708 - categorical_accuracy: 0.9162
31488/60000 [==============>...............] - ETA: 47s - loss: 0.2703 - categorical_accuracy: 0.9164
31520/60000 [==============>...............] - ETA: 47s - loss: 0.2701 - categorical_accuracy: 0.9165
31552/60000 [==============>...............] - ETA: 47s - loss: 0.2699 - categorical_accuracy: 0.9166
31584/60000 [==============>...............] - ETA: 47s - loss: 0.2696 - categorical_accuracy: 0.9166
31616/60000 [==============>...............] - ETA: 47s - loss: 0.2695 - categorical_accuracy: 0.9167
31680/60000 [==============>...............] - ETA: 47s - loss: 0.2693 - categorical_accuracy: 0.9168
31712/60000 [==============>...............] - ETA: 47s - loss: 0.2692 - categorical_accuracy: 0.9168
31744/60000 [==============>...............] - ETA: 46s - loss: 0.2692 - categorical_accuracy: 0.9168
31776/60000 [==============>...............] - ETA: 46s - loss: 0.2690 - categorical_accuracy: 0.9169
31808/60000 [==============>...............] - ETA: 46s - loss: 0.2687 - categorical_accuracy: 0.9169
31840/60000 [==============>...............] - ETA: 46s - loss: 0.2687 - categorical_accuracy: 0.9170
31872/60000 [==============>...............] - ETA: 46s - loss: 0.2685 - categorical_accuracy: 0.9170
31904/60000 [==============>...............] - ETA: 46s - loss: 0.2683 - categorical_accuracy: 0.9171
31936/60000 [==============>...............] - ETA: 46s - loss: 0.2685 - categorical_accuracy: 0.9171
31968/60000 [==============>...............] - ETA: 46s - loss: 0.2684 - categorical_accuracy: 0.9171
32000/60000 [===============>..............] - ETA: 46s - loss: 0.2684 - categorical_accuracy: 0.9171
32032/60000 [===============>..............] - ETA: 46s - loss: 0.2682 - categorical_accuracy: 0.9171
32064/60000 [===============>..............] - ETA: 46s - loss: 0.2680 - categorical_accuracy: 0.9172
32096/60000 [===============>..............] - ETA: 46s - loss: 0.2678 - categorical_accuracy: 0.9172
32128/60000 [===============>..............] - ETA: 46s - loss: 0.2677 - categorical_accuracy: 0.9172
32160/60000 [===============>..............] - ETA: 46s - loss: 0.2675 - categorical_accuracy: 0.9173
32192/60000 [===============>..............] - ETA: 46s - loss: 0.2674 - categorical_accuracy: 0.9173
32224/60000 [===============>..............] - ETA: 46s - loss: 0.2672 - categorical_accuracy: 0.9173
32256/60000 [===============>..............] - ETA: 46s - loss: 0.2671 - categorical_accuracy: 0.9174
32288/60000 [===============>..............] - ETA: 46s - loss: 0.2669 - categorical_accuracy: 0.9175
32320/60000 [===============>..............] - ETA: 46s - loss: 0.2669 - categorical_accuracy: 0.9175
32352/60000 [===============>..............] - ETA: 45s - loss: 0.2667 - categorical_accuracy: 0.9175
32384/60000 [===============>..............] - ETA: 45s - loss: 0.2666 - categorical_accuracy: 0.9176
32416/60000 [===============>..............] - ETA: 45s - loss: 0.2664 - categorical_accuracy: 0.9176
32448/60000 [===============>..............] - ETA: 45s - loss: 0.2662 - categorical_accuracy: 0.9177
32480/60000 [===============>..............] - ETA: 45s - loss: 0.2661 - categorical_accuracy: 0.9177
32512/60000 [===============>..............] - ETA: 45s - loss: 0.2658 - categorical_accuracy: 0.9178
32544/60000 [===============>..............] - ETA: 45s - loss: 0.2657 - categorical_accuracy: 0.9178
32576/60000 [===============>..............] - ETA: 45s - loss: 0.2656 - categorical_accuracy: 0.9178
32608/60000 [===============>..............] - ETA: 45s - loss: 0.2655 - categorical_accuracy: 0.9178
32640/60000 [===============>..............] - ETA: 45s - loss: 0.2657 - categorical_accuracy: 0.9178
32672/60000 [===============>..............] - ETA: 45s - loss: 0.2655 - categorical_accuracy: 0.9179
32704/60000 [===============>..............] - ETA: 45s - loss: 0.2653 - categorical_accuracy: 0.9180
32768/60000 [===============>..............] - ETA: 45s - loss: 0.2652 - categorical_accuracy: 0.9180
32832/60000 [===============>..............] - ETA: 45s - loss: 0.2652 - categorical_accuracy: 0.9180
32864/60000 [===============>..............] - ETA: 45s - loss: 0.2650 - categorical_accuracy: 0.9180
32896/60000 [===============>..............] - ETA: 45s - loss: 0.2648 - categorical_accuracy: 0.9181
32960/60000 [===============>..............] - ETA: 44s - loss: 0.2644 - categorical_accuracy: 0.9182
32992/60000 [===============>..............] - ETA: 44s - loss: 0.2643 - categorical_accuracy: 0.9183
33024/60000 [===============>..............] - ETA: 44s - loss: 0.2643 - categorical_accuracy: 0.9182
33056/60000 [===============>..............] - ETA: 44s - loss: 0.2643 - categorical_accuracy: 0.9182
33088/60000 [===============>..............] - ETA: 44s - loss: 0.2643 - categorical_accuracy: 0.9182
33120/60000 [===============>..............] - ETA: 44s - loss: 0.2641 - categorical_accuracy: 0.9182
33184/60000 [===============>..............] - ETA: 44s - loss: 0.2637 - categorical_accuracy: 0.9184
33216/60000 [===============>..............] - ETA: 44s - loss: 0.2635 - categorical_accuracy: 0.9184
33248/60000 [===============>..............] - ETA: 44s - loss: 0.2634 - categorical_accuracy: 0.9185
33280/60000 [===============>..............] - ETA: 44s - loss: 0.2635 - categorical_accuracy: 0.9184
33312/60000 [===============>..............] - ETA: 44s - loss: 0.2634 - categorical_accuracy: 0.9185
33344/60000 [===============>..............] - ETA: 44s - loss: 0.2632 - categorical_accuracy: 0.9185
33376/60000 [===============>..............] - ETA: 44s - loss: 0.2629 - categorical_accuracy: 0.9186
33408/60000 [===============>..............] - ETA: 44s - loss: 0.2627 - categorical_accuracy: 0.9187
33440/60000 [===============>..............] - ETA: 44s - loss: 0.2625 - categorical_accuracy: 0.9187
33472/60000 [===============>..............] - ETA: 44s - loss: 0.2623 - categorical_accuracy: 0.9188
33504/60000 [===============>..............] - ETA: 44s - loss: 0.2623 - categorical_accuracy: 0.9188
33536/60000 [===============>..............] - ETA: 44s - loss: 0.2624 - categorical_accuracy: 0.9188
33568/60000 [===============>..............] - ETA: 43s - loss: 0.2623 - categorical_accuracy: 0.9188
33600/60000 [===============>..............] - ETA: 43s - loss: 0.2623 - categorical_accuracy: 0.9188
33632/60000 [===============>..............] - ETA: 43s - loss: 0.2621 - categorical_accuracy: 0.9189
33664/60000 [===============>..............] - ETA: 43s - loss: 0.2619 - categorical_accuracy: 0.9189
33696/60000 [===============>..............] - ETA: 43s - loss: 0.2617 - categorical_accuracy: 0.9190
33728/60000 [===============>..............] - ETA: 43s - loss: 0.2616 - categorical_accuracy: 0.9191
33760/60000 [===============>..............] - ETA: 43s - loss: 0.2616 - categorical_accuracy: 0.9190
33792/60000 [===============>..............] - ETA: 43s - loss: 0.2615 - categorical_accuracy: 0.9191
33824/60000 [===============>..............] - ETA: 43s - loss: 0.2613 - categorical_accuracy: 0.9191
33856/60000 [===============>..............] - ETA: 43s - loss: 0.2612 - categorical_accuracy: 0.9192
33888/60000 [===============>..............] - ETA: 43s - loss: 0.2610 - categorical_accuracy: 0.9192
33920/60000 [===============>..............] - ETA: 43s - loss: 0.2610 - categorical_accuracy: 0.9193
33952/60000 [===============>..............] - ETA: 43s - loss: 0.2609 - categorical_accuracy: 0.9192
33984/60000 [===============>..............] - ETA: 43s - loss: 0.2608 - categorical_accuracy: 0.9193
34016/60000 [================>.............] - ETA: 43s - loss: 0.2607 - categorical_accuracy: 0.9193
34048/60000 [================>.............] - ETA: 43s - loss: 0.2605 - categorical_accuracy: 0.9194
34080/60000 [================>.............] - ETA: 43s - loss: 0.2604 - categorical_accuracy: 0.9194
34112/60000 [================>.............] - ETA: 43s - loss: 0.2602 - categorical_accuracy: 0.9195
34144/60000 [================>.............] - ETA: 43s - loss: 0.2601 - categorical_accuracy: 0.9195
34208/60000 [================>.............] - ETA: 42s - loss: 0.2601 - categorical_accuracy: 0.9195
34240/60000 [================>.............] - ETA: 42s - loss: 0.2600 - categorical_accuracy: 0.9196
34272/60000 [================>.............] - ETA: 42s - loss: 0.2598 - categorical_accuracy: 0.9196
34304/60000 [================>.............] - ETA: 42s - loss: 0.2596 - categorical_accuracy: 0.9197
34336/60000 [================>.............] - ETA: 42s - loss: 0.2593 - categorical_accuracy: 0.9198
34368/60000 [================>.............] - ETA: 42s - loss: 0.2592 - categorical_accuracy: 0.9198
34400/60000 [================>.............] - ETA: 42s - loss: 0.2591 - categorical_accuracy: 0.9199
34432/60000 [================>.............] - ETA: 42s - loss: 0.2592 - categorical_accuracy: 0.9198
34464/60000 [================>.............] - ETA: 42s - loss: 0.2590 - categorical_accuracy: 0.9199
34528/60000 [================>.............] - ETA: 42s - loss: 0.2591 - categorical_accuracy: 0.9199
34560/60000 [================>.............] - ETA: 42s - loss: 0.2589 - categorical_accuracy: 0.9199
34592/60000 [================>.............] - ETA: 42s - loss: 0.2588 - categorical_accuracy: 0.9200
34624/60000 [================>.............] - ETA: 42s - loss: 0.2585 - categorical_accuracy: 0.9201
34656/60000 [================>.............] - ETA: 42s - loss: 0.2584 - categorical_accuracy: 0.9201
34688/60000 [================>.............] - ETA: 42s - loss: 0.2582 - categorical_accuracy: 0.9201
34720/60000 [================>.............] - ETA: 42s - loss: 0.2582 - categorical_accuracy: 0.9202
34784/60000 [================>.............] - ETA: 41s - loss: 0.2581 - categorical_accuracy: 0.9202
34848/60000 [================>.............] - ETA: 41s - loss: 0.2578 - categorical_accuracy: 0.9203
34880/60000 [================>.............] - ETA: 41s - loss: 0.2576 - categorical_accuracy: 0.9203
34912/60000 [================>.............] - ETA: 41s - loss: 0.2575 - categorical_accuracy: 0.9204
34944/60000 [================>.............] - ETA: 41s - loss: 0.2574 - categorical_accuracy: 0.9204
34976/60000 [================>.............] - ETA: 41s - loss: 0.2573 - categorical_accuracy: 0.9205
35008/60000 [================>.............] - ETA: 41s - loss: 0.2571 - categorical_accuracy: 0.9205
35040/60000 [================>.............] - ETA: 41s - loss: 0.2570 - categorical_accuracy: 0.9206
35072/60000 [================>.............] - ETA: 41s - loss: 0.2568 - categorical_accuracy: 0.9206
35104/60000 [================>.............] - ETA: 41s - loss: 0.2567 - categorical_accuracy: 0.9206
35136/60000 [================>.............] - ETA: 41s - loss: 0.2565 - categorical_accuracy: 0.9207
35168/60000 [================>.............] - ETA: 41s - loss: 0.2565 - categorical_accuracy: 0.9207
35200/60000 [================>.............] - ETA: 41s - loss: 0.2564 - categorical_accuracy: 0.9208
35264/60000 [================>.............] - ETA: 41s - loss: 0.2562 - categorical_accuracy: 0.9208
35328/60000 [================>.............] - ETA: 41s - loss: 0.2560 - categorical_accuracy: 0.9209
35360/60000 [================>.............] - ETA: 41s - loss: 0.2558 - categorical_accuracy: 0.9210
35392/60000 [================>.............] - ETA: 40s - loss: 0.2559 - categorical_accuracy: 0.9210
35424/60000 [================>.............] - ETA: 40s - loss: 0.2558 - categorical_accuracy: 0.9210
35456/60000 [================>.............] - ETA: 40s - loss: 0.2556 - categorical_accuracy: 0.9211
35488/60000 [================>.............] - ETA: 40s - loss: 0.2555 - categorical_accuracy: 0.9211
35520/60000 [================>.............] - ETA: 40s - loss: 0.2553 - categorical_accuracy: 0.9212
35552/60000 [================>.............] - ETA: 40s - loss: 0.2551 - categorical_accuracy: 0.9212
35584/60000 [================>.............] - ETA: 40s - loss: 0.2551 - categorical_accuracy: 0.9212
35616/60000 [================>.............] - ETA: 40s - loss: 0.2549 - categorical_accuracy: 0.9213
35648/60000 [================>.............] - ETA: 40s - loss: 0.2548 - categorical_accuracy: 0.9213
35680/60000 [================>.............] - ETA: 40s - loss: 0.2548 - categorical_accuracy: 0.9213
35712/60000 [================>.............] - ETA: 40s - loss: 0.2547 - categorical_accuracy: 0.9214
35744/60000 [================>.............] - ETA: 40s - loss: 0.2546 - categorical_accuracy: 0.9214
35776/60000 [================>.............] - ETA: 40s - loss: 0.2548 - categorical_accuracy: 0.9213
35808/60000 [================>.............] - ETA: 40s - loss: 0.2547 - categorical_accuracy: 0.9214
35840/60000 [================>.............] - ETA: 40s - loss: 0.2545 - categorical_accuracy: 0.9214
35872/60000 [================>.............] - ETA: 40s - loss: 0.2544 - categorical_accuracy: 0.9214
35904/60000 [================>.............] - ETA: 40s - loss: 0.2542 - categorical_accuracy: 0.9215
35936/60000 [================>.............] - ETA: 40s - loss: 0.2540 - categorical_accuracy: 0.9215
35968/60000 [================>.............] - ETA: 40s - loss: 0.2539 - categorical_accuracy: 0.9216
36000/60000 [=================>............] - ETA: 39s - loss: 0.2538 - categorical_accuracy: 0.9216
36032/60000 [=================>............] - ETA: 39s - loss: 0.2537 - categorical_accuracy: 0.9216
36064/60000 [=================>............] - ETA: 39s - loss: 0.2535 - categorical_accuracy: 0.9217
36096/60000 [=================>............] - ETA: 39s - loss: 0.2534 - categorical_accuracy: 0.9217
36128/60000 [=================>............] - ETA: 39s - loss: 0.2533 - categorical_accuracy: 0.9217
36160/60000 [=================>............] - ETA: 39s - loss: 0.2531 - categorical_accuracy: 0.9218
36192/60000 [=================>............] - ETA: 39s - loss: 0.2529 - categorical_accuracy: 0.9218
36256/60000 [=================>............] - ETA: 39s - loss: 0.2526 - categorical_accuracy: 0.9219
36288/60000 [=================>............] - ETA: 39s - loss: 0.2524 - categorical_accuracy: 0.9220
36320/60000 [=================>............] - ETA: 39s - loss: 0.2523 - categorical_accuracy: 0.9221
36352/60000 [=================>............] - ETA: 39s - loss: 0.2521 - categorical_accuracy: 0.9221
36384/60000 [=================>............] - ETA: 39s - loss: 0.2520 - categorical_accuracy: 0.9221
36416/60000 [=================>............] - ETA: 39s - loss: 0.2518 - categorical_accuracy: 0.9222
36448/60000 [=================>............] - ETA: 39s - loss: 0.2518 - categorical_accuracy: 0.9222
36480/60000 [=================>............] - ETA: 39s - loss: 0.2517 - categorical_accuracy: 0.9222
36512/60000 [=================>............] - ETA: 39s - loss: 0.2518 - categorical_accuracy: 0.9222
36544/60000 [=================>............] - ETA: 39s - loss: 0.2516 - categorical_accuracy: 0.9223
36576/60000 [=================>............] - ETA: 39s - loss: 0.2514 - categorical_accuracy: 0.9223
36608/60000 [=================>............] - ETA: 38s - loss: 0.2512 - categorical_accuracy: 0.9224
36640/60000 [=================>............] - ETA: 38s - loss: 0.2512 - categorical_accuracy: 0.9224
36672/60000 [=================>............] - ETA: 38s - loss: 0.2510 - categorical_accuracy: 0.9225
36704/60000 [=================>............] - ETA: 38s - loss: 0.2508 - categorical_accuracy: 0.9225
36736/60000 [=================>............] - ETA: 38s - loss: 0.2507 - categorical_accuracy: 0.9226
36768/60000 [=================>............] - ETA: 38s - loss: 0.2506 - categorical_accuracy: 0.9226
36800/60000 [=================>............] - ETA: 38s - loss: 0.2504 - categorical_accuracy: 0.9226
36832/60000 [=================>............] - ETA: 38s - loss: 0.2503 - categorical_accuracy: 0.9226
36864/60000 [=================>............] - ETA: 38s - loss: 0.2503 - categorical_accuracy: 0.9226
36896/60000 [=================>............] - ETA: 38s - loss: 0.2502 - categorical_accuracy: 0.9227
36928/60000 [=================>............] - ETA: 38s - loss: 0.2500 - categorical_accuracy: 0.9227
36960/60000 [=================>............] - ETA: 38s - loss: 0.2499 - categorical_accuracy: 0.9228
36992/60000 [=================>............] - ETA: 38s - loss: 0.2498 - categorical_accuracy: 0.9228
37056/60000 [=================>............] - ETA: 38s - loss: 0.2495 - categorical_accuracy: 0.9229
37120/60000 [=================>............] - ETA: 38s - loss: 0.2492 - categorical_accuracy: 0.9230
37152/60000 [=================>............] - ETA: 38s - loss: 0.2491 - categorical_accuracy: 0.9230
37184/60000 [=================>............] - ETA: 37s - loss: 0.2489 - categorical_accuracy: 0.9231
37248/60000 [=================>............] - ETA: 37s - loss: 0.2488 - categorical_accuracy: 0.9231
37280/60000 [=================>............] - ETA: 37s - loss: 0.2486 - categorical_accuracy: 0.9232
37312/60000 [=================>............] - ETA: 37s - loss: 0.2484 - categorical_accuracy: 0.9232
37344/60000 [=================>............] - ETA: 37s - loss: 0.2484 - categorical_accuracy: 0.9233
37376/60000 [=================>............] - ETA: 37s - loss: 0.2482 - categorical_accuracy: 0.9233
37408/60000 [=================>............] - ETA: 37s - loss: 0.2481 - categorical_accuracy: 0.9233
37440/60000 [=================>............] - ETA: 37s - loss: 0.2480 - categorical_accuracy: 0.9233
37472/60000 [=================>............] - ETA: 37s - loss: 0.2478 - categorical_accuracy: 0.9234
37536/60000 [=================>............] - ETA: 37s - loss: 0.2475 - categorical_accuracy: 0.9235
37568/60000 [=================>............] - ETA: 37s - loss: 0.2473 - categorical_accuracy: 0.9236
37600/60000 [=================>............] - ETA: 37s - loss: 0.2473 - categorical_accuracy: 0.9236
37632/60000 [=================>............] - ETA: 37s - loss: 0.2472 - categorical_accuracy: 0.9237
37664/60000 [=================>............] - ETA: 37s - loss: 0.2470 - categorical_accuracy: 0.9237
37728/60000 [=================>............] - ETA: 37s - loss: 0.2468 - categorical_accuracy: 0.9238
37760/60000 [=================>............] - ETA: 37s - loss: 0.2467 - categorical_accuracy: 0.9238
37792/60000 [=================>............] - ETA: 36s - loss: 0.2465 - categorical_accuracy: 0.9239
37856/60000 [=================>............] - ETA: 36s - loss: 0.2462 - categorical_accuracy: 0.9240
37888/60000 [=================>............] - ETA: 36s - loss: 0.2460 - categorical_accuracy: 0.9240
37920/60000 [=================>............] - ETA: 36s - loss: 0.2460 - categorical_accuracy: 0.9241
37952/60000 [=================>............] - ETA: 36s - loss: 0.2459 - categorical_accuracy: 0.9241
38016/60000 [==================>...........] - ETA: 36s - loss: 0.2457 - categorical_accuracy: 0.9242
38080/60000 [==================>...........] - ETA: 36s - loss: 0.2453 - categorical_accuracy: 0.9243
38112/60000 [==================>...........] - ETA: 36s - loss: 0.2453 - categorical_accuracy: 0.9244
38176/60000 [==================>...........] - ETA: 36s - loss: 0.2450 - categorical_accuracy: 0.9245
38208/60000 [==================>...........] - ETA: 36s - loss: 0.2451 - categorical_accuracy: 0.9244
38240/60000 [==================>...........] - ETA: 36s - loss: 0.2450 - categorical_accuracy: 0.9245
38272/60000 [==================>...........] - ETA: 36s - loss: 0.2448 - categorical_accuracy: 0.9245
38304/60000 [==================>...........] - ETA: 36s - loss: 0.2447 - categorical_accuracy: 0.9246
38336/60000 [==================>...........] - ETA: 36s - loss: 0.2445 - categorical_accuracy: 0.9246
38368/60000 [==================>...........] - ETA: 36s - loss: 0.2445 - categorical_accuracy: 0.9246
38432/60000 [==================>...........] - ETA: 35s - loss: 0.2442 - categorical_accuracy: 0.9247
38464/60000 [==================>...........] - ETA: 35s - loss: 0.2441 - categorical_accuracy: 0.9247
38528/60000 [==================>...........] - ETA: 35s - loss: 0.2438 - categorical_accuracy: 0.9248
38560/60000 [==================>...........] - ETA: 35s - loss: 0.2436 - categorical_accuracy: 0.9248
38592/60000 [==================>...........] - ETA: 35s - loss: 0.2437 - categorical_accuracy: 0.9248
38624/60000 [==================>...........] - ETA: 35s - loss: 0.2435 - categorical_accuracy: 0.9249
38656/60000 [==================>...........] - ETA: 35s - loss: 0.2433 - categorical_accuracy: 0.9250
38688/60000 [==================>...........] - ETA: 35s - loss: 0.2431 - categorical_accuracy: 0.9250
38720/60000 [==================>...........] - ETA: 35s - loss: 0.2430 - categorical_accuracy: 0.9251
38752/60000 [==================>...........] - ETA: 35s - loss: 0.2428 - categorical_accuracy: 0.9251
38784/60000 [==================>...........] - ETA: 35s - loss: 0.2429 - categorical_accuracy: 0.9251
38816/60000 [==================>...........] - ETA: 35s - loss: 0.2428 - categorical_accuracy: 0.9251
38848/60000 [==================>...........] - ETA: 35s - loss: 0.2426 - categorical_accuracy: 0.9252
38880/60000 [==================>...........] - ETA: 35s - loss: 0.2425 - categorical_accuracy: 0.9252
38912/60000 [==================>...........] - ETA: 35s - loss: 0.2424 - categorical_accuracy: 0.9253
38944/60000 [==================>...........] - ETA: 35s - loss: 0.2424 - categorical_accuracy: 0.9253
38976/60000 [==================>...........] - ETA: 35s - loss: 0.2422 - categorical_accuracy: 0.9253
39008/60000 [==================>...........] - ETA: 34s - loss: 0.2422 - categorical_accuracy: 0.9253
39040/60000 [==================>...........] - ETA: 34s - loss: 0.2422 - categorical_accuracy: 0.9253
39072/60000 [==================>...........] - ETA: 34s - loss: 0.2420 - categorical_accuracy: 0.9254
39104/60000 [==================>...........] - ETA: 34s - loss: 0.2419 - categorical_accuracy: 0.9254
39136/60000 [==================>...........] - ETA: 34s - loss: 0.2417 - categorical_accuracy: 0.9255
39168/60000 [==================>...........] - ETA: 34s - loss: 0.2415 - categorical_accuracy: 0.9255
39232/60000 [==================>...........] - ETA: 34s - loss: 0.2414 - categorical_accuracy: 0.9255
39296/60000 [==================>...........] - ETA: 34s - loss: 0.2411 - categorical_accuracy: 0.9256
39328/60000 [==================>...........] - ETA: 34s - loss: 0.2410 - categorical_accuracy: 0.9256
39392/60000 [==================>...........] - ETA: 34s - loss: 0.2407 - categorical_accuracy: 0.9257
39424/60000 [==================>...........] - ETA: 34s - loss: 0.2406 - categorical_accuracy: 0.9257
39456/60000 [==================>...........] - ETA: 34s - loss: 0.2406 - categorical_accuracy: 0.9257
39488/60000 [==================>...........] - ETA: 34s - loss: 0.2407 - categorical_accuracy: 0.9257
39520/60000 [==================>...........] - ETA: 34s - loss: 0.2406 - categorical_accuracy: 0.9258
39552/60000 [==================>...........] - ETA: 34s - loss: 0.2404 - categorical_accuracy: 0.9258
39584/60000 [==================>...........] - ETA: 33s - loss: 0.2404 - categorical_accuracy: 0.9258
39616/60000 [==================>...........] - ETA: 33s - loss: 0.2403 - categorical_accuracy: 0.9258
39680/60000 [==================>...........] - ETA: 33s - loss: 0.2401 - categorical_accuracy: 0.9259
39712/60000 [==================>...........] - ETA: 33s - loss: 0.2399 - categorical_accuracy: 0.9259
39744/60000 [==================>...........] - ETA: 33s - loss: 0.2398 - categorical_accuracy: 0.9260
39776/60000 [==================>...........] - ETA: 33s - loss: 0.2397 - categorical_accuracy: 0.9260
39840/60000 [==================>...........] - ETA: 33s - loss: 0.2393 - categorical_accuracy: 0.9261
39872/60000 [==================>...........] - ETA: 33s - loss: 0.2393 - categorical_accuracy: 0.9261
39904/60000 [==================>...........] - ETA: 33s - loss: 0.2392 - categorical_accuracy: 0.9262
39936/60000 [==================>...........] - ETA: 33s - loss: 0.2390 - categorical_accuracy: 0.9262
39968/60000 [==================>...........] - ETA: 33s - loss: 0.2389 - categorical_accuracy: 0.9262
40000/60000 [===================>..........] - ETA: 33s - loss: 0.2389 - categorical_accuracy: 0.9263
40032/60000 [===================>..........] - ETA: 33s - loss: 0.2388 - categorical_accuracy: 0.9263
40064/60000 [===================>..........] - ETA: 33s - loss: 0.2386 - categorical_accuracy: 0.9264
40096/60000 [===================>..........] - ETA: 33s - loss: 0.2385 - categorical_accuracy: 0.9264
40128/60000 [===================>..........] - ETA: 33s - loss: 0.2384 - categorical_accuracy: 0.9265
40160/60000 [===================>..........] - ETA: 33s - loss: 0.2384 - categorical_accuracy: 0.9265
40192/60000 [===================>..........] - ETA: 32s - loss: 0.2382 - categorical_accuracy: 0.9265
40224/60000 [===================>..........] - ETA: 32s - loss: 0.2381 - categorical_accuracy: 0.9265
40288/60000 [===================>..........] - ETA: 32s - loss: 0.2383 - categorical_accuracy: 0.9265
40320/60000 [===================>..........] - ETA: 32s - loss: 0.2382 - categorical_accuracy: 0.9266
40352/60000 [===================>..........] - ETA: 32s - loss: 0.2380 - categorical_accuracy: 0.9266
40384/60000 [===================>..........] - ETA: 32s - loss: 0.2379 - categorical_accuracy: 0.9267
40416/60000 [===================>..........] - ETA: 32s - loss: 0.2377 - categorical_accuracy: 0.9267
40448/60000 [===================>..........] - ETA: 32s - loss: 0.2377 - categorical_accuracy: 0.9267
40480/60000 [===================>..........] - ETA: 32s - loss: 0.2376 - categorical_accuracy: 0.9268
40512/60000 [===================>..........] - ETA: 32s - loss: 0.2375 - categorical_accuracy: 0.9268
40544/60000 [===================>..........] - ETA: 32s - loss: 0.2375 - categorical_accuracy: 0.9268
40576/60000 [===================>..........] - ETA: 32s - loss: 0.2373 - categorical_accuracy: 0.9269
40608/60000 [===================>..........] - ETA: 32s - loss: 0.2371 - categorical_accuracy: 0.9269
40640/60000 [===================>..........] - ETA: 32s - loss: 0.2371 - categorical_accuracy: 0.9269
40672/60000 [===================>..........] - ETA: 32s - loss: 0.2371 - categorical_accuracy: 0.9270
40704/60000 [===================>..........] - ETA: 32s - loss: 0.2371 - categorical_accuracy: 0.9269
40736/60000 [===================>..........] - ETA: 32s - loss: 0.2370 - categorical_accuracy: 0.9269
40768/60000 [===================>..........] - ETA: 32s - loss: 0.2369 - categorical_accuracy: 0.9270
40800/60000 [===================>..........] - ETA: 31s - loss: 0.2367 - categorical_accuracy: 0.9270
40832/60000 [===================>..........] - ETA: 31s - loss: 0.2366 - categorical_accuracy: 0.9271
40864/60000 [===================>..........] - ETA: 31s - loss: 0.2365 - categorical_accuracy: 0.9271
40896/60000 [===================>..........] - ETA: 31s - loss: 0.2363 - categorical_accuracy: 0.9271
40928/60000 [===================>..........] - ETA: 31s - loss: 0.2363 - categorical_accuracy: 0.9271
40960/60000 [===================>..........] - ETA: 31s - loss: 0.2361 - categorical_accuracy: 0.9272
40992/60000 [===================>..........] - ETA: 31s - loss: 0.2360 - categorical_accuracy: 0.9272
41056/60000 [===================>..........] - ETA: 31s - loss: 0.2359 - categorical_accuracy: 0.9272
41120/60000 [===================>..........] - ETA: 31s - loss: 0.2357 - categorical_accuracy: 0.9273
41152/60000 [===================>..........] - ETA: 31s - loss: 0.2356 - categorical_accuracy: 0.9273
41184/60000 [===================>..........] - ETA: 31s - loss: 0.2355 - categorical_accuracy: 0.9274
41248/60000 [===================>..........] - ETA: 31s - loss: 0.2354 - categorical_accuracy: 0.9274
41312/60000 [===================>..........] - ETA: 31s - loss: 0.2351 - categorical_accuracy: 0.9275
41344/60000 [===================>..........] - ETA: 31s - loss: 0.2350 - categorical_accuracy: 0.9275
41376/60000 [===================>..........] - ETA: 31s - loss: 0.2349 - categorical_accuracy: 0.9275
41440/60000 [===================>..........] - ETA: 30s - loss: 0.2345 - categorical_accuracy: 0.9277
41504/60000 [===================>..........] - ETA: 30s - loss: 0.2346 - categorical_accuracy: 0.9277
41536/60000 [===================>..........] - ETA: 30s - loss: 0.2346 - categorical_accuracy: 0.9277
41568/60000 [===================>..........] - ETA: 30s - loss: 0.2344 - categorical_accuracy: 0.9277
41600/60000 [===================>..........] - ETA: 30s - loss: 0.2342 - categorical_accuracy: 0.9278
41632/60000 [===================>..........] - ETA: 30s - loss: 0.2342 - categorical_accuracy: 0.9278
41664/60000 [===================>..........] - ETA: 30s - loss: 0.2341 - categorical_accuracy: 0.9279
41696/60000 [===================>..........] - ETA: 30s - loss: 0.2339 - categorical_accuracy: 0.9279
41728/60000 [===================>..........] - ETA: 30s - loss: 0.2339 - categorical_accuracy: 0.9279
41760/60000 [===================>..........] - ETA: 30s - loss: 0.2337 - categorical_accuracy: 0.9280
41792/60000 [===================>..........] - ETA: 30s - loss: 0.2336 - categorical_accuracy: 0.9280
41824/60000 [===================>..........] - ETA: 30s - loss: 0.2336 - categorical_accuracy: 0.9281
41856/60000 [===================>..........] - ETA: 30s - loss: 0.2335 - categorical_accuracy: 0.9281
41888/60000 [===================>..........] - ETA: 30s - loss: 0.2333 - categorical_accuracy: 0.9282
41920/60000 [===================>..........] - ETA: 30s - loss: 0.2332 - categorical_accuracy: 0.9282
41952/60000 [===================>..........] - ETA: 30s - loss: 0.2331 - categorical_accuracy: 0.9283
41984/60000 [===================>..........] - ETA: 29s - loss: 0.2330 - categorical_accuracy: 0.9283
42016/60000 [====================>.........] - ETA: 29s - loss: 0.2328 - categorical_accuracy: 0.9284
42048/60000 [====================>.........] - ETA: 29s - loss: 0.2327 - categorical_accuracy: 0.9284
42080/60000 [====================>.........] - ETA: 29s - loss: 0.2325 - categorical_accuracy: 0.9285
42112/60000 [====================>.........] - ETA: 29s - loss: 0.2325 - categorical_accuracy: 0.9285
42144/60000 [====================>.........] - ETA: 29s - loss: 0.2323 - categorical_accuracy: 0.9286
42176/60000 [====================>.........] - ETA: 29s - loss: 0.2322 - categorical_accuracy: 0.9286
42208/60000 [====================>.........] - ETA: 29s - loss: 0.2320 - categorical_accuracy: 0.9287
42240/60000 [====================>.........] - ETA: 29s - loss: 0.2320 - categorical_accuracy: 0.9287
42272/60000 [====================>.........] - ETA: 29s - loss: 0.2318 - categorical_accuracy: 0.9287
42304/60000 [====================>.........] - ETA: 29s - loss: 0.2317 - categorical_accuracy: 0.9288
42336/60000 [====================>.........] - ETA: 29s - loss: 0.2315 - categorical_accuracy: 0.9289
42368/60000 [====================>.........] - ETA: 29s - loss: 0.2314 - categorical_accuracy: 0.9289
42432/60000 [====================>.........] - ETA: 29s - loss: 0.2312 - categorical_accuracy: 0.9289
42464/60000 [====================>.........] - ETA: 29s - loss: 0.2316 - categorical_accuracy: 0.9289
42496/60000 [====================>.........] - ETA: 29s - loss: 0.2314 - categorical_accuracy: 0.9289
42560/60000 [====================>.........] - ETA: 29s - loss: 0.2311 - categorical_accuracy: 0.9290
42592/60000 [====================>.........] - ETA: 28s - loss: 0.2312 - categorical_accuracy: 0.9290
42656/60000 [====================>.........] - ETA: 28s - loss: 0.2310 - categorical_accuracy: 0.9291
42688/60000 [====================>.........] - ETA: 28s - loss: 0.2309 - categorical_accuracy: 0.9291
42720/60000 [====================>.........] - ETA: 28s - loss: 0.2307 - categorical_accuracy: 0.9292
42752/60000 [====================>.........] - ETA: 28s - loss: 0.2306 - categorical_accuracy: 0.9292
42784/60000 [====================>.........] - ETA: 28s - loss: 0.2305 - categorical_accuracy: 0.9292
42848/60000 [====================>.........] - ETA: 28s - loss: 0.2304 - categorical_accuracy: 0.9293
42880/60000 [====================>.........] - ETA: 28s - loss: 0.2303 - categorical_accuracy: 0.9293
42912/60000 [====================>.........] - ETA: 28s - loss: 0.2301 - categorical_accuracy: 0.9294
42944/60000 [====================>.........] - ETA: 28s - loss: 0.2301 - categorical_accuracy: 0.9294
42976/60000 [====================>.........] - ETA: 28s - loss: 0.2301 - categorical_accuracy: 0.9294
43008/60000 [====================>.........] - ETA: 28s - loss: 0.2299 - categorical_accuracy: 0.9294
43040/60000 [====================>.........] - ETA: 28s - loss: 0.2298 - categorical_accuracy: 0.9295
43072/60000 [====================>.........] - ETA: 28s - loss: 0.2296 - categorical_accuracy: 0.9295
43104/60000 [====================>.........] - ETA: 28s - loss: 0.2295 - categorical_accuracy: 0.9296
43136/60000 [====================>.........] - ETA: 28s - loss: 0.2295 - categorical_accuracy: 0.9296
43168/60000 [====================>.........] - ETA: 28s - loss: 0.2295 - categorical_accuracy: 0.9296
43200/60000 [====================>.........] - ETA: 27s - loss: 0.2294 - categorical_accuracy: 0.9296
43232/60000 [====================>.........] - ETA: 27s - loss: 0.2293 - categorical_accuracy: 0.9296
43296/60000 [====================>.........] - ETA: 27s - loss: 0.2291 - categorical_accuracy: 0.9297
43360/60000 [====================>.........] - ETA: 27s - loss: 0.2289 - categorical_accuracy: 0.9298
43424/60000 [====================>.........] - ETA: 27s - loss: 0.2286 - categorical_accuracy: 0.9299
43488/60000 [====================>.........] - ETA: 27s - loss: 0.2286 - categorical_accuracy: 0.9299
43520/60000 [====================>.........] - ETA: 27s - loss: 0.2284 - categorical_accuracy: 0.9300
43584/60000 [====================>.........] - ETA: 27s - loss: 0.2281 - categorical_accuracy: 0.9301
43648/60000 [====================>.........] - ETA: 27s - loss: 0.2279 - categorical_accuracy: 0.9301
43680/60000 [====================>.........] - ETA: 27s - loss: 0.2278 - categorical_accuracy: 0.9301
43712/60000 [====================>.........] - ETA: 27s - loss: 0.2278 - categorical_accuracy: 0.9301
43744/60000 [====================>.........] - ETA: 27s - loss: 0.2276 - categorical_accuracy: 0.9301
43776/60000 [====================>.........] - ETA: 27s - loss: 0.2276 - categorical_accuracy: 0.9302
43840/60000 [====================>.........] - ETA: 26s - loss: 0.2273 - categorical_accuracy: 0.9302
43872/60000 [====================>.........] - ETA: 26s - loss: 0.2271 - categorical_accuracy: 0.9303
43936/60000 [====================>.........] - ETA: 26s - loss: 0.2270 - categorical_accuracy: 0.9303
43968/60000 [====================>.........] - ETA: 26s - loss: 0.2269 - categorical_accuracy: 0.9303
44000/60000 [=====================>........] - ETA: 26s - loss: 0.2267 - categorical_accuracy: 0.9304
44032/60000 [=====================>........] - ETA: 26s - loss: 0.2267 - categorical_accuracy: 0.9304
44064/60000 [=====================>........] - ETA: 26s - loss: 0.2265 - categorical_accuracy: 0.9305
44128/60000 [=====================>........] - ETA: 26s - loss: 0.2263 - categorical_accuracy: 0.9305
44160/60000 [=====================>........] - ETA: 26s - loss: 0.2262 - categorical_accuracy: 0.9305
44192/60000 [=====================>........] - ETA: 26s - loss: 0.2263 - categorical_accuracy: 0.9305
44224/60000 [=====================>........] - ETA: 26s - loss: 0.2263 - categorical_accuracy: 0.9305
44256/60000 [=====================>........] - ETA: 26s - loss: 0.2261 - categorical_accuracy: 0.9306
44288/60000 [=====================>........] - ETA: 26s - loss: 0.2260 - categorical_accuracy: 0.9306
44320/60000 [=====================>........] - ETA: 26s - loss: 0.2258 - categorical_accuracy: 0.9307
44352/60000 [=====================>........] - ETA: 26s - loss: 0.2257 - categorical_accuracy: 0.9307
44416/60000 [=====================>........] - ETA: 25s - loss: 0.2258 - categorical_accuracy: 0.9308
44448/60000 [=====================>........] - ETA: 25s - loss: 0.2257 - categorical_accuracy: 0.9308
44480/60000 [=====================>........] - ETA: 25s - loss: 0.2256 - categorical_accuracy: 0.9308
44512/60000 [=====================>........] - ETA: 25s - loss: 0.2255 - categorical_accuracy: 0.9309
44544/60000 [=====================>........] - ETA: 25s - loss: 0.2254 - categorical_accuracy: 0.9309
44576/60000 [=====================>........] - ETA: 25s - loss: 0.2253 - categorical_accuracy: 0.9309
44608/60000 [=====================>........] - ETA: 25s - loss: 0.2252 - categorical_accuracy: 0.9310
44640/60000 [=====================>........] - ETA: 25s - loss: 0.2250 - categorical_accuracy: 0.9310
44672/60000 [=====================>........] - ETA: 25s - loss: 0.2249 - categorical_accuracy: 0.9310
44704/60000 [=====================>........] - ETA: 25s - loss: 0.2248 - categorical_accuracy: 0.9311
44736/60000 [=====================>........] - ETA: 25s - loss: 0.2247 - categorical_accuracy: 0.9311
44768/60000 [=====================>........] - ETA: 25s - loss: 0.2245 - categorical_accuracy: 0.9312
44800/60000 [=====================>........] - ETA: 25s - loss: 0.2244 - categorical_accuracy: 0.9312
44832/60000 [=====================>........] - ETA: 25s - loss: 0.2243 - categorical_accuracy: 0.9312
44864/60000 [=====================>........] - ETA: 25s - loss: 0.2242 - categorical_accuracy: 0.9313
44896/60000 [=====================>........] - ETA: 25s - loss: 0.2240 - categorical_accuracy: 0.9313
44928/60000 [=====================>........] - ETA: 25s - loss: 0.2239 - categorical_accuracy: 0.9313
44960/60000 [=====================>........] - ETA: 25s - loss: 0.2238 - categorical_accuracy: 0.9314
44992/60000 [=====================>........] - ETA: 24s - loss: 0.2236 - categorical_accuracy: 0.9314
45024/60000 [=====================>........] - ETA: 24s - loss: 0.2235 - categorical_accuracy: 0.9315
45056/60000 [=====================>........] - ETA: 24s - loss: 0.2236 - categorical_accuracy: 0.9315
45088/60000 [=====================>........] - ETA: 24s - loss: 0.2236 - categorical_accuracy: 0.9315
45120/60000 [=====================>........] - ETA: 24s - loss: 0.2236 - categorical_accuracy: 0.9315
45152/60000 [=====================>........] - ETA: 24s - loss: 0.2235 - categorical_accuracy: 0.9315
45184/60000 [=====================>........] - ETA: 24s - loss: 0.2234 - categorical_accuracy: 0.9316
45216/60000 [=====================>........] - ETA: 24s - loss: 0.2234 - categorical_accuracy: 0.9316
45248/60000 [=====================>........] - ETA: 24s - loss: 0.2233 - categorical_accuracy: 0.9316
45312/60000 [=====================>........] - ETA: 24s - loss: 0.2233 - categorical_accuracy: 0.9317
45344/60000 [=====================>........] - ETA: 24s - loss: 0.2232 - categorical_accuracy: 0.9317
45376/60000 [=====================>........] - ETA: 24s - loss: 0.2231 - categorical_accuracy: 0.9317
45408/60000 [=====================>........] - ETA: 24s - loss: 0.2229 - categorical_accuracy: 0.9318
45472/60000 [=====================>........] - ETA: 24s - loss: 0.2229 - categorical_accuracy: 0.9319
45536/60000 [=====================>........] - ETA: 24s - loss: 0.2227 - categorical_accuracy: 0.9319
45568/60000 [=====================>........] - ETA: 24s - loss: 0.2226 - categorical_accuracy: 0.9320
45600/60000 [=====================>........] - ETA: 23s - loss: 0.2226 - categorical_accuracy: 0.9320
45664/60000 [=====================>........] - ETA: 23s - loss: 0.2224 - categorical_accuracy: 0.9320
45728/60000 [=====================>........] - ETA: 23s - loss: 0.2224 - categorical_accuracy: 0.9320
45760/60000 [=====================>........] - ETA: 23s - loss: 0.2223 - categorical_accuracy: 0.9320
45792/60000 [=====================>........] - ETA: 23s - loss: 0.2222 - categorical_accuracy: 0.9321
45824/60000 [=====================>........] - ETA: 23s - loss: 0.2221 - categorical_accuracy: 0.9321
45856/60000 [=====================>........] - ETA: 23s - loss: 0.2221 - categorical_accuracy: 0.9321
45888/60000 [=====================>........] - ETA: 23s - loss: 0.2220 - categorical_accuracy: 0.9321
45920/60000 [=====================>........] - ETA: 23s - loss: 0.2219 - categorical_accuracy: 0.9321
45952/60000 [=====================>........] - ETA: 23s - loss: 0.2218 - categorical_accuracy: 0.9322
45984/60000 [=====================>........] - ETA: 23s - loss: 0.2217 - categorical_accuracy: 0.9322
46016/60000 [======================>.......] - ETA: 23s - loss: 0.2217 - categorical_accuracy: 0.9322
46048/60000 [======================>.......] - ETA: 23s - loss: 0.2215 - categorical_accuracy: 0.9322
46080/60000 [======================>.......] - ETA: 23s - loss: 0.2214 - categorical_accuracy: 0.9323
46144/60000 [======================>.......] - ETA: 23s - loss: 0.2211 - categorical_accuracy: 0.9324
46208/60000 [======================>.......] - ETA: 22s - loss: 0.2209 - categorical_accuracy: 0.9325
46272/60000 [======================>.......] - ETA: 22s - loss: 0.2207 - categorical_accuracy: 0.9325
46304/60000 [======================>.......] - ETA: 22s - loss: 0.2206 - categorical_accuracy: 0.9326
46336/60000 [======================>.......] - ETA: 22s - loss: 0.2205 - categorical_accuracy: 0.9326
46368/60000 [======================>.......] - ETA: 22s - loss: 0.2204 - categorical_accuracy: 0.9326
46432/60000 [======================>.......] - ETA: 22s - loss: 0.2203 - categorical_accuracy: 0.9326
46496/60000 [======================>.......] - ETA: 22s - loss: 0.2202 - categorical_accuracy: 0.9327
46528/60000 [======================>.......] - ETA: 22s - loss: 0.2201 - categorical_accuracy: 0.9327
46560/60000 [======================>.......] - ETA: 22s - loss: 0.2201 - categorical_accuracy: 0.9327
46592/60000 [======================>.......] - ETA: 22s - loss: 0.2200 - categorical_accuracy: 0.9327
46624/60000 [======================>.......] - ETA: 22s - loss: 0.2199 - categorical_accuracy: 0.9327
46656/60000 [======================>.......] - ETA: 22s - loss: 0.2198 - categorical_accuracy: 0.9328
46688/60000 [======================>.......] - ETA: 22s - loss: 0.2199 - categorical_accuracy: 0.9328
46720/60000 [======================>.......] - ETA: 22s - loss: 0.2199 - categorical_accuracy: 0.9328
46752/60000 [======================>.......] - ETA: 22s - loss: 0.2198 - categorical_accuracy: 0.9328
46784/60000 [======================>.......] - ETA: 21s - loss: 0.2198 - categorical_accuracy: 0.9328
46816/60000 [======================>.......] - ETA: 21s - loss: 0.2197 - categorical_accuracy: 0.9328
46848/60000 [======================>.......] - ETA: 21s - loss: 0.2196 - categorical_accuracy: 0.9329
46880/60000 [======================>.......] - ETA: 21s - loss: 0.2195 - categorical_accuracy: 0.9329
46912/60000 [======================>.......] - ETA: 21s - loss: 0.2195 - categorical_accuracy: 0.9329
46976/60000 [======================>.......] - ETA: 21s - loss: 0.2194 - categorical_accuracy: 0.9329
47040/60000 [======================>.......] - ETA: 21s - loss: 0.2193 - categorical_accuracy: 0.9330
47072/60000 [======================>.......] - ETA: 21s - loss: 0.2192 - categorical_accuracy: 0.9330
47104/60000 [======================>.......] - ETA: 21s - loss: 0.2191 - categorical_accuracy: 0.9330
47136/60000 [======================>.......] - ETA: 21s - loss: 0.2190 - categorical_accuracy: 0.9330
47168/60000 [======================>.......] - ETA: 21s - loss: 0.2189 - categorical_accuracy: 0.9331
47200/60000 [======================>.......] - ETA: 21s - loss: 0.2188 - categorical_accuracy: 0.9331
47264/60000 [======================>.......] - ETA: 21s - loss: 0.2186 - categorical_accuracy: 0.9331
47328/60000 [======================>.......] - ETA: 21s - loss: 0.2186 - categorical_accuracy: 0.9331
47392/60000 [======================>.......] - ETA: 20s - loss: 0.2184 - categorical_accuracy: 0.9331
47424/60000 [======================>.......] - ETA: 20s - loss: 0.2183 - categorical_accuracy: 0.9332
47456/60000 [======================>.......] - ETA: 20s - loss: 0.2182 - categorical_accuracy: 0.9332
47488/60000 [======================>.......] - ETA: 20s - loss: 0.2180 - categorical_accuracy: 0.9332
47520/60000 [======================>.......] - ETA: 20s - loss: 0.2179 - categorical_accuracy: 0.9333
47552/60000 [======================>.......] - ETA: 20s - loss: 0.2179 - categorical_accuracy: 0.9333
47584/60000 [======================>.......] - ETA: 20s - loss: 0.2178 - categorical_accuracy: 0.9333
47616/60000 [======================>.......] - ETA: 20s - loss: 0.2178 - categorical_accuracy: 0.9333
47648/60000 [======================>.......] - ETA: 20s - loss: 0.2177 - categorical_accuracy: 0.9333
47680/60000 [======================>.......] - ETA: 20s - loss: 0.2177 - categorical_accuracy: 0.9333
47712/60000 [======================>.......] - ETA: 20s - loss: 0.2176 - categorical_accuracy: 0.9334
47744/60000 [======================>.......] - ETA: 20s - loss: 0.2175 - categorical_accuracy: 0.9334
47776/60000 [======================>.......] - ETA: 20s - loss: 0.2173 - categorical_accuracy: 0.9334
47808/60000 [======================>.......] - ETA: 20s - loss: 0.2172 - categorical_accuracy: 0.9335
47840/60000 [======================>.......] - ETA: 20s - loss: 0.2172 - categorical_accuracy: 0.9335
47872/60000 [======================>.......] - ETA: 20s - loss: 0.2171 - categorical_accuracy: 0.9335
47904/60000 [======================>.......] - ETA: 20s - loss: 0.2171 - categorical_accuracy: 0.9335
47936/60000 [======================>.......] - ETA: 20s - loss: 0.2170 - categorical_accuracy: 0.9336
47968/60000 [======================>.......] - ETA: 20s - loss: 0.2169 - categorical_accuracy: 0.9336
48000/60000 [=======================>......] - ETA: 19s - loss: 0.2168 - categorical_accuracy: 0.9336
48032/60000 [=======================>......] - ETA: 19s - loss: 0.2166 - categorical_accuracy: 0.9337
48064/60000 [=======================>......] - ETA: 19s - loss: 0.2165 - categorical_accuracy: 0.9337
48096/60000 [=======================>......] - ETA: 19s - loss: 0.2164 - categorical_accuracy: 0.9337
48128/60000 [=======================>......] - ETA: 19s - loss: 0.2165 - categorical_accuracy: 0.9337
48160/60000 [=======================>......] - ETA: 19s - loss: 0.2163 - categorical_accuracy: 0.9338
48192/60000 [=======================>......] - ETA: 19s - loss: 0.2162 - categorical_accuracy: 0.9338
48224/60000 [=======================>......] - ETA: 19s - loss: 0.2161 - categorical_accuracy: 0.9339
48256/60000 [=======================>......] - ETA: 19s - loss: 0.2160 - categorical_accuracy: 0.9339
48288/60000 [=======================>......] - ETA: 19s - loss: 0.2158 - categorical_accuracy: 0.9339
48320/60000 [=======================>......] - ETA: 19s - loss: 0.2157 - categorical_accuracy: 0.9340
48352/60000 [=======================>......] - ETA: 19s - loss: 0.2156 - categorical_accuracy: 0.9340
48384/60000 [=======================>......] - ETA: 19s - loss: 0.2157 - categorical_accuracy: 0.9340
48448/60000 [=======================>......] - ETA: 19s - loss: 0.2155 - categorical_accuracy: 0.9341
48480/60000 [=======================>......] - ETA: 19s - loss: 0.2154 - categorical_accuracy: 0.9341
48512/60000 [=======================>......] - ETA: 19s - loss: 0.2155 - categorical_accuracy: 0.9341
48576/60000 [=======================>......] - ETA: 19s - loss: 0.2153 - categorical_accuracy: 0.9341
48608/60000 [=======================>......] - ETA: 18s - loss: 0.2152 - categorical_accuracy: 0.9342
48640/60000 [=======================>......] - ETA: 18s - loss: 0.2151 - categorical_accuracy: 0.9342
48672/60000 [=======================>......] - ETA: 18s - loss: 0.2150 - categorical_accuracy: 0.9343
48704/60000 [=======================>......] - ETA: 18s - loss: 0.2150 - categorical_accuracy: 0.9343
48736/60000 [=======================>......] - ETA: 18s - loss: 0.2150 - categorical_accuracy: 0.9343
48768/60000 [=======================>......] - ETA: 18s - loss: 0.2149 - categorical_accuracy: 0.9343
48800/60000 [=======================>......] - ETA: 18s - loss: 0.2149 - categorical_accuracy: 0.9343
48832/60000 [=======================>......] - ETA: 18s - loss: 0.2147 - categorical_accuracy: 0.9343
48864/60000 [=======================>......] - ETA: 18s - loss: 0.2146 - categorical_accuracy: 0.9344
48896/60000 [=======================>......] - ETA: 18s - loss: 0.2146 - categorical_accuracy: 0.9344
48928/60000 [=======================>......] - ETA: 18s - loss: 0.2144 - categorical_accuracy: 0.9345
48992/60000 [=======================>......] - ETA: 18s - loss: 0.2142 - categorical_accuracy: 0.9345
49056/60000 [=======================>......] - ETA: 18s - loss: 0.2142 - categorical_accuracy: 0.9345
49120/60000 [=======================>......] - ETA: 18s - loss: 0.2140 - categorical_accuracy: 0.9346
49184/60000 [=======================>......] - ETA: 17s - loss: 0.2140 - categorical_accuracy: 0.9347
49216/60000 [=======================>......] - ETA: 17s - loss: 0.2139 - categorical_accuracy: 0.9347
49248/60000 [=======================>......] - ETA: 17s - loss: 0.2138 - categorical_accuracy: 0.9347
49280/60000 [=======================>......] - ETA: 17s - loss: 0.2138 - categorical_accuracy: 0.9347
49312/60000 [=======================>......] - ETA: 17s - loss: 0.2138 - categorical_accuracy: 0.9347
49344/60000 [=======================>......] - ETA: 17s - loss: 0.2137 - categorical_accuracy: 0.9347
49376/60000 [=======================>......] - ETA: 17s - loss: 0.2136 - categorical_accuracy: 0.9348
49408/60000 [=======================>......] - ETA: 17s - loss: 0.2135 - categorical_accuracy: 0.9348
49440/60000 [=======================>......] - ETA: 17s - loss: 0.2134 - categorical_accuracy: 0.9348
49472/60000 [=======================>......] - ETA: 17s - loss: 0.2133 - categorical_accuracy: 0.9349
49536/60000 [=======================>......] - ETA: 17s - loss: 0.2132 - categorical_accuracy: 0.9349
49568/60000 [=======================>......] - ETA: 17s - loss: 0.2131 - categorical_accuracy: 0.9349
49600/60000 [=======================>......] - ETA: 17s - loss: 0.2130 - categorical_accuracy: 0.9350
49632/60000 [=======================>......] - ETA: 17s - loss: 0.2129 - categorical_accuracy: 0.9350
49696/60000 [=======================>......] - ETA: 17s - loss: 0.2127 - categorical_accuracy: 0.9350
49728/60000 [=======================>......] - ETA: 17s - loss: 0.2126 - categorical_accuracy: 0.9350
49760/60000 [=======================>......] - ETA: 17s - loss: 0.2127 - categorical_accuracy: 0.9350
49824/60000 [=======================>......] - ETA: 16s - loss: 0.2128 - categorical_accuracy: 0.9350
49888/60000 [=======================>......] - ETA: 16s - loss: 0.2126 - categorical_accuracy: 0.9351
49920/60000 [=======================>......] - ETA: 16s - loss: 0.2125 - categorical_accuracy: 0.9351
49952/60000 [=======================>......] - ETA: 16s - loss: 0.2124 - categorical_accuracy: 0.9351
49984/60000 [=======================>......] - ETA: 16s - loss: 0.2123 - categorical_accuracy: 0.9352
50016/60000 [========================>.....] - ETA: 16s - loss: 0.2122 - categorical_accuracy: 0.9352
50080/60000 [========================>.....] - ETA: 16s - loss: 0.2123 - categorical_accuracy: 0.9352
50112/60000 [========================>.....] - ETA: 16s - loss: 0.2122 - categorical_accuracy: 0.9352
50144/60000 [========================>.....] - ETA: 16s - loss: 0.2122 - categorical_accuracy: 0.9352
50176/60000 [========================>.....] - ETA: 16s - loss: 0.2121 - categorical_accuracy: 0.9353
50208/60000 [========================>.....] - ETA: 16s - loss: 0.2120 - categorical_accuracy: 0.9353
50240/60000 [========================>.....] - ETA: 16s - loss: 0.2119 - categorical_accuracy: 0.9353
50272/60000 [========================>.....] - ETA: 16s - loss: 0.2118 - categorical_accuracy: 0.9354
50304/60000 [========================>.....] - ETA: 16s - loss: 0.2117 - categorical_accuracy: 0.9354
50336/60000 [========================>.....] - ETA: 16s - loss: 0.2116 - categorical_accuracy: 0.9354
50368/60000 [========================>.....] - ETA: 16s - loss: 0.2115 - categorical_accuracy: 0.9354
50400/60000 [========================>.....] - ETA: 15s - loss: 0.2114 - categorical_accuracy: 0.9354
50432/60000 [========================>.....] - ETA: 15s - loss: 0.2113 - categorical_accuracy: 0.9355
50464/60000 [========================>.....] - ETA: 15s - loss: 0.2113 - categorical_accuracy: 0.9355
50528/60000 [========================>.....] - ETA: 15s - loss: 0.2111 - categorical_accuracy: 0.9355
50560/60000 [========================>.....] - ETA: 15s - loss: 0.2110 - categorical_accuracy: 0.9356
50624/60000 [========================>.....] - ETA: 15s - loss: 0.2108 - categorical_accuracy: 0.9356
50688/60000 [========================>.....] - ETA: 15s - loss: 0.2107 - categorical_accuracy: 0.9356
50752/60000 [========================>.....] - ETA: 15s - loss: 0.2105 - categorical_accuracy: 0.9357
50784/60000 [========================>.....] - ETA: 15s - loss: 0.2104 - categorical_accuracy: 0.9357
50816/60000 [========================>.....] - ETA: 15s - loss: 0.2104 - categorical_accuracy: 0.9357
50848/60000 [========================>.....] - ETA: 15s - loss: 0.2103 - categorical_accuracy: 0.9357
50880/60000 [========================>.....] - ETA: 15s - loss: 0.2104 - categorical_accuracy: 0.9357
50912/60000 [========================>.....] - ETA: 15s - loss: 0.2102 - categorical_accuracy: 0.9358
50944/60000 [========================>.....] - ETA: 15s - loss: 0.2101 - categorical_accuracy: 0.9358
51008/60000 [========================>.....] - ETA: 14s - loss: 0.2101 - categorical_accuracy: 0.9358
51072/60000 [========================>.....] - ETA: 14s - loss: 0.2099 - categorical_accuracy: 0.9359
51104/60000 [========================>.....] - ETA: 14s - loss: 0.2099 - categorical_accuracy: 0.9359
51136/60000 [========================>.....] - ETA: 14s - loss: 0.2099 - categorical_accuracy: 0.9359
51168/60000 [========================>.....] - ETA: 14s - loss: 0.2098 - categorical_accuracy: 0.9359
51232/60000 [========================>.....] - ETA: 14s - loss: 0.2096 - categorical_accuracy: 0.9360
51264/60000 [========================>.....] - ETA: 14s - loss: 0.2096 - categorical_accuracy: 0.9360
51296/60000 [========================>.....] - ETA: 14s - loss: 0.2094 - categorical_accuracy: 0.9360
51328/60000 [========================>.....] - ETA: 14s - loss: 0.2093 - categorical_accuracy: 0.9361
51392/60000 [========================>.....] - ETA: 14s - loss: 0.2092 - categorical_accuracy: 0.9361
51424/60000 [========================>.....] - ETA: 14s - loss: 0.2093 - categorical_accuracy: 0.9361
51456/60000 [========================>.....] - ETA: 14s - loss: 0.2092 - categorical_accuracy: 0.9361
51488/60000 [========================>.....] - ETA: 14s - loss: 0.2092 - categorical_accuracy: 0.9361
51520/60000 [========================>.....] - ETA: 14s - loss: 0.2092 - categorical_accuracy: 0.9361
51552/60000 [========================>.....] - ETA: 14s - loss: 0.2091 - categorical_accuracy: 0.9362
51584/60000 [========================>.....] - ETA: 13s - loss: 0.2091 - categorical_accuracy: 0.9361
51616/60000 [========================>.....] - ETA: 13s - loss: 0.2091 - categorical_accuracy: 0.9362
51648/60000 [========================>.....] - ETA: 13s - loss: 0.2090 - categorical_accuracy: 0.9362
51680/60000 [========================>.....] - ETA: 13s - loss: 0.2089 - categorical_accuracy: 0.9362
51712/60000 [========================>.....] - ETA: 13s - loss: 0.2089 - categorical_accuracy: 0.9362
51744/60000 [========================>.....] - ETA: 13s - loss: 0.2088 - categorical_accuracy: 0.9363
51776/60000 [========================>.....] - ETA: 13s - loss: 0.2086 - categorical_accuracy: 0.9363
51808/60000 [========================>.....] - ETA: 13s - loss: 0.2085 - categorical_accuracy: 0.9363
51840/60000 [========================>.....] - ETA: 13s - loss: 0.2085 - categorical_accuracy: 0.9363
51872/60000 [========================>.....] - ETA: 13s - loss: 0.2086 - categorical_accuracy: 0.9363
51904/60000 [========================>.....] - ETA: 13s - loss: 0.2085 - categorical_accuracy: 0.9363
51968/60000 [========================>.....] - ETA: 13s - loss: 0.2086 - categorical_accuracy: 0.9364
52032/60000 [=========================>....] - ETA: 13s - loss: 0.2085 - categorical_accuracy: 0.9364
52064/60000 [=========================>....] - ETA: 13s - loss: 0.2085 - categorical_accuracy: 0.9364
52096/60000 [=========================>....] - ETA: 13s - loss: 0.2084 - categorical_accuracy: 0.9365
52128/60000 [=========================>....] - ETA: 13s - loss: 0.2083 - categorical_accuracy: 0.9365
52160/60000 [=========================>....] - ETA: 13s - loss: 0.2083 - categorical_accuracy: 0.9365
52192/60000 [=========================>....] - ETA: 12s - loss: 0.2082 - categorical_accuracy: 0.9365
52224/60000 [=========================>....] - ETA: 12s - loss: 0.2081 - categorical_accuracy: 0.9366
52256/60000 [=========================>....] - ETA: 12s - loss: 0.2080 - categorical_accuracy: 0.9366
52320/60000 [=========================>....] - ETA: 12s - loss: 0.2078 - categorical_accuracy: 0.9367
52352/60000 [=========================>....] - ETA: 12s - loss: 0.2077 - categorical_accuracy: 0.9367
52416/60000 [=========================>....] - ETA: 12s - loss: 0.2077 - categorical_accuracy: 0.9367
52480/60000 [=========================>....] - ETA: 12s - loss: 0.2076 - categorical_accuracy: 0.9367
52512/60000 [=========================>....] - ETA: 12s - loss: 0.2076 - categorical_accuracy: 0.9367
52576/60000 [=========================>....] - ETA: 12s - loss: 0.2074 - categorical_accuracy: 0.9368
52640/60000 [=========================>....] - ETA: 12s - loss: 0.2074 - categorical_accuracy: 0.9367
52672/60000 [=========================>....] - ETA: 12s - loss: 0.2073 - categorical_accuracy: 0.9368
52704/60000 [=========================>....] - ETA: 12s - loss: 0.2072 - categorical_accuracy: 0.9368
52736/60000 [=========================>....] - ETA: 12s - loss: 0.2072 - categorical_accuracy: 0.9368
52768/60000 [=========================>....] - ETA: 12s - loss: 0.2070 - categorical_accuracy: 0.9368
52800/60000 [=========================>....] - ETA: 11s - loss: 0.2070 - categorical_accuracy: 0.9369
52832/60000 [=========================>....] - ETA: 11s - loss: 0.2069 - categorical_accuracy: 0.9369
52864/60000 [=========================>....] - ETA: 11s - loss: 0.2068 - categorical_accuracy: 0.9369
52896/60000 [=========================>....] - ETA: 11s - loss: 0.2067 - categorical_accuracy: 0.9369
52928/60000 [=========================>....] - ETA: 11s - loss: 0.2067 - categorical_accuracy: 0.9370
52960/60000 [=========================>....] - ETA: 11s - loss: 0.2066 - categorical_accuracy: 0.9370
53024/60000 [=========================>....] - ETA: 11s - loss: 0.2064 - categorical_accuracy: 0.9370
53056/60000 [=========================>....] - ETA: 11s - loss: 0.2063 - categorical_accuracy: 0.9371
53088/60000 [=========================>....] - ETA: 11s - loss: 0.2062 - categorical_accuracy: 0.9371
53120/60000 [=========================>....] - ETA: 11s - loss: 0.2061 - categorical_accuracy: 0.9371
53152/60000 [=========================>....] - ETA: 11s - loss: 0.2060 - categorical_accuracy: 0.9371
53184/60000 [=========================>....] - ETA: 11s - loss: 0.2061 - categorical_accuracy: 0.9371
53216/60000 [=========================>....] - ETA: 11s - loss: 0.2060 - categorical_accuracy: 0.9371
53248/60000 [=========================>....] - ETA: 11s - loss: 0.2059 - categorical_accuracy: 0.9372
53280/60000 [=========================>....] - ETA: 11s - loss: 0.2058 - categorical_accuracy: 0.9372
53344/60000 [=========================>....] - ETA: 11s - loss: 0.2058 - categorical_accuracy: 0.9373
53376/60000 [=========================>....] - ETA: 11s - loss: 0.2056 - categorical_accuracy: 0.9373
53408/60000 [=========================>....] - ETA: 10s - loss: 0.2056 - categorical_accuracy: 0.9373
53472/60000 [=========================>....] - ETA: 10s - loss: 0.2054 - categorical_accuracy: 0.9373
53536/60000 [=========================>....] - ETA: 10s - loss: 0.2052 - categorical_accuracy: 0.9374
53568/60000 [=========================>....] - ETA: 10s - loss: 0.2051 - categorical_accuracy: 0.9374
53632/60000 [=========================>....] - ETA: 10s - loss: 0.2049 - categorical_accuracy: 0.9375
53664/60000 [=========================>....] - ETA: 10s - loss: 0.2048 - categorical_accuracy: 0.9375
53728/60000 [=========================>....] - ETA: 10s - loss: 0.2048 - categorical_accuracy: 0.9375
53760/60000 [=========================>....] - ETA: 10s - loss: 0.2048 - categorical_accuracy: 0.9376
53824/60000 [=========================>....] - ETA: 10s - loss: 0.2046 - categorical_accuracy: 0.9376
53856/60000 [=========================>....] - ETA: 10s - loss: 0.2045 - categorical_accuracy: 0.9376
53888/60000 [=========================>....] - ETA: 10s - loss: 0.2044 - categorical_accuracy: 0.9377
53920/60000 [=========================>....] - ETA: 10s - loss: 0.2043 - categorical_accuracy: 0.9377
53952/60000 [=========================>....] - ETA: 10s - loss: 0.2042 - categorical_accuracy: 0.9377
53984/60000 [=========================>....] - ETA: 9s - loss: 0.2042 - categorical_accuracy: 0.9378 
54016/60000 [==========================>...] - ETA: 9s - loss: 0.2041 - categorical_accuracy: 0.9378
54048/60000 [==========================>...] - ETA: 9s - loss: 0.2040 - categorical_accuracy: 0.9378
54080/60000 [==========================>...] - ETA: 9s - loss: 0.2040 - categorical_accuracy: 0.9378
54112/60000 [==========================>...] - ETA: 9s - loss: 0.2040 - categorical_accuracy: 0.9378
54144/60000 [==========================>...] - ETA: 9s - loss: 0.2040 - categorical_accuracy: 0.9379
54176/60000 [==========================>...] - ETA: 9s - loss: 0.2041 - categorical_accuracy: 0.9379
54208/60000 [==========================>...] - ETA: 9s - loss: 0.2043 - categorical_accuracy: 0.9378
54272/60000 [==========================>...] - ETA: 9s - loss: 0.2041 - categorical_accuracy: 0.9379
54304/60000 [==========================>...] - ETA: 9s - loss: 0.2040 - categorical_accuracy: 0.9379
54336/60000 [==========================>...] - ETA: 9s - loss: 0.2038 - categorical_accuracy: 0.9380
54368/60000 [==========================>...] - ETA: 9s - loss: 0.2038 - categorical_accuracy: 0.9380
54432/60000 [==========================>...] - ETA: 9s - loss: 0.2036 - categorical_accuracy: 0.9380
54496/60000 [==========================>...] - ETA: 9s - loss: 0.2035 - categorical_accuracy: 0.9381
54560/60000 [==========================>...] - ETA: 9s - loss: 0.2033 - categorical_accuracy: 0.9381
54592/60000 [==========================>...] - ETA: 8s - loss: 0.2032 - categorical_accuracy: 0.9382
54624/60000 [==========================>...] - ETA: 8s - loss: 0.2033 - categorical_accuracy: 0.9382
54656/60000 [==========================>...] - ETA: 8s - loss: 0.2032 - categorical_accuracy: 0.9382
54688/60000 [==========================>...] - ETA: 8s - loss: 0.2031 - categorical_accuracy: 0.9382
54752/60000 [==========================>...] - ETA: 8s - loss: 0.2030 - categorical_accuracy: 0.9382
54784/60000 [==========================>...] - ETA: 8s - loss: 0.2029 - categorical_accuracy: 0.9382
54848/60000 [==========================>...] - ETA: 8s - loss: 0.2027 - categorical_accuracy: 0.9383
54880/60000 [==========================>...] - ETA: 8s - loss: 0.2027 - categorical_accuracy: 0.9383
54912/60000 [==========================>...] - ETA: 8s - loss: 0.2026 - categorical_accuracy: 0.9384
54944/60000 [==========================>...] - ETA: 8s - loss: 0.2025 - categorical_accuracy: 0.9384
54976/60000 [==========================>...] - ETA: 8s - loss: 0.2025 - categorical_accuracy: 0.9384
55008/60000 [==========================>...] - ETA: 8s - loss: 0.2024 - categorical_accuracy: 0.9384
55040/60000 [==========================>...] - ETA: 8s - loss: 0.2024 - categorical_accuracy: 0.9384
55072/60000 [==========================>...] - ETA: 8s - loss: 0.2023 - categorical_accuracy: 0.9385
55104/60000 [==========================>...] - ETA: 8s - loss: 0.2022 - categorical_accuracy: 0.9385
55136/60000 [==========================>...] - ETA: 8s - loss: 0.2022 - categorical_accuracy: 0.9385
55168/60000 [==========================>...] - ETA: 8s - loss: 0.2021 - categorical_accuracy: 0.9385
55200/60000 [==========================>...] - ETA: 7s - loss: 0.2020 - categorical_accuracy: 0.9385
55264/60000 [==========================>...] - ETA: 7s - loss: 0.2018 - categorical_accuracy: 0.9386
55328/60000 [==========================>...] - ETA: 7s - loss: 0.2018 - categorical_accuracy: 0.9386
55360/60000 [==========================>...] - ETA: 7s - loss: 0.2017 - categorical_accuracy: 0.9386
55392/60000 [==========================>...] - ETA: 7s - loss: 0.2016 - categorical_accuracy: 0.9387
55424/60000 [==========================>...] - ETA: 7s - loss: 0.2015 - categorical_accuracy: 0.9387
55456/60000 [==========================>...] - ETA: 7s - loss: 0.2014 - categorical_accuracy: 0.9387
55488/60000 [==========================>...] - ETA: 7s - loss: 0.2013 - categorical_accuracy: 0.9388
55520/60000 [==========================>...] - ETA: 7s - loss: 0.2012 - categorical_accuracy: 0.9388
55552/60000 [==========================>...] - ETA: 7s - loss: 0.2011 - categorical_accuracy: 0.9388
55584/60000 [==========================>...] - ETA: 7s - loss: 0.2010 - categorical_accuracy: 0.9388
55616/60000 [==========================>...] - ETA: 7s - loss: 0.2009 - categorical_accuracy: 0.9389
55648/60000 [==========================>...] - ETA: 7s - loss: 0.2009 - categorical_accuracy: 0.9389
55680/60000 [==========================>...] - ETA: 7s - loss: 0.2008 - categorical_accuracy: 0.9389
55712/60000 [==========================>...] - ETA: 7s - loss: 0.2009 - categorical_accuracy: 0.9389
55744/60000 [==========================>...] - ETA: 7s - loss: 0.2008 - categorical_accuracy: 0.9390
55776/60000 [==========================>...] - ETA: 7s - loss: 0.2007 - categorical_accuracy: 0.9390
55840/60000 [==========================>...] - ETA: 6s - loss: 0.2005 - categorical_accuracy: 0.9391
55872/60000 [==========================>...] - ETA: 6s - loss: 0.2004 - categorical_accuracy: 0.9391
55936/60000 [==========================>...] - ETA: 6s - loss: 0.2005 - categorical_accuracy: 0.9390
55968/60000 [==========================>...] - ETA: 6s - loss: 0.2004 - categorical_accuracy: 0.9391
56000/60000 [===========================>..] - ETA: 6s - loss: 0.2004 - categorical_accuracy: 0.9391
56032/60000 [===========================>..] - ETA: 6s - loss: 0.2003 - categorical_accuracy: 0.9391
56096/60000 [===========================>..] - ETA: 6s - loss: 0.2001 - categorical_accuracy: 0.9391
56128/60000 [===========================>..] - ETA: 6s - loss: 0.2001 - categorical_accuracy: 0.9392
56160/60000 [===========================>..] - ETA: 6s - loss: 0.2000 - categorical_accuracy: 0.9392
56192/60000 [===========================>..] - ETA: 6s - loss: 0.1999 - categorical_accuracy: 0.9392
56256/60000 [===========================>..] - ETA: 6s - loss: 0.1997 - categorical_accuracy: 0.9393
56288/60000 [===========================>..] - ETA: 6s - loss: 0.1996 - categorical_accuracy: 0.9393
56352/60000 [===========================>..] - ETA: 6s - loss: 0.1995 - categorical_accuracy: 0.9393
56416/60000 [===========================>..] - ETA: 5s - loss: 0.1994 - categorical_accuracy: 0.9394
56480/60000 [===========================>..] - ETA: 5s - loss: 0.1993 - categorical_accuracy: 0.9394
56512/60000 [===========================>..] - ETA: 5s - loss: 0.1993 - categorical_accuracy: 0.9394
56544/60000 [===========================>..] - ETA: 5s - loss: 0.1992 - categorical_accuracy: 0.9395
56576/60000 [===========================>..] - ETA: 5s - loss: 0.1991 - categorical_accuracy: 0.9395
56608/60000 [===========================>..] - ETA: 5s - loss: 0.1990 - categorical_accuracy: 0.9395
56640/60000 [===========================>..] - ETA: 5s - loss: 0.1989 - categorical_accuracy: 0.9395
56672/60000 [===========================>..] - ETA: 5s - loss: 0.1988 - categorical_accuracy: 0.9396
56704/60000 [===========================>..] - ETA: 5s - loss: 0.1988 - categorical_accuracy: 0.9396
56736/60000 [===========================>..] - ETA: 5s - loss: 0.1987 - categorical_accuracy: 0.9396
56800/60000 [===========================>..] - ETA: 5s - loss: 0.1986 - categorical_accuracy: 0.9397
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1985 - categorical_accuracy: 0.9397
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1985 - categorical_accuracy: 0.9397
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1984 - categorical_accuracy: 0.9397
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1983 - categorical_accuracy: 0.9397
56960/60000 [===========================>..] - ETA: 5s - loss: 0.1983 - categorical_accuracy: 0.9398
56992/60000 [===========================>..] - ETA: 4s - loss: 0.1983 - categorical_accuracy: 0.9398
57024/60000 [===========================>..] - ETA: 4s - loss: 0.1982 - categorical_accuracy: 0.9398
57056/60000 [===========================>..] - ETA: 4s - loss: 0.1982 - categorical_accuracy: 0.9398
57088/60000 [===========================>..] - ETA: 4s - loss: 0.1982 - categorical_accuracy: 0.9398
57120/60000 [===========================>..] - ETA: 4s - loss: 0.1981 - categorical_accuracy: 0.9398
57152/60000 [===========================>..] - ETA: 4s - loss: 0.1980 - categorical_accuracy: 0.9398
57184/60000 [===========================>..] - ETA: 4s - loss: 0.1980 - categorical_accuracy: 0.9398
57216/60000 [===========================>..] - ETA: 4s - loss: 0.1979 - categorical_accuracy: 0.9398
57248/60000 [===========================>..] - ETA: 4s - loss: 0.1978 - categorical_accuracy: 0.9399
57280/60000 [===========================>..] - ETA: 4s - loss: 0.1978 - categorical_accuracy: 0.9399
57312/60000 [===========================>..] - ETA: 4s - loss: 0.1978 - categorical_accuracy: 0.9399
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1978 - categorical_accuracy: 0.9399
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1978 - categorical_accuracy: 0.9399
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1977 - categorical_accuracy: 0.9399
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1976 - categorical_accuracy: 0.9399
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1975 - categorical_accuracy: 0.9399
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1974 - categorical_accuracy: 0.9400
57600/60000 [===========================>..] - ETA: 3s - loss: 0.1973 - categorical_accuracy: 0.9400
57632/60000 [===========================>..] - ETA: 3s - loss: 0.1972 - categorical_accuracy: 0.9400
57664/60000 [===========================>..] - ETA: 3s - loss: 0.1971 - categorical_accuracy: 0.9401
57696/60000 [===========================>..] - ETA: 3s - loss: 0.1971 - categorical_accuracy: 0.9401
57760/60000 [===========================>..] - ETA: 3s - loss: 0.1971 - categorical_accuracy: 0.9401
57792/60000 [===========================>..] - ETA: 3s - loss: 0.1970 - categorical_accuracy: 0.9401
57824/60000 [===========================>..] - ETA: 3s - loss: 0.1969 - categorical_accuracy: 0.9401
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1968 - categorical_accuracy: 0.9402
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1968 - categorical_accuracy: 0.9402
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1967 - categorical_accuracy: 0.9402
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1967 - categorical_accuracy: 0.9402
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1967 - categorical_accuracy: 0.9402
58016/60000 [============================>.] - ETA: 3s - loss: 0.1966 - categorical_accuracy: 0.9402
58080/60000 [============================>.] - ETA: 3s - loss: 0.1965 - categorical_accuracy: 0.9402
58112/60000 [============================>.] - ETA: 3s - loss: 0.1964 - categorical_accuracy: 0.9402
58176/60000 [============================>.] - ETA: 3s - loss: 0.1963 - categorical_accuracy: 0.9403
58208/60000 [============================>.] - ETA: 2s - loss: 0.1965 - categorical_accuracy: 0.9403
58240/60000 [============================>.] - ETA: 2s - loss: 0.1964 - categorical_accuracy: 0.9403
58272/60000 [============================>.] - ETA: 2s - loss: 0.1964 - categorical_accuracy: 0.9403
58304/60000 [============================>.] - ETA: 2s - loss: 0.1963 - categorical_accuracy: 0.9403
58336/60000 [============================>.] - ETA: 2s - loss: 0.1962 - categorical_accuracy: 0.9404
58368/60000 [============================>.] - ETA: 2s - loss: 0.1963 - categorical_accuracy: 0.9403
58400/60000 [============================>.] - ETA: 2s - loss: 0.1963 - categorical_accuracy: 0.9404
58432/60000 [============================>.] - ETA: 2s - loss: 0.1963 - categorical_accuracy: 0.9404
58464/60000 [============================>.] - ETA: 2s - loss: 0.1963 - categorical_accuracy: 0.9403
58496/60000 [============================>.] - ETA: 2s - loss: 0.1963 - categorical_accuracy: 0.9404
58528/60000 [============================>.] - ETA: 2s - loss: 0.1962 - categorical_accuracy: 0.9404
58560/60000 [============================>.] - ETA: 2s - loss: 0.1961 - categorical_accuracy: 0.9404
58592/60000 [============================>.] - ETA: 2s - loss: 0.1960 - categorical_accuracy: 0.9405
58624/60000 [============================>.] - ETA: 2s - loss: 0.1959 - categorical_accuracy: 0.9405
58688/60000 [============================>.] - ETA: 2s - loss: 0.1958 - categorical_accuracy: 0.9405
58752/60000 [============================>.] - ETA: 2s - loss: 0.1956 - categorical_accuracy: 0.9405
58784/60000 [============================>.] - ETA: 2s - loss: 0.1955 - categorical_accuracy: 0.9406
58816/60000 [============================>.] - ETA: 1s - loss: 0.1955 - categorical_accuracy: 0.9406
58848/60000 [============================>.] - ETA: 1s - loss: 0.1954 - categorical_accuracy: 0.9406
58880/60000 [============================>.] - ETA: 1s - loss: 0.1954 - categorical_accuracy: 0.9406
58912/60000 [============================>.] - ETA: 1s - loss: 0.1953 - categorical_accuracy: 0.9407
58944/60000 [============================>.] - ETA: 1s - loss: 0.1952 - categorical_accuracy: 0.9407
58976/60000 [============================>.] - ETA: 1s - loss: 0.1952 - categorical_accuracy: 0.9407
59008/60000 [============================>.] - ETA: 1s - loss: 0.1951 - categorical_accuracy: 0.9407
59040/60000 [============================>.] - ETA: 1s - loss: 0.1950 - categorical_accuracy: 0.9407
59072/60000 [============================>.] - ETA: 1s - loss: 0.1950 - categorical_accuracy: 0.9408
59104/60000 [============================>.] - ETA: 1s - loss: 0.1949 - categorical_accuracy: 0.9408
59136/60000 [============================>.] - ETA: 1s - loss: 0.1948 - categorical_accuracy: 0.9408
59200/60000 [============================>.] - ETA: 1s - loss: 0.1946 - categorical_accuracy: 0.9408
59232/60000 [============================>.] - ETA: 1s - loss: 0.1945 - categorical_accuracy: 0.9409
59264/60000 [============================>.] - ETA: 1s - loss: 0.1945 - categorical_accuracy: 0.9409
59296/60000 [============================>.] - ETA: 1s - loss: 0.1944 - categorical_accuracy: 0.9409
59360/60000 [============================>.] - ETA: 1s - loss: 0.1945 - categorical_accuracy: 0.9409
59392/60000 [============================>.] - ETA: 1s - loss: 0.1945 - categorical_accuracy: 0.9409
59424/60000 [============================>.] - ETA: 0s - loss: 0.1945 - categorical_accuracy: 0.9409
59456/60000 [============================>.] - ETA: 0s - loss: 0.1944 - categorical_accuracy: 0.9409
59488/60000 [============================>.] - ETA: 0s - loss: 0.1943 - categorical_accuracy: 0.9410
59520/60000 [============================>.] - ETA: 0s - loss: 0.1942 - categorical_accuracy: 0.9410
59552/60000 [============================>.] - ETA: 0s - loss: 0.1941 - categorical_accuracy: 0.9410
59584/60000 [============================>.] - ETA: 0s - loss: 0.1941 - categorical_accuracy: 0.9410
59616/60000 [============================>.] - ETA: 0s - loss: 0.1941 - categorical_accuracy: 0.9410
59648/60000 [============================>.] - ETA: 0s - loss: 0.1941 - categorical_accuracy: 0.9411
59712/60000 [============================>.] - ETA: 0s - loss: 0.1939 - categorical_accuracy: 0.9411
59744/60000 [============================>.] - ETA: 0s - loss: 0.1938 - categorical_accuracy: 0.9411
59776/60000 [============================>.] - ETA: 0s - loss: 0.1938 - categorical_accuracy: 0.9411
59808/60000 [============================>.] - ETA: 0s - loss: 0.1937 - categorical_accuracy: 0.9411
59840/60000 [============================>.] - ETA: 0s - loss: 0.1936 - categorical_accuracy: 0.9412
59904/60000 [============================>.] - ETA: 0s - loss: 0.1935 - categorical_accuracy: 0.9412
59968/60000 [============================>.] - ETA: 0s - loss: 0.1934 - categorical_accuracy: 0.9413
60000/60000 [==============================] - 103s 2ms/step - loss: 0.1934 - categorical_accuracy: 0.9413 - val_loss: 0.0482 - val_categorical_accuracy: 0.9843

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 15s
  192/10000 [..............................] - ETA: 5s 
  352/10000 [>.............................] - ETA: 4s
  512/10000 [>.............................] - ETA: 3s
  672/10000 [=>............................] - ETA: 3s
  832/10000 [=>............................] - ETA: 3s
 1024/10000 [==>...........................] - ETA: 3s
 1216/10000 [==>...........................] - ETA: 3s
 1408/10000 [===>..........................] - ETA: 3s
 1600/10000 [===>..........................] - ETA: 2s
 1760/10000 [====>.........................] - ETA: 2s
 1952/10000 [====>.........................] - ETA: 2s
 2144/10000 [=====>........................] - ETA: 2s
 2336/10000 [======>.......................] - ETA: 2s
 2528/10000 [======>.......................] - ETA: 2s
 2720/10000 [=======>......................] - ETA: 2s
 2912/10000 [=======>......................] - ETA: 2s
 3072/10000 [========>.....................] - ETA: 2s
 3232/10000 [========>.....................] - ETA: 2s
 3424/10000 [=========>....................] - ETA: 2s
 3616/10000 [=========>....................] - ETA: 2s
 3808/10000 [==========>...................] - ETA: 2s
 4000/10000 [===========>..................] - ETA: 1s
 4160/10000 [===========>..................] - ETA: 1s
 4352/10000 [============>.................] - ETA: 1s
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
 6272/10000 [=================>............] - ETA: 1s
 6464/10000 [==================>...........] - ETA: 1s
 6656/10000 [==================>...........] - ETA: 1s
 6848/10000 [===================>..........] - ETA: 1s
 7040/10000 [====================>.........] - ETA: 0s
 7200/10000 [====================>.........] - ETA: 0s
 7360/10000 [=====================>........] - ETA: 0s
 7520/10000 [=====================>........] - ETA: 0s
 7712/10000 [======================>.......] - ETA: 0s
 7872/10000 [======================>.......] - ETA: 0s
 8032/10000 [=======================>......] - ETA: 0s
 8192/10000 [=======================>......] - ETA: 0s
 8352/10000 [========================>.....] - ETA: 0s
 8512/10000 [========================>.....] - ETA: 0s
 8672/10000 [=========================>....] - ETA: 0s
 8800/10000 [=========================>....] - ETA: 0s
 8960/10000 [=========================>....] - ETA: 0s
 9152/10000 [==========================>...] - ETA: 0s
 9312/10000 [==========================>...] - ETA: 0s
 9472/10000 [===========================>..] - ETA: 0s
 9632/10000 [===========================>..] - ETA: 0s
 9792/10000 [============================>.] - ETA: 0s
 9952/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 3s 327us/step
[[1.98023535e-08 1.12888525e-08 4.22795011e-07 ... 9.99996066e-01
  2.36577709e-08 5.56380371e-07]
 [1.90965166e-05 2.07777430e-05 9.99939322e-01 ... 5.00670971e-08
  6.94048322e-06 1.93952809e-09]
 [2.86190755e-07 9.99858141e-01 1.16191641e-05 ... 6.11033684e-05
  1.44995565e-05 3.78708069e-06]
 ...
 [7.55611929e-09 1.53102303e-06 1.07213445e-07 ... 1.43590378e-05
  9.13284475e-06 2.14725569e-05]
 [8.74319539e-07 2.60833872e-07 5.01407413e-08 ... 9.23620519e-07
  1.03040226e-02 2.02993442e-06]
 [1.30169713e-06 5.42083399e-07 2.37399058e-06 ... 9.64030522e-10
  6.13920179e-07 1.89375471e-09]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.04817535739473533, 'accuracy_test:': 0.9843000173568726}

  ('#### Save   #######################################################',) 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/charcnn/result'}

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   f85da80..a3f185b  master     -> origin/master
Updating f85da80..a3f185b
Fast-forward
 error_list/20200513/list_log_benchmark_20200513.md |  162 +-
 error_list/20200513/list_log_jupyter_20200513.md   | 2243 ++++++++++----------
 2 files changed, 1203 insertions(+), 1202 deletions(-)
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
[master 9dc9baa] ml_store
 1 file changed, 1783 insertions(+)
To github.com:arita37/mlmodels_store.git
   a3f185b..9dc9baa  master -> master





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
{'loss': 0.4224967584013939, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-13 00:38:18.394092: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
[master f9fa24a] ml_store
 1 file changed, 233 insertions(+)
To github.com:arita37/mlmodels_store.git
   9dc9baa..f9fa24a  master -> master





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
[master b355236] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   f9fa24a..b355236  master -> master





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
 40%|████      | 2/5 [00:16<00:24,  8.21s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.865703517781423, 'learning_rate': 0.06555104846695452, 'min_data_in_leaf': 28, 'num_leaves': 53} and reward: 0.3938
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeb\xb3\xd7\xdd\x1c\xe8`X\r\x00\x00\x00learning_rateq\x02G?\xb0\xc7\xf4\x19bS\x1bX\x10\x00\x00\x00min_data_in_leafq\x03K\x1cX\n\x00\x00\x00num_leavesq\x04K5u.' and reward: 0.3938
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeb\xb3\xd7\xdd\x1c\xe8`X\r\x00\x00\x00learning_rateq\x02G?\xb0\xc7\xf4\x19bS\x1bX\x10\x00\x00\x00min_data_in_leafq\x03K\x1cX\n\x00\x00\x00num_leavesq\x04K5u.' and reward: 0.3938
 60%|██████    | 3/5 [00:37<00:23, 11.95s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.7793143071712615, 'learning_rate': 0.019300758627962798, 'min_data_in_leaf': 3, 'num_leaves': 31} and reward: 0.3896
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8\xf0$\x8e\xd3_\xfbX\r\x00\x00\x00learning_rateq\x02G?\x93\xc3\x93\xfca_IX\x10\x00\x00\x00min_data_in_leafq\x03K\x03X\n\x00\x00\x00num_leavesq\x04K\x1fu.' and reward: 0.3896
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8\xf0$\x8e\xd3_\xfbX\r\x00\x00\x00learning_rateq\x02G?\x93\xc3\x93\xfca_IX\x10\x00\x00\x00min_data_in_leafq\x03K\x03X\n\x00\x00\x00num_leavesq\x04K\x1fu.' and reward: 0.3896
 80%|████████  | 4/5 [00:52<00:13, 13.03s/it]Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.9761841209275586, 'learning_rate': 0.005945548175774274, 'min_data_in_leaf': 18, 'num_leaves': 61} and reward: 0.3844
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xef<\xe6{HD\x98X\r\x00\x00\x00learning_rateq\x02G?xZ[\xef\x8cBZX\x10\x00\x00\x00min_data_in_leafq\x03K\x12X\n\x00\x00\x00num_leavesq\x04K=u.' and reward: 0.3844
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xef<\xe6{HD\x98X\r\x00\x00\x00learning_rateq\x02G?xZ[\xef\x8cBZX\x10\x00\x00\x00min_data_in_leafq\x03K\x12X\n\x00\x00\x00num_leavesq\x04K=u.' and reward: 0.3844
100%|██████████| 5/5 [01:16<00:00, 16.16s/it]100%|██████████| 5/5 [01:16<00:00, 15.22s/it]
Saving dataset/models/LightGBMClassifier/trial_4_model.pkl
Finished Task with config: {'feature_fraction': 0.9572781243544798, 'learning_rate': 0.023334680979566844, 'min_data_in_leaf': 19, 'num_leaves': 43} and reward: 0.391
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xee\xa2\x05\xbb\xa8\xeb5X\r\x00\x00\x00learning_rateq\x02G?\x97\xe5\x0b\xee\xad\xe8\xa9X\x10\x00\x00\x00min_data_in_leafq\x03K\x13X\n\x00\x00\x00num_leavesq\x04K+u.' and reward: 0.391
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xee\xa2\x05\xbb\xa8\xeb5X\r\x00\x00\x00learning_rateq\x02G?\x97\xe5\x0b\xee\xad\xe8\xa9X\x10\x00\x00\x00min_data_in_leafq\x03K\x13X\n\x00\x00\x00num_leavesq\x04K+u.' and reward: 0.391
Time for Gradient Boosting hyperparameter optimization: 95.1731219291687
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.865703517781423, 'learning_rate': 0.06555104846695452, 'min_data_in_leaf': 28, 'num_leaves': 53}
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
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3868
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3868
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3868
 40%|████      | 2/5 [00:55<01:23, 27.85s/it] 40%|████      | 2/5 [00:55<01:23, 27.85s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.015935133388541693, 'embedding_size_factor': 1.2769783736459959, 'layers.choice': 2, 'learning_rate': 0.0010279261402988123, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 3.1387015013842306e-10} and reward: 0.3864
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\x90QL\xb3\x0bs\xe1X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf4n\x80\xe0\x08$!X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?P\xd7oI\xf08~X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\xf5\x91\xa9}\xbf\xaf=u.' and reward: 0.3864
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\x90QL\xb3\x0bs\xe1X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf4n\x80\xe0\x08$!X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?P\xd7oI\xf08~X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\xf5\x91\xa9}\xbf\xaf=u.' and reward: 0.3864
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 112.46961045265198
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_4_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.78s of the -92.06s of remaining time.
Ensemble size: 73
Ensemble weights: 
[0.35616438 0.05479452 0.19178082 0.02739726 0.21917808 0.01369863
 0.1369863 ]
	0.3974	 = Validation accuracy score
	1.67s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 213.77s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_4_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f82dcd05080>

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   b355236..ff53cde  master     -> origin/master
Updating b355236..ff53cde
Fast-forward
 error_list/20200513/list_log_jupyter_20200513.md   | 2243 +++++++++---------
 error_list/20200513/list_log_test_cli_20200513.md  |  138 +-
 ...-18_6672e19fe4cfa7df885e45d91d645534b8989485.py | 2488 ++++++++++++++++++++
 3 files changed, 3678 insertions(+), 1191 deletions(-)
 create mode 100644 log_benchmark/log_benchmark_2020-05-13-00-18_6672e19fe4cfa7df885e45d91d645534b8989485.py
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
[master d373006] ml_store
 1 file changed, 214 insertions(+)
To github.com:arita37/mlmodels_store.git
   ff53cde..d373006  master -> master





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
[master 2867eb6] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   d373006..2867eb6  master -> master





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
[master 4df53e4] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   2867eb6..4df53e4  master -> master





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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7fe7dc786f98> 

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
