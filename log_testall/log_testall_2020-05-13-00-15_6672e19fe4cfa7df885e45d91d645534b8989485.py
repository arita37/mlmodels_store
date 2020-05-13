
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
[master e735a78] ml_store
 1 file changed, 108 insertions(+)
To github.com:arita37/mlmodels_store.git
   4df53e4..e735a78  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 0.86146256  0.07432055 -1.34501002 -0.19956072 -1.47533915 -0.65460317
  -0.31456386  0.3180143  -0.89027155 -1.29525789]
 [ 0.68188934 -1.15498263  1.22895559 -0.1776322   0.99854519 -1.51045638
  -0.27584606  1.01120706 -1.47656266  1.30970591]
 [ 1.16755486  0.0353601   0.7147896  -1.53879325  1.10863359 -0.44789518
  -1.75592564  0.61798553 -0.18417633  0.85270406]
 [ 0.6675918  -0.45252497 -0.60598132  1.16128569 -1.44620987  1.06996554
   1.92381543 -1.04553425  0.35528451  1.80358898]
 [ 0.62567337  0.5924728   0.67457071  1.19783084  1.23187251  1.70459417
  -0.76730983  1.04008915 -0.91844004  1.46089238]
 [ 1.01177337  0.09574677  0.73140252  1.0334508  -1.42203164 -0.14627327
  -0.01745495 -0.85749682 -0.93418184  0.95449567]
 [ 1.39198128 -0.19022103 -0.53722302 -0.44873803  0.70455707 -0.67244804
  -0.70134443 -0.55749472  0.93916874  0.15626385]
 [ 1.32720112 -0.16119832  0.6024509  -0.28638492 -0.5789623  -0.87088765
   1.37975819  0.50142959 -0.47861407 -0.89264667]
 [ 1.17867274 -0.59980453 -0.6946936   1.12341216  1.17899425  0.30526704
   0.01335268  1.3887794  -0.66134424  0.6218035 ]
 [ 0.81583612 -1.39169388  2.50598029  0.45021774 -0.88286982  0.62743708
  -1.19586151  0.75133724  0.14039544  1.91979229]
 [ 1.18947778 -0.68067814 -0.05682448 -0.08450803  0.82178321 -0.29736188
  -0.18657899  0.417302    0.78477065  0.49233656]
 [ 1.46893146 -1.47115693  0.58591043 -0.8301719   1.03345052 -0.8805776
  -0.95542526 -0.27909772  1.62284909  2.06578332]
 [ 2.07582971 -1.40232915 -0.47918492  0.45112294  1.03436581 -0.6949209
  -0.4189379   0.5154138  -1.11487105 -1.95210529]
 [ 0.84806927  0.45194604  0.63019567 -1.57915629  0.82798737 -0.82862798
  -0.10534471  0.52887975 -2.23708651 -0.4148469 ]
 [ 1.16777676 -0.66575452 -1.23312074 -1.67419581  1.01313574  0.82502982
  -0.12046457 -0.49821356 -0.31098498 -1.18231813]
 [ 0.10593645 -0.73728963  0.65032321  0.16466507 -1.53556118  0.77817418
   0.05031709  0.30981676  1.05132077  0.6065484 ]
 [ 0.62153099 -1.50957268 -0.10193204 -1.08071069 -1.13742855  0.725474
   0.7980638  -0.03917826 -0.22875417  0.74335654]
 [ 0.96703727  0.38271517 -0.80618482 -0.28899734  0.90852604 -0.39181624
   1.62091229  0.68400133 -0.35340998 -0.25167421]
 [ 0.56998385 -0.53302033 -0.17545897 -1.42655542  0.60660431  1.76795995
  -0.11598519 -0.47537288  0.47761018 -0.93391466]
 [ 1.03967316 -0.73153098  0.36184732 -1.56573815  0.95928819  1.01382247
  -1.78791289 -2.22711263 -1.6993336  -0.42449279]
 [ 1.01195228 -1.88141087  1.70018815  0.4972691  -0.91766462  0.2373327
  -1.09033833 -2.14444405 -0.36956243  0.60878366]
 [ 0.96457205 -0.10679399  1.12232832  1.45142926  1.21828168 -0.61803685
   0.43816635 -2.03720123 -1.94258918 -0.9970198 ]
 [ 1.02242019  1.85300949  0.64435367  0.14225137  1.15080755  0.51350548
  -0.45994283  0.37245685 -0.1484898   0.37167029]
 [ 0.47330777 -0.97326759 -0.22814069  0.17516773 -1.01366961 -0.05348369
   0.39378773 -0.18306199 -0.2210289   0.58033011]
 [ 1.12641981 -0.6294416   1.1010002  -1.1134361   0.94459507 -0.06741002
  -0.1834002   1.16143998 -0.02752939  0.78002714]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7fa999b4bef0>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7fa9b3ed26d8> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 8.53355545e-01 -7.04350332e-01 -6.79383783e-01 -4.58666861e-02
  -1.29936179e+00 -2.18733459e-01  5.90039464e-01  1.53920701e+00
  -1.14870423e+00 -9.50909251e-01]
 [ 1.05936450e-01 -7.37289628e-01  6.50323214e-01  1.64665066e-01
  -1.53556118e+00  7.78174179e-01  5.03170861e-02  3.09816759e-01
   1.05132077e+00  6.06548400e-01]
 [ 6.91743730e-01  1.00978733e+00 -1.21333813e+00 -1.55694156e+00
  -1.20257258e+00 -6.12442128e-01 -2.69836174e+00 -1.39351805e-01
  -7.28537489e-01  7.22518992e-02]
 [ 8.15836116e-01 -1.39169388e+00  2.50598029e+00  4.50217742e-01
  -8.82869820e-01  6.27437083e-01 -1.19586151e+00  7.51337235e-01
   1.40395436e-01  1.91979229e+00]
 [ 9.67037267e-01  3.82715174e-01 -8.06184817e-01 -2.88997343e-01
   9.08526041e-01 -3.91816240e-01  1.62091229e+00  6.84001328e-01
  -3.53409983e-01 -2.51674208e-01]
 [ 1.83829400e+00  5.02740882e-01  1.29101580e-01  1.55880554e+00
   1.32551412e+00  1.09402696e-01  1.40754000e+00 -1.21974440e+00
   2.44936865e+00  1.61694960e+00]
 [ 1.16777676e+00 -6.65754518e-01 -1.23312074e+00 -1.67419581e+00
   1.01313574e+00  8.25029824e-01 -1.20464572e-01 -4.98213564e-01
  -3.10984978e-01 -1.18231813e+00]
 [ 8.72267394e-01 -2.51630386e+00 -7.75070287e-01 -5.95667881e-01
   1.02600767e+00 -3.09121319e-01  1.74643509e+00  5.10937774e-01
   1.71066184e+00  1.41640538e-01]
 [ 1.24549398e+00 -7.22391905e-01  1.11813340e+00  1.09899633e+00
   1.00277655e+00 -9.01634490e-01 -5.32234021e-01 -8.22467189e-01
   7.21711292e-01  6.74396105e-01]
 [ 1.01177337e+00  9.57467711e-02  7.31402517e-01  1.03345080e+00
  -1.42203164e+00 -1.46273275e-01 -1.74549518e-02 -8.57496825e-01
  -9.34181843e-01  9.54495667e-01]
 [ 7.75285326e-01  1.47016034e+00  1.03298378e+00 -8.70008223e-01
   7.86556511e-01  3.69190470e-01 -1.43195745e-01  8.53282186e-01
  -1.39711730e-01 -2.22414029e-01]
 [ 6.10942600e-01 -2.79099641e+00 -1.33520272e+00 -4.56117555e-01
  -9.44959948e-01 -9.79890252e-01 -1.56993672e-01  6.92574348e-01
  -4.78672356e-01 -1.06460122e-01]
 [ 4.41189807e-01  4.79852371e-01 -1.92003697e-01 -1.55269878e+00
  -1.88873982e+00  5.78464420e-01  3.98598388e-01 -9.61263599e-01
  -1.45832446e+00 -3.05376438e+00]
 [ 8.57719529e-01  9.81122462e-02 -2.60466059e-01  1.06032751e+00
  -1.39003042e+00 -1.71116766e+00  2.65642403e-01  1.65712464e+00
   1.41767401e+00  4.45096710e-01]
 [ 3.45715997e-01 -4.13029310e-01 -4.68673816e-01  1.83471763e+00
   7.71514409e-01  5.64382855e-01  2.18628366e-02  2.13782807e+00
  -7.85533997e-01  8.53281222e-01]
 [ 7.61706684e-01 -1.48515645e+00  1.30253554e+00 -5.92461285e-01
  -1.64162479e+00 -2.30490794e+00 -1.34869645e+00 -3.18171727e-02
   1.12487742e-01 -3.62612088e-01]
 [ 8.88389445e-01  2.82995534e-01  1.79558917e-02  1.08030817e-01
  -8.49671873e-01  2.94176190e-02 -5.03973949e-01 -1.34793129e-01
   1.04921829e+00 -1.27046078e+00]
 [ 8.88611457e-01  8.49586845e-01 -3.09114176e-02 -1.22154015e-01
  -1.14722826e+00 -6.80851574e-01 -3.26061306e-01 -1.06787658e+00
  -7.66793627e-02  3.55717262e-01]
 [ 9.36211246e-01  2.04377395e-01 -1.49419377e+00  6.12232523e-01
  -9.84377246e-01  7.44884536e-01  4.94341651e-01 -3.62812886e-02
  -8.32395348e-01 -4.46699203e-01]
 [ 1.32857949e+00 -5.63236604e-01 -1.06179676e+00  2.39014596e+00
  -1.68450770e+00  2.45422849e-01 -5.69148654e-01  1.15259914e+00
  -2.24235772e-01  1.32247779e-01]
 [ 1.18947778e+00 -6.80678141e-01 -5.68244809e-02 -8.45080274e-02
   8.21783210e-01 -2.97361883e-01 -1.86578994e-01  4.17302005e-01
   7.84770651e-01  4.92336556e-01]
 [ 6.23629500e-01  9.86352180e-01  1.45391758e+00 -4.66154857e-01
   9.36403332e-01  1.38499134e+00  3.49435894e-02 -1.07296428e+00
   4.95158611e-01  6.61681076e-01]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 1.18559003e+00  8.64644065e-02  1.23289919e+00 -2.14246673e+00
   1.03334100e+00 -8.30168864e-01  3.67231814e-01  4.51615951e-01
   1.10417433e+00 -4.22856961e-01]
 [ 9.71395338e-01  7.13049050e-01  1.76041518e+00  1.30620607e+00
   1.05765490e+00 -6.04602969e-01  1.28376990e-01  6.36583409e-01
   1.40925339e+00  9.66539250e-01]]
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
[[ 0.62368852  1.2066079   0.90399917 -0.28286355 -1.18913787 -0.26632688
   1.42361443  1.06897162  0.04037143  1.57546791]
 [ 1.09488485 -0.06962454 -0.11644415  0.35387043 -1.44189096 -0.18695502
   1.2911889  -0.15323616 -2.43250851 -2.277298  ]
 [ 1.32857949 -0.5632366  -1.06179676  2.39014596 -1.6845077   0.24542285
  -0.56914865  1.15259914 -0.22423577  0.13224778]
 [ 0.55853873 -0.51634791 -0.51814555  0.3511169   0.82550695 -0.06877046
  -0.9520621  -1.34776494  1.47073986 -1.4614036 ]
 [ 0.56998385 -0.53302033 -0.17545897 -1.42655542  0.60660431  1.76795995
  -0.11598519 -0.47537288  0.47761018 -0.93391466]
 [ 1.18468624 -1.00016919 -0.59384307  1.04499441  0.96548233  0.6085147
  -0.625342   -0.0693287  -0.10839207 -0.34390071]
 [ 0.93621125  0.20437739 -1.49419377  0.61223252 -0.98437725  0.74488454
   0.49434165 -0.03628129 -0.83239535 -0.4466992 ]
 [ 1.07258847 -0.58652394 -1.34267579 -1.23685338  1.24328724  0.87583893
  -0.3264995   0.62336218 -0.43495668  1.11438298]
 [ 1.16755486  0.0353601   0.7147896  -1.53879325  1.10863359 -0.44789518
  -1.75592564  0.61798553 -0.18417633  0.85270406]
 [ 0.10593645 -0.73728963  0.65032321  0.16466507 -1.53556118  0.77817418
   0.05031709  0.30981676  1.05132077  0.6065484 ]
 [ 0.87122579 -0.20975294 -0.45698786  0.93514778 -0.87353582  1.81252782
   0.92550121  0.14010988 -1.41914878  1.06898597]
 [ 0.84806927  0.45194604  0.63019567 -1.57915629  0.82798737 -0.82862798
  -0.10534471  0.52887975 -2.23708651 -0.4148469 ]
 [ 1.1437713   0.7278135   0.35249436  0.51507361  1.17718111 -2.78253447
  -1.94332341  0.58464661  0.32427424 -0.23643695]
 [ 0.8786438   1.03703898 -0.47712421  0.67261975 -1.04948638  2.42887697
   0.52475049  1.00568668  0.35356722 -0.03599018]
 [ 1.06523311 -0.66486777  1.00806543 -1.94504696 -1.23017555 -0.91542437
   0.33722094  1.22515585 -1.05354607  0.78522692]
 [ 0.97139534  0.71304905  1.76041518  1.30620607  1.0576549  -0.60460297
   0.12837699  0.63658341  1.40925339  0.96653925]
 [ 1.46893146 -1.47115693  0.58591043 -0.8301719   1.03345052 -0.8805776
  -0.95542526 -0.27909772  1.62284909  2.06578332]
 [ 0.85335555 -0.70435033 -0.67938378 -0.04586669 -1.29936179 -0.21873346
   0.59003946  1.53920701 -1.14870423 -0.95090925]
 [ 1.01177337  0.09574677  0.73140252  1.0334508  -1.42203164 -0.14627327
  -0.01745495 -0.85749682 -0.93418184  0.95449567]
 [ 0.98379959 -0.40724002  0.93272141  0.16056499 -1.278618   -0.12014998
   0.19975956  0.38560229  0.71829074 -0.5301198 ]
 [ 2.07582971 -1.40232915 -0.47918492  0.45112294  1.03436581 -0.6949209
  -0.4189379   0.5154138  -1.11487105 -1.95210529]
 [ 0.77370361  1.27852808 -2.11416392 -0.44222928  1.06821044  0.32352735
  -2.50644065 -0.10999149  0.00854895 -0.41163916]
 [ 0.87226739 -2.51630386 -0.77507029 -0.59566788  1.02600767 -0.30912132
   1.74643509  0.51093777  1.71066184  0.14164054]
 [ 0.69211449 -0.06065249  2.05635552 -2.413503    1.17456965 -1.77756638
  -0.28173627 -0.77785883  1.11584111  1.76024923]
 [ 0.96703727  0.38271517 -0.80618482 -0.28899734  0.90852604 -0.39181624
   1.62091229  0.68400133 -0.35340998 -0.25167421]]
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
[master c15dcfb] ml_store
 1 file changed, 271 insertions(+)
To github.com:arita37/mlmodels_store.git
   e735a78..c15dcfb  master -> master





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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @139621450219472
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @139621450219248
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @139621450218016
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @139621450217568
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @139621450217064
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @139621450216728

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
grad_step = 000000, loss = 1.364914
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 1.151572
grad_step = 000002, loss = 0.987742
grad_step = 000003, loss = 0.814298
grad_step = 000004, loss = 0.609500
grad_step = 000005, loss = 0.383290
grad_step = 000006, loss = 0.193249
grad_step = 000007, loss = 0.145138
grad_step = 000008, loss = 0.224689
grad_step = 000009, loss = 0.179781
grad_step = 000010, loss = 0.076632
grad_step = 000011, loss = 0.026614
grad_step = 000012, loss = 0.032605
grad_step = 000013, loss = 0.052584
grad_step = 000014, loss = 0.066372
grad_step = 000015, loss = 0.069556
grad_step = 000016, loss = 0.063782
grad_step = 000017, loss = 0.052293
grad_step = 000018, loss = 0.038608
grad_step = 000019, loss = 0.026509
grad_step = 000020, loss = 0.019209
grad_step = 000021, loss = 0.017795
grad_step = 000022, loss = 0.019943
grad_step = 000023, loss = 0.021761
grad_step = 000024, loss = 0.020963
grad_step = 000025, loss = 0.018292
grad_step = 000026, loss = 0.016218
grad_step = 000027, loss = 0.015974
grad_step = 000028, loss = 0.016450
grad_step = 000029, loss = 0.016627
grad_step = 000030, loss = 0.015481
grad_step = 000031, loss = 0.013110
grad_step = 000032, loss = 0.010418
grad_step = 000033, loss = 0.008529
grad_step = 000034, loss = 0.008157
grad_step = 000035, loss = 0.009025
grad_step = 000036, loss = 0.010142
grad_step = 000037, loss = 0.010608
grad_step = 000038, loss = 0.010211
grad_step = 000039, loss = 0.009357
grad_step = 000040, loss = 0.008598
grad_step = 000041, loss = 0.008082
grad_step = 000042, loss = 0.007720
grad_step = 000043, loss = 0.007383
grad_step = 000044, loss = 0.007062
grad_step = 000045, loss = 0.006871
grad_step = 000046, loss = 0.006911
grad_step = 000047, loss = 0.007122
grad_step = 000048, loss = 0.007315
grad_step = 000049, loss = 0.007289
grad_step = 000050, loss = 0.006998
grad_step = 000051, loss = 0.006589
grad_step = 000052, loss = 0.006267
grad_step = 000053, loss = 0.006138
grad_step = 000054, loss = 0.006156
grad_step = 000055, loss = 0.006205
grad_step = 000056, loss = 0.006194
grad_step = 000057, loss = 0.006115
grad_step = 000058, loss = 0.006021
grad_step = 000059, loss = 0.005956
grad_step = 000060, loss = 0.005907
grad_step = 000061, loss = 0.005843
grad_step = 000062, loss = 0.005743
grad_step = 000063, loss = 0.005639
grad_step = 000064, loss = 0.005579
grad_step = 000065, loss = 0.005558
grad_step = 000066, loss = 0.005521
grad_step = 000067, loss = 0.005440
grad_step = 000068, loss = 0.005347
grad_step = 000069, loss = 0.005287
grad_step = 000070, loss = 0.005261
grad_step = 000071, loss = 0.005244
grad_step = 000072, loss = 0.005208
grad_step = 000073, loss = 0.005141
grad_step = 000074, loss = 0.005063
grad_step = 000075, loss = 0.004992
grad_step = 000076, loss = 0.004934
grad_step = 000077, loss = 0.004880
grad_step = 000078, loss = 0.004828
grad_step = 000079, loss = 0.004766
grad_step = 000080, loss = 0.004712
grad_step = 000081, loss = 0.004662
grad_step = 000082, loss = 0.004605
grad_step = 000083, loss = 0.004539
grad_step = 000084, loss = 0.004472
grad_step = 000085, loss = 0.004406
grad_step = 000086, loss = 0.004343
grad_step = 000087, loss = 0.004283
grad_step = 000088, loss = 0.004225
grad_step = 000089, loss = 0.004162
grad_step = 000090, loss = 0.004097
grad_step = 000091, loss = 0.004032
grad_step = 000092, loss = 0.003965
grad_step = 000093, loss = 0.003894
grad_step = 000094, loss = 0.003822
grad_step = 000095, loss = 0.003750
grad_step = 000096, loss = 0.003678
grad_step = 000097, loss = 0.003604
grad_step = 000098, loss = 0.003530
grad_step = 000099, loss = 0.003455
grad_step = 000100, loss = 0.003380
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.003306
grad_step = 000102, loss = 0.003231
grad_step = 000103, loss = 0.003153
grad_step = 000104, loss = 0.003074
grad_step = 000105, loss = 0.002997
grad_step = 000106, loss = 0.002920
grad_step = 000107, loss = 0.002845
grad_step = 000108, loss = 0.002773
grad_step = 000109, loss = 0.002704
grad_step = 000110, loss = 0.002636
grad_step = 000111, loss = 0.002571
grad_step = 000112, loss = 0.002509
grad_step = 000113, loss = 0.002451
grad_step = 000114, loss = 0.002394
grad_step = 000115, loss = 0.002341
grad_step = 000116, loss = 0.002289
grad_step = 000117, loss = 0.002236
grad_step = 000118, loss = 0.002183
grad_step = 000119, loss = 0.002129
grad_step = 000120, loss = 0.002072
grad_step = 000121, loss = 0.002012
grad_step = 000122, loss = 0.001951
grad_step = 000123, loss = 0.001890
grad_step = 000124, loss = 0.001834
grad_step = 000125, loss = 0.001779
grad_step = 000126, loss = 0.001725
grad_step = 000127, loss = 0.001673
grad_step = 000128, loss = 0.001621
grad_step = 000129, loss = 0.001568
grad_step = 000130, loss = 0.001515
grad_step = 000131, loss = 0.001462
grad_step = 000132, loss = 0.001411
grad_step = 000133, loss = 0.001363
grad_step = 000134, loss = 0.001319
grad_step = 000135, loss = 0.001281
grad_step = 000136, loss = 0.001248
grad_step = 000137, loss = 0.001223
grad_step = 000138, loss = 0.001205
grad_step = 000139, loss = 0.001184
grad_step = 000140, loss = 0.001153
grad_step = 000141, loss = 0.001125
grad_step = 000142, loss = 0.001116
grad_step = 000143, loss = 0.001117
grad_step = 000144, loss = 0.001104
grad_step = 000145, loss = 0.001081
grad_step = 000146, loss = 0.001069
grad_step = 000147, loss = 0.001066
grad_step = 000148, loss = 0.001056
grad_step = 000149, loss = 0.001034
grad_step = 000150, loss = 0.001020
grad_step = 000151, loss = 0.001013
grad_step = 000152, loss = 0.001000
grad_step = 000153, loss = 0.000981
grad_step = 000154, loss = 0.000970
grad_step = 000155, loss = 0.000962
grad_step = 000156, loss = 0.000949
grad_step = 000157, loss = 0.000936
grad_step = 000158, loss = 0.000928
grad_step = 000159, loss = 0.000920
grad_step = 000160, loss = 0.000908
grad_step = 000161, loss = 0.000899
grad_step = 000162, loss = 0.000893
grad_step = 000163, loss = 0.000884
grad_step = 000164, loss = 0.000875
grad_step = 000165, loss = 0.000868
grad_step = 000166, loss = 0.000861
grad_step = 000167, loss = 0.000854
grad_step = 000168, loss = 0.000845
grad_step = 000169, loss = 0.000838
grad_step = 000170, loss = 0.000832
grad_step = 000171, loss = 0.000827
grad_step = 000172, loss = 0.000819
grad_step = 000173, loss = 0.000812
grad_step = 000174, loss = 0.000805
grad_step = 000175, loss = 0.000799
grad_step = 000176, loss = 0.000793
grad_step = 000177, loss = 0.000788
grad_step = 000178, loss = 0.000782
grad_step = 000179, loss = 0.000776
grad_step = 000180, loss = 0.000769
grad_step = 000181, loss = 0.000763
grad_step = 000182, loss = 0.000757
grad_step = 000183, loss = 0.000752
grad_step = 000184, loss = 0.000747
grad_step = 000185, loss = 0.000742
grad_step = 000186, loss = 0.000737
grad_step = 000187, loss = 0.000734
grad_step = 000188, loss = 0.000735
grad_step = 000189, loss = 0.000741
grad_step = 000190, loss = 0.000746
grad_step = 000191, loss = 0.000742
grad_step = 000192, loss = 0.000736
grad_step = 000193, loss = 0.000712
grad_step = 000194, loss = 0.000702
grad_step = 000195, loss = 0.000713
grad_step = 000196, loss = 0.000712
grad_step = 000197, loss = 0.000698
grad_step = 000198, loss = 0.000688
grad_step = 000199, loss = 0.000687
grad_step = 000200, loss = 0.000688
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.000684
grad_step = 000202, loss = 0.000675
grad_step = 000203, loss = 0.000667
grad_step = 000204, loss = 0.000668
grad_step = 000205, loss = 0.000669
grad_step = 000206, loss = 0.000661
grad_step = 000207, loss = 0.000653
grad_step = 000208, loss = 0.000650
grad_step = 000209, loss = 0.000648
grad_step = 000210, loss = 0.000647
grad_step = 000211, loss = 0.000644
grad_step = 000212, loss = 0.000640
grad_step = 000213, loss = 0.000633
grad_step = 000214, loss = 0.000629
grad_step = 000215, loss = 0.000626
grad_step = 000216, loss = 0.000625
grad_step = 000217, loss = 0.000624
grad_step = 000218, loss = 0.000621
grad_step = 000219, loss = 0.000618
grad_step = 000220, loss = 0.000614
grad_step = 000221, loss = 0.000610
grad_step = 000222, loss = 0.000606
grad_step = 000223, loss = 0.000602
grad_step = 000224, loss = 0.000598
grad_step = 000225, loss = 0.000594
grad_step = 000226, loss = 0.000591
grad_step = 000227, loss = 0.000589
grad_step = 000228, loss = 0.000588
grad_step = 000229, loss = 0.000590
grad_step = 000230, loss = 0.000599
grad_step = 000231, loss = 0.000614
grad_step = 000232, loss = 0.000641
grad_step = 000233, loss = 0.000652
grad_step = 000234, loss = 0.000650
grad_step = 000235, loss = 0.000593
grad_step = 000236, loss = 0.000570
grad_step = 000237, loss = 0.000596
grad_step = 000238, loss = 0.000606
grad_step = 000239, loss = 0.000582
grad_step = 000240, loss = 0.000557
grad_step = 000241, loss = 0.000574
grad_step = 000242, loss = 0.000589
grad_step = 000243, loss = 0.000561
grad_step = 000244, loss = 0.000546
grad_step = 000245, loss = 0.000564
grad_step = 000246, loss = 0.000563
grad_step = 000247, loss = 0.000545
grad_step = 000248, loss = 0.000541
grad_step = 000249, loss = 0.000549
grad_step = 000250, loss = 0.000544
grad_step = 000251, loss = 0.000530
grad_step = 000252, loss = 0.000531
grad_step = 000253, loss = 0.000539
grad_step = 000254, loss = 0.000533
grad_step = 000255, loss = 0.000526
grad_step = 000256, loss = 0.000526
grad_step = 000257, loss = 0.000529
grad_step = 000258, loss = 0.000522
grad_step = 000259, loss = 0.000513
grad_step = 000260, loss = 0.000511
grad_step = 000261, loss = 0.000514
grad_step = 000262, loss = 0.000512
grad_step = 000263, loss = 0.000508
grad_step = 000264, loss = 0.000510
grad_step = 000265, loss = 0.000514
grad_step = 000266, loss = 0.000516
grad_step = 000267, loss = 0.000510
grad_step = 000268, loss = 0.000501
grad_step = 000269, loss = 0.000494
grad_step = 000270, loss = 0.000493
grad_step = 000271, loss = 0.000494
grad_step = 000272, loss = 0.000494
grad_step = 000273, loss = 0.000495
grad_step = 000274, loss = 0.000494
grad_step = 000275, loss = 0.000490
grad_step = 000276, loss = 0.000485
grad_step = 000277, loss = 0.000480
grad_step = 000278, loss = 0.000477
grad_step = 000279, loss = 0.000478
grad_step = 000280, loss = 0.000479
grad_step = 000281, loss = 0.000479
grad_step = 000282, loss = 0.000477
grad_step = 000283, loss = 0.000474
grad_step = 000284, loss = 0.000470
grad_step = 000285, loss = 0.000467
grad_step = 000286, loss = 0.000464
grad_step = 000287, loss = 0.000463
grad_step = 000288, loss = 0.000462
grad_step = 000289, loss = 0.000462
grad_step = 000290, loss = 0.000463
grad_step = 000291, loss = 0.000464
grad_step = 000292, loss = 0.000469
grad_step = 000293, loss = 0.000474
grad_step = 000294, loss = 0.000481
grad_step = 000295, loss = 0.000485
grad_step = 000296, loss = 0.000487
grad_step = 000297, loss = 0.000479
grad_step = 000298, loss = 0.000468
grad_step = 000299, loss = 0.000456
grad_step = 000300, loss = 0.000450
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000451
grad_step = 000302, loss = 0.000458
grad_step = 000303, loss = 0.000463
grad_step = 000304, loss = 0.000460
grad_step = 000305, loss = 0.000448
grad_step = 000306, loss = 0.000438
grad_step = 000307, loss = 0.000437
grad_step = 000308, loss = 0.000442
grad_step = 000309, loss = 0.000448
grad_step = 000310, loss = 0.000450
grad_step = 000311, loss = 0.000448
grad_step = 000312, loss = 0.000450
grad_step = 000313, loss = 0.000454
grad_step = 000314, loss = 0.000468
grad_step = 000315, loss = 0.000479
grad_step = 000316, loss = 0.000488
grad_step = 000317, loss = 0.000470
grad_step = 000318, loss = 0.000444
grad_step = 000319, loss = 0.000421
grad_step = 000320, loss = 0.000416
grad_step = 000321, loss = 0.000426
grad_step = 000322, loss = 0.000439
grad_step = 000323, loss = 0.000446
grad_step = 000324, loss = 0.000438
grad_step = 000325, loss = 0.000424
grad_step = 000326, loss = 0.000411
grad_step = 000327, loss = 0.000408
grad_step = 000328, loss = 0.000414
grad_step = 000329, loss = 0.000421
grad_step = 000330, loss = 0.000425
grad_step = 000331, loss = 0.000421
grad_step = 000332, loss = 0.000415
grad_step = 000333, loss = 0.000405
grad_step = 000334, loss = 0.000398
grad_step = 000335, loss = 0.000396
grad_step = 000336, loss = 0.000398
grad_step = 000337, loss = 0.000403
grad_step = 000338, loss = 0.000406
grad_step = 000339, loss = 0.000409
grad_step = 000340, loss = 0.000407
grad_step = 000341, loss = 0.000403
grad_step = 000342, loss = 0.000397
grad_step = 000343, loss = 0.000392
grad_step = 000344, loss = 0.000388
grad_step = 000345, loss = 0.000384
grad_step = 000346, loss = 0.000382
grad_step = 000347, loss = 0.000382
grad_step = 000348, loss = 0.000383
grad_step = 000349, loss = 0.000386
grad_step = 000350, loss = 0.000392
grad_step = 000351, loss = 0.000401
grad_step = 000352, loss = 0.000416
grad_step = 000353, loss = 0.000427
grad_step = 000354, loss = 0.000439
grad_step = 000355, loss = 0.000426
grad_step = 000356, loss = 0.000410
grad_step = 000357, loss = 0.000395
grad_step = 000358, loss = 0.000394
grad_step = 000359, loss = 0.000393
grad_step = 000360, loss = 0.000386
grad_step = 000361, loss = 0.000378
grad_step = 000362, loss = 0.000372
grad_step = 000363, loss = 0.000373
grad_step = 000364, loss = 0.000376
grad_step = 000365, loss = 0.000377
grad_step = 000366, loss = 0.000372
grad_step = 000367, loss = 0.000365
grad_step = 000368, loss = 0.000359
grad_step = 000369, loss = 0.000356
grad_step = 000370, loss = 0.000357
grad_step = 000371, loss = 0.000359
grad_step = 000372, loss = 0.000360
grad_step = 000373, loss = 0.000358
grad_step = 000374, loss = 0.000354
grad_step = 000375, loss = 0.000348
grad_step = 000376, loss = 0.000343
grad_step = 000377, loss = 0.000340
grad_step = 000378, loss = 0.000339
grad_step = 000379, loss = 0.000341
grad_step = 000380, loss = 0.000345
grad_step = 000381, loss = 0.000354
grad_step = 000382, loss = 0.000365
grad_step = 000383, loss = 0.000385
grad_step = 000384, loss = 0.000398
grad_step = 000385, loss = 0.000413
grad_step = 000386, loss = 0.000394
grad_step = 000387, loss = 0.000362
grad_step = 000388, loss = 0.000334
grad_step = 000389, loss = 0.000331
grad_step = 000390, loss = 0.000348
grad_step = 000391, loss = 0.000361
grad_step = 000392, loss = 0.000365
grad_step = 000393, loss = 0.000347
grad_step = 000394, loss = 0.000328
grad_step = 000395, loss = 0.000319
grad_step = 000396, loss = 0.000323
grad_step = 000397, loss = 0.000334
grad_step = 000398, loss = 0.000341
grad_step = 000399, loss = 0.000343
grad_step = 000400, loss = 0.000343
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000354
grad_step = 000402, loss = 0.000392
grad_step = 000403, loss = 0.000436
grad_step = 000404, loss = 0.000456
grad_step = 000405, loss = 0.000381
grad_step = 000406, loss = 0.000319
grad_step = 000407, loss = 0.000333
grad_step = 000408, loss = 0.000372
grad_step = 000409, loss = 0.000360
grad_step = 000410, loss = 0.000314
grad_step = 000411, loss = 0.000309
grad_step = 000412, loss = 0.000338
grad_step = 000413, loss = 0.000336
grad_step = 000414, loss = 0.000310
grad_step = 000415, loss = 0.000308
grad_step = 000416, loss = 0.000323
grad_step = 000417, loss = 0.000319
grad_step = 000418, loss = 0.000305
grad_step = 000419, loss = 0.000313
grad_step = 000420, loss = 0.000323
grad_step = 000421, loss = 0.000316
grad_step = 000422, loss = 0.000307
grad_step = 000423, loss = 0.000313
grad_step = 000424, loss = 0.000319
grad_step = 000425, loss = 0.000318
grad_step = 000426, loss = 0.000311
grad_step = 000427, loss = 0.000314
grad_step = 000428, loss = 0.000316
grad_step = 000429, loss = 0.000310
grad_step = 000430, loss = 0.000297
grad_step = 000431, loss = 0.000290
grad_step = 000432, loss = 0.000289
grad_step = 000433, loss = 0.000288
grad_step = 000434, loss = 0.000285
grad_step = 000435, loss = 0.000283
grad_step = 000436, loss = 0.000284
grad_step = 000437, loss = 0.000289
grad_step = 000438, loss = 0.000296
grad_step = 000439, loss = 0.000303
grad_step = 000440, loss = 0.000314
grad_step = 000441, loss = 0.000323
grad_step = 000442, loss = 0.000338
grad_step = 000443, loss = 0.000336
grad_step = 000444, loss = 0.000325
grad_step = 000445, loss = 0.000302
grad_step = 000446, loss = 0.000283
grad_step = 000447, loss = 0.000278
grad_step = 000448, loss = 0.000285
grad_step = 000449, loss = 0.000299
grad_step = 000450, loss = 0.000309
grad_step = 000451, loss = 0.000316
grad_step = 000452, loss = 0.000311
grad_step = 000453, loss = 0.000298
grad_step = 000454, loss = 0.000280
grad_step = 000455, loss = 0.000269
grad_step = 000456, loss = 0.000268
grad_step = 000457, loss = 0.000275
grad_step = 000458, loss = 0.000285
grad_step = 000459, loss = 0.000292
grad_step = 000460, loss = 0.000294
grad_step = 000461, loss = 0.000287
grad_step = 000462, loss = 0.000277
grad_step = 000463, loss = 0.000267
grad_step = 000464, loss = 0.000261
grad_step = 000465, loss = 0.000260
grad_step = 000466, loss = 0.000264
grad_step = 000467, loss = 0.000270
grad_step = 000468, loss = 0.000276
grad_step = 000469, loss = 0.000284
grad_step = 000470, loss = 0.000289
grad_step = 000471, loss = 0.000294
grad_step = 000472, loss = 0.000290
grad_step = 000473, loss = 0.000283
grad_step = 000474, loss = 0.000270
grad_step = 000475, loss = 0.000260
grad_step = 000476, loss = 0.000254
grad_step = 000477, loss = 0.000254
grad_step = 000478, loss = 0.000256
grad_step = 000479, loss = 0.000261
grad_step = 000480, loss = 0.000267
grad_step = 000481, loss = 0.000273
grad_step = 000482, loss = 0.000279
grad_step = 000483, loss = 0.000283
grad_step = 000484, loss = 0.000288
grad_step = 000485, loss = 0.000288
grad_step = 000486, loss = 0.000288
grad_step = 000487, loss = 0.000282
grad_step = 000488, loss = 0.000274
grad_step = 000489, loss = 0.000262
grad_step = 000490, loss = 0.000251
grad_step = 000491, loss = 0.000249
grad_step = 000492, loss = 0.000254
grad_step = 000493, loss = 0.000265
grad_step = 000494, loss = 0.000276
grad_step = 000495, loss = 0.000284
grad_step = 000496, loss = 0.000285
grad_step = 000497, loss = 0.000284
grad_step = 000498, loss = 0.000277
grad_step = 000499, loss = 0.000273
grad_step = 000500, loss = 0.000268
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000266
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
[[0.8588942  0.8602257  0.93111175 0.9541143  1.0075903 ]
 [0.8436369  0.9218097  0.9643978  1.0066938  0.9849736 ]
 [0.89483607 0.9284357  0.9898519  0.98899025 0.95715094]
 [0.92463547 1.0147842  1.0010242  0.94964814 0.90802157]
 [0.9932102  1.0022079  0.9418242  0.92567956 0.85211813]
 [0.99352515 0.9615785  0.9247694  0.86613166 0.8513402 ]
 [0.94517106 0.90940964 0.85261923 0.87200797 0.82091486]
 [0.8907626  0.8480674  0.8576121  0.82139933 0.8503653 ]
 [0.8240893  0.84413177 0.8228178  0.8487412  0.8460071 ]
 [0.8295937  0.8247157  0.81582534 0.8641517  0.8476987 ]
 [0.7861326  0.82073593 0.8501195  0.8215717  0.9253121 ]
 [0.8277862  0.8462211  0.84300625 0.9306371  0.92516124]
 [0.8483828  0.8574965  0.9297937  0.95153147 1.0025468 ]
 [0.8496695  0.9314819  0.9679144  1.006427   0.97140056]
 [0.9118835  0.9397141  0.9869784  0.9849415  0.9351303 ]
 [0.9415315  1.0195645  0.99073786 0.9415964  0.8856671 ]
 [0.99363285 0.9959954  0.9252703  0.909721   0.83372366]
 [0.986431   0.9468255  0.9095086  0.8411126  0.8405712 ]
 [0.93962216 0.9019904  0.8376218  0.8524648  0.8099272 ]
 [0.906484   0.84809434 0.8472431  0.81864166 0.8431432 ]
 [0.84014153 0.8547401  0.8271866  0.8476324  0.8468828 ]
 [0.8505318  0.8407419  0.82281613 0.87279856 0.84902245]
 [0.8008883  0.8335209  0.85591483 0.8245555  0.92185974]
 [0.83415    0.8535385  0.8470324  0.9322301  0.9294315 ]
 [0.865713   0.86507237 0.931387   0.9574628  1.0106075 ]
 [0.8536594  0.9264008  0.9689667  1.0144743  0.99356425]
 [0.9049493  0.9350838  0.9977769  1.0020405  0.96715945]
 [0.9391326  1.026855   1.0126947  0.96418583 0.91627026]
 [1.0010804  1.0154328  0.952443   0.9389751  0.8590046 ]
 [1.0041848  0.97461855 0.9356043  0.87336737 0.85845816]
 [0.9518256  0.91911685 0.8593253  0.87471414 0.82711333]]

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
[master 70c9376] ml_store
 1 file changed, 1122 insertions(+)
To github.com:arita37/mlmodels_store.git
   c15dcfb..70c9376  master -> master





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
[master cc9c97d] ml_store
 1 file changed, 37 insertions(+)
To github.com:arita37/mlmodels_store.git
   70c9376..cc9c97d  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//matchzoo_models.py 

  #### Loading params   ############################################## 

  {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'dataset_pars': {'data_pack': '', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'resample': True, 'sort': False, 'callbacks': 'PADDING'}, 'dataloader': '', 'dataloader_pars': {'device': 'cpu', 'dataset': 'None', 'stage': 'train', 'callback': 'PADDING'}, 'preprocess': {'train': {'transform': True, 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'stage': 'train', 'resample': True, 'sort': False, 'dataloader_callback': 'PADDING'}, 'test': {'transform': True, 'batch_size': 20, 'stage': 'dev', 'dataloader_callback': 'PADDING'}}} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Loading dataset   ############################################# 

  #### Model init   ################################################## 
  0%|          | 0/231508 [00:00<?, ?B/s] 23%|██▎       | 52224/231508 [00:00<00:00, 420202.83B/s] 53%|█████▎    | 121856/231508 [00:00<00:00, 451211.13B/s]100%|██████████| 231508/231508 [00:00<00:00, 899818.58B/s]
  0%|          | 0/433 [00:00<?, ?B/s]100%|██████████| 433/433 [00:00<00:00, 224435.69B/s]
  0%|          | 0/440473133 [00:00<?, ?B/s]  0%|          | 52224/440473133 [00:00<17:27, 420402.84B/s]  0%|          | 191488/440473133 [00:00<14:12, 516533.65B/s]  0%|          | 818176/440473133 [00:00<10:22, 706572.97B/s]  1%|          | 3201024/440473133 [00:00<07:18, 996722.61B/s]  1%|▏         | 6142976/440473133 [00:00<05:09, 1403510.51B/s]  2%|▏         | 9734144/440473133 [00:00<03:38, 1966952.67B/s]  3%|▎         | 13424640/440473133 [00:00<02:35, 2747168.71B/s]  4%|▎         | 16190464/440473133 [00:00<01:52, 3764098.93B/s]  5%|▍         | 19974144/440473133 [00:01<01:22, 5111774.96B/s]  5%|▌         | 23939072/440473133 [00:01<01:00, 6869016.99B/s]  6%|▋         | 28144640/440473133 [00:01<00:44, 9170911.67B/s]  7%|▋         | 31471616/440473133 [00:01<00:35, 11668624.30B/s]  8%|▊         | 35358720/440473133 [00:01<00:27, 14515115.25B/s]  9%|▉         | 39334912/440473133 [00:01<00:22, 17930490.72B/s] 10%|▉         | 42847232/440473133 [00:01<00:19, 20674088.61B/s] 11%|█         | 46499840/440473133 [00:01<00:16, 23378511.81B/s] 12%|█▏        | 51021824/440473133 [00:01<00:14, 26102741.68B/s] 13%|█▎        | 55285760/440473133 [00:01<00:13, 29539081.26B/s] 13%|█▎        | 58986496/440473133 [00:02<00:12, 30774352.57B/s] 14%|█▍        | 62621696/440473133 [00:02<00:11, 31789130.25B/s] 15%|█▌        | 66717696/440473133 [00:02<00:10, 34072410.44B/s] 16%|█▌        | 70439936/440473133 [00:02<00:10, 34506148.70B/s] 17%|█▋        | 74467328/440473133 [00:02<00:10, 35089083.32B/s] 18%|█▊        | 79242240/440473133 [00:02<00:09, 38121208.86B/s] 19%|█▉        | 83241984/440473133 [00:02<00:09, 37765892.76B/s] 20%|█▉        | 87394304/440473133 [00:02<00:09, 37885283.65B/s] 21%|██        | 92188672/440473133 [00:02<00:08, 40428732.94B/s] 22%|██▏       | 96344064/440473133 [00:03<00:08, 39995981.91B/s] 23%|██▎       | 100730880/440473133 [00:03<00:08, 40032352.95B/s] 24%|██▍       | 105257984/440473133 [00:03<00:08, 41468935.13B/s] 25%|██▍       | 109457408/440473133 [00:03<00:08, 41089316.44B/s] 26%|██▌       | 114051072/440473133 [00:03<00:07, 41046925.28B/s] 27%|██▋       | 119336960/440473133 [00:03<00:07, 43995397.24B/s] 28%|██▊       | 123814912/440473133 [00:03<00:07, 43566044.57B/s] 29%|██▉       | 128812032/440473133 [00:03<00:06, 45307506.28B/s] 30%|███       | 133399552/440473133 [00:03<00:06, 45450887.71B/s] 31%|███▏      | 137985024/440473133 [00:03<00:06, 43409815.57B/s] 32%|███▏      | 142608384/440473133 [00:04<00:06, 43186439.82B/s] 33%|███▎      | 147507200/440473133 [00:04<00:06, 44697554.57B/s] 35%|███▍      | 152013824/440473133 [00:04<00:06, 44432374.04B/s] 36%|███▌      | 156953600/440473133 [00:04<00:06, 45812561.72B/s] 37%|███▋      | 161564672/440473133 [00:04<00:06, 45620966.24B/s] 38%|███▊      | 166148096/440473133 [00:04<00:06, 45080191.56B/s] 39%|███▉      | 170739712/440473133 [00:04<00:05, 45324004.81B/s] 40%|███▉      | 175284224/440473133 [00:04<00:05, 44757170.43B/s] 41%|████      | 179886080/440473133 [00:04<00:05, 45128120.69B/s] 42%|████▏     | 184407040/440473133 [00:05<00:05, 44810386.95B/s] 43%|████▎     | 188894208/440473133 [00:05<00:05, 44598327.48B/s] 44%|████▍     | 193980416/440473133 [00:05<00:05, 46306695.23B/s] 45%|████▌     | 198631424/440473133 [00:05<00:05, 46113687.88B/s] 46%|████▌     | 203256832/440473133 [00:05<00:05, 45935343.94B/s] 47%|████▋     | 207884288/440473133 [00:05<00:05, 46034731.35B/s] 48%|████▊     | 212551680/440473133 [00:05<00:04, 46221020.90B/s] 49%|████▉     | 217206784/440473133 [00:05<00:04, 46319382.61B/s] 50%|█████     | 221974528/440473133 [00:05<00:04, 46718324.73B/s] 51%|█████▏    | 226666496/440473133 [00:05<00:04, 46777562.24B/s] 53%|█████▎    | 231479296/440473133 [00:06<00:04, 47171771.56B/s] 54%|█████▎    | 236276736/440473133 [00:06<00:04, 47409002.11B/s] 55%|█████▍    | 241121280/440473133 [00:06<00:04, 47712773.09B/s] 56%|█████▌    | 245895168/440473133 [00:06<00:04, 47506357.06B/s] 57%|█████▋    | 250647552/440473133 [00:06<00:04, 47127172.46B/s] 58%|█████▊    | 255392768/440473133 [00:06<00:03, 47223944.88B/s] 59%|█████▉    | 260179968/440473133 [00:06<00:03, 47415651.65B/s] 60%|██████    | 264923136/440473133 [00:06<00:03, 46998823.66B/s] 61%|██████    | 269672448/440473133 [00:06<00:03, 47144167.70B/s] 62%|██████▏   | 274388992/440473133 [00:06<00:03, 46554803.98B/s] 63%|██████▎   | 279047168/440473133 [00:07<00:03, 45083494.88B/s] 64%|██████▍   | 284004352/440473133 [00:07<00:03, 46341481.55B/s] 66%|██████▌   | 288656384/440473133 [00:07<00:03, 46349673.00B/s] 67%|██████▋   | 293488640/440473133 [00:07<00:03, 46922439.30B/s] 68%|██████▊   | 298191872/440473133 [00:07<00:03, 46533291.47B/s] 69%|██████▉   | 302915584/440473133 [00:07<00:02, 46719524.55B/s] 70%|██████▉   | 307593216/440473133 [00:07<00:02, 46533870.83B/s] 71%|███████   | 312510464/440473133 [00:07<00:02, 46989310.56B/s] 72%|███████▏  | 317364224/440473133 [00:07<00:02, 47442841.33B/s] 73%|███████▎  | 322259968/440473133 [00:07<00:02, 47886866.60B/s] 74%|███████▍  | 327053312/440473133 [00:08<00:02, 47569433.39B/s] 75%|███████▌  | 331840512/440473133 [00:08<00:02, 47658185.63B/s] 76%|███████▋  | 336609280/440473133 [00:08<00:02, 47315670.65B/s] 77%|███████▋  | 341343232/440473133 [00:08<00:02, 47114448.66B/s] 79%|███████▊  | 346177536/440473133 [00:08<00:01, 47475364.96B/s] 80%|███████▉  | 350927872/440473133 [00:08<00:01, 47234947.84B/s] 81%|████████  | 355944448/440473133 [00:08<00:01, 48076880.82B/s] 82%|████████▏ | 360757248/440473133 [00:08<00:01, 47129381.21B/s] 83%|████████▎ | 365512704/440473133 [00:08<00:01, 47129491.49B/s] 84%|████████▍ | 370231296/440473133 [00:08<00:01, 47013202.03B/s] 85%|████████▌ | 375113728/440473133 [00:09<00:01, 46825841.12B/s] 86%|████████▌ | 379906048/440473133 [00:09<00:01, 47145070.86B/s] 87%|████████▋ | 384635904/440473133 [00:09<00:01, 47189073.19B/s] 88%|████████▊ | 389482496/440473133 [00:09<00:01, 47559545.03B/s] 90%|████████▉ | 394241024/440473133 [00:09<00:00, 47337329.96B/s] 91%|█████████ | 399083520/440473133 [00:09<00:00, 47381135.72B/s] 92%|█████████▏| 404064256/440473133 [00:09<00:00, 48014437.25B/s] 93%|█████████▎| 408868864/440473133 [00:09<00:00, 47400638.27B/s] 94%|█████████▍| 413651968/440473133 [00:09<00:00, 47527740.72B/s] 95%|█████████▌| 418548736/440473133 [00:09<00:00, 47948917.73B/s] 96%|█████████▌| 423347200/440473133 [00:10<00:00, 40779482.62B/s] 97%|█████████▋| 427613184/440473133 [00:10<00:00, 40211123.08B/s] 98%|█████████▊| 431767552/440473133 [00:10<00:00, 39291103.10B/s] 99%|█████████▉| 435947520/440473133 [00:10<00:00, 38214357.44B/s]100%|█████████▉| 440368128/440473133 [00:10<00:00, 39833596.64B/s]100%|██████████| 440473133/440473133 [00:10<00:00, 41626732.48B/s]Downloading data from https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip

   8192/7094233 [..............................] - ETA: 0s
 851968/7094233 [==>...........................] - ETA: 0s
1892352/7094233 [=======>......................] - ETA: 0s
2932736/7094233 [===========>..................] - ETA: 0s
3719168/7094233 [==============>...............] - ETA: 0s
4759552/7094233 [===================>..........] - ETA: 0s
5537792/7094233 [======================>.......] - ETA: 0s
6324224/7094233 [=========================>....] - ETA: 0s
7094272/7094233 [==============================] - 0s 0us/step

Processing text_left with encode:   0%|          | 0/2118 [00:00<?, ?it/s]Processing text_left with encode:   9%|▉         | 196/2118 [00:00<00:00, 1957.90it/s]Processing text_left with encode:  32%|███▏      | 681/2118 [00:00<00:00, 2383.75it/s]Processing text_left with encode:  54%|█████▎    | 1136/2118 [00:00<00:00, 2780.65it/s]Processing text_left with encode:  74%|███████▎  | 1558/2118 [00:00<00:00, 3096.09it/s]Processing text_left with encode:  91%|█████████▏| 1933/2118 [00:00<00:00, 3265.54it/s]Processing text_left with encode: 100%|██████████| 2118/2118 [00:00<00:00, 3852.57it/s]
Processing text_right with encode:   0%|          | 0/18841 [00:00<?, ?it/s]Processing text_right with encode:   1%|          | 150/18841 [00:00<00:13, 1383.67it/s]Processing text_right with encode:   2%|▏         | 334/18841 [00:00<00:12, 1494.67it/s]Processing text_right with encode:   3%|▎         | 506/18841 [00:00<00:11, 1555.20it/s]Processing text_right with encode:   4%|▎         | 678/18841 [00:00<00:11, 1600.88it/s]Processing text_right with encode:   5%|▍         | 853/18841 [00:00<00:10, 1641.67it/s]Processing text_right with encode:   5%|▌         | 1036/18841 [00:00<00:10, 1692.86it/s]Processing text_right with encode:   6%|▋         | 1212/18841 [00:00<00:10, 1711.46it/s]Processing text_right with encode:   7%|▋         | 1375/18841 [00:00<00:10, 1684.04it/s]Processing text_right with encode:   8%|▊         | 1537/18841 [00:00<00:10, 1600.69it/s]Processing text_right with encode:   9%|▉         | 1694/18841 [00:01<00:10, 1588.90it/s]Processing text_right with encode:  10%|▉         | 1851/18841 [00:01<00:11, 1517.96it/s]Processing text_right with encode:  11%|█         | 2002/18841 [00:01<00:11, 1473.80it/s]Processing text_right with encode:  12%|█▏        | 2181/18841 [00:01<00:10, 1555.63it/s]Processing text_right with encode:  13%|█▎        | 2364/18841 [00:01<00:10, 1626.44it/s]Processing text_right with encode:  14%|█▎        | 2551/18841 [00:01<00:09, 1690.00it/s]Processing text_right with encode:  15%|█▍        | 2749/18841 [00:01<00:09, 1767.13it/s]Processing text_right with encode:  16%|█▌        | 2938/18841 [00:01<00:08, 1801.59it/s]Processing text_right with encode:  17%|█▋        | 3120/18841 [00:01<00:09, 1716.18it/s]Processing text_right with encode:  18%|█▊        | 3304/18841 [00:01<00:08, 1750.90it/s]Processing text_right with encode:  18%|█▊        | 3481/18841 [00:02<00:08, 1753.41it/s]Processing text_right with encode:  19%|█▉        | 3663/18841 [00:02<00:08, 1768.44it/s]Processing text_right with encode:  20%|██        | 3843/18841 [00:02<00:08, 1776.70it/s]Processing text_right with encode:  21%|██▏       | 4034/18841 [00:02<00:08, 1813.09it/s]Processing text_right with encode:  22%|██▏       | 4216/18841 [00:02<00:08, 1657.63it/s]Processing text_right with encode:  23%|██▎       | 4385/18841 [00:02<00:08, 1608.78it/s]Processing text_right with encode:  24%|██▍       | 4561/18841 [00:02<00:08, 1650.82it/s]Processing text_right with encode:  25%|██▌       | 4748/18841 [00:02<00:08, 1710.81it/s]Processing text_right with encode:  26%|██▌       | 4941/18841 [00:02<00:07, 1768.87it/s]Processing text_right with encode:  27%|██▋       | 5120/18841 [00:03<00:07, 1724.29it/s]Processing text_right with encode:  28%|██▊       | 5323/18841 [00:03<00:07, 1804.88it/s]Processing text_right with encode:  29%|██▉       | 5512/18841 [00:03<00:07, 1828.47it/s]Processing text_right with encode:  30%|███       | 5697/18841 [00:03<00:07, 1830.84it/s]Processing text_right with encode:  31%|███       | 5882/18841 [00:03<00:07, 1756.59it/s]Processing text_right with encode:  32%|███▏      | 6060/18841 [00:03<00:07, 1631.28it/s]Processing text_right with encode:  33%|███▎      | 6226/18841 [00:03<00:08, 1552.03it/s]Processing text_right with encode:  34%|███▍      | 6384/18841 [00:03<00:08, 1537.46it/s]Processing text_right with encode:  35%|███▍      | 6574/18841 [00:03<00:07, 1630.06it/s]Processing text_right with encode:  36%|███▌      | 6756/18841 [00:03<00:07, 1681.30it/s]Processing text_right with encode:  37%|███▋      | 6939/18841 [00:04<00:06, 1722.18it/s]Processing text_right with encode:  38%|███▊      | 7116/18841 [00:04<00:06, 1735.07it/s]Processing text_right with encode:  39%|███▉      | 7309/18841 [00:04<00:06, 1788.36it/s]Processing text_right with encode:  40%|███▉      | 7499/18841 [00:04<00:06, 1819.16it/s]Processing text_right with encode:  41%|████      | 7683/18841 [00:04<00:06, 1793.27it/s]Processing text_right with encode:  42%|████▏     | 7864/18841 [00:04<00:06, 1705.93it/s]Processing text_right with encode:  43%|████▎     | 8037/18841 [00:04<00:06, 1593.93it/s]Processing text_right with encode:  44%|████▎     | 8199/18841 [00:04<00:06, 1565.53it/s]Processing text_right with encode:  44%|████▍     | 8367/18841 [00:04<00:06, 1596.78it/s]Processing text_right with encode:  45%|████▌     | 8529/18841 [00:05<00:06, 1515.28it/s]Processing text_right with encode:  46%|████▌     | 8683/18841 [00:05<00:06, 1516.58it/s]Processing text_right with encode:  47%|████▋     | 8836/18841 [00:05<00:06, 1513.88it/s]Processing text_right with encode:  48%|████▊     | 9012/18841 [00:05<00:06, 1579.37it/s]Processing text_right with encode:  49%|████▉     | 9191/18841 [00:05<00:05, 1636.54it/s]Processing text_right with encode:  50%|████▉     | 9369/18841 [00:05<00:05, 1676.34it/s]Processing text_right with encode:  51%|█████     | 9538/18841 [00:05<00:05, 1609.77it/s]Processing text_right with encode:  51%|█████▏    | 9701/18841 [00:05<00:05, 1566.96it/s]Processing text_right with encode:  52%|█████▏    | 9859/18841 [00:05<00:06, 1490.94it/s]Processing text_right with encode:  53%|█████▎    | 10032/18841 [00:06<00:05, 1553.87it/s]Processing text_right with encode:  54%|█████▍    | 10190/18841 [00:06<00:05, 1555.94it/s]Processing text_right with encode:  55%|█████▌    | 10372/18841 [00:06<00:05, 1626.09it/s]Processing text_right with encode:  56%|█████▌    | 10564/18841 [00:06<00:04, 1704.35it/s]Processing text_right with encode:  57%|█████▋    | 10741/18841 [00:06<00:04, 1721.73it/s]Processing text_right with encode:  58%|█████▊    | 10923/18841 [00:06<00:04, 1749.72it/s]Processing text_right with encode:  59%|█████▉    | 11100/18841 [00:06<00:04, 1755.48it/s]Processing text_right with encode:  60%|█████▉    | 11279/18841 [00:06<00:04, 1764.60it/s]Processing text_right with encode:  61%|██████    | 11469/18841 [00:06<00:04, 1801.62it/s]Processing text_right with encode:  62%|██████▏   | 11650/18841 [00:06<00:04, 1790.99it/s]Processing text_right with encode:  63%|██████▎   | 11830/18841 [00:07<00:04, 1743.68it/s]Processing text_right with encode:  64%|██████▍   | 12021/18841 [00:07<00:03, 1785.56it/s]Processing text_right with encode:  65%|██████▍   | 12201/18841 [00:07<00:03, 1781.82it/s]Processing text_right with encode:  66%|██████▌   | 12389/18841 [00:07<00:03, 1809.17it/s]Processing text_right with encode:  67%|██████▋   | 12571/18841 [00:07<00:03, 1804.08it/s]Processing text_right with encode:  68%|██████▊   | 12752/18841 [00:07<00:03, 1762.44it/s]Processing text_right with encode:  69%|██████▊   | 12936/18841 [00:07<00:03, 1783.10it/s]Processing text_right with encode:  70%|██████▉   | 13116/18841 [00:07<00:03, 1785.51it/s]Processing text_right with encode:  71%|███████   | 13295/18841 [00:07<00:03, 1755.05it/s]Processing text_right with encode:  71%|███████▏  | 13471/18841 [00:07<00:03, 1749.55it/s]Processing text_right with encode:  72%|███████▏  | 13647/18841 [00:08<00:03, 1678.58it/s]Processing text_right with encode:  73%|███████▎  | 13832/18841 [00:08<00:02, 1724.66it/s]Processing text_right with encode:  74%|███████▍  | 14019/18841 [00:08<00:02, 1763.19it/s]Processing text_right with encode:  75%|███████▌  | 14197/18841 [00:08<00:02, 1759.48it/s]Processing text_right with encode:  76%|███████▋  | 14375/18841 [00:08<00:02, 1764.53it/s]Processing text_right with encode:  77%|███████▋  | 14552/18841 [00:08<00:02, 1753.87it/s]Processing text_right with encode:  78%|███████▊  | 14744/18841 [00:08<00:02, 1799.20it/s]Processing text_right with encode:  79%|███████▉  | 14925/18841 [00:08<00:02, 1786.76it/s]Processing text_right with encode:  80%|████████  | 15112/18841 [00:08<00:02, 1808.69it/s]Processing text_right with encode:  81%|████████  | 15294/18841 [00:08<00:01, 1792.07it/s]Processing text_right with encode:  82%|████████▏ | 15476/18841 [00:09<00:01, 1798.33it/s]Processing text_right with encode:  83%|████████▎ | 15657/18841 [00:09<00:01, 1783.58it/s]Processing text_right with encode:  84%|████████▍ | 15851/18841 [00:09<00:01, 1825.68it/s]Processing text_right with encode:  85%|████████▌ | 16034/18841 [00:09<00:01, 1819.85it/s]Processing text_right with encode:  86%|████████▌ | 16217/18841 [00:09<00:01, 1803.22it/s]Processing text_right with encode:  87%|████████▋ | 16398/18841 [00:09<00:01, 1789.30it/s]Processing text_right with encode:  88%|████████▊ | 16584/18841 [00:09<00:01, 1808.86it/s]Processing text_right with encode:  89%|████████▉ | 16766/18841 [00:09<00:01, 1806.22it/s]Processing text_right with encode:  90%|████████▉ | 16947/18841 [00:09<00:01, 1786.58it/s]Processing text_right with encode:  91%|█████████ | 17126/18841 [00:09<00:00, 1784.49it/s]Processing text_right with encode:  92%|█████████▏| 17306/18841 [00:10<00:00, 1786.99it/s]Processing text_right with encode:  93%|█████████▎| 17488/18841 [00:10<00:00, 1796.14it/s]Processing text_right with encode:  94%|█████████▍| 17681/18841 [00:10<00:00, 1833.50it/s]Processing text_right with encode:  95%|█████████▍| 17865/18841 [00:10<00:00, 1833.27it/s]Processing text_right with encode:  96%|█████████▌| 18064/18841 [00:10<00:00, 1872.72it/s]Processing text_right with encode:  97%|█████████▋| 18252/18841 [00:10<00:00, 1718.19it/s]Processing text_right with encode:  98%|█████████▊| 18427/18841 [00:10<00:00, 1669.97it/s]Processing text_right with encode:  99%|█████████▊| 18600/18841 [00:10<00:00, 1685.04it/s]Processing text_right with encode: 100%|█████████▉| 18785/18841 [00:10<00:00, 1731.21it/s]Processing text_right with encode: 100%|██████████| 18841/18841 [00:10<00:00, 1717.07it/s]
Processing length_left with len:   0%|          | 0/2118 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 2118/2118 [00:00<00:00, 531343.73it/s]
Processing length_right with len:   0%|          | 0/18841 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 18841/18841 [00:00<00:00, 811252.13it/s]
Processing text_left with encode:   0%|          | 0/633 [00:00<?, ?it/s]Processing text_left with encode:  66%|██████▌   | 416/633 [00:00<00:00, 4157.13it/s]Processing text_left with encode: 100%|██████████| 633/633 [00:00<00:00, 4056.39it/s]
Processing text_right with encode:   0%|          | 0/5961 [00:00<?, ?it/s]Processing text_right with encode:   3%|▎         | 183/5961 [00:00<00:03, 1825.60it/s]Processing text_right with encode:   6%|▌         | 363/5961 [00:00<00:03, 1816.56it/s]Processing text_right with encode:   9%|▉         | 542/5961 [00:00<00:02, 1807.22it/s]Processing text_right with encode:  12%|█▏        | 717/5961 [00:00<00:02, 1788.31it/s]Processing text_right with encode:  15%|█▌        | 907/5961 [00:00<00:02, 1820.05it/s]Processing text_right with encode:  18%|█▊        | 1091/5961 [00:00<00:02, 1822.21it/s]Processing text_right with encode:  22%|██▏       | 1286/5961 [00:00<00:02, 1858.46it/s]Processing text_right with encode:  25%|██▍       | 1474/5961 [00:00<00:02, 1862.89it/s]Processing text_right with encode:  28%|██▊       | 1656/5961 [00:00<00:02, 1845.32it/s]Processing text_right with encode:  31%|███       | 1838/5961 [00:01<00:02, 1836.57it/s]Processing text_right with encode:  34%|███▍      | 2025/5961 [00:01<00:02, 1845.82it/s]Processing text_right with encode:  37%|███▋      | 2224/5961 [00:01<00:01, 1884.02it/s]Processing text_right with encode:  40%|████      | 2411/5961 [00:01<00:01, 1779.36it/s]Processing text_right with encode:  44%|████▎     | 2600/5961 [00:01<00:01, 1810.81it/s]Processing text_right with encode:  47%|████▋     | 2798/5961 [00:01<00:01, 1857.95it/s]Processing text_right with encode:  50%|█████     | 2986/5961 [00:01<00:01, 1861.90it/s]Processing text_right with encode:  53%|█████▎    | 3184/5961 [00:01<00:01, 1893.72it/s]Processing text_right with encode:  57%|█████▋    | 3374/5961 [00:01<00:01, 1854.07it/s]Processing text_right with encode:  60%|█████▉    | 3560/5961 [00:01<00:01, 1769.36it/s]Processing text_right with encode:  63%|██████▎   | 3738/5961 [00:02<00:01, 1649.50it/s]Processing text_right with encode:  66%|██████▌   | 3944/5961 [00:02<00:01, 1751.34it/s]Processing text_right with encode:  69%|██████▉   | 4138/5961 [00:02<00:01, 1803.30it/s]Processing text_right with encode:  73%|███████▎  | 4328/5961 [00:02<00:00, 1829.14it/s]Processing text_right with encode:  76%|███████▌  | 4513/5961 [00:02<00:00, 1763.84it/s]Processing text_right with encode:  79%|███████▉  | 4708/5961 [00:02<00:00, 1813.55it/s]Processing text_right with encode:  82%|████████▏ | 4892/5961 [00:02<00:00, 1804.45it/s]Processing text_right with encode:  85%|████████▌ | 5074/5961 [00:02<00:00, 1806.86it/s]Processing text_right with encode:  88%|████████▊ | 5271/5961 [00:02<00:00, 1849.18it/s]Processing text_right with encode:  92%|█████████▏| 5457/5961 [00:03<00:00, 1773.38it/s]Processing text_right with encode:  95%|█████████▍| 5636/5961 [00:03<00:00, 1745.44it/s]Processing text_right with encode:  98%|█████████▊| 5812/5961 [00:03<00:00, 1705.91it/s]Processing text_right with encode: 100%|██████████| 5961/5961 [00:03<00:00, 1808.56it/s]
Processing length_left with len:   0%|          | 0/633 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 633/633 [00:00<00:00, 461176.73it/s]
Processing length_right with len:   0%|          | 0/5961 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 5961/5961 [00:00<00:00, 852562.44it/s]
  #### Model  fit   ############################################# 

  0%|          | 0/102 [00:00<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:27<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:27<?, ?it/s, loss=1.057]Epoch 1/1:   1%|          | 1/102 [00:27<45:46, 27.20s/it, loss=1.057]Epoch 1/1:   1%|          | 1/102 [01:31<45:46, 27.20s/it, loss=1.057]Epoch 1/1:   1%|          | 1/102 [01:31<45:46, 27.20s/it, loss=0.978]Epoch 1/1:   2%|▏         | 2/102 [01:31<1:03:37, 38.18s/it, loss=0.978]Epoch 1/1:   2%|▏         | 2/102 [03:12<1:03:37, 38.18s/it, loss=0.978]Epoch 1/1:   2%|▏         | 2/102 [03:12<1:03:37, 38.18s/it, loss=1.022]Epoch 1/1:   3%|▎         | 3/102 [03:12<1:34:07, 57.05s/it, loss=1.022]Epoch 1/1:   3%|▎         | 3/102 [04:18<1:34:07, 57.05s/it, loss=1.022]Epoch 1/1:   3%|▎         | 3/102 [04:18<1:34:07, 57.05s/it, loss=0.770]Epoch 1/1:   4%|▍         | 4/102 [04:18<1:37:32, 59.72s/it, loss=0.770]Epoch 1/1:   4%|▍         | 4/102 [06:18<1:37:32, 59.72s/it, loss=0.770]Epoch 1/1:   4%|▍         | 4/102 [06:18<1:37:32, 59.72s/it, loss=1.073]Epoch 1/1:   5%|▍         | 5/102 [06:18<2:05:57, 77.91s/it, loss=1.073]Epoch 1/1:   5%|▍         | 5/102 [07:26<2:05:57, 77.91s/it, loss=1.073]Epoch 1/1:   5%|▍         | 5/102 [07:26<2:05:57, 77.91s/it, loss=0.906]Epoch 1/1:   6%|▌         | 6/102 [07:26<2:00:08, 75.09s/it, loss=0.906]Epoch 1/1:   6%|▌         | 6/102 [09:17<2:00:08, 75.09s/it, loss=0.906]Epoch 1/1:   6%|▌         | 6/102 [09:17<2:00:08, 75.09s/it, loss=0.752]Epoch 1/1:   7%|▋         | 7/102 [09:17<2:15:35, 85.64s/it, loss=0.752]Killed

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Warning: Permanently added the RSA host key for IP address '52.64.108.95' to the list of known hosts.
From github.com:arita37/mlmodels_store
   cc9c97d..2204d2f  master     -> origin/master
Updating cc9c97d..2204d2f
Fast-forward
 error_list/20200513/list_log_benchmark_20200513.md |  196 +-
 error_list/20200513/list_log_jupyter_20200513.md   | 2259 ++++++++++----------
 error_list/20200513/list_log_test_cli_20200513.md  |  138 +-
 3 files changed, 1317 insertions(+), 1276 deletions(-)
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
[master 48a4983] ml_store
 1 file changed, 72 insertions(+)
To github.com:arita37/mlmodels_store.git
   2204d2f..48a4983  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py 

  #### Loading params   ############################################## 

  {'dataset': 'torchvision.datasets:MNIST', 'transform_uri': 'mlmodels.preprocess.image.py:torch_transform_mnist', '2nd___transform_uri': '/mnt/hgfs/d/gitdev/mlmodels/mlmodels/preprocess/image.py:torch_transform_mnist', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 4, 'test_batch_size': 1} {'checkpointdir': 'ztest/model_tch/torchhub/restnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18/'} 

  #### Loading dataset   ############################################# 

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:30, 318383.53it/s]  2%|▏         | 212992/9912422 [00:00<00:23, 411602.65it/s]  9%|▉         | 876544/9912422 [00:00<00:15, 569532.01it/s] 36%|███▌      | 3522560/9912422 [00:00<00:07, 804338.40it/s] 76%|███████▌  | 7553024/9912422 [00:00<00:02, 1137152.87it/s]9920512it [00:00, 11520169.65it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 143318.15it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 310845.65it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 403790.95it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 559048.41it/s]1654784it [00:00, 2836107.60it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 53862.99it/s]            dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
[master 1d9fac4] ml_store
 1 file changed, 84 insertions(+)
To github.com:arita37/mlmodels_store.git
   48a4983..1d9fac4  master -> master





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
[master 4ebbc74] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   1d9fac4..4ebbc74  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_sentence.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Epoch:   0%|          | 0/1 [00:00<?, ?it/s]
Iteration:   0%|          | 0/29440 [00:00<?, ?it/s][A
Iteration:   0%|          | 1/29440 [00:11<97:42:50, 11.95s/it][A
Iteration:   0%|          | 2/29440 [00:38<133:38:42, 16.34s/it][A
Iteration:   0%|          | 3/29440 [01:40<245:02:32, 29.97s/it][A
Iteration:   0%|          | 4/29440 [02:16<259:10:45, 31.70s/it][A
Iteration:   0%|          | 5/29440 [02:49<263:49:19, 32.27s/it][A
Iteration:   0%|          | 6/29440 [03:44<320:04:00, 39.15s/it][A
Iteration:   0%|          | 7/29440 [04:10<287:47:37, 35.20s/it][A
Iteration:   0%|          | 8/29440 [06:23<526:19:06, 64.38s/it][A
Iteration:   0%|          | 9/29440 [07:17<500:49:31, 61.26s/it][A
Iteration:   0%|          | 10/29440 [13:16<1230:38:51, 150.54s/it][A
Iteration:   0%|          | 11/29440 [14:20<1018:06:34, 124.54s/it][A
Iteration:   0%|          | 12/29440 [14:56<801:58:01, 98.11s/it]  [A
Iteration:   0%|          | 13/29440 [16:40<816:04:53, 99.84s/it][A
Iteration:   0%|          | 14/29440 [17:28<690:35:17, 84.49s/it][A
Iteration:   0%|          | 15/29440 [18:20<609:00:02, 74.51s/it][A
Iteration:   0%|          | 16/29440 [20:15<710:01:38, 86.87s/it][A
Iteration:   0%|          | 17/29440 [21:35<692:19:21, 84.71s/it][A
Iteration:   0%|          | 18/29440 [22:34<628:49:04, 76.94s/it][A
Iteration:   0%|          | 19/29440 [25:13<830:14:46, 101.59s/it][A
Iteration:   0%|          | 20/29440 [29:57<1277:55:51, 156.37s/it][A
Iteration:   0%|          | 21/29440 [31:13<1079:07:29, 132.05s/it][A
Iteration:   0%|          | 22/29440 [32:44<978:27:36, 119.74s/it] [A
Iteration:   0%|          | 23/29440 [33:22<780:07:07, 95.47s/it] [A
Iteration:   0%|          | 24/29440 [34:52<765:57:11, 93.74s/it][A
Iteration:   0%|          | 25/29440 [37:11<877:44:24, 107.42s/it][A
Iteration:   0%|          | 26/29440 [39:51<1004:56:57, 123.00s/it][A
Iteration:   0%|          | 27/29440 [40:49<846:01:42, 103.55s/it] [A
Iteration:   0%|          | 28/29440 [43:17<954:22:26, 116.81s/it][A
Iteration:   0%|          | 29/29440 [44:57<913:06:28, 111.77s/it][A
Iteration:   0%|          | 30/29440 [48:22<1142:06:15, 139.80s/it][A
Iteration:   0%|          | 31/29440 [49:25<954:02:33, 116.79s/it] [A
Iteration:   0%|          | 32/29440 [51:05<913:59:32, 111.89s/it][A
Iteration:   0%|          | 33/29440 [53:33<1002:24:35, 122.71s/it][A
Iteration:   0%|          | 34/29440 [54:21<819:07:12, 100.28s/it] [A
Iteration:   0%|          | 35/29440 [55:36<756:32:04, 92.62s/it] [A
Iteration:   0%|          | 36/29440 [56:09<609:07:23, 74.58s/it][A
Iteration:   0%|          | 37/29440 [57:44<660:12:59, 80.83s/it][A
Iteration:   0%|          | 38/29440 [58:41<600:54:32, 73.58s/it][AKilled

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   4ebbc74..1b15a08  master     -> origin/master
Updating 4ebbc74..1b15a08
Fast-forward
 error_list/20200513/list_log_jupyter_20200513.md   | 2261 +++++++++---------
 ...-13_6672e19fe4cfa7df885e45d91d645534b8989485.py | 2497 ++++++++++++++++++++
 ...-11_6672e19fe4cfa7df885e45d91d645534b8989485.py |  612 +++++
 3 files changed, 4234 insertions(+), 1136 deletions(-)
 create mode 100644 log_benchmark/log_benchmark_2020-05-13-01-13_6672e19fe4cfa7df885e45d91d645534b8989485.py
 create mode 100644 log_pullrequest/log_pr_2020-05-13-01-11_6672e19fe4cfa7df885e45d91d645534b8989485.py
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
[master f695ad3] ml_store
 1 file changed, 88 insertions(+)
To github.com:arita37/mlmodels_store.git
   1b15a08..f695ad3  master -> master





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
[master 65684cb] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   f695ad3..65684cb  master -> master





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


   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   65684cb..d80e053  master     -> origin/master
Updating 65684cb..d80e053
Fast-forward
 error_list/20200513/list_log_jupyter_20200513.md   | 2273 ++++++++++----------
 error_list/20200513/list_log_test_cli_20200513.md  |  152 +-
 error_list/20200513/list_log_testall_20200513.md   |  264 +--
 ...-10_6672e19fe4cfa7df885e45d91d645534b8989485.py |  621 ++++++
 4 files changed, 1994 insertions(+), 1316 deletions(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-13-02-10_6672e19fe4cfa7df885e45d91d645534b8989485.py
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
[master 8aa6bee] ml_store
 1 file changed, 54 insertions(+)
To github.com:arita37/mlmodels_store.git
   d80e053..8aa6bee  master -> master





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
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=5dcfaec8c4e726d1a042f40cb119097b2e37fb765776bebca64ec38fd558620e
  Stored in directory: /tmp/pip-ephem-wheel-cache-w427lcmy/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
[master 24d214a] ml_store
 1 file changed, 104 insertions(+)
To github.com:arita37/mlmodels_store.git
   8aa6bee..24d214a  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//mlp.py 

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
