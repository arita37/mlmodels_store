
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
Warning: Permanently added the RSA host key for IP address '192.30.255.113' to the list of known hosts.
From github.com:arita37/mlmodels_store
   d5a049c..53ea4ab  master     -> origin/master
Updating d5a049c..53ea4ab
Fast-forward
 .../20200513/list_log_dataloader_20200513.md       |   2 +-
 .../20200513/list_log_pullrequest_20200513.md      |   2 +-
 error_list/20200513/list_log_test_cli_20200513.md  | 364 ++++++++++----------
 error_list/20200513/list_log_testall_20200513.md   | 267 ++++++++-------
 ...-08_6672e19fe4cfa7df885e45d91d645534b8989485.py | 373 +++++++++++++++++++++
 5 files changed, 706 insertions(+), 302 deletions(-)
 create mode 100644 log_dataloader/log_2020-05-13-12-08_6672e19fe4cfa7df885e45d91d645534b8989485.py
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
[master 9c8b9b5] ml_store
 1 file changed, 70 insertions(+)
 create mode 100644 log_testall/log_testall_2020-05-13-12-12_6672e19fe4cfa7df885e45d91d645534b8989485.py
To github.com:arita37/mlmodels_store.git
   53ea4ab..9c8b9b5  master -> master





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
[master 26d3e6e] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   9c8b9b5..26d3e6e  master -> master





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
[master 62553f8] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   26d3e6e..62553f8  master -> master





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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         24          sequence_max[0][0]               
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
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         12          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-13 12:12:50.065866: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 12:12:50.071149: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-05-13 12:12:50.071302: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f2cd6dc430 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 12:12:50.071318: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
100/500 [=====>........................] - ETA: 1s - loss: 0.2475 - binary_crossentropy: 0.6862500/500 [==============================] - 1s 1ms/sample - loss: 0.2481 - binary_crossentropy: 0.6877 - val_loss: 0.2531 - val_binary_crossentropy: 0.7521

  #### metrics   #################################################### 
{'MSE': 0.25040656927584665}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         24          sequence_max[0][0]               
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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
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
100/500 [=====>........................] - ETA: 1s - loss: 0.2616 - binary_crossentropy: 1.3586500/500 [==============================] - 1s 1ms/sample - loss: 0.2979 - binary_crossentropy: 1.9647 - val_loss: 0.3029 - val_binary_crossentropy: 2.1301

  #### metrics   #################################################### 
{'MSE': 0.3003145888890926}

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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         24          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_2[0][0]           
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
Total params: 632
Trainable params: 632
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 2ms/sample - loss: 0.2497 - binary_crossentropy: 0.6926 - val_loss: 0.2498 - val_binary_crossentropy: 0.6927

  #### metrics   #################################################### 
{'MSE': 0.24950791228347252}

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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         24          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_2[0][0]           
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
Total params: 632
Trainable params: 632
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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         28          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_2[0][0]           
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
100/500 [=====>........................] - ETA: 2s - loss: 0.2745 - binary_crossentropy: 0.7507500/500 [==============================] - 1s 3ms/sample - loss: 0.2921 - binary_crossentropy: 0.7893 - val_loss: 0.2631 - val_binary_crossentropy: 0.7212

  #### metrics   #################################################### 
{'MSE': 0.2703050929550314}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         28          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_2[0][0]           
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
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
100/500 [=====>........................] - ETA: 2s - loss: 0.5000 - binary_crossentropy: 7.7125500/500 [==============================] - 2s 3ms/sample - loss: 0.5360 - binary_crossentropy: 8.2678 - val_loss: 0.5080 - val_binary_crossentropy: 7.8359

  #### metrics   #################################################### 
{'MSE': 0.522}

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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-13 12:14:04.626572: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 12:14:04.628624: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 12:14:04.634328: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-13 12:14:04.643956: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-13 12:14:04.645749: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 12:14:04.647260: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 12:14:04.648685: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2469 - val_binary_crossentropy: 0.6870
2020-05-13 12:14:05.856972: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 12:14:05.858658: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 12:14:05.862623: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-13 12:14:05.870748: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-13 12:14:05.872168: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 12:14:05.873448: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 12:14:05.874646: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24564689771595263}

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
2020-05-13 12:14:27.776131: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 12:14:27.777500: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 12:14:27.780993: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-13 12:14:27.787201: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-13 12:14:27.788264: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 12:14:27.789265: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 12:14:27.790199: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2483 - val_binary_crossentropy: 0.6898
2020-05-13 12:14:29.252793: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 12:14:29.253978: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 12:14:29.256779: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-13 12:14:29.261771: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-13 12:14:29.262632: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 12:14:29.263422: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 12:14:29.264160: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24761973828552541}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-13 12:15:01.779663: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 12:15:01.784206: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 12:15:01.797376: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-13 12:15:01.821234: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-13 12:15:01.825347: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 12:15:01.828942: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 12:15:01.832597: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.5997 - binary_crossentropy: 1.4890 - val_loss: 0.3133 - val_binary_crossentropy: 0.8392
2020-05-13 12:15:03.970671: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 12:15:03.975126: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 12:15:03.985942: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-13 12:15:04.008665: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-13 12:15:04.012459: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 12:15:04.016046: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 12:15:04.019536: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.39708481324331507}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
Total params: 660
Trainable params: 660
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2583 - binary_crossentropy: 0.7099500/500 [==============================] - 4s 8ms/sample - loss: 0.2590 - binary_crossentropy: 0.7115 - val_loss: 0.2498 - val_binary_crossentropy: 0.6927

  #### metrics   #################################################### 
{'MSE': 0.2539779462162896}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
Total params: 660
Trainable params: 660
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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 2)         18          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 2)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 2)         10          sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         10          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         6           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         10          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         6           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         2           sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_2[0][0]           
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
Total params: 227
Trainable params: 227
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2920 - binary_crossentropy: 0.7928500/500 [==============================] - 4s 9ms/sample - loss: 0.2815 - binary_crossentropy: 0.7646 - val_loss: 0.2661 - val_binary_crossentropy: 0.7290

  #### metrics   #################################################### 
{'MSE': 0.26562923659542703}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 2)         18          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 2)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 2)         10          sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         10          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         6           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         10          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         6           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         2           sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_2[0][0]           
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
Total params: 227
Trainable params: 227
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
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_max[0][0]               
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
Total params: 1,904
Trainable params: 1,904
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 5s - loss: 0.2533 - binary_crossentropy: 0.6999500/500 [==============================] - 4s 9ms/sample - loss: 0.2538 - binary_crossentropy: 0.7009 - val_loss: 0.2486 - val_binary_crossentropy: 0.6904

  #### metrics   #################################################### 
{'MSE': 0.24932608307135365}

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
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_max[0][0]               
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
regionsequence_sum (InputLayer) [(None, 1)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 9)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 1)]          0                                            
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
region_10sparse_seq_emb_regions (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 9, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         1           regionsequence_max[0][0]         
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
Total params: 144
Trainable params: 144
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.4601 - binary_crossentropy: 7.0955500/500 [==============================] - 5s 11ms/sample - loss: 0.4921 - binary_crossentropy: 7.5891 - val_loss: 0.5221 - val_binary_crossentropy: 8.0518

  #### metrics   #################################################### 
{'MSE': 0.507}

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
regionsequence_sum (InputLayer) [(None, 1)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 9)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 1)]          0                                            
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
region_10sparse_seq_emb_regions (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 9, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         1           regionsequence_max[0][0]         
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
Total params: 144
Trainable params: 144
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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
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
Total params: 1,417
Trainable params: 1,417
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2733 - binary_crossentropy: 0.7467500/500 [==============================] - 5s 11ms/sample - loss: 0.2833 - binary_crossentropy: 0.8738 - val_loss: 0.2753 - val_binary_crossentropy: 0.7735

  #### metrics   #################################################### 
{'MSE': 0.27674418523940036}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
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
Total params: 1,417
Trainable params: 1,417
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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         36          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         24          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 9, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 9, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 9, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 9, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           hash_10[0][0]                    
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
Total params: 3,103
Trainable params: 3,023
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2807 - binary_crossentropy: 0.7614500/500 [==============================] - 6s 12ms/sample - loss: 0.2853 - binary_crossentropy: 0.7724 - val_loss: 0.2758 - val_binary_crossentropy: 0.7494

  #### metrics   #################################################### 
{'MSE': 0.27836747842154197}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         36          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         24          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 9, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 9, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 9, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 9, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           hash_10[0][0]                    
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
Total params: 405
Trainable params: 405
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2574 - binary_crossentropy: 0.7093500/500 [==============================] - 6s 12ms/sample - loss: 0.2519 - binary_crossentropy: 0.6975 - val_loss: 0.2496 - val_binary_crossentropy: 0.6923

  #### metrics   #################################################### 
{'MSE': 0.24989567888937864}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
Total params: 405
Trainable params: 405
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_44 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_125 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_194[0][0] 
                                                                 sequence_pooling_layer_195[0][0] 
                                                                 sequence_pooling_layer_196[0][0] 
                                                                 sequence_pooling_layer_197[0][0] 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
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
Total params: 2,049
Trainable params: 2,049
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 6s 12ms/sample - loss: 0.2501 - binary_crossentropy: 0.6933 - val_loss: 0.2500 - val_binary_crossentropy: 0.6932

  #### metrics   #################################################### 
{'MSE': 0.2497988226839197}

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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_44 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_125 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_194[0][0] 
                                                                 sequence_pooling_layer_195[0][0] 
                                                                 sequence_pooling_layer_196[0][0] 
                                                                 sequence_pooling_layer_197[0][0] 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
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
Total params: 2,049
Trainable params: 2,049
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_47 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_90 (Concatenate)    (None, 1, 20)        0           no_mask_130[0][0]                
                                                                 no_mask_130[1][0]                
                                                                 no_mask_130[2][0]                
                                                                 no_mask_130[3][0]                
                                                                 no_mask_130[4][0]                
__________________________________________________________________________________________________
no_mask_131 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
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
100/500 [=====>........................] - ETA: 8s - loss: 0.3158 - binary_crossentropy: 0.9819500/500 [==============================] - 7s 13ms/sample - loss: 0.3072 - binary_crossentropy: 0.8910 - val_loss: 0.3322 - val_binary_crossentropy: 1.0832

  #### metrics   #################################################### 
{'MSE': 0.3180544594489469}

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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_47 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_90 (Concatenate)    (None, 1, 20)        0           no_mask_130[0][0]                
                                                                 no_mask_130[1][0]                
                                                                 no_mask_130[2][0]                
                                                                 no_mask_130[3][0]                
                                                                 no_mask_130[4][0]                
__________________________________________________________________________________________________
no_mask_131 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
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
Warning: Permanently added the RSA host key for IP address '192.30.255.112' to the list of known hosts.
From github.com:arita37/mlmodels_store
   62553f8..b10bfaa  master     -> origin/master
Updating 62553f8..b10bfaa
Fast-forward
 error_list/20200513/list_log_jupyter_20200513.md   | 2360 ++++++++++----------
 .../20200513/list_log_pullrequest_20200513.md      |    2 +-
 error_list/20200513/list_log_test_cli_20200513.md  |  364 +--
 ...-10_6672e19fe4cfa7df885e45d91d645534b8989485.py |  611 +++++
 4 files changed, 1978 insertions(+), 1359 deletions(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-13-12-10_6672e19fe4cfa7df885e45d91d645534b8989485.py
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
[master 4a83be2] ml_store
 1 file changed, 5671 insertions(+)
To github.com:arita37/mlmodels_store.git
   b10bfaa..4a83be2  master -> master





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
[master b97cf18] ml_store
 1 file changed, 50 insertions(+)
To github.com:arita37/mlmodels_store.git
   4a83be2..b97cf18  master -> master





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
[master 78839c7] ml_store
 1 file changed, 46 insertions(+)
To github.com:arita37/mlmodels_store.git
   b97cf18..78839c7  master -> master





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
[master 52b9a18] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   78839c7..52b9a18  master -> master





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

2020-05-13 12:27:25.396938: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 12:27:25.401748: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-05-13 12:27:25.401903: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56060ea926f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 12:27:25.401920: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 7s - loss: 1.3865
256/354 [====================>.........] - ETA: 3s - loss: 1.1813
354/354 [==============================] - 13s 37ms/step - loss: 1.3069 - val_loss: 2.3894

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
Warning: Permanently added the RSA host key for IP address '13.237.44.5' to the list of known hosts.
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
[master cff4ecb] ml_store
 1 file changed, 150 insertions(+)
To github.com:arita37/mlmodels_store.git
   52b9a18..cff4ecb  master -> master





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
[master 8f9347c] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   cff4ecb..8f9347c  master -> master





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
[master 350c602] ml_store
 1 file changed, 44 insertions(+)
To github.com:arita37/mlmodels_store.git
   8f9347c..350c602  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 50s
   49152/17464789 [..............................] - ETA: 50s
   98304/17464789 [..............................] - ETA: 37s
  139264/17464789 [..............................] - ETA: 35s
  204800/17464789 [..............................] - ETA: 29s
  376832/17464789 [..............................] - ETA: 19s
  622592/17464789 [>.............................] - ETA: 13s
 1179648/17464789 [=>............................] - ETA: 7s 
 2293760/17464789 [==>...........................] - ETA: 4s
 4505600/17464789 [======>.......................] - ETA: 2s
 7487488/17464789 [===========>..................] - ETA: 1s
10223616/17464789 [================>.............] - ETA: 0s
13352960/17464789 [=====================>........] - ETA: 0s
16433152/17464789 [===========================>..] - ETA: 0s
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
2020-05-13 12:28:46.629550: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 12:28:46.633929: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-05-13 12:28:46.634657: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f9f121d600 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 12:28:46.634675: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 11s - loss: 7.9580 - accuracy: 0.4810
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6206 - accuracy: 0.5030 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.5491 - accuracy: 0.5077
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6130 - accuracy: 0.5035
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5746 - accuracy: 0.5060
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5900 - accuracy: 0.5050
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5987 - accuracy: 0.5044
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5689 - accuracy: 0.5064
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6002 - accuracy: 0.5043
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6038 - accuracy: 0.5041
11000/25000 [============>.................] - ETA: 4s - loss: 7.5900 - accuracy: 0.5050
12000/25000 [=============>................] - ETA: 3s - loss: 7.5925 - accuracy: 0.5048
13000/25000 [==============>...............] - ETA: 3s - loss: 7.5805 - accuracy: 0.5056
14000/25000 [===============>..............] - ETA: 3s - loss: 7.5921 - accuracy: 0.5049
15000/25000 [=================>............] - ETA: 2s - loss: 7.6155 - accuracy: 0.5033
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6120 - accuracy: 0.5036
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6405 - accuracy: 0.5017
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6462 - accuracy: 0.5013
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6432 - accuracy: 0.5015
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6590 - accuracy: 0.5005
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6469 - accuracy: 0.5013
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6485 - accuracy: 0.5012
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6622 - accuracy: 0.5003
25000/25000 [==============================] - 8s 338us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7fc3286ca9e8>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7fc31f0f24a8> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 8.0193 - accuracy: 0.4770
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8046 - accuracy: 0.4910 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.8251 - accuracy: 0.4897
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7625 - accuracy: 0.4938
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7801 - accuracy: 0.4926
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7688 - accuracy: 0.4933
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7280 - accuracy: 0.4960
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7069 - accuracy: 0.4974
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7194 - accuracy: 0.4966
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7188 - accuracy: 0.4966
11000/25000 [============>.................] - ETA: 4s - loss: 7.6875 - accuracy: 0.4986
12000/25000 [=============>................] - ETA: 3s - loss: 7.6909 - accuracy: 0.4984
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6560 - accuracy: 0.5007
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6601 - accuracy: 0.5004
15000/25000 [=================>............] - ETA: 2s - loss: 7.6656 - accuracy: 0.5001
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6676 - accuracy: 0.4999
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6657 - accuracy: 0.5001
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6556 - accuracy: 0.5007
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6505 - accuracy: 0.5011
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6636 - accuracy: 0.5002
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6557 - accuracy: 0.5007
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6394 - accuracy: 0.5018
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6566 - accuracy: 0.5007
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 8s 338us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 11s - loss: 7.3140 - accuracy: 0.5230
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7280 - accuracy: 0.4960 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7126 - accuracy: 0.4970
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6973 - accuracy: 0.4980
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6912 - accuracy: 0.4984
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6717 - accuracy: 0.4997
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6535 - accuracy: 0.5009
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6206 - accuracy: 0.5030
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6445 - accuracy: 0.5014
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6406 - accuracy: 0.5017
11000/25000 [============>.................] - ETA: 3s - loss: 7.6248 - accuracy: 0.5027
12000/25000 [=============>................] - ETA: 3s - loss: 7.5963 - accuracy: 0.5046
13000/25000 [==============>...............] - ETA: 3s - loss: 7.5793 - accuracy: 0.5057
14000/25000 [===============>..............] - ETA: 3s - loss: 7.5790 - accuracy: 0.5057
15000/25000 [=================>............] - ETA: 2s - loss: 7.6043 - accuracy: 0.5041
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6206 - accuracy: 0.5030
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6269 - accuracy: 0.5026
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6308 - accuracy: 0.5023
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6222 - accuracy: 0.5029
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6597 - accuracy: 0.5005
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6695 - accuracy: 0.4998
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6792 - accuracy: 0.4992
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6766 - accuracy: 0.4993
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6864 - accuracy: 0.4987
25000/25000 [==============================] - 8s 336us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
   350c602..c69f43d  master     -> origin/master
Updating 350c602..c69f43d
Fast-forward
 .../20200513/list_log_dataloader_20200513.md       |    2 +-
 error_list/20200513/list_log_jupyter_20200513.md   | 1668 ++++++++++----------
 error_list/20200513/list_log_test_cli_20200513.md  |  152 +-
 3 files changed, 911 insertions(+), 911 deletions(-)
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
[master c382f84] ml_store
 1 file changed, 335 insertions(+)
To github.com:arita37/mlmodels_store.git
   c69f43d..c382f84  master -> master





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

13/13 [==============================] - 1s 112ms/step - loss: nan
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

13/13 [==============================] - 0s 3ms/step - loss: nan
Epoch 10/10

13/13 [==============================] - 0s 3ms/step - loss: nan

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
[master 2a547de] ml_store
 1 file changed, 125 insertions(+)
To github.com:arita37/mlmodels_store.git
   c382f84..2a547de  master -> master





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
   24576/11490434 [..............................] - ETA: 31s
   57344/11490434 [..............................] - ETA: 27s
   90112/11490434 [..............................] - ETA: 25s
  196608/11490434 [..............................] - ETA: 15s
  401408/11490434 [>.............................] - ETA: 9s 
  819200/11490434 [=>............................] - ETA: 5s
 1638400/11490434 [===>..........................] - ETA: 2s
 3244032/11490434 [=======>......................] - ETA: 1s
 6225920/11490434 [===============>..............] - ETA: 0s
 9109504/11490434 [======================>.......] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:02 - loss: 2.3104 - categorical_accuracy: 0.0625
   64/60000 [..............................] - ETA: 4:21 - loss: 2.2845 - categorical_accuracy: 0.1406
   96/60000 [..............................] - ETA: 3:29 - loss: 2.2724 - categorical_accuracy: 0.1771
  128/60000 [..............................] - ETA: 3:01 - loss: 2.2393 - categorical_accuracy: 0.1875
  160/60000 [..............................] - ETA: 2:44 - loss: 2.2646 - categorical_accuracy: 0.1688
  192/60000 [..............................] - ETA: 2:32 - loss: 2.2338 - categorical_accuracy: 0.1875
  224/60000 [..............................] - ETA: 2:24 - loss: 2.1977 - categorical_accuracy: 0.2366
  288/60000 [..............................] - ETA: 2:12 - loss: 2.1405 - categorical_accuracy: 0.2535
  320/60000 [..............................] - ETA: 2:08 - loss: 2.1109 - categorical_accuracy: 0.2688
  384/60000 [..............................] - ETA: 2:02 - loss: 2.0528 - categorical_accuracy: 0.2865
  416/60000 [..............................] - ETA: 2:00 - loss: 2.0375 - categorical_accuracy: 0.2933
  448/60000 [..............................] - ETA: 1:58 - loss: 1.9966 - categorical_accuracy: 0.3103
  480/60000 [..............................] - ETA: 1:57 - loss: 1.9788 - categorical_accuracy: 0.3104
  544/60000 [..............................] - ETA: 1:53 - loss: 1.9485 - categorical_accuracy: 0.3327
  576/60000 [..............................] - ETA: 1:52 - loss: 1.9318 - categorical_accuracy: 0.3420
  640/60000 [..............................] - ETA: 1:50 - loss: 1.8721 - categorical_accuracy: 0.3641
  704/60000 [..............................] - ETA: 1:48 - loss: 1.7972 - categorical_accuracy: 0.3949
  736/60000 [..............................] - ETA: 1:47 - loss: 1.7545 - categorical_accuracy: 0.4144
  768/60000 [..............................] - ETA: 1:47 - loss: 1.7165 - categorical_accuracy: 0.4310
  832/60000 [..............................] - ETA: 1:46 - loss: 1.6456 - categorical_accuracy: 0.4555
  896/60000 [..............................] - ETA: 1:45 - loss: 1.6458 - categorical_accuracy: 0.4587
  960/60000 [..............................] - ETA: 1:43 - loss: 1.5961 - categorical_accuracy: 0.4760
  992/60000 [..............................] - ETA: 1:43 - loss: 1.5623 - categorical_accuracy: 0.4869
 1024/60000 [..............................] - ETA: 1:43 - loss: 1.5399 - categorical_accuracy: 0.4932
 1056/60000 [..............................] - ETA: 1:43 - loss: 1.5266 - categorical_accuracy: 0.4972
 1120/60000 [..............................] - ETA: 1:42 - loss: 1.5013 - categorical_accuracy: 0.5036
 1184/60000 [..............................] - ETA: 1:42 - loss: 1.4563 - categorical_accuracy: 0.5203
 1248/60000 [..............................] - ETA: 1:41 - loss: 1.4172 - categorical_accuracy: 0.5329
 1312/60000 [..............................] - ETA: 1:40 - loss: 1.3744 - categorical_accuracy: 0.5465
 1344/60000 [..............................] - ETA: 1:40 - loss: 1.3539 - categorical_accuracy: 0.5528
 1408/60000 [..............................] - ETA: 1:39 - loss: 1.3235 - categorical_accuracy: 0.5639
 1440/60000 [..............................] - ETA: 1:39 - loss: 1.3089 - categorical_accuracy: 0.5688
 1472/60000 [..............................] - ETA: 1:39 - loss: 1.2919 - categorical_accuracy: 0.5754
 1536/60000 [..............................] - ETA: 1:38 - loss: 1.2714 - categorical_accuracy: 0.5820
 1600/60000 [..............................] - ETA: 1:38 - loss: 1.2496 - categorical_accuracy: 0.5900
 1664/60000 [..............................] - ETA: 1:38 - loss: 1.2324 - categorical_accuracy: 0.5986
 1696/60000 [..............................] - ETA: 1:38 - loss: 1.2184 - categorical_accuracy: 0.6038
 1760/60000 [..............................] - ETA: 1:37 - loss: 1.1999 - categorical_accuracy: 0.6119
 1792/60000 [..............................] - ETA: 1:37 - loss: 1.1873 - categorical_accuracy: 0.6166
 1856/60000 [..............................] - ETA: 1:37 - loss: 1.1680 - categorical_accuracy: 0.6223
 1920/60000 [..............................] - ETA: 1:36 - loss: 1.1458 - categorical_accuracy: 0.6297
 1984/60000 [..............................] - ETA: 1:36 - loss: 1.1235 - categorical_accuracy: 0.6361
 2016/60000 [>.............................] - ETA: 1:36 - loss: 1.1157 - categorical_accuracy: 0.6379
 2048/60000 [>.............................] - ETA: 1:36 - loss: 1.1096 - categorical_accuracy: 0.6401
 2080/60000 [>.............................] - ETA: 1:36 - loss: 1.1009 - categorical_accuracy: 0.6428
 2112/60000 [>.............................] - ETA: 1:36 - loss: 1.0945 - categorical_accuracy: 0.6444
 2144/60000 [>.............................] - ETA: 1:36 - loss: 1.0853 - categorical_accuracy: 0.6474
 2176/60000 [>.............................] - ETA: 1:36 - loss: 1.0760 - categorical_accuracy: 0.6498
 2208/60000 [>.............................] - ETA: 1:36 - loss: 1.0699 - categorical_accuracy: 0.6522
 2272/60000 [>.............................] - ETA: 1:35 - loss: 1.0532 - categorical_accuracy: 0.6576
 2304/60000 [>.............................] - ETA: 1:35 - loss: 1.0487 - categorical_accuracy: 0.6584
 2336/60000 [>.............................] - ETA: 1:35 - loss: 1.0415 - categorical_accuracy: 0.6618
 2400/60000 [>.............................] - ETA: 1:35 - loss: 1.0254 - categorical_accuracy: 0.6675
 2464/60000 [>.............................] - ETA: 1:35 - loss: 1.0140 - categorical_accuracy: 0.6721
 2528/60000 [>.............................] - ETA: 1:34 - loss: 1.0005 - categorical_accuracy: 0.6760
 2592/60000 [>.............................] - ETA: 1:34 - loss: 0.9879 - categorical_accuracy: 0.6809
 2656/60000 [>.............................] - ETA: 1:34 - loss: 0.9702 - categorical_accuracy: 0.6871
 2688/60000 [>.............................] - ETA: 1:34 - loss: 0.9643 - categorical_accuracy: 0.6894
 2720/60000 [>.............................] - ETA: 1:34 - loss: 0.9575 - categorical_accuracy: 0.6915
 2752/60000 [>.............................] - ETA: 1:34 - loss: 0.9553 - categorical_accuracy: 0.6922
 2784/60000 [>.............................] - ETA: 1:34 - loss: 0.9507 - categorical_accuracy: 0.6936
 2848/60000 [>.............................] - ETA: 1:34 - loss: 0.9403 - categorical_accuracy: 0.6959
 2912/60000 [>.............................] - ETA: 1:33 - loss: 0.9274 - categorical_accuracy: 0.7009
 2944/60000 [>.............................] - ETA: 1:33 - loss: 0.9199 - categorical_accuracy: 0.7035
 3008/60000 [>.............................] - ETA: 1:33 - loss: 0.9104 - categorical_accuracy: 0.7064
 3040/60000 [>.............................] - ETA: 1:33 - loss: 0.9044 - categorical_accuracy: 0.7089
 3072/60000 [>.............................] - ETA: 1:33 - loss: 0.9005 - categorical_accuracy: 0.7103
 3104/60000 [>.............................] - ETA: 1:33 - loss: 0.8949 - categorical_accuracy: 0.7120
 3136/60000 [>.............................] - ETA: 1:33 - loss: 0.8903 - categorical_accuracy: 0.7133
 3168/60000 [>.............................] - ETA: 1:33 - loss: 0.8842 - categorical_accuracy: 0.7153
 3200/60000 [>.............................] - ETA: 1:33 - loss: 0.8779 - categorical_accuracy: 0.7178
 3232/60000 [>.............................] - ETA: 1:33 - loss: 0.8721 - categorical_accuracy: 0.7197
 3264/60000 [>.............................] - ETA: 1:33 - loss: 0.8687 - categorical_accuracy: 0.7203
 3296/60000 [>.............................] - ETA: 1:33 - loss: 0.8627 - categorical_accuracy: 0.7221
 3360/60000 [>.............................] - ETA: 1:32 - loss: 0.8547 - categorical_accuracy: 0.7250
 3392/60000 [>.............................] - ETA: 1:32 - loss: 0.8494 - categorical_accuracy: 0.7273
 3424/60000 [>.............................] - ETA: 1:32 - loss: 0.8473 - categorical_accuracy: 0.7287
 3456/60000 [>.............................] - ETA: 1:32 - loss: 0.8439 - categorical_accuracy: 0.7303
 3488/60000 [>.............................] - ETA: 1:32 - loss: 0.8401 - categorical_accuracy: 0.7314
 3520/60000 [>.............................] - ETA: 1:32 - loss: 0.8348 - categorical_accuracy: 0.7330
 3552/60000 [>.............................] - ETA: 1:32 - loss: 0.8323 - categorical_accuracy: 0.7334
 3584/60000 [>.............................] - ETA: 1:32 - loss: 0.8273 - categorical_accuracy: 0.7349
 3648/60000 [>.............................] - ETA: 1:32 - loss: 0.8197 - categorical_accuracy: 0.7371
 3712/60000 [>.............................] - ETA: 1:32 - loss: 0.8141 - categorical_accuracy: 0.7395
 3776/60000 [>.............................] - ETA: 1:31 - loss: 0.8057 - categorical_accuracy: 0.7423
 3840/60000 [>.............................] - ETA: 1:31 - loss: 0.7974 - categorical_accuracy: 0.7451
 3904/60000 [>.............................] - ETA: 1:31 - loss: 0.7888 - categorical_accuracy: 0.7474
 3936/60000 [>.............................] - ETA: 1:31 - loss: 0.7841 - categorical_accuracy: 0.7487
 4000/60000 [=>............................] - ETA: 1:31 - loss: 0.7780 - categorical_accuracy: 0.7508
 4064/60000 [=>............................] - ETA: 1:31 - loss: 0.7756 - categorical_accuracy: 0.7527
 4128/60000 [=>............................] - ETA: 1:30 - loss: 0.7685 - categorical_accuracy: 0.7553
 4160/60000 [=>............................] - ETA: 1:30 - loss: 0.7657 - categorical_accuracy: 0.7563
 4192/60000 [=>............................] - ETA: 1:30 - loss: 0.7634 - categorical_accuracy: 0.7567
 4224/60000 [=>............................] - ETA: 1:30 - loss: 0.7603 - categorical_accuracy: 0.7576
 4256/60000 [=>............................] - ETA: 1:30 - loss: 0.7559 - categorical_accuracy: 0.7589
 4320/60000 [=>............................] - ETA: 1:30 - loss: 0.7494 - categorical_accuracy: 0.7611
 4384/60000 [=>............................] - ETA: 1:30 - loss: 0.7449 - categorical_accuracy: 0.7628
 4448/60000 [=>............................] - ETA: 1:30 - loss: 0.7391 - categorical_accuracy: 0.7646
 4480/60000 [=>............................] - ETA: 1:30 - loss: 0.7362 - categorical_accuracy: 0.7654
 4544/60000 [=>............................] - ETA: 1:30 - loss: 0.7288 - categorical_accuracy: 0.7680
 4608/60000 [=>............................] - ETA: 1:29 - loss: 0.7219 - categorical_accuracy: 0.7700
 4672/60000 [=>............................] - ETA: 1:29 - loss: 0.7156 - categorical_accuracy: 0.7720
 4704/60000 [=>............................] - ETA: 1:29 - loss: 0.7154 - categorical_accuracy: 0.7727
 4736/60000 [=>............................] - ETA: 1:29 - loss: 0.7128 - categorical_accuracy: 0.7736
 4800/60000 [=>............................] - ETA: 1:29 - loss: 0.7077 - categorical_accuracy: 0.7752
 4832/60000 [=>............................] - ETA: 1:29 - loss: 0.7047 - categorical_accuracy: 0.7765
 4896/60000 [=>............................] - ETA: 1:29 - loss: 0.6980 - categorical_accuracy: 0.7790
 4960/60000 [=>............................] - ETA: 1:29 - loss: 0.6927 - categorical_accuracy: 0.7802
 5024/60000 [=>............................] - ETA: 1:28 - loss: 0.6871 - categorical_accuracy: 0.7818
 5088/60000 [=>............................] - ETA: 1:28 - loss: 0.6823 - categorical_accuracy: 0.7836
 5152/60000 [=>............................] - ETA: 1:28 - loss: 0.6794 - categorical_accuracy: 0.7851
 5216/60000 [=>............................] - ETA: 1:28 - loss: 0.6754 - categorical_accuracy: 0.7868
 5248/60000 [=>............................] - ETA: 1:28 - loss: 0.6724 - categorical_accuracy: 0.7877
 5280/60000 [=>............................] - ETA: 1:28 - loss: 0.6708 - categorical_accuracy: 0.7879
 5312/60000 [=>............................] - ETA: 1:28 - loss: 0.6698 - categorical_accuracy: 0.7882
 5344/60000 [=>............................] - ETA: 1:28 - loss: 0.6668 - categorical_accuracy: 0.7891
 5408/60000 [=>............................] - ETA: 1:28 - loss: 0.6628 - categorical_accuracy: 0.7903
 5472/60000 [=>............................] - ETA: 1:27 - loss: 0.6579 - categorical_accuracy: 0.7922
 5504/60000 [=>............................] - ETA: 1:27 - loss: 0.6567 - categorical_accuracy: 0.7927
 5536/60000 [=>............................] - ETA: 1:27 - loss: 0.6542 - categorical_accuracy: 0.7937
 5568/60000 [=>............................] - ETA: 1:27 - loss: 0.6525 - categorical_accuracy: 0.7940
 5632/60000 [=>............................] - ETA: 1:27 - loss: 0.6473 - categorical_accuracy: 0.7955
 5696/60000 [=>............................] - ETA: 1:27 - loss: 0.6440 - categorical_accuracy: 0.7967
 5760/60000 [=>............................] - ETA: 1:27 - loss: 0.6397 - categorical_accuracy: 0.7981
 5824/60000 [=>............................] - ETA: 1:27 - loss: 0.6361 - categorical_accuracy: 0.7995
 5888/60000 [=>............................] - ETA: 1:27 - loss: 0.6318 - categorical_accuracy: 0.8006
 5952/60000 [=>............................] - ETA: 1:26 - loss: 0.6273 - categorical_accuracy: 0.8024
 6016/60000 [==>...........................] - ETA: 1:26 - loss: 0.6241 - categorical_accuracy: 0.8032
 6048/60000 [==>...........................] - ETA: 1:26 - loss: 0.6220 - categorical_accuracy: 0.8041
 6112/60000 [==>...........................] - ETA: 1:26 - loss: 0.6177 - categorical_accuracy: 0.8051
 6176/60000 [==>...........................] - ETA: 1:26 - loss: 0.6142 - categorical_accuracy: 0.8067
 6208/60000 [==>...........................] - ETA: 1:26 - loss: 0.6128 - categorical_accuracy: 0.8072
 6272/60000 [==>...........................] - ETA: 1:26 - loss: 0.6086 - categorical_accuracy: 0.8087
 6336/60000 [==>...........................] - ETA: 1:26 - loss: 0.6066 - categorical_accuracy: 0.8089
 6368/60000 [==>...........................] - ETA: 1:26 - loss: 0.6045 - categorical_accuracy: 0.8097
 6400/60000 [==>...........................] - ETA: 1:26 - loss: 0.6040 - categorical_accuracy: 0.8097
 6432/60000 [==>...........................] - ETA: 1:25 - loss: 0.6031 - categorical_accuracy: 0.8102
 6464/60000 [==>...........................] - ETA: 1:25 - loss: 0.6016 - categorical_accuracy: 0.8108
 6496/60000 [==>...........................] - ETA: 1:25 - loss: 0.6000 - categorical_accuracy: 0.8111
 6528/60000 [==>...........................] - ETA: 1:25 - loss: 0.5989 - categorical_accuracy: 0.8111
 6560/60000 [==>...........................] - ETA: 1:25 - loss: 0.5974 - categorical_accuracy: 0.8116
 6592/60000 [==>...........................] - ETA: 1:25 - loss: 0.5952 - categorical_accuracy: 0.8122
 6624/60000 [==>...........................] - ETA: 1:25 - loss: 0.5934 - categorical_accuracy: 0.8127
 6656/60000 [==>...........................] - ETA: 1:25 - loss: 0.5917 - categorical_accuracy: 0.8134
 6688/60000 [==>...........................] - ETA: 1:25 - loss: 0.5903 - categorical_accuracy: 0.8137
 6720/60000 [==>...........................] - ETA: 1:25 - loss: 0.5892 - categorical_accuracy: 0.8140
 6752/60000 [==>...........................] - ETA: 1:25 - loss: 0.5875 - categorical_accuracy: 0.8144
 6784/60000 [==>...........................] - ETA: 1:25 - loss: 0.5861 - categorical_accuracy: 0.8149
 6816/60000 [==>...........................] - ETA: 1:25 - loss: 0.5843 - categorical_accuracy: 0.8154
 6848/60000 [==>...........................] - ETA: 1:25 - loss: 0.5827 - categorical_accuracy: 0.8162
 6880/60000 [==>...........................] - ETA: 1:25 - loss: 0.5825 - categorical_accuracy: 0.8164
 6944/60000 [==>...........................] - ETA: 1:25 - loss: 0.5809 - categorical_accuracy: 0.8170
 6976/60000 [==>...........................] - ETA: 1:25 - loss: 0.5792 - categorical_accuracy: 0.8175
 7040/60000 [==>...........................] - ETA: 1:25 - loss: 0.5750 - categorical_accuracy: 0.8188
 7104/60000 [==>...........................] - ETA: 1:24 - loss: 0.5725 - categorical_accuracy: 0.8195
 7136/60000 [==>...........................] - ETA: 1:24 - loss: 0.5703 - categorical_accuracy: 0.8203
 7168/60000 [==>...........................] - ETA: 1:24 - loss: 0.5691 - categorical_accuracy: 0.8209
 7232/60000 [==>...........................] - ETA: 1:24 - loss: 0.5669 - categorical_accuracy: 0.8215
 7264/60000 [==>...........................] - ETA: 1:24 - loss: 0.5652 - categorical_accuracy: 0.8220
 7296/60000 [==>...........................] - ETA: 1:24 - loss: 0.5641 - categorical_accuracy: 0.8222
 7328/60000 [==>...........................] - ETA: 1:24 - loss: 0.5631 - categorical_accuracy: 0.8226
 7360/60000 [==>...........................] - ETA: 1:24 - loss: 0.5630 - categorical_accuracy: 0.8227
 7392/60000 [==>...........................] - ETA: 1:24 - loss: 0.5631 - categorical_accuracy: 0.8225
 7456/60000 [==>...........................] - ETA: 1:24 - loss: 0.5603 - categorical_accuracy: 0.8232
 7520/60000 [==>...........................] - ETA: 1:24 - loss: 0.5573 - categorical_accuracy: 0.8241
 7584/60000 [==>...........................] - ETA: 1:24 - loss: 0.5554 - categorical_accuracy: 0.8246
 7648/60000 [==>...........................] - ETA: 1:23 - loss: 0.5529 - categorical_accuracy: 0.8253
 7680/60000 [==>...........................] - ETA: 1:23 - loss: 0.5515 - categorical_accuracy: 0.8259
 7744/60000 [==>...........................] - ETA: 1:23 - loss: 0.5476 - categorical_accuracy: 0.8272
 7808/60000 [==>...........................] - ETA: 1:23 - loss: 0.5453 - categorical_accuracy: 0.8281
 7872/60000 [==>...........................] - ETA: 1:23 - loss: 0.5430 - categorical_accuracy: 0.8290
 7936/60000 [==>...........................] - ETA: 1:23 - loss: 0.5418 - categorical_accuracy: 0.8296
 8000/60000 [===>..........................] - ETA: 1:23 - loss: 0.5388 - categorical_accuracy: 0.8305
 8032/60000 [===>..........................] - ETA: 1:23 - loss: 0.5372 - categorical_accuracy: 0.8311
 8096/60000 [===>..........................] - ETA: 1:23 - loss: 0.5339 - categorical_accuracy: 0.8321
 8128/60000 [===>..........................] - ETA: 1:23 - loss: 0.5324 - categorical_accuracy: 0.8327
 8192/60000 [===>..........................] - ETA: 1:22 - loss: 0.5301 - categorical_accuracy: 0.8334
 8256/60000 [===>..........................] - ETA: 1:22 - loss: 0.5288 - categorical_accuracy: 0.8338
 8320/60000 [===>..........................] - ETA: 1:22 - loss: 0.5265 - categorical_accuracy: 0.8345
 8352/60000 [===>..........................] - ETA: 1:22 - loss: 0.5259 - categorical_accuracy: 0.8348
 8416/60000 [===>..........................] - ETA: 1:22 - loss: 0.5231 - categorical_accuracy: 0.8358
 8448/60000 [===>..........................] - ETA: 1:22 - loss: 0.5216 - categorical_accuracy: 0.8363
 8480/60000 [===>..........................] - ETA: 1:22 - loss: 0.5207 - categorical_accuracy: 0.8366
 8544/60000 [===>..........................] - ETA: 1:22 - loss: 0.5189 - categorical_accuracy: 0.8368
 8576/60000 [===>..........................] - ETA: 1:22 - loss: 0.5175 - categorical_accuracy: 0.8373
 8608/60000 [===>..........................] - ETA: 1:22 - loss: 0.5162 - categorical_accuracy: 0.8378
 8672/60000 [===>..........................] - ETA: 1:22 - loss: 0.5137 - categorical_accuracy: 0.8384
 8736/60000 [===>..........................] - ETA: 1:22 - loss: 0.5110 - categorical_accuracy: 0.8393
 8800/60000 [===>..........................] - ETA: 1:21 - loss: 0.5078 - categorical_accuracy: 0.8403
 8864/60000 [===>..........................] - ETA: 1:21 - loss: 0.5071 - categorical_accuracy: 0.8408
 8896/60000 [===>..........................] - ETA: 1:21 - loss: 0.5055 - categorical_accuracy: 0.8414
 8928/60000 [===>..........................] - ETA: 1:21 - loss: 0.5053 - categorical_accuracy: 0.8415
 8992/60000 [===>..........................] - ETA: 1:21 - loss: 0.5040 - categorical_accuracy: 0.8417
 9056/60000 [===>..........................] - ETA: 1:21 - loss: 0.5017 - categorical_accuracy: 0.8424
 9120/60000 [===>..........................] - ETA: 1:21 - loss: 0.5010 - categorical_accuracy: 0.8430
 9184/60000 [===>..........................] - ETA: 1:21 - loss: 0.4988 - categorical_accuracy: 0.8435
 9216/60000 [===>..........................] - ETA: 1:21 - loss: 0.4985 - categorical_accuracy: 0.8436
 9248/60000 [===>..........................] - ETA: 1:21 - loss: 0.4991 - categorical_accuracy: 0.8438
 9280/60000 [===>..........................] - ETA: 1:21 - loss: 0.4978 - categorical_accuracy: 0.8442
 9312/60000 [===>..........................] - ETA: 1:21 - loss: 0.4969 - categorical_accuracy: 0.8444
 9376/60000 [===>..........................] - ETA: 1:20 - loss: 0.4941 - categorical_accuracy: 0.8453
 9440/60000 [===>..........................] - ETA: 1:20 - loss: 0.4922 - categorical_accuracy: 0.8459
 9472/60000 [===>..........................] - ETA: 1:20 - loss: 0.4912 - categorical_accuracy: 0.8461
 9504/60000 [===>..........................] - ETA: 1:20 - loss: 0.4897 - categorical_accuracy: 0.8466
 9536/60000 [===>..........................] - ETA: 1:20 - loss: 0.4891 - categorical_accuracy: 0.8468
 9600/60000 [===>..........................] - ETA: 1:20 - loss: 0.4879 - categorical_accuracy: 0.8473
 9664/60000 [===>..........................] - ETA: 1:20 - loss: 0.4860 - categorical_accuracy: 0.8479
 9728/60000 [===>..........................] - ETA: 1:20 - loss: 0.4840 - categorical_accuracy: 0.8482
 9760/60000 [===>..........................] - ETA: 1:20 - loss: 0.4830 - categorical_accuracy: 0.8484
 9792/60000 [===>..........................] - ETA: 1:20 - loss: 0.4827 - categorical_accuracy: 0.8483
 9824/60000 [===>..........................] - ETA: 1:20 - loss: 0.4823 - categorical_accuracy: 0.8484
 9856/60000 [===>..........................] - ETA: 1:20 - loss: 0.4815 - categorical_accuracy: 0.8487
 9920/60000 [===>..........................] - ETA: 1:20 - loss: 0.4799 - categorical_accuracy: 0.8492
 9984/60000 [===>..........................] - ETA: 1:19 - loss: 0.4781 - categorical_accuracy: 0.8498
10048/60000 [====>.........................] - ETA: 1:19 - loss: 0.4768 - categorical_accuracy: 0.8502
10080/60000 [====>.........................] - ETA: 1:19 - loss: 0.4760 - categorical_accuracy: 0.8504
10112/60000 [====>.........................] - ETA: 1:19 - loss: 0.4748 - categorical_accuracy: 0.8508
10144/60000 [====>.........................] - ETA: 1:19 - loss: 0.4735 - categorical_accuracy: 0.8511
10176/60000 [====>.........................] - ETA: 1:19 - loss: 0.4733 - categorical_accuracy: 0.8512
10208/60000 [====>.........................] - ETA: 1:19 - loss: 0.4729 - categorical_accuracy: 0.8514
10272/60000 [====>.........................] - ETA: 1:19 - loss: 0.4708 - categorical_accuracy: 0.8519
10336/60000 [====>.........................] - ETA: 1:19 - loss: 0.4690 - categorical_accuracy: 0.8525
10368/60000 [====>.........................] - ETA: 1:19 - loss: 0.4677 - categorical_accuracy: 0.8529
10400/60000 [====>.........................] - ETA: 1:19 - loss: 0.4668 - categorical_accuracy: 0.8532
10432/60000 [====>.........................] - ETA: 1:19 - loss: 0.4665 - categorical_accuracy: 0.8534
10496/60000 [====>.........................] - ETA: 1:19 - loss: 0.4642 - categorical_accuracy: 0.8542
10528/60000 [====>.........................] - ETA: 1:19 - loss: 0.4637 - categorical_accuracy: 0.8543
10560/60000 [====>.........................] - ETA: 1:19 - loss: 0.4629 - categorical_accuracy: 0.8545
10624/60000 [====>.........................] - ETA: 1:18 - loss: 0.4607 - categorical_accuracy: 0.8550
10688/60000 [====>.........................] - ETA: 1:18 - loss: 0.4589 - categorical_accuracy: 0.8555
10720/60000 [====>.........................] - ETA: 1:18 - loss: 0.4580 - categorical_accuracy: 0.8559
10784/60000 [====>.........................] - ETA: 1:18 - loss: 0.4563 - categorical_accuracy: 0.8564
10848/60000 [====>.........................] - ETA: 1:18 - loss: 0.4555 - categorical_accuracy: 0.8567
10880/60000 [====>.........................] - ETA: 1:18 - loss: 0.4551 - categorical_accuracy: 0.8569
10944/60000 [====>.........................] - ETA: 1:18 - loss: 0.4529 - categorical_accuracy: 0.8576
11008/60000 [====>.........................] - ETA: 1:18 - loss: 0.4507 - categorical_accuracy: 0.8584
11072/60000 [====>.........................] - ETA: 1:18 - loss: 0.4486 - categorical_accuracy: 0.8590
11104/60000 [====>.........................] - ETA: 1:18 - loss: 0.4484 - categorical_accuracy: 0.8591
11136/60000 [====>.........................] - ETA: 1:17 - loss: 0.4480 - categorical_accuracy: 0.8594
11168/60000 [====>.........................] - ETA: 1:17 - loss: 0.4469 - categorical_accuracy: 0.8598
11200/60000 [====>.........................] - ETA: 1:17 - loss: 0.4459 - categorical_accuracy: 0.8601
11264/60000 [====>.........................] - ETA: 1:17 - loss: 0.4439 - categorical_accuracy: 0.8606
11328/60000 [====>.........................] - ETA: 1:17 - loss: 0.4427 - categorical_accuracy: 0.8610
11360/60000 [====>.........................] - ETA: 1:17 - loss: 0.4419 - categorical_accuracy: 0.8613
11424/60000 [====>.........................] - ETA: 1:17 - loss: 0.4404 - categorical_accuracy: 0.8617
11456/60000 [====>.........................] - ETA: 1:17 - loss: 0.4402 - categorical_accuracy: 0.8617
11488/60000 [====>.........................] - ETA: 1:17 - loss: 0.4393 - categorical_accuracy: 0.8620
11552/60000 [====>.........................] - ETA: 1:17 - loss: 0.4388 - categorical_accuracy: 0.8622
11584/60000 [====>.........................] - ETA: 1:17 - loss: 0.4382 - categorical_accuracy: 0.8622
11616/60000 [====>.........................] - ETA: 1:17 - loss: 0.4373 - categorical_accuracy: 0.8625
11680/60000 [====>.........................] - ETA: 1:17 - loss: 0.4365 - categorical_accuracy: 0.8629
11744/60000 [====>.........................] - ETA: 1:17 - loss: 0.4355 - categorical_accuracy: 0.8632
11776/60000 [====>.........................] - ETA: 1:16 - loss: 0.4355 - categorical_accuracy: 0.8632
11808/60000 [====>.........................] - ETA: 1:16 - loss: 0.4346 - categorical_accuracy: 0.8634
11840/60000 [====>.........................] - ETA: 1:16 - loss: 0.4341 - categorical_accuracy: 0.8637
11904/60000 [====>.........................] - ETA: 1:16 - loss: 0.4325 - categorical_accuracy: 0.8641
11968/60000 [====>.........................] - ETA: 1:16 - loss: 0.4319 - categorical_accuracy: 0.8645
12000/60000 [=====>........................] - ETA: 1:16 - loss: 0.4315 - categorical_accuracy: 0.8648
12064/60000 [=====>........................] - ETA: 1:16 - loss: 0.4297 - categorical_accuracy: 0.8653
12128/60000 [=====>........................] - ETA: 1:16 - loss: 0.4290 - categorical_accuracy: 0.8654
12160/60000 [=====>........................] - ETA: 1:16 - loss: 0.4283 - categorical_accuracy: 0.8655
12192/60000 [=====>........................] - ETA: 1:16 - loss: 0.4279 - categorical_accuracy: 0.8656
12256/60000 [=====>........................] - ETA: 1:16 - loss: 0.4267 - categorical_accuracy: 0.8660
12320/60000 [=====>........................] - ETA: 1:15 - loss: 0.4250 - categorical_accuracy: 0.8666
12384/60000 [=====>........................] - ETA: 1:15 - loss: 0.4248 - categorical_accuracy: 0.8670
12416/60000 [=====>........................] - ETA: 1:15 - loss: 0.4240 - categorical_accuracy: 0.8672
12448/60000 [=====>........................] - ETA: 1:15 - loss: 0.4235 - categorical_accuracy: 0.8673
12480/60000 [=====>........................] - ETA: 1:15 - loss: 0.4227 - categorical_accuracy: 0.8675
12544/60000 [=====>........................] - ETA: 1:15 - loss: 0.4208 - categorical_accuracy: 0.8681
12576/60000 [=====>........................] - ETA: 1:15 - loss: 0.4205 - categorical_accuracy: 0.8682
12608/60000 [=====>........................] - ETA: 1:15 - loss: 0.4195 - categorical_accuracy: 0.8685
12640/60000 [=====>........................] - ETA: 1:15 - loss: 0.4187 - categorical_accuracy: 0.8688
12672/60000 [=====>........................] - ETA: 1:15 - loss: 0.4183 - categorical_accuracy: 0.8690
12736/60000 [=====>........................] - ETA: 1:15 - loss: 0.4172 - categorical_accuracy: 0.8693
12768/60000 [=====>........................] - ETA: 1:15 - loss: 0.4164 - categorical_accuracy: 0.8695
12832/60000 [=====>........................] - ETA: 1:15 - loss: 0.4147 - categorical_accuracy: 0.8700
12896/60000 [=====>........................] - ETA: 1:15 - loss: 0.4133 - categorical_accuracy: 0.8703
12928/60000 [=====>........................] - ETA: 1:15 - loss: 0.4130 - categorical_accuracy: 0.8705
12960/60000 [=====>........................] - ETA: 1:14 - loss: 0.4129 - categorical_accuracy: 0.8707
13024/60000 [=====>........................] - ETA: 1:14 - loss: 0.4114 - categorical_accuracy: 0.8712
13088/60000 [=====>........................] - ETA: 1:14 - loss: 0.4101 - categorical_accuracy: 0.8716
13120/60000 [=====>........................] - ETA: 1:14 - loss: 0.4092 - categorical_accuracy: 0.8718
13184/60000 [=====>........................] - ETA: 1:14 - loss: 0.4083 - categorical_accuracy: 0.8720
13248/60000 [=====>........................] - ETA: 1:14 - loss: 0.4075 - categorical_accuracy: 0.8722
13312/60000 [=====>........................] - ETA: 1:14 - loss: 0.4061 - categorical_accuracy: 0.8727
13376/60000 [=====>........................] - ETA: 1:14 - loss: 0.4047 - categorical_accuracy: 0.8731
13408/60000 [=====>........................] - ETA: 1:14 - loss: 0.4045 - categorical_accuracy: 0.8733
13440/60000 [=====>........................] - ETA: 1:14 - loss: 0.4041 - categorical_accuracy: 0.8734
13472/60000 [=====>........................] - ETA: 1:14 - loss: 0.4033 - categorical_accuracy: 0.8737
13504/60000 [=====>........................] - ETA: 1:14 - loss: 0.4030 - categorical_accuracy: 0.8737
13568/60000 [=====>........................] - ETA: 1:13 - loss: 0.4020 - categorical_accuracy: 0.8742
13600/60000 [=====>........................] - ETA: 1:13 - loss: 0.4013 - categorical_accuracy: 0.8744
13632/60000 [=====>........................] - ETA: 1:13 - loss: 0.4006 - categorical_accuracy: 0.8746
13664/60000 [=====>........................] - ETA: 1:13 - loss: 0.3997 - categorical_accuracy: 0.8749
13696/60000 [=====>........................] - ETA: 1:13 - loss: 0.3991 - categorical_accuracy: 0.8750
13728/60000 [=====>........................] - ETA: 1:13 - loss: 0.3982 - categorical_accuracy: 0.8753
13760/60000 [=====>........................] - ETA: 1:13 - loss: 0.3975 - categorical_accuracy: 0.8756
13824/60000 [=====>........................] - ETA: 1:13 - loss: 0.3972 - categorical_accuracy: 0.8758
13888/60000 [=====>........................] - ETA: 1:13 - loss: 0.3958 - categorical_accuracy: 0.8762
13920/60000 [=====>........................] - ETA: 1:13 - loss: 0.3951 - categorical_accuracy: 0.8764
13952/60000 [=====>........................] - ETA: 1:13 - loss: 0.3943 - categorical_accuracy: 0.8767
14016/60000 [======>.......................] - ETA: 1:13 - loss: 0.3931 - categorical_accuracy: 0.8769
14080/60000 [======>.......................] - ETA: 1:13 - loss: 0.3916 - categorical_accuracy: 0.8774
14144/60000 [======>.......................] - ETA: 1:12 - loss: 0.3904 - categorical_accuracy: 0.8778
14176/60000 [======>.......................] - ETA: 1:12 - loss: 0.3898 - categorical_accuracy: 0.8780
14208/60000 [======>.......................] - ETA: 1:12 - loss: 0.3902 - categorical_accuracy: 0.8779
14240/60000 [======>.......................] - ETA: 1:12 - loss: 0.3896 - categorical_accuracy: 0.8780
14272/60000 [======>.......................] - ETA: 1:12 - loss: 0.3888 - categorical_accuracy: 0.8783
14304/60000 [======>.......................] - ETA: 1:12 - loss: 0.3883 - categorical_accuracy: 0.8784
14336/60000 [======>.......................] - ETA: 1:12 - loss: 0.3880 - categorical_accuracy: 0.8786
14368/60000 [======>.......................] - ETA: 1:12 - loss: 0.3879 - categorical_accuracy: 0.8785
14400/60000 [======>.......................] - ETA: 1:12 - loss: 0.3881 - categorical_accuracy: 0.8786
14432/60000 [======>.......................] - ETA: 1:12 - loss: 0.3883 - categorical_accuracy: 0.8787
14464/60000 [======>.......................] - ETA: 1:12 - loss: 0.3879 - categorical_accuracy: 0.8787
14496/60000 [======>.......................] - ETA: 1:12 - loss: 0.3878 - categorical_accuracy: 0.8787
14560/60000 [======>.......................] - ETA: 1:12 - loss: 0.3870 - categorical_accuracy: 0.8790
14624/60000 [======>.......................] - ETA: 1:12 - loss: 0.3858 - categorical_accuracy: 0.8794
14688/60000 [======>.......................] - ETA: 1:12 - loss: 0.3852 - categorical_accuracy: 0.8797
14720/60000 [======>.......................] - ETA: 1:12 - loss: 0.3848 - categorical_accuracy: 0.8798
14752/60000 [======>.......................] - ETA: 1:12 - loss: 0.3842 - categorical_accuracy: 0.8799
14784/60000 [======>.......................] - ETA: 1:12 - loss: 0.3835 - categorical_accuracy: 0.8802
14848/60000 [======>.......................] - ETA: 1:11 - loss: 0.3832 - categorical_accuracy: 0.8803
14912/60000 [======>.......................] - ETA: 1:11 - loss: 0.3821 - categorical_accuracy: 0.8806
14976/60000 [======>.......................] - ETA: 1:11 - loss: 0.3809 - categorical_accuracy: 0.8810
15008/60000 [======>.......................] - ETA: 1:11 - loss: 0.3801 - categorical_accuracy: 0.8813
15072/60000 [======>.......................] - ETA: 1:11 - loss: 0.3794 - categorical_accuracy: 0.8814
15136/60000 [======>.......................] - ETA: 1:11 - loss: 0.3782 - categorical_accuracy: 0.8817
15200/60000 [======>.......................] - ETA: 1:11 - loss: 0.3770 - categorical_accuracy: 0.8820
15232/60000 [======>.......................] - ETA: 1:11 - loss: 0.3767 - categorical_accuracy: 0.8822
15264/60000 [======>.......................] - ETA: 1:11 - loss: 0.3763 - categorical_accuracy: 0.8823
15296/60000 [======>.......................] - ETA: 1:11 - loss: 0.3761 - categorical_accuracy: 0.8823
15360/60000 [======>.......................] - ETA: 1:11 - loss: 0.3752 - categorical_accuracy: 0.8826
15424/60000 [======>.......................] - ETA: 1:10 - loss: 0.3742 - categorical_accuracy: 0.8830
15488/60000 [======>.......................] - ETA: 1:10 - loss: 0.3732 - categorical_accuracy: 0.8833
15552/60000 [======>.......................] - ETA: 1:10 - loss: 0.3723 - categorical_accuracy: 0.8835
15584/60000 [======>.......................] - ETA: 1:10 - loss: 0.3717 - categorical_accuracy: 0.8837
15648/60000 [======>.......................] - ETA: 1:10 - loss: 0.3710 - categorical_accuracy: 0.8840
15712/60000 [======>.......................] - ETA: 1:10 - loss: 0.3699 - categorical_accuracy: 0.8843
15776/60000 [======>.......................] - ETA: 1:10 - loss: 0.3689 - categorical_accuracy: 0.8846
15808/60000 [======>.......................] - ETA: 1:10 - loss: 0.3684 - categorical_accuracy: 0.8848
15840/60000 [======>.......................] - ETA: 1:10 - loss: 0.3681 - categorical_accuracy: 0.8849
15872/60000 [======>.......................] - ETA: 1:10 - loss: 0.3674 - categorical_accuracy: 0.8851
15904/60000 [======>.......................] - ETA: 1:10 - loss: 0.3669 - categorical_accuracy: 0.8853
15968/60000 [======>.......................] - ETA: 1:10 - loss: 0.3659 - categorical_accuracy: 0.8856
16000/60000 [=======>......................] - ETA: 1:09 - loss: 0.3654 - categorical_accuracy: 0.8858
16032/60000 [=======>......................] - ETA: 1:09 - loss: 0.3647 - categorical_accuracy: 0.8860
16096/60000 [=======>......................] - ETA: 1:09 - loss: 0.3638 - categorical_accuracy: 0.8864
16160/60000 [=======>......................] - ETA: 1:09 - loss: 0.3639 - categorical_accuracy: 0.8864
16192/60000 [=======>......................] - ETA: 1:09 - loss: 0.3635 - categorical_accuracy: 0.8866
16256/60000 [=======>......................] - ETA: 1:09 - loss: 0.3625 - categorical_accuracy: 0.8869
16320/60000 [=======>......................] - ETA: 1:09 - loss: 0.3618 - categorical_accuracy: 0.8869
16352/60000 [=======>......................] - ETA: 1:09 - loss: 0.3611 - categorical_accuracy: 0.8872
16416/60000 [=======>......................] - ETA: 1:09 - loss: 0.3604 - categorical_accuracy: 0.8874
16480/60000 [=======>......................] - ETA: 1:09 - loss: 0.3594 - categorical_accuracy: 0.8876
16512/60000 [=======>......................] - ETA: 1:09 - loss: 0.3588 - categorical_accuracy: 0.8878
16576/60000 [=======>......................] - ETA: 1:09 - loss: 0.3585 - categorical_accuracy: 0.8880
16640/60000 [=======>......................] - ETA: 1:08 - loss: 0.3581 - categorical_accuracy: 0.8882
16704/60000 [=======>......................] - ETA: 1:08 - loss: 0.3570 - categorical_accuracy: 0.8885
16736/60000 [=======>......................] - ETA: 1:08 - loss: 0.3566 - categorical_accuracy: 0.8886
16768/60000 [=======>......................] - ETA: 1:08 - loss: 0.3563 - categorical_accuracy: 0.8887
16800/60000 [=======>......................] - ETA: 1:08 - loss: 0.3557 - categorical_accuracy: 0.8889
16832/60000 [=======>......................] - ETA: 1:08 - loss: 0.3552 - categorical_accuracy: 0.8891
16864/60000 [=======>......................] - ETA: 1:08 - loss: 0.3548 - categorical_accuracy: 0.8892
16896/60000 [=======>......................] - ETA: 1:08 - loss: 0.3545 - categorical_accuracy: 0.8892
16928/60000 [=======>......................] - ETA: 1:08 - loss: 0.3541 - categorical_accuracy: 0.8894
16960/60000 [=======>......................] - ETA: 1:08 - loss: 0.3537 - categorical_accuracy: 0.8894
17024/60000 [=======>......................] - ETA: 1:08 - loss: 0.3536 - categorical_accuracy: 0.8896
17088/60000 [=======>......................] - ETA: 1:08 - loss: 0.3525 - categorical_accuracy: 0.8900
17152/60000 [=======>......................] - ETA: 1:08 - loss: 0.3525 - categorical_accuracy: 0.8901
17184/60000 [=======>......................] - ETA: 1:08 - loss: 0.3522 - categorical_accuracy: 0.8901
17248/60000 [=======>......................] - ETA: 1:07 - loss: 0.3513 - categorical_accuracy: 0.8904
17312/60000 [=======>......................] - ETA: 1:07 - loss: 0.3506 - categorical_accuracy: 0.8907
17376/60000 [=======>......................] - ETA: 1:07 - loss: 0.3498 - categorical_accuracy: 0.8909
17440/60000 [=======>......................] - ETA: 1:07 - loss: 0.3497 - categorical_accuracy: 0.8911
17472/60000 [=======>......................] - ETA: 1:07 - loss: 0.3495 - categorical_accuracy: 0.8911
17504/60000 [=======>......................] - ETA: 1:07 - loss: 0.3493 - categorical_accuracy: 0.8912
17536/60000 [=======>......................] - ETA: 1:07 - loss: 0.3489 - categorical_accuracy: 0.8913
17600/60000 [=======>......................] - ETA: 1:07 - loss: 0.3484 - categorical_accuracy: 0.8913
17664/60000 [=======>......................] - ETA: 1:07 - loss: 0.3477 - categorical_accuracy: 0.8916
17728/60000 [=======>......................] - ETA: 1:07 - loss: 0.3472 - categorical_accuracy: 0.8918
17792/60000 [=======>......................] - ETA: 1:07 - loss: 0.3472 - categorical_accuracy: 0.8919
17824/60000 [=======>......................] - ETA: 1:06 - loss: 0.3469 - categorical_accuracy: 0.8919
17888/60000 [=======>......................] - ETA: 1:06 - loss: 0.3462 - categorical_accuracy: 0.8922
17920/60000 [=======>......................] - ETA: 1:06 - loss: 0.3457 - categorical_accuracy: 0.8924
17952/60000 [=======>......................] - ETA: 1:06 - loss: 0.3453 - categorical_accuracy: 0.8925
18016/60000 [========>.....................] - ETA: 1:06 - loss: 0.3449 - categorical_accuracy: 0.8927
18048/60000 [========>.....................] - ETA: 1:06 - loss: 0.3444 - categorical_accuracy: 0.8928
18080/60000 [========>.....................] - ETA: 1:06 - loss: 0.3441 - categorical_accuracy: 0.8929
18112/60000 [========>.....................] - ETA: 1:06 - loss: 0.3438 - categorical_accuracy: 0.8930
18144/60000 [========>.....................] - ETA: 1:06 - loss: 0.3434 - categorical_accuracy: 0.8931
18176/60000 [========>.....................] - ETA: 1:06 - loss: 0.3432 - categorical_accuracy: 0.8932
18208/60000 [========>.....................] - ETA: 1:06 - loss: 0.3428 - categorical_accuracy: 0.8933
18240/60000 [========>.....................] - ETA: 1:06 - loss: 0.3422 - categorical_accuracy: 0.8935
18272/60000 [========>.....................] - ETA: 1:06 - loss: 0.3418 - categorical_accuracy: 0.8936
18304/60000 [========>.....................] - ETA: 1:06 - loss: 0.3413 - categorical_accuracy: 0.8937
18368/60000 [========>.....................] - ETA: 1:06 - loss: 0.3404 - categorical_accuracy: 0.8939
18432/60000 [========>.....................] - ETA: 1:06 - loss: 0.3399 - categorical_accuracy: 0.8942
18496/60000 [========>.....................] - ETA: 1:05 - loss: 0.3390 - categorical_accuracy: 0.8945
18528/60000 [========>.....................] - ETA: 1:05 - loss: 0.3384 - categorical_accuracy: 0.8946
18592/60000 [========>.....................] - ETA: 1:05 - loss: 0.3375 - categorical_accuracy: 0.8950
18656/60000 [========>.....................] - ETA: 1:05 - loss: 0.3364 - categorical_accuracy: 0.8953
18720/60000 [========>.....................] - ETA: 1:05 - loss: 0.3363 - categorical_accuracy: 0.8954
18784/60000 [========>.....................] - ETA: 1:05 - loss: 0.3354 - categorical_accuracy: 0.8957
18816/60000 [========>.....................] - ETA: 1:05 - loss: 0.3352 - categorical_accuracy: 0.8957
18848/60000 [========>.....................] - ETA: 1:05 - loss: 0.3349 - categorical_accuracy: 0.8957
18880/60000 [========>.....................] - ETA: 1:05 - loss: 0.3346 - categorical_accuracy: 0.8959
18912/60000 [========>.....................] - ETA: 1:05 - loss: 0.3341 - categorical_accuracy: 0.8960
18944/60000 [========>.....................] - ETA: 1:05 - loss: 0.3337 - categorical_accuracy: 0.8962
18976/60000 [========>.....................] - ETA: 1:05 - loss: 0.3340 - categorical_accuracy: 0.8962
19008/60000 [========>.....................] - ETA: 1:05 - loss: 0.3336 - categorical_accuracy: 0.8964
19040/60000 [========>.....................] - ETA: 1:05 - loss: 0.3331 - categorical_accuracy: 0.8966
19072/60000 [========>.....................] - ETA: 1:05 - loss: 0.3328 - categorical_accuracy: 0.8967
19104/60000 [========>.....................] - ETA: 1:04 - loss: 0.3323 - categorical_accuracy: 0.8968
19136/60000 [========>.....................] - ETA: 1:04 - loss: 0.3318 - categorical_accuracy: 0.8969
19168/60000 [========>.....................] - ETA: 1:04 - loss: 0.3319 - categorical_accuracy: 0.8970
19232/60000 [========>.....................] - ETA: 1:04 - loss: 0.3310 - categorical_accuracy: 0.8973
19296/60000 [========>.....................] - ETA: 1:04 - loss: 0.3307 - categorical_accuracy: 0.8974
19328/60000 [========>.....................] - ETA: 1:04 - loss: 0.3303 - categorical_accuracy: 0.8976
19392/60000 [========>.....................] - ETA: 1:04 - loss: 0.3295 - categorical_accuracy: 0.8978
19424/60000 [========>.....................] - ETA: 1:04 - loss: 0.3291 - categorical_accuracy: 0.8979
19488/60000 [========>.....................] - ETA: 1:04 - loss: 0.3283 - categorical_accuracy: 0.8981
19552/60000 [========>.....................] - ETA: 1:04 - loss: 0.3274 - categorical_accuracy: 0.8985
19584/60000 [========>.....................] - ETA: 1:04 - loss: 0.3272 - categorical_accuracy: 0.8986
19648/60000 [========>.....................] - ETA: 1:04 - loss: 0.3265 - categorical_accuracy: 0.8988
19712/60000 [========>.....................] - ETA: 1:03 - loss: 0.3257 - categorical_accuracy: 0.8990
19776/60000 [========>.....................] - ETA: 1:03 - loss: 0.3250 - categorical_accuracy: 0.8992
19840/60000 [========>.....................] - ETA: 1:03 - loss: 0.3244 - categorical_accuracy: 0.8993
19904/60000 [========>.....................] - ETA: 1:03 - loss: 0.3238 - categorical_accuracy: 0.8995
19968/60000 [========>.....................] - ETA: 1:03 - loss: 0.3232 - categorical_accuracy: 0.8996
20000/60000 [=========>....................] - ETA: 1:03 - loss: 0.3227 - categorical_accuracy: 0.8997
20032/60000 [=========>....................] - ETA: 1:03 - loss: 0.3224 - categorical_accuracy: 0.8999
20064/60000 [=========>....................] - ETA: 1:03 - loss: 0.3219 - categorical_accuracy: 0.9000
20096/60000 [=========>....................] - ETA: 1:03 - loss: 0.3215 - categorical_accuracy: 0.9001
20128/60000 [=========>....................] - ETA: 1:03 - loss: 0.3211 - categorical_accuracy: 0.9002
20192/60000 [=========>....................] - ETA: 1:03 - loss: 0.3206 - categorical_accuracy: 0.9004
20224/60000 [=========>....................] - ETA: 1:03 - loss: 0.3202 - categorical_accuracy: 0.9004
20256/60000 [=========>....................] - ETA: 1:03 - loss: 0.3199 - categorical_accuracy: 0.9005
20320/60000 [=========>....................] - ETA: 1:02 - loss: 0.3195 - categorical_accuracy: 0.9007
20384/60000 [=========>....................] - ETA: 1:02 - loss: 0.3188 - categorical_accuracy: 0.9009
20416/60000 [=========>....................] - ETA: 1:02 - loss: 0.3185 - categorical_accuracy: 0.9010
20448/60000 [=========>....................] - ETA: 1:02 - loss: 0.3181 - categorical_accuracy: 0.9011
20512/60000 [=========>....................] - ETA: 1:02 - loss: 0.3179 - categorical_accuracy: 0.9011
20576/60000 [=========>....................] - ETA: 1:02 - loss: 0.3170 - categorical_accuracy: 0.9014
20640/60000 [=========>....................] - ETA: 1:02 - loss: 0.3163 - categorical_accuracy: 0.9016
20704/60000 [=========>....................] - ETA: 1:02 - loss: 0.3155 - categorical_accuracy: 0.9019
20736/60000 [=========>....................] - ETA: 1:02 - loss: 0.3156 - categorical_accuracy: 0.9019
20800/60000 [=========>....................] - ETA: 1:02 - loss: 0.3149 - categorical_accuracy: 0.9021
20864/60000 [=========>....................] - ETA: 1:02 - loss: 0.3150 - categorical_accuracy: 0.9021
20928/60000 [=========>....................] - ETA: 1:01 - loss: 0.3146 - categorical_accuracy: 0.9021
20992/60000 [=========>....................] - ETA: 1:01 - loss: 0.3145 - categorical_accuracy: 0.9022
21056/60000 [=========>....................] - ETA: 1:01 - loss: 0.3137 - categorical_accuracy: 0.9025
21088/60000 [=========>....................] - ETA: 1:01 - loss: 0.3133 - categorical_accuracy: 0.9026
21120/60000 [=========>....................] - ETA: 1:01 - loss: 0.3133 - categorical_accuracy: 0.9027
21152/60000 [=========>....................] - ETA: 1:01 - loss: 0.3129 - categorical_accuracy: 0.9028
21216/60000 [=========>....................] - ETA: 1:01 - loss: 0.3126 - categorical_accuracy: 0.9030
21280/60000 [=========>....................] - ETA: 1:01 - loss: 0.3118 - categorical_accuracy: 0.9032
21344/60000 [=========>....................] - ETA: 1:01 - loss: 0.3110 - categorical_accuracy: 0.9034
21376/60000 [=========>....................] - ETA: 1:01 - loss: 0.3108 - categorical_accuracy: 0.9035
21408/60000 [=========>....................] - ETA: 1:01 - loss: 0.3104 - categorical_accuracy: 0.9036
21472/60000 [=========>....................] - ETA: 1:01 - loss: 0.3098 - categorical_accuracy: 0.9037
21536/60000 [=========>....................] - ETA: 1:00 - loss: 0.3100 - categorical_accuracy: 0.9036
21600/60000 [=========>....................] - ETA: 1:00 - loss: 0.3095 - categorical_accuracy: 0.9038
21664/60000 [=========>....................] - ETA: 1:00 - loss: 0.3091 - categorical_accuracy: 0.9041
21696/60000 [=========>....................] - ETA: 1:00 - loss: 0.3090 - categorical_accuracy: 0.9040
21760/60000 [=========>....................] - ETA: 1:00 - loss: 0.3089 - categorical_accuracy: 0.9040
21824/60000 [=========>....................] - ETA: 1:00 - loss: 0.3084 - categorical_accuracy: 0.9042
21888/60000 [=========>....................] - ETA: 1:00 - loss: 0.3084 - categorical_accuracy: 0.9042
21952/60000 [=========>....................] - ETA: 1:00 - loss: 0.3082 - categorical_accuracy: 0.9042
22016/60000 [==========>...................] - ETA: 1:00 - loss: 0.3076 - categorical_accuracy: 0.9043
22080/60000 [==========>...................] - ETA: 1:00 - loss: 0.3072 - categorical_accuracy: 0.9044
22144/60000 [==========>...................] - ETA: 59s - loss: 0.3066 - categorical_accuracy: 0.9046 
22176/60000 [==========>...................] - ETA: 59s - loss: 0.3063 - categorical_accuracy: 0.9046
22208/60000 [==========>...................] - ETA: 59s - loss: 0.3065 - categorical_accuracy: 0.9047
22240/60000 [==========>...................] - ETA: 59s - loss: 0.3062 - categorical_accuracy: 0.9048
22272/60000 [==========>...................] - ETA: 59s - loss: 0.3059 - categorical_accuracy: 0.9049
22336/60000 [==========>...................] - ETA: 59s - loss: 0.3053 - categorical_accuracy: 0.9050
22368/60000 [==========>...................] - ETA: 59s - loss: 0.3053 - categorical_accuracy: 0.9051
22400/60000 [==========>...................] - ETA: 59s - loss: 0.3052 - categorical_accuracy: 0.9051
22464/60000 [==========>...................] - ETA: 59s - loss: 0.3045 - categorical_accuracy: 0.9053
22528/60000 [==========>...................] - ETA: 59s - loss: 0.3041 - categorical_accuracy: 0.9055
22592/60000 [==========>...................] - ETA: 59s - loss: 0.3038 - categorical_accuracy: 0.9056
22656/60000 [==========>...................] - ETA: 59s - loss: 0.3034 - categorical_accuracy: 0.9057
22720/60000 [==========>...................] - ETA: 59s - loss: 0.3030 - categorical_accuracy: 0.9059
22784/60000 [==========>...................] - ETA: 58s - loss: 0.3027 - categorical_accuracy: 0.9059
22816/60000 [==========>...................] - ETA: 58s - loss: 0.3025 - categorical_accuracy: 0.9060
22848/60000 [==========>...................] - ETA: 58s - loss: 0.3021 - categorical_accuracy: 0.9061
22912/60000 [==========>...................] - ETA: 58s - loss: 0.3015 - categorical_accuracy: 0.9063
22976/60000 [==========>...................] - ETA: 58s - loss: 0.3008 - categorical_accuracy: 0.9066
23040/60000 [==========>...................] - ETA: 58s - loss: 0.3003 - categorical_accuracy: 0.9067
23072/60000 [==========>...................] - ETA: 58s - loss: 0.2999 - categorical_accuracy: 0.9069
23136/60000 [==========>...................] - ETA: 58s - loss: 0.2991 - categorical_accuracy: 0.9071
23200/60000 [==========>...................] - ETA: 58s - loss: 0.2992 - categorical_accuracy: 0.9072
23232/60000 [==========>...................] - ETA: 58s - loss: 0.2990 - categorical_accuracy: 0.9072
23264/60000 [==========>...................] - ETA: 58s - loss: 0.2992 - categorical_accuracy: 0.9071
23296/60000 [==========>...................] - ETA: 58s - loss: 0.2989 - categorical_accuracy: 0.9072
23360/60000 [==========>...................] - ETA: 58s - loss: 0.2985 - categorical_accuracy: 0.9074
23392/60000 [==========>...................] - ETA: 57s - loss: 0.2983 - categorical_accuracy: 0.9074
23424/60000 [==========>...................] - ETA: 57s - loss: 0.2981 - categorical_accuracy: 0.9075
23488/60000 [==========>...................] - ETA: 57s - loss: 0.2980 - categorical_accuracy: 0.9075
23520/60000 [==========>...................] - ETA: 57s - loss: 0.2977 - categorical_accuracy: 0.9077
23584/60000 [==========>...................] - ETA: 57s - loss: 0.2972 - categorical_accuracy: 0.9078
23648/60000 [==========>...................] - ETA: 57s - loss: 0.2966 - categorical_accuracy: 0.9080
23712/60000 [==========>...................] - ETA: 57s - loss: 0.2962 - categorical_accuracy: 0.9082
23776/60000 [==========>...................] - ETA: 57s - loss: 0.2956 - categorical_accuracy: 0.9084
23840/60000 [==========>...................] - ETA: 57s - loss: 0.2951 - categorical_accuracy: 0.9085
23904/60000 [==========>...................] - ETA: 57s - loss: 0.2945 - categorical_accuracy: 0.9087
23936/60000 [==========>...................] - ETA: 57s - loss: 0.2943 - categorical_accuracy: 0.9088
23968/60000 [==========>...................] - ETA: 57s - loss: 0.2940 - categorical_accuracy: 0.9088
24000/60000 [===========>..................] - ETA: 56s - loss: 0.2939 - categorical_accuracy: 0.9089
24064/60000 [===========>..................] - ETA: 56s - loss: 0.2940 - categorical_accuracy: 0.9089
24128/60000 [===========>..................] - ETA: 56s - loss: 0.2934 - categorical_accuracy: 0.9091
24160/60000 [===========>..................] - ETA: 56s - loss: 0.2931 - categorical_accuracy: 0.9091
24224/60000 [===========>..................] - ETA: 56s - loss: 0.2929 - categorical_accuracy: 0.9092
24256/60000 [===========>..................] - ETA: 56s - loss: 0.2927 - categorical_accuracy: 0.9093
24288/60000 [===========>..................] - ETA: 56s - loss: 0.2926 - categorical_accuracy: 0.9093
24320/60000 [===========>..................] - ETA: 56s - loss: 0.2923 - categorical_accuracy: 0.9093
24384/60000 [===========>..................] - ETA: 56s - loss: 0.2920 - categorical_accuracy: 0.9094
24448/60000 [===========>..................] - ETA: 56s - loss: 0.2914 - categorical_accuracy: 0.9096
24480/60000 [===========>..................] - ETA: 56s - loss: 0.2917 - categorical_accuracy: 0.9096
24544/60000 [===========>..................] - ETA: 56s - loss: 0.2913 - categorical_accuracy: 0.9097
24576/60000 [===========>..................] - ETA: 56s - loss: 0.2911 - categorical_accuracy: 0.9098
24608/60000 [===========>..................] - ETA: 56s - loss: 0.2908 - categorical_accuracy: 0.9099
24672/60000 [===========>..................] - ETA: 55s - loss: 0.2904 - categorical_accuracy: 0.9101
24736/60000 [===========>..................] - ETA: 55s - loss: 0.2902 - categorical_accuracy: 0.9101
24800/60000 [===========>..................] - ETA: 55s - loss: 0.2896 - categorical_accuracy: 0.9103
24864/60000 [===========>..................] - ETA: 55s - loss: 0.2891 - categorical_accuracy: 0.9104
24896/60000 [===========>..................] - ETA: 55s - loss: 0.2889 - categorical_accuracy: 0.9105
24928/60000 [===========>..................] - ETA: 55s - loss: 0.2889 - categorical_accuracy: 0.9105
24992/60000 [===========>..................] - ETA: 55s - loss: 0.2888 - categorical_accuracy: 0.9105
25056/60000 [===========>..................] - ETA: 55s - loss: 0.2881 - categorical_accuracy: 0.9107
25120/60000 [===========>..................] - ETA: 55s - loss: 0.2875 - categorical_accuracy: 0.9109
25184/60000 [===========>..................] - ETA: 55s - loss: 0.2869 - categorical_accuracy: 0.9112
25216/60000 [===========>..................] - ETA: 55s - loss: 0.2866 - categorical_accuracy: 0.9113
25248/60000 [===========>..................] - ETA: 55s - loss: 0.2863 - categorical_accuracy: 0.9114
25280/60000 [===========>..................] - ETA: 54s - loss: 0.2860 - categorical_accuracy: 0.9115
25312/60000 [===========>..................] - ETA: 54s - loss: 0.2857 - categorical_accuracy: 0.9116
25344/60000 [===========>..................] - ETA: 54s - loss: 0.2857 - categorical_accuracy: 0.9116
25376/60000 [===========>..................] - ETA: 54s - loss: 0.2854 - categorical_accuracy: 0.9116
25440/60000 [===========>..................] - ETA: 54s - loss: 0.2848 - categorical_accuracy: 0.9118
25504/60000 [===========>..................] - ETA: 54s - loss: 0.2846 - categorical_accuracy: 0.9119
25568/60000 [===========>..................] - ETA: 54s - loss: 0.2841 - categorical_accuracy: 0.9120
25632/60000 [===========>..................] - ETA: 54s - loss: 0.2838 - categorical_accuracy: 0.9121
25696/60000 [===========>..................] - ETA: 54s - loss: 0.2835 - categorical_accuracy: 0.9121
25728/60000 [===========>..................] - ETA: 54s - loss: 0.2831 - categorical_accuracy: 0.9122
25760/60000 [===========>..................] - ETA: 54s - loss: 0.2828 - categorical_accuracy: 0.9123
25792/60000 [===========>..................] - ETA: 54s - loss: 0.2826 - categorical_accuracy: 0.9124
25856/60000 [===========>..................] - ETA: 54s - loss: 0.2825 - categorical_accuracy: 0.9124
25888/60000 [===========>..................] - ETA: 53s - loss: 0.2823 - categorical_accuracy: 0.9125
25952/60000 [===========>..................] - ETA: 53s - loss: 0.2818 - categorical_accuracy: 0.9126
26016/60000 [============>.................] - ETA: 53s - loss: 0.2812 - categorical_accuracy: 0.9128
26048/60000 [============>.................] - ETA: 53s - loss: 0.2809 - categorical_accuracy: 0.9129
26080/60000 [============>.................] - ETA: 53s - loss: 0.2806 - categorical_accuracy: 0.9130
26144/60000 [============>.................] - ETA: 53s - loss: 0.2803 - categorical_accuracy: 0.9131
26208/60000 [============>.................] - ETA: 53s - loss: 0.2801 - categorical_accuracy: 0.9133
26272/60000 [============>.................] - ETA: 53s - loss: 0.2795 - categorical_accuracy: 0.9135
26336/60000 [============>.................] - ETA: 53s - loss: 0.2791 - categorical_accuracy: 0.9137
26368/60000 [============>.................] - ETA: 53s - loss: 0.2791 - categorical_accuracy: 0.9137
26400/60000 [============>.................] - ETA: 53s - loss: 0.2787 - categorical_accuracy: 0.9138
26464/60000 [============>.................] - ETA: 53s - loss: 0.2785 - categorical_accuracy: 0.9139
26496/60000 [============>.................] - ETA: 53s - loss: 0.2787 - categorical_accuracy: 0.9139
26528/60000 [============>.................] - ETA: 52s - loss: 0.2784 - categorical_accuracy: 0.9140
26592/60000 [============>.................] - ETA: 52s - loss: 0.2778 - categorical_accuracy: 0.9142
26656/60000 [============>.................] - ETA: 52s - loss: 0.2775 - categorical_accuracy: 0.9143
26688/60000 [============>.................] - ETA: 52s - loss: 0.2775 - categorical_accuracy: 0.9143
26720/60000 [============>.................] - ETA: 52s - loss: 0.2778 - categorical_accuracy: 0.9143
26752/60000 [============>.................] - ETA: 52s - loss: 0.2775 - categorical_accuracy: 0.9144
26784/60000 [============>.................] - ETA: 52s - loss: 0.2773 - categorical_accuracy: 0.9145
26816/60000 [============>.................] - ETA: 52s - loss: 0.2770 - categorical_accuracy: 0.9146
26848/60000 [============>.................] - ETA: 52s - loss: 0.2767 - categorical_accuracy: 0.9146
26880/60000 [============>.................] - ETA: 52s - loss: 0.2766 - categorical_accuracy: 0.9147
26944/60000 [============>.................] - ETA: 52s - loss: 0.2765 - categorical_accuracy: 0.9147
27008/60000 [============>.................] - ETA: 52s - loss: 0.2761 - categorical_accuracy: 0.9148
27040/60000 [============>.................] - ETA: 52s - loss: 0.2761 - categorical_accuracy: 0.9148
27072/60000 [============>.................] - ETA: 52s - loss: 0.2761 - categorical_accuracy: 0.9148
27104/60000 [============>.................] - ETA: 52s - loss: 0.2759 - categorical_accuracy: 0.9148
27168/60000 [============>.................] - ETA: 51s - loss: 0.2758 - categorical_accuracy: 0.9148
27232/60000 [============>.................] - ETA: 51s - loss: 0.2756 - categorical_accuracy: 0.9149
27296/60000 [============>.................] - ETA: 51s - loss: 0.2754 - categorical_accuracy: 0.9150
27360/60000 [============>.................] - ETA: 51s - loss: 0.2750 - categorical_accuracy: 0.9151
27392/60000 [============>.................] - ETA: 51s - loss: 0.2747 - categorical_accuracy: 0.9151
27424/60000 [============>.................] - ETA: 51s - loss: 0.2745 - categorical_accuracy: 0.9152
27456/60000 [============>.................] - ETA: 51s - loss: 0.2742 - categorical_accuracy: 0.9153
27488/60000 [============>.................] - ETA: 51s - loss: 0.2742 - categorical_accuracy: 0.9153
27520/60000 [============>.................] - ETA: 51s - loss: 0.2739 - categorical_accuracy: 0.9154
27552/60000 [============>.................] - ETA: 51s - loss: 0.2736 - categorical_accuracy: 0.9155
27584/60000 [============>.................] - ETA: 51s - loss: 0.2734 - categorical_accuracy: 0.9156
27648/60000 [============>.................] - ETA: 51s - loss: 0.2729 - categorical_accuracy: 0.9157
27712/60000 [============>.................] - ETA: 51s - loss: 0.2729 - categorical_accuracy: 0.9158
27744/60000 [============>.................] - ETA: 51s - loss: 0.2728 - categorical_accuracy: 0.9158
27808/60000 [============>.................] - ETA: 50s - loss: 0.2724 - categorical_accuracy: 0.9159
27872/60000 [============>.................] - ETA: 50s - loss: 0.2723 - categorical_accuracy: 0.9160
27936/60000 [============>.................] - ETA: 50s - loss: 0.2718 - categorical_accuracy: 0.9161
28000/60000 [=============>................] - ETA: 50s - loss: 0.2713 - categorical_accuracy: 0.9163
28032/60000 [=============>................] - ETA: 50s - loss: 0.2710 - categorical_accuracy: 0.9164
28064/60000 [=============>................] - ETA: 50s - loss: 0.2707 - categorical_accuracy: 0.9165
28096/60000 [=============>................] - ETA: 50s - loss: 0.2705 - categorical_accuracy: 0.9166
28160/60000 [=============>................] - ETA: 50s - loss: 0.2699 - categorical_accuracy: 0.9168
28224/60000 [=============>................] - ETA: 50s - loss: 0.2697 - categorical_accuracy: 0.9168
28288/60000 [=============>................] - ETA: 50s - loss: 0.2694 - categorical_accuracy: 0.9170
28320/60000 [=============>................] - ETA: 50s - loss: 0.2692 - categorical_accuracy: 0.9170
28352/60000 [=============>................] - ETA: 50s - loss: 0.2692 - categorical_accuracy: 0.9169
28416/60000 [=============>................] - ETA: 49s - loss: 0.2687 - categorical_accuracy: 0.9171
28480/60000 [=============>................] - ETA: 49s - loss: 0.2683 - categorical_accuracy: 0.9172
28512/60000 [=============>................] - ETA: 49s - loss: 0.2682 - categorical_accuracy: 0.9172
28544/60000 [=============>................] - ETA: 49s - loss: 0.2680 - categorical_accuracy: 0.9173
28576/60000 [=============>................] - ETA: 49s - loss: 0.2680 - categorical_accuracy: 0.9173
28608/60000 [=============>................] - ETA: 49s - loss: 0.2679 - categorical_accuracy: 0.9174
28672/60000 [=============>................] - ETA: 49s - loss: 0.2675 - categorical_accuracy: 0.9175
28736/60000 [=============>................] - ETA: 49s - loss: 0.2672 - categorical_accuracy: 0.9176
28768/60000 [=============>................] - ETA: 49s - loss: 0.2672 - categorical_accuracy: 0.9176
28800/60000 [=============>................] - ETA: 49s - loss: 0.2671 - categorical_accuracy: 0.9176
28832/60000 [=============>................] - ETA: 49s - loss: 0.2670 - categorical_accuracy: 0.9176
28864/60000 [=============>................] - ETA: 49s - loss: 0.2670 - categorical_accuracy: 0.9175
28896/60000 [=============>................] - ETA: 49s - loss: 0.2668 - categorical_accuracy: 0.9176
28960/60000 [=============>................] - ETA: 49s - loss: 0.2664 - categorical_accuracy: 0.9177
28992/60000 [=============>................] - ETA: 49s - loss: 0.2663 - categorical_accuracy: 0.9177
29056/60000 [=============>................] - ETA: 48s - loss: 0.2661 - categorical_accuracy: 0.9177
29120/60000 [=============>................] - ETA: 48s - loss: 0.2658 - categorical_accuracy: 0.9178
29184/60000 [=============>................] - ETA: 48s - loss: 0.2656 - categorical_accuracy: 0.9179
29248/60000 [=============>................] - ETA: 48s - loss: 0.2655 - categorical_accuracy: 0.9179
29280/60000 [=============>................] - ETA: 48s - loss: 0.2655 - categorical_accuracy: 0.9180
29312/60000 [=============>................] - ETA: 48s - loss: 0.2653 - categorical_accuracy: 0.9181
29344/60000 [=============>................] - ETA: 48s - loss: 0.2651 - categorical_accuracy: 0.9181
29376/60000 [=============>................] - ETA: 48s - loss: 0.2651 - categorical_accuracy: 0.9181
29408/60000 [=============>................] - ETA: 48s - loss: 0.2648 - categorical_accuracy: 0.9182
29440/60000 [=============>................] - ETA: 48s - loss: 0.2648 - categorical_accuracy: 0.9181
29472/60000 [=============>................] - ETA: 48s - loss: 0.2645 - categorical_accuracy: 0.9182
29536/60000 [=============>................] - ETA: 48s - loss: 0.2645 - categorical_accuracy: 0.9183
29568/60000 [=============>................] - ETA: 48s - loss: 0.2644 - categorical_accuracy: 0.9183
29600/60000 [=============>................] - ETA: 48s - loss: 0.2642 - categorical_accuracy: 0.9183
29632/60000 [=============>................] - ETA: 48s - loss: 0.2642 - categorical_accuracy: 0.9183
29664/60000 [=============>................] - ETA: 48s - loss: 0.2640 - categorical_accuracy: 0.9183
29728/60000 [=============>................] - ETA: 47s - loss: 0.2637 - categorical_accuracy: 0.9184
29760/60000 [=============>................] - ETA: 47s - loss: 0.2637 - categorical_accuracy: 0.9184
29792/60000 [=============>................] - ETA: 47s - loss: 0.2636 - categorical_accuracy: 0.9184
29856/60000 [=============>................] - ETA: 47s - loss: 0.2631 - categorical_accuracy: 0.9186
29920/60000 [=============>................] - ETA: 47s - loss: 0.2626 - categorical_accuracy: 0.9188
29952/60000 [=============>................] - ETA: 47s - loss: 0.2625 - categorical_accuracy: 0.9188
29984/60000 [=============>................] - ETA: 47s - loss: 0.2623 - categorical_accuracy: 0.9189
30016/60000 [==============>...............] - ETA: 47s - loss: 0.2622 - categorical_accuracy: 0.9189
30048/60000 [==============>...............] - ETA: 47s - loss: 0.2621 - categorical_accuracy: 0.9190
30080/60000 [==============>...............] - ETA: 47s - loss: 0.2620 - categorical_accuracy: 0.9190
30112/60000 [==============>...............] - ETA: 47s - loss: 0.2618 - categorical_accuracy: 0.9191
30144/60000 [==============>...............] - ETA: 47s - loss: 0.2616 - categorical_accuracy: 0.9191
30176/60000 [==============>...............] - ETA: 47s - loss: 0.2614 - categorical_accuracy: 0.9192
30240/60000 [==============>...............] - ETA: 47s - loss: 0.2609 - categorical_accuracy: 0.9193
30272/60000 [==============>...............] - ETA: 47s - loss: 0.2607 - categorical_accuracy: 0.9194
30304/60000 [==============>...............] - ETA: 47s - loss: 0.2607 - categorical_accuracy: 0.9194
30368/60000 [==============>...............] - ETA: 46s - loss: 0.2606 - categorical_accuracy: 0.9195
30400/60000 [==============>...............] - ETA: 46s - loss: 0.2606 - categorical_accuracy: 0.9195
30464/60000 [==============>...............] - ETA: 46s - loss: 0.2604 - categorical_accuracy: 0.9196
30528/60000 [==============>...............] - ETA: 46s - loss: 0.2601 - categorical_accuracy: 0.9197
30592/60000 [==============>...............] - ETA: 46s - loss: 0.2598 - categorical_accuracy: 0.9199
30624/60000 [==============>...............] - ETA: 46s - loss: 0.2597 - categorical_accuracy: 0.9199
30656/60000 [==============>...............] - ETA: 46s - loss: 0.2595 - categorical_accuracy: 0.9200
30688/60000 [==============>...............] - ETA: 46s - loss: 0.2595 - categorical_accuracy: 0.9201
30720/60000 [==============>...............] - ETA: 46s - loss: 0.2592 - categorical_accuracy: 0.9201
30752/60000 [==============>...............] - ETA: 46s - loss: 0.2590 - categorical_accuracy: 0.9202
30816/60000 [==============>...............] - ETA: 46s - loss: 0.2586 - categorical_accuracy: 0.9203
30880/60000 [==============>...............] - ETA: 46s - loss: 0.2587 - categorical_accuracy: 0.9202
30912/60000 [==============>...............] - ETA: 46s - loss: 0.2586 - categorical_accuracy: 0.9203
30976/60000 [==============>...............] - ETA: 45s - loss: 0.2584 - categorical_accuracy: 0.9204
31040/60000 [==============>...............] - ETA: 45s - loss: 0.2581 - categorical_accuracy: 0.9205
31104/60000 [==============>...............] - ETA: 45s - loss: 0.2577 - categorical_accuracy: 0.9206
31168/60000 [==============>...............] - ETA: 45s - loss: 0.2576 - categorical_accuracy: 0.9206
31232/60000 [==============>...............] - ETA: 45s - loss: 0.2573 - categorical_accuracy: 0.9207
31296/60000 [==============>...............] - ETA: 45s - loss: 0.2569 - categorical_accuracy: 0.9209
31360/60000 [==============>...............] - ETA: 45s - loss: 0.2569 - categorical_accuracy: 0.9209
31392/60000 [==============>...............] - ETA: 45s - loss: 0.2567 - categorical_accuracy: 0.9210
31456/60000 [==============>...............] - ETA: 45s - loss: 0.2563 - categorical_accuracy: 0.9211
31488/60000 [==============>...............] - ETA: 45s - loss: 0.2563 - categorical_accuracy: 0.9211
31520/60000 [==============>...............] - ETA: 45s - loss: 0.2561 - categorical_accuracy: 0.9212
31584/60000 [==============>...............] - ETA: 44s - loss: 0.2556 - categorical_accuracy: 0.9213
31648/60000 [==============>...............] - ETA: 44s - loss: 0.2553 - categorical_accuracy: 0.9214
31712/60000 [==============>...............] - ETA: 44s - loss: 0.2549 - categorical_accuracy: 0.9216
31744/60000 [==============>...............] - ETA: 44s - loss: 0.2548 - categorical_accuracy: 0.9216
31776/60000 [==============>...............] - ETA: 44s - loss: 0.2547 - categorical_accuracy: 0.9216
31808/60000 [==============>...............] - ETA: 44s - loss: 0.2545 - categorical_accuracy: 0.9217
31840/60000 [==============>...............] - ETA: 44s - loss: 0.2543 - categorical_accuracy: 0.9217
31872/60000 [==============>...............] - ETA: 44s - loss: 0.2541 - categorical_accuracy: 0.9218
31936/60000 [==============>...............] - ETA: 44s - loss: 0.2537 - categorical_accuracy: 0.9219
31968/60000 [==============>...............] - ETA: 44s - loss: 0.2535 - categorical_accuracy: 0.9220
32000/60000 [===============>..............] - ETA: 44s - loss: 0.2533 - categorical_accuracy: 0.9221
32032/60000 [===============>..............] - ETA: 44s - loss: 0.2531 - categorical_accuracy: 0.9221
32064/60000 [===============>..............] - ETA: 44s - loss: 0.2530 - categorical_accuracy: 0.9222
32096/60000 [===============>..............] - ETA: 44s - loss: 0.2527 - categorical_accuracy: 0.9222
32160/60000 [===============>..............] - ETA: 44s - loss: 0.2525 - categorical_accuracy: 0.9223
32224/60000 [===============>..............] - ETA: 43s - loss: 0.2521 - categorical_accuracy: 0.9224
32256/60000 [===============>..............] - ETA: 43s - loss: 0.2520 - categorical_accuracy: 0.9225
32320/60000 [===============>..............] - ETA: 43s - loss: 0.2516 - categorical_accuracy: 0.9226
32384/60000 [===============>..............] - ETA: 43s - loss: 0.2516 - categorical_accuracy: 0.9227
32448/60000 [===============>..............] - ETA: 43s - loss: 0.2512 - categorical_accuracy: 0.9228
32480/60000 [===============>..............] - ETA: 43s - loss: 0.2511 - categorical_accuracy: 0.9228
32512/60000 [===============>..............] - ETA: 43s - loss: 0.2509 - categorical_accuracy: 0.9229
32576/60000 [===============>..............] - ETA: 43s - loss: 0.2506 - categorical_accuracy: 0.9230
32608/60000 [===============>..............] - ETA: 43s - loss: 0.2504 - categorical_accuracy: 0.9231
32672/60000 [===============>..............] - ETA: 43s - loss: 0.2500 - categorical_accuracy: 0.9232
32736/60000 [===============>..............] - ETA: 43s - loss: 0.2498 - categorical_accuracy: 0.9233
32768/60000 [===============>..............] - ETA: 43s - loss: 0.2496 - categorical_accuracy: 0.9233
32800/60000 [===============>..............] - ETA: 43s - loss: 0.2495 - categorical_accuracy: 0.9234
32832/60000 [===============>..............] - ETA: 42s - loss: 0.2496 - categorical_accuracy: 0.9234
32896/60000 [===============>..............] - ETA: 42s - loss: 0.2497 - categorical_accuracy: 0.9234
32960/60000 [===============>..............] - ETA: 42s - loss: 0.2494 - categorical_accuracy: 0.9234
32992/60000 [===============>..............] - ETA: 42s - loss: 0.2492 - categorical_accuracy: 0.9235
33056/60000 [===============>..............] - ETA: 42s - loss: 0.2488 - categorical_accuracy: 0.9236
33120/60000 [===============>..............] - ETA: 42s - loss: 0.2485 - categorical_accuracy: 0.9237
33184/60000 [===============>..............] - ETA: 42s - loss: 0.2484 - categorical_accuracy: 0.9238
33216/60000 [===============>..............] - ETA: 42s - loss: 0.2482 - categorical_accuracy: 0.9238
33248/60000 [===============>..............] - ETA: 42s - loss: 0.2480 - categorical_accuracy: 0.9238
33280/60000 [===============>..............] - ETA: 42s - loss: 0.2479 - categorical_accuracy: 0.9239
33312/60000 [===============>..............] - ETA: 42s - loss: 0.2476 - categorical_accuracy: 0.9240
33344/60000 [===============>..............] - ETA: 42s - loss: 0.2475 - categorical_accuracy: 0.9240
33376/60000 [===============>..............] - ETA: 42s - loss: 0.2474 - categorical_accuracy: 0.9240
33408/60000 [===============>..............] - ETA: 42s - loss: 0.2472 - categorical_accuracy: 0.9241
33440/60000 [===============>..............] - ETA: 42s - loss: 0.2470 - categorical_accuracy: 0.9241
33472/60000 [===============>..............] - ETA: 41s - loss: 0.2468 - categorical_accuracy: 0.9242
33536/60000 [===============>..............] - ETA: 41s - loss: 0.2467 - categorical_accuracy: 0.9242
33600/60000 [===============>..............] - ETA: 41s - loss: 0.2465 - categorical_accuracy: 0.9243
33632/60000 [===============>..............] - ETA: 41s - loss: 0.2464 - categorical_accuracy: 0.9243
33696/60000 [===============>..............] - ETA: 41s - loss: 0.2463 - categorical_accuracy: 0.9244
33760/60000 [===============>..............] - ETA: 41s - loss: 0.2460 - categorical_accuracy: 0.9245
33792/60000 [===============>..............] - ETA: 41s - loss: 0.2459 - categorical_accuracy: 0.9245
33824/60000 [===============>..............] - ETA: 41s - loss: 0.2457 - categorical_accuracy: 0.9245
33856/60000 [===============>..............] - ETA: 41s - loss: 0.2456 - categorical_accuracy: 0.9245
33888/60000 [===============>..............] - ETA: 41s - loss: 0.2455 - categorical_accuracy: 0.9245
33920/60000 [===============>..............] - ETA: 41s - loss: 0.2453 - categorical_accuracy: 0.9246
33952/60000 [===============>..............] - ETA: 41s - loss: 0.2451 - categorical_accuracy: 0.9247
33984/60000 [===============>..............] - ETA: 41s - loss: 0.2452 - categorical_accuracy: 0.9247
34048/60000 [================>.............] - ETA: 41s - loss: 0.2448 - categorical_accuracy: 0.9248
34080/60000 [================>.............] - ETA: 41s - loss: 0.2446 - categorical_accuracy: 0.9248
34144/60000 [================>.............] - ETA: 40s - loss: 0.2444 - categorical_accuracy: 0.9248
34208/60000 [================>.............] - ETA: 40s - loss: 0.2444 - categorical_accuracy: 0.9249
34272/60000 [================>.............] - ETA: 40s - loss: 0.2440 - categorical_accuracy: 0.9251
34304/60000 [================>.............] - ETA: 40s - loss: 0.2440 - categorical_accuracy: 0.9251
34336/60000 [================>.............] - ETA: 40s - loss: 0.2439 - categorical_accuracy: 0.9252
34400/60000 [================>.............] - ETA: 40s - loss: 0.2437 - categorical_accuracy: 0.9252
34464/60000 [================>.............] - ETA: 40s - loss: 0.2434 - categorical_accuracy: 0.9253
34528/60000 [================>.............] - ETA: 40s - loss: 0.2431 - categorical_accuracy: 0.9254
34560/60000 [================>.............] - ETA: 40s - loss: 0.2431 - categorical_accuracy: 0.9254
34592/60000 [================>.............] - ETA: 40s - loss: 0.2429 - categorical_accuracy: 0.9254
34656/60000 [================>.............] - ETA: 40s - loss: 0.2425 - categorical_accuracy: 0.9256
34720/60000 [================>.............] - ETA: 39s - loss: 0.2421 - categorical_accuracy: 0.9257
34784/60000 [================>.............] - ETA: 39s - loss: 0.2420 - categorical_accuracy: 0.9257
34848/60000 [================>.............] - ETA: 39s - loss: 0.2416 - categorical_accuracy: 0.9258
34880/60000 [================>.............] - ETA: 39s - loss: 0.2417 - categorical_accuracy: 0.9259
34912/60000 [================>.............] - ETA: 39s - loss: 0.2417 - categorical_accuracy: 0.9259
34944/60000 [================>.............] - ETA: 39s - loss: 0.2417 - categorical_accuracy: 0.9259
34976/60000 [================>.............] - ETA: 39s - loss: 0.2415 - categorical_accuracy: 0.9259
35008/60000 [================>.............] - ETA: 39s - loss: 0.2413 - categorical_accuracy: 0.9260
35072/60000 [================>.............] - ETA: 39s - loss: 0.2410 - categorical_accuracy: 0.9261
35104/60000 [================>.............] - ETA: 39s - loss: 0.2408 - categorical_accuracy: 0.9261
35136/60000 [================>.............] - ETA: 39s - loss: 0.2407 - categorical_accuracy: 0.9262
35168/60000 [================>.............] - ETA: 39s - loss: 0.2405 - categorical_accuracy: 0.9262
35200/60000 [================>.............] - ETA: 39s - loss: 0.2405 - categorical_accuracy: 0.9262
35232/60000 [================>.............] - ETA: 39s - loss: 0.2403 - categorical_accuracy: 0.9263
35264/60000 [================>.............] - ETA: 39s - loss: 0.2404 - categorical_accuracy: 0.9263
35328/60000 [================>.............] - ETA: 39s - loss: 0.2401 - categorical_accuracy: 0.9264
35392/60000 [================>.............] - ETA: 38s - loss: 0.2400 - categorical_accuracy: 0.9264
35456/60000 [================>.............] - ETA: 38s - loss: 0.2399 - categorical_accuracy: 0.9264
35520/60000 [================>.............] - ETA: 38s - loss: 0.2395 - categorical_accuracy: 0.9265
35552/60000 [================>.............] - ETA: 38s - loss: 0.2395 - categorical_accuracy: 0.9266
35616/60000 [================>.............] - ETA: 38s - loss: 0.2391 - categorical_accuracy: 0.9267
35648/60000 [================>.............] - ETA: 38s - loss: 0.2390 - categorical_accuracy: 0.9267
35712/60000 [================>.............] - ETA: 38s - loss: 0.2388 - categorical_accuracy: 0.9268
35744/60000 [================>.............] - ETA: 38s - loss: 0.2387 - categorical_accuracy: 0.9268
35776/60000 [================>.............] - ETA: 38s - loss: 0.2386 - categorical_accuracy: 0.9268
35808/60000 [================>.............] - ETA: 38s - loss: 0.2384 - categorical_accuracy: 0.9269
35840/60000 [================>.............] - ETA: 38s - loss: 0.2383 - categorical_accuracy: 0.9269
35872/60000 [================>.............] - ETA: 38s - loss: 0.2381 - categorical_accuracy: 0.9269
35936/60000 [================>.............] - ETA: 38s - loss: 0.2377 - categorical_accuracy: 0.9271
35968/60000 [================>.............] - ETA: 38s - loss: 0.2376 - categorical_accuracy: 0.9271
36000/60000 [=================>............] - ETA: 37s - loss: 0.2375 - categorical_accuracy: 0.9271
36064/60000 [=================>............] - ETA: 37s - loss: 0.2371 - categorical_accuracy: 0.9272
36096/60000 [=================>............] - ETA: 37s - loss: 0.2371 - categorical_accuracy: 0.9272
36128/60000 [=================>............] - ETA: 37s - loss: 0.2370 - categorical_accuracy: 0.9273
36160/60000 [=================>............] - ETA: 37s - loss: 0.2368 - categorical_accuracy: 0.9274
36192/60000 [=================>............] - ETA: 37s - loss: 0.2366 - categorical_accuracy: 0.9274
36256/60000 [=================>............] - ETA: 37s - loss: 0.2364 - categorical_accuracy: 0.9275
36320/60000 [=================>............] - ETA: 37s - loss: 0.2365 - categorical_accuracy: 0.9275
36384/60000 [=================>............] - ETA: 37s - loss: 0.2362 - categorical_accuracy: 0.9276
36416/60000 [=================>............] - ETA: 37s - loss: 0.2364 - categorical_accuracy: 0.9275
36480/60000 [=================>............] - ETA: 37s - loss: 0.2361 - categorical_accuracy: 0.9276
36544/60000 [=================>............] - ETA: 37s - loss: 0.2358 - categorical_accuracy: 0.9276
36608/60000 [=================>............] - ETA: 36s - loss: 0.2356 - categorical_accuracy: 0.9277
36640/60000 [=================>............] - ETA: 36s - loss: 0.2358 - categorical_accuracy: 0.9277
36672/60000 [=================>............] - ETA: 36s - loss: 0.2357 - categorical_accuracy: 0.9277
36704/60000 [=================>............] - ETA: 36s - loss: 0.2355 - categorical_accuracy: 0.9278
36736/60000 [=================>............] - ETA: 36s - loss: 0.2354 - categorical_accuracy: 0.9278
36768/60000 [=================>............] - ETA: 36s - loss: 0.2353 - categorical_accuracy: 0.9279
36800/60000 [=================>............] - ETA: 36s - loss: 0.2351 - categorical_accuracy: 0.9279
36832/60000 [=================>............] - ETA: 36s - loss: 0.2349 - categorical_accuracy: 0.9280
36864/60000 [=================>............] - ETA: 36s - loss: 0.2348 - categorical_accuracy: 0.9280
36896/60000 [=================>............] - ETA: 36s - loss: 0.2347 - categorical_accuracy: 0.9281
36960/60000 [=================>............] - ETA: 36s - loss: 0.2343 - categorical_accuracy: 0.9282
36992/60000 [=================>............] - ETA: 36s - loss: 0.2342 - categorical_accuracy: 0.9282
37024/60000 [=================>............] - ETA: 36s - loss: 0.2341 - categorical_accuracy: 0.9282
37056/60000 [=================>............] - ETA: 36s - loss: 0.2340 - categorical_accuracy: 0.9282
37120/60000 [=================>............] - ETA: 36s - loss: 0.2338 - categorical_accuracy: 0.9283
37184/60000 [=================>............] - ETA: 36s - loss: 0.2337 - categorical_accuracy: 0.9283
37248/60000 [=================>............] - ETA: 35s - loss: 0.2334 - categorical_accuracy: 0.9284
37280/60000 [=================>............] - ETA: 35s - loss: 0.2335 - categorical_accuracy: 0.9284
37344/60000 [=================>............] - ETA: 35s - loss: 0.2332 - categorical_accuracy: 0.9284
37376/60000 [=================>............] - ETA: 35s - loss: 0.2331 - categorical_accuracy: 0.9285
37440/60000 [=================>............] - ETA: 35s - loss: 0.2328 - categorical_accuracy: 0.9286
37504/60000 [=================>............] - ETA: 35s - loss: 0.2327 - categorical_accuracy: 0.9286
37568/60000 [=================>............] - ETA: 35s - loss: 0.2326 - categorical_accuracy: 0.9287
37600/60000 [=================>............] - ETA: 35s - loss: 0.2324 - categorical_accuracy: 0.9287
37632/60000 [=================>............] - ETA: 35s - loss: 0.2324 - categorical_accuracy: 0.9288
37664/60000 [=================>............] - ETA: 35s - loss: 0.2322 - categorical_accuracy: 0.9288
37728/60000 [=================>............] - ETA: 35s - loss: 0.2319 - categorical_accuracy: 0.9289
37760/60000 [=================>............] - ETA: 35s - loss: 0.2317 - categorical_accuracy: 0.9290
37792/60000 [=================>............] - ETA: 35s - loss: 0.2316 - categorical_accuracy: 0.9290
37856/60000 [=================>............] - ETA: 35s - loss: 0.2314 - categorical_accuracy: 0.9290
37888/60000 [=================>............] - ETA: 34s - loss: 0.2315 - categorical_accuracy: 0.9290
37920/60000 [=================>............] - ETA: 34s - loss: 0.2316 - categorical_accuracy: 0.9290
37984/60000 [=================>............] - ETA: 34s - loss: 0.2312 - categorical_accuracy: 0.9292
38016/60000 [==================>...........] - ETA: 34s - loss: 0.2311 - categorical_accuracy: 0.9292
38080/60000 [==================>...........] - ETA: 34s - loss: 0.2309 - categorical_accuracy: 0.9293
38112/60000 [==================>...........] - ETA: 34s - loss: 0.2309 - categorical_accuracy: 0.9292
38176/60000 [==================>...........] - ETA: 34s - loss: 0.2306 - categorical_accuracy: 0.9293
38208/60000 [==================>...........] - ETA: 34s - loss: 0.2306 - categorical_accuracy: 0.9293
38240/60000 [==================>...........] - ETA: 34s - loss: 0.2304 - categorical_accuracy: 0.9294
38272/60000 [==================>...........] - ETA: 34s - loss: 0.2303 - categorical_accuracy: 0.9294
38336/60000 [==================>...........] - ETA: 34s - loss: 0.2302 - categorical_accuracy: 0.9294
38368/60000 [==================>...........] - ETA: 34s - loss: 0.2301 - categorical_accuracy: 0.9294
38432/60000 [==================>...........] - ETA: 34s - loss: 0.2299 - categorical_accuracy: 0.9295
38464/60000 [==================>...........] - ETA: 34s - loss: 0.2298 - categorical_accuracy: 0.9295
38496/60000 [==================>...........] - ETA: 34s - loss: 0.2299 - categorical_accuracy: 0.9296
38528/60000 [==================>...........] - ETA: 33s - loss: 0.2299 - categorical_accuracy: 0.9295
38592/60000 [==================>...........] - ETA: 33s - loss: 0.2297 - categorical_accuracy: 0.9296
38656/60000 [==================>...........] - ETA: 33s - loss: 0.2294 - categorical_accuracy: 0.9297
38688/60000 [==================>...........] - ETA: 33s - loss: 0.2294 - categorical_accuracy: 0.9297
38752/60000 [==================>...........] - ETA: 33s - loss: 0.2291 - categorical_accuracy: 0.9298
38784/60000 [==================>...........] - ETA: 33s - loss: 0.2289 - categorical_accuracy: 0.9298
38816/60000 [==================>...........] - ETA: 33s - loss: 0.2289 - categorical_accuracy: 0.9298
38880/60000 [==================>...........] - ETA: 33s - loss: 0.2289 - categorical_accuracy: 0.9299
38944/60000 [==================>...........] - ETA: 33s - loss: 0.2287 - categorical_accuracy: 0.9299
39008/60000 [==================>...........] - ETA: 33s - loss: 0.2284 - categorical_accuracy: 0.9300
39072/60000 [==================>...........] - ETA: 33s - loss: 0.2283 - categorical_accuracy: 0.9301
39104/60000 [==================>...........] - ETA: 33s - loss: 0.2283 - categorical_accuracy: 0.9301
39136/60000 [==================>...........] - ETA: 33s - loss: 0.2281 - categorical_accuracy: 0.9301
39200/60000 [==================>...........] - ETA: 32s - loss: 0.2279 - categorical_accuracy: 0.9302
39264/60000 [==================>...........] - ETA: 32s - loss: 0.2279 - categorical_accuracy: 0.9302
39328/60000 [==================>...........] - ETA: 32s - loss: 0.2276 - categorical_accuracy: 0.9303
39392/60000 [==================>...........] - ETA: 32s - loss: 0.2274 - categorical_accuracy: 0.9304
39424/60000 [==================>...........] - ETA: 32s - loss: 0.2273 - categorical_accuracy: 0.9304
39456/60000 [==================>...........] - ETA: 32s - loss: 0.2272 - categorical_accuracy: 0.9305
39488/60000 [==================>...........] - ETA: 32s - loss: 0.2270 - categorical_accuracy: 0.9305
39520/60000 [==================>...........] - ETA: 32s - loss: 0.2269 - categorical_accuracy: 0.9305
39584/60000 [==================>...........] - ETA: 32s - loss: 0.2266 - categorical_accuracy: 0.9307
39648/60000 [==================>...........] - ETA: 32s - loss: 0.2264 - categorical_accuracy: 0.9307
39680/60000 [==================>...........] - ETA: 32s - loss: 0.2263 - categorical_accuracy: 0.9308
39712/60000 [==================>...........] - ETA: 32s - loss: 0.2261 - categorical_accuracy: 0.9308
39744/60000 [==================>...........] - ETA: 32s - loss: 0.2261 - categorical_accuracy: 0.9308
39776/60000 [==================>...........] - ETA: 31s - loss: 0.2260 - categorical_accuracy: 0.9308
39840/60000 [==================>...........] - ETA: 31s - loss: 0.2257 - categorical_accuracy: 0.9309
39904/60000 [==================>...........] - ETA: 31s - loss: 0.2254 - categorical_accuracy: 0.9310
39936/60000 [==================>...........] - ETA: 31s - loss: 0.2253 - categorical_accuracy: 0.9310
39968/60000 [==================>...........] - ETA: 31s - loss: 0.2252 - categorical_accuracy: 0.9310
40000/60000 [===================>..........] - ETA: 31s - loss: 0.2251 - categorical_accuracy: 0.9310
40032/60000 [===================>..........] - ETA: 31s - loss: 0.2251 - categorical_accuracy: 0.9311
40064/60000 [===================>..........] - ETA: 31s - loss: 0.2249 - categorical_accuracy: 0.9311
40096/60000 [===================>..........] - ETA: 31s - loss: 0.2248 - categorical_accuracy: 0.9311
40128/60000 [===================>..........] - ETA: 31s - loss: 0.2249 - categorical_accuracy: 0.9311
40160/60000 [===================>..........] - ETA: 31s - loss: 0.2248 - categorical_accuracy: 0.9312
40224/60000 [===================>..........] - ETA: 31s - loss: 0.2249 - categorical_accuracy: 0.9312
40288/60000 [===================>..........] - ETA: 31s - loss: 0.2247 - categorical_accuracy: 0.9313
40352/60000 [===================>..........] - ETA: 31s - loss: 0.2244 - categorical_accuracy: 0.9314
40384/60000 [===================>..........] - ETA: 31s - loss: 0.2243 - categorical_accuracy: 0.9314
40416/60000 [===================>..........] - ETA: 30s - loss: 0.2241 - categorical_accuracy: 0.9314
40480/60000 [===================>..........] - ETA: 30s - loss: 0.2238 - categorical_accuracy: 0.9315
40512/60000 [===================>..........] - ETA: 30s - loss: 0.2238 - categorical_accuracy: 0.9316
40576/60000 [===================>..........] - ETA: 30s - loss: 0.2236 - categorical_accuracy: 0.9316
40640/60000 [===================>..........] - ETA: 30s - loss: 0.2234 - categorical_accuracy: 0.9317
40704/60000 [===================>..........] - ETA: 30s - loss: 0.2232 - categorical_accuracy: 0.9317
40736/60000 [===================>..........] - ETA: 30s - loss: 0.2230 - categorical_accuracy: 0.9318
40800/60000 [===================>..........] - ETA: 30s - loss: 0.2229 - categorical_accuracy: 0.9318
40864/60000 [===================>..........] - ETA: 30s - loss: 0.2226 - categorical_accuracy: 0.9319
40928/60000 [===================>..........] - ETA: 30s - loss: 0.2223 - categorical_accuracy: 0.9321
40992/60000 [===================>..........] - ETA: 30s - loss: 0.2220 - categorical_accuracy: 0.9321
41056/60000 [===================>..........] - ETA: 29s - loss: 0.2218 - categorical_accuracy: 0.9322
41120/60000 [===================>..........] - ETA: 29s - loss: 0.2215 - categorical_accuracy: 0.9323
41184/60000 [===================>..........] - ETA: 29s - loss: 0.2214 - categorical_accuracy: 0.9324
41216/60000 [===================>..........] - ETA: 29s - loss: 0.2212 - categorical_accuracy: 0.9324
41248/60000 [===================>..........] - ETA: 29s - loss: 0.2212 - categorical_accuracy: 0.9324
41280/60000 [===================>..........] - ETA: 29s - loss: 0.2211 - categorical_accuracy: 0.9324
41344/60000 [===================>..........] - ETA: 29s - loss: 0.2210 - categorical_accuracy: 0.9325
41376/60000 [===================>..........] - ETA: 29s - loss: 0.2210 - categorical_accuracy: 0.9325
41408/60000 [===================>..........] - ETA: 29s - loss: 0.2208 - categorical_accuracy: 0.9325
41472/60000 [===================>..........] - ETA: 29s - loss: 0.2205 - categorical_accuracy: 0.9326
41536/60000 [===================>..........] - ETA: 29s - loss: 0.2203 - categorical_accuracy: 0.9327
41600/60000 [===================>..........] - ETA: 29s - loss: 0.2200 - categorical_accuracy: 0.9328
41664/60000 [===================>..........] - ETA: 28s - loss: 0.2198 - categorical_accuracy: 0.9329
41728/60000 [===================>..........] - ETA: 28s - loss: 0.2195 - categorical_accuracy: 0.9329
41792/60000 [===================>..........] - ETA: 28s - loss: 0.2193 - categorical_accuracy: 0.9330
41856/60000 [===================>..........] - ETA: 28s - loss: 0.2192 - categorical_accuracy: 0.9331
41920/60000 [===================>..........] - ETA: 28s - loss: 0.2189 - categorical_accuracy: 0.9331
41984/60000 [===================>..........] - ETA: 28s - loss: 0.2188 - categorical_accuracy: 0.9332
42016/60000 [====================>.........] - ETA: 28s - loss: 0.2189 - categorical_accuracy: 0.9331
42048/60000 [====================>.........] - ETA: 28s - loss: 0.2187 - categorical_accuracy: 0.9332
42112/60000 [====================>.........] - ETA: 28s - loss: 0.2185 - categorical_accuracy: 0.9333
42176/60000 [====================>.........] - ETA: 28s - loss: 0.2182 - categorical_accuracy: 0.9334
42208/60000 [====================>.........] - ETA: 28s - loss: 0.2180 - categorical_accuracy: 0.9334
42240/60000 [====================>.........] - ETA: 28s - loss: 0.2179 - categorical_accuracy: 0.9335
42272/60000 [====================>.........] - ETA: 28s - loss: 0.2179 - categorical_accuracy: 0.9335
42304/60000 [====================>.........] - ETA: 27s - loss: 0.2177 - categorical_accuracy: 0.9336
42336/60000 [====================>.........] - ETA: 27s - loss: 0.2176 - categorical_accuracy: 0.9336
42400/60000 [====================>.........] - ETA: 27s - loss: 0.2175 - categorical_accuracy: 0.9337
42464/60000 [====================>.........] - ETA: 27s - loss: 0.2172 - categorical_accuracy: 0.9337
42528/60000 [====================>.........] - ETA: 27s - loss: 0.2169 - categorical_accuracy: 0.9338
42592/60000 [====================>.........] - ETA: 27s - loss: 0.2168 - categorical_accuracy: 0.9338
42656/60000 [====================>.........] - ETA: 27s - loss: 0.2166 - categorical_accuracy: 0.9338
42720/60000 [====================>.........] - ETA: 27s - loss: 0.2164 - categorical_accuracy: 0.9339
42752/60000 [====================>.........] - ETA: 27s - loss: 0.2163 - categorical_accuracy: 0.9339
42816/60000 [====================>.........] - ETA: 27s - loss: 0.2160 - categorical_accuracy: 0.9340
42880/60000 [====================>.........] - ETA: 27s - loss: 0.2158 - categorical_accuracy: 0.9341
42912/60000 [====================>.........] - ETA: 27s - loss: 0.2157 - categorical_accuracy: 0.9341
42944/60000 [====================>.........] - ETA: 26s - loss: 0.2156 - categorical_accuracy: 0.9341
42976/60000 [====================>.........] - ETA: 26s - loss: 0.2155 - categorical_accuracy: 0.9342
43008/60000 [====================>.........] - ETA: 26s - loss: 0.2153 - categorical_accuracy: 0.9342
43040/60000 [====================>.........] - ETA: 26s - loss: 0.2152 - categorical_accuracy: 0.9343
43072/60000 [====================>.........] - ETA: 26s - loss: 0.2151 - categorical_accuracy: 0.9343
43104/60000 [====================>.........] - ETA: 26s - loss: 0.2154 - categorical_accuracy: 0.9343
43136/60000 [====================>.........] - ETA: 26s - loss: 0.2153 - categorical_accuracy: 0.9343
43168/60000 [====================>.........] - ETA: 26s - loss: 0.2152 - categorical_accuracy: 0.9343
43200/60000 [====================>.........] - ETA: 26s - loss: 0.2151 - categorical_accuracy: 0.9343
43232/60000 [====================>.........] - ETA: 26s - loss: 0.2150 - categorical_accuracy: 0.9343
43264/60000 [====================>.........] - ETA: 26s - loss: 0.2148 - categorical_accuracy: 0.9344
43296/60000 [====================>.........] - ETA: 26s - loss: 0.2149 - categorical_accuracy: 0.9343
43328/60000 [====================>.........] - ETA: 26s - loss: 0.2148 - categorical_accuracy: 0.9344
43360/60000 [====================>.........] - ETA: 26s - loss: 0.2147 - categorical_accuracy: 0.9344
43392/60000 [====================>.........] - ETA: 26s - loss: 0.2148 - categorical_accuracy: 0.9344
43424/60000 [====================>.........] - ETA: 26s - loss: 0.2148 - categorical_accuracy: 0.9344
43456/60000 [====================>.........] - ETA: 26s - loss: 0.2147 - categorical_accuracy: 0.9345
43488/60000 [====================>.........] - ETA: 26s - loss: 0.2147 - categorical_accuracy: 0.9344
43520/60000 [====================>.........] - ETA: 26s - loss: 0.2146 - categorical_accuracy: 0.9344
43552/60000 [====================>.........] - ETA: 26s - loss: 0.2146 - categorical_accuracy: 0.9344
43584/60000 [====================>.........] - ETA: 25s - loss: 0.2146 - categorical_accuracy: 0.9344
43616/60000 [====================>.........] - ETA: 25s - loss: 0.2145 - categorical_accuracy: 0.9345
43648/60000 [====================>.........] - ETA: 25s - loss: 0.2145 - categorical_accuracy: 0.9345
43680/60000 [====================>.........] - ETA: 25s - loss: 0.2143 - categorical_accuracy: 0.9345
43712/60000 [====================>.........] - ETA: 25s - loss: 0.2142 - categorical_accuracy: 0.9346
43744/60000 [====================>.........] - ETA: 25s - loss: 0.2142 - categorical_accuracy: 0.9346
43776/60000 [====================>.........] - ETA: 25s - loss: 0.2141 - categorical_accuracy: 0.9346
43808/60000 [====================>.........] - ETA: 25s - loss: 0.2140 - categorical_accuracy: 0.9346
43840/60000 [====================>.........] - ETA: 25s - loss: 0.2139 - categorical_accuracy: 0.9346
43872/60000 [====================>.........] - ETA: 25s - loss: 0.2138 - categorical_accuracy: 0.9347
43936/60000 [====================>.........] - ETA: 25s - loss: 0.2135 - categorical_accuracy: 0.9347
44000/60000 [=====================>........] - ETA: 25s - loss: 0.2134 - categorical_accuracy: 0.9348
44064/60000 [=====================>........] - ETA: 25s - loss: 0.2133 - categorical_accuracy: 0.9348
44096/60000 [=====================>........] - ETA: 25s - loss: 0.2133 - categorical_accuracy: 0.9348
44160/60000 [=====================>........] - ETA: 25s - loss: 0.2131 - categorical_accuracy: 0.9349
44224/60000 [=====================>........] - ETA: 24s - loss: 0.2130 - categorical_accuracy: 0.9349
44256/60000 [=====================>........] - ETA: 24s - loss: 0.2129 - categorical_accuracy: 0.9349
44288/60000 [=====================>........] - ETA: 24s - loss: 0.2128 - categorical_accuracy: 0.9350
44320/60000 [=====================>........] - ETA: 24s - loss: 0.2128 - categorical_accuracy: 0.9350
44352/60000 [=====================>........] - ETA: 24s - loss: 0.2127 - categorical_accuracy: 0.9350
44384/60000 [=====================>........] - ETA: 24s - loss: 0.2126 - categorical_accuracy: 0.9350
44416/60000 [=====================>........] - ETA: 24s - loss: 0.2124 - categorical_accuracy: 0.9350
44480/60000 [=====================>........] - ETA: 24s - loss: 0.2125 - categorical_accuracy: 0.9350
44544/60000 [=====================>........] - ETA: 24s - loss: 0.2124 - categorical_accuracy: 0.9351
44576/60000 [=====================>........] - ETA: 24s - loss: 0.2123 - categorical_accuracy: 0.9351
44608/60000 [=====================>........] - ETA: 24s - loss: 0.2125 - categorical_accuracy: 0.9351
44672/60000 [=====================>........] - ETA: 24s - loss: 0.2125 - categorical_accuracy: 0.9351
44704/60000 [=====================>........] - ETA: 24s - loss: 0.2123 - categorical_accuracy: 0.9351
44768/60000 [=====================>........] - ETA: 24s - loss: 0.2122 - categorical_accuracy: 0.9352
44832/60000 [=====================>........] - ETA: 23s - loss: 0.2120 - categorical_accuracy: 0.9352
44896/60000 [=====================>........] - ETA: 23s - loss: 0.2119 - categorical_accuracy: 0.9353
44960/60000 [=====================>........] - ETA: 23s - loss: 0.2117 - categorical_accuracy: 0.9353
45024/60000 [=====================>........] - ETA: 23s - loss: 0.2114 - categorical_accuracy: 0.9354
45088/60000 [=====================>........] - ETA: 23s - loss: 0.2112 - categorical_accuracy: 0.9355
45152/60000 [=====================>........] - ETA: 23s - loss: 0.2110 - categorical_accuracy: 0.9356
45216/60000 [=====================>........] - ETA: 23s - loss: 0.2107 - categorical_accuracy: 0.9356
45280/60000 [=====================>........] - ETA: 23s - loss: 0.2105 - categorical_accuracy: 0.9357
45344/60000 [=====================>........] - ETA: 23s - loss: 0.2102 - categorical_accuracy: 0.9358
45376/60000 [=====================>........] - ETA: 23s - loss: 0.2101 - categorical_accuracy: 0.9358
45408/60000 [=====================>........] - ETA: 23s - loss: 0.2099 - categorical_accuracy: 0.9359
45440/60000 [=====================>........] - ETA: 23s - loss: 0.2098 - categorical_accuracy: 0.9359
45472/60000 [=====================>........] - ETA: 22s - loss: 0.2097 - categorical_accuracy: 0.9359
45504/60000 [=====================>........] - ETA: 22s - loss: 0.2096 - categorical_accuracy: 0.9359
45536/60000 [=====================>........] - ETA: 22s - loss: 0.2096 - categorical_accuracy: 0.9360
45600/60000 [=====================>........] - ETA: 22s - loss: 0.2095 - categorical_accuracy: 0.9360
45664/60000 [=====================>........] - ETA: 22s - loss: 0.2095 - categorical_accuracy: 0.9360
45696/60000 [=====================>........] - ETA: 22s - loss: 0.2094 - categorical_accuracy: 0.9360
45760/60000 [=====================>........] - ETA: 22s - loss: 0.2093 - categorical_accuracy: 0.9360
45824/60000 [=====================>........] - ETA: 22s - loss: 0.2091 - categorical_accuracy: 0.9361
45856/60000 [=====================>........] - ETA: 22s - loss: 0.2089 - categorical_accuracy: 0.9361
45920/60000 [=====================>........] - ETA: 22s - loss: 0.2087 - categorical_accuracy: 0.9362
45984/60000 [=====================>........] - ETA: 22s - loss: 0.2085 - categorical_accuracy: 0.9363
46048/60000 [======================>.......] - ETA: 22s - loss: 0.2082 - categorical_accuracy: 0.9363
46112/60000 [======================>.......] - ETA: 21s - loss: 0.2083 - categorical_accuracy: 0.9364
46144/60000 [======================>.......] - ETA: 21s - loss: 0.2083 - categorical_accuracy: 0.9364
46208/60000 [======================>.......] - ETA: 21s - loss: 0.2080 - categorical_accuracy: 0.9364
46272/60000 [======================>.......] - ETA: 21s - loss: 0.2079 - categorical_accuracy: 0.9365
46336/60000 [======================>.......] - ETA: 21s - loss: 0.2076 - categorical_accuracy: 0.9366
46368/60000 [======================>.......] - ETA: 21s - loss: 0.2075 - categorical_accuracy: 0.9366
46432/60000 [======================>.......] - ETA: 21s - loss: 0.2073 - categorical_accuracy: 0.9366
46464/60000 [======================>.......] - ETA: 21s - loss: 0.2073 - categorical_accuracy: 0.9366
46496/60000 [======================>.......] - ETA: 21s - loss: 0.2072 - categorical_accuracy: 0.9367
46528/60000 [======================>.......] - ETA: 21s - loss: 0.2071 - categorical_accuracy: 0.9367
46592/60000 [======================>.......] - ETA: 21s - loss: 0.2070 - categorical_accuracy: 0.9367
46624/60000 [======================>.......] - ETA: 21s - loss: 0.2069 - categorical_accuracy: 0.9368
46656/60000 [======================>.......] - ETA: 21s - loss: 0.2068 - categorical_accuracy: 0.9368
46688/60000 [======================>.......] - ETA: 21s - loss: 0.2067 - categorical_accuracy: 0.9368
46720/60000 [======================>.......] - ETA: 20s - loss: 0.2066 - categorical_accuracy: 0.9369
46752/60000 [======================>.......] - ETA: 20s - loss: 0.2064 - categorical_accuracy: 0.9369
46784/60000 [======================>.......] - ETA: 20s - loss: 0.2063 - categorical_accuracy: 0.9369
46848/60000 [======================>.......] - ETA: 20s - loss: 0.2064 - categorical_accuracy: 0.9369
46880/60000 [======================>.......] - ETA: 20s - loss: 0.2064 - categorical_accuracy: 0.9369
46912/60000 [======================>.......] - ETA: 20s - loss: 0.2063 - categorical_accuracy: 0.9369
46976/60000 [======================>.......] - ETA: 20s - loss: 0.2061 - categorical_accuracy: 0.9369
47040/60000 [======================>.......] - ETA: 20s - loss: 0.2059 - categorical_accuracy: 0.9370
47104/60000 [======================>.......] - ETA: 20s - loss: 0.2058 - categorical_accuracy: 0.9371
47136/60000 [======================>.......] - ETA: 20s - loss: 0.2058 - categorical_accuracy: 0.9371
47200/60000 [======================>.......] - ETA: 20s - loss: 0.2057 - categorical_accuracy: 0.9371
47232/60000 [======================>.......] - ETA: 20s - loss: 0.2056 - categorical_accuracy: 0.9371
47296/60000 [======================>.......] - ETA: 20s - loss: 0.2054 - categorical_accuracy: 0.9372
47328/60000 [======================>.......] - ETA: 20s - loss: 0.2053 - categorical_accuracy: 0.9372
47392/60000 [======================>.......] - ETA: 19s - loss: 0.2051 - categorical_accuracy: 0.9372
47424/60000 [======================>.......] - ETA: 19s - loss: 0.2051 - categorical_accuracy: 0.9372
47488/60000 [======================>.......] - ETA: 19s - loss: 0.2052 - categorical_accuracy: 0.9372
47520/60000 [======================>.......] - ETA: 19s - loss: 0.2051 - categorical_accuracy: 0.9372
47552/60000 [======================>.......] - ETA: 19s - loss: 0.2051 - categorical_accuracy: 0.9373
47584/60000 [======================>.......] - ETA: 19s - loss: 0.2050 - categorical_accuracy: 0.9373
47616/60000 [======================>.......] - ETA: 19s - loss: 0.2049 - categorical_accuracy: 0.9373
47680/60000 [======================>.......] - ETA: 19s - loss: 0.2051 - categorical_accuracy: 0.9373
47744/60000 [======================>.......] - ETA: 19s - loss: 0.2048 - categorical_accuracy: 0.9374
47808/60000 [======================>.......] - ETA: 19s - loss: 0.2046 - categorical_accuracy: 0.9375
47872/60000 [======================>.......] - ETA: 19s - loss: 0.2044 - categorical_accuracy: 0.9375
47904/60000 [======================>.......] - ETA: 19s - loss: 0.2043 - categorical_accuracy: 0.9375
47936/60000 [======================>.......] - ETA: 19s - loss: 0.2044 - categorical_accuracy: 0.9375
48000/60000 [=======================>......] - ETA: 18s - loss: 0.2041 - categorical_accuracy: 0.9376
48064/60000 [=======================>......] - ETA: 18s - loss: 0.2039 - categorical_accuracy: 0.9377
48096/60000 [=======================>......] - ETA: 18s - loss: 0.2039 - categorical_accuracy: 0.9377
48160/60000 [=======================>......] - ETA: 18s - loss: 0.2038 - categorical_accuracy: 0.9377
48192/60000 [=======================>......] - ETA: 18s - loss: 0.2037 - categorical_accuracy: 0.9378
48256/60000 [=======================>......] - ETA: 18s - loss: 0.2036 - categorical_accuracy: 0.9378
48320/60000 [=======================>......] - ETA: 18s - loss: 0.2034 - categorical_accuracy: 0.9379
48384/60000 [=======================>......] - ETA: 18s - loss: 0.2032 - categorical_accuracy: 0.9379
48448/60000 [=======================>......] - ETA: 18s - loss: 0.2031 - categorical_accuracy: 0.9379
48480/60000 [=======================>......] - ETA: 18s - loss: 0.2030 - categorical_accuracy: 0.9380
48512/60000 [=======================>......] - ETA: 18s - loss: 0.2030 - categorical_accuracy: 0.9380
48544/60000 [=======================>......] - ETA: 18s - loss: 0.2028 - categorical_accuracy: 0.9380
48576/60000 [=======================>......] - ETA: 18s - loss: 0.2027 - categorical_accuracy: 0.9380
48608/60000 [=======================>......] - ETA: 18s - loss: 0.2027 - categorical_accuracy: 0.9380
48672/60000 [=======================>......] - ETA: 17s - loss: 0.2025 - categorical_accuracy: 0.9381
48704/60000 [=======================>......] - ETA: 17s - loss: 0.2025 - categorical_accuracy: 0.9381
48736/60000 [=======================>......] - ETA: 17s - loss: 0.2026 - categorical_accuracy: 0.9381
48768/60000 [=======================>......] - ETA: 17s - loss: 0.2026 - categorical_accuracy: 0.9381
48800/60000 [=======================>......] - ETA: 17s - loss: 0.2025 - categorical_accuracy: 0.9381
48864/60000 [=======================>......] - ETA: 17s - loss: 0.2023 - categorical_accuracy: 0.9382
48896/60000 [=======================>......] - ETA: 17s - loss: 0.2023 - categorical_accuracy: 0.9382
48960/60000 [=======================>......] - ETA: 17s - loss: 0.2022 - categorical_accuracy: 0.9382
49024/60000 [=======================>......] - ETA: 17s - loss: 0.2020 - categorical_accuracy: 0.9382
49088/60000 [=======================>......] - ETA: 17s - loss: 0.2018 - categorical_accuracy: 0.9383
49152/60000 [=======================>......] - ETA: 17s - loss: 0.2017 - categorical_accuracy: 0.9384
49184/60000 [=======================>......] - ETA: 17s - loss: 0.2017 - categorical_accuracy: 0.9384
49216/60000 [=======================>......] - ETA: 17s - loss: 0.2016 - categorical_accuracy: 0.9384
49248/60000 [=======================>......] - ETA: 16s - loss: 0.2015 - categorical_accuracy: 0.9384
49280/60000 [=======================>......] - ETA: 16s - loss: 0.2014 - categorical_accuracy: 0.9385
49312/60000 [=======================>......] - ETA: 16s - loss: 0.2013 - categorical_accuracy: 0.9385
49344/60000 [=======================>......] - ETA: 16s - loss: 0.2012 - categorical_accuracy: 0.9385
49408/60000 [=======================>......] - ETA: 16s - loss: 0.2010 - categorical_accuracy: 0.9386
49440/60000 [=======================>......] - ETA: 16s - loss: 0.2010 - categorical_accuracy: 0.9386
49472/60000 [=======================>......] - ETA: 16s - loss: 0.2009 - categorical_accuracy: 0.9386
49536/60000 [=======================>......] - ETA: 16s - loss: 0.2008 - categorical_accuracy: 0.9387
49568/60000 [=======================>......] - ETA: 16s - loss: 0.2006 - categorical_accuracy: 0.9387
49600/60000 [=======================>......] - ETA: 16s - loss: 0.2005 - categorical_accuracy: 0.9387
49632/60000 [=======================>......] - ETA: 16s - loss: 0.2004 - categorical_accuracy: 0.9388
49664/60000 [=======================>......] - ETA: 16s - loss: 0.2004 - categorical_accuracy: 0.9388
49696/60000 [=======================>......] - ETA: 16s - loss: 0.2003 - categorical_accuracy: 0.9388
49728/60000 [=======================>......] - ETA: 16s - loss: 0.2002 - categorical_accuracy: 0.9389
49792/60000 [=======================>......] - ETA: 16s - loss: 0.2001 - categorical_accuracy: 0.9389
49824/60000 [=======================>......] - ETA: 16s - loss: 0.2000 - categorical_accuracy: 0.9389
49888/60000 [=======================>......] - ETA: 15s - loss: 0.2000 - categorical_accuracy: 0.9389
49920/60000 [=======================>......] - ETA: 15s - loss: 0.1998 - categorical_accuracy: 0.9390
49984/60000 [=======================>......] - ETA: 15s - loss: 0.1997 - categorical_accuracy: 0.9390
50016/60000 [========================>.....] - ETA: 15s - loss: 0.1996 - categorical_accuracy: 0.9391
50048/60000 [========================>.....] - ETA: 15s - loss: 0.1995 - categorical_accuracy: 0.9391
50080/60000 [========================>.....] - ETA: 15s - loss: 0.1994 - categorical_accuracy: 0.9391
50112/60000 [========================>.....] - ETA: 15s - loss: 0.1993 - categorical_accuracy: 0.9391
50176/60000 [========================>.....] - ETA: 15s - loss: 0.1992 - categorical_accuracy: 0.9391
50240/60000 [========================>.....] - ETA: 15s - loss: 0.1990 - categorical_accuracy: 0.9392
50304/60000 [========================>.....] - ETA: 15s - loss: 0.1988 - categorical_accuracy: 0.9393
50368/60000 [========================>.....] - ETA: 15s - loss: 0.1986 - categorical_accuracy: 0.9393
50432/60000 [========================>.....] - ETA: 15s - loss: 0.1986 - categorical_accuracy: 0.9394
50496/60000 [========================>.....] - ETA: 15s - loss: 0.1985 - categorical_accuracy: 0.9394
50560/60000 [========================>.....] - ETA: 14s - loss: 0.1983 - categorical_accuracy: 0.9394
50624/60000 [========================>.....] - ETA: 14s - loss: 0.1982 - categorical_accuracy: 0.9394
50656/60000 [========================>.....] - ETA: 14s - loss: 0.1981 - categorical_accuracy: 0.9395
50688/60000 [========================>.....] - ETA: 14s - loss: 0.1980 - categorical_accuracy: 0.9395
50720/60000 [========================>.....] - ETA: 14s - loss: 0.1979 - categorical_accuracy: 0.9395
50752/60000 [========================>.....] - ETA: 14s - loss: 0.1978 - categorical_accuracy: 0.9395
50784/60000 [========================>.....] - ETA: 14s - loss: 0.1977 - categorical_accuracy: 0.9395
50816/60000 [========================>.....] - ETA: 14s - loss: 0.1977 - categorical_accuracy: 0.9395
50848/60000 [========================>.....] - ETA: 14s - loss: 0.1976 - categorical_accuracy: 0.9396
50912/60000 [========================>.....] - ETA: 14s - loss: 0.1975 - categorical_accuracy: 0.9396
50976/60000 [========================>.....] - ETA: 14s - loss: 0.1973 - categorical_accuracy: 0.9397
51040/60000 [========================>.....] - ETA: 14s - loss: 0.1971 - categorical_accuracy: 0.9397
51072/60000 [========================>.....] - ETA: 14s - loss: 0.1970 - categorical_accuracy: 0.9397
51136/60000 [========================>.....] - ETA: 14s - loss: 0.1968 - categorical_accuracy: 0.9398
51168/60000 [========================>.....] - ETA: 13s - loss: 0.1967 - categorical_accuracy: 0.9398
51200/60000 [========================>.....] - ETA: 13s - loss: 0.1967 - categorical_accuracy: 0.9398
51264/60000 [========================>.....] - ETA: 13s - loss: 0.1965 - categorical_accuracy: 0.9398
51328/60000 [========================>.....] - ETA: 13s - loss: 0.1963 - categorical_accuracy: 0.9399
51360/60000 [========================>.....] - ETA: 13s - loss: 0.1963 - categorical_accuracy: 0.9399
51392/60000 [========================>.....] - ETA: 13s - loss: 0.1963 - categorical_accuracy: 0.9399
51456/60000 [========================>.....] - ETA: 13s - loss: 0.1962 - categorical_accuracy: 0.9399
51520/60000 [========================>.....] - ETA: 13s - loss: 0.1960 - categorical_accuracy: 0.9400
51584/60000 [========================>.....] - ETA: 13s - loss: 0.1958 - categorical_accuracy: 0.9400
51648/60000 [========================>.....] - ETA: 13s - loss: 0.1958 - categorical_accuracy: 0.9400
51712/60000 [========================>.....] - ETA: 13s - loss: 0.1956 - categorical_accuracy: 0.9401
51744/60000 [========================>.....] - ETA: 13s - loss: 0.1955 - categorical_accuracy: 0.9401
51776/60000 [========================>.....] - ETA: 13s - loss: 0.1954 - categorical_accuracy: 0.9401
51840/60000 [========================>.....] - ETA: 12s - loss: 0.1953 - categorical_accuracy: 0.9401
51904/60000 [========================>.....] - ETA: 12s - loss: 0.1952 - categorical_accuracy: 0.9401
51936/60000 [========================>.....] - ETA: 12s - loss: 0.1951 - categorical_accuracy: 0.9402
51968/60000 [========================>.....] - ETA: 12s - loss: 0.1951 - categorical_accuracy: 0.9402
52000/60000 [=========================>....] - ETA: 12s - loss: 0.1950 - categorical_accuracy: 0.9402
52032/60000 [=========================>....] - ETA: 12s - loss: 0.1949 - categorical_accuracy: 0.9402
52064/60000 [=========================>....] - ETA: 12s - loss: 0.1949 - categorical_accuracy: 0.9402
52096/60000 [=========================>....] - ETA: 12s - loss: 0.1948 - categorical_accuracy: 0.9402
52160/60000 [=========================>....] - ETA: 12s - loss: 0.1947 - categorical_accuracy: 0.9403
52224/60000 [=========================>....] - ETA: 12s - loss: 0.1946 - categorical_accuracy: 0.9403
52288/60000 [=========================>....] - ETA: 12s - loss: 0.1946 - categorical_accuracy: 0.9403
52320/60000 [=========================>....] - ETA: 12s - loss: 0.1946 - categorical_accuracy: 0.9403
52352/60000 [=========================>....] - ETA: 12s - loss: 0.1946 - categorical_accuracy: 0.9403
52416/60000 [=========================>....] - ETA: 11s - loss: 0.1944 - categorical_accuracy: 0.9403
52448/60000 [=========================>....] - ETA: 11s - loss: 0.1943 - categorical_accuracy: 0.9403
52480/60000 [=========================>....] - ETA: 11s - loss: 0.1943 - categorical_accuracy: 0.9403
52544/60000 [=========================>....] - ETA: 11s - loss: 0.1942 - categorical_accuracy: 0.9404
52608/60000 [=========================>....] - ETA: 11s - loss: 0.1941 - categorical_accuracy: 0.9404
52640/60000 [=========================>....] - ETA: 11s - loss: 0.1941 - categorical_accuracy: 0.9404
52672/60000 [=========================>....] - ETA: 11s - loss: 0.1941 - categorical_accuracy: 0.9405
52704/60000 [=========================>....] - ETA: 11s - loss: 0.1940 - categorical_accuracy: 0.9405
52768/60000 [=========================>....] - ETA: 11s - loss: 0.1939 - categorical_accuracy: 0.9405
52800/60000 [=========================>....] - ETA: 11s - loss: 0.1938 - categorical_accuracy: 0.9406
52832/60000 [=========================>....] - ETA: 11s - loss: 0.1937 - categorical_accuracy: 0.9406
52864/60000 [=========================>....] - ETA: 11s - loss: 0.1935 - categorical_accuracy: 0.9406
52928/60000 [=========================>....] - ETA: 11s - loss: 0.1934 - categorical_accuracy: 0.9407
52960/60000 [=========================>....] - ETA: 11s - loss: 0.1932 - categorical_accuracy: 0.9407
52992/60000 [=========================>....] - ETA: 11s - loss: 0.1931 - categorical_accuracy: 0.9408
53056/60000 [=========================>....] - ETA: 10s - loss: 0.1930 - categorical_accuracy: 0.9408
53120/60000 [=========================>....] - ETA: 10s - loss: 0.1929 - categorical_accuracy: 0.9409
53152/60000 [=========================>....] - ETA: 10s - loss: 0.1929 - categorical_accuracy: 0.9409
53216/60000 [=========================>....] - ETA: 10s - loss: 0.1928 - categorical_accuracy: 0.9409
53248/60000 [=========================>....] - ETA: 10s - loss: 0.1928 - categorical_accuracy: 0.9410
53280/60000 [=========================>....] - ETA: 10s - loss: 0.1927 - categorical_accuracy: 0.9410
53312/60000 [=========================>....] - ETA: 10s - loss: 0.1926 - categorical_accuracy: 0.9410
53344/60000 [=========================>....] - ETA: 10s - loss: 0.1925 - categorical_accuracy: 0.9410
53376/60000 [=========================>....] - ETA: 10s - loss: 0.1925 - categorical_accuracy: 0.9410
53408/60000 [=========================>....] - ETA: 10s - loss: 0.1925 - categorical_accuracy: 0.9410
53472/60000 [=========================>....] - ETA: 10s - loss: 0.1924 - categorical_accuracy: 0.9411
53536/60000 [=========================>....] - ETA: 10s - loss: 0.1923 - categorical_accuracy: 0.9411
53568/60000 [=========================>....] - ETA: 10s - loss: 0.1923 - categorical_accuracy: 0.9411
53632/60000 [=========================>....] - ETA: 10s - loss: 0.1921 - categorical_accuracy: 0.9412
53664/60000 [=========================>....] - ETA: 10s - loss: 0.1920 - categorical_accuracy: 0.9412
53728/60000 [=========================>....] - ETA: 9s - loss: 0.1921 - categorical_accuracy: 0.9412 
53792/60000 [=========================>....] - ETA: 9s - loss: 0.1920 - categorical_accuracy: 0.9413
53824/60000 [=========================>....] - ETA: 9s - loss: 0.1919 - categorical_accuracy: 0.9413
53856/60000 [=========================>....] - ETA: 9s - loss: 0.1919 - categorical_accuracy: 0.9413
53888/60000 [=========================>....] - ETA: 9s - loss: 0.1918 - categorical_accuracy: 0.9413
53952/60000 [=========================>....] - ETA: 9s - loss: 0.1916 - categorical_accuracy: 0.9414
54016/60000 [==========================>...] - ETA: 9s - loss: 0.1916 - categorical_accuracy: 0.9414
54048/60000 [==========================>...] - ETA: 9s - loss: 0.1916 - categorical_accuracy: 0.9414
54112/60000 [==========================>...] - ETA: 9s - loss: 0.1915 - categorical_accuracy: 0.9415
54176/60000 [==========================>...] - ETA: 9s - loss: 0.1915 - categorical_accuracy: 0.9415
54240/60000 [==========================>...] - ETA: 9s - loss: 0.1913 - categorical_accuracy: 0.9415
54272/60000 [==========================>...] - ETA: 9s - loss: 0.1913 - categorical_accuracy: 0.9415
54336/60000 [==========================>...] - ETA: 8s - loss: 0.1911 - categorical_accuracy: 0.9416
54368/60000 [==========================>...] - ETA: 8s - loss: 0.1912 - categorical_accuracy: 0.9416
54432/60000 [==========================>...] - ETA: 8s - loss: 0.1912 - categorical_accuracy: 0.9416
54496/60000 [==========================>...] - ETA: 8s - loss: 0.1912 - categorical_accuracy: 0.9416
54528/60000 [==========================>...] - ETA: 8s - loss: 0.1912 - categorical_accuracy: 0.9416
54560/60000 [==========================>...] - ETA: 8s - loss: 0.1911 - categorical_accuracy: 0.9416
54624/60000 [==========================>...] - ETA: 8s - loss: 0.1910 - categorical_accuracy: 0.9417
54688/60000 [==========================>...] - ETA: 8s - loss: 0.1908 - categorical_accuracy: 0.9417
54720/60000 [==========================>...] - ETA: 8s - loss: 0.1908 - categorical_accuracy: 0.9417
54784/60000 [==========================>...] - ETA: 8s - loss: 0.1907 - categorical_accuracy: 0.9417
54848/60000 [==========================>...] - ETA: 8s - loss: 0.1907 - categorical_accuracy: 0.9417
54880/60000 [==========================>...] - ETA: 8s - loss: 0.1906 - categorical_accuracy: 0.9418
54912/60000 [==========================>...] - ETA: 8s - loss: 0.1905 - categorical_accuracy: 0.9418
54944/60000 [==========================>...] - ETA: 7s - loss: 0.1904 - categorical_accuracy: 0.9418
55008/60000 [==========================>...] - ETA: 7s - loss: 0.1903 - categorical_accuracy: 0.9419
55072/60000 [==========================>...] - ETA: 7s - loss: 0.1901 - categorical_accuracy: 0.9419
55136/60000 [==========================>...] - ETA: 7s - loss: 0.1900 - categorical_accuracy: 0.9419
55200/60000 [==========================>...] - ETA: 7s - loss: 0.1898 - categorical_accuracy: 0.9420
55232/60000 [==========================>...] - ETA: 7s - loss: 0.1898 - categorical_accuracy: 0.9420
55264/60000 [==========================>...] - ETA: 7s - loss: 0.1897 - categorical_accuracy: 0.9420
55328/60000 [==========================>...] - ETA: 7s - loss: 0.1898 - categorical_accuracy: 0.9421
55392/60000 [==========================>...] - ETA: 7s - loss: 0.1896 - categorical_accuracy: 0.9421
55456/60000 [==========================>...] - ETA: 7s - loss: 0.1895 - categorical_accuracy: 0.9421
55520/60000 [==========================>...] - ETA: 7s - loss: 0.1894 - categorical_accuracy: 0.9422
55584/60000 [==========================>...] - ETA: 6s - loss: 0.1893 - categorical_accuracy: 0.9422
55648/60000 [==========================>...] - ETA: 6s - loss: 0.1893 - categorical_accuracy: 0.9422
55680/60000 [==========================>...] - ETA: 6s - loss: 0.1892 - categorical_accuracy: 0.9422
55744/60000 [==========================>...] - ETA: 6s - loss: 0.1892 - categorical_accuracy: 0.9423
55808/60000 [==========================>...] - ETA: 6s - loss: 0.1891 - categorical_accuracy: 0.9423
55840/60000 [==========================>...] - ETA: 6s - loss: 0.1891 - categorical_accuracy: 0.9423
55904/60000 [==========================>...] - ETA: 6s - loss: 0.1890 - categorical_accuracy: 0.9423
55936/60000 [==========================>...] - ETA: 6s - loss: 0.1889 - categorical_accuracy: 0.9423
55968/60000 [==========================>...] - ETA: 6s - loss: 0.1888 - categorical_accuracy: 0.9424
56000/60000 [===========================>..] - ETA: 6s - loss: 0.1887 - categorical_accuracy: 0.9424
56032/60000 [===========================>..] - ETA: 6s - loss: 0.1887 - categorical_accuracy: 0.9424
56064/60000 [===========================>..] - ETA: 6s - loss: 0.1887 - categorical_accuracy: 0.9424
56128/60000 [===========================>..] - ETA: 6s - loss: 0.1887 - categorical_accuracy: 0.9424
56192/60000 [===========================>..] - ETA: 6s - loss: 0.1885 - categorical_accuracy: 0.9424
56256/60000 [===========================>..] - ETA: 5s - loss: 0.1884 - categorical_accuracy: 0.9425
56320/60000 [===========================>..] - ETA: 5s - loss: 0.1882 - categorical_accuracy: 0.9425
56384/60000 [===========================>..] - ETA: 5s - loss: 0.1881 - categorical_accuracy: 0.9425
56448/60000 [===========================>..] - ETA: 5s - loss: 0.1880 - categorical_accuracy: 0.9426
56512/60000 [===========================>..] - ETA: 5s - loss: 0.1879 - categorical_accuracy: 0.9426
56576/60000 [===========================>..] - ETA: 5s - loss: 0.1878 - categorical_accuracy: 0.9426
56608/60000 [===========================>..] - ETA: 5s - loss: 0.1877 - categorical_accuracy: 0.9427
56640/60000 [===========================>..] - ETA: 5s - loss: 0.1876 - categorical_accuracy: 0.9427
56672/60000 [===========================>..] - ETA: 5s - loss: 0.1875 - categorical_accuracy: 0.9427
56704/60000 [===========================>..] - ETA: 5s - loss: 0.1875 - categorical_accuracy: 0.9427
56736/60000 [===========================>..] - ETA: 5s - loss: 0.1875 - categorical_accuracy: 0.9427
56768/60000 [===========================>..] - ETA: 5s - loss: 0.1875 - categorical_accuracy: 0.9428
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1873 - categorical_accuracy: 0.9428
56864/60000 [===========================>..] - ETA: 4s - loss: 0.1872 - categorical_accuracy: 0.9428
56896/60000 [===========================>..] - ETA: 4s - loss: 0.1871 - categorical_accuracy: 0.9428
56928/60000 [===========================>..] - ETA: 4s - loss: 0.1870 - categorical_accuracy: 0.9429
56960/60000 [===========================>..] - ETA: 4s - loss: 0.1870 - categorical_accuracy: 0.9429
57024/60000 [===========================>..] - ETA: 4s - loss: 0.1868 - categorical_accuracy: 0.9430
57088/60000 [===========================>..] - ETA: 4s - loss: 0.1867 - categorical_accuracy: 0.9430
57152/60000 [===========================>..] - ETA: 4s - loss: 0.1866 - categorical_accuracy: 0.9430
57184/60000 [===========================>..] - ETA: 4s - loss: 0.1865 - categorical_accuracy: 0.9430
57216/60000 [===========================>..] - ETA: 4s - loss: 0.1864 - categorical_accuracy: 0.9430
57280/60000 [===========================>..] - ETA: 4s - loss: 0.1863 - categorical_accuracy: 0.9431
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1862 - categorical_accuracy: 0.9431
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1860 - categorical_accuracy: 0.9431
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1860 - categorical_accuracy: 0.9431
57504/60000 [===========================>..] - ETA: 3s - loss: 0.1858 - categorical_accuracy: 0.9432
57568/60000 [===========================>..] - ETA: 3s - loss: 0.1858 - categorical_accuracy: 0.9432
57632/60000 [===========================>..] - ETA: 3s - loss: 0.1856 - categorical_accuracy: 0.9432
57696/60000 [===========================>..] - ETA: 3s - loss: 0.1855 - categorical_accuracy: 0.9433
57728/60000 [===========================>..] - ETA: 3s - loss: 0.1854 - categorical_accuracy: 0.9433
57760/60000 [===========================>..] - ETA: 3s - loss: 0.1854 - categorical_accuracy: 0.9433
57792/60000 [===========================>..] - ETA: 3s - loss: 0.1853 - categorical_accuracy: 0.9434
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1852 - categorical_accuracy: 0.9434
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1851 - categorical_accuracy: 0.9435
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1850 - categorical_accuracy: 0.9435
58048/60000 [============================>.] - ETA: 3s - loss: 0.1849 - categorical_accuracy: 0.9435
58080/60000 [============================>.] - ETA: 3s - loss: 0.1848 - categorical_accuracy: 0.9435
58112/60000 [============================>.] - ETA: 2s - loss: 0.1848 - categorical_accuracy: 0.9435
58144/60000 [============================>.] - ETA: 2s - loss: 0.1848 - categorical_accuracy: 0.9435
58176/60000 [============================>.] - ETA: 2s - loss: 0.1847 - categorical_accuracy: 0.9435
58208/60000 [============================>.] - ETA: 2s - loss: 0.1846 - categorical_accuracy: 0.9436
58240/60000 [============================>.] - ETA: 2s - loss: 0.1846 - categorical_accuracy: 0.9436
58272/60000 [============================>.] - ETA: 2s - loss: 0.1846 - categorical_accuracy: 0.9436
58336/60000 [============================>.] - ETA: 2s - loss: 0.1845 - categorical_accuracy: 0.9436
58400/60000 [============================>.] - ETA: 2s - loss: 0.1843 - categorical_accuracy: 0.9437
58464/60000 [============================>.] - ETA: 2s - loss: 0.1842 - categorical_accuracy: 0.9437
58528/60000 [============================>.] - ETA: 2s - loss: 0.1840 - categorical_accuracy: 0.9438
58560/60000 [============================>.] - ETA: 2s - loss: 0.1840 - categorical_accuracy: 0.9438
58592/60000 [============================>.] - ETA: 2s - loss: 0.1839 - categorical_accuracy: 0.9438
58624/60000 [============================>.] - ETA: 2s - loss: 0.1838 - categorical_accuracy: 0.9438
58688/60000 [============================>.] - ETA: 2s - loss: 0.1836 - categorical_accuracy: 0.9439
58752/60000 [============================>.] - ETA: 1s - loss: 0.1835 - categorical_accuracy: 0.9439
58816/60000 [============================>.] - ETA: 1s - loss: 0.1834 - categorical_accuracy: 0.9439
58848/60000 [============================>.] - ETA: 1s - loss: 0.1833 - categorical_accuracy: 0.9440
58912/60000 [============================>.] - ETA: 1s - loss: 0.1832 - categorical_accuracy: 0.9440
58944/60000 [============================>.] - ETA: 1s - loss: 0.1832 - categorical_accuracy: 0.9440
59008/60000 [============================>.] - ETA: 1s - loss: 0.1831 - categorical_accuracy: 0.9440
59072/60000 [============================>.] - ETA: 1s - loss: 0.1829 - categorical_accuracy: 0.9441
59104/60000 [============================>.] - ETA: 1s - loss: 0.1829 - categorical_accuracy: 0.9441
59136/60000 [============================>.] - ETA: 1s - loss: 0.1828 - categorical_accuracy: 0.9441
59168/60000 [============================>.] - ETA: 1s - loss: 0.1829 - categorical_accuracy: 0.9441
59200/60000 [============================>.] - ETA: 1s - loss: 0.1828 - categorical_accuracy: 0.9441
59264/60000 [============================>.] - ETA: 1s - loss: 0.1827 - categorical_accuracy: 0.9441
59296/60000 [============================>.] - ETA: 1s - loss: 0.1826 - categorical_accuracy: 0.9442
59360/60000 [============================>.] - ETA: 1s - loss: 0.1826 - categorical_accuracy: 0.9442
59392/60000 [============================>.] - ETA: 0s - loss: 0.1825 - categorical_accuracy: 0.9442
59424/60000 [============================>.] - ETA: 0s - loss: 0.1825 - categorical_accuracy: 0.9442
59488/60000 [============================>.] - ETA: 0s - loss: 0.1825 - categorical_accuracy: 0.9442
59552/60000 [============================>.] - ETA: 0s - loss: 0.1825 - categorical_accuracy: 0.9442
59616/60000 [============================>.] - ETA: 0s - loss: 0.1823 - categorical_accuracy: 0.9443
59648/60000 [============================>.] - ETA: 0s - loss: 0.1823 - categorical_accuracy: 0.9443
59712/60000 [============================>.] - ETA: 0s - loss: 0.1821 - categorical_accuracy: 0.9443
59776/60000 [============================>.] - ETA: 0s - loss: 0.1820 - categorical_accuracy: 0.9443
59808/60000 [============================>.] - ETA: 0s - loss: 0.1819 - categorical_accuracy: 0.9444
59872/60000 [============================>.] - ETA: 0s - loss: 0.1817 - categorical_accuracy: 0.9444
59904/60000 [============================>.] - ETA: 0s - loss: 0.1817 - categorical_accuracy: 0.9444
59936/60000 [============================>.] - ETA: 0s - loss: 0.1816 - categorical_accuracy: 0.9444
59968/60000 [============================>.] - ETA: 0s - loss: 0.1816 - categorical_accuracy: 0.9444
60000/60000 [==============================] - 98s 2ms/step - loss: 0.1815 - categorical_accuracy: 0.9445 - val_loss: 0.0476 - val_categorical_accuracy: 0.9846

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 15s
  192/10000 [..............................] - ETA: 5s 
  352/10000 [>.............................] - ETA: 4s
  512/10000 [>.............................] - ETA: 3s
  704/10000 [=>............................] - ETA: 3s
  896/10000 [=>............................] - ETA: 3s
 1088/10000 [==>...........................] - ETA: 3s
 1280/10000 [==>...........................] - ETA: 2s
 1472/10000 [===>..........................] - ETA: 2s
 1664/10000 [===>..........................] - ETA: 2s
 1856/10000 [====>.........................] - ETA: 2s
 2048/10000 [=====>........................] - ETA: 2s
 2240/10000 [=====>........................] - ETA: 2s
 2432/10000 [======>.......................] - ETA: 2s
 2624/10000 [======>.......................] - ETA: 2s
 2784/10000 [=======>......................] - ETA: 2s
 2976/10000 [=======>......................] - ETA: 2s
 3168/10000 [========>.....................] - ETA: 2s
 3360/10000 [=========>....................] - ETA: 2s
 3552/10000 [=========>....................] - ETA: 2s
 3712/10000 [==========>...................] - ETA: 1s
 3872/10000 [==========>...................] - ETA: 1s
 4064/10000 [===========>..................] - ETA: 1s
 4256/10000 [===========>..................] - ETA: 1s
 4448/10000 [============>.................] - ETA: 1s
 4640/10000 [============>.................] - ETA: 1s
 4832/10000 [=============>................] - ETA: 1s
 5024/10000 [==============>...............] - ETA: 1s
 5216/10000 [==============>...............] - ETA: 1s
 5408/10000 [===============>..............] - ETA: 1s
 5600/10000 [===============>..............] - ETA: 1s
 5760/10000 [================>.............] - ETA: 1s
 5920/10000 [================>.............] - ETA: 1s
 6112/10000 [=================>............] - ETA: 1s
 6304/10000 [=================>............] - ETA: 1s
 6496/10000 [==================>...........] - ETA: 1s
 6688/10000 [===================>..........] - ETA: 1s
 6880/10000 [===================>..........] - ETA: 0s
 7072/10000 [====================>.........] - ETA: 0s
 7264/10000 [====================>.........] - ETA: 0s
 7424/10000 [=====================>........] - ETA: 0s
 7616/10000 [=====================>........] - ETA: 0s
 7808/10000 [======================>.......] - ETA: 0s
 8000/10000 [=======================>......] - ETA: 0s
 8192/10000 [=======================>......] - ETA: 0s
 8352/10000 [========================>.....] - ETA: 0s
 8544/10000 [========================>.....] - ETA: 0s
 8736/10000 [=========================>....] - ETA: 0s
 8928/10000 [=========================>....] - ETA: 0s
 9120/10000 [==========================>...] - ETA: 0s
 9312/10000 [==========================>...] - ETA: 0s
 9504/10000 [===========================>..] - ETA: 0s
 9696/10000 [============================>.] - ETA: 0s
 9888/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 3s 307us/step
[[1.05422195e-08 3.35057244e-08 9.76051979e-07 ... 9.99998331e-01
  7.04283387e-09 4.04424725e-07]
 [2.23838942e-05 3.62395549e-05 9.99934793e-01 ... 1.54796993e-08
  5.39519419e-07 5.66429625e-10]
 [8.73753322e-07 9.99910712e-01 4.98139161e-06 ... 1.05550034e-05
  5.10716745e-06 4.59776572e-07]
 ...
 [2.45145131e-08 1.00001273e-06 5.80414516e-09 ... 9.76095976e-07
  1.29958664e-06 2.54353199e-05]
 [1.12797272e-06 1.52705908e-07 8.28353031e-09 ... 5.84058341e-07
  5.48307784e-04 3.99541022e-07]
 [1.93215237e-05 9.72726980e-07 1.29728605e-05 ... 1.59014029e-08
  6.52563074e-07 8.48401527e-09]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.047585165653703736, 'accuracy_test:': 0.9846000075340271}

  ('#### Save   #######################################################',) 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/charcnn/result'}

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   2a547de..0fc2261  master     -> origin/master
Updating 2a547de..0fc2261
Fast-forward
 error_list/20200513/list_log_import_20200513.md  |    2 +-
 error_list/20200513/list_log_json_20200513.md    | 1146 +++++++++++-----------
 error_list/20200513/list_log_testall_20200513.md |  511 +++-------
 3 files changed, 696 insertions(+), 963 deletions(-)
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
[master 39dbb65] ml_store
 1 file changed, 1455 insertions(+)
To github.com:arita37/mlmodels_store.git
   0fc2261..39dbb65  master -> master





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
{'loss': 0.4021092765033245, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-13 12:32:16.367516: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
[master efc6832] ml_store
 1 file changed, 233 insertions(+)
To github.com:arita37/mlmodels_store.git
   39dbb65..efc6832  master -> master





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
[master 50808cf] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   efc6832..50808cf  master -> master





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
 40%|████      | 2/5 [00:15<00:23,  7.76s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.8496128371910541, 'learning_rate': 0.049071269972155815, 'min_data_in_leaf': 19, 'num_leaves': 37} and reward: 0.396
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeb0\x07B\xbf\xeaJX\r\x00\x00\x00learning_rateq\x02G?\xa9\x1f\xde\x97h=\xbeX\x10\x00\x00\x00min_data_in_leafq\x03K\x13X\n\x00\x00\x00num_leavesq\x04K%u.' and reward: 0.396
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeb0\x07B\xbf\xeaJX\r\x00\x00\x00learning_rateq\x02G?\xa9\x1f\xde\x97h=\xbeX\x10\x00\x00\x00min_data_in_leafq\x03K\x13X\n\x00\x00\x00num_leavesq\x04K%u.' and reward: 0.396
 60%|██████    | 3/5 [00:31<00:20, 10.19s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.853019522841504, 'learning_rate': 0.008941771670491882, 'min_data_in_leaf': 9, 'num_leaves': 59} and reward: 0.3884
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xebK\xef\x99.\x85gX\r\x00\x00\x00learning_rateq\x02G?\x82P\x10G$\xfd\xa6X\x10\x00\x00\x00min_data_in_leafq\x03K\tX\n\x00\x00\x00num_leavesq\x04K;u.' and reward: 0.3884
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xebK\xef\x99.\x85gX\r\x00\x00\x00learning_rateq\x02G?\x82P\x10G$\xfd\xa6X\x10\x00\x00\x00min_data_in_leafq\x03K\tX\n\x00\x00\x00num_leavesq\x04K;u.' and reward: 0.3884
 80%|████████  | 4/5 [00:52<00:13, 13.43s/it]Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.8566529831340322, 'learning_rate': 0.04006953948180989, 'min_data_in_leaf': 24, 'num_leaves': 61} and reward: 0.3904
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xebi\xb3\x84R\x9b\xb6X\r\x00\x00\x00learning_rateq\x02G?\xa4\x83\xfe\xa3G\xc2MX\x10\x00\x00\x00min_data_in_leafq\x03K\x18X\n\x00\x00\x00num_leavesq\x04K=u.' and reward: 0.3904
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xebi\xb3\x84R\x9b\xb6X\r\x00\x00\x00learning_rateq\x02G?\xa4\x83\xfe\xa3G\xc2MX\x10\x00\x00\x00min_data_in_leafq\x03K\x18X\n\x00\x00\x00num_leavesq\x04K=u.' and reward: 0.3904
100%|██████████| 5/5 [01:13<00:00, 15.87s/it]100%|██████████| 5/5 [01:13<00:00, 14.79s/it]
Saving dataset/models/LightGBMClassifier/trial_4_model.pkl
Finished Task with config: {'feature_fraction': 0.9338037112543609, 'learning_rate': 0.04215714544206581, 'min_data_in_leaf': 22, 'num_leaves': 40} and reward: 0.3924
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\xe1\xb8R\x17\x11\xacX\r\x00\x00\x00learning_rateq\x02G?\xa5\x95\x9f\x11\xee\xcalX\x10\x00\x00\x00min_data_in_leafq\x03K\x16X\n\x00\x00\x00num_leavesq\x04K(u.' and reward: 0.3924
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\xe1\xb8R\x17\x11\xacX\r\x00\x00\x00learning_rateq\x02G?\xa5\x95\x9f\x11\xee\xcalX\x10\x00\x00\x00min_data_in_leafq\x03K\x16X\n\x00\x00\x00num_leavesq\x04K(u.' and reward: 0.3924
Time for Gradient Boosting hyperparameter optimization: 90.88961601257324
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.8496128371910541, 'learning_rate': 0.049071269972155815, 'min_data_in_leaf': 19, 'num_leaves': 37}
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
 40%|████      | 2/5 [00:50<01:16, 25.38s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.47148165478251314, 'embedding_size_factor': 0.8698657059972175, 'layers.choice': 3, 'learning_rate': 0.0028831659276243593, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 1.0877683848466928e-09} and reward: 0.3584
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xde,\xc1c\xfd\x1c\\X\x15\x00\x00\x00embedding_size_factorq\x03G?\xeb\xd5\xf0\x9a\xe5p\xa4X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?g\x9eo\xeb\xc7`\xc0X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\x12\xb0\x0eR\xba\xf5\x16u.' and reward: 0.3584
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xde,\xc1c\xfd\x1c\\X\x15\x00\x00\x00embedding_size_factorq\x03G?\xeb\xd5\xf0\x9a\xe5p\xa4X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?g\x9eo\xeb\xc7`\xc0X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\x12\xb0\x0eR\xba\xf5\x16u.' and reward: 0.3584
 60%|██████    | 3/5 [01:44<01:07, 33.95s/it] 60%|██████    | 3/5 [01:44<01:09, 34.91s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_7_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.28279345444183607, 'embedding_size_factor': 1.0791623044170908, 'layers.choice': 2, 'learning_rate': 0.00011060834180726944, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 1.7360927944534643e-12} and reward: 0.3712
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd2\x19I\xb7\x96oqX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf1D?\xb1H\xc2(X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?\x1c\xfe\xcc\xd7\xc8\xd2`X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=~\x8a\xaa\xb7z\xdbMu.' and reward: 0.3712
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd2\x19I\xb7\x96oqX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf1D?\xb1H\xc2(X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?\x1c\xfe\xcc\xd7\xc8\xd2`X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=~\x8a\xaa\xb7z\xdbMu.' and reward: 0.3712
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 157.60665273666382
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
Loading: dataset/models/NeuralNetClassifier/trial_7_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.77s of the -133.05s of remaining time.
Ensemble size: 1
Ensemble weights: 
[1. 0. 0. 0. 0. 0. 0. 0.]
	0.396	 = Validation accuracy score
	1.72s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 254.82s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f3ab41c47f0>

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   50808cf..85d2146  master     -> origin/master
Updating 50808cf..85d2146
Fast-forward
 .../20200513/list_log_dataloader_20200513.md       |    2 +-
 error_list/20200513/list_log_import_20200513.md    |    2 +-
 error_list/20200513/list_log_json_20200513.md      | 1146 +++++-----
 error_list/20200513/list_log_jupyter_20200513.md   | 2348 ++++++++++----------
 .../20200513/list_log_pullrequest_20200513.md      |    2 +-
 error_list/20200513/list_log_test_cli_20200513.md  |  378 ++--
 error_list/20200513/list_log_testall_20200513.md   |  511 ++++-
 7 files changed, 2324 insertions(+), 2065 deletions(-)
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
[master ff2189d] ml_store
 1 file changed, 217 insertions(+)
To github.com:arita37/mlmodels_store.git
   85d2146..ff2189d  master -> master





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
[master 0e07a36] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   ff2189d..0e07a36  master -> master





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
[master f677a2c] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   0e07a36..f677a2c  master -> master





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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7f5832217fd0> 

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
[master cabb333] ml_store
 1 file changed, 108 insertions(+)
To github.com:arita37/mlmodels_store.git
   f677a2c..cabb333  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 1.24549398 -0.72239191  1.1181334   1.09899633  1.00277655 -0.90163449
  -0.53223402 -0.82246719  0.72171129  0.6743961 ]
 [ 1.07258847 -0.58652394 -1.34267579 -1.23685338  1.24328724  0.87583893
  -0.3264995   0.62336218 -0.43495668  1.11438298]
 [ 0.62368852  1.2066079   0.90399917 -0.28286355 -1.18913787 -0.26632688
   1.42361443  1.06897162  0.04037143  1.57546791]
 [ 0.76170668 -1.48515645  1.30253554 -0.59246129 -1.64162479 -2.30490794
  -1.34869645 -0.03181717  0.11248774 -0.36261209]
 [ 0.46739791 -0.23787527 -0.15449119 -0.75566277 -0.54706224  1.85143789
  -1.46405357  0.20909668  1.55501599 -0.09243232]
 [ 0.98379959 -0.40724002  0.93272141  0.16056499 -1.278618   -0.12014998
   0.19975956  0.38560229  0.71829074 -0.5301198 ]
 [ 0.61363671  0.3166589   1.34710546 -1.89526695 -0.76045809  0.08972912
  -0.32905155  0.41026575  0.85987097 -1.04906775]
 [ 0.9292506  -1.10657307 -1.95816909 -0.3592241  -1.21258781  0.5053819
   0.54264529  1.2179409  -1.94068096  0.67780757]
 [ 0.72297801  0.18553562  0.91549927  0.39442803 -0.84983074  0.72552256
  -0.15050433  1.49588477  0.67545381 -0.43820027]
 [ 1.34728643 -0.36453805  0.08075099 -0.45971768 -0.8894876   1.70548352
   0.09499611  0.24050555 -0.9994265  -0.76780375]
 [ 0.44118981  0.47985237 -0.1920037  -1.55269878 -1.88873982  0.57846442
   0.39859839 -0.9612636  -1.45832446 -3.05376438]
 [ 0.77528533  1.47016034  1.03298378 -0.87000822  0.78655651  0.36919047
  -0.14319575  0.85328219 -0.13971173 -0.22241403]
 [ 1.32857949 -0.5632366  -1.06179676  2.39014596 -1.6845077   0.24542285
  -0.56914865  1.15259914 -0.22423577  0.13224778]
 [ 0.94781411 -1.13379204  0.64098587 -0.1905483  -1.23912256  0.23333913
  -0.3169012   0.43499832  0.9104236   1.21987438]
 [ 0.68188934 -1.15498263  1.22895559 -0.1776322   0.99854519 -1.51045638
  -0.27584606  1.01120706 -1.47656266  1.30970591]
 [ 1.58463774  0.057121   -0.01771832 -0.79954749  1.32970299 -0.2915946
  -1.1077125  -0.25898285  0.1892932  -1.71939447]
 [ 1.18559003  0.08646441  1.23289919 -2.14246673  1.033341   -0.83016886
   0.36723181  0.45161595  1.10417433 -0.42285696]
 [ 0.88861146  0.84958685 -0.03091142 -0.12215402 -1.14722826 -0.68085157
  -0.32606131 -1.06787658 -0.07667936  0.35571726]
 [ 1.39198128 -0.19022103 -0.53722302 -0.44873803  0.70455707 -0.67244804
  -0.70134443 -0.55749472  0.93916874  0.15626385]
 [ 1.4468218   0.80745592  1.49810818  0.31223869 -0.68243019 -0.19332164
   0.28807817 -2.07680202  0.94750117 -0.30097615]
 [ 0.81583612 -1.39169388  2.50598029  0.45021774 -0.88286982  0.62743708
  -1.19586151  0.75133724  0.14039544  1.91979229]
 [ 0.345716   -0.41302931 -0.46867382  1.83471763  0.77151441  0.56438286
   0.02186284  2.13782807 -0.785534    0.85328122]
 [ 1.01195228 -1.88141087  1.70018815  0.4972691  -0.91766462  0.2373327
  -1.09033833 -2.14444405 -0.36956243  0.60878366]
 [ 1.18947778 -0.68067814 -0.05682448 -0.08450803  0.82178321 -0.29736188
  -0.18657899  0.417302    0.78477065  0.49233656]
 [ 0.6109426  -2.79099641 -1.33520272 -0.45611756 -0.94495995 -0.97989025
  -0.15699367  0.69257435 -0.47867236 -0.10646012]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7f569bf06ef0>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7f56b628c748> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 0.97139534  0.71304905  1.76041518  1.30620607  1.0576549  -0.60460297
   0.12837699  0.63658341  1.40925339  0.96653925]
 [ 1.4468218   0.80745592  1.49810818  0.31223869 -0.68243019 -0.19332164
   0.28807817 -2.07680202  0.94750117 -0.30097615]
 [ 1.77547698 -0.20339445 -0.19883786  0.24266944  0.96435056  0.20183018
  -0.54577417  0.66102029  1.79215821 -0.7003985 ]
 [ 1.01177337  0.09574677  0.73140252  1.0334508  -1.42203164 -0.14627327
  -0.01745495 -0.85749682 -0.93418184  0.95449567]
 [ 0.78801845  0.30196005  0.70098212 -0.39468968 -1.20376927 -1.17181338
   0.75539203  0.98401224 -0.55968142 -0.19893745]
 [ 0.69211449 -0.06065249  2.05635552 -2.413503    1.17456965 -1.77756638
  -0.28173627 -0.77785883  1.11584111  1.76024923]
 [ 1.17867274 -0.59980453 -0.6946936   1.12341216  1.17899425  0.30526704
   0.01335268  1.3887794  -0.66134424  0.6218035 ]
 [ 1.06702918 -0.42914228  0.35016716  1.20845633  0.75148062  1.1157018
  -0.4791571   0.84086156 -0.10288722  0.01716473]
 [ 0.87874071 -0.01923163  0.31965694  0.15001628 -1.46662161  0.46353432
  -0.89868319  0.39788042 -0.99601089  0.3181542 ]
 [ 0.8786438   1.03703898 -0.47712421  0.67261975 -1.04948638  2.42887697
   0.52475049  1.00568668  0.35356722 -0.03599018]
 [ 0.93621125  0.20437739 -1.49419377  0.61223252 -0.98437725  0.74488454
   0.49434165 -0.03628129 -0.83239535 -0.4466992 ]
 [ 0.87226739 -2.51630386 -0.77507029 -0.59566788  1.02600767 -0.30912132
   1.74643509  0.51093777  1.71066184  0.14164054]
 [ 0.61363671  0.3166589   1.34710546 -1.89526695 -0.76045809  0.08972912
  -0.32905155  0.41026575  0.85987097 -1.04906775]
 [ 0.87122579 -0.20975294 -0.45698786  0.93514778 -0.87353582  1.81252782
   0.92550121  0.14010988 -1.41914878  1.06898597]
 [ 0.85982375  0.17195713 -0.34898419  0.49056104 -1.15649503 -1.39528303
   0.61472628 -0.52235647 -0.3692559  -0.977773  ]
 [ 0.5630779  -1.17598267 -0.17418034  1.01012718  1.06796368  0.92001793
  -0.16819884 -0.19505734  0.80539342  0.4611641 ]
 [ 1.34728643 -0.36453805  0.08075099 -0.45971768 -0.8894876   1.70548352
   0.09499611  0.24050555 -0.9994265  -0.76780375]
 [ 0.85877496  2.29371761 -1.47023709 -0.83001099 -0.67204982 -1.01951985
   0.59921324 -0.21465384  1.02124813  0.60640394]
 [ 0.89891716  0.55743945 -0.75806733  0.18103874  0.84146721  1.10717545
   0.69336623  1.44287693 -0.53968156 -0.8088472 ]
 [ 1.34740825  0.73302323  0.83863475 -1.89881206 -0.54245992 -1.11711069
  -1.09715436 -0.50897228 -0.16648595 -1.03918232]
 [ 1.18559003  0.08646441  1.23289919 -2.14246673  1.033341   -0.83016886
   0.36723181  0.45161595  1.10417433 -0.42285696]
 [ 0.87699465  1.23225307 -0.86778722 -0.25417987  0.89189141  1.39984394
  -0.87728152 -0.78191168 -0.43750898 -1.44087602]
 [ 0.88838944  0.28299553  0.01795589  0.10803082 -0.84967187  0.02941762
  -0.50397395 -0.13479313  1.04921829 -1.27046078]
 [ 0.85335555 -0.70435033 -0.67938378 -0.04586669 -1.29936179 -0.21873346
   0.59003946  1.53920701 -1.14870423 -0.95090925]
 [ 0.62567337  0.5924728   0.67457071  1.19783084  1.23187251  1.70459417
  -0.76730983  1.04008915 -0.91844004  1.46089238]]
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
[[ 1.27991386 -0.87142207 -0.32403233 -0.86482994 -0.96853969  0.60874908
   0.50798434  0.5616381   1.51475038 -1.51107661]
 [ 0.85877496  2.29371761 -1.47023709 -0.83001099 -0.67204982 -1.01951985
   0.59921324 -0.21465384  1.02124813  0.60640394]
 [ 1.13545112  0.8616231   0.04906169 -2.08639057 -1.1146902   0.36180164
  -0.80617821  0.42592018  0.0490804  -0.59608633]
 [ 0.62368852  1.2066079   0.90399917 -0.28286355 -1.18913787 -0.26632688
   1.42361443  1.06897162  0.04037143  1.57546791]
 [ 1.17867274 -0.59980453 -0.6946936   1.12341216  1.17899425  0.30526704
   0.01335268  1.3887794  -0.66134424  0.6218035 ]
 [ 0.87122579 -0.20975294 -0.45698786  0.93514778 -0.87353582  1.81252782
   0.92550121  0.14010988 -1.41914878  1.06898597]
 [ 1.06040861  0.5103076   0.50172511 -0.91579185 -0.90731836 -0.40725204
  -0.17961229  0.98495167  1.07125243 -0.59334375]
 [ 1.64661853 -1.52568032 -0.6069984   0.79502609  1.08480038 -0.37443832
   0.42952614  0.1340482   1.20205486  0.10622272]
 [ 0.56998385 -0.53302033 -0.17545897 -1.42655542  0.60660431  1.76795995
  -0.11598519 -0.47537288  0.47761018 -0.93391466]
 [ 0.62567337  0.5924728   0.67457071  1.19783084  1.23187251  1.70459417
  -0.76730983  1.04008915 -0.91844004  1.46089238]
 [ 0.44118981  0.47985237 -0.1920037  -1.55269878 -1.88873982  0.57846442
   0.39859839 -0.9612636  -1.45832446 -3.05376438]
 [ 0.47330777 -0.97326759 -0.22814069  0.17516773 -1.01366961 -0.05348369
   0.39378773 -0.18306199 -0.2210289   0.58033011]
 [ 0.61363671  0.3166589   1.34710546 -1.89526695 -0.76045809  0.08972912
  -0.32905155  0.41026575  0.85987097 -1.04906775]
 [ 0.345716   -0.41302931 -0.46867382  1.83471763  0.77151441  0.56438286
   0.02186284  2.13782807 -0.785534    0.85328122]
 [ 1.39198128 -0.19022103 -0.53722302 -0.44873803  0.70455707 -0.67244804
  -0.70134443 -0.55749472  0.93916874  0.15626385]
 [ 1.36586461  3.9586027   0.54812958  0.64864364  0.84917607  0.10734329
   1.38631426 -1.39881282  0.08176782 -1.63744959]
 [ 0.92686981  0.39233491 -0.4234783   0.44838065 -1.09230828  1.1253235
  -0.94843966  0.10405339  0.52800342  1.00796648]
 [ 1.01177337  0.09574677  0.73140252  1.0334508  -1.42203164 -0.14627327
  -0.01745495 -0.85749682 -0.93418184  0.95449567]
 [ 0.69211449 -0.06065249  2.05635552 -2.413503    1.17456965 -1.77756638
  -0.28173627 -0.77785883  1.11584111  1.76024923]
 [ 1.58463774  0.057121   -0.01771832 -0.79954749  1.32970299 -0.2915946
  -1.1077125  -0.25898285  0.1892932  -1.71939447]
 [ 0.88883881  1.03368687 -0.04970258  0.80884436  0.81405135  1.78975468
   1.14690038  0.45128402 -1.68405999  0.46664327]
 [ 1.1437713   0.7278135   0.35249436  0.51507361  1.17718111 -2.78253447
  -1.94332341  0.58464661  0.32427424 -0.23643695]
 [ 0.78801845  0.30196005  0.70098212 -0.39468968 -1.20376927 -1.17181338
   0.75539203  0.98401224 -0.55968142 -0.19893745]
 [ 1.22867367  0.13437312 -0.18242041 -0.2683713  -1.73963799 -0.13167563
  -0.92687194  1.01855247  1.2305582  -0.49112514]
 [ 0.6675918  -0.45252497 -0.60598132  1.16128569 -1.44620987  1.06996554
   1.92381543 -1.04553425  0.35528451  1.80358898]]
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
[master 4aae7ac] ml_store
 1 file changed, 246 insertions(+)
To github.com:arita37/mlmodels_store.git
   cabb333..4aae7ac  master -> master





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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @139967664213912
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @139967664212960
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @139967664212624
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @139967664212120
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @139967664211616
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @139967664211280

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
grad_step = 000000, loss = 0.739434
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.543129
grad_step = 000002, loss = 0.374742
grad_step = 000003, loss = 0.210512
grad_step = 000004, loss = 0.081940
grad_step = 000005, loss = 0.061550
grad_step = 000006, loss = 0.113695
grad_step = 000007, loss = 0.096309
grad_step = 000008, loss = 0.041285
grad_step = 000009, loss = 0.012517
grad_step = 000010, loss = 0.015952
grad_step = 000011, loss = 0.031959
grad_step = 000012, loss = 0.043887
grad_step = 000013, loss = 0.045777
grad_step = 000014, loss = 0.038813
grad_step = 000015, loss = 0.027264
grad_step = 000016, loss = 0.016544
grad_step = 000017, loss = 0.011106
grad_step = 000018, loss = 0.012206
grad_step = 000019, loss = 0.016526
grad_step = 000020, loss = 0.018618
grad_step = 000021, loss = 0.016155
grad_step = 000022, loss = 0.011417
grad_step = 000023, loss = 0.008060
grad_step = 000024, loss = 0.007756
grad_step = 000025, loss = 0.009592
grad_step = 000026, loss = 0.011515
grad_step = 000027, loss = 0.012115
grad_step = 000028, loss = 0.011105
grad_step = 000029, loss = 0.009184
grad_step = 000030, loss = 0.007470
grad_step = 000031, loss = 0.006824
grad_step = 000032, loss = 0.007316
grad_step = 000033, loss = 0.008156
grad_step = 000034, loss = 0.008370
grad_step = 000035, loss = 0.007641
grad_step = 000036, loss = 0.006485
grad_step = 000037, loss = 0.005680
grad_step = 000038, loss = 0.005615
grad_step = 000039, loss = 0.006107
grad_step = 000040, loss = 0.006629
grad_step = 000041, loss = 0.006794
grad_step = 000042, loss = 0.006531
grad_step = 000043, loss = 0.006061
grad_step = 000044, loss = 0.005703
grad_step = 000045, loss = 0.005635
grad_step = 000046, loss = 0.005773
grad_step = 000047, loss = 0.005867
grad_step = 000048, loss = 0.005740
grad_step = 000049, loss = 0.005442
grad_step = 000050, loss = 0.005172
grad_step = 000051, loss = 0.005080
grad_step = 000052, loss = 0.005155
grad_step = 000053, loss = 0.005265
grad_step = 000054, loss = 0.005285
grad_step = 000055, loss = 0.005190
grad_step = 000056, loss = 0.005050
grad_step = 000057, loss = 0.004955
grad_step = 000058, loss = 0.004935
grad_step = 000059, loss = 0.004944
grad_step = 000060, loss = 0.004915
grad_step = 000061, loss = 0.004827
grad_step = 000062, loss = 0.004722
grad_step = 000063, loss = 0.004655
grad_step = 000064, loss = 0.004641
grad_step = 000065, loss = 0.004646
grad_step = 000066, loss = 0.004626
grad_step = 000067, loss = 0.004568
grad_step = 000068, loss = 0.004499
grad_step = 000069, loss = 0.004452
grad_step = 000070, loss = 0.004432
grad_step = 000071, loss = 0.004415
grad_step = 000072, loss = 0.004375
grad_step = 000073, loss = 0.004314
grad_step = 000074, loss = 0.004256
grad_step = 000075, loss = 0.004217
grad_step = 000076, loss = 0.004193
grad_step = 000077, loss = 0.004165
grad_step = 000078, loss = 0.004123
grad_step = 000079, loss = 0.004073
grad_step = 000080, loss = 0.004030
grad_step = 000081, loss = 0.003999
grad_step = 000082, loss = 0.003968
grad_step = 000083, loss = 0.003927
grad_step = 000084, loss = 0.003878
grad_step = 000085, loss = 0.003830
grad_step = 000086, loss = 0.003788
grad_step = 000087, loss = 0.003747
grad_step = 000088, loss = 0.003700
grad_step = 000089, loss = 0.003648
grad_step = 000090, loss = 0.003597
grad_step = 000091, loss = 0.003551
grad_step = 000092, loss = 0.003499
grad_step = 000093, loss = 0.003442
grad_step = 000094, loss = 0.003384
grad_step = 000095, loss = 0.003324
grad_step = 000096, loss = 0.003262
grad_step = 000097, loss = 0.003197
grad_step = 000098, loss = 0.003133
grad_step = 000099, loss = 0.003064
grad_step = 000100, loss = 0.002994
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002922
grad_step = 000102, loss = 0.002848
grad_step = 000103, loss = 0.002772
grad_step = 000104, loss = 0.002694
grad_step = 000105, loss = 0.002616
grad_step = 000106, loss = 0.002537
grad_step = 000107, loss = 0.002458
grad_step = 000108, loss = 0.002377
grad_step = 000109, loss = 0.002295
grad_step = 000110, loss = 0.002215
grad_step = 000111, loss = 0.002136
grad_step = 000112, loss = 0.002058
grad_step = 000113, loss = 0.001984
grad_step = 000114, loss = 0.001913
grad_step = 000115, loss = 0.001851
grad_step = 000116, loss = 0.001803
grad_step = 000117, loss = 0.001755
grad_step = 000118, loss = 0.001688
grad_step = 000119, loss = 0.001631
grad_step = 000120, loss = 0.001607
grad_step = 000121, loss = 0.001582
grad_step = 000122, loss = 0.001531
grad_step = 000123, loss = 0.001493
grad_step = 000124, loss = 0.001483
grad_step = 000125, loss = 0.001467
grad_step = 000126, loss = 0.001432
grad_step = 000127, loss = 0.001407
grad_step = 000128, loss = 0.001402
grad_step = 000129, loss = 0.001391
grad_step = 000130, loss = 0.001365
grad_step = 000131, loss = 0.001344
grad_step = 000132, loss = 0.001335
grad_step = 000133, loss = 0.001325
grad_step = 000134, loss = 0.001305
grad_step = 000135, loss = 0.001282
grad_step = 000136, loss = 0.001268
grad_step = 000137, loss = 0.001257
grad_step = 000138, loss = 0.001238
grad_step = 000139, loss = 0.001219
grad_step = 000140, loss = 0.001200
grad_step = 000141, loss = 0.001189
grad_step = 000142, loss = 0.001174
grad_step = 000143, loss = 0.001151
grad_step = 000144, loss = 0.001132
grad_step = 000145, loss = 0.001120
grad_step = 000146, loss = 0.001109
grad_step = 000147, loss = 0.001094
grad_step = 000148, loss = 0.001077
grad_step = 000149, loss = 0.001061
grad_step = 000150, loss = 0.001049
grad_step = 000151, loss = 0.001036
grad_step = 000152, loss = 0.001020
grad_step = 000153, loss = 0.001004
grad_step = 000154, loss = 0.000989
grad_step = 000155, loss = 0.000977
grad_step = 000156, loss = 0.000965
grad_step = 000157, loss = 0.000951
grad_step = 000158, loss = 0.000936
grad_step = 000159, loss = 0.000921
grad_step = 000160, loss = 0.000908
grad_step = 000161, loss = 0.000897
grad_step = 000162, loss = 0.000885
grad_step = 000163, loss = 0.000873
grad_step = 000164, loss = 0.000860
grad_step = 000165, loss = 0.000848
grad_step = 000166, loss = 0.000836
grad_step = 000167, loss = 0.000826
grad_step = 000168, loss = 0.000817
grad_step = 000169, loss = 0.000808
grad_step = 000170, loss = 0.000799
grad_step = 000171, loss = 0.000790
grad_step = 000172, loss = 0.000783
grad_step = 000173, loss = 0.000776
grad_step = 000174, loss = 0.000769
grad_step = 000175, loss = 0.000764
grad_step = 000176, loss = 0.000759
grad_step = 000177, loss = 0.000754
grad_step = 000178, loss = 0.000748
grad_step = 000179, loss = 0.000740
grad_step = 000180, loss = 0.000729
grad_step = 000181, loss = 0.000718
grad_step = 000182, loss = 0.000710
grad_step = 000183, loss = 0.000702
grad_step = 000184, loss = 0.000698
grad_step = 000185, loss = 0.000696
grad_step = 000186, loss = 0.000694
grad_step = 000187, loss = 0.000693
grad_step = 000188, loss = 0.000693
grad_step = 000189, loss = 0.000691
grad_step = 000190, loss = 0.000687
grad_step = 000191, loss = 0.000681
grad_step = 000192, loss = 0.000672
grad_step = 000193, loss = 0.000662
grad_step = 000194, loss = 0.000654
grad_step = 000195, loss = 0.000649
grad_step = 000196, loss = 0.000646
grad_step = 000197, loss = 0.000644
grad_step = 000198, loss = 0.000644
grad_step = 000199, loss = 0.000645
grad_step = 000200, loss = 0.000643
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.000641
grad_step = 000202, loss = 0.000637
grad_step = 000203, loss = 0.000632
grad_step = 000204, loss = 0.000624
grad_step = 000205, loss = 0.000617
grad_step = 000206, loss = 0.000611
grad_step = 000207, loss = 0.000606
grad_step = 000208, loss = 0.000601
grad_step = 000209, loss = 0.000597
grad_step = 000210, loss = 0.000595
grad_step = 000211, loss = 0.000593
grad_step = 000212, loss = 0.000591
grad_step = 000213, loss = 0.000592
grad_step = 000214, loss = 0.000596
grad_step = 000215, loss = 0.000604
grad_step = 000216, loss = 0.000619
grad_step = 000217, loss = 0.000635
grad_step = 000218, loss = 0.000648
grad_step = 000219, loss = 0.000628
grad_step = 000220, loss = 0.000590
grad_step = 000221, loss = 0.000564
grad_step = 000222, loss = 0.000570
grad_step = 000223, loss = 0.000588
grad_step = 000224, loss = 0.000593
grad_step = 000225, loss = 0.000577
grad_step = 000226, loss = 0.000554
grad_step = 000227, loss = 0.000550
grad_step = 000228, loss = 0.000562
grad_step = 000229, loss = 0.000567
grad_step = 000230, loss = 0.000559
grad_step = 000231, loss = 0.000545
grad_step = 000232, loss = 0.000537
grad_step = 000233, loss = 0.000540
grad_step = 000234, loss = 0.000546
grad_step = 000235, loss = 0.000546
grad_step = 000236, loss = 0.000536
grad_step = 000237, loss = 0.000527
grad_step = 000238, loss = 0.000524
grad_step = 000239, loss = 0.000525
grad_step = 000240, loss = 0.000527
grad_step = 000241, loss = 0.000526
grad_step = 000242, loss = 0.000522
grad_step = 000243, loss = 0.000515
grad_step = 000244, loss = 0.000510
grad_step = 000245, loss = 0.000508
grad_step = 000246, loss = 0.000508
grad_step = 000247, loss = 0.000509
grad_step = 000248, loss = 0.000508
grad_step = 000249, loss = 0.000506
grad_step = 000250, loss = 0.000502
grad_step = 000251, loss = 0.000498
grad_step = 000252, loss = 0.000494
grad_step = 000253, loss = 0.000491
grad_step = 000254, loss = 0.000488
grad_step = 000255, loss = 0.000486
grad_step = 000256, loss = 0.000484
grad_step = 000257, loss = 0.000483
grad_step = 000258, loss = 0.000482
grad_step = 000259, loss = 0.000481
grad_step = 000260, loss = 0.000482
grad_step = 000261, loss = 0.000483
grad_step = 000262, loss = 0.000485
grad_step = 000263, loss = 0.000490
grad_step = 000264, loss = 0.000498
grad_step = 000265, loss = 0.000504
grad_step = 000266, loss = 0.000512
grad_step = 000267, loss = 0.000509
grad_step = 000268, loss = 0.000498
grad_step = 000269, loss = 0.000478
grad_step = 000270, loss = 0.000460
grad_step = 000271, loss = 0.000452
grad_step = 000272, loss = 0.000454
grad_step = 000273, loss = 0.000463
grad_step = 000274, loss = 0.000471
grad_step = 000275, loss = 0.000473
grad_step = 000276, loss = 0.000467
grad_step = 000277, loss = 0.000456
grad_step = 000278, loss = 0.000444
grad_step = 000279, loss = 0.000436
grad_step = 000280, loss = 0.000433
grad_step = 000281, loss = 0.000434
grad_step = 000282, loss = 0.000438
grad_step = 000283, loss = 0.000440
grad_step = 000284, loss = 0.000441
grad_step = 000285, loss = 0.000438
grad_step = 000286, loss = 0.000433
grad_step = 000287, loss = 0.000426
grad_step = 000288, loss = 0.000419
grad_step = 000289, loss = 0.000414
grad_step = 000290, loss = 0.000410
grad_step = 000291, loss = 0.000409
grad_step = 000292, loss = 0.000408
grad_step = 000293, loss = 0.000409
grad_step = 000294, loss = 0.000410
grad_step = 000295, loss = 0.000411
grad_step = 000296, loss = 0.000412
grad_step = 000297, loss = 0.000413
grad_step = 000298, loss = 0.000413
grad_step = 000299, loss = 0.000413
grad_step = 000300, loss = 0.000410
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000407
grad_step = 000302, loss = 0.000399
grad_step = 000303, loss = 0.000389
grad_step = 000304, loss = 0.000379
grad_step = 000305, loss = 0.000372
grad_step = 000306, loss = 0.000370
grad_step = 000307, loss = 0.000370
grad_step = 000308, loss = 0.000370
grad_step = 000309, loss = 0.000368
grad_step = 000310, loss = 0.000365
grad_step = 000311, loss = 0.000365
grad_step = 000312, loss = 0.000369
grad_step = 000313, loss = 0.000374
grad_step = 000314, loss = 0.000379
grad_step = 000315, loss = 0.000381
grad_step = 000316, loss = 0.000389
grad_step = 000317, loss = 0.000394
grad_step = 000318, loss = 0.000393
grad_step = 000319, loss = 0.000376
grad_step = 000320, loss = 0.000358
grad_step = 000321, loss = 0.000343
grad_step = 000322, loss = 0.000332
grad_step = 000323, loss = 0.000328
grad_step = 000324, loss = 0.000334
grad_step = 000325, loss = 0.000343
grad_step = 000326, loss = 0.000346
grad_step = 000327, loss = 0.000347
grad_step = 000328, loss = 0.000343
grad_step = 000329, loss = 0.000338
grad_step = 000330, loss = 0.000327
grad_step = 000331, loss = 0.000317
grad_step = 000332, loss = 0.000310
grad_step = 000333, loss = 0.000306
grad_step = 000334, loss = 0.000304
grad_step = 000335, loss = 0.000304
grad_step = 000336, loss = 0.000306
grad_step = 000337, loss = 0.000312
grad_step = 000338, loss = 0.000321
grad_step = 000339, loss = 0.000330
grad_step = 000340, loss = 0.000342
grad_step = 000341, loss = 0.000341
grad_step = 000342, loss = 0.000340
grad_step = 000343, loss = 0.000327
grad_step = 000344, loss = 0.000311
grad_step = 000345, loss = 0.000292
grad_step = 000346, loss = 0.000284
grad_step = 000347, loss = 0.000286
grad_step = 000348, loss = 0.000296
grad_step = 000349, loss = 0.000303
grad_step = 000350, loss = 0.000305
grad_step = 000351, loss = 0.000304
grad_step = 000352, loss = 0.000297
grad_step = 000353, loss = 0.000288
grad_step = 000354, loss = 0.000277
grad_step = 000355, loss = 0.000271
grad_step = 000356, loss = 0.000269
grad_step = 000357, loss = 0.000271
grad_step = 000358, loss = 0.000274
grad_step = 000359, loss = 0.000278
grad_step = 000360, loss = 0.000281
grad_step = 000361, loss = 0.000283
grad_step = 000362, loss = 0.000284
grad_step = 000363, loss = 0.000281
grad_step = 000364, loss = 0.000277
grad_step = 000365, loss = 0.000271
grad_step = 000366, loss = 0.000266
grad_step = 000367, loss = 0.000260
grad_step = 000368, loss = 0.000255
grad_step = 000369, loss = 0.000251
grad_step = 000370, loss = 0.000250
grad_step = 000371, loss = 0.000250
grad_step = 000372, loss = 0.000251
grad_step = 000373, loss = 0.000253
grad_step = 000374, loss = 0.000255
grad_step = 000375, loss = 0.000260
grad_step = 000376, loss = 0.000264
grad_step = 000377, loss = 0.000273
grad_step = 000378, loss = 0.000278
grad_step = 000379, loss = 0.000286
grad_step = 000380, loss = 0.000286
grad_step = 000381, loss = 0.000286
grad_step = 000382, loss = 0.000275
grad_step = 000383, loss = 0.000263
grad_step = 000384, loss = 0.000248
grad_step = 000385, loss = 0.000238
grad_step = 000386, loss = 0.000233
grad_step = 000387, loss = 0.000233
grad_step = 000388, loss = 0.000237
grad_step = 000389, loss = 0.000243
grad_step = 000390, loss = 0.000250
grad_step = 000391, loss = 0.000255
grad_step = 000392, loss = 0.000260
grad_step = 000393, loss = 0.000258
grad_step = 000394, loss = 0.000255
grad_step = 000395, loss = 0.000246
grad_step = 000396, loss = 0.000238
grad_step = 000397, loss = 0.000229
grad_step = 000398, loss = 0.000224
grad_step = 000399, loss = 0.000221
grad_step = 000400, loss = 0.000222
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000225
grad_step = 000402, loss = 0.000228
grad_step = 000403, loss = 0.000233
grad_step = 000404, loss = 0.000239
grad_step = 000405, loss = 0.000246
grad_step = 000406, loss = 0.000251
grad_step = 000407, loss = 0.000256
grad_step = 000408, loss = 0.000252
grad_step = 000409, loss = 0.000247
grad_step = 000410, loss = 0.000236
grad_step = 000411, loss = 0.000225
grad_step = 000412, loss = 0.000217
grad_step = 000413, loss = 0.000213
grad_step = 000414, loss = 0.000213
grad_step = 000415, loss = 0.000216
grad_step = 000416, loss = 0.000221
grad_step = 000417, loss = 0.000228
grad_step = 000418, loss = 0.000236
grad_step = 000419, loss = 0.000240
grad_step = 000420, loss = 0.000241
grad_step = 000421, loss = 0.000235
grad_step = 000422, loss = 0.000228
grad_step = 000423, loss = 0.000223
grad_step = 000424, loss = 0.000224
grad_step = 000425, loss = 0.000229
grad_step = 000426, loss = 0.000233
grad_step = 000427, loss = 0.000228
grad_step = 000428, loss = 0.000221
grad_step = 000429, loss = 0.000215
grad_step = 000430, loss = 0.000218
grad_step = 000431, loss = 0.000223
grad_step = 000432, loss = 0.000224
grad_step = 000433, loss = 0.000218
grad_step = 000434, loss = 0.000212
grad_step = 000435, loss = 0.000213
grad_step = 000436, loss = 0.000218
grad_step = 000437, loss = 0.000222
grad_step = 000438, loss = 0.000220
grad_step = 000439, loss = 0.000215
grad_step = 000440, loss = 0.000216
grad_step = 000441, loss = 0.000222
grad_step = 000442, loss = 0.000232
grad_step = 000443, loss = 0.000238
grad_step = 000444, loss = 0.000247
grad_step = 000445, loss = 0.000250
grad_step = 000446, loss = 0.000254
grad_step = 000447, loss = 0.000247
grad_step = 000448, loss = 0.000241
grad_step = 000449, loss = 0.000231
grad_step = 000450, loss = 0.000221
grad_step = 000451, loss = 0.000208
grad_step = 000452, loss = 0.000198
grad_step = 000453, loss = 0.000195
grad_step = 000454, loss = 0.000202
grad_step = 000455, loss = 0.000212
grad_step = 000456, loss = 0.000219
grad_step = 000457, loss = 0.000221
grad_step = 000458, loss = 0.000218
grad_step = 000459, loss = 0.000217
grad_step = 000460, loss = 0.000216
grad_step = 000461, loss = 0.000215
grad_step = 000462, loss = 0.000211
grad_step = 000463, loss = 0.000204
grad_step = 000464, loss = 0.000198
grad_step = 000465, loss = 0.000193
grad_step = 000466, loss = 0.000191
grad_step = 000467, loss = 0.000191
grad_step = 000468, loss = 0.000194
grad_step = 000469, loss = 0.000198
grad_step = 000470, loss = 0.000204
grad_step = 000471, loss = 0.000210
grad_step = 000472, loss = 0.000218
grad_step = 000473, loss = 0.000221
grad_step = 000474, loss = 0.000225
grad_step = 000475, loss = 0.000223
grad_step = 000476, loss = 0.000222
grad_step = 000477, loss = 0.000218
grad_step = 000478, loss = 0.000218
grad_step = 000479, loss = 0.000218
grad_step = 000480, loss = 0.000218
grad_step = 000481, loss = 0.000211
grad_step = 000482, loss = 0.000200
grad_step = 000483, loss = 0.000189
grad_step = 000484, loss = 0.000185
grad_step = 000485, loss = 0.000189
grad_step = 000486, loss = 0.000197
grad_step = 000487, loss = 0.000202
grad_step = 000488, loss = 0.000201
grad_step = 000489, loss = 0.000198
grad_step = 000490, loss = 0.000195
grad_step = 000491, loss = 0.000197
grad_step = 000492, loss = 0.000202
grad_step = 000493, loss = 0.000211
grad_step = 000494, loss = 0.000218
grad_step = 000495, loss = 0.000227
grad_step = 000496, loss = 0.000228
grad_step = 000497, loss = 0.000230
grad_step = 000498, loss = 0.000224
grad_step = 000499, loss = 0.000221
grad_step = 000500, loss = 0.000215
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000211
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
[[0.8604659  0.8525581  0.9356502  0.9513116  1.0207665 ]
 [0.85553265 0.9177433  0.949911   1.0117488  0.9800602 ]
 [0.9016315  0.92720693 0.99943703 0.987595   0.937219  ]
 [0.92471397 1.0009937  1.0024     0.9525149  0.9134784 ]
 [0.9965356  0.99061906 0.96102417 0.9248475  0.8552984 ]
 [0.9792063  0.9420896  0.921823   0.87240684 0.85471034]
 [0.95342755 0.8983935  0.859298   0.8542194  0.8276917 ]
 [0.9161675  0.83573693 0.86120903 0.82138765 0.83358115]
 [0.84098184 0.8419908  0.81571996 0.85621893 0.8592557 ]
 [0.84265107 0.80036277 0.8432982  0.8493145  0.8142703 ]
 [0.80407786 0.8182919  0.8578645  0.8256854  0.9182837 ]
 [0.84167904 0.8628282  0.8192837  0.9395917  0.93304336]
 [0.8537397  0.8493249  0.93371    0.9510777  1.018479  ]
 [0.85329556 0.9266962  0.95409155 1.0121121  0.9735676 ]
 [0.91367066 0.93541676 1.0035106  0.9786476  0.92334294]
 [0.9392715  1.0116323  0.9936163  0.9436589  0.89476603]
 [1.0065467  0.9898753  0.94239044 0.90895355 0.8339999 ]
 [0.97955525 0.92876464 0.90054226 0.850847   0.8395186 ]
 [0.94203484 0.89068246 0.837283   0.8405237  0.8239609 ]
 [0.9182979  0.83479226 0.8503165  0.81766886 0.8352085 ]
 [0.8508561  0.8508347  0.8147584  0.8551385  0.8664081 ]
 [0.86272    0.8134705  0.84525585 0.85771936 0.82321465]
 [0.8165519  0.8281519  0.8638094  0.8299799  0.91837746]
 [0.85045284 0.8700744  0.8260746  0.9448509  0.9386418 ]
 [0.86353934 0.8605057  0.93401814 0.95367825 1.0231642 ]
 [0.8622597  0.92084605 0.9532605  1.0193474  0.99587935]
 [0.9067356  0.93583524 1.0054123  0.9994742  0.9502598 ]
 [0.9357908  1.0147618  1.0149076  0.96689874 0.924499  ]
 [1.0092819  1.0072744  0.9718741  0.9379516  0.8617548 ]
 [0.9904252  0.9533192  0.93002915 0.88044226 0.8587779 ]
 [0.9602032  0.9072666  0.865008   0.85819924 0.8346312 ]]

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
[master ca67abe] ml_store
 1 file changed, 1122 insertions(+)
To github.com:arita37/mlmodels_store.git
   4aae7ac..ca67abe  master -> master





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
