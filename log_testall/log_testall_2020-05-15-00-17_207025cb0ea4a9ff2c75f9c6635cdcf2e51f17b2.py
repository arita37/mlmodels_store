
  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_all 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2

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
Warning: Permanently added the RSA host key for IP address '140.82.113.4' to the list of known hosts.
From github.com:arita37/mlmodels_store
   7874825..2ccd957  master     -> origin/master
Updating 7874825..2ccd957
Fast-forward
 .../20200515/list_log_dataloader_20200515.md       |   2 +-
 error_list/20200515/list_log_import_20200515.md    |   2 +-
 error_list/20200515/list_log_test_cli_20200515.md  | 138 ++---
 ...-08_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py | 373 +++++++++++++
 ...-14_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py | 614 +++++++++++++++++++++
 5 files changed, 1058 insertions(+), 71 deletions(-)
 create mode 100644 log_dataloader/log_2020-05-15-00-08_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py
 create mode 100644 log_pullrequest/log_pr_2020-05-15-00-14_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py
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
[master 13d3a1d] ml_store
 1 file changed, 71 insertions(+)
 create mode 100644 log_testall/log_testall_2020-05-15-00-17_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py
To github.com:arita37/mlmodels_store.git
   2ccd957..13d3a1d  master -> master





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
[master bd05f69] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   13d3a1d..bd05f69  master -> master





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
[master 9c2aa79] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   bd05f69..9c2aa79  master -> master





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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         20          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         36          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-15 00:18:19.288081: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 00:18:19.306066: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-15 00:18:19.306346: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562dbbe32910 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 00:18:19.306368: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 1s 1ms/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2502 - val_binary_crossentropy: 0.6934

  #### metrics   #################################################### 
{'MSE': 0.2499240894099016}

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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         20          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 393
Trainable params: 393
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2814 - binary_crossentropy: 0.7613500/500 [==============================] - 1s 1ms/sample - loss: 0.2784 - binary_crossentropy: 0.7555 - val_loss: 0.2790 - val_binary_crossentropy: 0.7563

  #### metrics   #################################################### 
{'MSE': 0.27767074571036904}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 393
Trainable params: 393
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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
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
Total params: 647
Trainable params: 647
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 1s 2ms/sample - loss: 0.2501 - binary_crossentropy: 0.6934 - val_loss: 0.2501 - val_binary_crossentropy: 0.6933

  #### metrics   #################################################### 
{'MSE': 0.2499178220262715}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
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
Total params: 388
Trainable params: 388
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2768 - binary_crossentropy: 1.1374500/500 [==============================] - 1s 2ms/sample - loss: 0.2781 - binary_crossentropy: 1.1421 - val_loss: 0.2699 - val_binary_crossentropy: 1.0707

  #### metrics   #################################################### 
{'MSE': 0.269497420653459}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 123
Trainable params: 123
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2906 - binary_crossentropy: 0.8030500/500 [==============================] - 2s 3ms/sample - loss: 0.3022 - binary_crossentropy: 0.8271 - val_loss: 0.3035 - val_binary_crossentropy: 0.8259

  #### metrics   #################################################### 
{'MSE': 0.3023069535921556}

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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 123
Trainable params: 123
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-15 00:19:37.700106: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 00:19:37.702292: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 00:19:37.708283: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-15 00:19:37.718454: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-15 00:19:37.720340: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-15 00:19:37.721893: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 00:19:37.723684: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2461 - val_binary_crossentropy: 0.6854
2020-05-15 00:19:38.950178: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 00:19:38.951865: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 00:19:38.956187: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-15 00:19:38.964880: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-15 00:19:38.966417: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-15 00:19:38.967742: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 00:19:38.969083: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2444477601037415}

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
2020-05-15 00:20:02.645626: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 00:20:02.647189: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 00:20:02.651467: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-15 00:20:02.658694: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-15 00:20:02.659860: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-15 00:20:02.661266: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 00:20:02.662198: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2503 - val_binary_crossentropy: 0.6937
2020-05-15 00:20:04.216336: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 00:20:04.217718: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 00:20:04.220750: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-15 00:20:04.227751: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-15 00:20:04.229227: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-15 00:20:04.230742: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 00:20:04.232205: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.250295703853916}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-15 00:20:38.296615: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 00:20:38.301369: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 00:20:38.315309: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-15 00:20:38.339797: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-15 00:20:38.343968: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-15 00:20:38.347897: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 00:20:38.352022: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.2038 - binary_crossentropy: 0.6004 - val_loss: 0.2520 - val_binary_crossentropy: 0.6972
2020-05-15 00:20:40.658577: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 00:20:40.663907: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 00:20:40.675358: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-15 00:20:40.702353: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-15 00:20:40.706685: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-15 00:20:40.711074: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 00:20:40.714811: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2362214341803567}

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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
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
100/500 [=====>........................] - ETA: 6s - loss: 0.2462 - binary_crossentropy: 0.6854500/500 [==============================] - 4s 8ms/sample - loss: 0.2549 - binary_crossentropy: 0.7295 - val_loss: 0.2619 - val_binary_crossentropy: 0.8230

  #### metrics   #################################################### 
{'MSE': 0.2581489773673219}

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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 2)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 2)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 2)         6           sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         16          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         14          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_2[0][0]           
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
Total params: 254
Trainable params: 254
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2636 - binary_crossentropy: 0.7216500/500 [==============================] - 5s 10ms/sample - loss: 0.2781 - binary_crossentropy: 0.9124 - val_loss: 0.2663 - val_binary_crossentropy: 0.9886

  #### metrics   #################################################### 
{'MSE': 0.27073187084346295}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 2)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 2)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 2)         6           sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         16          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         14          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_2[0][0]           
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
Total params: 254
Trainable params: 254
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_max[0][0]               
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
Total params: 1,889
Trainable params: 1,889
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2584 - binary_crossentropy: 0.7102500/500 [==============================] - 5s 9ms/sample - loss: 0.2564 - binary_crossentropy: 0.7065 - val_loss: 0.2622 - val_binary_crossentropy: 0.7183

  #### metrics   #################################################### 
{'MSE': 0.2599002586928466}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_max[0][0]               
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
regionsequence_sum (InputLayer) [(None, 4)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 5)]          0                                            
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
region_10sparse_seq_emb_regions (None, 4, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 5, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 4, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 4, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 4, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 4, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 4, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 4, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 4, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 4, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 4, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 4, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 4, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 4, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 4, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 4, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 4, 1)         1           regionsequence_max[0][0]         
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
Total params: 112
Trainable params: 112
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.5301 - binary_crossentropy: 8.1752500/500 [==============================] - 6s 12ms/sample - loss: 0.4661 - binary_crossentropy: 7.1880 - val_loss: 0.5081 - val_binary_crossentropy: 7.8359

  #### metrics   #################################################### 
{'MSE': 0.487}

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
regionsequence_sum (InputLayer) [(None, 4)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 5)]          0                                            
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
region_10sparse_seq_emb_regions (None, 4, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 5, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 4, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 4, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 4, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 4, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 4, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 4, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 4, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 4, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 4, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 4, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 4, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 4, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 4, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 4, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         4           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 4, 1)         1           regionsequence_max[0][0]         
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
Total params: 112
Trainable params: 112
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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_max[0][0]               
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
Total params: 1,397
Trainable params: 1,397
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.3554 - binary_crossentropy: 3.5198500/500 [==============================] - 6s 12ms/sample - loss: 0.3475 - binary_crossentropy: 3.1633 - val_loss: 0.3319 - val_binary_crossentropy: 2.9686

  #### metrics   #################################################### 
{'MSE': 0.338508458023321}

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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_max[0][0]               
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
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
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 2, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 4, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         12          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         12          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 2, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         12          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 4, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 2, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 4, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 2, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 4, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_max[0][0]               
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
Total params: 3,001
Trainable params: 2,921
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2702 - binary_crossentropy: 0.7349500/500 [==============================] - 7s 13ms/sample - loss: 0.2577 - binary_crossentropy: 0.7089 - val_loss: 0.2534 - val_binary_crossentropy: 0.7261

  #### metrics   #################################################### 
{'MSE': 0.2526957757044986}

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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
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
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 2, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 4, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         12          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         12          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 2, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         12          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 4, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 2, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 4, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 2, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 4, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_max[0][0]               
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
Total params: 3,001
Trainable params: 2,921
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
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
Total params: 489
Trainable params: 489
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2513 - binary_crossentropy: 0.6957500/500 [==============================] - 7s 13ms/sample - loss: 0.2503 - binary_crossentropy: 0.6939 - val_loss: 0.2514 - val_binary_crossentropy: 0.6960

  #### metrics   #################################################### 
{'MSE': 0.25044492420822734}

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
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
Total params: 489
Trainable params: 489
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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_mean[0][0]              
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
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
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
100/500 [=====>........................] - ETA: 8s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 7s 13ms/sample - loss: 0.2503 - binary_crossentropy: 0.6937 - val_loss: 0.2499 - val_binary_crossentropy: 0.6930

  #### metrics   #################################################### 
{'MSE': 0.24996582013347485}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_mean[0][0]              
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
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_47 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_90 (Concatenate)    (None, 1, 20)        0           no_mask_130[0][0]                
                                                                 no_mask_130[1][0]                
                                                                 no_mask_130[2][0]                
                                                                 no_mask_130[3][0]                
                                                                 no_mask_130[4][0]                
__________________________________________________________________________________________________
no_mask_131 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
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
Total params: 361
Trainable params: 361
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2540 - binary_crossentropy: 0.7014500/500 [==============================] - 7s 14ms/sample - loss: 0.2531 - binary_crossentropy: 0.6995 - val_loss: 0.2508 - val_binary_crossentropy: 0.6948

  #### metrics   #################################################### 
{'MSE': 0.25077194517425605}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_47 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_90 (Concatenate)    (None, 1, 20)        0           no_mask_130[0][0]                
                                                                 no_mask_130[1][0]                
                                                                 no_mask_130[2][0]                
                                                                 no_mask_130[3][0]                
                                                                 no_mask_130[4][0]                
__________________________________________________________________________________________________
no_mask_131 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
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
Total params: 361
Trainable params: 361
Non-trainable params: 0
__________________________________________________________________________________________________

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Warning: Permanently added the RSA host key for IP address '140.82.113.3' to the list of known hosts.
From github.com:arita37/mlmodels_store
   9c2aa79..31b65f2  master     -> origin/master
Updating 9c2aa79..31b65f2
Fast-forward
 .../20200515/list_log_dataloader_20200515.md       |    2 +-
 error_list/20200515/list_log_import_20200515.md    |    2 +-
 .../20200515/list_log_pullrequest_20200515.md      |    2 +-
 error_list/20200515/list_log_test_cli_20200515.md  |  152 +-
 ...-20_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py | 3968 ++++++++++++++++++++
 ...-20_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py | 2000 ++++++++++
 ...-22_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py | 3523 +++++++++++++++++
 7 files changed, 9575 insertions(+), 74 deletions(-)
 create mode 100644 log_json/log_json_2020-05-15-00-20_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py
 create mode 100644 log_jupyter/log_jupyter_2020-05-15-00-20_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py
 create mode 100644 log_test_cli/log_cli_2020-05-15-00-22_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py
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
[master 4c5509e] ml_store
 1 file changed, 5676 insertions(+)
To github.com:arita37/mlmodels_store.git
   31b65f2..4c5509e  master -> master





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
[master 3f2443f] ml_store
 1 file changed, 50 insertions(+)
To github.com:arita37/mlmodels_store.git
   4c5509e..3f2443f  master -> master





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
[master b7aae40] ml_store
 1 file changed, 46 insertions(+)
To github.com:arita37/mlmodels_store.git
   3f2443f..b7aae40  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//Autokeras.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//Autokeras.py", line 12, in <module>
    import autokeras as ak
ModuleNotFoundError: No module named 'autokeras'

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
[master c86aae3] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   b7aae40..c86aae3  master -> master





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

2020-05-15 00:34:01.334582: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 00:34:01.339088: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-15 00:34:01.339261: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e2edaefb40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 00:34:01.339276: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 8s - loss: 1.3881
256/354 [====================>.........] - ETA: 3s - loss: 1.2374
354/354 [==============================] - 15s 43ms/step - loss: 1.3112 - val_loss: 2.3373

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
[master 1bf7843] ml_store
 1 file changed, 149 insertions(+)
To github.com:arita37/mlmodels_store.git
   c86aae3..1bf7843  master -> master





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
[master 5daa4c6] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   1bf7843..5daa4c6  master -> master





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
[master 8662385] ml_store
 1 file changed, 44 insertions(+)
To github.com:arita37/mlmodels_store.git
   5daa4c6..8662385  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2211840/17464789 [==>...........................] - ETA: 0s
 9093120/17464789 [==============>...............] - ETA: 0s
16506880/17464789 [===========================>..] - ETA: 0s
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
2020-05-15 00:35:01.815018: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 00:35:01.819072: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-15 00:35:01.819225: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562f3600dd00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 00:35:01.819240: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 13s - loss: 7.5440 - accuracy: 0.5080
 2000/25000 [=>............................] - ETA: 9s - loss: 7.4060 - accuracy: 0.5170 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5388 - accuracy: 0.5083
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6130 - accuracy: 0.5035
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5225 - accuracy: 0.5094
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.4494 - accuracy: 0.5142
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.5199 - accuracy: 0.5096
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5593 - accuracy: 0.5070
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5644 - accuracy: 0.5067
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5486 - accuracy: 0.5077
11000/25000 [============>.................] - ETA: 4s - loss: 7.5774 - accuracy: 0.5058
12000/25000 [=============>................] - ETA: 4s - loss: 7.6066 - accuracy: 0.5039
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6218 - accuracy: 0.5029
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6392 - accuracy: 0.5018
15000/25000 [=================>............] - ETA: 3s - loss: 7.6564 - accuracy: 0.5007
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6695 - accuracy: 0.4998
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6702 - accuracy: 0.4998
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6777 - accuracy: 0.4993
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6828 - accuracy: 0.4989
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6789 - accuracy: 0.4992
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6674 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6778 - accuracy: 0.4993
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6760 - accuracy: 0.4994
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6743 - accuracy: 0.4995
25000/25000 [==============================] - 9s 374us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7f917e6bec50>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f9164db59b0> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.4980 - accuracy: 0.5110
 2000/25000 [=>............................] - ETA: 9s - loss: 7.4520 - accuracy: 0.5140 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6768 - accuracy: 0.4993
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6743 - accuracy: 0.4995
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6973 - accuracy: 0.4980
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6385 - accuracy: 0.5018
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6360 - accuracy: 0.5020
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6647 - accuracy: 0.5001
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6717 - accuracy: 0.4997
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6360 - accuracy: 0.5020
11000/25000 [============>.................] - ETA: 4s - loss: 7.6067 - accuracy: 0.5039
12000/25000 [=============>................] - ETA: 4s - loss: 7.5989 - accuracy: 0.5044
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6053 - accuracy: 0.5040
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6042 - accuracy: 0.5041
15000/25000 [=================>............] - ETA: 3s - loss: 7.6227 - accuracy: 0.5029
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6168 - accuracy: 0.5033
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6360 - accuracy: 0.5020
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6462 - accuracy: 0.5013
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6295 - accuracy: 0.5024
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6283 - accuracy: 0.5025
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6389 - accuracy: 0.5018
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6353 - accuracy: 0.5020
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6486 - accuracy: 0.5012
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6545 - accuracy: 0.5008
25000/25000 [==============================] - 9s 376us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.8046 - accuracy: 0.4910
 2000/25000 [=>............................] - ETA: 9s - loss: 7.7126 - accuracy: 0.4970 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7075 - accuracy: 0.4973
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6628 - accuracy: 0.5002
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6881 - accuracy: 0.4986
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7177 - accuracy: 0.4967
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6885 - accuracy: 0.4986
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6475 - accuracy: 0.5013
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6564 - accuracy: 0.5007
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6528 - accuracy: 0.5009
11000/25000 [============>.................] - ETA: 4s - loss: 7.6387 - accuracy: 0.5018
12000/25000 [=============>................] - ETA: 4s - loss: 7.6768 - accuracy: 0.4993
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6749 - accuracy: 0.4995
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6962 - accuracy: 0.4981
15000/25000 [=================>............] - ETA: 3s - loss: 7.7085 - accuracy: 0.4973
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7136 - accuracy: 0.4969
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6964 - accuracy: 0.4981
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6990 - accuracy: 0.4979
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6957 - accuracy: 0.4981
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6958 - accuracy: 0.4981
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6936 - accuracy: 0.4982
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6792 - accuracy: 0.4992
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6793 - accuracy: 0.4992
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6743 - accuracy: 0.4995
25000/25000 [==============================] - 9s 371us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
Warning: Permanently added the RSA host key for IP address '192.30.255.113' to the list of known hosts.
From github.com:arita37/mlmodels_store
   8662385..ee36a24  master     -> origin/master
Updating 8662385..ee36a24
Fast-forward
 error_list/20200515/list_log_benchmark_20200515.md |  184 ++-
 error_list/20200515/list_log_jupyter_20200515.md   | 1658 ++++++++++----------
 .../20200515/list_log_pullrequest_20200515.md      |    2 +-
 error_list/20200515/list_log_testall_20200515.md   |  833 +---------
 4 files changed, 937 insertions(+), 1740 deletions(-)
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
[master cd35afc] ml_store
 1 file changed, 326 insertions(+)
To github.com:arita37/mlmodels_store.git
   ee36a24..cd35afc  master -> master





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

13/13 [==============================] - 1s 111ms/step - loss: nan
Epoch 2/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 3/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 4/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 5/10

13/13 [==============================] - 0s 5ms/step - loss: nan
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
[master c969729] ml_store
 1 file changed, 125 insertions(+)
To github.com:arita37/mlmodels_store.git
   cd35afc..c969729  master -> master





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

    8192/11490434 [..............................] - ETA: 1s
 1556480/11490434 [===>..........................] - ETA: 0s
 6873088/11490434 [================>.............] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:17 - loss: 2.3092 - categorical_accuracy: 0.1250
   64/60000 [..............................] - ETA: 4:43 - loss: 2.2925 - categorical_accuracy: 0.1562
   96/60000 [..............................] - ETA: 3:51 - loss: 2.2923 - categorical_accuracy: 0.1875
  128/60000 [..............................] - ETA: 3:20 - loss: 2.2655 - categorical_accuracy: 0.2266
  160/60000 [..............................] - ETA: 3:02 - loss: 2.2179 - categorical_accuracy: 0.2313
  192/60000 [..............................] - ETA: 2:49 - loss: 2.1779 - categorical_accuracy: 0.2500
  224/60000 [..............................] - ETA: 2:41 - loss: 2.1515 - categorical_accuracy: 0.2545
  256/60000 [..............................] - ETA: 2:36 - loss: 2.0975 - categorical_accuracy: 0.2695
  288/60000 [..............................] - ETA: 2:31 - loss: 2.0770 - categorical_accuracy: 0.2917
  320/60000 [..............................] - ETA: 2:26 - loss: 2.0350 - categorical_accuracy: 0.3094
  352/60000 [..............................] - ETA: 2:22 - loss: 2.0052 - categorical_accuracy: 0.3267
  384/60000 [..............................] - ETA: 2:19 - loss: 1.9596 - categorical_accuracy: 0.3411
  416/60000 [..............................] - ETA: 2:17 - loss: 1.9346 - categorical_accuracy: 0.3462
  448/60000 [..............................] - ETA: 2:14 - loss: 1.8907 - categorical_accuracy: 0.3616
  480/60000 [..............................] - ETA: 2:12 - loss: 1.8650 - categorical_accuracy: 0.3729
  512/60000 [..............................] - ETA: 2:10 - loss: 1.8341 - categorical_accuracy: 0.3848
  544/60000 [..............................] - ETA: 2:09 - loss: 1.8174 - categorical_accuracy: 0.3879
  576/60000 [..............................] - ETA: 2:07 - loss: 1.7841 - categorical_accuracy: 0.3993
  608/60000 [..............................] - ETA: 2:06 - loss: 1.7494 - categorical_accuracy: 0.4145
  640/60000 [..............................] - ETA: 2:05 - loss: 1.7211 - categorical_accuracy: 0.4234
  672/60000 [..............................] - ETA: 2:04 - loss: 1.6996 - categorical_accuracy: 0.4301
  704/60000 [..............................] - ETA: 2:03 - loss: 1.6654 - categorical_accuracy: 0.4418
  736/60000 [..............................] - ETA: 2:03 - loss: 1.6332 - categorical_accuracy: 0.4511
  768/60000 [..............................] - ETA: 2:02 - loss: 1.5994 - categorical_accuracy: 0.4635
  800/60000 [..............................] - ETA: 2:01 - loss: 1.5733 - categorical_accuracy: 0.4712
  832/60000 [..............................] - ETA: 2:00 - loss: 1.5402 - categorical_accuracy: 0.4832
  864/60000 [..............................] - ETA: 2:00 - loss: 1.5376 - categorical_accuracy: 0.4850
  896/60000 [..............................] - ETA: 1:59 - loss: 1.5240 - categorical_accuracy: 0.4888
  928/60000 [..............................] - ETA: 1:59 - loss: 1.5046 - categorical_accuracy: 0.4968
  960/60000 [..............................] - ETA: 1:58 - loss: 1.4840 - categorical_accuracy: 0.5042
  992/60000 [..............................] - ETA: 1:57 - loss: 1.4623 - categorical_accuracy: 0.5101
 1024/60000 [..............................] - ETA: 1:57 - loss: 1.4425 - categorical_accuracy: 0.5176
 1056/60000 [..............................] - ETA: 1:56 - loss: 1.4517 - categorical_accuracy: 0.5189
 1088/60000 [..............................] - ETA: 1:56 - loss: 1.4388 - categorical_accuracy: 0.5230
 1120/60000 [..............................] - ETA: 1:56 - loss: 1.4233 - categorical_accuracy: 0.5312
 1152/60000 [..............................] - ETA: 1:56 - loss: 1.4154 - categorical_accuracy: 0.5347
 1184/60000 [..............................] - ETA: 1:55 - loss: 1.4053 - categorical_accuracy: 0.5380
 1216/60000 [..............................] - ETA: 1:55 - loss: 1.3864 - categorical_accuracy: 0.5461
 1248/60000 [..............................] - ETA: 1:54 - loss: 1.3655 - categorical_accuracy: 0.5545
 1280/60000 [..............................] - ETA: 1:54 - loss: 1.3592 - categorical_accuracy: 0.5562
 1312/60000 [..............................] - ETA: 1:54 - loss: 1.3401 - categorical_accuracy: 0.5633
 1344/60000 [..............................] - ETA: 1:54 - loss: 1.3260 - categorical_accuracy: 0.5685
 1376/60000 [..............................] - ETA: 1:53 - loss: 1.3176 - categorical_accuracy: 0.5705
 1408/60000 [..............................] - ETA: 1:54 - loss: 1.3103 - categorical_accuracy: 0.5717
 1440/60000 [..............................] - ETA: 1:54 - loss: 1.2919 - categorical_accuracy: 0.5778
 1472/60000 [..............................] - ETA: 1:53 - loss: 1.2794 - categorical_accuracy: 0.5822
 1504/60000 [..............................] - ETA: 1:53 - loss: 1.2717 - categorical_accuracy: 0.5858
 1536/60000 [..............................] - ETA: 1:53 - loss: 1.2630 - categorical_accuracy: 0.5872
 1568/60000 [..............................] - ETA: 1:53 - loss: 1.2516 - categorical_accuracy: 0.5912
 1600/60000 [..............................] - ETA: 1:52 - loss: 1.2422 - categorical_accuracy: 0.5950
 1632/60000 [..............................] - ETA: 1:52 - loss: 1.2396 - categorical_accuracy: 0.5962
 1664/60000 [..............................] - ETA: 1:53 - loss: 1.2279 - categorical_accuracy: 0.6010
 1696/60000 [..............................] - ETA: 1:52 - loss: 1.2177 - categorical_accuracy: 0.6038
 1728/60000 [..............................] - ETA: 1:52 - loss: 1.2070 - categorical_accuracy: 0.6076
 1760/60000 [..............................] - ETA: 1:52 - loss: 1.1931 - categorical_accuracy: 0.6119
 1792/60000 [..............................] - ETA: 1:52 - loss: 1.1846 - categorical_accuracy: 0.6144
 1824/60000 [..............................] - ETA: 1:52 - loss: 1.1742 - categorical_accuracy: 0.6179
 1856/60000 [..............................] - ETA: 1:52 - loss: 1.1579 - categorical_accuracy: 0.6245
 1888/60000 [..............................] - ETA: 1:51 - loss: 1.1468 - categorical_accuracy: 0.6287
 1920/60000 [..............................] - ETA: 1:51 - loss: 1.1408 - categorical_accuracy: 0.6313
 1952/60000 [..............................] - ETA: 1:51 - loss: 1.1319 - categorical_accuracy: 0.6332
 1984/60000 [..............................] - ETA: 1:51 - loss: 1.1250 - categorical_accuracy: 0.6341
 2016/60000 [>.............................] - ETA: 1:51 - loss: 1.1108 - categorical_accuracy: 0.6394
 2048/60000 [>.............................] - ETA: 1:51 - loss: 1.0984 - categorical_accuracy: 0.6431
 2080/60000 [>.............................] - ETA: 1:50 - loss: 1.0890 - categorical_accuracy: 0.6457
 2112/60000 [>.............................] - ETA: 1:50 - loss: 1.0788 - categorical_accuracy: 0.6487
 2144/60000 [>.............................] - ETA: 1:50 - loss: 1.0677 - categorical_accuracy: 0.6521
 2176/60000 [>.............................] - ETA: 1:50 - loss: 1.0616 - categorical_accuracy: 0.6535
 2208/60000 [>.............................] - ETA: 1:50 - loss: 1.0568 - categorical_accuracy: 0.6544
 2240/60000 [>.............................] - ETA: 1:50 - loss: 1.0457 - categorical_accuracy: 0.6580
 2272/60000 [>.............................] - ETA: 1:50 - loss: 1.0372 - categorical_accuracy: 0.6593
 2304/60000 [>.............................] - ETA: 1:49 - loss: 1.0309 - categorical_accuracy: 0.6606
 2336/60000 [>.............................] - ETA: 1:49 - loss: 1.0218 - categorical_accuracy: 0.6640
 2368/60000 [>.............................] - ETA: 1:49 - loss: 1.0128 - categorical_accuracy: 0.6664
 2400/60000 [>.............................] - ETA: 1:49 - loss: 1.0062 - categorical_accuracy: 0.6683
 2432/60000 [>.............................] - ETA: 1:49 - loss: 1.0132 - categorical_accuracy: 0.6674
 2464/60000 [>.............................] - ETA: 1:49 - loss: 1.0052 - categorical_accuracy: 0.6705
 2496/60000 [>.............................] - ETA: 1:48 - loss: 0.9991 - categorical_accuracy: 0.6723
 2528/60000 [>.............................] - ETA: 1:48 - loss: 0.9934 - categorical_accuracy: 0.6741
 2560/60000 [>.............................] - ETA: 1:48 - loss: 0.9835 - categorical_accuracy: 0.6777
 2592/60000 [>.............................] - ETA: 1:48 - loss: 0.9832 - categorical_accuracy: 0.6786
 2624/60000 [>.............................] - ETA: 1:48 - loss: 0.9779 - categorical_accuracy: 0.6810
 2656/60000 [>.............................] - ETA: 1:48 - loss: 0.9726 - categorical_accuracy: 0.6822
 2688/60000 [>.............................] - ETA: 1:48 - loss: 0.9666 - categorical_accuracy: 0.6842
 2720/60000 [>.............................] - ETA: 1:48 - loss: 0.9596 - categorical_accuracy: 0.6864
 2752/60000 [>.............................] - ETA: 1:47 - loss: 0.9548 - categorical_accuracy: 0.6882
 2784/60000 [>.............................] - ETA: 1:47 - loss: 0.9512 - categorical_accuracy: 0.6893
 2816/60000 [>.............................] - ETA: 1:47 - loss: 0.9495 - categorical_accuracy: 0.6900
 2848/60000 [>.............................] - ETA: 1:47 - loss: 0.9482 - categorical_accuracy: 0.6903
 2880/60000 [>.............................] - ETA: 1:47 - loss: 0.9436 - categorical_accuracy: 0.6920
 2912/60000 [>.............................] - ETA: 1:47 - loss: 0.9387 - categorical_accuracy: 0.6933
 2944/60000 [>.............................] - ETA: 1:46 - loss: 0.9325 - categorical_accuracy: 0.6960
 2976/60000 [>.............................] - ETA: 1:46 - loss: 0.9288 - categorical_accuracy: 0.6976
 3008/60000 [>.............................] - ETA: 1:46 - loss: 0.9229 - categorical_accuracy: 0.6991
 3040/60000 [>.............................] - ETA: 1:46 - loss: 0.9209 - categorical_accuracy: 0.7003
 3072/60000 [>.............................] - ETA: 1:46 - loss: 0.9145 - categorical_accuracy: 0.7025
 3104/60000 [>.............................] - ETA: 1:46 - loss: 0.9096 - categorical_accuracy: 0.7036
 3136/60000 [>.............................] - ETA: 1:46 - loss: 0.9051 - categorical_accuracy: 0.7057
 3168/60000 [>.............................] - ETA: 1:46 - loss: 0.9004 - categorical_accuracy: 0.7071
 3200/60000 [>.............................] - ETA: 1:46 - loss: 0.8975 - categorical_accuracy: 0.7084
 3232/60000 [>.............................] - ETA: 1:45 - loss: 0.8942 - categorical_accuracy: 0.7101
 3264/60000 [>.............................] - ETA: 1:45 - loss: 0.8884 - categorical_accuracy: 0.7120
 3296/60000 [>.............................] - ETA: 1:45 - loss: 0.8829 - categorical_accuracy: 0.7139
 3328/60000 [>.............................] - ETA: 1:45 - loss: 0.8799 - categorical_accuracy: 0.7142
 3360/60000 [>.............................] - ETA: 1:45 - loss: 0.8763 - categorical_accuracy: 0.7152
 3392/60000 [>.............................] - ETA: 1:45 - loss: 0.8702 - categorical_accuracy: 0.7170
 3424/60000 [>.............................] - ETA: 1:45 - loss: 0.8649 - categorical_accuracy: 0.7185
 3456/60000 [>.............................] - ETA: 1:45 - loss: 0.8601 - categorical_accuracy: 0.7193
 3488/60000 [>.............................] - ETA: 1:45 - loss: 0.8549 - categorical_accuracy: 0.7208
 3520/60000 [>.............................] - ETA: 1:45 - loss: 0.8567 - categorical_accuracy: 0.7207
 3552/60000 [>.............................] - ETA: 1:45 - loss: 0.8515 - categorical_accuracy: 0.7230
 3584/60000 [>.............................] - ETA: 1:45 - loss: 0.8464 - categorical_accuracy: 0.7243
 3616/60000 [>.............................] - ETA: 1:45 - loss: 0.8439 - categorical_accuracy: 0.7251
 3648/60000 [>.............................] - ETA: 1:45 - loss: 0.8401 - categorical_accuracy: 0.7262
 3680/60000 [>.............................] - ETA: 1:44 - loss: 0.8368 - categorical_accuracy: 0.7266
 3712/60000 [>.............................] - ETA: 1:44 - loss: 0.8336 - categorical_accuracy: 0.7276
 3744/60000 [>.............................] - ETA: 1:44 - loss: 0.8293 - categorical_accuracy: 0.7292
 3776/60000 [>.............................] - ETA: 1:44 - loss: 0.8266 - categorical_accuracy: 0.7296
 3808/60000 [>.............................] - ETA: 1:44 - loss: 0.8214 - categorical_accuracy: 0.7314
 3840/60000 [>.............................] - ETA: 1:44 - loss: 0.8179 - categorical_accuracy: 0.7323
 3872/60000 [>.............................] - ETA: 1:44 - loss: 0.8163 - categorical_accuracy: 0.7322
 3904/60000 [>.............................] - ETA: 1:44 - loss: 0.8114 - categorical_accuracy: 0.7341
 3936/60000 [>.............................] - ETA: 1:44 - loss: 0.8074 - categorical_accuracy: 0.7355
 3968/60000 [>.............................] - ETA: 1:44 - loss: 0.8043 - categorical_accuracy: 0.7369
 4000/60000 [=>............................] - ETA: 1:43 - loss: 0.8033 - categorical_accuracy: 0.7370
 4032/60000 [=>............................] - ETA: 1:43 - loss: 0.8015 - categorical_accuracy: 0.7366
 4064/60000 [=>............................] - ETA: 1:43 - loss: 0.7981 - categorical_accuracy: 0.7379
 4096/60000 [=>............................] - ETA: 1:43 - loss: 0.7941 - categorical_accuracy: 0.7395
 4128/60000 [=>............................] - ETA: 1:43 - loss: 0.7900 - categorical_accuracy: 0.7413
 4160/60000 [=>............................] - ETA: 1:43 - loss: 0.7852 - categorical_accuracy: 0.7428
 4192/60000 [=>............................] - ETA: 1:43 - loss: 0.7822 - categorical_accuracy: 0.7436
 4224/60000 [=>............................] - ETA: 1:43 - loss: 0.7777 - categorical_accuracy: 0.7448
 4256/60000 [=>............................] - ETA: 1:43 - loss: 0.7745 - categorical_accuracy: 0.7458
 4288/60000 [=>............................] - ETA: 1:43 - loss: 0.7702 - categorical_accuracy: 0.7472
 4320/60000 [=>............................] - ETA: 1:42 - loss: 0.7669 - categorical_accuracy: 0.7481
 4352/60000 [=>............................] - ETA: 1:42 - loss: 0.7666 - categorical_accuracy: 0.7489
 4384/60000 [=>............................] - ETA: 1:42 - loss: 0.7631 - categorical_accuracy: 0.7505
 4416/60000 [=>............................] - ETA: 1:42 - loss: 0.7612 - categorical_accuracy: 0.7507
 4448/60000 [=>............................] - ETA: 1:42 - loss: 0.7603 - categorical_accuracy: 0.7513
 4480/60000 [=>............................] - ETA: 1:42 - loss: 0.7569 - categorical_accuracy: 0.7522
 4512/60000 [=>............................] - ETA: 1:42 - loss: 0.7535 - categorical_accuracy: 0.7533
 4544/60000 [=>............................] - ETA: 1:42 - loss: 0.7508 - categorical_accuracy: 0.7540
 4576/60000 [=>............................] - ETA: 1:42 - loss: 0.7466 - categorical_accuracy: 0.7550
 4608/60000 [=>............................] - ETA: 1:42 - loss: 0.7422 - categorical_accuracy: 0.7565
 4640/60000 [=>............................] - ETA: 1:42 - loss: 0.7396 - categorical_accuracy: 0.7573
 4672/60000 [=>............................] - ETA: 1:41 - loss: 0.7368 - categorical_accuracy: 0.7579
 4704/60000 [=>............................] - ETA: 1:41 - loss: 0.7339 - categorical_accuracy: 0.7594
 4736/60000 [=>............................] - ETA: 1:41 - loss: 0.7317 - categorical_accuracy: 0.7601
 4768/60000 [=>............................] - ETA: 1:41 - loss: 0.7284 - categorical_accuracy: 0.7613
 4800/60000 [=>............................] - ETA: 1:41 - loss: 0.7249 - categorical_accuracy: 0.7623
 4832/60000 [=>............................] - ETA: 1:41 - loss: 0.7250 - categorical_accuracy: 0.7622
 4864/60000 [=>............................] - ETA: 1:41 - loss: 0.7224 - categorical_accuracy: 0.7632
 4896/60000 [=>............................] - ETA: 1:41 - loss: 0.7197 - categorical_accuracy: 0.7637
 4928/60000 [=>............................] - ETA: 1:41 - loss: 0.7181 - categorical_accuracy: 0.7636
 4960/60000 [=>............................] - ETA: 1:41 - loss: 0.7162 - categorical_accuracy: 0.7643
 4992/60000 [=>............................] - ETA: 1:41 - loss: 0.7125 - categorical_accuracy: 0.7656
 5024/60000 [=>............................] - ETA: 1:41 - loss: 0.7092 - categorical_accuracy: 0.7667
 5056/60000 [=>............................] - ETA: 1:41 - loss: 0.7072 - categorical_accuracy: 0.7672
 5088/60000 [=>............................] - ETA: 1:40 - loss: 0.7043 - categorical_accuracy: 0.7683
 5120/60000 [=>............................] - ETA: 1:40 - loss: 0.7013 - categorical_accuracy: 0.7691
 5152/60000 [=>............................] - ETA: 1:40 - loss: 0.6988 - categorical_accuracy: 0.7700
 5184/60000 [=>............................] - ETA: 1:40 - loss: 0.6950 - categorical_accuracy: 0.7714
 5216/60000 [=>............................] - ETA: 1:40 - loss: 0.6921 - categorical_accuracy: 0.7726
 5248/60000 [=>............................] - ETA: 1:40 - loss: 0.6896 - categorical_accuracy: 0.7738
 5280/60000 [=>............................] - ETA: 1:40 - loss: 0.6879 - categorical_accuracy: 0.7744
 5312/60000 [=>............................] - ETA: 1:40 - loss: 0.6859 - categorical_accuracy: 0.7750
 5344/60000 [=>............................] - ETA: 1:40 - loss: 0.6848 - categorical_accuracy: 0.7758
 5376/60000 [=>............................] - ETA: 1:40 - loss: 0.6834 - categorical_accuracy: 0.7766
 5408/60000 [=>............................] - ETA: 1:40 - loss: 0.6806 - categorical_accuracy: 0.7776
 5440/60000 [=>............................] - ETA: 1:40 - loss: 0.6799 - categorical_accuracy: 0.7776
 5472/60000 [=>............................] - ETA: 1:40 - loss: 0.6765 - categorical_accuracy: 0.7787
 5504/60000 [=>............................] - ETA: 1:40 - loss: 0.6760 - categorical_accuracy: 0.7791
 5536/60000 [=>............................] - ETA: 1:40 - loss: 0.6737 - categorical_accuracy: 0.7794
 5568/60000 [=>............................] - ETA: 1:40 - loss: 0.6706 - categorical_accuracy: 0.7805
 5600/60000 [=>............................] - ETA: 1:40 - loss: 0.6675 - categorical_accuracy: 0.7816
 5632/60000 [=>............................] - ETA: 1:40 - loss: 0.6657 - categorical_accuracy: 0.7823
 5664/60000 [=>............................] - ETA: 1:39 - loss: 0.6628 - categorical_accuracy: 0.7834
 5696/60000 [=>............................] - ETA: 1:39 - loss: 0.6616 - categorical_accuracy: 0.7839
 5728/60000 [=>............................] - ETA: 1:39 - loss: 0.6589 - categorical_accuracy: 0.7849
 5760/60000 [=>............................] - ETA: 1:39 - loss: 0.6561 - categorical_accuracy: 0.7859
 5792/60000 [=>............................] - ETA: 1:39 - loss: 0.6532 - categorical_accuracy: 0.7869
 5824/60000 [=>............................] - ETA: 1:39 - loss: 0.6517 - categorical_accuracy: 0.7874
 5856/60000 [=>............................] - ETA: 1:39 - loss: 0.6489 - categorical_accuracy: 0.7884
 5888/60000 [=>............................] - ETA: 1:39 - loss: 0.6468 - categorical_accuracy: 0.7892
 5920/60000 [=>............................] - ETA: 1:39 - loss: 0.6452 - categorical_accuracy: 0.7897
 5952/60000 [=>............................] - ETA: 1:39 - loss: 0.6427 - categorical_accuracy: 0.7907
 5984/60000 [=>............................] - ETA: 1:39 - loss: 0.6404 - categorical_accuracy: 0.7913
 6016/60000 [==>...........................] - ETA: 1:39 - loss: 0.6401 - categorical_accuracy: 0.7917
 6048/60000 [==>...........................] - ETA: 1:39 - loss: 0.6398 - categorical_accuracy: 0.7920
 6080/60000 [==>...........................] - ETA: 1:38 - loss: 0.6377 - categorical_accuracy: 0.7928
 6112/60000 [==>...........................] - ETA: 1:38 - loss: 0.6352 - categorical_accuracy: 0.7937
 6144/60000 [==>...........................] - ETA: 1:38 - loss: 0.6333 - categorical_accuracy: 0.7941
 6176/60000 [==>...........................] - ETA: 1:38 - loss: 0.6313 - categorical_accuracy: 0.7947
 6208/60000 [==>...........................] - ETA: 1:38 - loss: 0.6293 - categorical_accuracy: 0.7951
 6240/60000 [==>...........................] - ETA: 1:38 - loss: 0.6268 - categorical_accuracy: 0.7960
 6272/60000 [==>...........................] - ETA: 1:38 - loss: 0.6258 - categorical_accuracy: 0.7966
 6304/60000 [==>...........................] - ETA: 1:38 - loss: 0.6237 - categorical_accuracy: 0.7970
 6336/60000 [==>...........................] - ETA: 1:38 - loss: 0.6232 - categorical_accuracy: 0.7973
 6368/60000 [==>...........................] - ETA: 1:38 - loss: 0.6208 - categorical_accuracy: 0.7982
 6400/60000 [==>...........................] - ETA: 1:38 - loss: 0.6187 - categorical_accuracy: 0.7987
 6432/60000 [==>...........................] - ETA: 1:38 - loss: 0.6185 - categorical_accuracy: 0.7991
 6464/60000 [==>...........................] - ETA: 1:38 - loss: 0.6173 - categorical_accuracy: 0.7997
 6496/60000 [==>...........................] - ETA: 1:38 - loss: 0.6155 - categorical_accuracy: 0.8000
 6528/60000 [==>...........................] - ETA: 1:38 - loss: 0.6150 - categorical_accuracy: 0.8001
 6560/60000 [==>...........................] - ETA: 1:37 - loss: 0.6124 - categorical_accuracy: 0.8008
 6592/60000 [==>...........................] - ETA: 1:37 - loss: 0.6107 - categorical_accuracy: 0.8014
 6624/60000 [==>...........................] - ETA: 1:37 - loss: 0.6106 - categorical_accuracy: 0.8016
 6656/60000 [==>...........................] - ETA: 1:37 - loss: 0.6089 - categorical_accuracy: 0.8021
 6688/60000 [==>...........................] - ETA: 1:37 - loss: 0.6076 - categorical_accuracy: 0.8026
 6720/60000 [==>...........................] - ETA: 1:37 - loss: 0.6065 - categorical_accuracy: 0.8027
 6752/60000 [==>...........................] - ETA: 1:37 - loss: 0.6051 - categorical_accuracy: 0.8032
 6784/60000 [==>...........................] - ETA: 1:37 - loss: 0.6041 - categorical_accuracy: 0.8032
 6816/60000 [==>...........................] - ETA: 1:37 - loss: 0.6023 - categorical_accuracy: 0.8036
 6848/60000 [==>...........................] - ETA: 1:37 - loss: 0.6000 - categorical_accuracy: 0.8043
 6880/60000 [==>...........................] - ETA: 1:37 - loss: 0.5983 - categorical_accuracy: 0.8049
 6912/60000 [==>...........................] - ETA: 1:37 - loss: 0.5966 - categorical_accuracy: 0.8056
 6944/60000 [==>...........................] - ETA: 1:37 - loss: 0.5955 - categorical_accuracy: 0.8057
 6976/60000 [==>...........................] - ETA: 1:37 - loss: 0.5939 - categorical_accuracy: 0.8063
 7008/60000 [==>...........................] - ETA: 1:37 - loss: 0.5921 - categorical_accuracy: 0.8068
 7040/60000 [==>...........................] - ETA: 1:36 - loss: 0.5921 - categorical_accuracy: 0.8070
 7072/60000 [==>...........................] - ETA: 1:36 - loss: 0.5902 - categorical_accuracy: 0.8074
 7104/60000 [==>...........................] - ETA: 1:36 - loss: 0.5898 - categorical_accuracy: 0.8079
 7136/60000 [==>...........................] - ETA: 1:36 - loss: 0.5889 - categorical_accuracy: 0.8082
 7168/60000 [==>...........................] - ETA: 1:36 - loss: 0.5869 - categorical_accuracy: 0.8087
 7200/60000 [==>...........................] - ETA: 1:36 - loss: 0.5849 - categorical_accuracy: 0.8094
 7232/60000 [==>...........................] - ETA: 1:36 - loss: 0.5830 - categorical_accuracy: 0.8101
 7264/60000 [==>...........................] - ETA: 1:36 - loss: 0.5810 - categorical_accuracy: 0.8108
 7296/60000 [==>...........................] - ETA: 1:36 - loss: 0.5798 - categorical_accuracy: 0.8113
 7328/60000 [==>...........................] - ETA: 1:36 - loss: 0.5784 - categorical_accuracy: 0.8118
 7360/60000 [==>...........................] - ETA: 1:36 - loss: 0.5774 - categorical_accuracy: 0.8124
 7392/60000 [==>...........................] - ETA: 1:36 - loss: 0.5766 - categorical_accuracy: 0.8129
 7424/60000 [==>...........................] - ETA: 1:36 - loss: 0.5756 - categorical_accuracy: 0.8134
 7456/60000 [==>...........................] - ETA: 1:36 - loss: 0.5740 - categorical_accuracy: 0.8140
 7488/60000 [==>...........................] - ETA: 1:35 - loss: 0.5731 - categorical_accuracy: 0.8144
 7520/60000 [==>...........................] - ETA: 1:35 - loss: 0.5709 - categorical_accuracy: 0.8152
 7552/60000 [==>...........................] - ETA: 1:35 - loss: 0.5698 - categorical_accuracy: 0.8157
 7584/60000 [==>...........................] - ETA: 1:35 - loss: 0.5690 - categorical_accuracy: 0.8158
 7616/60000 [==>...........................] - ETA: 1:35 - loss: 0.5673 - categorical_accuracy: 0.8163
 7648/60000 [==>...........................] - ETA: 1:35 - loss: 0.5654 - categorical_accuracy: 0.8168
 7680/60000 [==>...........................] - ETA: 1:35 - loss: 0.5652 - categorical_accuracy: 0.8171
 7712/60000 [==>...........................] - ETA: 1:35 - loss: 0.5636 - categorical_accuracy: 0.8176
 7744/60000 [==>...........................] - ETA: 1:35 - loss: 0.5623 - categorical_accuracy: 0.8182
 7776/60000 [==>...........................] - ETA: 1:35 - loss: 0.5602 - categorical_accuracy: 0.8189
 7808/60000 [==>...........................] - ETA: 1:35 - loss: 0.5585 - categorical_accuracy: 0.8195
 7840/60000 [==>...........................] - ETA: 1:35 - loss: 0.5568 - categorical_accuracy: 0.8200
 7872/60000 [==>...........................] - ETA: 1:35 - loss: 0.5558 - categorical_accuracy: 0.8204
 7904/60000 [==>...........................] - ETA: 1:35 - loss: 0.5544 - categorical_accuracy: 0.8209
 7936/60000 [==>...........................] - ETA: 1:35 - loss: 0.5525 - categorical_accuracy: 0.8214
 7968/60000 [==>...........................] - ETA: 1:34 - loss: 0.5516 - categorical_accuracy: 0.8218
 8000/60000 [===>..........................] - ETA: 1:34 - loss: 0.5495 - categorical_accuracy: 0.8225
 8032/60000 [===>..........................] - ETA: 1:34 - loss: 0.5475 - categorical_accuracy: 0.8232
 8064/60000 [===>..........................] - ETA: 1:34 - loss: 0.5472 - categorical_accuracy: 0.8234
 8096/60000 [===>..........................] - ETA: 1:34 - loss: 0.5456 - categorical_accuracy: 0.8240
 8128/60000 [===>..........................] - ETA: 1:34 - loss: 0.5440 - categorical_accuracy: 0.8246
 8160/60000 [===>..........................] - ETA: 1:34 - loss: 0.5424 - categorical_accuracy: 0.8250
 8192/60000 [===>..........................] - ETA: 1:34 - loss: 0.5411 - categorical_accuracy: 0.8253
 8224/60000 [===>..........................] - ETA: 1:34 - loss: 0.5397 - categorical_accuracy: 0.8256
 8256/60000 [===>..........................] - ETA: 1:34 - loss: 0.5386 - categorical_accuracy: 0.8259
 8288/60000 [===>..........................] - ETA: 1:34 - loss: 0.5366 - categorical_accuracy: 0.8266
 8320/60000 [===>..........................] - ETA: 1:34 - loss: 0.5357 - categorical_accuracy: 0.8268
 8352/60000 [===>..........................] - ETA: 1:34 - loss: 0.5347 - categorical_accuracy: 0.8271
 8384/60000 [===>..........................] - ETA: 1:34 - loss: 0.5335 - categorical_accuracy: 0.8275
 8416/60000 [===>..........................] - ETA: 1:34 - loss: 0.5323 - categorical_accuracy: 0.8281
 8448/60000 [===>..........................] - ETA: 1:34 - loss: 0.5307 - categorical_accuracy: 0.8287
 8480/60000 [===>..........................] - ETA: 1:33 - loss: 0.5297 - categorical_accuracy: 0.8289
 8512/60000 [===>..........................] - ETA: 1:33 - loss: 0.5287 - categorical_accuracy: 0.8293
 8544/60000 [===>..........................] - ETA: 1:33 - loss: 0.5269 - categorical_accuracy: 0.8299
 8576/60000 [===>..........................] - ETA: 1:33 - loss: 0.5261 - categorical_accuracy: 0.8305
 8608/60000 [===>..........................] - ETA: 1:33 - loss: 0.5255 - categorical_accuracy: 0.8305
 8640/60000 [===>..........................] - ETA: 1:33 - loss: 0.5238 - categorical_accuracy: 0.8311
 8672/60000 [===>..........................] - ETA: 1:33 - loss: 0.5225 - categorical_accuracy: 0.8315
 8704/60000 [===>..........................] - ETA: 1:33 - loss: 0.5217 - categorical_accuracy: 0.8316
 8736/60000 [===>..........................] - ETA: 1:33 - loss: 0.5213 - categorical_accuracy: 0.8317
 8768/60000 [===>..........................] - ETA: 1:33 - loss: 0.5200 - categorical_accuracy: 0.8322
 8800/60000 [===>..........................] - ETA: 1:33 - loss: 0.5182 - categorical_accuracy: 0.8328
 8832/60000 [===>..........................] - ETA: 1:33 - loss: 0.5186 - categorical_accuracy: 0.8329
 8864/60000 [===>..........................] - ETA: 1:33 - loss: 0.5174 - categorical_accuracy: 0.8333
 8896/60000 [===>..........................] - ETA: 1:33 - loss: 0.5162 - categorical_accuracy: 0.8337
 8928/60000 [===>..........................] - ETA: 1:33 - loss: 0.5149 - categorical_accuracy: 0.8341
 8960/60000 [===>..........................] - ETA: 1:32 - loss: 0.5141 - categorical_accuracy: 0.8343
 8992/60000 [===>..........................] - ETA: 1:32 - loss: 0.5140 - categorical_accuracy: 0.8342
 9024/60000 [===>..........................] - ETA: 1:32 - loss: 0.5129 - categorical_accuracy: 0.8346
 9056/60000 [===>..........................] - ETA: 1:32 - loss: 0.5125 - categorical_accuracy: 0.8346
 9088/60000 [===>..........................] - ETA: 1:32 - loss: 0.5121 - categorical_accuracy: 0.8347
 9120/60000 [===>..........................] - ETA: 1:32 - loss: 0.5109 - categorical_accuracy: 0.8350
 9152/60000 [===>..........................] - ETA: 1:32 - loss: 0.5097 - categorical_accuracy: 0.8354
 9184/60000 [===>..........................] - ETA: 1:32 - loss: 0.5085 - categorical_accuracy: 0.8358
 9216/60000 [===>..........................] - ETA: 1:32 - loss: 0.5081 - categorical_accuracy: 0.8363
 9248/60000 [===>..........................] - ETA: 1:32 - loss: 0.5069 - categorical_accuracy: 0.8366
 9280/60000 [===>..........................] - ETA: 1:32 - loss: 0.5053 - categorical_accuracy: 0.8372
 9312/60000 [===>..........................] - ETA: 1:32 - loss: 0.5039 - categorical_accuracy: 0.8375
 9344/60000 [===>..........................] - ETA: 1:32 - loss: 0.5029 - categorical_accuracy: 0.8379
 9376/60000 [===>..........................] - ETA: 1:32 - loss: 0.5020 - categorical_accuracy: 0.8379
 9408/60000 [===>..........................] - ETA: 1:32 - loss: 0.5016 - categorical_accuracy: 0.8378
 9440/60000 [===>..........................] - ETA: 1:32 - loss: 0.5009 - categorical_accuracy: 0.8380
 9472/60000 [===>..........................] - ETA: 1:31 - loss: 0.4996 - categorical_accuracy: 0.8385
 9504/60000 [===>..........................] - ETA: 1:31 - loss: 0.4981 - categorical_accuracy: 0.8390
 9536/60000 [===>..........................] - ETA: 1:31 - loss: 0.4967 - categorical_accuracy: 0.8396
 9568/60000 [===>..........................] - ETA: 1:31 - loss: 0.4951 - categorical_accuracy: 0.8401
 9600/60000 [===>..........................] - ETA: 1:31 - loss: 0.4939 - categorical_accuracy: 0.8405
 9632/60000 [===>..........................] - ETA: 1:31 - loss: 0.4928 - categorical_accuracy: 0.8408
 9664/60000 [===>..........................] - ETA: 1:31 - loss: 0.4916 - categorical_accuracy: 0.8412
 9696/60000 [===>..........................] - ETA: 1:31 - loss: 0.4916 - categorical_accuracy: 0.8412
 9728/60000 [===>..........................] - ETA: 1:31 - loss: 0.4909 - categorical_accuracy: 0.8412
 9760/60000 [===>..........................] - ETA: 1:31 - loss: 0.4895 - categorical_accuracy: 0.8417
 9792/60000 [===>..........................] - ETA: 1:31 - loss: 0.4884 - categorical_accuracy: 0.8419
 9824/60000 [===>..........................] - ETA: 1:31 - loss: 0.4874 - categorical_accuracy: 0.8422
 9856/60000 [===>..........................] - ETA: 1:31 - loss: 0.4860 - categorical_accuracy: 0.8426
 9888/60000 [===>..........................] - ETA: 1:31 - loss: 0.4854 - categorical_accuracy: 0.8429
 9920/60000 [===>..........................] - ETA: 1:31 - loss: 0.4846 - categorical_accuracy: 0.8432
 9952/60000 [===>..........................] - ETA: 1:31 - loss: 0.4847 - categorical_accuracy: 0.8433
 9984/60000 [===>..........................] - ETA: 1:31 - loss: 0.4846 - categorical_accuracy: 0.8433
10016/60000 [====>.........................] - ETA: 1:31 - loss: 0.4850 - categorical_accuracy: 0.8434
10048/60000 [====>.........................] - ETA: 1:30 - loss: 0.4837 - categorical_accuracy: 0.8438
10080/60000 [====>.........................] - ETA: 1:30 - loss: 0.4829 - categorical_accuracy: 0.8439
10112/60000 [====>.........................] - ETA: 1:30 - loss: 0.4828 - categorical_accuracy: 0.8441
10144/60000 [====>.........................] - ETA: 1:30 - loss: 0.4815 - categorical_accuracy: 0.8446
10176/60000 [====>.........................] - ETA: 1:30 - loss: 0.4808 - categorical_accuracy: 0.8448
10208/60000 [====>.........................] - ETA: 1:30 - loss: 0.4797 - categorical_accuracy: 0.8452
10240/60000 [====>.........................] - ETA: 1:30 - loss: 0.4789 - categorical_accuracy: 0.8454
10272/60000 [====>.........................] - ETA: 1:30 - loss: 0.4776 - categorical_accuracy: 0.8459
10304/60000 [====>.........................] - ETA: 1:30 - loss: 0.4778 - categorical_accuracy: 0.8459
10336/60000 [====>.........................] - ETA: 1:30 - loss: 0.4766 - categorical_accuracy: 0.8463
10368/60000 [====>.........................] - ETA: 1:30 - loss: 0.4758 - categorical_accuracy: 0.8465
10400/60000 [====>.........................] - ETA: 1:30 - loss: 0.4745 - categorical_accuracy: 0.8470
10432/60000 [====>.........................] - ETA: 1:30 - loss: 0.4733 - categorical_accuracy: 0.8474
10464/60000 [====>.........................] - ETA: 1:30 - loss: 0.4730 - categorical_accuracy: 0.8476
10496/60000 [====>.........................] - ETA: 1:30 - loss: 0.4720 - categorical_accuracy: 0.8479
10528/60000 [====>.........................] - ETA: 1:30 - loss: 0.4708 - categorical_accuracy: 0.8483
10560/60000 [====>.........................] - ETA: 1:30 - loss: 0.4695 - categorical_accuracy: 0.8488
10592/60000 [====>.........................] - ETA: 1:29 - loss: 0.4682 - categorical_accuracy: 0.8492
10624/60000 [====>.........................] - ETA: 1:29 - loss: 0.4671 - categorical_accuracy: 0.8495
10656/60000 [====>.........................] - ETA: 1:29 - loss: 0.4660 - categorical_accuracy: 0.8498
10688/60000 [====>.........................] - ETA: 1:29 - loss: 0.4655 - categorical_accuracy: 0.8499
10720/60000 [====>.........................] - ETA: 1:29 - loss: 0.4657 - categorical_accuracy: 0.8500
10752/60000 [====>.........................] - ETA: 1:29 - loss: 0.4646 - categorical_accuracy: 0.8504
10784/60000 [====>.........................] - ETA: 1:29 - loss: 0.4647 - categorical_accuracy: 0.8505
10816/60000 [====>.........................] - ETA: 1:29 - loss: 0.4637 - categorical_accuracy: 0.8509
10848/60000 [====>.........................] - ETA: 1:29 - loss: 0.4628 - categorical_accuracy: 0.8511
10880/60000 [====>.........................] - ETA: 1:29 - loss: 0.4616 - categorical_accuracy: 0.8516
10912/60000 [====>.........................] - ETA: 1:29 - loss: 0.4606 - categorical_accuracy: 0.8518
10944/60000 [====>.........................] - ETA: 1:29 - loss: 0.4603 - categorical_accuracy: 0.8519
10976/60000 [====>.........................] - ETA: 1:29 - loss: 0.4605 - categorical_accuracy: 0.8519
11008/60000 [====>.........................] - ETA: 1:29 - loss: 0.4597 - categorical_accuracy: 0.8521
11040/60000 [====>.........................] - ETA: 1:29 - loss: 0.4590 - categorical_accuracy: 0.8523
11072/60000 [====>.........................] - ETA: 1:29 - loss: 0.4581 - categorical_accuracy: 0.8526
11104/60000 [====>.........................] - ETA: 1:29 - loss: 0.4578 - categorical_accuracy: 0.8527
11136/60000 [====>.........................] - ETA: 1:28 - loss: 0.4569 - categorical_accuracy: 0.8530
11168/60000 [====>.........................] - ETA: 1:28 - loss: 0.4568 - categorical_accuracy: 0.8531
11200/60000 [====>.........................] - ETA: 1:28 - loss: 0.4564 - categorical_accuracy: 0.8531
11232/60000 [====>.........................] - ETA: 1:28 - loss: 0.4554 - categorical_accuracy: 0.8535
11264/60000 [====>.........................] - ETA: 1:28 - loss: 0.4560 - categorical_accuracy: 0.8534
11296/60000 [====>.........................] - ETA: 1:28 - loss: 0.4549 - categorical_accuracy: 0.8538
11328/60000 [====>.........................] - ETA: 1:28 - loss: 0.4540 - categorical_accuracy: 0.8541
11360/60000 [====>.........................] - ETA: 1:28 - loss: 0.4534 - categorical_accuracy: 0.8543
11392/60000 [====>.........................] - ETA: 1:28 - loss: 0.4525 - categorical_accuracy: 0.8545
11424/60000 [====>.........................] - ETA: 1:28 - loss: 0.4519 - categorical_accuracy: 0.8547
11456/60000 [====>.........................] - ETA: 1:28 - loss: 0.4509 - categorical_accuracy: 0.8551
11488/60000 [====>.........................] - ETA: 1:28 - loss: 0.4501 - categorical_accuracy: 0.8553
11520/60000 [====>.........................] - ETA: 1:28 - loss: 0.4496 - categorical_accuracy: 0.8555
11552/60000 [====>.........................] - ETA: 1:28 - loss: 0.4492 - categorical_accuracy: 0.8555
11584/60000 [====>.........................] - ETA: 1:28 - loss: 0.4483 - categorical_accuracy: 0.8558
11616/60000 [====>.........................] - ETA: 1:28 - loss: 0.4483 - categorical_accuracy: 0.8558
11648/60000 [====>.........................] - ETA: 1:27 - loss: 0.4474 - categorical_accuracy: 0.8561
11680/60000 [====>.........................] - ETA: 1:27 - loss: 0.4464 - categorical_accuracy: 0.8564
11712/60000 [====>.........................] - ETA: 1:27 - loss: 0.4455 - categorical_accuracy: 0.8567
11744/60000 [====>.........................] - ETA: 1:27 - loss: 0.4447 - categorical_accuracy: 0.8570
11776/60000 [====>.........................] - ETA: 1:27 - loss: 0.4453 - categorical_accuracy: 0.8573
11808/60000 [====>.........................] - ETA: 1:27 - loss: 0.4445 - categorical_accuracy: 0.8575
11840/60000 [====>.........................] - ETA: 1:27 - loss: 0.4447 - categorical_accuracy: 0.8576
11872/60000 [====>.........................] - ETA: 1:27 - loss: 0.4438 - categorical_accuracy: 0.8578
11904/60000 [====>.........................] - ETA: 1:27 - loss: 0.4438 - categorical_accuracy: 0.8579
11936/60000 [====>.........................] - ETA: 1:27 - loss: 0.4435 - categorical_accuracy: 0.8579
11968/60000 [====>.........................] - ETA: 1:27 - loss: 0.4429 - categorical_accuracy: 0.8581
12000/60000 [=====>........................] - ETA: 1:27 - loss: 0.4420 - categorical_accuracy: 0.8584
12032/60000 [=====>........................] - ETA: 1:27 - loss: 0.4414 - categorical_accuracy: 0.8586
12064/60000 [=====>........................] - ETA: 1:27 - loss: 0.4412 - categorical_accuracy: 0.8587
12096/60000 [=====>........................] - ETA: 1:27 - loss: 0.4409 - categorical_accuracy: 0.8588
12128/60000 [=====>........................] - ETA: 1:27 - loss: 0.4405 - categorical_accuracy: 0.8591
12160/60000 [=====>........................] - ETA: 1:27 - loss: 0.4397 - categorical_accuracy: 0.8594
12192/60000 [=====>........................] - ETA: 1:26 - loss: 0.4387 - categorical_accuracy: 0.8597
12224/60000 [=====>........................] - ETA: 1:26 - loss: 0.4378 - categorical_accuracy: 0.8601
12256/60000 [=====>........................] - ETA: 1:26 - loss: 0.4369 - categorical_accuracy: 0.8603
12288/60000 [=====>........................] - ETA: 1:26 - loss: 0.4363 - categorical_accuracy: 0.8605
12320/60000 [=====>........................] - ETA: 1:26 - loss: 0.4353 - categorical_accuracy: 0.8609
12352/60000 [=====>........................] - ETA: 1:26 - loss: 0.4347 - categorical_accuracy: 0.8611
12384/60000 [=====>........................] - ETA: 1:26 - loss: 0.4337 - categorical_accuracy: 0.8614
12416/60000 [=====>........................] - ETA: 1:26 - loss: 0.4333 - categorical_accuracy: 0.8615
12448/60000 [=====>........................] - ETA: 1:26 - loss: 0.4327 - categorical_accuracy: 0.8617
12480/60000 [=====>........................] - ETA: 1:26 - loss: 0.4321 - categorical_accuracy: 0.8619
12512/60000 [=====>........................] - ETA: 1:26 - loss: 0.4318 - categorical_accuracy: 0.8618
12544/60000 [=====>........................] - ETA: 1:26 - loss: 0.4315 - categorical_accuracy: 0.8619
12576/60000 [=====>........................] - ETA: 1:26 - loss: 0.4309 - categorical_accuracy: 0.8622
12608/60000 [=====>........................] - ETA: 1:26 - loss: 0.4302 - categorical_accuracy: 0.8625
12640/60000 [=====>........................] - ETA: 1:26 - loss: 0.4299 - categorical_accuracy: 0.8626
12672/60000 [=====>........................] - ETA: 1:26 - loss: 0.4289 - categorical_accuracy: 0.8629
12704/60000 [=====>........................] - ETA: 1:26 - loss: 0.4285 - categorical_accuracy: 0.8631
12736/60000 [=====>........................] - ETA: 1:25 - loss: 0.4279 - categorical_accuracy: 0.8634
12768/60000 [=====>........................] - ETA: 1:25 - loss: 0.4274 - categorical_accuracy: 0.8635
12800/60000 [=====>........................] - ETA: 1:25 - loss: 0.4264 - categorical_accuracy: 0.8638
12832/60000 [=====>........................] - ETA: 1:25 - loss: 0.4255 - categorical_accuracy: 0.8642
12864/60000 [=====>........................] - ETA: 1:25 - loss: 0.4246 - categorical_accuracy: 0.8644
12896/60000 [=====>........................] - ETA: 1:25 - loss: 0.4237 - categorical_accuracy: 0.8648
12928/60000 [=====>........................] - ETA: 1:25 - loss: 0.4234 - categorical_accuracy: 0.8649
12960/60000 [=====>........................] - ETA: 1:25 - loss: 0.4228 - categorical_accuracy: 0.8650
12992/60000 [=====>........................] - ETA: 1:25 - loss: 0.4223 - categorical_accuracy: 0.8651
13024/60000 [=====>........................] - ETA: 1:25 - loss: 0.4217 - categorical_accuracy: 0.8653
13056/60000 [=====>........................] - ETA: 1:25 - loss: 0.4210 - categorical_accuracy: 0.8655
13088/60000 [=====>........................] - ETA: 1:25 - loss: 0.4202 - categorical_accuracy: 0.8657
13120/60000 [=====>........................] - ETA: 1:25 - loss: 0.4195 - categorical_accuracy: 0.8659
13152/60000 [=====>........................] - ETA: 1:25 - loss: 0.4193 - categorical_accuracy: 0.8660
13184/60000 [=====>........................] - ETA: 1:25 - loss: 0.4186 - categorical_accuracy: 0.8662
13216/60000 [=====>........................] - ETA: 1:25 - loss: 0.4179 - categorical_accuracy: 0.8664
13248/60000 [=====>........................] - ETA: 1:25 - loss: 0.4183 - categorical_accuracy: 0.8662
13280/60000 [=====>........................] - ETA: 1:24 - loss: 0.4176 - categorical_accuracy: 0.8665
13312/60000 [=====>........................] - ETA: 1:24 - loss: 0.4173 - categorical_accuracy: 0.8667
13344/60000 [=====>........................] - ETA: 1:24 - loss: 0.4165 - categorical_accuracy: 0.8670
13376/60000 [=====>........................] - ETA: 1:24 - loss: 0.4158 - categorical_accuracy: 0.8673
13408/60000 [=====>........................] - ETA: 1:24 - loss: 0.4149 - categorical_accuracy: 0.8676
13440/60000 [=====>........................] - ETA: 1:24 - loss: 0.4141 - categorical_accuracy: 0.8679
13472/60000 [=====>........................] - ETA: 1:24 - loss: 0.4132 - categorical_accuracy: 0.8682
13504/60000 [=====>........................] - ETA: 1:24 - loss: 0.4124 - categorical_accuracy: 0.8684
13536/60000 [=====>........................] - ETA: 1:24 - loss: 0.4119 - categorical_accuracy: 0.8684
13568/60000 [=====>........................] - ETA: 1:24 - loss: 0.4114 - categorical_accuracy: 0.8686
13600/60000 [=====>........................] - ETA: 1:24 - loss: 0.4112 - categorical_accuracy: 0.8685
13632/60000 [=====>........................] - ETA: 1:24 - loss: 0.4105 - categorical_accuracy: 0.8688
13664/60000 [=====>........................] - ETA: 1:24 - loss: 0.4101 - categorical_accuracy: 0.8689
13696/60000 [=====>........................] - ETA: 1:24 - loss: 0.4100 - categorical_accuracy: 0.8690
13728/60000 [=====>........................] - ETA: 1:24 - loss: 0.4098 - categorical_accuracy: 0.8692
13760/60000 [=====>........................] - ETA: 1:24 - loss: 0.4095 - categorical_accuracy: 0.8692
13792/60000 [=====>........................] - ETA: 1:24 - loss: 0.4086 - categorical_accuracy: 0.8695
13824/60000 [=====>........................] - ETA: 1:23 - loss: 0.4081 - categorical_accuracy: 0.8696
13856/60000 [=====>........................] - ETA: 1:23 - loss: 0.4076 - categorical_accuracy: 0.8699
13888/60000 [=====>........................] - ETA: 1:23 - loss: 0.4070 - categorical_accuracy: 0.8700
13920/60000 [=====>........................] - ETA: 1:23 - loss: 0.4065 - categorical_accuracy: 0.8700
13952/60000 [=====>........................] - ETA: 1:23 - loss: 0.4057 - categorical_accuracy: 0.8703
13984/60000 [=====>........................] - ETA: 1:23 - loss: 0.4049 - categorical_accuracy: 0.8706
14016/60000 [======>.......................] - ETA: 1:23 - loss: 0.4041 - categorical_accuracy: 0.8709
14048/60000 [======>.......................] - ETA: 1:23 - loss: 0.4033 - categorical_accuracy: 0.8711
14080/60000 [======>.......................] - ETA: 1:23 - loss: 0.4041 - categorical_accuracy: 0.8711
14112/60000 [======>.......................] - ETA: 1:23 - loss: 0.4040 - categorical_accuracy: 0.8711
14144/60000 [======>.......................] - ETA: 1:23 - loss: 0.4041 - categorical_accuracy: 0.8712
14176/60000 [======>.......................] - ETA: 1:23 - loss: 0.4036 - categorical_accuracy: 0.8713
14208/60000 [======>.......................] - ETA: 1:23 - loss: 0.4028 - categorical_accuracy: 0.8716
14240/60000 [======>.......................] - ETA: 1:23 - loss: 0.4021 - categorical_accuracy: 0.8718
14272/60000 [======>.......................] - ETA: 1:23 - loss: 0.4018 - categorical_accuracy: 0.8718
14304/60000 [======>.......................] - ETA: 1:23 - loss: 0.4016 - categorical_accuracy: 0.8719
14336/60000 [======>.......................] - ETA: 1:22 - loss: 0.4010 - categorical_accuracy: 0.8721
14368/60000 [======>.......................] - ETA: 1:22 - loss: 0.4002 - categorical_accuracy: 0.8724
14400/60000 [======>.......................] - ETA: 1:22 - loss: 0.3995 - categorical_accuracy: 0.8726
14432/60000 [======>.......................] - ETA: 1:22 - loss: 0.3991 - categorical_accuracy: 0.8728
14464/60000 [======>.......................] - ETA: 1:22 - loss: 0.3984 - categorical_accuracy: 0.8731
14496/60000 [======>.......................] - ETA: 1:22 - loss: 0.3977 - categorical_accuracy: 0.8733
14528/60000 [======>.......................] - ETA: 1:22 - loss: 0.3969 - categorical_accuracy: 0.8736
14560/60000 [======>.......................] - ETA: 1:22 - loss: 0.3973 - categorical_accuracy: 0.8734
14592/60000 [======>.......................] - ETA: 1:22 - loss: 0.3971 - categorical_accuracy: 0.8735
14624/60000 [======>.......................] - ETA: 1:22 - loss: 0.3963 - categorical_accuracy: 0.8738
14656/60000 [======>.......................] - ETA: 1:22 - loss: 0.3958 - categorical_accuracy: 0.8739
14688/60000 [======>.......................] - ETA: 1:22 - loss: 0.3951 - categorical_accuracy: 0.8742
14720/60000 [======>.......................] - ETA: 1:22 - loss: 0.3948 - categorical_accuracy: 0.8743
14752/60000 [======>.......................] - ETA: 1:22 - loss: 0.3942 - categorical_accuracy: 0.8745
14784/60000 [======>.......................] - ETA: 1:22 - loss: 0.3934 - categorical_accuracy: 0.8747
14816/60000 [======>.......................] - ETA: 1:22 - loss: 0.3929 - categorical_accuracy: 0.8749
14848/60000 [======>.......................] - ETA: 1:22 - loss: 0.3922 - categorical_accuracy: 0.8751
14880/60000 [======>.......................] - ETA: 1:22 - loss: 0.3921 - categorical_accuracy: 0.8752
14912/60000 [======>.......................] - ETA: 1:22 - loss: 0.3915 - categorical_accuracy: 0.8753
14944/60000 [======>.......................] - ETA: 1:21 - loss: 0.3912 - categorical_accuracy: 0.8755
14976/60000 [======>.......................] - ETA: 1:21 - loss: 0.3907 - categorical_accuracy: 0.8757
15008/60000 [======>.......................] - ETA: 1:21 - loss: 0.3902 - categorical_accuracy: 0.8759
15040/60000 [======>.......................] - ETA: 1:21 - loss: 0.3897 - categorical_accuracy: 0.8761
15072/60000 [======>.......................] - ETA: 1:21 - loss: 0.3891 - categorical_accuracy: 0.8763
15104/60000 [======>.......................] - ETA: 1:21 - loss: 0.3885 - categorical_accuracy: 0.8764
15136/60000 [======>.......................] - ETA: 1:21 - loss: 0.3889 - categorical_accuracy: 0.8763
15168/60000 [======>.......................] - ETA: 1:21 - loss: 0.3882 - categorical_accuracy: 0.8766
15200/60000 [======>.......................] - ETA: 1:21 - loss: 0.3881 - categorical_accuracy: 0.8766
15232/60000 [======>.......................] - ETA: 1:21 - loss: 0.3876 - categorical_accuracy: 0.8767
15264/60000 [======>.......................] - ETA: 1:21 - loss: 0.3870 - categorical_accuracy: 0.8769
15296/60000 [======>.......................] - ETA: 1:21 - loss: 0.3868 - categorical_accuracy: 0.8770
15328/60000 [======>.......................] - ETA: 1:21 - loss: 0.3864 - categorical_accuracy: 0.8770
15360/60000 [======>.......................] - ETA: 1:21 - loss: 0.3857 - categorical_accuracy: 0.8773
15392/60000 [======>.......................] - ETA: 1:21 - loss: 0.3852 - categorical_accuracy: 0.8774
15424/60000 [======>.......................] - ETA: 1:21 - loss: 0.3847 - categorical_accuracy: 0.8776
15456/60000 [======>.......................] - ETA: 1:21 - loss: 0.3840 - categorical_accuracy: 0.8778
15488/60000 [======>.......................] - ETA: 1:21 - loss: 0.3838 - categorical_accuracy: 0.8778
15520/60000 [======>.......................] - ETA: 1:20 - loss: 0.3832 - categorical_accuracy: 0.8780
15552/60000 [======>.......................] - ETA: 1:20 - loss: 0.3826 - categorical_accuracy: 0.8782
15584/60000 [======>.......................] - ETA: 1:20 - loss: 0.3819 - categorical_accuracy: 0.8784
15616/60000 [======>.......................] - ETA: 1:20 - loss: 0.3813 - categorical_accuracy: 0.8786
15648/60000 [======>.......................] - ETA: 1:20 - loss: 0.3807 - categorical_accuracy: 0.8788
15680/60000 [======>.......................] - ETA: 1:20 - loss: 0.3801 - categorical_accuracy: 0.8790
15712/60000 [======>.......................] - ETA: 1:20 - loss: 0.3794 - categorical_accuracy: 0.8792
15744/60000 [======>.......................] - ETA: 1:20 - loss: 0.3786 - categorical_accuracy: 0.8794
15776/60000 [======>.......................] - ETA: 1:20 - loss: 0.3781 - categorical_accuracy: 0.8795
15808/60000 [======>.......................] - ETA: 1:20 - loss: 0.3780 - categorical_accuracy: 0.8796
15840/60000 [======>.......................] - ETA: 1:20 - loss: 0.3780 - categorical_accuracy: 0.8797
15872/60000 [======>.......................] - ETA: 1:20 - loss: 0.3774 - categorical_accuracy: 0.8799
15904/60000 [======>.......................] - ETA: 1:20 - loss: 0.3769 - categorical_accuracy: 0.8801
15936/60000 [======>.......................] - ETA: 1:20 - loss: 0.3765 - categorical_accuracy: 0.8803
15968/60000 [======>.......................] - ETA: 1:20 - loss: 0.3763 - categorical_accuracy: 0.8804
16000/60000 [=======>......................] - ETA: 1:20 - loss: 0.3759 - categorical_accuracy: 0.8806
16032/60000 [=======>......................] - ETA: 1:20 - loss: 0.3752 - categorical_accuracy: 0.8808
16064/60000 [=======>......................] - ETA: 1:20 - loss: 0.3747 - categorical_accuracy: 0.8810
16096/60000 [=======>......................] - ETA: 1:19 - loss: 0.3740 - categorical_accuracy: 0.8812
16128/60000 [=======>......................] - ETA: 1:19 - loss: 0.3733 - categorical_accuracy: 0.8814
16160/60000 [=======>......................] - ETA: 1:19 - loss: 0.3731 - categorical_accuracy: 0.8816
16192/60000 [=======>......................] - ETA: 1:19 - loss: 0.3727 - categorical_accuracy: 0.8817
16224/60000 [=======>......................] - ETA: 1:19 - loss: 0.3730 - categorical_accuracy: 0.8817
16256/60000 [=======>......................] - ETA: 1:19 - loss: 0.3724 - categorical_accuracy: 0.8819
16288/60000 [=======>......................] - ETA: 1:19 - loss: 0.3721 - categorical_accuracy: 0.8820
16320/60000 [=======>......................] - ETA: 1:19 - loss: 0.3716 - categorical_accuracy: 0.8822
16352/60000 [=======>......................] - ETA: 1:19 - loss: 0.3711 - categorical_accuracy: 0.8823
16384/60000 [=======>......................] - ETA: 1:19 - loss: 0.3709 - categorical_accuracy: 0.8823
16416/60000 [=======>......................] - ETA: 1:19 - loss: 0.3704 - categorical_accuracy: 0.8824
16448/60000 [=======>......................] - ETA: 1:19 - loss: 0.3698 - categorical_accuracy: 0.8826
16480/60000 [=======>......................] - ETA: 1:19 - loss: 0.3695 - categorical_accuracy: 0.8826
16512/60000 [=======>......................] - ETA: 1:19 - loss: 0.3691 - categorical_accuracy: 0.8828
16544/60000 [=======>......................] - ETA: 1:19 - loss: 0.3686 - categorical_accuracy: 0.8829
16576/60000 [=======>......................] - ETA: 1:19 - loss: 0.3681 - categorical_accuracy: 0.8831
16608/60000 [=======>......................] - ETA: 1:18 - loss: 0.3676 - categorical_accuracy: 0.8832
16640/60000 [=======>......................] - ETA: 1:18 - loss: 0.3672 - categorical_accuracy: 0.8834
16672/60000 [=======>......................] - ETA: 1:18 - loss: 0.3668 - categorical_accuracy: 0.8836
16704/60000 [=======>......................] - ETA: 1:18 - loss: 0.3663 - categorical_accuracy: 0.8837
16736/60000 [=======>......................] - ETA: 1:18 - loss: 0.3662 - categorical_accuracy: 0.8838
16768/60000 [=======>......................] - ETA: 1:18 - loss: 0.3663 - categorical_accuracy: 0.8839
16800/60000 [=======>......................] - ETA: 1:18 - loss: 0.3658 - categorical_accuracy: 0.8840
16832/60000 [=======>......................] - ETA: 1:18 - loss: 0.3652 - categorical_accuracy: 0.8842
16864/60000 [=======>......................] - ETA: 1:18 - loss: 0.3647 - categorical_accuracy: 0.8844
16896/60000 [=======>......................] - ETA: 1:18 - loss: 0.3642 - categorical_accuracy: 0.8845
16928/60000 [=======>......................] - ETA: 1:18 - loss: 0.3638 - categorical_accuracy: 0.8846
16960/60000 [=======>......................] - ETA: 1:18 - loss: 0.3634 - categorical_accuracy: 0.8847
16992/60000 [=======>......................] - ETA: 1:18 - loss: 0.3633 - categorical_accuracy: 0.8847
17024/60000 [=======>......................] - ETA: 1:18 - loss: 0.3633 - categorical_accuracy: 0.8847
17056/60000 [=======>......................] - ETA: 1:18 - loss: 0.3628 - categorical_accuracy: 0.8848
17088/60000 [=======>......................] - ETA: 1:18 - loss: 0.3624 - categorical_accuracy: 0.8850
17120/60000 [=======>......................] - ETA: 1:18 - loss: 0.3619 - categorical_accuracy: 0.8852
17152/60000 [=======>......................] - ETA: 1:17 - loss: 0.3616 - categorical_accuracy: 0.8853
17184/60000 [=======>......................] - ETA: 1:17 - loss: 0.3616 - categorical_accuracy: 0.8855
17216/60000 [=======>......................] - ETA: 1:17 - loss: 0.3613 - categorical_accuracy: 0.8855
17248/60000 [=======>......................] - ETA: 1:17 - loss: 0.3616 - categorical_accuracy: 0.8856
17280/60000 [=======>......................] - ETA: 1:17 - loss: 0.3614 - categorical_accuracy: 0.8856
17312/60000 [=======>......................] - ETA: 1:17 - loss: 0.3611 - categorical_accuracy: 0.8857
17344/60000 [=======>......................] - ETA: 1:17 - loss: 0.3606 - categorical_accuracy: 0.8860
17376/60000 [=======>......................] - ETA: 1:17 - loss: 0.3603 - categorical_accuracy: 0.8860
17408/60000 [=======>......................] - ETA: 1:17 - loss: 0.3597 - categorical_accuracy: 0.8863
17440/60000 [=======>......................] - ETA: 1:17 - loss: 0.3594 - categorical_accuracy: 0.8864
17472/60000 [=======>......................] - ETA: 1:17 - loss: 0.3592 - categorical_accuracy: 0.8864
17504/60000 [=======>......................] - ETA: 1:17 - loss: 0.3586 - categorical_accuracy: 0.8867
17536/60000 [=======>......................] - ETA: 1:17 - loss: 0.3582 - categorical_accuracy: 0.8868
17568/60000 [=======>......................] - ETA: 1:17 - loss: 0.3579 - categorical_accuracy: 0.8869
17600/60000 [=======>......................] - ETA: 1:17 - loss: 0.3578 - categorical_accuracy: 0.8869
17632/60000 [=======>......................] - ETA: 1:17 - loss: 0.3575 - categorical_accuracy: 0.8870
17664/60000 [=======>......................] - ETA: 1:16 - loss: 0.3571 - categorical_accuracy: 0.8871
17696/60000 [=======>......................] - ETA: 1:16 - loss: 0.3566 - categorical_accuracy: 0.8873
17728/60000 [=======>......................] - ETA: 1:16 - loss: 0.3562 - categorical_accuracy: 0.8874
17760/60000 [=======>......................] - ETA: 1:16 - loss: 0.3559 - categorical_accuracy: 0.8874
17792/60000 [=======>......................] - ETA: 1:16 - loss: 0.3553 - categorical_accuracy: 0.8876
17824/60000 [=======>......................] - ETA: 1:16 - loss: 0.3550 - categorical_accuracy: 0.8877
17856/60000 [=======>......................] - ETA: 1:16 - loss: 0.3549 - categorical_accuracy: 0.8878
17888/60000 [=======>......................] - ETA: 1:16 - loss: 0.3545 - categorical_accuracy: 0.8879
17920/60000 [=======>......................] - ETA: 1:16 - loss: 0.3543 - categorical_accuracy: 0.8879
17952/60000 [=======>......................] - ETA: 1:16 - loss: 0.3539 - categorical_accuracy: 0.8881
17984/60000 [=======>......................] - ETA: 1:16 - loss: 0.3537 - categorical_accuracy: 0.8882
18016/60000 [========>.....................] - ETA: 1:16 - loss: 0.3532 - categorical_accuracy: 0.8883
18048/60000 [========>.....................] - ETA: 1:16 - loss: 0.3527 - categorical_accuracy: 0.8885
18080/60000 [========>.....................] - ETA: 1:16 - loss: 0.3524 - categorical_accuracy: 0.8886
18112/60000 [========>.....................] - ETA: 1:16 - loss: 0.3519 - categorical_accuracy: 0.8888
18144/60000 [========>.....................] - ETA: 1:16 - loss: 0.3522 - categorical_accuracy: 0.8888
18176/60000 [========>.....................] - ETA: 1:15 - loss: 0.3521 - categorical_accuracy: 0.8889
18208/60000 [========>.....................] - ETA: 1:15 - loss: 0.3518 - categorical_accuracy: 0.8889
18240/60000 [========>.....................] - ETA: 1:15 - loss: 0.3514 - categorical_accuracy: 0.8890
18272/60000 [========>.....................] - ETA: 1:15 - loss: 0.3510 - categorical_accuracy: 0.8892
18304/60000 [========>.....................] - ETA: 1:15 - loss: 0.3504 - categorical_accuracy: 0.8894
18336/60000 [========>.....................] - ETA: 1:15 - loss: 0.3505 - categorical_accuracy: 0.8893
18368/60000 [========>.....................] - ETA: 1:15 - loss: 0.3501 - categorical_accuracy: 0.8894
18400/60000 [========>.....................] - ETA: 1:15 - loss: 0.3496 - categorical_accuracy: 0.8896
18432/60000 [========>.....................] - ETA: 1:15 - loss: 0.3497 - categorical_accuracy: 0.8896
18464/60000 [========>.....................] - ETA: 1:15 - loss: 0.3492 - categorical_accuracy: 0.8898
18496/60000 [========>.....................] - ETA: 1:15 - loss: 0.3488 - categorical_accuracy: 0.8899
18528/60000 [========>.....................] - ETA: 1:15 - loss: 0.3487 - categorical_accuracy: 0.8901
18560/60000 [========>.....................] - ETA: 1:15 - loss: 0.3482 - categorical_accuracy: 0.8902
18592/60000 [========>.....................] - ETA: 1:15 - loss: 0.3477 - categorical_accuracy: 0.8904
18624/60000 [========>.....................] - ETA: 1:15 - loss: 0.3472 - categorical_accuracy: 0.8906
18656/60000 [========>.....................] - ETA: 1:15 - loss: 0.3471 - categorical_accuracy: 0.8907
18688/60000 [========>.....................] - ETA: 1:15 - loss: 0.3467 - categorical_accuracy: 0.8907
18720/60000 [========>.....................] - ETA: 1:14 - loss: 0.3466 - categorical_accuracy: 0.8908
18752/60000 [========>.....................] - ETA: 1:14 - loss: 0.3464 - categorical_accuracy: 0.8908
18784/60000 [========>.....................] - ETA: 1:14 - loss: 0.3461 - categorical_accuracy: 0.8909
18816/60000 [========>.....................] - ETA: 1:14 - loss: 0.3459 - categorical_accuracy: 0.8910
18848/60000 [========>.....................] - ETA: 1:14 - loss: 0.3455 - categorical_accuracy: 0.8911
18880/60000 [========>.....................] - ETA: 1:14 - loss: 0.3451 - categorical_accuracy: 0.8912
18912/60000 [========>.....................] - ETA: 1:14 - loss: 0.3446 - categorical_accuracy: 0.8913
18944/60000 [========>.....................] - ETA: 1:14 - loss: 0.3448 - categorical_accuracy: 0.8915
18976/60000 [========>.....................] - ETA: 1:14 - loss: 0.3445 - categorical_accuracy: 0.8915
19008/60000 [========>.....................] - ETA: 1:14 - loss: 0.3440 - categorical_accuracy: 0.8917
19040/60000 [========>.....................] - ETA: 1:14 - loss: 0.3436 - categorical_accuracy: 0.8919
19072/60000 [========>.....................] - ETA: 1:14 - loss: 0.3431 - categorical_accuracy: 0.8920
19104/60000 [========>.....................] - ETA: 1:14 - loss: 0.3426 - categorical_accuracy: 0.8922
19136/60000 [========>.....................] - ETA: 1:14 - loss: 0.3423 - categorical_accuracy: 0.8922
19168/60000 [========>.....................] - ETA: 1:14 - loss: 0.3420 - categorical_accuracy: 0.8923
19200/60000 [========>.....................] - ETA: 1:14 - loss: 0.3415 - categorical_accuracy: 0.8925
19232/60000 [========>.....................] - ETA: 1:14 - loss: 0.3412 - categorical_accuracy: 0.8926
19264/60000 [========>.....................] - ETA: 1:13 - loss: 0.3407 - categorical_accuracy: 0.8928
19296/60000 [========>.....................] - ETA: 1:13 - loss: 0.3408 - categorical_accuracy: 0.8929
19328/60000 [========>.....................] - ETA: 1:13 - loss: 0.3406 - categorical_accuracy: 0.8930
19360/60000 [========>.....................] - ETA: 1:13 - loss: 0.3402 - categorical_accuracy: 0.8931
19392/60000 [========>.....................] - ETA: 1:13 - loss: 0.3398 - categorical_accuracy: 0.8933
19424/60000 [========>.....................] - ETA: 1:13 - loss: 0.3394 - categorical_accuracy: 0.8934
19456/60000 [========>.....................] - ETA: 1:13 - loss: 0.3389 - categorical_accuracy: 0.8936
19488/60000 [========>.....................] - ETA: 1:13 - loss: 0.3386 - categorical_accuracy: 0.8936
19520/60000 [========>.....................] - ETA: 1:13 - loss: 0.3387 - categorical_accuracy: 0.8935
19552/60000 [========>.....................] - ETA: 1:13 - loss: 0.3384 - categorical_accuracy: 0.8936
19584/60000 [========>.....................] - ETA: 1:13 - loss: 0.3381 - categorical_accuracy: 0.8936
19616/60000 [========>.....................] - ETA: 1:13 - loss: 0.3378 - categorical_accuracy: 0.8937
19648/60000 [========>.....................] - ETA: 1:13 - loss: 0.3375 - categorical_accuracy: 0.8938
19680/60000 [========>.....................] - ETA: 1:13 - loss: 0.3374 - categorical_accuracy: 0.8939
19712/60000 [========>.....................] - ETA: 1:13 - loss: 0.3370 - categorical_accuracy: 0.8939
19744/60000 [========>.....................] - ETA: 1:13 - loss: 0.3366 - categorical_accuracy: 0.8940
19776/60000 [========>.....................] - ETA: 1:13 - loss: 0.3363 - categorical_accuracy: 0.8941
19808/60000 [========>.....................] - ETA: 1:12 - loss: 0.3361 - categorical_accuracy: 0.8942
19840/60000 [========>.....................] - ETA: 1:12 - loss: 0.3357 - categorical_accuracy: 0.8943
19872/60000 [========>.....................] - ETA: 1:12 - loss: 0.3359 - categorical_accuracy: 0.8944
19904/60000 [========>.....................] - ETA: 1:12 - loss: 0.3359 - categorical_accuracy: 0.8943
19936/60000 [========>.....................] - ETA: 1:12 - loss: 0.3357 - categorical_accuracy: 0.8944
19968/60000 [========>.....................] - ETA: 1:12 - loss: 0.3353 - categorical_accuracy: 0.8945
20000/60000 [=========>....................] - ETA: 1:12 - loss: 0.3350 - categorical_accuracy: 0.8946
20032/60000 [=========>....................] - ETA: 1:12 - loss: 0.3346 - categorical_accuracy: 0.8947
20064/60000 [=========>....................] - ETA: 1:12 - loss: 0.3344 - categorical_accuracy: 0.8948
20096/60000 [=========>....................] - ETA: 1:12 - loss: 0.3340 - categorical_accuracy: 0.8950
20128/60000 [=========>....................] - ETA: 1:12 - loss: 0.3338 - categorical_accuracy: 0.8950
20160/60000 [=========>....................] - ETA: 1:12 - loss: 0.3335 - categorical_accuracy: 0.8951
20192/60000 [=========>....................] - ETA: 1:12 - loss: 0.3331 - categorical_accuracy: 0.8953
20224/60000 [=========>....................] - ETA: 1:12 - loss: 0.3326 - categorical_accuracy: 0.8954
20256/60000 [=========>....................] - ETA: 1:12 - loss: 0.3327 - categorical_accuracy: 0.8954
20288/60000 [=========>....................] - ETA: 1:12 - loss: 0.3322 - categorical_accuracy: 0.8956
20320/60000 [=========>....................] - ETA: 1:12 - loss: 0.3322 - categorical_accuracy: 0.8957
20352/60000 [=========>....................] - ETA: 1:11 - loss: 0.3322 - categorical_accuracy: 0.8957
20384/60000 [=========>....................] - ETA: 1:11 - loss: 0.3322 - categorical_accuracy: 0.8958
20416/60000 [=========>....................] - ETA: 1:11 - loss: 0.3320 - categorical_accuracy: 0.8959
20448/60000 [=========>....................] - ETA: 1:11 - loss: 0.3319 - categorical_accuracy: 0.8960
20480/60000 [=========>....................] - ETA: 1:11 - loss: 0.3318 - categorical_accuracy: 0.8960
20512/60000 [=========>....................] - ETA: 1:11 - loss: 0.3315 - categorical_accuracy: 0.8961
20544/60000 [=========>....................] - ETA: 1:11 - loss: 0.3316 - categorical_accuracy: 0.8961
20576/60000 [=========>....................] - ETA: 1:11 - loss: 0.3313 - categorical_accuracy: 0.8961
20608/60000 [=========>....................] - ETA: 1:11 - loss: 0.3314 - categorical_accuracy: 0.8961
20640/60000 [=========>....................] - ETA: 1:11 - loss: 0.3312 - categorical_accuracy: 0.8962
20672/60000 [=========>....................] - ETA: 1:11 - loss: 0.3310 - categorical_accuracy: 0.8962
20704/60000 [=========>....................] - ETA: 1:11 - loss: 0.3307 - categorical_accuracy: 0.8963
20736/60000 [=========>....................] - ETA: 1:11 - loss: 0.3305 - categorical_accuracy: 0.8965
20768/60000 [=========>....................] - ETA: 1:11 - loss: 0.3304 - categorical_accuracy: 0.8966
20800/60000 [=========>....................] - ETA: 1:11 - loss: 0.3301 - categorical_accuracy: 0.8967
20832/60000 [=========>....................] - ETA: 1:11 - loss: 0.3298 - categorical_accuracy: 0.8968
20864/60000 [=========>....................] - ETA: 1:11 - loss: 0.3293 - categorical_accuracy: 0.8970
20896/60000 [=========>....................] - ETA: 1:10 - loss: 0.3289 - categorical_accuracy: 0.8971
20928/60000 [=========>....................] - ETA: 1:10 - loss: 0.3285 - categorical_accuracy: 0.8972
20960/60000 [=========>....................] - ETA: 1:10 - loss: 0.3283 - categorical_accuracy: 0.8972
20992/60000 [=========>....................] - ETA: 1:10 - loss: 0.3280 - categorical_accuracy: 0.8973
21024/60000 [=========>....................] - ETA: 1:10 - loss: 0.3275 - categorical_accuracy: 0.8975
21056/60000 [=========>....................] - ETA: 1:10 - loss: 0.3278 - categorical_accuracy: 0.8974
21088/60000 [=========>....................] - ETA: 1:10 - loss: 0.3274 - categorical_accuracy: 0.8975
21120/60000 [=========>....................] - ETA: 1:10 - loss: 0.3273 - categorical_accuracy: 0.8975
21152/60000 [=========>....................] - ETA: 1:10 - loss: 0.3268 - categorical_accuracy: 0.8977
21184/60000 [=========>....................] - ETA: 1:10 - loss: 0.3263 - categorical_accuracy: 0.8978
21216/60000 [=========>....................] - ETA: 1:10 - loss: 0.3259 - categorical_accuracy: 0.8980
21248/60000 [=========>....................] - ETA: 1:10 - loss: 0.3256 - categorical_accuracy: 0.8981
21280/60000 [=========>....................] - ETA: 1:10 - loss: 0.3252 - categorical_accuracy: 0.8982
21312/60000 [=========>....................] - ETA: 1:10 - loss: 0.3248 - categorical_accuracy: 0.8984
21344/60000 [=========>....................] - ETA: 1:10 - loss: 0.3247 - categorical_accuracy: 0.8984
21376/60000 [=========>....................] - ETA: 1:10 - loss: 0.3243 - categorical_accuracy: 0.8985
21408/60000 [=========>....................] - ETA: 1:10 - loss: 0.3239 - categorical_accuracy: 0.8986
21440/60000 [=========>....................] - ETA: 1:10 - loss: 0.3236 - categorical_accuracy: 0.8987
21472/60000 [=========>....................] - ETA: 1:09 - loss: 0.3232 - categorical_accuracy: 0.8988
21504/60000 [=========>....................] - ETA: 1:09 - loss: 0.3228 - categorical_accuracy: 0.8989
21536/60000 [=========>....................] - ETA: 1:09 - loss: 0.3225 - categorical_accuracy: 0.8991
21568/60000 [=========>....................] - ETA: 1:09 - loss: 0.3222 - categorical_accuracy: 0.8992
21600/60000 [=========>....................] - ETA: 1:09 - loss: 0.3217 - categorical_accuracy: 0.8994
21632/60000 [=========>....................] - ETA: 1:09 - loss: 0.3215 - categorical_accuracy: 0.8995
21664/60000 [=========>....................] - ETA: 1:09 - loss: 0.3213 - categorical_accuracy: 0.8995
21696/60000 [=========>....................] - ETA: 1:09 - loss: 0.3210 - categorical_accuracy: 0.8996
21728/60000 [=========>....................] - ETA: 1:09 - loss: 0.3209 - categorical_accuracy: 0.8997
21760/60000 [=========>....................] - ETA: 1:09 - loss: 0.3206 - categorical_accuracy: 0.8998
21792/60000 [=========>....................] - ETA: 1:09 - loss: 0.3203 - categorical_accuracy: 0.8999
21824/60000 [=========>....................] - ETA: 1:09 - loss: 0.3200 - categorical_accuracy: 0.9000
21856/60000 [=========>....................] - ETA: 1:09 - loss: 0.3197 - categorical_accuracy: 0.9001
21888/60000 [=========>....................] - ETA: 1:09 - loss: 0.3194 - categorical_accuracy: 0.9002
21920/60000 [=========>....................] - ETA: 1:09 - loss: 0.3191 - categorical_accuracy: 0.9003
21952/60000 [=========>....................] - ETA: 1:09 - loss: 0.3190 - categorical_accuracy: 0.9004
21984/60000 [=========>....................] - ETA: 1:09 - loss: 0.3186 - categorical_accuracy: 0.9005
22016/60000 [==========>...................] - ETA: 1:08 - loss: 0.3185 - categorical_accuracy: 0.9005
22048/60000 [==========>...................] - ETA: 1:08 - loss: 0.3180 - categorical_accuracy: 0.9007
22080/60000 [==========>...................] - ETA: 1:08 - loss: 0.3177 - categorical_accuracy: 0.9008
22112/60000 [==========>...................] - ETA: 1:08 - loss: 0.3177 - categorical_accuracy: 0.9008
22144/60000 [==========>...................] - ETA: 1:08 - loss: 0.3177 - categorical_accuracy: 0.9009
22176/60000 [==========>...................] - ETA: 1:08 - loss: 0.3174 - categorical_accuracy: 0.9009
22208/60000 [==========>...................] - ETA: 1:08 - loss: 0.3170 - categorical_accuracy: 0.9010
22240/60000 [==========>...................] - ETA: 1:08 - loss: 0.3167 - categorical_accuracy: 0.9012
22272/60000 [==========>...................] - ETA: 1:08 - loss: 0.3166 - categorical_accuracy: 0.9012
22304/60000 [==========>...................] - ETA: 1:08 - loss: 0.3164 - categorical_accuracy: 0.9013
22336/60000 [==========>...................] - ETA: 1:08 - loss: 0.3160 - categorical_accuracy: 0.9015
22368/60000 [==========>...................] - ETA: 1:08 - loss: 0.3158 - categorical_accuracy: 0.9015
22400/60000 [==========>...................] - ETA: 1:08 - loss: 0.3157 - categorical_accuracy: 0.9016
22432/60000 [==========>...................] - ETA: 1:08 - loss: 0.3156 - categorical_accuracy: 0.9016
22464/60000 [==========>...................] - ETA: 1:08 - loss: 0.3153 - categorical_accuracy: 0.9017
22496/60000 [==========>...................] - ETA: 1:08 - loss: 0.3156 - categorical_accuracy: 0.9017
22528/60000 [==========>...................] - ETA: 1:08 - loss: 0.3153 - categorical_accuracy: 0.9018
22560/60000 [==========>...................] - ETA: 1:08 - loss: 0.3150 - categorical_accuracy: 0.9019
22592/60000 [==========>...................] - ETA: 1:07 - loss: 0.3146 - categorical_accuracy: 0.9020
22624/60000 [==========>...................] - ETA: 1:07 - loss: 0.3144 - categorical_accuracy: 0.9021
22656/60000 [==========>...................] - ETA: 1:07 - loss: 0.3141 - categorical_accuracy: 0.9022
22688/60000 [==========>...................] - ETA: 1:07 - loss: 0.3139 - categorical_accuracy: 0.9023
22720/60000 [==========>...................] - ETA: 1:07 - loss: 0.3137 - categorical_accuracy: 0.9023
22752/60000 [==========>...................] - ETA: 1:07 - loss: 0.3136 - categorical_accuracy: 0.9024
22784/60000 [==========>...................] - ETA: 1:07 - loss: 0.3139 - categorical_accuracy: 0.9023
22816/60000 [==========>...................] - ETA: 1:07 - loss: 0.3139 - categorical_accuracy: 0.9024
22848/60000 [==========>...................] - ETA: 1:07 - loss: 0.3137 - categorical_accuracy: 0.9025
22880/60000 [==========>...................] - ETA: 1:07 - loss: 0.3137 - categorical_accuracy: 0.9024
22912/60000 [==========>...................] - ETA: 1:07 - loss: 0.3136 - categorical_accuracy: 0.9025
22944/60000 [==========>...................] - ETA: 1:07 - loss: 0.3132 - categorical_accuracy: 0.9026
22976/60000 [==========>...................] - ETA: 1:07 - loss: 0.3128 - categorical_accuracy: 0.9027
23008/60000 [==========>...................] - ETA: 1:07 - loss: 0.3125 - categorical_accuracy: 0.9028
23040/60000 [==========>...................] - ETA: 1:07 - loss: 0.3121 - categorical_accuracy: 0.9030
23072/60000 [==========>...................] - ETA: 1:07 - loss: 0.3118 - categorical_accuracy: 0.9031
23104/60000 [==========>...................] - ETA: 1:07 - loss: 0.3115 - categorical_accuracy: 0.9032
23136/60000 [==========>...................] - ETA: 1:06 - loss: 0.3114 - categorical_accuracy: 0.9032
23168/60000 [==========>...................] - ETA: 1:06 - loss: 0.3111 - categorical_accuracy: 0.9033
23200/60000 [==========>...................] - ETA: 1:06 - loss: 0.3110 - categorical_accuracy: 0.9034
23232/60000 [==========>...................] - ETA: 1:06 - loss: 0.3108 - categorical_accuracy: 0.9035
23264/60000 [==========>...................] - ETA: 1:06 - loss: 0.3104 - categorical_accuracy: 0.9036
23296/60000 [==========>...................] - ETA: 1:06 - loss: 0.3101 - categorical_accuracy: 0.9036
23328/60000 [==========>...................] - ETA: 1:06 - loss: 0.3097 - categorical_accuracy: 0.9038
23360/60000 [==========>...................] - ETA: 1:06 - loss: 0.3096 - categorical_accuracy: 0.9038
23392/60000 [==========>...................] - ETA: 1:06 - loss: 0.3094 - categorical_accuracy: 0.9038
23424/60000 [==========>...................] - ETA: 1:06 - loss: 0.3090 - categorical_accuracy: 0.9039
23456/60000 [==========>...................] - ETA: 1:06 - loss: 0.3091 - categorical_accuracy: 0.9040
23488/60000 [==========>...................] - ETA: 1:06 - loss: 0.3087 - categorical_accuracy: 0.9041
23520/60000 [==========>...................] - ETA: 1:06 - loss: 0.3085 - categorical_accuracy: 0.9042
23552/60000 [==========>...................] - ETA: 1:06 - loss: 0.3082 - categorical_accuracy: 0.9043
23584/60000 [==========>...................] - ETA: 1:06 - loss: 0.3078 - categorical_accuracy: 0.9044
23616/60000 [==========>...................] - ETA: 1:06 - loss: 0.3077 - categorical_accuracy: 0.9044
23648/60000 [==========>...................] - ETA: 1:06 - loss: 0.3074 - categorical_accuracy: 0.9045
23680/60000 [==========>...................] - ETA: 1:05 - loss: 0.3072 - categorical_accuracy: 0.9046
23712/60000 [==========>...................] - ETA: 1:05 - loss: 0.3069 - categorical_accuracy: 0.9047
23744/60000 [==========>...................] - ETA: 1:05 - loss: 0.3067 - categorical_accuracy: 0.9048
23776/60000 [==========>...................] - ETA: 1:05 - loss: 0.3064 - categorical_accuracy: 0.9049
23808/60000 [==========>...................] - ETA: 1:05 - loss: 0.3060 - categorical_accuracy: 0.9050
23840/60000 [==========>...................] - ETA: 1:05 - loss: 0.3058 - categorical_accuracy: 0.9051
23872/60000 [==========>...................] - ETA: 1:05 - loss: 0.3055 - categorical_accuracy: 0.9052
23904/60000 [==========>...................] - ETA: 1:05 - loss: 0.3052 - categorical_accuracy: 0.9052
23936/60000 [==========>...................] - ETA: 1:05 - loss: 0.3051 - categorical_accuracy: 0.9053
23968/60000 [==========>...................] - ETA: 1:05 - loss: 0.3048 - categorical_accuracy: 0.9054
24000/60000 [===========>..................] - ETA: 1:05 - loss: 0.3046 - categorical_accuracy: 0.9054
24032/60000 [===========>..................] - ETA: 1:05 - loss: 0.3046 - categorical_accuracy: 0.9054
24064/60000 [===========>..................] - ETA: 1:05 - loss: 0.3043 - categorical_accuracy: 0.9055
24096/60000 [===========>..................] - ETA: 1:05 - loss: 0.3043 - categorical_accuracy: 0.9056
24128/60000 [===========>..................] - ETA: 1:05 - loss: 0.3040 - categorical_accuracy: 0.9057
24160/60000 [===========>..................] - ETA: 1:05 - loss: 0.3037 - categorical_accuracy: 0.9058
24192/60000 [===========>..................] - ETA: 1:05 - loss: 0.3034 - categorical_accuracy: 0.9059
24224/60000 [===========>..................] - ETA: 1:05 - loss: 0.3030 - categorical_accuracy: 0.9060
24256/60000 [===========>..................] - ETA: 1:04 - loss: 0.3028 - categorical_accuracy: 0.9061
24288/60000 [===========>..................] - ETA: 1:04 - loss: 0.3026 - categorical_accuracy: 0.9062
24320/60000 [===========>..................] - ETA: 1:04 - loss: 0.3024 - categorical_accuracy: 0.9062
24352/60000 [===========>..................] - ETA: 1:04 - loss: 0.3020 - categorical_accuracy: 0.9064
24384/60000 [===========>..................] - ETA: 1:04 - loss: 0.3021 - categorical_accuracy: 0.9064
24416/60000 [===========>..................] - ETA: 1:04 - loss: 0.3018 - categorical_accuracy: 0.9065
24448/60000 [===========>..................] - ETA: 1:04 - loss: 0.3015 - categorical_accuracy: 0.9067
24480/60000 [===========>..................] - ETA: 1:04 - loss: 0.3013 - categorical_accuracy: 0.9067
24512/60000 [===========>..................] - ETA: 1:04 - loss: 0.3011 - categorical_accuracy: 0.9068
24544/60000 [===========>..................] - ETA: 1:04 - loss: 0.3010 - categorical_accuracy: 0.9068
24576/60000 [===========>..................] - ETA: 1:04 - loss: 0.3009 - categorical_accuracy: 0.9068
24608/60000 [===========>..................] - ETA: 1:04 - loss: 0.3007 - categorical_accuracy: 0.9069
24640/60000 [===========>..................] - ETA: 1:04 - loss: 0.3004 - categorical_accuracy: 0.9070
24672/60000 [===========>..................] - ETA: 1:04 - loss: 0.3002 - categorical_accuracy: 0.9071
24704/60000 [===========>..................] - ETA: 1:04 - loss: 0.2999 - categorical_accuracy: 0.9072
24736/60000 [===========>..................] - ETA: 1:04 - loss: 0.2997 - categorical_accuracy: 0.9072
24768/60000 [===========>..................] - ETA: 1:04 - loss: 0.2996 - categorical_accuracy: 0.9072
24800/60000 [===========>..................] - ETA: 1:03 - loss: 0.2994 - categorical_accuracy: 0.9073
24832/60000 [===========>..................] - ETA: 1:03 - loss: 0.2994 - categorical_accuracy: 0.9073
24864/60000 [===========>..................] - ETA: 1:03 - loss: 0.2991 - categorical_accuracy: 0.9074
24896/60000 [===========>..................] - ETA: 1:03 - loss: 0.2989 - categorical_accuracy: 0.9075
24928/60000 [===========>..................] - ETA: 1:03 - loss: 0.2987 - categorical_accuracy: 0.9075
24960/60000 [===========>..................] - ETA: 1:03 - loss: 0.2984 - categorical_accuracy: 0.9076
24992/60000 [===========>..................] - ETA: 1:03 - loss: 0.2981 - categorical_accuracy: 0.9077
25024/60000 [===========>..................] - ETA: 1:03 - loss: 0.2982 - categorical_accuracy: 0.9077
25056/60000 [===========>..................] - ETA: 1:03 - loss: 0.2979 - categorical_accuracy: 0.9078
25088/60000 [===========>..................] - ETA: 1:03 - loss: 0.2978 - categorical_accuracy: 0.9078
25120/60000 [===========>..................] - ETA: 1:03 - loss: 0.2976 - categorical_accuracy: 0.9079
25152/60000 [===========>..................] - ETA: 1:03 - loss: 0.2973 - categorical_accuracy: 0.9080
25184/60000 [===========>..................] - ETA: 1:03 - loss: 0.2971 - categorical_accuracy: 0.9080
25216/60000 [===========>..................] - ETA: 1:03 - loss: 0.2970 - categorical_accuracy: 0.9081
25248/60000 [===========>..................] - ETA: 1:03 - loss: 0.2966 - categorical_accuracy: 0.9082
25280/60000 [===========>..................] - ETA: 1:03 - loss: 0.2963 - categorical_accuracy: 0.9083
25312/60000 [===========>..................] - ETA: 1:03 - loss: 0.2962 - categorical_accuracy: 0.9083
25344/60000 [===========>..................] - ETA: 1:02 - loss: 0.2962 - categorical_accuracy: 0.9082
25376/60000 [===========>..................] - ETA: 1:02 - loss: 0.2960 - categorical_accuracy: 0.9082
25408/60000 [===========>..................] - ETA: 1:02 - loss: 0.2958 - categorical_accuracy: 0.9082
25440/60000 [===========>..................] - ETA: 1:02 - loss: 0.2959 - categorical_accuracy: 0.9083
25472/60000 [===========>..................] - ETA: 1:02 - loss: 0.2956 - categorical_accuracy: 0.9084
25504/60000 [===========>..................] - ETA: 1:02 - loss: 0.2953 - categorical_accuracy: 0.9085
25536/60000 [===========>..................] - ETA: 1:02 - loss: 0.2950 - categorical_accuracy: 0.9086
25568/60000 [===========>..................] - ETA: 1:02 - loss: 0.2947 - categorical_accuracy: 0.9087
25600/60000 [===========>..................] - ETA: 1:02 - loss: 0.2945 - categorical_accuracy: 0.9087
25632/60000 [===========>..................] - ETA: 1:02 - loss: 0.2944 - categorical_accuracy: 0.9087
25664/60000 [===========>..................] - ETA: 1:02 - loss: 0.2941 - categorical_accuracy: 0.9088
25696/60000 [===========>..................] - ETA: 1:02 - loss: 0.2939 - categorical_accuracy: 0.9089
25728/60000 [===========>..................] - ETA: 1:02 - loss: 0.2939 - categorical_accuracy: 0.9089
25760/60000 [===========>..................] - ETA: 1:02 - loss: 0.2938 - categorical_accuracy: 0.9090
25792/60000 [===========>..................] - ETA: 1:02 - loss: 0.2936 - categorical_accuracy: 0.9090
25824/60000 [===========>..................] - ETA: 1:02 - loss: 0.2933 - categorical_accuracy: 0.9092
25856/60000 [===========>..................] - ETA: 1:02 - loss: 0.2932 - categorical_accuracy: 0.9092
25888/60000 [===========>..................] - ETA: 1:02 - loss: 0.2930 - categorical_accuracy: 0.9092
25920/60000 [===========>..................] - ETA: 1:01 - loss: 0.2931 - categorical_accuracy: 0.9092
25952/60000 [===========>..................] - ETA: 1:01 - loss: 0.2931 - categorical_accuracy: 0.9092
25984/60000 [===========>..................] - ETA: 1:01 - loss: 0.2931 - categorical_accuracy: 0.9092
26016/60000 [============>.................] - ETA: 1:01 - loss: 0.2930 - categorical_accuracy: 0.9092
26048/60000 [============>.................] - ETA: 1:01 - loss: 0.2927 - categorical_accuracy: 0.9093
26080/60000 [============>.................] - ETA: 1:01 - loss: 0.2928 - categorical_accuracy: 0.9094
26112/60000 [============>.................] - ETA: 1:01 - loss: 0.2925 - categorical_accuracy: 0.9095
26144/60000 [============>.................] - ETA: 1:01 - loss: 0.2923 - categorical_accuracy: 0.9095
26176/60000 [============>.................] - ETA: 1:01 - loss: 0.2920 - categorical_accuracy: 0.9096
26208/60000 [============>.................] - ETA: 1:01 - loss: 0.2920 - categorical_accuracy: 0.9096
26240/60000 [============>.................] - ETA: 1:01 - loss: 0.2917 - categorical_accuracy: 0.9098
26272/60000 [============>.................] - ETA: 1:01 - loss: 0.2915 - categorical_accuracy: 0.9098
26304/60000 [============>.................] - ETA: 1:01 - loss: 0.2912 - categorical_accuracy: 0.9099
26336/60000 [============>.................] - ETA: 1:01 - loss: 0.2910 - categorical_accuracy: 0.9099
26368/60000 [============>.................] - ETA: 1:01 - loss: 0.2908 - categorical_accuracy: 0.9100
26400/60000 [============>.................] - ETA: 1:01 - loss: 0.2905 - categorical_accuracy: 0.9101
26432/60000 [============>.................] - ETA: 1:00 - loss: 0.2903 - categorical_accuracy: 0.9101
26464/60000 [============>.................] - ETA: 1:00 - loss: 0.2903 - categorical_accuracy: 0.9102
26496/60000 [============>.................] - ETA: 1:00 - loss: 0.2900 - categorical_accuracy: 0.9103
26528/60000 [============>.................] - ETA: 1:00 - loss: 0.2899 - categorical_accuracy: 0.9103
26560/60000 [============>.................] - ETA: 1:00 - loss: 0.2897 - categorical_accuracy: 0.9104
26592/60000 [============>.................] - ETA: 1:00 - loss: 0.2895 - categorical_accuracy: 0.9104
26624/60000 [============>.................] - ETA: 1:00 - loss: 0.2893 - categorical_accuracy: 0.9105
26656/60000 [============>.................] - ETA: 1:00 - loss: 0.2893 - categorical_accuracy: 0.9105
26688/60000 [============>.................] - ETA: 1:00 - loss: 0.2894 - categorical_accuracy: 0.9106
26720/60000 [============>.................] - ETA: 1:00 - loss: 0.2891 - categorical_accuracy: 0.9106
26752/60000 [============>.................] - ETA: 1:00 - loss: 0.2888 - categorical_accuracy: 0.9107
26784/60000 [============>.................] - ETA: 1:00 - loss: 0.2885 - categorical_accuracy: 0.9108
26816/60000 [============>.................] - ETA: 1:00 - loss: 0.2884 - categorical_accuracy: 0.9108
26848/60000 [============>.................] - ETA: 1:00 - loss: 0.2881 - categorical_accuracy: 0.9109
26880/60000 [============>.................] - ETA: 1:00 - loss: 0.2881 - categorical_accuracy: 0.9109
26912/60000 [============>.................] - ETA: 1:00 - loss: 0.2880 - categorical_accuracy: 0.9109
26944/60000 [============>.................] - ETA: 1:00 - loss: 0.2877 - categorical_accuracy: 0.9110
26976/60000 [============>.................] - ETA: 59s - loss: 0.2876 - categorical_accuracy: 0.9110 
27008/60000 [============>.................] - ETA: 59s - loss: 0.2876 - categorical_accuracy: 0.9111
27040/60000 [============>.................] - ETA: 59s - loss: 0.2875 - categorical_accuracy: 0.9111
27072/60000 [============>.................] - ETA: 59s - loss: 0.2874 - categorical_accuracy: 0.9111
27104/60000 [============>.................] - ETA: 59s - loss: 0.2871 - categorical_accuracy: 0.9112
27136/60000 [============>.................] - ETA: 59s - loss: 0.2868 - categorical_accuracy: 0.9113
27168/60000 [============>.................] - ETA: 59s - loss: 0.2865 - categorical_accuracy: 0.9114
27200/60000 [============>.................] - ETA: 59s - loss: 0.2862 - categorical_accuracy: 0.9115
27232/60000 [============>.................] - ETA: 59s - loss: 0.2862 - categorical_accuracy: 0.9115
27264/60000 [============>.................] - ETA: 59s - loss: 0.2860 - categorical_accuracy: 0.9116
27296/60000 [============>.................] - ETA: 59s - loss: 0.2858 - categorical_accuracy: 0.9116
27328/60000 [============>.................] - ETA: 59s - loss: 0.2856 - categorical_accuracy: 0.9117
27360/60000 [============>.................] - ETA: 59s - loss: 0.2853 - categorical_accuracy: 0.9118
27392/60000 [============>.................] - ETA: 59s - loss: 0.2853 - categorical_accuracy: 0.9118
27424/60000 [============>.................] - ETA: 59s - loss: 0.2850 - categorical_accuracy: 0.9119
27456/60000 [============>.................] - ETA: 59s - loss: 0.2848 - categorical_accuracy: 0.9119
27488/60000 [============>.................] - ETA: 59s - loss: 0.2846 - categorical_accuracy: 0.9120
27520/60000 [============>.................] - ETA: 59s - loss: 0.2846 - categorical_accuracy: 0.9120
27552/60000 [============>.................] - ETA: 58s - loss: 0.2844 - categorical_accuracy: 0.9121
27584/60000 [============>.................] - ETA: 58s - loss: 0.2843 - categorical_accuracy: 0.9121
27616/60000 [============>.................] - ETA: 58s - loss: 0.2840 - categorical_accuracy: 0.9122
27648/60000 [============>.................] - ETA: 58s - loss: 0.2840 - categorical_accuracy: 0.9122
27680/60000 [============>.................] - ETA: 58s - loss: 0.2837 - categorical_accuracy: 0.9123
27712/60000 [============>.................] - ETA: 58s - loss: 0.2835 - categorical_accuracy: 0.9124
27744/60000 [============>.................] - ETA: 58s - loss: 0.2833 - categorical_accuracy: 0.9125
27776/60000 [============>.................] - ETA: 58s - loss: 0.2830 - categorical_accuracy: 0.9125
27808/60000 [============>.................] - ETA: 58s - loss: 0.2835 - categorical_accuracy: 0.9125
27840/60000 [============>.................] - ETA: 58s - loss: 0.2832 - categorical_accuracy: 0.9125
27872/60000 [============>.................] - ETA: 58s - loss: 0.2831 - categorical_accuracy: 0.9126
27904/60000 [============>.................] - ETA: 58s - loss: 0.2829 - categorical_accuracy: 0.9126
27936/60000 [============>.................] - ETA: 58s - loss: 0.2828 - categorical_accuracy: 0.9127
27968/60000 [============>.................] - ETA: 58s - loss: 0.2826 - categorical_accuracy: 0.9128
28000/60000 [=============>................] - ETA: 58s - loss: 0.2824 - categorical_accuracy: 0.9128
28032/60000 [=============>................] - ETA: 58s - loss: 0.2822 - categorical_accuracy: 0.9129
28064/60000 [=============>................] - ETA: 58s - loss: 0.2819 - categorical_accuracy: 0.9130
28096/60000 [=============>................] - ETA: 57s - loss: 0.2816 - categorical_accuracy: 0.9130
28128/60000 [=============>................] - ETA: 57s - loss: 0.2814 - categorical_accuracy: 0.9131
28160/60000 [=============>................] - ETA: 57s - loss: 0.2813 - categorical_accuracy: 0.9131
28192/60000 [=============>................] - ETA: 57s - loss: 0.2811 - categorical_accuracy: 0.9132
28224/60000 [=============>................] - ETA: 57s - loss: 0.2809 - categorical_accuracy: 0.9133
28256/60000 [=============>................] - ETA: 57s - loss: 0.2807 - categorical_accuracy: 0.9134
28288/60000 [=============>................] - ETA: 57s - loss: 0.2804 - categorical_accuracy: 0.9135
28320/60000 [=============>................] - ETA: 57s - loss: 0.2804 - categorical_accuracy: 0.9135
28352/60000 [=============>................] - ETA: 57s - loss: 0.2801 - categorical_accuracy: 0.9136
28384/60000 [=============>................] - ETA: 57s - loss: 0.2799 - categorical_accuracy: 0.9136
28416/60000 [=============>................] - ETA: 57s - loss: 0.2797 - categorical_accuracy: 0.9137
28448/60000 [=============>................] - ETA: 57s - loss: 0.2795 - categorical_accuracy: 0.9138
28480/60000 [=============>................] - ETA: 57s - loss: 0.2795 - categorical_accuracy: 0.9138
28512/60000 [=============>................] - ETA: 57s - loss: 0.2793 - categorical_accuracy: 0.9139
28544/60000 [=============>................] - ETA: 57s - loss: 0.2792 - categorical_accuracy: 0.9139
28576/60000 [=============>................] - ETA: 57s - loss: 0.2789 - categorical_accuracy: 0.9140
28608/60000 [=============>................] - ETA: 57s - loss: 0.2787 - categorical_accuracy: 0.9140
28640/60000 [=============>................] - ETA: 56s - loss: 0.2787 - categorical_accuracy: 0.9140
28672/60000 [=============>................] - ETA: 56s - loss: 0.2789 - categorical_accuracy: 0.9140
28704/60000 [=============>................] - ETA: 56s - loss: 0.2787 - categorical_accuracy: 0.9141
28736/60000 [=============>................] - ETA: 56s - loss: 0.2785 - categorical_accuracy: 0.9141
28768/60000 [=============>................] - ETA: 56s - loss: 0.2784 - categorical_accuracy: 0.9141
28800/60000 [=============>................] - ETA: 56s - loss: 0.2785 - categorical_accuracy: 0.9141
28832/60000 [=============>................] - ETA: 56s - loss: 0.2783 - categorical_accuracy: 0.9142
28864/60000 [=============>................] - ETA: 56s - loss: 0.2780 - categorical_accuracy: 0.9143
28896/60000 [=============>................] - ETA: 56s - loss: 0.2777 - categorical_accuracy: 0.9143
28928/60000 [=============>................] - ETA: 56s - loss: 0.2777 - categorical_accuracy: 0.9144
28960/60000 [=============>................] - ETA: 56s - loss: 0.2775 - categorical_accuracy: 0.9145
28992/60000 [=============>................] - ETA: 56s - loss: 0.2772 - categorical_accuracy: 0.9146
29024/60000 [=============>................] - ETA: 56s - loss: 0.2770 - categorical_accuracy: 0.9147
29056/60000 [=============>................] - ETA: 56s - loss: 0.2767 - categorical_accuracy: 0.9148
29088/60000 [=============>................] - ETA: 56s - loss: 0.2764 - categorical_accuracy: 0.9148
29120/60000 [=============>................] - ETA: 56s - loss: 0.2764 - categorical_accuracy: 0.9149
29152/60000 [=============>................] - ETA: 56s - loss: 0.2761 - categorical_accuracy: 0.9150
29184/60000 [=============>................] - ETA: 55s - loss: 0.2760 - categorical_accuracy: 0.9150
29216/60000 [=============>................] - ETA: 55s - loss: 0.2759 - categorical_accuracy: 0.9150
29248/60000 [=============>................] - ETA: 55s - loss: 0.2758 - categorical_accuracy: 0.9151
29280/60000 [=============>................] - ETA: 55s - loss: 0.2755 - categorical_accuracy: 0.9152
29312/60000 [=============>................] - ETA: 55s - loss: 0.2754 - categorical_accuracy: 0.9152
29344/60000 [=============>................] - ETA: 55s - loss: 0.2753 - categorical_accuracy: 0.9152
29376/60000 [=============>................] - ETA: 55s - loss: 0.2751 - categorical_accuracy: 0.9153
29408/60000 [=============>................] - ETA: 55s - loss: 0.2750 - categorical_accuracy: 0.9153
29440/60000 [=============>................] - ETA: 55s - loss: 0.2747 - categorical_accuracy: 0.9153
29472/60000 [=============>................] - ETA: 55s - loss: 0.2744 - categorical_accuracy: 0.9154
29504/60000 [=============>................] - ETA: 55s - loss: 0.2743 - categorical_accuracy: 0.9155
29536/60000 [=============>................] - ETA: 55s - loss: 0.2742 - categorical_accuracy: 0.9155
29568/60000 [=============>................] - ETA: 55s - loss: 0.2741 - categorical_accuracy: 0.9155
29600/60000 [=============>................] - ETA: 55s - loss: 0.2740 - categorical_accuracy: 0.9156
29632/60000 [=============>................] - ETA: 55s - loss: 0.2738 - categorical_accuracy: 0.9156
29664/60000 [=============>................] - ETA: 55s - loss: 0.2737 - categorical_accuracy: 0.9157
29696/60000 [=============>................] - ETA: 55s - loss: 0.2735 - categorical_accuracy: 0.9157
29728/60000 [=============>................] - ETA: 54s - loss: 0.2733 - categorical_accuracy: 0.9158
29760/60000 [=============>................] - ETA: 54s - loss: 0.2732 - categorical_accuracy: 0.9157
29792/60000 [=============>................] - ETA: 54s - loss: 0.2730 - categorical_accuracy: 0.9158
29824/60000 [=============>................] - ETA: 54s - loss: 0.2728 - categorical_accuracy: 0.9158
29856/60000 [=============>................] - ETA: 54s - loss: 0.2727 - categorical_accuracy: 0.9159
29888/60000 [=============>................] - ETA: 54s - loss: 0.2724 - categorical_accuracy: 0.9160
29920/60000 [=============>................] - ETA: 54s - loss: 0.2723 - categorical_accuracy: 0.9160
29952/60000 [=============>................] - ETA: 54s - loss: 0.2721 - categorical_accuracy: 0.9161
29984/60000 [=============>................] - ETA: 54s - loss: 0.2720 - categorical_accuracy: 0.9161
30016/60000 [==============>...............] - ETA: 54s - loss: 0.2720 - categorical_accuracy: 0.9160
30048/60000 [==============>...............] - ETA: 54s - loss: 0.2724 - categorical_accuracy: 0.9160
30080/60000 [==============>...............] - ETA: 54s - loss: 0.2723 - categorical_accuracy: 0.9161
30112/60000 [==============>...............] - ETA: 54s - loss: 0.2720 - categorical_accuracy: 0.9162
30144/60000 [==============>...............] - ETA: 54s - loss: 0.2719 - categorical_accuracy: 0.9162
30176/60000 [==============>...............] - ETA: 54s - loss: 0.2717 - categorical_accuracy: 0.9163
30208/60000 [==============>...............] - ETA: 54s - loss: 0.2715 - categorical_accuracy: 0.9164
30240/60000 [==============>...............] - ETA: 54s - loss: 0.2713 - categorical_accuracy: 0.9164
30272/60000 [==============>...............] - ETA: 53s - loss: 0.2712 - categorical_accuracy: 0.9165
30304/60000 [==============>...............] - ETA: 53s - loss: 0.2712 - categorical_accuracy: 0.9165
30336/60000 [==============>...............] - ETA: 53s - loss: 0.2710 - categorical_accuracy: 0.9165
30368/60000 [==============>...............] - ETA: 53s - loss: 0.2710 - categorical_accuracy: 0.9165
30400/60000 [==============>...............] - ETA: 53s - loss: 0.2710 - categorical_accuracy: 0.9165
30432/60000 [==============>...............] - ETA: 53s - loss: 0.2709 - categorical_accuracy: 0.9165
30464/60000 [==============>...............] - ETA: 53s - loss: 0.2707 - categorical_accuracy: 0.9165
30496/60000 [==============>...............] - ETA: 53s - loss: 0.2704 - categorical_accuracy: 0.9166
30528/60000 [==============>...............] - ETA: 53s - loss: 0.2703 - categorical_accuracy: 0.9166
30560/60000 [==============>...............] - ETA: 53s - loss: 0.2702 - categorical_accuracy: 0.9166
30592/60000 [==============>...............] - ETA: 53s - loss: 0.2700 - categorical_accuracy: 0.9167
30624/60000 [==============>...............] - ETA: 53s - loss: 0.2697 - categorical_accuracy: 0.9168
30656/60000 [==============>...............] - ETA: 53s - loss: 0.2695 - categorical_accuracy: 0.9169
30688/60000 [==============>...............] - ETA: 53s - loss: 0.2693 - categorical_accuracy: 0.9169
30720/60000 [==============>...............] - ETA: 53s - loss: 0.2692 - categorical_accuracy: 0.9170
30752/60000 [==============>...............] - ETA: 53s - loss: 0.2693 - categorical_accuracy: 0.9170
30784/60000 [==============>...............] - ETA: 53s - loss: 0.2692 - categorical_accuracy: 0.9170
30816/60000 [==============>...............] - ETA: 53s - loss: 0.2691 - categorical_accuracy: 0.9170
30848/60000 [==============>...............] - ETA: 52s - loss: 0.2689 - categorical_accuracy: 0.9171
30880/60000 [==============>...............] - ETA: 52s - loss: 0.2687 - categorical_accuracy: 0.9172
30912/60000 [==============>...............] - ETA: 52s - loss: 0.2686 - categorical_accuracy: 0.9172
30944/60000 [==============>...............] - ETA: 52s - loss: 0.2685 - categorical_accuracy: 0.9172
30976/60000 [==============>...............] - ETA: 52s - loss: 0.2684 - categorical_accuracy: 0.9172
31008/60000 [==============>...............] - ETA: 52s - loss: 0.2683 - categorical_accuracy: 0.9172
31040/60000 [==============>...............] - ETA: 52s - loss: 0.2683 - categorical_accuracy: 0.9172
31072/60000 [==============>...............] - ETA: 52s - loss: 0.2683 - categorical_accuracy: 0.9173
31104/60000 [==============>...............] - ETA: 52s - loss: 0.2680 - categorical_accuracy: 0.9174
31136/60000 [==============>...............] - ETA: 52s - loss: 0.2678 - categorical_accuracy: 0.9175
31168/60000 [==============>...............] - ETA: 52s - loss: 0.2678 - categorical_accuracy: 0.9175
31200/60000 [==============>...............] - ETA: 52s - loss: 0.2676 - categorical_accuracy: 0.9175
31232/60000 [==============>...............] - ETA: 52s - loss: 0.2676 - categorical_accuracy: 0.9175
31264/60000 [==============>...............] - ETA: 52s - loss: 0.2675 - categorical_accuracy: 0.9175
31296/60000 [==============>...............] - ETA: 52s - loss: 0.2673 - categorical_accuracy: 0.9176
31328/60000 [==============>...............] - ETA: 52s - loss: 0.2671 - categorical_accuracy: 0.9177
31360/60000 [==============>...............] - ETA: 52s - loss: 0.2669 - categorical_accuracy: 0.9178
31392/60000 [==============>...............] - ETA: 51s - loss: 0.2667 - categorical_accuracy: 0.9178
31424/60000 [==============>...............] - ETA: 51s - loss: 0.2667 - categorical_accuracy: 0.9178
31456/60000 [==============>...............] - ETA: 51s - loss: 0.2665 - categorical_accuracy: 0.9179
31488/60000 [==============>...............] - ETA: 51s - loss: 0.2664 - categorical_accuracy: 0.9179
31520/60000 [==============>...............] - ETA: 51s - loss: 0.2662 - categorical_accuracy: 0.9180
31552/60000 [==============>...............] - ETA: 51s - loss: 0.2660 - categorical_accuracy: 0.9180
31584/60000 [==============>...............] - ETA: 51s - loss: 0.2657 - categorical_accuracy: 0.9181
31616/60000 [==============>...............] - ETA: 51s - loss: 0.2656 - categorical_accuracy: 0.9181
31648/60000 [==============>...............] - ETA: 51s - loss: 0.2657 - categorical_accuracy: 0.9181
31680/60000 [==============>...............] - ETA: 51s - loss: 0.2656 - categorical_accuracy: 0.9182
31712/60000 [==============>...............] - ETA: 51s - loss: 0.2655 - categorical_accuracy: 0.9182
31744/60000 [==============>...............] - ETA: 51s - loss: 0.2653 - categorical_accuracy: 0.9183
31776/60000 [==============>...............] - ETA: 51s - loss: 0.2651 - categorical_accuracy: 0.9184
31808/60000 [==============>...............] - ETA: 51s - loss: 0.2649 - categorical_accuracy: 0.9184
31840/60000 [==============>...............] - ETA: 51s - loss: 0.2648 - categorical_accuracy: 0.9184
31872/60000 [==============>...............] - ETA: 51s - loss: 0.2648 - categorical_accuracy: 0.9185
31904/60000 [==============>...............] - ETA: 51s - loss: 0.2646 - categorical_accuracy: 0.9185
31936/60000 [==============>...............] - ETA: 50s - loss: 0.2643 - categorical_accuracy: 0.9186
31968/60000 [==============>...............] - ETA: 50s - loss: 0.2642 - categorical_accuracy: 0.9187
32000/60000 [===============>..............] - ETA: 50s - loss: 0.2643 - categorical_accuracy: 0.9187
32032/60000 [===============>..............] - ETA: 50s - loss: 0.2641 - categorical_accuracy: 0.9187
32064/60000 [===============>..............] - ETA: 50s - loss: 0.2640 - categorical_accuracy: 0.9188
32096/60000 [===============>..............] - ETA: 50s - loss: 0.2637 - categorical_accuracy: 0.9188
32128/60000 [===============>..............] - ETA: 50s - loss: 0.2635 - categorical_accuracy: 0.9189
32160/60000 [===============>..............] - ETA: 50s - loss: 0.2634 - categorical_accuracy: 0.9189
32192/60000 [===============>..............] - ETA: 50s - loss: 0.2634 - categorical_accuracy: 0.9190
32224/60000 [===============>..............] - ETA: 50s - loss: 0.2632 - categorical_accuracy: 0.9190
32256/60000 [===============>..............] - ETA: 50s - loss: 0.2632 - categorical_accuracy: 0.9190
32288/60000 [===============>..............] - ETA: 50s - loss: 0.2632 - categorical_accuracy: 0.9190
32320/60000 [===============>..............] - ETA: 50s - loss: 0.2629 - categorical_accuracy: 0.9191
32352/60000 [===============>..............] - ETA: 50s - loss: 0.2627 - categorical_accuracy: 0.9192
32384/60000 [===============>..............] - ETA: 50s - loss: 0.2628 - categorical_accuracy: 0.9191
32416/60000 [===============>..............] - ETA: 50s - loss: 0.2627 - categorical_accuracy: 0.9192
32448/60000 [===============>..............] - ETA: 50s - loss: 0.2625 - categorical_accuracy: 0.9193
32480/60000 [===============>..............] - ETA: 50s - loss: 0.2623 - categorical_accuracy: 0.9193
32512/60000 [===============>..............] - ETA: 49s - loss: 0.2623 - categorical_accuracy: 0.9193
32544/60000 [===============>..............] - ETA: 49s - loss: 0.2622 - categorical_accuracy: 0.9194
32576/60000 [===============>..............] - ETA: 49s - loss: 0.2620 - categorical_accuracy: 0.9194
32608/60000 [===============>..............] - ETA: 49s - loss: 0.2618 - categorical_accuracy: 0.9195
32640/60000 [===============>..............] - ETA: 49s - loss: 0.2616 - categorical_accuracy: 0.9195
32672/60000 [===============>..............] - ETA: 49s - loss: 0.2615 - categorical_accuracy: 0.9196
32704/60000 [===============>..............] - ETA: 49s - loss: 0.2613 - categorical_accuracy: 0.9196
32736/60000 [===============>..............] - ETA: 49s - loss: 0.2613 - categorical_accuracy: 0.9197
32768/60000 [===============>..............] - ETA: 49s - loss: 0.2611 - categorical_accuracy: 0.9197
32800/60000 [===============>..............] - ETA: 49s - loss: 0.2610 - categorical_accuracy: 0.9197
32832/60000 [===============>..............] - ETA: 49s - loss: 0.2607 - categorical_accuracy: 0.9198
32864/60000 [===============>..............] - ETA: 49s - loss: 0.2606 - categorical_accuracy: 0.9199
32896/60000 [===============>..............] - ETA: 49s - loss: 0.2606 - categorical_accuracy: 0.9198
32928/60000 [===============>..............] - ETA: 49s - loss: 0.2604 - categorical_accuracy: 0.9199
32960/60000 [===============>..............] - ETA: 49s - loss: 0.2602 - categorical_accuracy: 0.9200
32992/60000 [===============>..............] - ETA: 49s - loss: 0.2601 - categorical_accuracy: 0.9200
33024/60000 [===============>..............] - ETA: 49s - loss: 0.2599 - categorical_accuracy: 0.9200
33056/60000 [===============>..............] - ETA: 48s - loss: 0.2597 - categorical_accuracy: 0.9201
33088/60000 [===============>..............] - ETA: 48s - loss: 0.2596 - categorical_accuracy: 0.9201
33120/60000 [===============>..............] - ETA: 48s - loss: 0.2596 - categorical_accuracy: 0.9201
33152/60000 [===============>..............] - ETA: 48s - loss: 0.2594 - categorical_accuracy: 0.9202
33184/60000 [===============>..............] - ETA: 48s - loss: 0.2592 - categorical_accuracy: 0.9203
33216/60000 [===============>..............] - ETA: 48s - loss: 0.2591 - categorical_accuracy: 0.9203
33248/60000 [===============>..............] - ETA: 48s - loss: 0.2588 - categorical_accuracy: 0.9204
33280/60000 [===============>..............] - ETA: 48s - loss: 0.2587 - categorical_accuracy: 0.9204
33312/60000 [===============>..............] - ETA: 48s - loss: 0.2586 - categorical_accuracy: 0.9204
33344/60000 [===============>..............] - ETA: 48s - loss: 0.2584 - categorical_accuracy: 0.9205
33376/60000 [===============>..............] - ETA: 48s - loss: 0.2584 - categorical_accuracy: 0.9205
33408/60000 [===============>..............] - ETA: 48s - loss: 0.2583 - categorical_accuracy: 0.9206
33440/60000 [===============>..............] - ETA: 48s - loss: 0.2581 - categorical_accuracy: 0.9206
33472/60000 [===============>..............] - ETA: 48s - loss: 0.2580 - categorical_accuracy: 0.9207
33504/60000 [===============>..............] - ETA: 48s - loss: 0.2579 - categorical_accuracy: 0.9207
33536/60000 [===============>..............] - ETA: 48s - loss: 0.2577 - categorical_accuracy: 0.9208
33568/60000 [===============>..............] - ETA: 48s - loss: 0.2575 - categorical_accuracy: 0.9208
33600/60000 [===============>..............] - ETA: 47s - loss: 0.2573 - categorical_accuracy: 0.9209
33632/60000 [===============>..............] - ETA: 47s - loss: 0.2571 - categorical_accuracy: 0.9210
33664/60000 [===============>..............] - ETA: 47s - loss: 0.2570 - categorical_accuracy: 0.9210
33696/60000 [===============>..............] - ETA: 47s - loss: 0.2569 - categorical_accuracy: 0.9210
33728/60000 [===============>..............] - ETA: 47s - loss: 0.2567 - categorical_accuracy: 0.9210
33760/60000 [===============>..............] - ETA: 47s - loss: 0.2565 - categorical_accuracy: 0.9211
33792/60000 [===============>..............] - ETA: 47s - loss: 0.2564 - categorical_accuracy: 0.9211
33824/60000 [===============>..............] - ETA: 47s - loss: 0.2562 - categorical_accuracy: 0.9212
33856/60000 [===============>..............] - ETA: 47s - loss: 0.2561 - categorical_accuracy: 0.9212
33888/60000 [===============>..............] - ETA: 47s - loss: 0.2560 - categorical_accuracy: 0.9213
33920/60000 [===============>..............] - ETA: 47s - loss: 0.2560 - categorical_accuracy: 0.9212
33952/60000 [===============>..............] - ETA: 47s - loss: 0.2559 - categorical_accuracy: 0.9213
33984/60000 [===============>..............] - ETA: 47s - loss: 0.2557 - categorical_accuracy: 0.9213
34016/60000 [================>.............] - ETA: 47s - loss: 0.2555 - categorical_accuracy: 0.9214
34048/60000 [================>.............] - ETA: 47s - loss: 0.2553 - categorical_accuracy: 0.9215
34080/60000 [================>.............] - ETA: 47s - loss: 0.2552 - categorical_accuracy: 0.9214
34112/60000 [================>.............] - ETA: 47s - loss: 0.2550 - categorical_accuracy: 0.9215
34144/60000 [================>.............] - ETA: 46s - loss: 0.2550 - categorical_accuracy: 0.9215
34176/60000 [================>.............] - ETA: 46s - loss: 0.2549 - categorical_accuracy: 0.9215
34208/60000 [================>.............] - ETA: 46s - loss: 0.2547 - categorical_accuracy: 0.9216
34240/60000 [================>.............] - ETA: 46s - loss: 0.2545 - categorical_accuracy: 0.9216
34272/60000 [================>.............] - ETA: 46s - loss: 0.2544 - categorical_accuracy: 0.9217
34304/60000 [================>.............] - ETA: 46s - loss: 0.2547 - categorical_accuracy: 0.9217
34336/60000 [================>.............] - ETA: 46s - loss: 0.2545 - categorical_accuracy: 0.9217
34368/60000 [================>.............] - ETA: 46s - loss: 0.2544 - categorical_accuracy: 0.9218
34400/60000 [================>.............] - ETA: 46s - loss: 0.2542 - categorical_accuracy: 0.9218
34432/60000 [================>.............] - ETA: 46s - loss: 0.2541 - categorical_accuracy: 0.9219
34464/60000 [================>.............] - ETA: 46s - loss: 0.2541 - categorical_accuracy: 0.9219
34496/60000 [================>.............] - ETA: 46s - loss: 0.2539 - categorical_accuracy: 0.9219
34528/60000 [================>.............] - ETA: 46s - loss: 0.2540 - categorical_accuracy: 0.9219
34560/60000 [================>.............] - ETA: 46s - loss: 0.2538 - categorical_accuracy: 0.9220
34592/60000 [================>.............] - ETA: 46s - loss: 0.2536 - categorical_accuracy: 0.9221
34624/60000 [================>.............] - ETA: 46s - loss: 0.2534 - categorical_accuracy: 0.9222
34656/60000 [================>.............] - ETA: 46s - loss: 0.2532 - categorical_accuracy: 0.9222
34688/60000 [================>.............] - ETA: 45s - loss: 0.2531 - categorical_accuracy: 0.9222
34720/60000 [================>.............] - ETA: 45s - loss: 0.2530 - categorical_accuracy: 0.9222
34752/60000 [================>.............] - ETA: 45s - loss: 0.2528 - categorical_accuracy: 0.9223
34784/60000 [================>.............] - ETA: 45s - loss: 0.2527 - categorical_accuracy: 0.9223
34816/60000 [================>.............] - ETA: 45s - loss: 0.2525 - categorical_accuracy: 0.9224
34848/60000 [================>.............] - ETA: 45s - loss: 0.2523 - categorical_accuracy: 0.9225
34880/60000 [================>.............] - ETA: 45s - loss: 0.2521 - categorical_accuracy: 0.9225
34912/60000 [================>.............] - ETA: 45s - loss: 0.2519 - categorical_accuracy: 0.9226
34944/60000 [================>.............] - ETA: 45s - loss: 0.2518 - categorical_accuracy: 0.9226
34976/60000 [================>.............] - ETA: 45s - loss: 0.2516 - categorical_accuracy: 0.9227
35008/60000 [================>.............] - ETA: 45s - loss: 0.2515 - categorical_accuracy: 0.9227
35040/60000 [================>.............] - ETA: 45s - loss: 0.2514 - categorical_accuracy: 0.9227
35072/60000 [================>.............] - ETA: 45s - loss: 0.2512 - categorical_accuracy: 0.9228
35104/60000 [================>.............] - ETA: 45s - loss: 0.2510 - categorical_accuracy: 0.9228
35136/60000 [================>.............] - ETA: 45s - loss: 0.2509 - categorical_accuracy: 0.9228
35168/60000 [================>.............] - ETA: 45s - loss: 0.2509 - categorical_accuracy: 0.9229
35200/60000 [================>.............] - ETA: 45s - loss: 0.2507 - categorical_accuracy: 0.9229
35232/60000 [================>.............] - ETA: 44s - loss: 0.2505 - categorical_accuracy: 0.9230
35264/60000 [================>.............] - ETA: 44s - loss: 0.2503 - categorical_accuracy: 0.9230
35296/60000 [================>.............] - ETA: 44s - loss: 0.2501 - categorical_accuracy: 0.9231
35328/60000 [================>.............] - ETA: 44s - loss: 0.2500 - categorical_accuracy: 0.9231
35360/60000 [================>.............] - ETA: 44s - loss: 0.2498 - categorical_accuracy: 0.9232
35392/60000 [================>.............] - ETA: 44s - loss: 0.2496 - categorical_accuracy: 0.9232
35424/60000 [================>.............] - ETA: 44s - loss: 0.2494 - categorical_accuracy: 0.9233
35456/60000 [================>.............] - ETA: 44s - loss: 0.2492 - categorical_accuracy: 0.9234
35488/60000 [================>.............] - ETA: 44s - loss: 0.2490 - categorical_accuracy: 0.9234
35520/60000 [================>.............] - ETA: 44s - loss: 0.2488 - categorical_accuracy: 0.9235
35552/60000 [================>.............] - ETA: 44s - loss: 0.2488 - categorical_accuracy: 0.9235
35584/60000 [================>.............] - ETA: 44s - loss: 0.2486 - categorical_accuracy: 0.9236
35616/60000 [================>.............] - ETA: 44s - loss: 0.2484 - categorical_accuracy: 0.9236
35648/60000 [================>.............] - ETA: 44s - loss: 0.2484 - categorical_accuracy: 0.9236
35680/60000 [================>.............] - ETA: 44s - loss: 0.2483 - categorical_accuracy: 0.9237
35712/60000 [================>.............] - ETA: 44s - loss: 0.2482 - categorical_accuracy: 0.9237
35744/60000 [================>.............] - ETA: 44s - loss: 0.2480 - categorical_accuracy: 0.9237
35776/60000 [================>.............] - ETA: 43s - loss: 0.2478 - categorical_accuracy: 0.9238
35808/60000 [================>.............] - ETA: 43s - loss: 0.2476 - categorical_accuracy: 0.9238
35840/60000 [================>.............] - ETA: 43s - loss: 0.2474 - categorical_accuracy: 0.9239
35872/60000 [================>.............] - ETA: 43s - loss: 0.2473 - categorical_accuracy: 0.9240
35904/60000 [================>.............] - ETA: 43s - loss: 0.2472 - categorical_accuracy: 0.9240
35936/60000 [================>.............] - ETA: 43s - loss: 0.2471 - categorical_accuracy: 0.9240
35968/60000 [================>.............] - ETA: 43s - loss: 0.2471 - categorical_accuracy: 0.9240
36000/60000 [=================>............] - ETA: 43s - loss: 0.2470 - categorical_accuracy: 0.9241
36032/60000 [=================>............] - ETA: 43s - loss: 0.2469 - categorical_accuracy: 0.9241
36064/60000 [=================>............] - ETA: 43s - loss: 0.2468 - categorical_accuracy: 0.9241
36096/60000 [=================>............] - ETA: 43s - loss: 0.2466 - categorical_accuracy: 0.9242
36128/60000 [=================>............] - ETA: 43s - loss: 0.2464 - categorical_accuracy: 0.9242
36160/60000 [=================>............] - ETA: 43s - loss: 0.2463 - categorical_accuracy: 0.9243
36192/60000 [=================>............] - ETA: 43s - loss: 0.2463 - categorical_accuracy: 0.9243
36224/60000 [=================>............] - ETA: 43s - loss: 0.2462 - categorical_accuracy: 0.9244
36256/60000 [=================>............] - ETA: 43s - loss: 0.2461 - categorical_accuracy: 0.9244
36288/60000 [=================>............] - ETA: 43s - loss: 0.2462 - categorical_accuracy: 0.9244
36320/60000 [=================>............] - ETA: 43s - loss: 0.2462 - categorical_accuracy: 0.9244
36352/60000 [=================>............] - ETA: 42s - loss: 0.2462 - categorical_accuracy: 0.9244
36384/60000 [=================>............] - ETA: 42s - loss: 0.2460 - categorical_accuracy: 0.9244
36416/60000 [=================>............] - ETA: 42s - loss: 0.2459 - categorical_accuracy: 0.9245
36448/60000 [=================>............] - ETA: 42s - loss: 0.2459 - categorical_accuracy: 0.9244
36480/60000 [=================>............] - ETA: 42s - loss: 0.2458 - categorical_accuracy: 0.9245
36512/60000 [=================>............] - ETA: 42s - loss: 0.2457 - categorical_accuracy: 0.9245
36544/60000 [=================>............] - ETA: 42s - loss: 0.2456 - categorical_accuracy: 0.9246
36576/60000 [=================>............] - ETA: 42s - loss: 0.2457 - categorical_accuracy: 0.9245
36608/60000 [=================>............] - ETA: 42s - loss: 0.2456 - categorical_accuracy: 0.9246
36640/60000 [=================>............] - ETA: 42s - loss: 0.2454 - categorical_accuracy: 0.9246
36672/60000 [=================>............] - ETA: 42s - loss: 0.2453 - categorical_accuracy: 0.9247
36704/60000 [=================>............] - ETA: 42s - loss: 0.2452 - categorical_accuracy: 0.9247
36736/60000 [=================>............] - ETA: 42s - loss: 0.2451 - categorical_accuracy: 0.9247
36768/60000 [=================>............] - ETA: 42s - loss: 0.2450 - categorical_accuracy: 0.9247
36800/60000 [=================>............] - ETA: 42s - loss: 0.2449 - categorical_accuracy: 0.9248
36832/60000 [=================>............] - ETA: 42s - loss: 0.2450 - categorical_accuracy: 0.9248
36864/60000 [=================>............] - ETA: 42s - loss: 0.2449 - categorical_accuracy: 0.9248
36896/60000 [=================>............] - ETA: 41s - loss: 0.2447 - categorical_accuracy: 0.9249
36928/60000 [=================>............] - ETA: 41s - loss: 0.2445 - categorical_accuracy: 0.9249
36960/60000 [=================>............] - ETA: 41s - loss: 0.2444 - categorical_accuracy: 0.9250
36992/60000 [=================>............] - ETA: 41s - loss: 0.2445 - categorical_accuracy: 0.9250
37024/60000 [=================>............] - ETA: 41s - loss: 0.2444 - categorical_accuracy: 0.9250
37056/60000 [=================>............] - ETA: 41s - loss: 0.2442 - categorical_accuracy: 0.9251
37088/60000 [=================>............] - ETA: 41s - loss: 0.2441 - categorical_accuracy: 0.9251
37120/60000 [=================>............] - ETA: 41s - loss: 0.2441 - categorical_accuracy: 0.9252
37152/60000 [=================>............] - ETA: 41s - loss: 0.2439 - categorical_accuracy: 0.9252
37184/60000 [=================>............] - ETA: 41s - loss: 0.2437 - categorical_accuracy: 0.9252
37216/60000 [=================>............] - ETA: 41s - loss: 0.2437 - categorical_accuracy: 0.9252
37248/60000 [=================>............] - ETA: 41s - loss: 0.2436 - categorical_accuracy: 0.9253
37280/60000 [=================>............] - ETA: 41s - loss: 0.2435 - categorical_accuracy: 0.9253
37312/60000 [=================>............] - ETA: 41s - loss: 0.2433 - categorical_accuracy: 0.9254
37344/60000 [=================>............] - ETA: 41s - loss: 0.2431 - categorical_accuracy: 0.9254
37376/60000 [=================>............] - ETA: 41s - loss: 0.2429 - categorical_accuracy: 0.9255
37408/60000 [=================>............] - ETA: 41s - loss: 0.2428 - categorical_accuracy: 0.9256
37440/60000 [=================>............] - ETA: 40s - loss: 0.2426 - categorical_accuracy: 0.9256
37472/60000 [=================>............] - ETA: 40s - loss: 0.2424 - categorical_accuracy: 0.9257
37504/60000 [=================>............] - ETA: 40s - loss: 0.2422 - categorical_accuracy: 0.9257
37536/60000 [=================>............] - ETA: 40s - loss: 0.2422 - categorical_accuracy: 0.9257
37568/60000 [=================>............] - ETA: 40s - loss: 0.2420 - categorical_accuracy: 0.9258
37600/60000 [=================>............] - ETA: 40s - loss: 0.2418 - categorical_accuracy: 0.9259
37632/60000 [=================>............] - ETA: 40s - loss: 0.2418 - categorical_accuracy: 0.9259
37664/60000 [=================>............] - ETA: 40s - loss: 0.2417 - categorical_accuracy: 0.9259
37696/60000 [=================>............] - ETA: 40s - loss: 0.2416 - categorical_accuracy: 0.9260
37728/60000 [=================>............] - ETA: 40s - loss: 0.2414 - categorical_accuracy: 0.9260
37760/60000 [=================>............] - ETA: 40s - loss: 0.2413 - categorical_accuracy: 0.9260
37792/60000 [=================>............] - ETA: 40s - loss: 0.2412 - categorical_accuracy: 0.9261
37824/60000 [=================>............] - ETA: 40s - loss: 0.2411 - categorical_accuracy: 0.9261
37856/60000 [=================>............] - ETA: 40s - loss: 0.2412 - categorical_accuracy: 0.9261
37888/60000 [=================>............] - ETA: 40s - loss: 0.2410 - categorical_accuracy: 0.9262
37920/60000 [=================>............] - ETA: 40s - loss: 0.2411 - categorical_accuracy: 0.9261
37952/60000 [=================>............] - ETA: 40s - loss: 0.2410 - categorical_accuracy: 0.9261
37984/60000 [=================>............] - ETA: 39s - loss: 0.2408 - categorical_accuracy: 0.9262
38016/60000 [==================>...........] - ETA: 39s - loss: 0.2407 - categorical_accuracy: 0.9262
38048/60000 [==================>...........] - ETA: 39s - loss: 0.2406 - categorical_accuracy: 0.9263
38080/60000 [==================>...........] - ETA: 39s - loss: 0.2405 - categorical_accuracy: 0.9262
38112/60000 [==================>...........] - ETA: 39s - loss: 0.2403 - categorical_accuracy: 0.9263
38144/60000 [==================>...........] - ETA: 39s - loss: 0.2402 - categorical_accuracy: 0.9263
38176/60000 [==================>...........] - ETA: 39s - loss: 0.2401 - categorical_accuracy: 0.9264
38208/60000 [==================>...........] - ETA: 39s - loss: 0.2400 - categorical_accuracy: 0.9264
38240/60000 [==================>...........] - ETA: 39s - loss: 0.2399 - categorical_accuracy: 0.9265
38272/60000 [==================>...........] - ETA: 39s - loss: 0.2397 - categorical_accuracy: 0.9265
38304/60000 [==================>...........] - ETA: 39s - loss: 0.2396 - categorical_accuracy: 0.9266
38336/60000 [==================>...........] - ETA: 39s - loss: 0.2395 - categorical_accuracy: 0.9266
38368/60000 [==================>...........] - ETA: 39s - loss: 0.2394 - categorical_accuracy: 0.9266
38400/60000 [==================>...........] - ETA: 39s - loss: 0.2392 - categorical_accuracy: 0.9267
38432/60000 [==================>...........] - ETA: 39s - loss: 0.2391 - categorical_accuracy: 0.9267
38464/60000 [==================>...........] - ETA: 39s - loss: 0.2390 - categorical_accuracy: 0.9267
38496/60000 [==================>...........] - ETA: 39s - loss: 0.2389 - categorical_accuracy: 0.9267
38528/60000 [==================>...........] - ETA: 38s - loss: 0.2387 - categorical_accuracy: 0.9268
38560/60000 [==================>...........] - ETA: 38s - loss: 0.2386 - categorical_accuracy: 0.9268
38592/60000 [==================>...........] - ETA: 38s - loss: 0.2384 - categorical_accuracy: 0.9269
38624/60000 [==================>...........] - ETA: 38s - loss: 0.2382 - categorical_accuracy: 0.9270
38656/60000 [==================>...........] - ETA: 38s - loss: 0.2380 - categorical_accuracy: 0.9270
38688/60000 [==================>...........] - ETA: 38s - loss: 0.2379 - categorical_accuracy: 0.9271
38720/60000 [==================>...........] - ETA: 38s - loss: 0.2378 - categorical_accuracy: 0.9271
38752/60000 [==================>...........] - ETA: 38s - loss: 0.2377 - categorical_accuracy: 0.9271
38784/60000 [==================>...........] - ETA: 38s - loss: 0.2375 - categorical_accuracy: 0.9272
38816/60000 [==================>...........] - ETA: 38s - loss: 0.2376 - categorical_accuracy: 0.9272
38848/60000 [==================>...........] - ETA: 38s - loss: 0.2375 - categorical_accuracy: 0.9273
38880/60000 [==================>...........] - ETA: 38s - loss: 0.2374 - categorical_accuracy: 0.9273
38912/60000 [==================>...........] - ETA: 38s - loss: 0.2374 - categorical_accuracy: 0.9273
38944/60000 [==================>...........] - ETA: 38s - loss: 0.2374 - categorical_accuracy: 0.9273
38976/60000 [==================>...........] - ETA: 38s - loss: 0.2373 - categorical_accuracy: 0.9273
39008/60000 [==================>...........] - ETA: 38s - loss: 0.2373 - categorical_accuracy: 0.9273
39040/60000 [==================>...........] - ETA: 38s - loss: 0.2371 - categorical_accuracy: 0.9274
39072/60000 [==================>...........] - ETA: 37s - loss: 0.2371 - categorical_accuracy: 0.9274
39104/60000 [==================>...........] - ETA: 37s - loss: 0.2371 - categorical_accuracy: 0.9274
39136/60000 [==================>...........] - ETA: 37s - loss: 0.2372 - categorical_accuracy: 0.9274
39168/60000 [==================>...........] - ETA: 37s - loss: 0.2370 - categorical_accuracy: 0.9274
39200/60000 [==================>...........] - ETA: 37s - loss: 0.2369 - categorical_accuracy: 0.9275
39232/60000 [==================>...........] - ETA: 37s - loss: 0.2367 - categorical_accuracy: 0.9275
39264/60000 [==================>...........] - ETA: 37s - loss: 0.2366 - categorical_accuracy: 0.9275
39296/60000 [==================>...........] - ETA: 37s - loss: 0.2366 - categorical_accuracy: 0.9275
39328/60000 [==================>...........] - ETA: 37s - loss: 0.2364 - categorical_accuracy: 0.9276
39360/60000 [==================>...........] - ETA: 37s - loss: 0.2362 - categorical_accuracy: 0.9276
39392/60000 [==================>...........] - ETA: 37s - loss: 0.2361 - categorical_accuracy: 0.9277
39424/60000 [==================>...........] - ETA: 37s - loss: 0.2360 - categorical_accuracy: 0.9277
39456/60000 [==================>...........] - ETA: 37s - loss: 0.2360 - categorical_accuracy: 0.9277
39488/60000 [==================>...........] - ETA: 37s - loss: 0.2360 - categorical_accuracy: 0.9278
39520/60000 [==================>...........] - ETA: 37s - loss: 0.2360 - categorical_accuracy: 0.9277
39552/60000 [==================>...........] - ETA: 37s - loss: 0.2360 - categorical_accuracy: 0.9278
39584/60000 [==================>...........] - ETA: 37s - loss: 0.2359 - categorical_accuracy: 0.9278
39616/60000 [==================>...........] - ETA: 36s - loss: 0.2357 - categorical_accuracy: 0.9279
39648/60000 [==================>...........] - ETA: 36s - loss: 0.2356 - categorical_accuracy: 0.9279
39680/60000 [==================>...........] - ETA: 36s - loss: 0.2355 - categorical_accuracy: 0.9279
39712/60000 [==================>...........] - ETA: 36s - loss: 0.2353 - categorical_accuracy: 0.9280
39744/60000 [==================>...........] - ETA: 36s - loss: 0.2352 - categorical_accuracy: 0.9280
39776/60000 [==================>...........] - ETA: 36s - loss: 0.2351 - categorical_accuracy: 0.9281
39808/60000 [==================>...........] - ETA: 36s - loss: 0.2349 - categorical_accuracy: 0.9281
39840/60000 [==================>...........] - ETA: 36s - loss: 0.2349 - categorical_accuracy: 0.9281
39872/60000 [==================>...........] - ETA: 36s - loss: 0.2348 - categorical_accuracy: 0.9282
39904/60000 [==================>...........] - ETA: 36s - loss: 0.2349 - categorical_accuracy: 0.9282
39936/60000 [==================>...........] - ETA: 36s - loss: 0.2349 - categorical_accuracy: 0.9282
39968/60000 [==================>...........] - ETA: 36s - loss: 0.2349 - categorical_accuracy: 0.9282
40000/60000 [===================>..........] - ETA: 36s - loss: 0.2347 - categorical_accuracy: 0.9282
40032/60000 [===================>..........] - ETA: 36s - loss: 0.2347 - categorical_accuracy: 0.9282
40064/60000 [===================>..........] - ETA: 36s - loss: 0.2347 - categorical_accuracy: 0.9282
40096/60000 [===================>..........] - ETA: 36s - loss: 0.2347 - categorical_accuracy: 0.9282
40128/60000 [===================>..........] - ETA: 36s - loss: 0.2346 - categorical_accuracy: 0.9283
40160/60000 [===================>..........] - ETA: 35s - loss: 0.2345 - categorical_accuracy: 0.9283
40192/60000 [===================>..........] - ETA: 35s - loss: 0.2345 - categorical_accuracy: 0.9283
40224/60000 [===================>..........] - ETA: 35s - loss: 0.2344 - categorical_accuracy: 0.9283
40256/60000 [===================>..........] - ETA: 35s - loss: 0.2342 - categorical_accuracy: 0.9284
40288/60000 [===================>..........] - ETA: 35s - loss: 0.2342 - categorical_accuracy: 0.9284
40320/60000 [===================>..........] - ETA: 35s - loss: 0.2341 - categorical_accuracy: 0.9283
40352/60000 [===================>..........] - ETA: 35s - loss: 0.2339 - categorical_accuracy: 0.9284
40384/60000 [===================>..........] - ETA: 35s - loss: 0.2338 - categorical_accuracy: 0.9284
40416/60000 [===================>..........] - ETA: 35s - loss: 0.2336 - categorical_accuracy: 0.9285
40448/60000 [===================>..........] - ETA: 35s - loss: 0.2336 - categorical_accuracy: 0.9285
40480/60000 [===================>..........] - ETA: 35s - loss: 0.2336 - categorical_accuracy: 0.9285
40512/60000 [===================>..........] - ETA: 35s - loss: 0.2335 - categorical_accuracy: 0.9285
40544/60000 [===================>..........] - ETA: 35s - loss: 0.2333 - categorical_accuracy: 0.9286
40576/60000 [===================>..........] - ETA: 35s - loss: 0.2332 - categorical_accuracy: 0.9286
40608/60000 [===================>..........] - ETA: 35s - loss: 0.2331 - categorical_accuracy: 0.9286
40640/60000 [===================>..........] - ETA: 35s - loss: 0.2332 - categorical_accuracy: 0.9287
40672/60000 [===================>..........] - ETA: 35s - loss: 0.2331 - categorical_accuracy: 0.9287
40704/60000 [===================>..........] - ETA: 34s - loss: 0.2329 - categorical_accuracy: 0.9288
40736/60000 [===================>..........] - ETA: 34s - loss: 0.2328 - categorical_accuracy: 0.9288
40768/60000 [===================>..........] - ETA: 34s - loss: 0.2328 - categorical_accuracy: 0.9288
40800/60000 [===================>..........] - ETA: 34s - loss: 0.2328 - categorical_accuracy: 0.9288
40832/60000 [===================>..........] - ETA: 34s - loss: 0.2328 - categorical_accuracy: 0.9288
40864/60000 [===================>..........] - ETA: 34s - loss: 0.2328 - categorical_accuracy: 0.9288
40896/60000 [===================>..........] - ETA: 34s - loss: 0.2328 - categorical_accuracy: 0.9288
40928/60000 [===================>..........] - ETA: 34s - loss: 0.2326 - categorical_accuracy: 0.9289
40960/60000 [===================>..........] - ETA: 34s - loss: 0.2325 - categorical_accuracy: 0.9289
40992/60000 [===================>..........] - ETA: 34s - loss: 0.2323 - categorical_accuracy: 0.9290
41024/60000 [===================>..........] - ETA: 34s - loss: 0.2324 - categorical_accuracy: 0.9289
41056/60000 [===================>..........] - ETA: 34s - loss: 0.2323 - categorical_accuracy: 0.9290
41088/60000 [===================>..........] - ETA: 34s - loss: 0.2321 - categorical_accuracy: 0.9290
41120/60000 [===================>..........] - ETA: 34s - loss: 0.2321 - categorical_accuracy: 0.9290
41152/60000 [===================>..........] - ETA: 34s - loss: 0.2320 - categorical_accuracy: 0.9290
41184/60000 [===================>..........] - ETA: 34s - loss: 0.2319 - categorical_accuracy: 0.9291
41216/60000 [===================>..........] - ETA: 34s - loss: 0.2318 - categorical_accuracy: 0.9291
41248/60000 [===================>..........] - ETA: 33s - loss: 0.2317 - categorical_accuracy: 0.9291
41280/60000 [===================>..........] - ETA: 33s - loss: 0.2316 - categorical_accuracy: 0.9291
41312/60000 [===================>..........] - ETA: 33s - loss: 0.2315 - categorical_accuracy: 0.9292
41344/60000 [===================>..........] - ETA: 33s - loss: 0.2313 - categorical_accuracy: 0.9292
41376/60000 [===================>..........] - ETA: 33s - loss: 0.2312 - categorical_accuracy: 0.9293
41408/60000 [===================>..........] - ETA: 33s - loss: 0.2310 - categorical_accuracy: 0.9293
41440/60000 [===================>..........] - ETA: 33s - loss: 0.2311 - categorical_accuracy: 0.9293
41472/60000 [===================>..........] - ETA: 33s - loss: 0.2310 - categorical_accuracy: 0.9293
41504/60000 [===================>..........] - ETA: 33s - loss: 0.2309 - categorical_accuracy: 0.9294
41536/60000 [===================>..........] - ETA: 33s - loss: 0.2308 - categorical_accuracy: 0.9294
41568/60000 [===================>..........] - ETA: 33s - loss: 0.2306 - categorical_accuracy: 0.9294
41600/60000 [===================>..........] - ETA: 33s - loss: 0.2306 - categorical_accuracy: 0.9294
41632/60000 [===================>..........] - ETA: 33s - loss: 0.2304 - categorical_accuracy: 0.9295
41664/60000 [===================>..........] - ETA: 33s - loss: 0.2303 - categorical_accuracy: 0.9295
41696/60000 [===================>..........] - ETA: 33s - loss: 0.2302 - categorical_accuracy: 0.9296
41728/60000 [===================>..........] - ETA: 33s - loss: 0.2300 - categorical_accuracy: 0.9296
41760/60000 [===================>..........] - ETA: 33s - loss: 0.2300 - categorical_accuracy: 0.9296
41792/60000 [===================>..........] - ETA: 32s - loss: 0.2300 - categorical_accuracy: 0.9297
41824/60000 [===================>..........] - ETA: 32s - loss: 0.2298 - categorical_accuracy: 0.9297
41856/60000 [===================>..........] - ETA: 32s - loss: 0.2297 - categorical_accuracy: 0.9297
41888/60000 [===================>..........] - ETA: 32s - loss: 0.2295 - categorical_accuracy: 0.9298
41920/60000 [===================>..........] - ETA: 32s - loss: 0.2293 - categorical_accuracy: 0.9298
41952/60000 [===================>..........] - ETA: 32s - loss: 0.2292 - categorical_accuracy: 0.9299
41984/60000 [===================>..........] - ETA: 32s - loss: 0.2292 - categorical_accuracy: 0.9299
42016/60000 [====================>.........] - ETA: 32s - loss: 0.2291 - categorical_accuracy: 0.9299
42048/60000 [====================>.........] - ETA: 32s - loss: 0.2290 - categorical_accuracy: 0.9300
42080/60000 [====================>.........] - ETA: 32s - loss: 0.2288 - categorical_accuracy: 0.9300
42112/60000 [====================>.........] - ETA: 32s - loss: 0.2287 - categorical_accuracy: 0.9300
42144/60000 [====================>.........] - ETA: 32s - loss: 0.2286 - categorical_accuracy: 0.9300
42176/60000 [====================>.........] - ETA: 32s - loss: 0.2284 - categorical_accuracy: 0.9301
42208/60000 [====================>.........] - ETA: 32s - loss: 0.2283 - categorical_accuracy: 0.9302
42240/60000 [====================>.........] - ETA: 32s - loss: 0.2281 - categorical_accuracy: 0.9302
42272/60000 [====================>.........] - ETA: 32s - loss: 0.2281 - categorical_accuracy: 0.9302
42304/60000 [====================>.........] - ETA: 32s - loss: 0.2279 - categorical_accuracy: 0.9303
42336/60000 [====================>.........] - ETA: 32s - loss: 0.2277 - categorical_accuracy: 0.9303
42368/60000 [====================>.........] - ETA: 31s - loss: 0.2276 - categorical_accuracy: 0.9304
42400/60000 [====================>.........] - ETA: 31s - loss: 0.2274 - categorical_accuracy: 0.9304
42432/60000 [====================>.........] - ETA: 31s - loss: 0.2273 - categorical_accuracy: 0.9305
42464/60000 [====================>.........] - ETA: 31s - loss: 0.2273 - categorical_accuracy: 0.9305
42496/60000 [====================>.........] - ETA: 31s - loss: 0.2272 - categorical_accuracy: 0.9305
42528/60000 [====================>.........] - ETA: 31s - loss: 0.2271 - categorical_accuracy: 0.9305
42560/60000 [====================>.........] - ETA: 31s - loss: 0.2270 - categorical_accuracy: 0.9306
42592/60000 [====================>.........] - ETA: 31s - loss: 0.2269 - categorical_accuracy: 0.9306
42624/60000 [====================>.........] - ETA: 31s - loss: 0.2268 - categorical_accuracy: 0.9306
42656/60000 [====================>.........] - ETA: 31s - loss: 0.2266 - categorical_accuracy: 0.9307
42688/60000 [====================>.........] - ETA: 31s - loss: 0.2267 - categorical_accuracy: 0.9307
42720/60000 [====================>.........] - ETA: 31s - loss: 0.2266 - categorical_accuracy: 0.9307
42752/60000 [====================>.........] - ETA: 31s - loss: 0.2265 - categorical_accuracy: 0.9307
42784/60000 [====================>.........] - ETA: 31s - loss: 0.2264 - categorical_accuracy: 0.9307
42816/60000 [====================>.........] - ETA: 31s - loss: 0.2265 - categorical_accuracy: 0.9308
42848/60000 [====================>.........] - ETA: 31s - loss: 0.2264 - categorical_accuracy: 0.9308
42880/60000 [====================>.........] - ETA: 31s - loss: 0.2263 - categorical_accuracy: 0.9308
42912/60000 [====================>.........] - ETA: 30s - loss: 0.2261 - categorical_accuracy: 0.9309
42944/60000 [====================>.........] - ETA: 30s - loss: 0.2261 - categorical_accuracy: 0.9309
42976/60000 [====================>.........] - ETA: 30s - loss: 0.2260 - categorical_accuracy: 0.9309
43008/60000 [====================>.........] - ETA: 30s - loss: 0.2259 - categorical_accuracy: 0.9309
43040/60000 [====================>.........] - ETA: 30s - loss: 0.2258 - categorical_accuracy: 0.9309
43072/60000 [====================>.........] - ETA: 30s - loss: 0.2257 - categorical_accuracy: 0.9310
43104/60000 [====================>.........] - ETA: 30s - loss: 0.2256 - categorical_accuracy: 0.9310
43136/60000 [====================>.........] - ETA: 30s - loss: 0.2255 - categorical_accuracy: 0.9310
43168/60000 [====================>.........] - ETA: 30s - loss: 0.2254 - categorical_accuracy: 0.9311
43200/60000 [====================>.........] - ETA: 30s - loss: 0.2252 - categorical_accuracy: 0.9311
43232/60000 [====================>.........] - ETA: 30s - loss: 0.2253 - categorical_accuracy: 0.9311
43264/60000 [====================>.........] - ETA: 30s - loss: 0.2252 - categorical_accuracy: 0.9312
43296/60000 [====================>.........] - ETA: 30s - loss: 0.2250 - categorical_accuracy: 0.9312
43328/60000 [====================>.........] - ETA: 30s - loss: 0.2250 - categorical_accuracy: 0.9312
43360/60000 [====================>.........] - ETA: 30s - loss: 0.2252 - categorical_accuracy: 0.9312
43392/60000 [====================>.........] - ETA: 30s - loss: 0.2252 - categorical_accuracy: 0.9312
43424/60000 [====================>.........] - ETA: 30s - loss: 0.2252 - categorical_accuracy: 0.9312
43456/60000 [====================>.........] - ETA: 29s - loss: 0.2251 - categorical_accuracy: 0.9312
43488/60000 [====================>.........] - ETA: 29s - loss: 0.2251 - categorical_accuracy: 0.9312
43520/60000 [====================>.........] - ETA: 29s - loss: 0.2250 - categorical_accuracy: 0.9313
43552/60000 [====================>.........] - ETA: 29s - loss: 0.2249 - categorical_accuracy: 0.9313
43584/60000 [====================>.........] - ETA: 29s - loss: 0.2250 - categorical_accuracy: 0.9313
43616/60000 [====================>.........] - ETA: 29s - loss: 0.2249 - categorical_accuracy: 0.9313
43648/60000 [====================>.........] - ETA: 29s - loss: 0.2249 - categorical_accuracy: 0.9313
43680/60000 [====================>.........] - ETA: 29s - loss: 0.2249 - categorical_accuracy: 0.9313
43712/60000 [====================>.........] - ETA: 29s - loss: 0.2248 - categorical_accuracy: 0.9313
43744/60000 [====================>.........] - ETA: 29s - loss: 0.2247 - categorical_accuracy: 0.9314
43776/60000 [====================>.........] - ETA: 29s - loss: 0.2246 - categorical_accuracy: 0.9314
43808/60000 [====================>.........] - ETA: 29s - loss: 0.2244 - categorical_accuracy: 0.9315
43840/60000 [====================>.........] - ETA: 29s - loss: 0.2244 - categorical_accuracy: 0.9315
43872/60000 [====================>.........] - ETA: 29s - loss: 0.2243 - categorical_accuracy: 0.9315
43904/60000 [====================>.........] - ETA: 29s - loss: 0.2243 - categorical_accuracy: 0.9315
43936/60000 [====================>.........] - ETA: 29s - loss: 0.2242 - categorical_accuracy: 0.9315
43968/60000 [====================>.........] - ETA: 29s - loss: 0.2241 - categorical_accuracy: 0.9315
44000/60000 [=====================>........] - ETA: 28s - loss: 0.2240 - categorical_accuracy: 0.9316
44032/60000 [=====================>........] - ETA: 28s - loss: 0.2239 - categorical_accuracy: 0.9316
44064/60000 [=====================>........] - ETA: 28s - loss: 0.2238 - categorical_accuracy: 0.9316
44096/60000 [=====================>........] - ETA: 28s - loss: 0.2237 - categorical_accuracy: 0.9316
44128/60000 [=====================>........] - ETA: 28s - loss: 0.2236 - categorical_accuracy: 0.9317
44160/60000 [=====================>........] - ETA: 28s - loss: 0.2235 - categorical_accuracy: 0.9317
44192/60000 [=====================>........] - ETA: 28s - loss: 0.2234 - categorical_accuracy: 0.9317
44224/60000 [=====================>........] - ETA: 28s - loss: 0.2234 - categorical_accuracy: 0.9318
44256/60000 [=====================>........] - ETA: 28s - loss: 0.2232 - categorical_accuracy: 0.9318
44288/60000 [=====================>........] - ETA: 28s - loss: 0.2232 - categorical_accuracy: 0.9318
44320/60000 [=====================>........] - ETA: 28s - loss: 0.2231 - categorical_accuracy: 0.9318
44352/60000 [=====================>........] - ETA: 28s - loss: 0.2230 - categorical_accuracy: 0.9319
44384/60000 [=====================>........] - ETA: 28s - loss: 0.2229 - categorical_accuracy: 0.9319
44416/60000 [=====================>........] - ETA: 28s - loss: 0.2228 - categorical_accuracy: 0.9319
44448/60000 [=====================>........] - ETA: 28s - loss: 0.2229 - categorical_accuracy: 0.9319
44480/60000 [=====================>........] - ETA: 28s - loss: 0.2227 - categorical_accuracy: 0.9320
44512/60000 [=====================>........] - ETA: 28s - loss: 0.2227 - categorical_accuracy: 0.9319
44544/60000 [=====================>........] - ETA: 28s - loss: 0.2226 - categorical_accuracy: 0.9320
44576/60000 [=====================>........] - ETA: 27s - loss: 0.2224 - categorical_accuracy: 0.9320
44608/60000 [=====================>........] - ETA: 27s - loss: 0.2224 - categorical_accuracy: 0.9320
44640/60000 [=====================>........] - ETA: 27s - loss: 0.2222 - categorical_accuracy: 0.9321
44672/60000 [=====================>........] - ETA: 27s - loss: 0.2221 - categorical_accuracy: 0.9321
44704/60000 [=====================>........] - ETA: 27s - loss: 0.2220 - categorical_accuracy: 0.9322
44736/60000 [=====================>........] - ETA: 27s - loss: 0.2219 - categorical_accuracy: 0.9322
44768/60000 [=====================>........] - ETA: 27s - loss: 0.2218 - categorical_accuracy: 0.9322
44800/60000 [=====================>........] - ETA: 27s - loss: 0.2218 - categorical_accuracy: 0.9322
44832/60000 [=====================>........] - ETA: 27s - loss: 0.2216 - categorical_accuracy: 0.9322
44864/60000 [=====================>........] - ETA: 27s - loss: 0.2215 - categorical_accuracy: 0.9322
44896/60000 [=====================>........] - ETA: 27s - loss: 0.2214 - categorical_accuracy: 0.9323
44928/60000 [=====================>........] - ETA: 27s - loss: 0.2214 - categorical_accuracy: 0.9323
44960/60000 [=====================>........] - ETA: 27s - loss: 0.2213 - categorical_accuracy: 0.9323
44992/60000 [=====================>........] - ETA: 27s - loss: 0.2212 - categorical_accuracy: 0.9323
45024/60000 [=====================>........] - ETA: 27s - loss: 0.2211 - categorical_accuracy: 0.9324
45056/60000 [=====================>........] - ETA: 27s - loss: 0.2209 - categorical_accuracy: 0.9324
45088/60000 [=====================>........] - ETA: 27s - loss: 0.2208 - categorical_accuracy: 0.9325
45120/60000 [=====================>........] - ETA: 26s - loss: 0.2206 - categorical_accuracy: 0.9325
45152/60000 [=====================>........] - ETA: 26s - loss: 0.2206 - categorical_accuracy: 0.9325
45184/60000 [=====================>........] - ETA: 26s - loss: 0.2205 - categorical_accuracy: 0.9325
45216/60000 [=====================>........] - ETA: 26s - loss: 0.2203 - categorical_accuracy: 0.9326
45248/60000 [=====================>........] - ETA: 26s - loss: 0.2203 - categorical_accuracy: 0.9325
45280/60000 [=====================>........] - ETA: 26s - loss: 0.2201 - categorical_accuracy: 0.9326
45312/60000 [=====================>........] - ETA: 26s - loss: 0.2202 - categorical_accuracy: 0.9326
45344/60000 [=====================>........] - ETA: 26s - loss: 0.2203 - categorical_accuracy: 0.9326
45376/60000 [=====================>........] - ETA: 26s - loss: 0.2202 - categorical_accuracy: 0.9327
45408/60000 [=====================>........] - ETA: 26s - loss: 0.2201 - categorical_accuracy: 0.9327
45440/60000 [=====================>........] - ETA: 26s - loss: 0.2199 - categorical_accuracy: 0.9327
45472/60000 [=====================>........] - ETA: 26s - loss: 0.2198 - categorical_accuracy: 0.9328
45504/60000 [=====================>........] - ETA: 26s - loss: 0.2197 - categorical_accuracy: 0.9328
45536/60000 [=====================>........] - ETA: 26s - loss: 0.2195 - categorical_accuracy: 0.9329
45568/60000 [=====================>........] - ETA: 26s - loss: 0.2195 - categorical_accuracy: 0.9328
45600/60000 [=====================>........] - ETA: 26s - loss: 0.2194 - categorical_accuracy: 0.9329
45632/60000 [=====================>........] - ETA: 26s - loss: 0.2193 - categorical_accuracy: 0.9329
45664/60000 [=====================>........] - ETA: 25s - loss: 0.2193 - categorical_accuracy: 0.9329
45696/60000 [=====================>........] - ETA: 25s - loss: 0.2194 - categorical_accuracy: 0.9329
45728/60000 [=====================>........] - ETA: 25s - loss: 0.2193 - categorical_accuracy: 0.9329
45760/60000 [=====================>........] - ETA: 25s - loss: 0.2192 - categorical_accuracy: 0.9330
45792/60000 [=====================>........] - ETA: 25s - loss: 0.2191 - categorical_accuracy: 0.9330
45824/60000 [=====================>........] - ETA: 25s - loss: 0.2190 - categorical_accuracy: 0.9330
45856/60000 [=====================>........] - ETA: 25s - loss: 0.2189 - categorical_accuracy: 0.9331
45888/60000 [=====================>........] - ETA: 25s - loss: 0.2188 - categorical_accuracy: 0.9331
45920/60000 [=====================>........] - ETA: 25s - loss: 0.2188 - categorical_accuracy: 0.9331
45952/60000 [=====================>........] - ETA: 25s - loss: 0.2188 - categorical_accuracy: 0.9331
45984/60000 [=====================>........] - ETA: 25s - loss: 0.2187 - categorical_accuracy: 0.9331
46016/60000 [======================>.......] - ETA: 25s - loss: 0.2187 - categorical_accuracy: 0.9331
46048/60000 [======================>.......] - ETA: 25s - loss: 0.2186 - categorical_accuracy: 0.9331
46080/60000 [======================>.......] - ETA: 25s - loss: 0.2185 - categorical_accuracy: 0.9332
46112/60000 [======================>.......] - ETA: 25s - loss: 0.2184 - categorical_accuracy: 0.9332
46144/60000 [======================>.......] - ETA: 25s - loss: 0.2183 - categorical_accuracy: 0.9332
46176/60000 [======================>.......] - ETA: 25s - loss: 0.2182 - categorical_accuracy: 0.9332
46208/60000 [======================>.......] - ETA: 24s - loss: 0.2181 - categorical_accuracy: 0.9333
46240/60000 [======================>.......] - ETA: 24s - loss: 0.2180 - categorical_accuracy: 0.9333
46272/60000 [======================>.......] - ETA: 24s - loss: 0.2179 - categorical_accuracy: 0.9333
46304/60000 [======================>.......] - ETA: 24s - loss: 0.2178 - categorical_accuracy: 0.9333
46336/60000 [======================>.......] - ETA: 24s - loss: 0.2177 - categorical_accuracy: 0.9334
46368/60000 [======================>.......] - ETA: 24s - loss: 0.2176 - categorical_accuracy: 0.9334
46400/60000 [======================>.......] - ETA: 24s - loss: 0.2175 - categorical_accuracy: 0.9334
46432/60000 [======================>.......] - ETA: 24s - loss: 0.2174 - categorical_accuracy: 0.9335
46464/60000 [======================>.......] - ETA: 24s - loss: 0.2173 - categorical_accuracy: 0.9335
46496/60000 [======================>.......] - ETA: 24s - loss: 0.2172 - categorical_accuracy: 0.9335
46528/60000 [======================>.......] - ETA: 24s - loss: 0.2172 - categorical_accuracy: 0.9336
46560/60000 [======================>.......] - ETA: 24s - loss: 0.2170 - categorical_accuracy: 0.9336
46592/60000 [======================>.......] - ETA: 24s - loss: 0.2169 - categorical_accuracy: 0.9336
46624/60000 [======================>.......] - ETA: 24s - loss: 0.2170 - categorical_accuracy: 0.9336
46656/60000 [======================>.......] - ETA: 24s - loss: 0.2168 - categorical_accuracy: 0.9336
46688/60000 [======================>.......] - ETA: 24s - loss: 0.2167 - categorical_accuracy: 0.9337
46720/60000 [======================>.......] - ETA: 24s - loss: 0.2166 - categorical_accuracy: 0.9337
46752/60000 [======================>.......] - ETA: 24s - loss: 0.2165 - categorical_accuracy: 0.9338
46784/60000 [======================>.......] - ETA: 23s - loss: 0.2165 - categorical_accuracy: 0.9338
46816/60000 [======================>.......] - ETA: 23s - loss: 0.2164 - categorical_accuracy: 0.9338
46848/60000 [======================>.......] - ETA: 23s - loss: 0.2162 - categorical_accuracy: 0.9339
46880/60000 [======================>.......] - ETA: 23s - loss: 0.2162 - categorical_accuracy: 0.9339
46912/60000 [======================>.......] - ETA: 23s - loss: 0.2162 - categorical_accuracy: 0.9339
46944/60000 [======================>.......] - ETA: 23s - loss: 0.2161 - categorical_accuracy: 0.9339
46976/60000 [======================>.......] - ETA: 23s - loss: 0.2160 - categorical_accuracy: 0.9339
47008/60000 [======================>.......] - ETA: 23s - loss: 0.2159 - categorical_accuracy: 0.9339
47040/60000 [======================>.......] - ETA: 23s - loss: 0.2158 - categorical_accuracy: 0.9339
47072/60000 [======================>.......] - ETA: 23s - loss: 0.2157 - categorical_accuracy: 0.9340
47104/60000 [======================>.......] - ETA: 23s - loss: 0.2157 - categorical_accuracy: 0.9340
47136/60000 [======================>.......] - ETA: 23s - loss: 0.2156 - categorical_accuracy: 0.9340
47168/60000 [======================>.......] - ETA: 23s - loss: 0.2155 - categorical_accuracy: 0.9340
47200/60000 [======================>.......] - ETA: 23s - loss: 0.2153 - categorical_accuracy: 0.9341
47232/60000 [======================>.......] - ETA: 23s - loss: 0.2154 - categorical_accuracy: 0.9340
47264/60000 [======================>.......] - ETA: 23s - loss: 0.2153 - categorical_accuracy: 0.9341
47296/60000 [======================>.......] - ETA: 23s - loss: 0.2154 - categorical_accuracy: 0.9341
47328/60000 [======================>.......] - ETA: 22s - loss: 0.2152 - categorical_accuracy: 0.9341
47360/60000 [======================>.......] - ETA: 22s - loss: 0.2151 - categorical_accuracy: 0.9341
47392/60000 [======================>.......] - ETA: 22s - loss: 0.2151 - categorical_accuracy: 0.9341
47424/60000 [======================>.......] - ETA: 22s - loss: 0.2151 - categorical_accuracy: 0.9341
47456/60000 [======================>.......] - ETA: 22s - loss: 0.2150 - categorical_accuracy: 0.9342
47488/60000 [======================>.......] - ETA: 22s - loss: 0.2149 - categorical_accuracy: 0.9342
47520/60000 [======================>.......] - ETA: 22s - loss: 0.2148 - categorical_accuracy: 0.9342
47552/60000 [======================>.......] - ETA: 22s - loss: 0.2146 - categorical_accuracy: 0.9343
47584/60000 [======================>.......] - ETA: 22s - loss: 0.2146 - categorical_accuracy: 0.9343
47616/60000 [======================>.......] - ETA: 22s - loss: 0.2144 - categorical_accuracy: 0.9343
47648/60000 [======================>.......] - ETA: 22s - loss: 0.2143 - categorical_accuracy: 0.9344
47680/60000 [======================>.......] - ETA: 22s - loss: 0.2142 - categorical_accuracy: 0.9344
47712/60000 [======================>.......] - ETA: 22s - loss: 0.2141 - categorical_accuracy: 0.9344
47744/60000 [======================>.......] - ETA: 22s - loss: 0.2142 - categorical_accuracy: 0.9344
47776/60000 [======================>.......] - ETA: 22s - loss: 0.2140 - categorical_accuracy: 0.9344
47808/60000 [======================>.......] - ETA: 22s - loss: 0.2139 - categorical_accuracy: 0.9345
47840/60000 [======================>.......] - ETA: 22s - loss: 0.2138 - categorical_accuracy: 0.9345
47872/60000 [======================>.......] - ETA: 21s - loss: 0.2137 - categorical_accuracy: 0.9346
47904/60000 [======================>.......] - ETA: 21s - loss: 0.2136 - categorical_accuracy: 0.9346
47936/60000 [======================>.......] - ETA: 21s - loss: 0.2135 - categorical_accuracy: 0.9346
47968/60000 [======================>.......] - ETA: 21s - loss: 0.2134 - categorical_accuracy: 0.9346
48000/60000 [=======================>......] - ETA: 21s - loss: 0.2133 - categorical_accuracy: 0.9347
48032/60000 [=======================>......] - ETA: 21s - loss: 0.2133 - categorical_accuracy: 0.9347
48064/60000 [=======================>......] - ETA: 21s - loss: 0.2134 - categorical_accuracy: 0.9347
48096/60000 [=======================>......] - ETA: 21s - loss: 0.2133 - categorical_accuracy: 0.9347
48128/60000 [=======================>......] - ETA: 21s - loss: 0.2132 - categorical_accuracy: 0.9347
48160/60000 [=======================>......] - ETA: 21s - loss: 0.2131 - categorical_accuracy: 0.9348
48192/60000 [=======================>......] - ETA: 21s - loss: 0.2130 - categorical_accuracy: 0.9348
48224/60000 [=======================>......] - ETA: 21s - loss: 0.2129 - categorical_accuracy: 0.9348
48256/60000 [=======================>......] - ETA: 21s - loss: 0.2128 - categorical_accuracy: 0.9349
48288/60000 [=======================>......] - ETA: 21s - loss: 0.2127 - categorical_accuracy: 0.9349
48320/60000 [=======================>......] - ETA: 21s - loss: 0.2126 - categorical_accuracy: 0.9349
48352/60000 [=======================>......] - ETA: 21s - loss: 0.2125 - categorical_accuracy: 0.9349
48384/60000 [=======================>......] - ETA: 21s - loss: 0.2124 - categorical_accuracy: 0.9350
48416/60000 [=======================>......] - ETA: 20s - loss: 0.2122 - categorical_accuracy: 0.9350
48448/60000 [=======================>......] - ETA: 20s - loss: 0.2121 - categorical_accuracy: 0.9351
48480/60000 [=======================>......] - ETA: 20s - loss: 0.2121 - categorical_accuracy: 0.9351
48512/60000 [=======================>......] - ETA: 20s - loss: 0.2120 - categorical_accuracy: 0.9351
48544/60000 [=======================>......] - ETA: 20s - loss: 0.2119 - categorical_accuracy: 0.9352
48576/60000 [=======================>......] - ETA: 20s - loss: 0.2119 - categorical_accuracy: 0.9351
48608/60000 [=======================>......] - ETA: 20s - loss: 0.2118 - categorical_accuracy: 0.9352
48640/60000 [=======================>......] - ETA: 20s - loss: 0.2117 - categorical_accuracy: 0.9352
48672/60000 [=======================>......] - ETA: 20s - loss: 0.2117 - categorical_accuracy: 0.9352
48704/60000 [=======================>......] - ETA: 20s - loss: 0.2116 - categorical_accuracy: 0.9353
48736/60000 [=======================>......] - ETA: 20s - loss: 0.2115 - categorical_accuracy: 0.9353
48768/60000 [=======================>......] - ETA: 20s - loss: 0.2114 - categorical_accuracy: 0.9353
48800/60000 [=======================>......] - ETA: 20s - loss: 0.2114 - categorical_accuracy: 0.9353
48832/60000 [=======================>......] - ETA: 20s - loss: 0.2112 - categorical_accuracy: 0.9353
48864/60000 [=======================>......] - ETA: 20s - loss: 0.2111 - categorical_accuracy: 0.9354
48896/60000 [=======================>......] - ETA: 20s - loss: 0.2110 - categorical_accuracy: 0.9354
48928/60000 [=======================>......] - ETA: 20s - loss: 0.2109 - categorical_accuracy: 0.9354
48960/60000 [=======================>......] - ETA: 19s - loss: 0.2108 - categorical_accuracy: 0.9355
48992/60000 [=======================>......] - ETA: 19s - loss: 0.2107 - categorical_accuracy: 0.9355
49024/60000 [=======================>......] - ETA: 19s - loss: 0.2106 - categorical_accuracy: 0.9355
49056/60000 [=======================>......] - ETA: 19s - loss: 0.2105 - categorical_accuracy: 0.9356
49088/60000 [=======================>......] - ETA: 19s - loss: 0.2104 - categorical_accuracy: 0.9356
49120/60000 [=======================>......] - ETA: 19s - loss: 0.2105 - categorical_accuracy: 0.9356
49152/60000 [=======================>......] - ETA: 19s - loss: 0.2103 - categorical_accuracy: 0.9356
49184/60000 [=======================>......] - ETA: 19s - loss: 0.2104 - categorical_accuracy: 0.9356
49216/60000 [=======================>......] - ETA: 19s - loss: 0.2103 - categorical_accuracy: 0.9357
49248/60000 [=======================>......] - ETA: 19s - loss: 0.2102 - categorical_accuracy: 0.9357
49280/60000 [=======================>......] - ETA: 19s - loss: 0.2102 - categorical_accuracy: 0.9357
49312/60000 [=======================>......] - ETA: 19s - loss: 0.2102 - categorical_accuracy: 0.9357
49344/60000 [=======================>......] - ETA: 19s - loss: 0.2101 - categorical_accuracy: 0.9357
49376/60000 [=======================>......] - ETA: 19s - loss: 0.2100 - categorical_accuracy: 0.9358
49408/60000 [=======================>......] - ETA: 19s - loss: 0.2099 - categorical_accuracy: 0.9358
49440/60000 [=======================>......] - ETA: 19s - loss: 0.2097 - categorical_accuracy: 0.9358
49472/60000 [=======================>......] - ETA: 19s - loss: 0.2096 - categorical_accuracy: 0.9359
49504/60000 [=======================>......] - ETA: 19s - loss: 0.2095 - categorical_accuracy: 0.9359
49536/60000 [=======================>......] - ETA: 18s - loss: 0.2094 - categorical_accuracy: 0.9360
49568/60000 [=======================>......] - ETA: 18s - loss: 0.2093 - categorical_accuracy: 0.9360
49600/60000 [=======================>......] - ETA: 18s - loss: 0.2091 - categorical_accuracy: 0.9360
49632/60000 [=======================>......] - ETA: 18s - loss: 0.2090 - categorical_accuracy: 0.9361
49664/60000 [=======================>......] - ETA: 18s - loss: 0.2091 - categorical_accuracy: 0.9361
49696/60000 [=======================>......] - ETA: 18s - loss: 0.2090 - categorical_accuracy: 0.9361
49728/60000 [=======================>......] - ETA: 18s - loss: 0.2089 - categorical_accuracy: 0.9361
49760/60000 [=======================>......] - ETA: 18s - loss: 0.2088 - categorical_accuracy: 0.9362
49792/60000 [=======================>......] - ETA: 18s - loss: 0.2087 - categorical_accuracy: 0.9362
49824/60000 [=======================>......] - ETA: 18s - loss: 0.2086 - categorical_accuracy: 0.9362
49856/60000 [=======================>......] - ETA: 18s - loss: 0.2085 - categorical_accuracy: 0.9363
49888/60000 [=======================>......] - ETA: 18s - loss: 0.2085 - categorical_accuracy: 0.9363
49920/60000 [=======================>......] - ETA: 18s - loss: 0.2084 - categorical_accuracy: 0.9363
49952/60000 [=======================>......] - ETA: 18s - loss: 0.2082 - categorical_accuracy: 0.9364
49984/60000 [=======================>......] - ETA: 18s - loss: 0.2081 - categorical_accuracy: 0.9364
50016/60000 [========================>.....] - ETA: 18s - loss: 0.2080 - categorical_accuracy: 0.9364
50048/60000 [========================>.....] - ETA: 18s - loss: 0.2080 - categorical_accuracy: 0.9364
50080/60000 [========================>.....] - ETA: 17s - loss: 0.2080 - categorical_accuracy: 0.9364
50112/60000 [========================>.....] - ETA: 17s - loss: 0.2079 - categorical_accuracy: 0.9365
50144/60000 [========================>.....] - ETA: 17s - loss: 0.2078 - categorical_accuracy: 0.9365
50176/60000 [========================>.....] - ETA: 17s - loss: 0.2077 - categorical_accuracy: 0.9365
50208/60000 [========================>.....] - ETA: 17s - loss: 0.2076 - categorical_accuracy: 0.9365
50240/60000 [========================>.....] - ETA: 17s - loss: 0.2076 - categorical_accuracy: 0.9365
50272/60000 [========================>.....] - ETA: 17s - loss: 0.2075 - categorical_accuracy: 0.9366
50304/60000 [========================>.....] - ETA: 17s - loss: 0.2074 - categorical_accuracy: 0.9366
50336/60000 [========================>.....] - ETA: 17s - loss: 0.2075 - categorical_accuracy: 0.9366
50368/60000 [========================>.....] - ETA: 17s - loss: 0.2075 - categorical_accuracy: 0.9366
50400/60000 [========================>.....] - ETA: 17s - loss: 0.2074 - categorical_accuracy: 0.9366
50432/60000 [========================>.....] - ETA: 17s - loss: 0.2074 - categorical_accuracy: 0.9366
50464/60000 [========================>.....] - ETA: 17s - loss: 0.2074 - categorical_accuracy: 0.9366
50496/60000 [========================>.....] - ETA: 17s - loss: 0.2074 - categorical_accuracy: 0.9366
50528/60000 [========================>.....] - ETA: 17s - loss: 0.2073 - categorical_accuracy: 0.9367
50560/60000 [========================>.....] - ETA: 17s - loss: 0.2072 - categorical_accuracy: 0.9367
50592/60000 [========================>.....] - ETA: 17s - loss: 0.2071 - categorical_accuracy: 0.9367
50624/60000 [========================>.....] - ETA: 16s - loss: 0.2070 - categorical_accuracy: 0.9367
50656/60000 [========================>.....] - ETA: 16s - loss: 0.2069 - categorical_accuracy: 0.9368
50688/60000 [========================>.....] - ETA: 16s - loss: 0.2068 - categorical_accuracy: 0.9368
50720/60000 [========================>.....] - ETA: 16s - loss: 0.2067 - categorical_accuracy: 0.9368
50752/60000 [========================>.....] - ETA: 16s - loss: 0.2067 - categorical_accuracy: 0.9368
50784/60000 [========================>.....] - ETA: 16s - loss: 0.2066 - categorical_accuracy: 0.9369
50816/60000 [========================>.....] - ETA: 16s - loss: 0.2065 - categorical_accuracy: 0.9369
50848/60000 [========================>.....] - ETA: 16s - loss: 0.2064 - categorical_accuracy: 0.9369
50880/60000 [========================>.....] - ETA: 16s - loss: 0.2063 - categorical_accuracy: 0.9370
50912/60000 [========================>.....] - ETA: 16s - loss: 0.2062 - categorical_accuracy: 0.9370
50944/60000 [========================>.....] - ETA: 16s - loss: 0.2061 - categorical_accuracy: 0.9370
50976/60000 [========================>.....] - ETA: 16s - loss: 0.2060 - categorical_accuracy: 0.9371
51008/60000 [========================>.....] - ETA: 16s - loss: 0.2060 - categorical_accuracy: 0.9371
51040/60000 [========================>.....] - ETA: 16s - loss: 0.2059 - categorical_accuracy: 0.9371
51072/60000 [========================>.....] - ETA: 16s - loss: 0.2058 - categorical_accuracy: 0.9372
51104/60000 [========================>.....] - ETA: 16s - loss: 0.2056 - categorical_accuracy: 0.9372
51136/60000 [========================>.....] - ETA: 16s - loss: 0.2055 - categorical_accuracy: 0.9372
51168/60000 [========================>.....] - ETA: 15s - loss: 0.2055 - categorical_accuracy: 0.9373
51200/60000 [========================>.....] - ETA: 15s - loss: 0.2055 - categorical_accuracy: 0.9373
51232/60000 [========================>.....] - ETA: 15s - loss: 0.2054 - categorical_accuracy: 0.9373
51264/60000 [========================>.....] - ETA: 15s - loss: 0.2053 - categorical_accuracy: 0.9374
51296/60000 [========================>.....] - ETA: 15s - loss: 0.2054 - categorical_accuracy: 0.9373
51328/60000 [========================>.....] - ETA: 15s - loss: 0.2054 - categorical_accuracy: 0.9373
51360/60000 [========================>.....] - ETA: 15s - loss: 0.2053 - categorical_accuracy: 0.9374
51392/60000 [========================>.....] - ETA: 15s - loss: 0.2053 - categorical_accuracy: 0.9374
51424/60000 [========================>.....] - ETA: 15s - loss: 0.2051 - categorical_accuracy: 0.9374
51456/60000 [========================>.....] - ETA: 15s - loss: 0.2051 - categorical_accuracy: 0.9375
51488/60000 [========================>.....] - ETA: 15s - loss: 0.2051 - categorical_accuracy: 0.9375
51520/60000 [========================>.....] - ETA: 15s - loss: 0.2049 - categorical_accuracy: 0.9375
51552/60000 [========================>.....] - ETA: 15s - loss: 0.2049 - categorical_accuracy: 0.9375
51584/60000 [========================>.....] - ETA: 15s - loss: 0.2047 - categorical_accuracy: 0.9376
51616/60000 [========================>.....] - ETA: 15s - loss: 0.2047 - categorical_accuracy: 0.9376
51648/60000 [========================>.....] - ETA: 15s - loss: 0.2047 - categorical_accuracy: 0.9376
51680/60000 [========================>.....] - ETA: 15s - loss: 0.2045 - categorical_accuracy: 0.9376
51712/60000 [========================>.....] - ETA: 15s - loss: 0.2046 - categorical_accuracy: 0.9376
51744/60000 [========================>.....] - ETA: 14s - loss: 0.2045 - categorical_accuracy: 0.9376
51776/60000 [========================>.....] - ETA: 14s - loss: 0.2044 - categorical_accuracy: 0.9376
51808/60000 [========================>.....] - ETA: 14s - loss: 0.2044 - categorical_accuracy: 0.9376
51840/60000 [========================>.....] - ETA: 14s - loss: 0.2043 - categorical_accuracy: 0.9377
51872/60000 [========================>.....] - ETA: 14s - loss: 0.2042 - categorical_accuracy: 0.9377
51904/60000 [========================>.....] - ETA: 14s - loss: 0.2041 - categorical_accuracy: 0.9377
51936/60000 [========================>.....] - ETA: 14s - loss: 0.2040 - categorical_accuracy: 0.9378
51968/60000 [========================>.....] - ETA: 14s - loss: 0.2041 - categorical_accuracy: 0.9378
52000/60000 [=========================>....] - ETA: 14s - loss: 0.2040 - categorical_accuracy: 0.9378
52032/60000 [=========================>....] - ETA: 14s - loss: 0.2039 - categorical_accuracy: 0.9378
52064/60000 [=========================>....] - ETA: 14s - loss: 0.2038 - categorical_accuracy: 0.9378
52096/60000 [=========================>....] - ETA: 14s - loss: 0.2037 - categorical_accuracy: 0.9379
52128/60000 [=========================>....] - ETA: 14s - loss: 0.2036 - categorical_accuracy: 0.9379
52160/60000 [=========================>....] - ETA: 14s - loss: 0.2036 - categorical_accuracy: 0.9379
52192/60000 [=========================>....] - ETA: 14s - loss: 0.2035 - categorical_accuracy: 0.9379
52224/60000 [=========================>....] - ETA: 14s - loss: 0.2034 - categorical_accuracy: 0.9380
52256/60000 [=========================>....] - ETA: 14s - loss: 0.2032 - categorical_accuracy: 0.9380
52288/60000 [=========================>....] - ETA: 13s - loss: 0.2034 - categorical_accuracy: 0.9380
52320/60000 [=========================>....] - ETA: 13s - loss: 0.2033 - categorical_accuracy: 0.9380
52352/60000 [=========================>....] - ETA: 13s - loss: 0.2032 - categorical_accuracy: 0.9381
52384/60000 [=========================>....] - ETA: 13s - loss: 0.2032 - categorical_accuracy: 0.9381
52416/60000 [=========================>....] - ETA: 13s - loss: 0.2030 - categorical_accuracy: 0.9381
52448/60000 [=========================>....] - ETA: 13s - loss: 0.2029 - categorical_accuracy: 0.9381
52480/60000 [=========================>....] - ETA: 13s - loss: 0.2029 - categorical_accuracy: 0.9382
52512/60000 [=========================>....] - ETA: 13s - loss: 0.2029 - categorical_accuracy: 0.9382
52544/60000 [=========================>....] - ETA: 13s - loss: 0.2028 - categorical_accuracy: 0.9382
52576/60000 [=========================>....] - ETA: 13s - loss: 0.2027 - categorical_accuracy: 0.9383
52608/60000 [=========================>....] - ETA: 13s - loss: 0.2028 - categorical_accuracy: 0.9382
52640/60000 [=========================>....] - ETA: 13s - loss: 0.2028 - categorical_accuracy: 0.9382
52672/60000 [=========================>....] - ETA: 13s - loss: 0.2026 - categorical_accuracy: 0.9383
52704/60000 [=========================>....] - ETA: 13s - loss: 0.2026 - categorical_accuracy: 0.9383
52736/60000 [=========================>....] - ETA: 13s - loss: 0.2025 - categorical_accuracy: 0.9383
52768/60000 [=========================>....] - ETA: 13s - loss: 0.2024 - categorical_accuracy: 0.9383
52800/60000 [=========================>....] - ETA: 13s - loss: 0.2023 - categorical_accuracy: 0.9384
52832/60000 [=========================>....] - ETA: 12s - loss: 0.2023 - categorical_accuracy: 0.9384
52864/60000 [=========================>....] - ETA: 12s - loss: 0.2022 - categorical_accuracy: 0.9384
52896/60000 [=========================>....] - ETA: 12s - loss: 0.2021 - categorical_accuracy: 0.9384
52928/60000 [=========================>....] - ETA: 12s - loss: 0.2020 - categorical_accuracy: 0.9384
52960/60000 [=========================>....] - ETA: 12s - loss: 0.2019 - categorical_accuracy: 0.9384
52992/60000 [=========================>....] - ETA: 12s - loss: 0.2019 - categorical_accuracy: 0.9384
53024/60000 [=========================>....] - ETA: 12s - loss: 0.2018 - categorical_accuracy: 0.9385
53056/60000 [=========================>....] - ETA: 12s - loss: 0.2017 - categorical_accuracy: 0.9385
53088/60000 [=========================>....] - ETA: 12s - loss: 0.2016 - categorical_accuracy: 0.9385
53120/60000 [=========================>....] - ETA: 12s - loss: 0.2017 - categorical_accuracy: 0.9385
53152/60000 [=========================>....] - ETA: 12s - loss: 0.2016 - categorical_accuracy: 0.9386
53184/60000 [=========================>....] - ETA: 12s - loss: 0.2015 - categorical_accuracy: 0.9386
53216/60000 [=========================>....] - ETA: 12s - loss: 0.2015 - categorical_accuracy: 0.9386
53248/60000 [=========================>....] - ETA: 12s - loss: 0.2014 - categorical_accuracy: 0.9386
53280/60000 [=========================>....] - ETA: 12s - loss: 0.2014 - categorical_accuracy: 0.9386
53312/60000 [=========================>....] - ETA: 12s - loss: 0.2014 - categorical_accuracy: 0.9387
53344/60000 [=========================>....] - ETA: 12s - loss: 0.2013 - categorical_accuracy: 0.9387
53376/60000 [=========================>....] - ETA: 12s - loss: 0.2014 - categorical_accuracy: 0.9387
53408/60000 [=========================>....] - ETA: 11s - loss: 0.2014 - categorical_accuracy: 0.9387
53440/60000 [=========================>....] - ETA: 11s - loss: 0.2014 - categorical_accuracy: 0.9387
53472/60000 [=========================>....] - ETA: 11s - loss: 0.2013 - categorical_accuracy: 0.9387
53504/60000 [=========================>....] - ETA: 11s - loss: 0.2013 - categorical_accuracy: 0.9387
53536/60000 [=========================>....] - ETA: 11s - loss: 0.2012 - categorical_accuracy: 0.9388
53568/60000 [=========================>....] - ETA: 11s - loss: 0.2013 - categorical_accuracy: 0.9388
53600/60000 [=========================>....] - ETA: 11s - loss: 0.2012 - categorical_accuracy: 0.9388
53632/60000 [=========================>....] - ETA: 11s - loss: 0.2011 - categorical_accuracy: 0.9388
53664/60000 [=========================>....] - ETA: 11s - loss: 0.2010 - categorical_accuracy: 0.9389
53696/60000 [=========================>....] - ETA: 11s - loss: 0.2009 - categorical_accuracy: 0.9389
53728/60000 [=========================>....] - ETA: 11s - loss: 0.2008 - categorical_accuracy: 0.9389
53760/60000 [=========================>....] - ETA: 11s - loss: 0.2009 - categorical_accuracy: 0.9389
53792/60000 [=========================>....] - ETA: 11s - loss: 0.2008 - categorical_accuracy: 0.9390
53824/60000 [=========================>....] - ETA: 11s - loss: 0.2007 - categorical_accuracy: 0.9390
53856/60000 [=========================>....] - ETA: 11s - loss: 0.2006 - categorical_accuracy: 0.9390
53888/60000 [=========================>....] - ETA: 11s - loss: 0.2005 - categorical_accuracy: 0.9391
53920/60000 [=========================>....] - ETA: 11s - loss: 0.2004 - categorical_accuracy: 0.9391
53952/60000 [=========================>....] - ETA: 10s - loss: 0.2003 - categorical_accuracy: 0.9391
53984/60000 [=========================>....] - ETA: 10s - loss: 0.2002 - categorical_accuracy: 0.9392
54016/60000 [==========================>...] - ETA: 10s - loss: 0.2003 - categorical_accuracy: 0.9392
54048/60000 [==========================>...] - ETA: 10s - loss: 0.2002 - categorical_accuracy: 0.9392
54080/60000 [==========================>...] - ETA: 10s - loss: 0.2001 - categorical_accuracy: 0.9392
54112/60000 [==========================>...] - ETA: 10s - loss: 0.2000 - categorical_accuracy: 0.9393
54144/60000 [==========================>...] - ETA: 10s - loss: 0.1999 - categorical_accuracy: 0.9393
54176/60000 [==========================>...] - ETA: 10s - loss: 0.1998 - categorical_accuracy: 0.9393
54208/60000 [==========================>...] - ETA: 10s - loss: 0.1997 - categorical_accuracy: 0.9394
54240/60000 [==========================>...] - ETA: 10s - loss: 0.1996 - categorical_accuracy: 0.9394
54272/60000 [==========================>...] - ETA: 10s - loss: 0.1996 - categorical_accuracy: 0.9394
54304/60000 [==========================>...] - ETA: 10s - loss: 0.1995 - categorical_accuracy: 0.9394
54336/60000 [==========================>...] - ETA: 10s - loss: 0.1995 - categorical_accuracy: 0.9394
54368/60000 [==========================>...] - ETA: 10s - loss: 0.1994 - categorical_accuracy: 0.9395
54400/60000 [==========================>...] - ETA: 10s - loss: 0.1993 - categorical_accuracy: 0.9395
54432/60000 [==========================>...] - ETA: 10s - loss: 0.1993 - categorical_accuracy: 0.9395
54464/60000 [==========================>...] - ETA: 10s - loss: 0.1993 - categorical_accuracy: 0.9395
54496/60000 [==========================>...] - ETA: 9s - loss: 0.1992 - categorical_accuracy: 0.9395 
54528/60000 [==========================>...] - ETA: 9s - loss: 0.1991 - categorical_accuracy: 0.9396
54560/60000 [==========================>...] - ETA: 9s - loss: 0.1990 - categorical_accuracy: 0.9396
54592/60000 [==========================>...] - ETA: 9s - loss: 0.1989 - categorical_accuracy: 0.9396
54624/60000 [==========================>...] - ETA: 9s - loss: 0.1989 - categorical_accuracy: 0.9397
54656/60000 [==========================>...] - ETA: 9s - loss: 0.1989 - categorical_accuracy: 0.9397
54688/60000 [==========================>...] - ETA: 9s - loss: 0.1988 - categorical_accuracy: 0.9397
54720/60000 [==========================>...] - ETA: 9s - loss: 0.1987 - categorical_accuracy: 0.9397
54752/60000 [==========================>...] - ETA: 9s - loss: 0.1987 - categorical_accuracy: 0.9397
54784/60000 [==========================>...] - ETA: 9s - loss: 0.1987 - categorical_accuracy: 0.9397
54816/60000 [==========================>...] - ETA: 9s - loss: 0.1986 - categorical_accuracy: 0.9397
54848/60000 [==========================>...] - ETA: 9s - loss: 0.1988 - categorical_accuracy: 0.9397
54880/60000 [==========================>...] - ETA: 9s - loss: 0.1988 - categorical_accuracy: 0.9398
54912/60000 [==========================>...] - ETA: 9s - loss: 0.1987 - categorical_accuracy: 0.9398
54944/60000 [==========================>...] - ETA: 9s - loss: 0.1986 - categorical_accuracy: 0.9398
54976/60000 [==========================>...] - ETA: 9s - loss: 0.1985 - categorical_accuracy: 0.9399
55008/60000 [==========================>...] - ETA: 9s - loss: 0.1984 - categorical_accuracy: 0.9399
55040/60000 [==========================>...] - ETA: 8s - loss: 0.1983 - categorical_accuracy: 0.9399
55072/60000 [==========================>...] - ETA: 8s - loss: 0.1982 - categorical_accuracy: 0.9399
55104/60000 [==========================>...] - ETA: 8s - loss: 0.1981 - categorical_accuracy: 0.9399
55136/60000 [==========================>...] - ETA: 8s - loss: 0.1981 - categorical_accuracy: 0.9400
55168/60000 [==========================>...] - ETA: 8s - loss: 0.1980 - categorical_accuracy: 0.9400
55200/60000 [==========================>...] - ETA: 8s - loss: 0.1980 - categorical_accuracy: 0.9400
55232/60000 [==========================>...] - ETA: 8s - loss: 0.1980 - categorical_accuracy: 0.9400
55264/60000 [==========================>...] - ETA: 8s - loss: 0.1979 - categorical_accuracy: 0.9401
55296/60000 [==========================>...] - ETA: 8s - loss: 0.1978 - categorical_accuracy: 0.9401
55328/60000 [==========================>...] - ETA: 8s - loss: 0.1977 - categorical_accuracy: 0.9401
55360/60000 [==========================>...] - ETA: 8s - loss: 0.1976 - categorical_accuracy: 0.9401
55392/60000 [==========================>...] - ETA: 8s - loss: 0.1975 - categorical_accuracy: 0.9402
55424/60000 [==========================>...] - ETA: 8s - loss: 0.1976 - categorical_accuracy: 0.9401
55456/60000 [==========================>...] - ETA: 8s - loss: 0.1976 - categorical_accuracy: 0.9401
55488/60000 [==========================>...] - ETA: 8s - loss: 0.1976 - categorical_accuracy: 0.9401
55520/60000 [==========================>...] - ETA: 8s - loss: 0.1976 - categorical_accuracy: 0.9401
55552/60000 [==========================>...] - ETA: 8s - loss: 0.1975 - categorical_accuracy: 0.9401
55584/60000 [==========================>...] - ETA: 7s - loss: 0.1974 - categorical_accuracy: 0.9401
55616/60000 [==========================>...] - ETA: 7s - loss: 0.1974 - categorical_accuracy: 0.9402
55648/60000 [==========================>...] - ETA: 7s - loss: 0.1973 - categorical_accuracy: 0.9402
55680/60000 [==========================>...] - ETA: 7s - loss: 0.1972 - categorical_accuracy: 0.9402
55712/60000 [==========================>...] - ETA: 7s - loss: 0.1971 - categorical_accuracy: 0.9402
55744/60000 [==========================>...] - ETA: 7s - loss: 0.1971 - categorical_accuracy: 0.9402
55776/60000 [==========================>...] - ETA: 7s - loss: 0.1971 - categorical_accuracy: 0.9403
55808/60000 [==========================>...] - ETA: 7s - loss: 0.1971 - categorical_accuracy: 0.9403
55840/60000 [==========================>...] - ETA: 7s - loss: 0.1970 - categorical_accuracy: 0.9403
55872/60000 [==========================>...] - ETA: 7s - loss: 0.1969 - categorical_accuracy: 0.9403
55904/60000 [==========================>...] - ETA: 7s - loss: 0.1969 - categorical_accuracy: 0.9403
55936/60000 [==========================>...] - ETA: 7s - loss: 0.1968 - categorical_accuracy: 0.9403
55968/60000 [==========================>...] - ETA: 7s - loss: 0.1968 - categorical_accuracy: 0.9404
56000/60000 [===========================>..] - ETA: 7s - loss: 0.1968 - categorical_accuracy: 0.9404
56032/60000 [===========================>..] - ETA: 7s - loss: 0.1967 - categorical_accuracy: 0.9404
56064/60000 [===========================>..] - ETA: 7s - loss: 0.1966 - categorical_accuracy: 0.9404
56096/60000 [===========================>..] - ETA: 7s - loss: 0.1965 - categorical_accuracy: 0.9404
56128/60000 [===========================>..] - ETA: 7s - loss: 0.1964 - categorical_accuracy: 0.9404
56160/60000 [===========================>..] - ETA: 6s - loss: 0.1963 - categorical_accuracy: 0.9405
56192/60000 [===========================>..] - ETA: 6s - loss: 0.1962 - categorical_accuracy: 0.9405
56224/60000 [===========================>..] - ETA: 6s - loss: 0.1962 - categorical_accuracy: 0.9405
56256/60000 [===========================>..] - ETA: 6s - loss: 0.1961 - categorical_accuracy: 0.9406
56288/60000 [===========================>..] - ETA: 6s - loss: 0.1960 - categorical_accuracy: 0.9406
56320/60000 [===========================>..] - ETA: 6s - loss: 0.1959 - categorical_accuracy: 0.9406
56352/60000 [===========================>..] - ETA: 6s - loss: 0.1958 - categorical_accuracy: 0.9406
56384/60000 [===========================>..] - ETA: 6s - loss: 0.1957 - categorical_accuracy: 0.9407
56416/60000 [===========================>..] - ETA: 6s - loss: 0.1956 - categorical_accuracy: 0.9407
56448/60000 [===========================>..] - ETA: 6s - loss: 0.1956 - categorical_accuracy: 0.9407
56480/60000 [===========================>..] - ETA: 6s - loss: 0.1955 - categorical_accuracy: 0.9407
56512/60000 [===========================>..] - ETA: 6s - loss: 0.1954 - categorical_accuracy: 0.9407
56544/60000 [===========================>..] - ETA: 6s - loss: 0.1955 - categorical_accuracy: 0.9407
56576/60000 [===========================>..] - ETA: 6s - loss: 0.1955 - categorical_accuracy: 0.9407
56608/60000 [===========================>..] - ETA: 6s - loss: 0.1954 - categorical_accuracy: 0.9408
56640/60000 [===========================>..] - ETA: 6s - loss: 0.1953 - categorical_accuracy: 0.9408
56672/60000 [===========================>..] - ETA: 6s - loss: 0.1953 - categorical_accuracy: 0.9408
56704/60000 [===========================>..] - ETA: 5s - loss: 0.1952 - categorical_accuracy: 0.9409
56736/60000 [===========================>..] - ETA: 5s - loss: 0.1951 - categorical_accuracy: 0.9409
56768/60000 [===========================>..] - ETA: 5s - loss: 0.1950 - categorical_accuracy: 0.9409
56800/60000 [===========================>..] - ETA: 5s - loss: 0.1949 - categorical_accuracy: 0.9409
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1950 - categorical_accuracy: 0.9409
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1949 - categorical_accuracy: 0.9410
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1948 - categorical_accuracy: 0.9410
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1947 - categorical_accuracy: 0.9410
56960/60000 [===========================>..] - ETA: 5s - loss: 0.1947 - categorical_accuracy: 0.9410
56992/60000 [===========================>..] - ETA: 5s - loss: 0.1946 - categorical_accuracy: 0.9410
57024/60000 [===========================>..] - ETA: 5s - loss: 0.1946 - categorical_accuracy: 0.9410
57056/60000 [===========================>..] - ETA: 5s - loss: 0.1945 - categorical_accuracy: 0.9410
57088/60000 [===========================>..] - ETA: 5s - loss: 0.1945 - categorical_accuracy: 0.9410
57120/60000 [===========================>..] - ETA: 5s - loss: 0.1944 - categorical_accuracy: 0.9411
57152/60000 [===========================>..] - ETA: 5s - loss: 0.1944 - categorical_accuracy: 0.9411
57184/60000 [===========================>..] - ETA: 5s - loss: 0.1943 - categorical_accuracy: 0.9411
57216/60000 [===========================>..] - ETA: 5s - loss: 0.1943 - categorical_accuracy: 0.9411
57248/60000 [===========================>..] - ETA: 4s - loss: 0.1943 - categorical_accuracy: 0.9411
57280/60000 [===========================>..] - ETA: 4s - loss: 0.1945 - categorical_accuracy: 0.9411
57312/60000 [===========================>..] - ETA: 4s - loss: 0.1944 - categorical_accuracy: 0.9411
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1944 - categorical_accuracy: 0.9411
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1943 - categorical_accuracy: 0.9412
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1942 - categorical_accuracy: 0.9412
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1941 - categorical_accuracy: 0.9412
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1940 - categorical_accuracy: 0.9413
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1940 - categorical_accuracy: 0.9413
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1939 - categorical_accuracy: 0.9413
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1938 - categorical_accuracy: 0.9413
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1937 - categorical_accuracy: 0.9413
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1937 - categorical_accuracy: 0.9414
57664/60000 [===========================>..] - ETA: 4s - loss: 0.1937 - categorical_accuracy: 0.9413
57696/60000 [===========================>..] - ETA: 4s - loss: 0.1936 - categorical_accuracy: 0.9414
57728/60000 [===========================>..] - ETA: 4s - loss: 0.1935 - categorical_accuracy: 0.9414
57760/60000 [===========================>..] - ETA: 4s - loss: 0.1934 - categorical_accuracy: 0.9414
57792/60000 [===========================>..] - ETA: 3s - loss: 0.1935 - categorical_accuracy: 0.9414
57824/60000 [===========================>..] - ETA: 3s - loss: 0.1934 - categorical_accuracy: 0.9415
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1934 - categorical_accuracy: 0.9415
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1933 - categorical_accuracy: 0.9415
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1932 - categorical_accuracy: 0.9416
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1932 - categorical_accuracy: 0.9416
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1931 - categorical_accuracy: 0.9416
58016/60000 [============================>.] - ETA: 3s - loss: 0.1931 - categorical_accuracy: 0.9416
58048/60000 [============================>.] - ETA: 3s - loss: 0.1930 - categorical_accuracy: 0.9416
58080/60000 [============================>.] - ETA: 3s - loss: 0.1932 - categorical_accuracy: 0.9416
58112/60000 [============================>.] - ETA: 3s - loss: 0.1931 - categorical_accuracy: 0.9416
58144/60000 [============================>.] - ETA: 3s - loss: 0.1931 - categorical_accuracy: 0.9416
58176/60000 [============================>.] - ETA: 3s - loss: 0.1930 - categorical_accuracy: 0.9416
58208/60000 [============================>.] - ETA: 3s - loss: 0.1929 - categorical_accuracy: 0.9417
58240/60000 [============================>.] - ETA: 3s - loss: 0.1929 - categorical_accuracy: 0.9417
58272/60000 [============================>.] - ETA: 3s - loss: 0.1929 - categorical_accuracy: 0.9417
58304/60000 [============================>.] - ETA: 3s - loss: 0.1929 - categorical_accuracy: 0.9417
58336/60000 [============================>.] - ETA: 3s - loss: 0.1928 - categorical_accuracy: 0.9417
58368/60000 [============================>.] - ETA: 2s - loss: 0.1928 - categorical_accuracy: 0.9417
58400/60000 [============================>.] - ETA: 2s - loss: 0.1927 - categorical_accuracy: 0.9417
58432/60000 [============================>.] - ETA: 2s - loss: 0.1926 - categorical_accuracy: 0.9417
58464/60000 [============================>.] - ETA: 2s - loss: 0.1926 - categorical_accuracy: 0.9417
58496/60000 [============================>.] - ETA: 2s - loss: 0.1926 - categorical_accuracy: 0.9417
58528/60000 [============================>.] - ETA: 2s - loss: 0.1925 - categorical_accuracy: 0.9418
58560/60000 [============================>.] - ETA: 2s - loss: 0.1924 - categorical_accuracy: 0.9418
58592/60000 [============================>.] - ETA: 2s - loss: 0.1923 - categorical_accuracy: 0.9418
58624/60000 [============================>.] - ETA: 2s - loss: 0.1922 - categorical_accuracy: 0.9418
58656/60000 [============================>.] - ETA: 2s - loss: 0.1921 - categorical_accuracy: 0.9419
58688/60000 [============================>.] - ETA: 2s - loss: 0.1920 - categorical_accuracy: 0.9419
58720/60000 [============================>.] - ETA: 2s - loss: 0.1921 - categorical_accuracy: 0.9419
58752/60000 [============================>.] - ETA: 2s - loss: 0.1920 - categorical_accuracy: 0.9419
58784/60000 [============================>.] - ETA: 2s - loss: 0.1919 - categorical_accuracy: 0.9419
58816/60000 [============================>.] - ETA: 2s - loss: 0.1919 - categorical_accuracy: 0.9419
58848/60000 [============================>.] - ETA: 2s - loss: 0.1920 - categorical_accuracy: 0.9419
58880/60000 [============================>.] - ETA: 2s - loss: 0.1921 - categorical_accuracy: 0.9419
58912/60000 [============================>.] - ETA: 1s - loss: 0.1921 - categorical_accuracy: 0.9419
58944/60000 [============================>.] - ETA: 1s - loss: 0.1921 - categorical_accuracy: 0.9419
58976/60000 [============================>.] - ETA: 1s - loss: 0.1921 - categorical_accuracy: 0.9419
59008/60000 [============================>.] - ETA: 1s - loss: 0.1920 - categorical_accuracy: 0.9419
59040/60000 [============================>.] - ETA: 1s - loss: 0.1919 - categorical_accuracy: 0.9420
59072/60000 [============================>.] - ETA: 1s - loss: 0.1919 - categorical_accuracy: 0.9420
59104/60000 [============================>.] - ETA: 1s - loss: 0.1918 - categorical_accuracy: 0.9420
59136/60000 [============================>.] - ETA: 1s - loss: 0.1918 - categorical_accuracy: 0.9420
59168/60000 [============================>.] - ETA: 1s - loss: 0.1917 - categorical_accuracy: 0.9420
59200/60000 [============================>.] - ETA: 1s - loss: 0.1916 - categorical_accuracy: 0.9420
59232/60000 [============================>.] - ETA: 1s - loss: 0.1915 - categorical_accuracy: 0.9421
59264/60000 [============================>.] - ETA: 1s - loss: 0.1915 - categorical_accuracy: 0.9421
59296/60000 [============================>.] - ETA: 1s - loss: 0.1915 - categorical_accuracy: 0.9421
59328/60000 [============================>.] - ETA: 1s - loss: 0.1914 - categorical_accuracy: 0.9421
59360/60000 [============================>.] - ETA: 1s - loss: 0.1913 - categorical_accuracy: 0.9421
59392/60000 [============================>.] - ETA: 1s - loss: 0.1914 - categorical_accuracy: 0.9421
59424/60000 [============================>.] - ETA: 1s - loss: 0.1913 - categorical_accuracy: 0.9421
59456/60000 [============================>.] - ETA: 0s - loss: 0.1913 - categorical_accuracy: 0.9421
59488/60000 [============================>.] - ETA: 0s - loss: 0.1912 - categorical_accuracy: 0.9421
59520/60000 [============================>.] - ETA: 0s - loss: 0.1911 - categorical_accuracy: 0.9421
59552/60000 [============================>.] - ETA: 0s - loss: 0.1910 - categorical_accuracy: 0.9422
59584/60000 [============================>.] - ETA: 0s - loss: 0.1910 - categorical_accuracy: 0.9421
59616/60000 [============================>.] - ETA: 0s - loss: 0.1910 - categorical_accuracy: 0.9422
59648/60000 [============================>.] - ETA: 0s - loss: 0.1909 - categorical_accuracy: 0.9422
59680/60000 [============================>.] - ETA: 0s - loss: 0.1910 - categorical_accuracy: 0.9421
59712/60000 [============================>.] - ETA: 0s - loss: 0.1909 - categorical_accuracy: 0.9422
59744/60000 [============================>.] - ETA: 0s - loss: 0.1909 - categorical_accuracy: 0.9422
59776/60000 [============================>.] - ETA: 0s - loss: 0.1909 - categorical_accuracy: 0.9422
59808/60000 [============================>.] - ETA: 0s - loss: 0.1908 - categorical_accuracy: 0.9422
59840/60000 [============================>.] - ETA: 0s - loss: 0.1907 - categorical_accuracy: 0.9422
59872/60000 [============================>.] - ETA: 0s - loss: 0.1906 - categorical_accuracy: 0.9422
59904/60000 [============================>.] - ETA: 0s - loss: 0.1905 - categorical_accuracy: 0.9423
59936/60000 [============================>.] - ETA: 0s - loss: 0.1905 - categorical_accuracy: 0.9423
59968/60000 [============================>.] - ETA: 0s - loss: 0.1905 - categorical_accuracy: 0.9423
60000/60000 [==============================] - 112s 2ms/step - loss: 0.1904 - categorical_accuracy: 0.9423 - val_loss: 0.0509 - val_categorical_accuracy: 0.9828

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 15s
  192/10000 [..............................] - ETA: 5s 
  352/10000 [>.............................] - ETA: 4s
  512/10000 [>.............................] - ETA: 3s
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
 3200/10000 [========>.....................] - ETA: 2s
 3360/10000 [=========>....................] - ETA: 2s
 3520/10000 [=========>....................] - ETA: 2s
 3680/10000 [==========>...................] - ETA: 2s
 3840/10000 [==========>...................] - ETA: 2s
 4000/10000 [===========>..................] - ETA: 2s
 4160/10000 [===========>..................] - ETA: 2s
 4320/10000 [===========>..................] - ETA: 2s
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
 7040/10000 [====================>.........] - ETA: 1s
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
 9248/10000 [==========================>...] - ETA: 0s
 9408/10000 [===========================>..] - ETA: 0s
 9568/10000 [===========================>..] - ETA: 0s
 9696/10000 [============================>.] - ETA: 0s
 9856/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 4s 355us/step
[[1.97237640e-08 2.59302073e-08 1.14851923e-06 ... 9.99988675e-01
  5.24345367e-08 1.05700656e-06]
 [1.99146016e-05 3.73673465e-05 9.99798119e-01 ... 9.06122537e-08
  2.20870352e-05 4.40381420e-09]
 [7.96561835e-07 9.99845624e-01 2.23473653e-05 ... 4.21221921e-05
  7.36078164e-06 1.56002682e-06]
 ...
 [3.91139210e-09 5.03703859e-07 1.01869766e-07 ... 7.78341837e-06
  9.11959376e-07 6.97685846e-06]
 [1.11920217e-05 5.73513574e-08 4.67497472e-08 ... 2.54745856e-07
  1.96265802e-03 6.11750750e-07]
 [9.13767622e-07 4.60112872e-08 4.70315445e-06 ... 3.78928194e-10
  7.53599124e-07 3.13274451e-09]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.050899067131779156, 'accuracy_test:': 0.9828000068664551}

  ('#### Save   #######################################################',) 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/charcnn/result'}

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   c969729..088045d  master     -> origin/master
Updating c969729..088045d
Fast-forward
 error_list/20200515/list_log_benchmark_20200515.md |  184 +--
 .../20200515/list_log_dataloader_20200515.md       |    2 +-
 error_list/20200515/list_log_jupyter_20200515.md   | 1684 ++++++++++----------
 .../20200515/list_log_pullrequest_20200515.md      |    2 +-
 error_list/20200515/list_log_testall_20200515.md   |  435 +++++
 5 files changed, 1377 insertions(+), 930 deletions(-)
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
