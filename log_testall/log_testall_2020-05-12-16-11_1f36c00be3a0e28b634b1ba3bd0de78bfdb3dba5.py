
  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_all 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5

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
[master 3d041ad] ml_store
 1 file changed, 60 insertions(+)
 create mode 100644 log_testall/log_testall_2020-05-12-16-11_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py
To github.com:arita37/mlmodels_store.git
   c9eb565..3d041ad  master -> master





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
From github.com:arita37/mlmodels_store
   3d041ad..4f01e9b  master     -> origin/master
Updating 3d041ad..4f01e9b
Fast-forward
 ...-10_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py | 621 +++++++++++++++++++++
 1 file changed, 621 insertions(+)
 create mode 100644 log_pullrequest/log_pr_2020-05-12-16-10_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py
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
[master 0f84d3b] ml_store
 1 file changed, 53 insertions(+)
To github.com:arita37/mlmodels_store.git
   4f01e9b..0f84d3b  master -> master





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
[master d5862aa] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   0f84d3b..d5862aa  master -> master





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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         12          sequence_mean[0][0]              
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
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         28          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-12 16:12:42.721240: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 16:12:42.726417: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-05-12 16:12:42.726602: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55daa86107a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 16:12:42.726621: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
100/500 [=====>........................] - ETA: 1s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 1ms/sample - loss: 0.2501 - binary_crossentropy: 0.6933 - val_loss: 0.2500 - val_binary_crossentropy: 0.6931

  #### metrics   #################################################### 
{'MSE': 0.2498911207606773}

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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         12          sequence_mean[0][0]              
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
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         28          sparse_feature_2[0][0]           
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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_max[0][0]               
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
Total params: 443
Trainable params: 443
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2592 - binary_crossentropy: 0.8439500/500 [==============================] - 1s 2ms/sample - loss: 0.2518 - binary_crossentropy: 0.7231 - val_loss: 0.2495 - val_binary_crossentropy: 0.6918

  #### metrics   #################################################### 
{'MSE': 0.2504568425307592}

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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_max[0][0]               
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
Total params: 443
Trainable params: 443
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
Total params: 582
Trainable params: 582
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2575 - binary_crossentropy: 0.9670500/500 [==============================] - 1s 2ms/sample - loss: 0.2504 - binary_crossentropy: 0.7449 - val_loss: 0.2560 - val_binary_crossentropy: 0.8109

  #### metrics   #################################################### 
{'MSE': 0.25299610628133723}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
Total params: 582
Trainable params: 582
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
weighted_sequence_layer_9 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
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
Total params: 493
Trainable params: 493
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2547 - binary_crossentropy: 0.7038500/500 [==============================] - 2s 3ms/sample - loss: 0.2746 - binary_crossentropy: 0.7474 - val_loss: 0.2626 - val_binary_crossentropy: 0.7192

  #### metrics   #################################################### 
{'MSE': 0.2617009045036267}

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
weighted_sequence_layer_9 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
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
Total params: 493
Trainable params: 493
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         8           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         2           sequence_max[0][0]               
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
Total params: 148
Trainable params: 148
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2868 - binary_crossentropy: 2.4329500/500 [==============================] - 2s 4ms/sample - loss: 0.3347 - binary_crossentropy: 2.7593 - val_loss: 0.3200 - val_binary_crossentropy: 2.1875

  #### metrics   #################################################### 
{'MSE': 0.33587863929697254}

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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         8           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         2           sequence_max[0][0]               
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
Total params: 148
Trainable params: 148
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-12 16:14:10.794287: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-12 16:14:10.796437: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-12 16:14:10.802643: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-12 16:14:10.813338: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-12 16:14:10.815258: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-12 16:14:10.816933: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-12 16:14:10.818553: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2486 - val_binary_crossentropy: 0.6904
2020-05-12 16:14:12.219457: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-12 16:14:12.221513: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-12 16:14:12.226111: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-12 16:14:12.235329: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-12 16:14:12.237151: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-12 16:14:12.238768: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-12 16:14:12.240344: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24803735782567315}

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
2020-05-12 16:14:38.029288: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-12 16:14:38.030726: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-12 16:14:38.034701: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-12 16:14:38.041508: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-12 16:14:38.042670: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-12 16:14:38.043758: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-12 16:14:38.044736: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2480 - val_binary_crossentropy: 0.6892
2020-05-12 16:14:39.709017: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-12 16:14:39.710381: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-12 16:14:39.713599: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-12 16:14:39.719895: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-12 16:14:39.720998: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-12 16:14:39.722053: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-12 16:14:39.722912: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24721637808570188}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-12 16:15:16.327143: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-12 16:15:16.332446: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-12 16:15:16.348755: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-12 16:15:16.377219: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-12 16:15:16.382014: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-12 16:15:16.386406: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-12 16:15:16.390651: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 6s 6s/sample - loss: 0.1457 - binary_crossentropy: 0.4808 - val_loss: 0.2934 - val_binary_crossentropy: 0.7884
2020-05-12 16:15:18.847563: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-12 16:15:18.852361: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-12 16:15:18.865574: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-12 16:15:18.890595: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-12 16:15:18.894799: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-12 16:15:18.898685: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-12 16:15:18.902446: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2239374157429168}

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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_max[0][0]               
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
Total params: 740
Trainable params: 740
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2968 - binary_crossentropy: 0.7996500/500 [==============================] - 4s 9ms/sample - loss: 0.2863 - binary_crossentropy: 0.7753 - val_loss: 0.2777 - val_binary_crossentropy: 0.7565

  #### metrics   #################################################### 
{'MSE': 0.281181914394414}

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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_max[0][0]               
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
Total params: 740
Trainable params: 740
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 2)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 2)         18          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 2)         8           sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         10          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         18          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         6           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         10          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         2           sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_5[0][0]           
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
Total params: 248
Trainable params: 248
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2820 - binary_crossentropy: 0.7670500/500 [==============================] - 5s 9ms/sample - loss: 0.2855 - binary_crossentropy: 1.0840 - val_loss: 0.2844 - val_binary_crossentropy: 1.0772

  #### metrics   #################################################### 
{'MSE': 0.2819197524199271}

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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 2)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 2)         18          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 2)         8           sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         10          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         18          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         6           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         10          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         2           sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_5[0][0]           
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
Total params: 248
Trainable params: 248
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
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
100/500 [=====>........................] - ETA: 6s - loss: 0.4500 - binary_crossentropy: 6.9377500/500 [==============================] - 5s 10ms/sample - loss: 0.4680 - binary_crossentropy: 7.2115 - val_loss: 0.4760 - val_binary_crossentropy: 7.3423

  #### metrics   #################################################### 
{'MSE': 0.471}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
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
regionsequence_sum (InputLayer) [(None, 4)]          0                                            
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
region_10sparse_seq_emb_regions (None, 4, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 4, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 4, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 4, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 4, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 4, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 4, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 4, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
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
Total params: 96
Trainable params: 96
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 10s - loss: 0.2516 - binary_crossentropy: 0.6962500/500 [==============================] - 6s 13ms/sample - loss: 0.2489 - binary_crossentropy: 0.6907 - val_loss: 0.2516 - val_binary_crossentropy: 0.6962

  #### metrics   #################################################### 
{'MSE': 0.2500807228002633}

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
region_10sparse_seq_emb_regions (None, 4, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 4, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 4, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 4, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 4, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 4, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 4, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 4, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
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
Total params: 96
Trainable params: 96
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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_max[0][0]               
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
Total params: 1,387
Trainable params: 1,387
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2466 - binary_crossentropy: 0.6858500/500 [==============================] - 6s 12ms/sample - loss: 0.2595 - binary_crossentropy: 0.7127 - val_loss: 0.2589 - val_binary_crossentropy: 0.7113

  #### metrics   #################################################### 
{'MSE': 0.25847930948758235}

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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_max[0][0]               
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
Total params: 1,387
Trainable params: 1,387
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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         36          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         12          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 4, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         12          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         12          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 4, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         12          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 4, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 2, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 4, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 2, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           hash_10[0][0]                    
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
100/500 [=====>........................] - ETA: 9s - loss: 0.2552 - binary_crossentropy: 0.9604500/500 [==============================] - 7s 13ms/sample - loss: 0.2704 - binary_crossentropy: 1.2549 - val_loss: 0.2850 - val_binary_crossentropy: 1.5989

  #### metrics   #################################################### 
{'MSE': 0.27785004146775344}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         36          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         12          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 4, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         12          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         12          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 4, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         12          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 4, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 2, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 4, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 2, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           hash_10[0][0]                    
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
Total params: 469
Trainable params: 469
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2447 - binary_crossentropy: 0.6825500/500 [==============================] - 6s 13ms/sample - loss: 0.2479 - binary_crossentropy: 0.6888 - val_loss: 0.2490 - val_binary_crossentropy: 0.6912

  #### metrics   #################################################### 
{'MSE': 0.2491405099724845}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
Total params: 469
Trainable params: 469
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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_44 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         36          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_125 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_194[0][0] 
                                                                 sequence_pooling_layer_195[0][0] 
                                                                 sequence_pooling_layer_196[0][0] 
                                                                 sequence_pooling_layer_197[0][0] 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
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
100/500 [=====>........................] - ETA: 8s - loss: 0.4600 - binary_crossentropy: 7.0603500/500 [==============================] - 7s 13ms/sample - loss: 0.4860 - binary_crossentropy: 7.4645 - val_loss: 0.4860 - val_binary_crossentropy: 7.4965

  #### metrics   #################################################### 
{'MSE': 0.491}

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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_44 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         36          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_125 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_194[0][0] 
                                                                 sequence_pooling_layer_195[0][0] 
                                                                 sequence_pooling_layer_196[0][0] 
                                                                 sequence_pooling_layer_197[0][0] 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_90 (Concatenate)    (None, 1, 20)        0           no_mask_130[0][0]                
                                                                 no_mask_130[1][0]                
                                                                 no_mask_130[2][0]                
                                                                 no_mask_130[3][0]                
                                                                 no_mask_130[4][0]                
__________________________________________________________________________________________________
no_mask_131 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 271
Trainable params: 271
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2617 - binary_crossentropy: 0.7180500/500 [==============================] - 7s 14ms/sample - loss: 0.2795 - binary_crossentropy: 0.7588 - val_loss: 0.2751 - val_binary_crossentropy: 0.7469

  #### metrics   #################################################### 
{'MSE': 0.27347917294088614}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_90 (Concatenate)    (None, 1, 20)        0           no_mask_130[0][0]                
                                                                 no_mask_130[1][0]                
                                                                 no_mask_130[2][0]                
                                                                 no_mask_130[3][0]                
                                                                 no_mask_130[4][0]                
__________________________________________________________________________________________________
no_mask_131 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 271
Trainable params: 271
Non-trainable params: 0
__________________________________________________________________________________________________

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
[master 838610e] ml_store
 1 file changed, 5661 insertions(+)
To github.com:arita37/mlmodels_store.git
   d5862aa..838610e  master -> master





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
[master 2da6415] ml_store
 1 file changed, 50 insertions(+)
To github.com:arita37/mlmodels_store.git
   838610e..2da6415  master -> master





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
[master ef432eb] ml_store
 1 file changed, 46 insertions(+)
To github.com:arita37/mlmodels_store.git
   2da6415..ef432eb  master -> master





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
[master ee69e9d] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   ef432eb..ee69e9d  master -> master





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

2020-05-12 16:28:39.580680: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 16:28:39.586084: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-05-12 16:28:39.586326: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55bd63f12360 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 16:28:39.586344: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 7s - loss: 1.3847
256/354 [====================>.........] - ETA: 3s - loss: 1.2592
354/354 [==============================] - 13s 38ms/step - loss: 1.1936 - val_loss: 2.2015

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
[master e4fa6ea] ml_store
 1 file changed, 149 insertions(+)
To github.com:arita37/mlmodels_store.git
   ee69e9d..e4fa6ea  master -> master





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
[master 7674ff0] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   e4fa6ea..7674ff0  master -> master





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
[master a7166f6] ml_store
 1 file changed, 44 insertions(+)
To github.com:arita37/mlmodels_store.git
   7674ff0..a7166f6  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 44s
   57344/17464789 [..............................] - ETA: 37s
  106496/17464789 [..............................] - ETA: 30s
  245760/17464789 [..............................] - ETA: 17s
  524288/17464789 [..............................] - ETA: 10s
 1081344/17464789 [>.............................] - ETA: 5s 
 2179072/17464789 [==>...........................] - ETA: 3s
 4390912/17464789 [======>.......................] - ETA: 1s
 7438336/17464789 [===========>..................] - ETA: 0s
10436608/17464789 [================>.............] - ETA: 0s
13484032/17464789 [======================>.......] - ETA: 0s
16515072/17464789 [===========================>..] - ETA: 0s
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
2020-05-12 16:29:42.697659: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 16:29:42.701819: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-05-12 16:29:42.701962: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561e297f4b60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 16:29:42.701990: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.2220 - accuracy: 0.5290
 2000/25000 [=>............................] - ETA: 9s - loss: 7.4060 - accuracy: 0.5170 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5951 - accuracy: 0.5047
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5938 - accuracy: 0.5048
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5470 - accuracy: 0.5078
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5465 - accuracy: 0.5078
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5615 - accuracy: 0.5069
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5459 - accuracy: 0.5079
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5661 - accuracy: 0.5066
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6038 - accuracy: 0.5041
11000/25000 [============>.................] - ETA: 4s - loss: 7.6067 - accuracy: 0.5039
12000/25000 [=============>................] - ETA: 3s - loss: 7.6487 - accuracy: 0.5012
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6065 - accuracy: 0.5039
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6075 - accuracy: 0.5039
15000/25000 [=================>............] - ETA: 3s - loss: 7.6308 - accuracy: 0.5023
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6340 - accuracy: 0.5021
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6441 - accuracy: 0.5015
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6649 - accuracy: 0.5001
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6763 - accuracy: 0.4994
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6689 - accuracy: 0.4999
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6695 - accuracy: 0.4998
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6624 - accuracy: 0.5003
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6593 - accuracy: 0.5005
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6577 - accuracy: 0.5006
25000/25000 [==============================] - 9s 357us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7f4675e29a20>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f46922b0a58> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.7740 - accuracy: 0.4930
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6360 - accuracy: 0.5020 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7842 - accuracy: 0.4923
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7970 - accuracy: 0.4915
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7464 - accuracy: 0.4948
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6947 - accuracy: 0.4982
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6907 - accuracy: 0.4984
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7145 - accuracy: 0.4969
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6564 - accuracy: 0.5007
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7264 - accuracy: 0.4961
11000/25000 [============>.................] - ETA: 4s - loss: 7.7294 - accuracy: 0.4959
12000/25000 [=============>................] - ETA: 3s - loss: 7.7165 - accuracy: 0.4967
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7055 - accuracy: 0.4975
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6885 - accuracy: 0.4986
15000/25000 [=================>............] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6791 - accuracy: 0.4992
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6856 - accuracy: 0.4988
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6743 - accuracy: 0.4995
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6577 - accuracy: 0.5006
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6597 - accuracy: 0.5005
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6469 - accuracy: 0.5013
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6701 - accuracy: 0.4998
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6713 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6813 - accuracy: 0.4990
25000/25000 [==============================] - 9s 359us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.6360 - accuracy: 0.5020
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6436 - accuracy: 0.5015 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.5900 - accuracy: 0.5050
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5210 - accuracy: 0.5095
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5041 - accuracy: 0.5106
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.4750 - accuracy: 0.5125
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5242 - accuracy: 0.5093
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5152 - accuracy: 0.5099
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5457 - accuracy: 0.5079
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5516 - accuracy: 0.5075
11000/25000 [============>.................] - ETA: 4s - loss: 7.6109 - accuracy: 0.5036
12000/25000 [=============>................] - ETA: 3s - loss: 7.6015 - accuracy: 0.5042
13000/25000 [==============>...............] - ETA: 3s - loss: 7.5900 - accuracy: 0.5050
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6097 - accuracy: 0.5037
15000/25000 [=================>............] - ETA: 2s - loss: 7.6124 - accuracy: 0.5035
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6254 - accuracy: 0.5027
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6314 - accuracy: 0.5023
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6351 - accuracy: 0.5021
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6319 - accuracy: 0.5023
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6406 - accuracy: 0.5017
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6506 - accuracy: 0.5010
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6548 - accuracy: 0.5008
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6706 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6647 - accuracy: 0.5001
25000/25000 [==============================] - 9s 358us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
[master 2e1316d] ml_store
 1 file changed, 326 insertions(+)
To github.com:arita37/mlmodels_store.git
   a7166f6..2e1316d  master -> master





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
[master 140db21] ml_store
 1 file changed, 125 insertions(+)
To github.com:arita37/mlmodels_store.git
   2e1316d..140db21  master -> master





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
   24576/11490434 [..............................] - ETA: 33s
   57344/11490434 [..............................] - ETA: 28s
  106496/11490434 [..............................] - ETA: 22s
  196608/11490434 [..............................] - ETA: 16s
  385024/11490434 [>.............................] - ETA: 10s
  786432/11490434 [=>............................] - ETA: 5s 
 1556480/11490434 [===>..........................] - ETA: 3s
 3112960/11490434 [=======>......................] - ETA: 1s
 6144000/11490434 [===============>..............] - ETA: 0s
 9158656/11490434 [======================>.......] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:27 - loss: 2.3066 - categorical_accuracy: 0.0625
   64/60000 [..............................] - ETA: 4:35 - loss: 2.2571 - categorical_accuracy: 0.1250
   96/60000 [..............................] - ETA: 3:35 - loss: 2.2282 - categorical_accuracy: 0.1562
  128/60000 [..............................] - ETA: 3:05 - loss: 2.1813 - categorical_accuracy: 0.1953
  160/60000 [..............................] - ETA: 2:50 - loss: 2.1791 - categorical_accuracy: 0.2250
  192/60000 [..............................] - ETA: 2:38 - loss: 2.1607 - categorical_accuracy: 0.2396
  224/60000 [..............................] - ETA: 2:29 - loss: 2.1283 - categorical_accuracy: 0.2634
  256/60000 [..............................] - ETA: 2:22 - loss: 2.0766 - categorical_accuracy: 0.2852
  288/60000 [..............................] - ETA: 2:16 - loss: 2.0475 - categorical_accuracy: 0.3021
  320/60000 [..............................] - ETA: 2:12 - loss: 1.9958 - categorical_accuracy: 0.3219
  352/60000 [..............................] - ETA: 2:09 - loss: 1.9766 - categorical_accuracy: 0.3295
  416/60000 [..............................] - ETA: 2:03 - loss: 1.9046 - categorical_accuracy: 0.3606
  448/60000 [..............................] - ETA: 2:01 - loss: 1.8781 - categorical_accuracy: 0.3661
  480/60000 [..............................] - ETA: 1:59 - loss: 1.8660 - categorical_accuracy: 0.3729
  512/60000 [..............................] - ETA: 1:58 - loss: 1.8438 - categorical_accuracy: 0.3809
  544/60000 [..............................] - ETA: 1:56 - loss: 1.8186 - categorical_accuracy: 0.3860
  576/60000 [..............................] - ETA: 1:55 - loss: 1.7960 - categorical_accuracy: 0.3941
  608/60000 [..............................] - ETA: 1:55 - loss: 1.7737 - categorical_accuracy: 0.4013
  640/60000 [..............................] - ETA: 1:55 - loss: 1.7471 - categorical_accuracy: 0.4172
  672/60000 [..............................] - ETA: 1:53 - loss: 1.7213 - categorical_accuracy: 0.4256
  704/60000 [..............................] - ETA: 1:53 - loss: 1.6885 - categorical_accuracy: 0.4332
  736/60000 [..............................] - ETA: 1:52 - loss: 1.6899 - categorical_accuracy: 0.4334
  768/60000 [..............................] - ETA: 1:51 - loss: 1.6798 - categorical_accuracy: 0.4362
  800/60000 [..............................] - ETA: 1:52 - loss: 1.6540 - categorical_accuracy: 0.4512
  832/60000 [..............................] - ETA: 1:51 - loss: 1.6385 - categorical_accuracy: 0.4567
  864/60000 [..............................] - ETA: 1:51 - loss: 1.6213 - categorical_accuracy: 0.4641
  896/60000 [..............................] - ETA: 1:50 - loss: 1.5900 - categorical_accuracy: 0.4754
  928/60000 [..............................] - ETA: 1:50 - loss: 1.5707 - categorical_accuracy: 0.4817
  960/60000 [..............................] - ETA: 1:49 - loss: 1.5612 - categorical_accuracy: 0.4823
  992/60000 [..............................] - ETA: 1:49 - loss: 1.5489 - categorical_accuracy: 0.4869
 1024/60000 [..............................] - ETA: 1:48 - loss: 1.5243 - categorical_accuracy: 0.4961
 1056/60000 [..............................] - ETA: 1:48 - loss: 1.5049 - categorical_accuracy: 0.5038
 1088/60000 [..............................] - ETA: 1:47 - loss: 1.4803 - categorical_accuracy: 0.5110
 1120/60000 [..............................] - ETA: 1:47 - loss: 1.4619 - categorical_accuracy: 0.5179
 1152/60000 [..............................] - ETA: 1:47 - loss: 1.4383 - categorical_accuracy: 0.5252
 1184/60000 [..............................] - ETA: 1:46 - loss: 1.4319 - categorical_accuracy: 0.5262
 1216/60000 [..............................] - ETA: 1:46 - loss: 1.4233 - categorical_accuracy: 0.5296
 1248/60000 [..............................] - ETA: 1:46 - loss: 1.4129 - categorical_accuracy: 0.5329
 1280/60000 [..............................] - ETA: 1:46 - loss: 1.4007 - categorical_accuracy: 0.5375
 1312/60000 [..............................] - ETA: 1:46 - loss: 1.3877 - categorical_accuracy: 0.5412
 1344/60000 [..............................] - ETA: 1:45 - loss: 1.3777 - categorical_accuracy: 0.5446
 1376/60000 [..............................] - ETA: 1:45 - loss: 1.3649 - categorical_accuracy: 0.5494
 1408/60000 [..............................] - ETA: 1:45 - loss: 1.3578 - categorical_accuracy: 0.5504
 1440/60000 [..............................] - ETA: 1:45 - loss: 1.3496 - categorical_accuracy: 0.5521
 1472/60000 [..............................] - ETA: 1:44 - loss: 1.3410 - categorical_accuracy: 0.5550
 1504/60000 [..............................] - ETA: 1:44 - loss: 1.3219 - categorical_accuracy: 0.5618
 1536/60000 [..............................] - ETA: 1:44 - loss: 1.3102 - categorical_accuracy: 0.5658
 1568/60000 [..............................] - ETA: 1:44 - loss: 1.2974 - categorical_accuracy: 0.5695
 1600/60000 [..............................] - ETA: 1:44 - loss: 1.2843 - categorical_accuracy: 0.5738
 1632/60000 [..............................] - ETA: 1:44 - loss: 1.2783 - categorical_accuracy: 0.5741
 1664/60000 [..............................] - ETA: 1:43 - loss: 1.2622 - categorical_accuracy: 0.5799
 1696/60000 [..............................] - ETA: 1:43 - loss: 1.2471 - categorical_accuracy: 0.5837
 1728/60000 [..............................] - ETA: 1:43 - loss: 1.2367 - categorical_accuracy: 0.5880
 1760/60000 [..............................] - ETA: 1:43 - loss: 1.2281 - categorical_accuracy: 0.5909
 1792/60000 [..............................] - ETA: 1:43 - loss: 1.2236 - categorical_accuracy: 0.5915
 1856/60000 [..............................] - ETA: 1:42 - loss: 1.2129 - categorical_accuracy: 0.5964
 1888/60000 [..............................] - ETA: 1:42 - loss: 1.2018 - categorical_accuracy: 0.6006
 1920/60000 [..............................] - ETA: 1:42 - loss: 1.1858 - categorical_accuracy: 0.6057
 1952/60000 [..............................] - ETA: 1:42 - loss: 1.1753 - categorical_accuracy: 0.6107
 1984/60000 [..............................] - ETA: 1:41 - loss: 1.1656 - categorical_accuracy: 0.6149
 2016/60000 [>.............................] - ETA: 1:41 - loss: 1.1537 - categorical_accuracy: 0.6186
 2048/60000 [>.............................] - ETA: 1:41 - loss: 1.1472 - categorical_accuracy: 0.6216
 2112/60000 [>.............................] - ETA: 1:41 - loss: 1.1382 - categorical_accuracy: 0.6241
 2144/60000 [>.............................] - ETA: 1:40 - loss: 1.1299 - categorical_accuracy: 0.6278
 2176/60000 [>.............................] - ETA: 1:40 - loss: 1.1183 - categorical_accuracy: 0.6314
 2208/60000 [>.............................] - ETA: 1:40 - loss: 1.1125 - categorical_accuracy: 0.6327
 2240/60000 [>.............................] - ETA: 1:40 - loss: 1.1064 - categorical_accuracy: 0.6353
 2272/60000 [>.............................] - ETA: 1:40 - loss: 1.0964 - categorical_accuracy: 0.6391
 2304/60000 [>.............................] - ETA: 1:40 - loss: 1.0883 - categorical_accuracy: 0.6419
 2336/60000 [>.............................] - ETA: 1:40 - loss: 1.0783 - categorical_accuracy: 0.6447
 2368/60000 [>.............................] - ETA: 1:40 - loss: 1.0698 - categorical_accuracy: 0.6482
 2400/60000 [>.............................] - ETA: 1:39 - loss: 1.0707 - categorical_accuracy: 0.6475
 2432/60000 [>.............................] - ETA: 1:39 - loss: 1.0664 - categorical_accuracy: 0.6484
 2464/60000 [>.............................] - ETA: 1:39 - loss: 1.0586 - categorical_accuracy: 0.6506
 2496/60000 [>.............................] - ETA: 1:39 - loss: 1.0502 - categorical_accuracy: 0.6534
 2528/60000 [>.............................] - ETA: 1:39 - loss: 1.0414 - categorical_accuracy: 0.6566
 2560/60000 [>.............................] - ETA: 1:39 - loss: 1.0328 - categorical_accuracy: 0.6598
 2592/60000 [>.............................] - ETA: 1:39 - loss: 1.0257 - categorical_accuracy: 0.6620
 2624/60000 [>.............................] - ETA: 1:39 - loss: 1.0179 - categorical_accuracy: 0.6650
 2656/60000 [>.............................] - ETA: 1:39 - loss: 1.0137 - categorical_accuracy: 0.6664
 2688/60000 [>.............................] - ETA: 1:39 - loss: 1.0100 - categorical_accuracy: 0.6670
 2720/60000 [>.............................] - ETA: 1:38 - loss: 1.0027 - categorical_accuracy: 0.6691
 2752/60000 [>.............................] - ETA: 1:38 - loss: 0.9957 - categorical_accuracy: 0.6711
 2816/60000 [>.............................] - ETA: 1:38 - loss: 0.9861 - categorical_accuracy: 0.6744
 2848/60000 [>.............................] - ETA: 1:38 - loss: 0.9830 - categorical_accuracy: 0.6759
 2880/60000 [>.............................] - ETA: 1:38 - loss: 0.9774 - categorical_accuracy: 0.6774
 2912/60000 [>.............................] - ETA: 1:38 - loss: 0.9711 - categorical_accuracy: 0.6793
 2944/60000 [>.............................] - ETA: 1:38 - loss: 0.9658 - categorical_accuracy: 0.6810
 2976/60000 [>.............................] - ETA: 1:37 - loss: 0.9585 - categorical_accuracy: 0.6838
 3008/60000 [>.............................] - ETA: 1:37 - loss: 0.9536 - categorical_accuracy: 0.6855
 3040/60000 [>.............................] - ETA: 1:37 - loss: 0.9528 - categorical_accuracy: 0.6868
 3072/60000 [>.............................] - ETA: 1:37 - loss: 0.9504 - categorical_accuracy: 0.6885
 3104/60000 [>.............................] - ETA: 1:37 - loss: 0.9445 - categorical_accuracy: 0.6901
 3136/60000 [>.............................] - ETA: 1:37 - loss: 0.9388 - categorical_accuracy: 0.6916
 3168/60000 [>.............................] - ETA: 1:37 - loss: 0.9354 - categorical_accuracy: 0.6932
 3200/60000 [>.............................] - ETA: 1:37 - loss: 0.9309 - categorical_accuracy: 0.6953
 3232/60000 [>.............................] - ETA: 1:36 - loss: 0.9245 - categorical_accuracy: 0.6977
 3264/60000 [>.............................] - ETA: 1:36 - loss: 0.9191 - categorical_accuracy: 0.6991
 3296/60000 [>.............................] - ETA: 1:36 - loss: 0.9147 - categorical_accuracy: 0.7005
 3328/60000 [>.............................] - ETA: 1:36 - loss: 0.9100 - categorical_accuracy: 0.7028
 3360/60000 [>.............................] - ETA: 1:36 - loss: 0.9086 - categorical_accuracy: 0.7033
 3392/60000 [>.............................] - ETA: 1:36 - loss: 0.9065 - categorical_accuracy: 0.7046
 3424/60000 [>.............................] - ETA: 1:36 - loss: 0.9022 - categorical_accuracy: 0.7059
 3456/60000 [>.............................] - ETA: 1:36 - loss: 0.8970 - categorical_accuracy: 0.7078
 3488/60000 [>.............................] - ETA: 1:36 - loss: 0.8911 - categorical_accuracy: 0.7096
 3520/60000 [>.............................] - ETA: 1:35 - loss: 0.8843 - categorical_accuracy: 0.7122
 3552/60000 [>.............................] - ETA: 1:35 - loss: 0.8798 - categorical_accuracy: 0.7137
 3616/60000 [>.............................] - ETA: 1:35 - loss: 0.8685 - categorical_accuracy: 0.7174
 3648/60000 [>.............................] - ETA: 1:35 - loss: 0.8654 - categorical_accuracy: 0.7182
 3680/60000 [>.............................] - ETA: 1:35 - loss: 0.8655 - categorical_accuracy: 0.7190
 3744/60000 [>.............................] - ETA: 1:35 - loss: 0.8587 - categorical_accuracy: 0.7209
 3808/60000 [>.............................] - ETA: 1:35 - loss: 0.8537 - categorical_accuracy: 0.7222
 3840/60000 [>.............................] - ETA: 1:35 - loss: 0.8522 - categorical_accuracy: 0.7219
 3872/60000 [>.............................] - ETA: 1:34 - loss: 0.8476 - categorical_accuracy: 0.7237
 3904/60000 [>.............................] - ETA: 1:34 - loss: 0.8445 - categorical_accuracy: 0.7246
 3936/60000 [>.............................] - ETA: 1:34 - loss: 0.8400 - categorical_accuracy: 0.7261
 3968/60000 [>.............................] - ETA: 1:34 - loss: 0.8358 - categorical_accuracy: 0.7271
 4000/60000 [=>............................] - ETA: 1:34 - loss: 0.8334 - categorical_accuracy: 0.7280
 4032/60000 [=>............................] - ETA: 1:34 - loss: 0.8303 - categorical_accuracy: 0.7292
 4064/60000 [=>............................] - ETA: 1:34 - loss: 0.8268 - categorical_accuracy: 0.7301
 4096/60000 [=>............................] - ETA: 1:34 - loss: 0.8258 - categorical_accuracy: 0.7302
 4128/60000 [=>............................] - ETA: 1:34 - loss: 0.8223 - categorical_accuracy: 0.7321
 4160/60000 [=>............................] - ETA: 1:34 - loss: 0.8187 - categorical_accuracy: 0.7332
 4192/60000 [=>............................] - ETA: 1:34 - loss: 0.8151 - categorical_accuracy: 0.7343
 4224/60000 [=>............................] - ETA: 1:33 - loss: 0.8105 - categorical_accuracy: 0.7356
 4256/60000 [=>............................] - ETA: 1:33 - loss: 0.8079 - categorical_accuracy: 0.7364
 4288/60000 [=>............................] - ETA: 1:33 - loss: 0.8033 - categorical_accuracy: 0.7379
 4320/60000 [=>............................] - ETA: 1:33 - loss: 0.7994 - categorical_accuracy: 0.7391
 4352/60000 [=>............................] - ETA: 1:33 - loss: 0.7963 - categorical_accuracy: 0.7401
 4384/60000 [=>............................] - ETA: 1:33 - loss: 0.7937 - categorical_accuracy: 0.7409
 4416/60000 [=>............................] - ETA: 1:33 - loss: 0.7907 - categorical_accuracy: 0.7414
 4448/60000 [=>............................] - ETA: 1:33 - loss: 0.7900 - categorical_accuracy: 0.7421
 4480/60000 [=>............................] - ETA: 1:33 - loss: 0.7880 - categorical_accuracy: 0.7429
 4512/60000 [=>............................] - ETA: 1:33 - loss: 0.7849 - categorical_accuracy: 0.7440
 4544/60000 [=>............................] - ETA: 1:33 - loss: 0.7815 - categorical_accuracy: 0.7452
 4576/60000 [=>............................] - ETA: 1:33 - loss: 0.7769 - categorical_accuracy: 0.7469
 4608/60000 [=>............................] - ETA: 1:33 - loss: 0.7742 - categorical_accuracy: 0.7476
 4640/60000 [=>............................] - ETA: 1:33 - loss: 0.7722 - categorical_accuracy: 0.7483
 4672/60000 [=>............................] - ETA: 1:33 - loss: 0.7686 - categorical_accuracy: 0.7498
 4704/60000 [=>............................] - ETA: 1:33 - loss: 0.7645 - categorical_accuracy: 0.7511
 4736/60000 [=>............................] - ETA: 1:32 - loss: 0.7609 - categorical_accuracy: 0.7525
 4768/60000 [=>............................] - ETA: 1:32 - loss: 0.7572 - categorical_accuracy: 0.7538
 4800/60000 [=>............................] - ETA: 1:32 - loss: 0.7534 - categorical_accuracy: 0.7550
 4864/60000 [=>............................] - ETA: 1:32 - loss: 0.7466 - categorical_accuracy: 0.7576
 4896/60000 [=>............................] - ETA: 1:32 - loss: 0.7432 - categorical_accuracy: 0.7588
 4960/60000 [=>............................] - ETA: 1:32 - loss: 0.7384 - categorical_accuracy: 0.7599
 4992/60000 [=>............................] - ETA: 1:32 - loss: 0.7349 - categorical_accuracy: 0.7608
 5024/60000 [=>............................] - ETA: 1:32 - loss: 0.7321 - categorical_accuracy: 0.7617
 5056/60000 [=>............................] - ETA: 1:32 - loss: 0.7284 - categorical_accuracy: 0.7633
 5120/60000 [=>............................] - ETA: 1:31 - loss: 0.7223 - categorical_accuracy: 0.7656
 5184/60000 [=>............................] - ETA: 1:31 - loss: 0.7171 - categorical_accuracy: 0.7677
 5216/60000 [=>............................] - ETA: 1:31 - loss: 0.7154 - categorical_accuracy: 0.7682
 5248/60000 [=>............................] - ETA: 1:31 - loss: 0.7141 - categorical_accuracy: 0.7687
 5280/60000 [=>............................] - ETA: 1:31 - loss: 0.7113 - categorical_accuracy: 0.7697
 5344/60000 [=>............................] - ETA: 1:31 - loss: 0.7071 - categorical_accuracy: 0.7710
 5408/60000 [=>............................] - ETA: 1:31 - loss: 0.7033 - categorical_accuracy: 0.7714
 5440/60000 [=>............................] - ETA: 1:31 - loss: 0.7008 - categorical_accuracy: 0.7721
 5472/60000 [=>............................] - ETA: 1:31 - loss: 0.6978 - categorical_accuracy: 0.7730
 5504/60000 [=>............................] - ETA: 1:30 - loss: 0.6951 - categorical_accuracy: 0.7740
 5536/60000 [=>............................] - ETA: 1:30 - loss: 0.6949 - categorical_accuracy: 0.7740
 5568/60000 [=>............................] - ETA: 1:30 - loss: 0.6925 - categorical_accuracy: 0.7750
 5600/60000 [=>............................] - ETA: 1:30 - loss: 0.6895 - categorical_accuracy: 0.7759
 5632/60000 [=>............................] - ETA: 1:30 - loss: 0.6884 - categorical_accuracy: 0.7768
 5664/60000 [=>............................] - ETA: 1:30 - loss: 0.6860 - categorical_accuracy: 0.7775
 5696/60000 [=>............................] - ETA: 1:30 - loss: 0.6835 - categorical_accuracy: 0.7783
 5728/60000 [=>............................] - ETA: 1:30 - loss: 0.6830 - categorical_accuracy: 0.7790
 5760/60000 [=>............................] - ETA: 1:30 - loss: 0.6804 - categorical_accuracy: 0.7799
 5792/60000 [=>............................] - ETA: 1:30 - loss: 0.6788 - categorical_accuracy: 0.7806
 5824/60000 [=>............................] - ETA: 1:30 - loss: 0.6774 - categorical_accuracy: 0.7812
 5856/60000 [=>............................] - ETA: 1:30 - loss: 0.6751 - categorical_accuracy: 0.7819
 5920/60000 [=>............................] - ETA: 1:30 - loss: 0.6702 - categorical_accuracy: 0.7836
 5952/60000 [=>............................] - ETA: 1:30 - loss: 0.6688 - categorical_accuracy: 0.7839
 5984/60000 [=>............................] - ETA: 1:29 - loss: 0.6668 - categorical_accuracy: 0.7849
 6016/60000 [==>...........................] - ETA: 1:29 - loss: 0.6644 - categorical_accuracy: 0.7856
 6048/60000 [==>...........................] - ETA: 1:29 - loss: 0.6619 - categorical_accuracy: 0.7865
 6080/60000 [==>...........................] - ETA: 1:29 - loss: 0.6604 - categorical_accuracy: 0.7873
 6112/60000 [==>...........................] - ETA: 1:29 - loss: 0.6587 - categorical_accuracy: 0.7880
 6144/60000 [==>...........................] - ETA: 1:29 - loss: 0.6565 - categorical_accuracy: 0.7886
 6176/60000 [==>...........................] - ETA: 1:29 - loss: 0.6542 - categorical_accuracy: 0.7893
 6208/60000 [==>...........................] - ETA: 1:29 - loss: 0.6523 - categorical_accuracy: 0.7899
 6240/60000 [==>...........................] - ETA: 1:29 - loss: 0.6498 - categorical_accuracy: 0.7907
 6272/60000 [==>...........................] - ETA: 1:29 - loss: 0.6470 - categorical_accuracy: 0.7916
 6304/60000 [==>...........................] - ETA: 1:29 - loss: 0.6467 - categorical_accuracy: 0.7919
 6336/60000 [==>...........................] - ETA: 1:29 - loss: 0.6455 - categorical_accuracy: 0.7923
 6368/60000 [==>...........................] - ETA: 1:29 - loss: 0.6441 - categorical_accuracy: 0.7927
 6432/60000 [==>...........................] - ETA: 1:29 - loss: 0.6395 - categorical_accuracy: 0.7940
 6464/60000 [==>...........................] - ETA: 1:29 - loss: 0.6378 - categorical_accuracy: 0.7946
 6496/60000 [==>...........................] - ETA: 1:29 - loss: 0.6362 - categorical_accuracy: 0.7951
 6528/60000 [==>...........................] - ETA: 1:29 - loss: 0.6342 - categorical_accuracy: 0.7958
 6560/60000 [==>...........................] - ETA: 1:28 - loss: 0.6313 - categorical_accuracy: 0.7968
 6592/60000 [==>...........................] - ETA: 1:28 - loss: 0.6298 - categorical_accuracy: 0.7973
 6624/60000 [==>...........................] - ETA: 1:28 - loss: 0.6285 - categorical_accuracy: 0.7979
 6656/60000 [==>...........................] - ETA: 1:28 - loss: 0.6262 - categorical_accuracy: 0.7987
 6688/60000 [==>...........................] - ETA: 1:28 - loss: 0.6239 - categorical_accuracy: 0.7995
 6720/60000 [==>...........................] - ETA: 1:28 - loss: 0.6216 - categorical_accuracy: 0.8000
 6752/60000 [==>...........................] - ETA: 1:28 - loss: 0.6206 - categorical_accuracy: 0.8007
 6784/60000 [==>...........................] - ETA: 1:28 - loss: 0.6213 - categorical_accuracy: 0.8007
 6816/60000 [==>...........................] - ETA: 1:28 - loss: 0.6201 - categorical_accuracy: 0.8012
 6848/60000 [==>...........................] - ETA: 1:28 - loss: 0.6186 - categorical_accuracy: 0.8018
 6880/60000 [==>...........................] - ETA: 1:28 - loss: 0.6169 - categorical_accuracy: 0.8025
 6944/60000 [==>...........................] - ETA: 1:28 - loss: 0.6133 - categorical_accuracy: 0.8033
 6976/60000 [==>...........................] - ETA: 1:28 - loss: 0.6118 - categorical_accuracy: 0.8038
 7008/60000 [==>...........................] - ETA: 1:28 - loss: 0.6104 - categorical_accuracy: 0.8042
 7040/60000 [==>...........................] - ETA: 1:28 - loss: 0.6090 - categorical_accuracy: 0.8045
 7072/60000 [==>...........................] - ETA: 1:28 - loss: 0.6081 - categorical_accuracy: 0.8049
 7104/60000 [==>...........................] - ETA: 1:27 - loss: 0.6065 - categorical_accuracy: 0.8053
 7136/60000 [==>...........................] - ETA: 1:27 - loss: 0.6053 - categorical_accuracy: 0.8056
 7168/60000 [==>...........................] - ETA: 1:27 - loss: 0.6030 - categorical_accuracy: 0.8065
 7200/60000 [==>...........................] - ETA: 1:27 - loss: 0.6020 - categorical_accuracy: 0.8068
 7232/60000 [==>...........................] - ETA: 1:27 - loss: 0.6005 - categorical_accuracy: 0.8074
 7296/60000 [==>...........................] - ETA: 1:27 - loss: 0.5968 - categorical_accuracy: 0.8085
 7360/60000 [==>...........................] - ETA: 1:27 - loss: 0.5954 - categorical_accuracy: 0.8091
 7424/60000 [==>...........................] - ETA: 1:27 - loss: 0.5938 - categorical_accuracy: 0.8101
 7456/60000 [==>...........................] - ETA: 1:27 - loss: 0.5930 - categorical_accuracy: 0.8105
 7488/60000 [==>...........................] - ETA: 1:27 - loss: 0.5918 - categorical_accuracy: 0.8108
 7520/60000 [==>...........................] - ETA: 1:27 - loss: 0.5908 - categorical_accuracy: 0.8110
 7552/60000 [==>...........................] - ETA: 1:27 - loss: 0.5887 - categorical_accuracy: 0.8118
 7616/60000 [==>...........................] - ETA: 1:26 - loss: 0.5866 - categorical_accuracy: 0.8125
 7648/60000 [==>...........................] - ETA: 1:26 - loss: 0.5848 - categorical_accuracy: 0.8130
 7680/60000 [==>...........................] - ETA: 1:26 - loss: 0.5825 - categorical_accuracy: 0.8138
 7712/60000 [==>...........................] - ETA: 1:26 - loss: 0.5807 - categorical_accuracy: 0.8144
 7744/60000 [==>...........................] - ETA: 1:26 - loss: 0.5793 - categorical_accuracy: 0.8150
 7808/60000 [==>...........................] - ETA: 1:26 - loss: 0.5768 - categorical_accuracy: 0.8157
 7840/60000 [==>...........................] - ETA: 1:26 - loss: 0.5764 - categorical_accuracy: 0.8161
 7872/60000 [==>...........................] - ETA: 1:26 - loss: 0.5745 - categorical_accuracy: 0.8167
 7904/60000 [==>...........................] - ETA: 1:26 - loss: 0.5724 - categorical_accuracy: 0.8173
 7968/60000 [==>...........................] - ETA: 1:26 - loss: 0.5695 - categorical_accuracy: 0.8185
 8000/60000 [===>..........................] - ETA: 1:26 - loss: 0.5690 - categorical_accuracy: 0.8186
 8064/60000 [===>..........................] - ETA: 1:25 - loss: 0.5661 - categorical_accuracy: 0.8194
 8096/60000 [===>..........................] - ETA: 1:25 - loss: 0.5651 - categorical_accuracy: 0.8199
 8128/60000 [===>..........................] - ETA: 1:25 - loss: 0.5634 - categorical_accuracy: 0.8205
 8160/60000 [===>..........................] - ETA: 1:25 - loss: 0.5620 - categorical_accuracy: 0.8210
 8192/60000 [===>..........................] - ETA: 1:25 - loss: 0.5611 - categorical_accuracy: 0.8213
 8224/60000 [===>..........................] - ETA: 1:25 - loss: 0.5599 - categorical_accuracy: 0.8216
 8288/60000 [===>..........................] - ETA: 1:25 - loss: 0.5584 - categorical_accuracy: 0.8219
 8320/60000 [===>..........................] - ETA: 1:25 - loss: 0.5575 - categorical_accuracy: 0.8224
 8352/60000 [===>..........................] - ETA: 1:25 - loss: 0.5560 - categorical_accuracy: 0.8228
 8384/60000 [===>..........................] - ETA: 1:25 - loss: 0.5547 - categorical_accuracy: 0.8234
 8416/60000 [===>..........................] - ETA: 1:25 - loss: 0.5529 - categorical_accuracy: 0.8240
 8448/60000 [===>..........................] - ETA: 1:25 - loss: 0.5515 - categorical_accuracy: 0.8243
 8480/60000 [===>..........................] - ETA: 1:25 - loss: 0.5498 - categorical_accuracy: 0.8249
 8512/60000 [===>..........................] - ETA: 1:25 - loss: 0.5482 - categorical_accuracy: 0.8254
 8576/60000 [===>..........................] - ETA: 1:25 - loss: 0.5457 - categorical_accuracy: 0.8261
 8608/60000 [===>..........................] - ETA: 1:25 - loss: 0.5457 - categorical_accuracy: 0.8262
 8640/60000 [===>..........................] - ETA: 1:25 - loss: 0.5446 - categorical_accuracy: 0.8266
 8672/60000 [===>..........................] - ETA: 1:24 - loss: 0.5431 - categorical_accuracy: 0.8271
 8704/60000 [===>..........................] - ETA: 1:24 - loss: 0.5417 - categorical_accuracy: 0.8274
 8736/60000 [===>..........................] - ETA: 1:24 - loss: 0.5407 - categorical_accuracy: 0.8276
 8768/60000 [===>..........................] - ETA: 1:24 - loss: 0.5391 - categorical_accuracy: 0.8281
 8800/60000 [===>..........................] - ETA: 1:24 - loss: 0.5385 - categorical_accuracy: 0.8282
 8832/60000 [===>..........................] - ETA: 1:24 - loss: 0.5375 - categorical_accuracy: 0.8286
 8864/60000 [===>..........................] - ETA: 1:24 - loss: 0.5366 - categorical_accuracy: 0.8290
 8896/60000 [===>..........................] - ETA: 1:24 - loss: 0.5356 - categorical_accuracy: 0.8294
 8928/60000 [===>..........................] - ETA: 1:24 - loss: 0.5347 - categorical_accuracy: 0.8296
 8960/60000 [===>..........................] - ETA: 1:24 - loss: 0.5333 - categorical_accuracy: 0.8300
 8992/60000 [===>..........................] - ETA: 1:24 - loss: 0.5319 - categorical_accuracy: 0.8305
 9024/60000 [===>..........................] - ETA: 1:24 - loss: 0.5307 - categorical_accuracy: 0.8309
 9056/60000 [===>..........................] - ETA: 1:24 - loss: 0.5291 - categorical_accuracy: 0.8314
 9088/60000 [===>..........................] - ETA: 1:24 - loss: 0.5282 - categorical_accuracy: 0.8318
 9120/60000 [===>..........................] - ETA: 1:24 - loss: 0.5275 - categorical_accuracy: 0.8319
 9152/60000 [===>..........................] - ETA: 1:24 - loss: 0.5259 - categorical_accuracy: 0.8325
 9184/60000 [===>..........................] - ETA: 1:24 - loss: 0.5251 - categorical_accuracy: 0.8325
 9216/60000 [===>..........................] - ETA: 1:24 - loss: 0.5240 - categorical_accuracy: 0.8328
 9248/60000 [===>..........................] - ETA: 1:23 - loss: 0.5235 - categorical_accuracy: 0.8329
 9280/60000 [===>..........................] - ETA: 1:23 - loss: 0.5220 - categorical_accuracy: 0.8334
 9312/60000 [===>..........................] - ETA: 1:23 - loss: 0.5207 - categorical_accuracy: 0.8339
 9344/60000 [===>..........................] - ETA: 1:23 - loss: 0.5195 - categorical_accuracy: 0.8342
 9376/60000 [===>..........................] - ETA: 1:23 - loss: 0.5179 - categorical_accuracy: 0.8348
 9440/60000 [===>..........................] - ETA: 1:23 - loss: 0.5150 - categorical_accuracy: 0.8357
 9472/60000 [===>..........................] - ETA: 1:23 - loss: 0.5143 - categorical_accuracy: 0.8359
 9504/60000 [===>..........................] - ETA: 1:23 - loss: 0.5131 - categorical_accuracy: 0.8363
 9536/60000 [===>..........................] - ETA: 1:23 - loss: 0.5120 - categorical_accuracy: 0.8365
 9568/60000 [===>..........................] - ETA: 1:23 - loss: 0.5114 - categorical_accuracy: 0.8366
 9600/60000 [===>..........................] - ETA: 1:23 - loss: 0.5105 - categorical_accuracy: 0.8369
 9632/60000 [===>..........................] - ETA: 1:23 - loss: 0.5097 - categorical_accuracy: 0.8371
 9696/60000 [===>..........................] - ETA: 1:23 - loss: 0.5076 - categorical_accuracy: 0.8379
 9728/60000 [===>..........................] - ETA: 1:23 - loss: 0.5063 - categorical_accuracy: 0.8383
 9760/60000 [===>..........................] - ETA: 1:23 - loss: 0.5052 - categorical_accuracy: 0.8386
 9792/60000 [===>..........................] - ETA: 1:22 - loss: 0.5040 - categorical_accuracy: 0.8390
 9824/60000 [===>..........................] - ETA: 1:22 - loss: 0.5026 - categorical_accuracy: 0.8394
 9856/60000 [===>..........................] - ETA: 1:22 - loss: 0.5018 - categorical_accuracy: 0.8398
 9888/60000 [===>..........................] - ETA: 1:22 - loss: 0.5012 - categorical_accuracy: 0.8400
 9920/60000 [===>..........................] - ETA: 1:22 - loss: 0.5003 - categorical_accuracy: 0.8403
 9952/60000 [===>..........................] - ETA: 1:22 - loss: 0.4992 - categorical_accuracy: 0.8407
 9984/60000 [===>..........................] - ETA: 1:22 - loss: 0.4987 - categorical_accuracy: 0.8410
10016/60000 [====>.........................] - ETA: 1:22 - loss: 0.4985 - categorical_accuracy: 0.8414
10048/60000 [====>.........................] - ETA: 1:22 - loss: 0.4978 - categorical_accuracy: 0.8416
10080/60000 [====>.........................] - ETA: 1:22 - loss: 0.4973 - categorical_accuracy: 0.8418
10112/60000 [====>.........................] - ETA: 1:22 - loss: 0.4966 - categorical_accuracy: 0.8420
10144/60000 [====>.........................] - ETA: 1:22 - loss: 0.4956 - categorical_accuracy: 0.8424
10176/60000 [====>.........................] - ETA: 1:22 - loss: 0.4949 - categorical_accuracy: 0.8426
10208/60000 [====>.........................] - ETA: 1:22 - loss: 0.4944 - categorical_accuracy: 0.8428
10240/60000 [====>.........................] - ETA: 1:22 - loss: 0.4932 - categorical_accuracy: 0.8431
10272/60000 [====>.........................] - ETA: 1:22 - loss: 0.4926 - categorical_accuracy: 0.8434
10304/60000 [====>.........................] - ETA: 1:22 - loss: 0.4916 - categorical_accuracy: 0.8437
10336/60000 [====>.........................] - ETA: 1:22 - loss: 0.4913 - categorical_accuracy: 0.8438
10368/60000 [====>.........................] - ETA: 1:21 - loss: 0.4907 - categorical_accuracy: 0.8440
10400/60000 [====>.........................] - ETA: 1:21 - loss: 0.4904 - categorical_accuracy: 0.8442
10432/60000 [====>.........................] - ETA: 1:21 - loss: 0.4895 - categorical_accuracy: 0.8446
10464/60000 [====>.........................] - ETA: 1:21 - loss: 0.4890 - categorical_accuracy: 0.8450
10496/60000 [====>.........................] - ETA: 1:21 - loss: 0.4883 - categorical_accuracy: 0.8453
10528/60000 [====>.........................] - ETA: 1:21 - loss: 0.4884 - categorical_accuracy: 0.8453
10560/60000 [====>.........................] - ETA: 1:21 - loss: 0.4880 - categorical_accuracy: 0.8455
10592/60000 [====>.........................] - ETA: 1:21 - loss: 0.4870 - categorical_accuracy: 0.8457
10624/60000 [====>.........................] - ETA: 1:21 - loss: 0.4860 - categorical_accuracy: 0.8461
10656/60000 [====>.........................] - ETA: 1:21 - loss: 0.4852 - categorical_accuracy: 0.8463
10720/60000 [====>.........................] - ETA: 1:21 - loss: 0.4833 - categorical_accuracy: 0.8469
10752/60000 [====>.........................] - ETA: 1:21 - loss: 0.4824 - categorical_accuracy: 0.8472
10784/60000 [====>.........................] - ETA: 1:21 - loss: 0.4813 - categorical_accuracy: 0.8475
10816/60000 [====>.........................] - ETA: 1:21 - loss: 0.4804 - categorical_accuracy: 0.8477
10848/60000 [====>.........................] - ETA: 1:21 - loss: 0.4792 - categorical_accuracy: 0.8482
10880/60000 [====>.........................] - ETA: 1:21 - loss: 0.4784 - categorical_accuracy: 0.8485
10912/60000 [====>.........................] - ETA: 1:21 - loss: 0.4774 - categorical_accuracy: 0.8488
10944/60000 [====>.........................] - ETA: 1:20 - loss: 0.4771 - categorical_accuracy: 0.8488
10976/60000 [====>.........................] - ETA: 1:20 - loss: 0.4766 - categorical_accuracy: 0.8490
11008/60000 [====>.........................] - ETA: 1:20 - loss: 0.4761 - categorical_accuracy: 0.8493
11040/60000 [====>.........................] - ETA: 1:20 - loss: 0.4754 - categorical_accuracy: 0.8495
11072/60000 [====>.........................] - ETA: 1:20 - loss: 0.4758 - categorical_accuracy: 0.8495
11104/60000 [====>.........................] - ETA: 1:20 - loss: 0.4752 - categorical_accuracy: 0.8497
11168/60000 [====>.........................] - ETA: 1:20 - loss: 0.4735 - categorical_accuracy: 0.8503
11200/60000 [====>.........................] - ETA: 1:20 - loss: 0.4725 - categorical_accuracy: 0.8506
11232/60000 [====>.........................] - ETA: 1:20 - loss: 0.4713 - categorical_accuracy: 0.8511
11264/60000 [====>.........................] - ETA: 1:20 - loss: 0.4704 - categorical_accuracy: 0.8512
11328/60000 [====>.........................] - ETA: 1:20 - loss: 0.4687 - categorical_accuracy: 0.8517
11360/60000 [====>.........................] - ETA: 1:20 - loss: 0.4679 - categorical_accuracy: 0.8519
11392/60000 [====>.........................] - ETA: 1:20 - loss: 0.4674 - categorical_accuracy: 0.8519
11424/60000 [====>.........................] - ETA: 1:20 - loss: 0.4665 - categorical_accuracy: 0.8522
11456/60000 [====>.........................] - ETA: 1:20 - loss: 0.4653 - categorical_accuracy: 0.8526
11488/60000 [====>.........................] - ETA: 1:19 - loss: 0.4655 - categorical_accuracy: 0.8525
11520/60000 [====>.........................] - ETA: 1:19 - loss: 0.4650 - categorical_accuracy: 0.8528
11552/60000 [====>.........................] - ETA: 1:19 - loss: 0.4642 - categorical_accuracy: 0.8531
11584/60000 [====>.........................] - ETA: 1:19 - loss: 0.4631 - categorical_accuracy: 0.8534
11616/60000 [====>.........................] - ETA: 1:19 - loss: 0.4622 - categorical_accuracy: 0.8537
11648/60000 [====>.........................] - ETA: 1:19 - loss: 0.4615 - categorical_accuracy: 0.8540
11680/60000 [====>.........................] - ETA: 1:19 - loss: 0.4608 - categorical_accuracy: 0.8543
11712/60000 [====>.........................] - ETA: 1:19 - loss: 0.4598 - categorical_accuracy: 0.8547
11744/60000 [====>.........................] - ETA: 1:19 - loss: 0.4590 - categorical_accuracy: 0.8550
11776/60000 [====>.........................] - ETA: 1:19 - loss: 0.4581 - categorical_accuracy: 0.8552
11808/60000 [====>.........................] - ETA: 1:19 - loss: 0.4571 - categorical_accuracy: 0.8555
11840/60000 [====>.........................] - ETA: 1:19 - loss: 0.4560 - categorical_accuracy: 0.8559
11872/60000 [====>.........................] - ETA: 1:19 - loss: 0.4555 - categorical_accuracy: 0.8560
11904/60000 [====>.........................] - ETA: 1:19 - loss: 0.4548 - categorical_accuracy: 0.8561
11968/60000 [====>.........................] - ETA: 1:19 - loss: 0.4536 - categorical_accuracy: 0.8565
12000/60000 [=====>........................] - ETA: 1:19 - loss: 0.4537 - categorical_accuracy: 0.8568
12032/60000 [=====>........................] - ETA: 1:19 - loss: 0.4533 - categorical_accuracy: 0.8570
12064/60000 [=====>........................] - ETA: 1:18 - loss: 0.4525 - categorical_accuracy: 0.8571
12096/60000 [=====>........................] - ETA: 1:18 - loss: 0.4519 - categorical_accuracy: 0.8573
12128/60000 [=====>........................] - ETA: 1:18 - loss: 0.4516 - categorical_accuracy: 0.8574
12160/60000 [=====>........................] - ETA: 1:18 - loss: 0.4509 - categorical_accuracy: 0.8576
12224/60000 [=====>........................] - ETA: 1:18 - loss: 0.4502 - categorical_accuracy: 0.8577
12256/60000 [=====>........................] - ETA: 1:18 - loss: 0.4500 - categorical_accuracy: 0.8578
12288/60000 [=====>........................] - ETA: 1:18 - loss: 0.4493 - categorical_accuracy: 0.8581
12320/60000 [=====>........................] - ETA: 1:18 - loss: 0.4492 - categorical_accuracy: 0.8582
12352/60000 [=====>........................] - ETA: 1:18 - loss: 0.4483 - categorical_accuracy: 0.8585
12384/60000 [=====>........................] - ETA: 1:18 - loss: 0.4473 - categorical_accuracy: 0.8589
12448/60000 [=====>........................] - ETA: 1:18 - loss: 0.4461 - categorical_accuracy: 0.8594
12512/60000 [=====>........................] - ETA: 1:18 - loss: 0.4445 - categorical_accuracy: 0.8601
12544/60000 [=====>........................] - ETA: 1:18 - loss: 0.4436 - categorical_accuracy: 0.8604
12608/60000 [=====>........................] - ETA: 1:18 - loss: 0.4435 - categorical_accuracy: 0.8608
12640/60000 [=====>........................] - ETA: 1:17 - loss: 0.4427 - categorical_accuracy: 0.8610
12672/60000 [=====>........................] - ETA: 1:17 - loss: 0.4422 - categorical_accuracy: 0.8611
12704/60000 [=====>........................] - ETA: 1:17 - loss: 0.4417 - categorical_accuracy: 0.8612
12736/60000 [=====>........................] - ETA: 1:17 - loss: 0.4415 - categorical_accuracy: 0.8613
12768/60000 [=====>........................] - ETA: 1:17 - loss: 0.4418 - categorical_accuracy: 0.8611
12800/60000 [=====>........................] - ETA: 1:17 - loss: 0.4413 - categorical_accuracy: 0.8613
12832/60000 [=====>........................] - ETA: 1:17 - loss: 0.4404 - categorical_accuracy: 0.8617
12864/60000 [=====>........................] - ETA: 1:17 - loss: 0.4405 - categorical_accuracy: 0.8618
12896/60000 [=====>........................] - ETA: 1:17 - loss: 0.4402 - categorical_accuracy: 0.8620
12928/60000 [=====>........................] - ETA: 1:17 - loss: 0.4397 - categorical_accuracy: 0.8622
12960/60000 [=====>........................] - ETA: 1:17 - loss: 0.4387 - categorical_accuracy: 0.8625
12992/60000 [=====>........................] - ETA: 1:17 - loss: 0.4380 - categorical_accuracy: 0.8628
13024/60000 [=====>........................] - ETA: 1:17 - loss: 0.4372 - categorical_accuracy: 0.8630
13056/60000 [=====>........................] - ETA: 1:17 - loss: 0.4368 - categorical_accuracy: 0.8631
13088/60000 [=====>........................] - ETA: 1:17 - loss: 0.4364 - categorical_accuracy: 0.8631
13120/60000 [=====>........................] - ETA: 1:17 - loss: 0.4356 - categorical_accuracy: 0.8633
13152/60000 [=====>........................] - ETA: 1:17 - loss: 0.4352 - categorical_accuracy: 0.8634
13184/60000 [=====>........................] - ETA: 1:17 - loss: 0.4343 - categorical_accuracy: 0.8638
13216/60000 [=====>........................] - ETA: 1:17 - loss: 0.4342 - categorical_accuracy: 0.8638
13248/60000 [=====>........................] - ETA: 1:16 - loss: 0.4337 - categorical_accuracy: 0.8641
13280/60000 [=====>........................] - ETA: 1:16 - loss: 0.4332 - categorical_accuracy: 0.8642
13312/60000 [=====>........................] - ETA: 1:16 - loss: 0.4324 - categorical_accuracy: 0.8645
13344/60000 [=====>........................] - ETA: 1:16 - loss: 0.4319 - categorical_accuracy: 0.8647
13376/60000 [=====>........................] - ETA: 1:16 - loss: 0.4311 - categorical_accuracy: 0.8650
13440/60000 [=====>........................] - ETA: 1:16 - loss: 0.4301 - categorical_accuracy: 0.8653
13504/60000 [=====>........................] - ETA: 1:16 - loss: 0.4295 - categorical_accuracy: 0.8656
13536/60000 [=====>........................] - ETA: 1:16 - loss: 0.4287 - categorical_accuracy: 0.8658
13600/60000 [=====>........................] - ETA: 1:16 - loss: 0.4271 - categorical_accuracy: 0.8662
13632/60000 [=====>........................] - ETA: 1:16 - loss: 0.4265 - categorical_accuracy: 0.8665
13696/60000 [=====>........................] - ETA: 1:16 - loss: 0.4253 - categorical_accuracy: 0.8667
13728/60000 [=====>........................] - ETA: 1:16 - loss: 0.4259 - categorical_accuracy: 0.8667
13760/60000 [=====>........................] - ETA: 1:16 - loss: 0.4252 - categorical_accuracy: 0.8669
13792/60000 [=====>........................] - ETA: 1:15 - loss: 0.4244 - categorical_accuracy: 0.8672
13824/60000 [=====>........................] - ETA: 1:15 - loss: 0.4237 - categorical_accuracy: 0.8674
13856/60000 [=====>........................] - ETA: 1:15 - loss: 0.4230 - categorical_accuracy: 0.8676
13888/60000 [=====>........................] - ETA: 1:15 - loss: 0.4225 - categorical_accuracy: 0.8677
13920/60000 [=====>........................] - ETA: 1:15 - loss: 0.4225 - categorical_accuracy: 0.8677
13952/60000 [=====>........................] - ETA: 1:15 - loss: 0.4219 - categorical_accuracy: 0.8678
13984/60000 [=====>........................] - ETA: 1:15 - loss: 0.4214 - categorical_accuracy: 0.8680
14016/60000 [======>.......................] - ETA: 1:15 - loss: 0.4207 - categorical_accuracy: 0.8682
14048/60000 [======>.......................] - ETA: 1:15 - loss: 0.4199 - categorical_accuracy: 0.8685
14080/60000 [======>.......................] - ETA: 1:15 - loss: 0.4196 - categorical_accuracy: 0.8687
14112/60000 [======>.......................] - ETA: 1:15 - loss: 0.4196 - categorical_accuracy: 0.8688
14144/60000 [======>.......................] - ETA: 1:15 - loss: 0.4192 - categorical_accuracy: 0.8689
14176/60000 [======>.......................] - ETA: 1:15 - loss: 0.4186 - categorical_accuracy: 0.8691
14208/60000 [======>.......................] - ETA: 1:15 - loss: 0.4179 - categorical_accuracy: 0.8693
14240/60000 [======>.......................] - ETA: 1:15 - loss: 0.4173 - categorical_accuracy: 0.8695
14272/60000 [======>.......................] - ETA: 1:15 - loss: 0.4165 - categorical_accuracy: 0.8698
14304/60000 [======>.......................] - ETA: 1:15 - loss: 0.4160 - categorical_accuracy: 0.8699
14336/60000 [======>.......................] - ETA: 1:15 - loss: 0.4154 - categorical_accuracy: 0.8700
14368/60000 [======>.......................] - ETA: 1:15 - loss: 0.4155 - categorical_accuracy: 0.8702
14400/60000 [======>.......................] - ETA: 1:14 - loss: 0.4152 - categorical_accuracy: 0.8703
14432/60000 [======>.......................] - ETA: 1:14 - loss: 0.4145 - categorical_accuracy: 0.8705
14464/60000 [======>.......................] - ETA: 1:14 - loss: 0.4137 - categorical_accuracy: 0.8708
14496/60000 [======>.......................] - ETA: 1:14 - loss: 0.4141 - categorical_accuracy: 0.8708
14560/60000 [======>.......................] - ETA: 1:14 - loss: 0.4133 - categorical_accuracy: 0.8711
14592/60000 [======>.......................] - ETA: 1:14 - loss: 0.4128 - categorical_accuracy: 0.8712
14624/60000 [======>.......................] - ETA: 1:14 - loss: 0.4122 - categorical_accuracy: 0.8713
14656/60000 [======>.......................] - ETA: 1:14 - loss: 0.4120 - categorical_accuracy: 0.8715
14720/60000 [======>.......................] - ETA: 1:14 - loss: 0.4117 - categorical_accuracy: 0.8717
14784/60000 [======>.......................] - ETA: 1:14 - loss: 0.4104 - categorical_accuracy: 0.8722
14816/60000 [======>.......................] - ETA: 1:14 - loss: 0.4096 - categorical_accuracy: 0.8724
14848/60000 [======>.......................] - ETA: 1:14 - loss: 0.4098 - categorical_accuracy: 0.8724
14880/60000 [======>.......................] - ETA: 1:14 - loss: 0.4097 - categorical_accuracy: 0.8724
14912/60000 [======>.......................] - ETA: 1:14 - loss: 0.4095 - categorical_accuracy: 0.8725
14944/60000 [======>.......................] - ETA: 1:14 - loss: 0.4087 - categorical_accuracy: 0.8728
14976/60000 [======>.......................] - ETA: 1:14 - loss: 0.4081 - categorical_accuracy: 0.8730
15040/60000 [======>.......................] - ETA: 1:13 - loss: 0.4068 - categorical_accuracy: 0.8735
15072/60000 [======>.......................] - ETA: 1:13 - loss: 0.4062 - categorical_accuracy: 0.8736
15104/60000 [======>.......................] - ETA: 1:13 - loss: 0.4058 - categorical_accuracy: 0.8738
15136/60000 [======>.......................] - ETA: 1:13 - loss: 0.4054 - categorical_accuracy: 0.8739
15168/60000 [======>.......................] - ETA: 1:13 - loss: 0.4047 - categorical_accuracy: 0.8742
15200/60000 [======>.......................] - ETA: 1:13 - loss: 0.4040 - categorical_accuracy: 0.8745
15232/60000 [======>.......................] - ETA: 1:13 - loss: 0.4033 - categorical_accuracy: 0.8747
15296/60000 [======>.......................] - ETA: 1:13 - loss: 0.4029 - categorical_accuracy: 0.8749
15328/60000 [======>.......................] - ETA: 1:13 - loss: 0.4024 - categorical_accuracy: 0.8751
15360/60000 [======>.......................] - ETA: 1:13 - loss: 0.4019 - categorical_accuracy: 0.8753
15392/60000 [======>.......................] - ETA: 1:13 - loss: 0.4017 - categorical_accuracy: 0.8753
15424/60000 [======>.......................] - ETA: 1:13 - loss: 0.4014 - categorical_accuracy: 0.8753
15456/60000 [======>.......................] - ETA: 1:13 - loss: 0.4007 - categorical_accuracy: 0.8755
15488/60000 [======>.......................] - ETA: 1:13 - loss: 0.4000 - categorical_accuracy: 0.8757
15520/60000 [======>.......................] - ETA: 1:13 - loss: 0.3996 - categorical_accuracy: 0.8758
15552/60000 [======>.......................] - ETA: 1:13 - loss: 0.3994 - categorical_accuracy: 0.8760
15584/60000 [======>.......................] - ETA: 1:13 - loss: 0.3989 - categorical_accuracy: 0.8760
15616/60000 [======>.......................] - ETA: 1:12 - loss: 0.3988 - categorical_accuracy: 0.8761
15648/60000 [======>.......................] - ETA: 1:12 - loss: 0.3982 - categorical_accuracy: 0.8763
15712/60000 [======>.......................] - ETA: 1:12 - loss: 0.3971 - categorical_accuracy: 0.8766
15744/60000 [======>.......................] - ETA: 1:12 - loss: 0.3966 - categorical_accuracy: 0.8767
15776/60000 [======>.......................] - ETA: 1:12 - loss: 0.3963 - categorical_accuracy: 0.8768
15808/60000 [======>.......................] - ETA: 1:12 - loss: 0.3960 - categorical_accuracy: 0.8768
15840/60000 [======>.......................] - ETA: 1:12 - loss: 0.3957 - categorical_accuracy: 0.8769
15872/60000 [======>.......................] - ETA: 1:12 - loss: 0.3953 - categorical_accuracy: 0.8770
15904/60000 [======>.......................] - ETA: 1:12 - loss: 0.3949 - categorical_accuracy: 0.8772
15936/60000 [======>.......................] - ETA: 1:12 - loss: 0.3941 - categorical_accuracy: 0.8774
16000/60000 [=======>......................] - ETA: 1:12 - loss: 0.3928 - categorical_accuracy: 0.8779
16032/60000 [=======>......................] - ETA: 1:12 - loss: 0.3927 - categorical_accuracy: 0.8781
16064/60000 [=======>......................] - ETA: 1:12 - loss: 0.3921 - categorical_accuracy: 0.8783
16096/60000 [=======>......................] - ETA: 1:12 - loss: 0.3917 - categorical_accuracy: 0.8783
16128/60000 [=======>......................] - ETA: 1:12 - loss: 0.3912 - categorical_accuracy: 0.8785
16160/60000 [=======>......................] - ETA: 1:12 - loss: 0.3910 - categorical_accuracy: 0.8786
16192/60000 [=======>......................] - ETA: 1:12 - loss: 0.3904 - categorical_accuracy: 0.8788
16224/60000 [=======>......................] - ETA: 1:11 - loss: 0.3901 - categorical_accuracy: 0.8789
16256/60000 [=======>......................] - ETA: 1:11 - loss: 0.3900 - categorical_accuracy: 0.8789
16288/60000 [=======>......................] - ETA: 1:11 - loss: 0.3899 - categorical_accuracy: 0.8789
16352/60000 [=======>......................] - ETA: 1:11 - loss: 0.3889 - categorical_accuracy: 0.8792
16384/60000 [=======>......................] - ETA: 1:11 - loss: 0.3882 - categorical_accuracy: 0.8794
16416/60000 [=======>......................] - ETA: 1:11 - loss: 0.3878 - categorical_accuracy: 0.8796
16448/60000 [=======>......................] - ETA: 1:11 - loss: 0.3871 - categorical_accuracy: 0.8798
16480/60000 [=======>......................] - ETA: 1:11 - loss: 0.3871 - categorical_accuracy: 0.8799
16544/60000 [=======>......................] - ETA: 1:11 - loss: 0.3866 - categorical_accuracy: 0.8800
16608/60000 [=======>......................] - ETA: 1:11 - loss: 0.3853 - categorical_accuracy: 0.8805
16672/60000 [=======>......................] - ETA: 1:11 - loss: 0.3842 - categorical_accuracy: 0.8808
16736/60000 [=======>......................] - ETA: 1:11 - loss: 0.3834 - categorical_accuracy: 0.8811
16800/60000 [=======>......................] - ETA: 1:10 - loss: 0.3827 - categorical_accuracy: 0.8814
16864/60000 [=======>......................] - ETA: 1:10 - loss: 0.3819 - categorical_accuracy: 0.8816
16896/60000 [=======>......................] - ETA: 1:10 - loss: 0.3815 - categorical_accuracy: 0.8817
16928/60000 [=======>......................] - ETA: 1:10 - loss: 0.3810 - categorical_accuracy: 0.8819
16960/60000 [=======>......................] - ETA: 1:10 - loss: 0.3804 - categorical_accuracy: 0.8821
16992/60000 [=======>......................] - ETA: 1:10 - loss: 0.3803 - categorical_accuracy: 0.8821
17024/60000 [=======>......................] - ETA: 1:10 - loss: 0.3799 - categorical_accuracy: 0.8822
17088/60000 [=======>......................] - ETA: 1:10 - loss: 0.3786 - categorical_accuracy: 0.8826
17120/60000 [=======>......................] - ETA: 1:10 - loss: 0.3785 - categorical_accuracy: 0.8828
17152/60000 [=======>......................] - ETA: 1:10 - loss: 0.3779 - categorical_accuracy: 0.8830
17184/60000 [=======>......................] - ETA: 1:10 - loss: 0.3777 - categorical_accuracy: 0.8829
17216/60000 [=======>......................] - ETA: 1:10 - loss: 0.3772 - categorical_accuracy: 0.8831
17280/60000 [=======>......................] - ETA: 1:10 - loss: 0.3761 - categorical_accuracy: 0.8834
17312/60000 [=======>......................] - ETA: 1:09 - loss: 0.3756 - categorical_accuracy: 0.8836
17344/60000 [=======>......................] - ETA: 1:09 - loss: 0.3750 - categorical_accuracy: 0.8838
17408/60000 [=======>......................] - ETA: 1:09 - loss: 0.3738 - categorical_accuracy: 0.8842
17440/60000 [=======>......................] - ETA: 1:09 - loss: 0.3738 - categorical_accuracy: 0.8843
17472/60000 [=======>......................] - ETA: 1:09 - loss: 0.3745 - categorical_accuracy: 0.8843
17536/60000 [=======>......................] - ETA: 1:09 - loss: 0.3737 - categorical_accuracy: 0.8845
17600/60000 [=======>......................] - ETA: 1:09 - loss: 0.3729 - categorical_accuracy: 0.8847
17664/60000 [=======>......................] - ETA: 1:09 - loss: 0.3719 - categorical_accuracy: 0.8850
17728/60000 [=======>......................] - ETA: 1:09 - loss: 0.3714 - categorical_accuracy: 0.8852
17760/60000 [=======>......................] - ETA: 1:09 - loss: 0.3710 - categorical_accuracy: 0.8853
17792/60000 [=======>......................] - ETA: 1:09 - loss: 0.3707 - categorical_accuracy: 0.8853
17856/60000 [=======>......................] - ETA: 1:09 - loss: 0.3699 - categorical_accuracy: 0.8856
17920/60000 [=======>......................] - ETA: 1:08 - loss: 0.3687 - categorical_accuracy: 0.8860
17952/60000 [=======>......................] - ETA: 1:08 - loss: 0.3681 - categorical_accuracy: 0.8862
17984/60000 [=======>......................] - ETA: 1:08 - loss: 0.3676 - categorical_accuracy: 0.8864
18048/60000 [========>.....................] - ETA: 1:08 - loss: 0.3667 - categorical_accuracy: 0.8867
18080/60000 [========>.....................] - ETA: 1:08 - loss: 0.3666 - categorical_accuracy: 0.8867
18112/60000 [========>.....................] - ETA: 1:08 - loss: 0.3661 - categorical_accuracy: 0.8869
18176/60000 [========>.....................] - ETA: 1:08 - loss: 0.3661 - categorical_accuracy: 0.8869
18240/60000 [========>.....................] - ETA: 1:08 - loss: 0.3652 - categorical_accuracy: 0.8872
18272/60000 [========>.....................] - ETA: 1:08 - loss: 0.3648 - categorical_accuracy: 0.8873
18304/60000 [========>.....................] - ETA: 1:08 - loss: 0.3645 - categorical_accuracy: 0.8873
18336/60000 [========>.....................] - ETA: 1:08 - loss: 0.3646 - categorical_accuracy: 0.8874
18368/60000 [========>.....................] - ETA: 1:08 - loss: 0.3643 - categorical_accuracy: 0.8875
18400/60000 [========>.....................] - ETA: 1:08 - loss: 0.3639 - categorical_accuracy: 0.8876
18464/60000 [========>.....................] - ETA: 1:07 - loss: 0.3633 - categorical_accuracy: 0.8878
18496/60000 [========>.....................] - ETA: 1:07 - loss: 0.3630 - categorical_accuracy: 0.8879
18560/60000 [========>.....................] - ETA: 1:07 - loss: 0.3628 - categorical_accuracy: 0.8879
18624/60000 [========>.....................] - ETA: 1:07 - loss: 0.3619 - categorical_accuracy: 0.8883
18656/60000 [========>.....................] - ETA: 1:07 - loss: 0.3616 - categorical_accuracy: 0.8883
18688/60000 [========>.....................] - ETA: 1:07 - loss: 0.3613 - categorical_accuracy: 0.8884
18720/60000 [========>.....................] - ETA: 1:07 - loss: 0.3609 - categorical_accuracy: 0.8886
18752/60000 [========>.....................] - ETA: 1:07 - loss: 0.3605 - categorical_accuracy: 0.8887
18784/60000 [========>.....................] - ETA: 1:07 - loss: 0.3605 - categorical_accuracy: 0.8887
18816/60000 [========>.....................] - ETA: 1:07 - loss: 0.3600 - categorical_accuracy: 0.8889
18848/60000 [========>.....................] - ETA: 1:07 - loss: 0.3597 - categorical_accuracy: 0.8890
18912/60000 [========>.....................] - ETA: 1:07 - loss: 0.3591 - categorical_accuracy: 0.8893
18944/60000 [========>.....................] - ETA: 1:07 - loss: 0.3586 - categorical_accuracy: 0.8895
18976/60000 [========>.....................] - ETA: 1:07 - loss: 0.3581 - categorical_accuracy: 0.8897
19008/60000 [========>.....................] - ETA: 1:07 - loss: 0.3577 - categorical_accuracy: 0.8897
19040/60000 [========>.....................] - ETA: 1:06 - loss: 0.3573 - categorical_accuracy: 0.8898
19104/60000 [========>.....................] - ETA: 1:06 - loss: 0.3567 - categorical_accuracy: 0.8900
19168/60000 [========>.....................] - ETA: 1:06 - loss: 0.3568 - categorical_accuracy: 0.8900
19232/60000 [========>.....................] - ETA: 1:06 - loss: 0.3558 - categorical_accuracy: 0.8903
19264/60000 [========>.....................] - ETA: 1:06 - loss: 0.3553 - categorical_accuracy: 0.8904
19328/60000 [========>.....................] - ETA: 1:06 - loss: 0.3549 - categorical_accuracy: 0.8905
19360/60000 [========>.....................] - ETA: 1:06 - loss: 0.3547 - categorical_accuracy: 0.8905
19392/60000 [========>.....................] - ETA: 1:06 - loss: 0.3541 - categorical_accuracy: 0.8907
19424/60000 [========>.....................] - ETA: 1:06 - loss: 0.3546 - categorical_accuracy: 0.8905
19456/60000 [========>.....................] - ETA: 1:06 - loss: 0.3542 - categorical_accuracy: 0.8906
19488/60000 [========>.....................] - ETA: 1:06 - loss: 0.3537 - categorical_accuracy: 0.8907
19552/60000 [========>.....................] - ETA: 1:06 - loss: 0.3529 - categorical_accuracy: 0.8909
19584/60000 [========>.....................] - ETA: 1:06 - loss: 0.3528 - categorical_accuracy: 0.8909
19648/60000 [========>.....................] - ETA: 1:05 - loss: 0.3524 - categorical_accuracy: 0.8910
19712/60000 [========>.....................] - ETA: 1:05 - loss: 0.3514 - categorical_accuracy: 0.8914
19776/60000 [========>.....................] - ETA: 1:05 - loss: 0.3510 - categorical_accuracy: 0.8914
19840/60000 [========>.....................] - ETA: 1:05 - loss: 0.3501 - categorical_accuracy: 0.8916
19872/60000 [========>.....................] - ETA: 1:05 - loss: 0.3501 - categorical_accuracy: 0.8916
19904/60000 [========>.....................] - ETA: 1:05 - loss: 0.3498 - categorical_accuracy: 0.8917
19936/60000 [========>.....................] - ETA: 1:05 - loss: 0.3496 - categorical_accuracy: 0.8918
19968/60000 [========>.....................] - ETA: 1:05 - loss: 0.3493 - categorical_accuracy: 0.8918
20000/60000 [=========>....................] - ETA: 1:05 - loss: 0.3488 - categorical_accuracy: 0.8920
20032/60000 [=========>....................] - ETA: 1:05 - loss: 0.3483 - categorical_accuracy: 0.8921
20064/60000 [=========>....................] - ETA: 1:05 - loss: 0.3478 - categorical_accuracy: 0.8923
20096/60000 [=========>....................] - ETA: 1:05 - loss: 0.3476 - categorical_accuracy: 0.8924
20160/60000 [=========>....................] - ETA: 1:05 - loss: 0.3469 - categorical_accuracy: 0.8927
20224/60000 [=========>....................] - ETA: 1:04 - loss: 0.3463 - categorical_accuracy: 0.8928
20256/60000 [=========>....................] - ETA: 1:04 - loss: 0.3459 - categorical_accuracy: 0.8930
20288/60000 [=========>....................] - ETA: 1:04 - loss: 0.3458 - categorical_accuracy: 0.8930
20320/60000 [=========>....................] - ETA: 1:04 - loss: 0.3455 - categorical_accuracy: 0.8931
20352/60000 [=========>....................] - ETA: 1:04 - loss: 0.3450 - categorical_accuracy: 0.8933
20416/60000 [=========>....................] - ETA: 1:04 - loss: 0.3445 - categorical_accuracy: 0.8934
20480/60000 [=========>....................] - ETA: 1:04 - loss: 0.3438 - categorical_accuracy: 0.8937
20544/60000 [=========>....................] - ETA: 1:04 - loss: 0.3430 - categorical_accuracy: 0.8939
20608/60000 [=========>....................] - ETA: 1:04 - loss: 0.3421 - categorical_accuracy: 0.8942
20672/60000 [=========>....................] - ETA: 1:04 - loss: 0.3412 - categorical_accuracy: 0.8945
20704/60000 [=========>....................] - ETA: 1:04 - loss: 0.3415 - categorical_accuracy: 0.8946
20736/60000 [=========>....................] - ETA: 1:04 - loss: 0.3411 - categorical_accuracy: 0.8947
20800/60000 [=========>....................] - ETA: 1:03 - loss: 0.3404 - categorical_accuracy: 0.8950
20864/60000 [=========>....................] - ETA: 1:03 - loss: 0.3397 - categorical_accuracy: 0.8951
20896/60000 [=========>....................] - ETA: 1:03 - loss: 0.3392 - categorical_accuracy: 0.8953
20928/60000 [=========>....................] - ETA: 1:03 - loss: 0.3391 - categorical_accuracy: 0.8954
20992/60000 [=========>....................] - ETA: 1:03 - loss: 0.3386 - categorical_accuracy: 0.8956
21024/60000 [=========>....................] - ETA: 1:03 - loss: 0.3385 - categorical_accuracy: 0.8956
21056/60000 [=========>....................] - ETA: 1:03 - loss: 0.3385 - categorical_accuracy: 0.8957
21088/60000 [=========>....................] - ETA: 1:03 - loss: 0.3387 - categorical_accuracy: 0.8956
21120/60000 [=========>....................] - ETA: 1:03 - loss: 0.3383 - categorical_accuracy: 0.8957
21152/60000 [=========>....................] - ETA: 1:03 - loss: 0.3380 - categorical_accuracy: 0.8958
21216/60000 [=========>....................] - ETA: 1:03 - loss: 0.3376 - categorical_accuracy: 0.8959
21280/60000 [=========>....................] - ETA: 1:03 - loss: 0.3370 - categorical_accuracy: 0.8961
21312/60000 [=========>....................] - ETA: 1:03 - loss: 0.3366 - categorical_accuracy: 0.8962
21376/60000 [=========>....................] - ETA: 1:02 - loss: 0.3363 - categorical_accuracy: 0.8964
21440/60000 [=========>....................] - ETA: 1:02 - loss: 0.3356 - categorical_accuracy: 0.8966
21472/60000 [=========>....................] - ETA: 1:02 - loss: 0.3353 - categorical_accuracy: 0.8966
21536/60000 [=========>....................] - ETA: 1:02 - loss: 0.3349 - categorical_accuracy: 0.8967
21600/60000 [=========>....................] - ETA: 1:02 - loss: 0.3345 - categorical_accuracy: 0.8968
21632/60000 [=========>....................] - ETA: 1:02 - loss: 0.3340 - categorical_accuracy: 0.8970
21664/60000 [=========>....................] - ETA: 1:02 - loss: 0.3337 - categorical_accuracy: 0.8971
21728/60000 [=========>....................] - ETA: 1:02 - loss: 0.3331 - categorical_accuracy: 0.8972
21760/60000 [=========>....................] - ETA: 1:02 - loss: 0.3327 - categorical_accuracy: 0.8974
21792/60000 [=========>....................] - ETA: 1:02 - loss: 0.3324 - categorical_accuracy: 0.8975
21824/60000 [=========>....................] - ETA: 1:02 - loss: 0.3320 - categorical_accuracy: 0.8976
21856/60000 [=========>....................] - ETA: 1:02 - loss: 0.3319 - categorical_accuracy: 0.8976
21888/60000 [=========>....................] - ETA: 1:02 - loss: 0.3319 - categorical_accuracy: 0.8977
21920/60000 [=========>....................] - ETA: 1:02 - loss: 0.3316 - categorical_accuracy: 0.8978
21952/60000 [=========>....................] - ETA: 1:01 - loss: 0.3313 - categorical_accuracy: 0.8979
21984/60000 [=========>....................] - ETA: 1:01 - loss: 0.3310 - categorical_accuracy: 0.8980
22016/60000 [==========>...................] - ETA: 1:01 - loss: 0.3305 - categorical_accuracy: 0.8982
22048/60000 [==========>...................] - ETA: 1:01 - loss: 0.3302 - categorical_accuracy: 0.8982
22112/60000 [==========>...................] - ETA: 1:01 - loss: 0.3301 - categorical_accuracy: 0.8983
22176/60000 [==========>...................] - ETA: 1:01 - loss: 0.3293 - categorical_accuracy: 0.8985
22240/60000 [==========>...................] - ETA: 1:01 - loss: 0.3287 - categorical_accuracy: 0.8987
22272/60000 [==========>...................] - ETA: 1:01 - loss: 0.3283 - categorical_accuracy: 0.8988
22336/60000 [==========>...................] - ETA: 1:01 - loss: 0.3281 - categorical_accuracy: 0.8989
22400/60000 [==========>...................] - ETA: 1:01 - loss: 0.3276 - categorical_accuracy: 0.8990
22464/60000 [==========>...................] - ETA: 1:01 - loss: 0.3270 - categorical_accuracy: 0.8992
22496/60000 [==========>...................] - ETA: 1:01 - loss: 0.3268 - categorical_accuracy: 0.8992
22560/60000 [==========>...................] - ETA: 1:00 - loss: 0.3261 - categorical_accuracy: 0.8995
22592/60000 [==========>...................] - ETA: 1:00 - loss: 0.3261 - categorical_accuracy: 0.8995
22624/60000 [==========>...................] - ETA: 1:00 - loss: 0.3260 - categorical_accuracy: 0.8995
22688/60000 [==========>...................] - ETA: 1:00 - loss: 0.3253 - categorical_accuracy: 0.8997
22720/60000 [==========>...................] - ETA: 1:00 - loss: 0.3249 - categorical_accuracy: 0.8999
22752/60000 [==========>...................] - ETA: 1:00 - loss: 0.3246 - categorical_accuracy: 0.9000
22784/60000 [==========>...................] - ETA: 1:00 - loss: 0.3242 - categorical_accuracy: 0.9001
22816/60000 [==========>...................] - ETA: 1:00 - loss: 0.3240 - categorical_accuracy: 0.9002
22880/60000 [==========>...................] - ETA: 1:00 - loss: 0.3233 - categorical_accuracy: 0.9003
22912/60000 [==========>...................] - ETA: 1:00 - loss: 0.3230 - categorical_accuracy: 0.9004
22976/60000 [==========>...................] - ETA: 1:00 - loss: 0.3227 - categorical_accuracy: 0.9005
23008/60000 [==========>...................] - ETA: 1:00 - loss: 0.3226 - categorical_accuracy: 0.9006
23072/60000 [==========>...................] - ETA: 1:00 - loss: 0.3219 - categorical_accuracy: 0.9007
23104/60000 [==========>...................] - ETA: 1:00 - loss: 0.3216 - categorical_accuracy: 0.9008
23136/60000 [==========>...................] - ETA: 59s - loss: 0.3215 - categorical_accuracy: 0.9009 
23168/60000 [==========>...................] - ETA: 59s - loss: 0.3212 - categorical_accuracy: 0.9010
23200/60000 [==========>...................] - ETA: 59s - loss: 0.3209 - categorical_accuracy: 0.9011
23232/60000 [==========>...................] - ETA: 59s - loss: 0.3205 - categorical_accuracy: 0.9012
23264/60000 [==========>...................] - ETA: 59s - loss: 0.3201 - categorical_accuracy: 0.9013
23328/60000 [==========>...................] - ETA: 59s - loss: 0.3197 - categorical_accuracy: 0.9014
23392/60000 [==========>...................] - ETA: 59s - loss: 0.3192 - categorical_accuracy: 0.9016
23456/60000 [==========>...................] - ETA: 59s - loss: 0.3187 - categorical_accuracy: 0.9017
23520/60000 [==========>...................] - ETA: 59s - loss: 0.3182 - categorical_accuracy: 0.9019
23552/60000 [==========>...................] - ETA: 59s - loss: 0.3178 - categorical_accuracy: 0.9020
23584/60000 [==========>...................] - ETA: 59s - loss: 0.3177 - categorical_accuracy: 0.9021
23648/60000 [==========>...................] - ETA: 59s - loss: 0.3183 - categorical_accuracy: 0.9021
23680/60000 [==========>...................] - ETA: 59s - loss: 0.3181 - categorical_accuracy: 0.9022
23744/60000 [==========>...................] - ETA: 58s - loss: 0.3178 - categorical_accuracy: 0.9023
23776/60000 [==========>...................] - ETA: 58s - loss: 0.3175 - categorical_accuracy: 0.9024
23808/60000 [==========>...................] - ETA: 58s - loss: 0.3171 - categorical_accuracy: 0.9025
23840/60000 [==========>...................] - ETA: 58s - loss: 0.3167 - categorical_accuracy: 0.9026
23872/60000 [==========>...................] - ETA: 58s - loss: 0.3165 - categorical_accuracy: 0.9027
23936/60000 [==========>...................] - ETA: 58s - loss: 0.3160 - categorical_accuracy: 0.9028
24000/60000 [===========>..................] - ETA: 58s - loss: 0.3155 - categorical_accuracy: 0.9029
24064/60000 [===========>..................] - ETA: 58s - loss: 0.3150 - categorical_accuracy: 0.9031
24096/60000 [===========>..................] - ETA: 58s - loss: 0.3147 - categorical_accuracy: 0.9032
24160/60000 [===========>..................] - ETA: 58s - loss: 0.3140 - categorical_accuracy: 0.9034
24192/60000 [===========>..................] - ETA: 58s - loss: 0.3140 - categorical_accuracy: 0.9034
24224/60000 [===========>..................] - ETA: 58s - loss: 0.3137 - categorical_accuracy: 0.9035
24256/60000 [===========>..................] - ETA: 58s - loss: 0.3134 - categorical_accuracy: 0.9036
24288/60000 [===========>..................] - ETA: 58s - loss: 0.3134 - categorical_accuracy: 0.9036
24320/60000 [===========>..................] - ETA: 57s - loss: 0.3131 - categorical_accuracy: 0.9036
24352/60000 [===========>..................] - ETA: 57s - loss: 0.3128 - categorical_accuracy: 0.9037
24416/60000 [===========>..................] - ETA: 57s - loss: 0.3123 - categorical_accuracy: 0.9039
24480/60000 [===========>..................] - ETA: 57s - loss: 0.3119 - categorical_accuracy: 0.9041
24512/60000 [===========>..................] - ETA: 57s - loss: 0.3118 - categorical_accuracy: 0.9041
24544/60000 [===========>..................] - ETA: 57s - loss: 0.3117 - categorical_accuracy: 0.9042
24576/60000 [===========>..................] - ETA: 57s - loss: 0.3114 - categorical_accuracy: 0.9043
24640/60000 [===========>..................] - ETA: 57s - loss: 0.3111 - categorical_accuracy: 0.9043
24704/60000 [===========>..................] - ETA: 57s - loss: 0.3107 - categorical_accuracy: 0.9045
24768/60000 [===========>..................] - ETA: 57s - loss: 0.3104 - categorical_accuracy: 0.9046
24800/60000 [===========>..................] - ETA: 57s - loss: 0.3101 - categorical_accuracy: 0.9047
24832/60000 [===========>..................] - ETA: 57s - loss: 0.3098 - categorical_accuracy: 0.9047
24896/60000 [===========>..................] - ETA: 57s - loss: 0.3092 - categorical_accuracy: 0.9049
24960/60000 [===========>..................] - ETA: 56s - loss: 0.3086 - categorical_accuracy: 0.9051
24992/60000 [===========>..................] - ETA: 56s - loss: 0.3083 - categorical_accuracy: 0.9051
25024/60000 [===========>..................] - ETA: 56s - loss: 0.3080 - categorical_accuracy: 0.9053
25056/60000 [===========>..................] - ETA: 56s - loss: 0.3076 - categorical_accuracy: 0.9054
25088/60000 [===========>..................] - ETA: 56s - loss: 0.3073 - categorical_accuracy: 0.9055
25120/60000 [===========>..................] - ETA: 56s - loss: 0.3074 - categorical_accuracy: 0.9055
25152/60000 [===========>..................] - ETA: 56s - loss: 0.3071 - categorical_accuracy: 0.9056
25184/60000 [===========>..................] - ETA: 56s - loss: 0.3068 - categorical_accuracy: 0.9057
25216/60000 [===========>..................] - ETA: 56s - loss: 0.3065 - categorical_accuracy: 0.9058
25248/60000 [===========>..................] - ETA: 56s - loss: 0.3066 - categorical_accuracy: 0.9058
25280/60000 [===========>..................] - ETA: 56s - loss: 0.3064 - categorical_accuracy: 0.9059
25312/60000 [===========>..................] - ETA: 56s - loss: 0.3061 - categorical_accuracy: 0.9059
25376/60000 [===========>..................] - ETA: 56s - loss: 0.3057 - categorical_accuracy: 0.9061
25440/60000 [===========>..................] - ETA: 56s - loss: 0.3051 - categorical_accuracy: 0.9062
25472/60000 [===========>..................] - ETA: 56s - loss: 0.3049 - categorical_accuracy: 0.9063
25504/60000 [===========>..................] - ETA: 56s - loss: 0.3046 - categorical_accuracy: 0.9064
25536/60000 [===========>..................] - ETA: 55s - loss: 0.3046 - categorical_accuracy: 0.9064
25568/60000 [===========>..................] - ETA: 55s - loss: 0.3046 - categorical_accuracy: 0.9064
25632/60000 [===========>..................] - ETA: 55s - loss: 0.3044 - categorical_accuracy: 0.9064
25664/60000 [===========>..................] - ETA: 55s - loss: 0.3041 - categorical_accuracy: 0.9066
25696/60000 [===========>..................] - ETA: 55s - loss: 0.3039 - categorical_accuracy: 0.9066
25760/60000 [===========>..................] - ETA: 55s - loss: 0.3034 - categorical_accuracy: 0.9067
25792/60000 [===========>..................] - ETA: 55s - loss: 0.3031 - categorical_accuracy: 0.9068
25856/60000 [===========>..................] - ETA: 55s - loss: 0.3031 - categorical_accuracy: 0.9069
25888/60000 [===========>..................] - ETA: 55s - loss: 0.3028 - categorical_accuracy: 0.9070
25952/60000 [===========>..................] - ETA: 55s - loss: 0.3026 - categorical_accuracy: 0.9071
26016/60000 [============>.................] - ETA: 55s - loss: 0.3020 - categorical_accuracy: 0.9073
26048/60000 [============>.................] - ETA: 55s - loss: 0.3017 - categorical_accuracy: 0.9074
26112/60000 [============>.................] - ETA: 54s - loss: 0.3014 - categorical_accuracy: 0.9076
26144/60000 [============>.................] - ETA: 54s - loss: 0.3011 - categorical_accuracy: 0.9076
26176/60000 [============>.................] - ETA: 54s - loss: 0.3011 - categorical_accuracy: 0.9077
26208/60000 [============>.................] - ETA: 54s - loss: 0.3008 - categorical_accuracy: 0.9078
26240/60000 [============>.................] - ETA: 54s - loss: 0.3005 - categorical_accuracy: 0.9079
26272/60000 [============>.................] - ETA: 54s - loss: 0.3003 - categorical_accuracy: 0.9079
26304/60000 [============>.................] - ETA: 54s - loss: 0.3000 - categorical_accuracy: 0.9080
26336/60000 [============>.................] - ETA: 54s - loss: 0.2998 - categorical_accuracy: 0.9080
26400/60000 [============>.................] - ETA: 54s - loss: 0.2993 - categorical_accuracy: 0.9082
26432/60000 [============>.................] - ETA: 54s - loss: 0.2990 - categorical_accuracy: 0.9082
26464/60000 [============>.................] - ETA: 54s - loss: 0.2987 - categorical_accuracy: 0.9083
26496/60000 [============>.................] - ETA: 54s - loss: 0.2984 - categorical_accuracy: 0.9084
26528/60000 [============>.................] - ETA: 54s - loss: 0.2987 - categorical_accuracy: 0.9084
26592/60000 [============>.................] - ETA: 54s - loss: 0.2982 - categorical_accuracy: 0.9085
26656/60000 [============>.................] - ETA: 54s - loss: 0.2978 - categorical_accuracy: 0.9087
26688/60000 [============>.................] - ETA: 54s - loss: 0.2975 - categorical_accuracy: 0.9088
26720/60000 [============>.................] - ETA: 53s - loss: 0.2972 - categorical_accuracy: 0.9089
26752/60000 [============>.................] - ETA: 53s - loss: 0.2969 - categorical_accuracy: 0.9090
26784/60000 [============>.................] - ETA: 53s - loss: 0.2967 - categorical_accuracy: 0.9091
26816/60000 [============>.................] - ETA: 53s - loss: 0.2967 - categorical_accuracy: 0.9090
26880/60000 [============>.................] - ETA: 53s - loss: 0.2962 - categorical_accuracy: 0.9092
26944/60000 [============>.................] - ETA: 53s - loss: 0.2962 - categorical_accuracy: 0.9091
27008/60000 [============>.................] - ETA: 53s - loss: 0.2961 - categorical_accuracy: 0.9092
27072/60000 [============>.................] - ETA: 53s - loss: 0.2955 - categorical_accuracy: 0.9094
27104/60000 [============>.................] - ETA: 53s - loss: 0.2954 - categorical_accuracy: 0.9093
27168/60000 [============>.................] - ETA: 53s - loss: 0.2949 - categorical_accuracy: 0.9095
27232/60000 [============>.................] - ETA: 53s - loss: 0.2943 - categorical_accuracy: 0.9097
27264/60000 [============>.................] - ETA: 53s - loss: 0.2941 - categorical_accuracy: 0.9097
27328/60000 [============>.................] - ETA: 52s - loss: 0.2935 - categorical_accuracy: 0.9099
27360/60000 [============>.................] - ETA: 52s - loss: 0.2940 - categorical_accuracy: 0.9099
27424/60000 [============>.................] - ETA: 52s - loss: 0.2937 - categorical_accuracy: 0.9099
27456/60000 [============>.................] - ETA: 52s - loss: 0.2935 - categorical_accuracy: 0.9100
27520/60000 [============>.................] - ETA: 52s - loss: 0.2930 - categorical_accuracy: 0.9101
27584/60000 [============>.................] - ETA: 52s - loss: 0.2926 - categorical_accuracy: 0.9102
27616/60000 [============>.................] - ETA: 52s - loss: 0.2923 - categorical_accuracy: 0.9103
27648/60000 [============>.................] - ETA: 52s - loss: 0.2922 - categorical_accuracy: 0.9103
27680/60000 [============>.................] - ETA: 52s - loss: 0.2920 - categorical_accuracy: 0.9104
27712/60000 [============>.................] - ETA: 52s - loss: 0.2917 - categorical_accuracy: 0.9105
27744/60000 [============>.................] - ETA: 52s - loss: 0.2916 - categorical_accuracy: 0.9105
27776/60000 [============>.................] - ETA: 52s - loss: 0.2914 - categorical_accuracy: 0.9105
27808/60000 [============>.................] - ETA: 52s - loss: 0.2911 - categorical_accuracy: 0.9106
27872/60000 [============>.................] - ETA: 52s - loss: 0.2906 - categorical_accuracy: 0.9107
27936/60000 [============>.................] - ETA: 51s - loss: 0.2903 - categorical_accuracy: 0.9108
27968/60000 [============>.................] - ETA: 51s - loss: 0.2903 - categorical_accuracy: 0.9109
28000/60000 [=============>................] - ETA: 51s - loss: 0.2901 - categorical_accuracy: 0.9109
28032/60000 [=============>................] - ETA: 51s - loss: 0.2898 - categorical_accuracy: 0.9110
28064/60000 [=============>................] - ETA: 51s - loss: 0.2898 - categorical_accuracy: 0.9111
28096/60000 [=============>................] - ETA: 51s - loss: 0.2896 - categorical_accuracy: 0.9111
28128/60000 [=============>................] - ETA: 51s - loss: 0.2893 - categorical_accuracy: 0.9112
28160/60000 [=============>................] - ETA: 51s - loss: 0.2891 - categorical_accuracy: 0.9113
28192/60000 [=============>................] - ETA: 51s - loss: 0.2890 - categorical_accuracy: 0.9114
28224/60000 [=============>................] - ETA: 51s - loss: 0.2888 - categorical_accuracy: 0.9115
28288/60000 [=============>................] - ETA: 51s - loss: 0.2885 - categorical_accuracy: 0.9114
28320/60000 [=============>................] - ETA: 51s - loss: 0.2884 - categorical_accuracy: 0.9115
28352/60000 [=============>................] - ETA: 51s - loss: 0.2882 - categorical_accuracy: 0.9116
28384/60000 [=============>................] - ETA: 51s - loss: 0.2879 - categorical_accuracy: 0.9117
28416/60000 [=============>................] - ETA: 51s - loss: 0.2878 - categorical_accuracy: 0.9117
28448/60000 [=============>................] - ETA: 51s - loss: 0.2875 - categorical_accuracy: 0.9118
28480/60000 [=============>................] - ETA: 51s - loss: 0.2873 - categorical_accuracy: 0.9119
28512/60000 [=============>................] - ETA: 51s - loss: 0.2871 - categorical_accuracy: 0.9119
28544/60000 [=============>................] - ETA: 50s - loss: 0.2871 - categorical_accuracy: 0.9120
28608/60000 [=============>................] - ETA: 50s - loss: 0.2870 - categorical_accuracy: 0.9120
28672/60000 [=============>................] - ETA: 50s - loss: 0.2867 - categorical_accuracy: 0.9121
28704/60000 [=============>................] - ETA: 50s - loss: 0.2867 - categorical_accuracy: 0.9122
28768/60000 [=============>................] - ETA: 50s - loss: 0.2867 - categorical_accuracy: 0.9122
28832/60000 [=============>................] - ETA: 50s - loss: 0.2863 - categorical_accuracy: 0.9124
28896/60000 [=============>................] - ETA: 50s - loss: 0.2858 - categorical_accuracy: 0.9125
28928/60000 [=============>................] - ETA: 50s - loss: 0.2856 - categorical_accuracy: 0.9125
28960/60000 [=============>................] - ETA: 50s - loss: 0.2858 - categorical_accuracy: 0.9125
28992/60000 [=============>................] - ETA: 50s - loss: 0.2856 - categorical_accuracy: 0.9126
29024/60000 [=============>................] - ETA: 50s - loss: 0.2855 - categorical_accuracy: 0.9126
29088/60000 [=============>................] - ETA: 50s - loss: 0.2849 - categorical_accuracy: 0.9128
29120/60000 [=============>................] - ETA: 50s - loss: 0.2848 - categorical_accuracy: 0.9128
29184/60000 [=============>................] - ETA: 49s - loss: 0.2845 - categorical_accuracy: 0.9129
29248/60000 [=============>................] - ETA: 49s - loss: 0.2842 - categorical_accuracy: 0.9130
29280/60000 [=============>................] - ETA: 49s - loss: 0.2840 - categorical_accuracy: 0.9131
29312/60000 [=============>................] - ETA: 49s - loss: 0.2839 - categorical_accuracy: 0.9131
29344/60000 [=============>................] - ETA: 49s - loss: 0.2837 - categorical_accuracy: 0.9132
29376/60000 [=============>................] - ETA: 49s - loss: 0.2834 - categorical_accuracy: 0.9132
29408/60000 [=============>................] - ETA: 49s - loss: 0.2832 - categorical_accuracy: 0.9133
29472/60000 [=============>................] - ETA: 49s - loss: 0.2826 - categorical_accuracy: 0.9135
29504/60000 [=============>................] - ETA: 49s - loss: 0.2825 - categorical_accuracy: 0.9135
29536/60000 [=============>................] - ETA: 49s - loss: 0.2822 - categorical_accuracy: 0.9136
29600/60000 [=============>................] - ETA: 49s - loss: 0.2821 - categorical_accuracy: 0.9136
29632/60000 [=============>................] - ETA: 49s - loss: 0.2818 - categorical_accuracy: 0.9137
29664/60000 [=============>................] - ETA: 49s - loss: 0.2817 - categorical_accuracy: 0.9138
29696/60000 [=============>................] - ETA: 49s - loss: 0.2815 - categorical_accuracy: 0.9139
29760/60000 [=============>................] - ETA: 48s - loss: 0.2809 - categorical_accuracy: 0.9140
29792/60000 [=============>................] - ETA: 48s - loss: 0.2807 - categorical_accuracy: 0.9140
29856/60000 [=============>................] - ETA: 48s - loss: 0.2802 - categorical_accuracy: 0.9142
29920/60000 [=============>................] - ETA: 48s - loss: 0.2798 - categorical_accuracy: 0.9143
29984/60000 [=============>................] - ETA: 48s - loss: 0.2795 - categorical_accuracy: 0.9144
30016/60000 [==============>...............] - ETA: 48s - loss: 0.2795 - categorical_accuracy: 0.9144
30048/60000 [==============>...............] - ETA: 48s - loss: 0.2793 - categorical_accuracy: 0.9145
30080/60000 [==============>...............] - ETA: 48s - loss: 0.2790 - categorical_accuracy: 0.9145
30144/60000 [==============>...............] - ETA: 48s - loss: 0.2786 - categorical_accuracy: 0.9147
30208/60000 [==============>...............] - ETA: 48s - loss: 0.2784 - categorical_accuracy: 0.9147
30272/60000 [==============>...............] - ETA: 48s - loss: 0.2782 - categorical_accuracy: 0.9148
30304/60000 [==============>...............] - ETA: 48s - loss: 0.2781 - categorical_accuracy: 0.9148
30336/60000 [==============>...............] - ETA: 48s - loss: 0.2778 - categorical_accuracy: 0.9149
30400/60000 [==============>...............] - ETA: 47s - loss: 0.2776 - categorical_accuracy: 0.9149
30432/60000 [==============>...............] - ETA: 47s - loss: 0.2773 - categorical_accuracy: 0.9150
30464/60000 [==============>...............] - ETA: 47s - loss: 0.2773 - categorical_accuracy: 0.9150
30496/60000 [==============>...............] - ETA: 47s - loss: 0.2770 - categorical_accuracy: 0.9151
30528/60000 [==============>...............] - ETA: 47s - loss: 0.2769 - categorical_accuracy: 0.9151
30560/60000 [==============>...............] - ETA: 47s - loss: 0.2769 - categorical_accuracy: 0.9151
30592/60000 [==============>...............] - ETA: 47s - loss: 0.2768 - categorical_accuracy: 0.9151
30656/60000 [==============>...............] - ETA: 47s - loss: 0.2764 - categorical_accuracy: 0.9152
30720/60000 [==============>...............] - ETA: 47s - loss: 0.2761 - categorical_accuracy: 0.9153
30752/60000 [==============>...............] - ETA: 47s - loss: 0.2759 - categorical_accuracy: 0.9153
30816/60000 [==============>...............] - ETA: 47s - loss: 0.2755 - categorical_accuracy: 0.9154
30848/60000 [==============>...............] - ETA: 47s - loss: 0.2754 - categorical_accuracy: 0.9155
30880/60000 [==============>...............] - ETA: 47s - loss: 0.2753 - categorical_accuracy: 0.9154
30912/60000 [==============>...............] - ETA: 47s - loss: 0.2755 - categorical_accuracy: 0.9154
30944/60000 [==============>...............] - ETA: 47s - loss: 0.2753 - categorical_accuracy: 0.9155
30976/60000 [==============>...............] - ETA: 46s - loss: 0.2751 - categorical_accuracy: 0.9155
31008/60000 [==============>...............] - ETA: 46s - loss: 0.2749 - categorical_accuracy: 0.9156
31040/60000 [==============>...............] - ETA: 46s - loss: 0.2747 - categorical_accuracy: 0.9157
31072/60000 [==============>...............] - ETA: 46s - loss: 0.2744 - categorical_accuracy: 0.9157
31104/60000 [==============>...............] - ETA: 46s - loss: 0.2742 - categorical_accuracy: 0.9158
31136/60000 [==============>...............] - ETA: 46s - loss: 0.2744 - categorical_accuracy: 0.9158
31200/60000 [==============>...............] - ETA: 46s - loss: 0.2741 - categorical_accuracy: 0.9159
31232/60000 [==============>...............] - ETA: 46s - loss: 0.2740 - categorical_accuracy: 0.9159
31264/60000 [==============>...............] - ETA: 46s - loss: 0.2739 - categorical_accuracy: 0.9159
31296/60000 [==============>...............] - ETA: 46s - loss: 0.2738 - categorical_accuracy: 0.9159
31360/60000 [==============>...............] - ETA: 46s - loss: 0.2736 - categorical_accuracy: 0.9159
31392/60000 [==============>...............] - ETA: 46s - loss: 0.2735 - categorical_accuracy: 0.9160
31424/60000 [==============>...............] - ETA: 46s - loss: 0.2734 - categorical_accuracy: 0.9160
31456/60000 [==============>...............] - ETA: 46s - loss: 0.2731 - categorical_accuracy: 0.9161
31488/60000 [==============>...............] - ETA: 46s - loss: 0.2729 - categorical_accuracy: 0.9162
31520/60000 [==============>...............] - ETA: 46s - loss: 0.2728 - categorical_accuracy: 0.9162
31552/60000 [==============>...............] - ETA: 46s - loss: 0.2727 - categorical_accuracy: 0.9162
31616/60000 [==============>...............] - ETA: 45s - loss: 0.2723 - categorical_accuracy: 0.9164
31648/60000 [==============>...............] - ETA: 45s - loss: 0.2721 - categorical_accuracy: 0.9165
31680/60000 [==============>...............] - ETA: 45s - loss: 0.2720 - categorical_accuracy: 0.9165
31712/60000 [==============>...............] - ETA: 45s - loss: 0.2720 - categorical_accuracy: 0.9165
31744/60000 [==============>...............] - ETA: 45s - loss: 0.2718 - categorical_accuracy: 0.9166
31776/60000 [==============>...............] - ETA: 45s - loss: 0.2716 - categorical_accuracy: 0.9166
31808/60000 [==============>...............] - ETA: 45s - loss: 0.2716 - categorical_accuracy: 0.9167
31840/60000 [==============>...............] - ETA: 45s - loss: 0.2714 - categorical_accuracy: 0.9167
31872/60000 [==============>...............] - ETA: 45s - loss: 0.2711 - categorical_accuracy: 0.9168
31904/60000 [==============>...............] - ETA: 45s - loss: 0.2709 - categorical_accuracy: 0.9168
31936/60000 [==============>...............] - ETA: 45s - loss: 0.2708 - categorical_accuracy: 0.9169
31968/60000 [==============>...............] - ETA: 45s - loss: 0.2705 - categorical_accuracy: 0.9169
32000/60000 [===============>..............] - ETA: 45s - loss: 0.2703 - categorical_accuracy: 0.9170
32032/60000 [===============>..............] - ETA: 45s - loss: 0.2701 - categorical_accuracy: 0.9171
32096/60000 [===============>..............] - ETA: 45s - loss: 0.2701 - categorical_accuracy: 0.9171
32160/60000 [===============>..............] - ETA: 45s - loss: 0.2698 - categorical_accuracy: 0.9171
32192/60000 [===============>..............] - ETA: 44s - loss: 0.2700 - categorical_accuracy: 0.9170
32256/60000 [===============>..............] - ETA: 44s - loss: 0.2698 - categorical_accuracy: 0.9171
32288/60000 [===============>..............] - ETA: 44s - loss: 0.2697 - categorical_accuracy: 0.9171
32320/60000 [===============>..............] - ETA: 44s - loss: 0.2698 - categorical_accuracy: 0.9171
32352/60000 [===============>..............] - ETA: 44s - loss: 0.2698 - categorical_accuracy: 0.9172
32384/60000 [===============>..............] - ETA: 44s - loss: 0.2697 - categorical_accuracy: 0.9172
32416/60000 [===============>..............] - ETA: 44s - loss: 0.2694 - categorical_accuracy: 0.9173
32448/60000 [===============>..............] - ETA: 44s - loss: 0.2692 - categorical_accuracy: 0.9174
32480/60000 [===============>..............] - ETA: 44s - loss: 0.2690 - categorical_accuracy: 0.9175
32512/60000 [===============>..............] - ETA: 44s - loss: 0.2688 - categorical_accuracy: 0.9175
32544/60000 [===============>..............] - ETA: 44s - loss: 0.2687 - categorical_accuracy: 0.9175
32576/60000 [===============>..............] - ETA: 44s - loss: 0.2686 - categorical_accuracy: 0.9175
32608/60000 [===============>..............] - ETA: 44s - loss: 0.2684 - categorical_accuracy: 0.9176
32640/60000 [===============>..............] - ETA: 44s - loss: 0.2681 - categorical_accuracy: 0.9177
32672/60000 [===============>..............] - ETA: 44s - loss: 0.2681 - categorical_accuracy: 0.9178
32704/60000 [===============>..............] - ETA: 44s - loss: 0.2678 - categorical_accuracy: 0.9178
32736/60000 [===============>..............] - ETA: 44s - loss: 0.2677 - categorical_accuracy: 0.9179
32768/60000 [===============>..............] - ETA: 44s - loss: 0.2675 - categorical_accuracy: 0.9179
32800/60000 [===============>..............] - ETA: 44s - loss: 0.2673 - categorical_accuracy: 0.9180
32832/60000 [===============>..............] - ETA: 43s - loss: 0.2670 - categorical_accuracy: 0.9180
32864/60000 [===============>..............] - ETA: 43s - loss: 0.2670 - categorical_accuracy: 0.9181
32896/60000 [===============>..............] - ETA: 43s - loss: 0.2668 - categorical_accuracy: 0.9181
32928/60000 [===============>..............] - ETA: 43s - loss: 0.2668 - categorical_accuracy: 0.9181
32960/60000 [===============>..............] - ETA: 43s - loss: 0.2667 - categorical_accuracy: 0.9181
32992/60000 [===============>..............] - ETA: 43s - loss: 0.2666 - categorical_accuracy: 0.9182
33024/60000 [===============>..............] - ETA: 43s - loss: 0.2664 - categorical_accuracy: 0.9182
33056/60000 [===============>..............] - ETA: 43s - loss: 0.2662 - categorical_accuracy: 0.9183
33088/60000 [===============>..............] - ETA: 43s - loss: 0.2663 - categorical_accuracy: 0.9183
33120/60000 [===============>..............] - ETA: 43s - loss: 0.2662 - categorical_accuracy: 0.9183
33152/60000 [===============>..............] - ETA: 43s - loss: 0.2660 - categorical_accuracy: 0.9183
33184/60000 [===============>..............] - ETA: 43s - loss: 0.2658 - categorical_accuracy: 0.9184
33216/60000 [===============>..............] - ETA: 43s - loss: 0.2657 - categorical_accuracy: 0.9184
33248/60000 [===============>..............] - ETA: 43s - loss: 0.2654 - categorical_accuracy: 0.9185
33280/60000 [===============>..............] - ETA: 43s - loss: 0.2653 - categorical_accuracy: 0.9185
33312/60000 [===============>..............] - ETA: 43s - loss: 0.2651 - categorical_accuracy: 0.9186
33344/60000 [===============>..............] - ETA: 43s - loss: 0.2650 - categorical_accuracy: 0.9186
33376/60000 [===============>..............] - ETA: 43s - loss: 0.2648 - categorical_accuracy: 0.9187
33408/60000 [===============>..............] - ETA: 43s - loss: 0.2645 - categorical_accuracy: 0.9188
33440/60000 [===============>..............] - ETA: 42s - loss: 0.2643 - categorical_accuracy: 0.9189
33472/60000 [===============>..............] - ETA: 42s - loss: 0.2643 - categorical_accuracy: 0.9189
33504/60000 [===============>..............] - ETA: 42s - loss: 0.2641 - categorical_accuracy: 0.9190
33536/60000 [===============>..............] - ETA: 42s - loss: 0.2639 - categorical_accuracy: 0.9190
33568/60000 [===============>..............] - ETA: 42s - loss: 0.2638 - categorical_accuracy: 0.9191
33632/60000 [===============>..............] - ETA: 42s - loss: 0.2635 - categorical_accuracy: 0.9192
33664/60000 [===============>..............] - ETA: 42s - loss: 0.2636 - categorical_accuracy: 0.9191
33696/60000 [===============>..............] - ETA: 42s - loss: 0.2634 - categorical_accuracy: 0.9192
33760/60000 [===============>..............] - ETA: 42s - loss: 0.2632 - categorical_accuracy: 0.9193
33792/60000 [===============>..............] - ETA: 42s - loss: 0.2631 - categorical_accuracy: 0.9193
33856/60000 [===============>..............] - ETA: 42s - loss: 0.2627 - categorical_accuracy: 0.9195
33920/60000 [===============>..............] - ETA: 42s - loss: 0.2623 - categorical_accuracy: 0.9196
33952/60000 [===============>..............] - ETA: 42s - loss: 0.2621 - categorical_accuracy: 0.9197
34016/60000 [================>.............] - ETA: 42s - loss: 0.2620 - categorical_accuracy: 0.9197
34080/60000 [================>.............] - ETA: 41s - loss: 0.2618 - categorical_accuracy: 0.9197
34112/60000 [================>.............] - ETA: 41s - loss: 0.2619 - categorical_accuracy: 0.9197
34144/60000 [================>.............] - ETA: 41s - loss: 0.2619 - categorical_accuracy: 0.9198
34176/60000 [================>.............] - ETA: 41s - loss: 0.2618 - categorical_accuracy: 0.9198
34208/60000 [================>.............] - ETA: 41s - loss: 0.2621 - categorical_accuracy: 0.9198
34272/60000 [================>.............] - ETA: 41s - loss: 0.2617 - categorical_accuracy: 0.9199
34304/60000 [================>.............] - ETA: 41s - loss: 0.2615 - categorical_accuracy: 0.9200
34336/60000 [================>.............] - ETA: 41s - loss: 0.2614 - categorical_accuracy: 0.9200
34368/60000 [================>.............] - ETA: 41s - loss: 0.2612 - categorical_accuracy: 0.9201
34400/60000 [================>.............] - ETA: 41s - loss: 0.2610 - categorical_accuracy: 0.9201
34432/60000 [================>.............] - ETA: 41s - loss: 0.2608 - categorical_accuracy: 0.9202
34464/60000 [================>.............] - ETA: 41s - loss: 0.2606 - categorical_accuracy: 0.9202
34496/60000 [================>.............] - ETA: 41s - loss: 0.2605 - categorical_accuracy: 0.9203
34528/60000 [================>.............] - ETA: 41s - loss: 0.2603 - categorical_accuracy: 0.9204
34560/60000 [================>.............] - ETA: 41s - loss: 0.2602 - categorical_accuracy: 0.9203
34592/60000 [================>.............] - ETA: 41s - loss: 0.2600 - categorical_accuracy: 0.9204
34624/60000 [================>.............] - ETA: 41s - loss: 0.2598 - categorical_accuracy: 0.9205
34656/60000 [================>.............] - ETA: 40s - loss: 0.2597 - categorical_accuracy: 0.9205
34720/60000 [================>.............] - ETA: 40s - loss: 0.2595 - categorical_accuracy: 0.9206
34752/60000 [================>.............] - ETA: 40s - loss: 0.2595 - categorical_accuracy: 0.9206
34784/60000 [================>.............] - ETA: 40s - loss: 0.2593 - categorical_accuracy: 0.9207
34848/60000 [================>.............] - ETA: 40s - loss: 0.2590 - categorical_accuracy: 0.9208
34880/60000 [================>.............] - ETA: 40s - loss: 0.2588 - categorical_accuracy: 0.9208
34912/60000 [================>.............] - ETA: 40s - loss: 0.2586 - categorical_accuracy: 0.9209
34944/60000 [================>.............] - ETA: 40s - loss: 0.2584 - categorical_accuracy: 0.9210
34976/60000 [================>.............] - ETA: 40s - loss: 0.2582 - categorical_accuracy: 0.9210
35008/60000 [================>.............] - ETA: 40s - loss: 0.2579 - categorical_accuracy: 0.9211
35040/60000 [================>.............] - ETA: 40s - loss: 0.2578 - categorical_accuracy: 0.9211
35072/60000 [================>.............] - ETA: 40s - loss: 0.2578 - categorical_accuracy: 0.9211
35104/60000 [================>.............] - ETA: 40s - loss: 0.2576 - categorical_accuracy: 0.9211
35136/60000 [================>.............] - ETA: 40s - loss: 0.2576 - categorical_accuracy: 0.9212
35200/60000 [================>.............] - ETA: 40s - loss: 0.2571 - categorical_accuracy: 0.9213
35232/60000 [================>.............] - ETA: 40s - loss: 0.2569 - categorical_accuracy: 0.9214
35264/60000 [================>.............] - ETA: 40s - loss: 0.2567 - categorical_accuracy: 0.9215
35296/60000 [================>.............] - ETA: 39s - loss: 0.2565 - categorical_accuracy: 0.9215
35328/60000 [================>.............] - ETA: 39s - loss: 0.2563 - categorical_accuracy: 0.9216
35360/60000 [================>.............] - ETA: 39s - loss: 0.2561 - categorical_accuracy: 0.9217
35424/60000 [================>.............] - ETA: 39s - loss: 0.2560 - categorical_accuracy: 0.9217
35488/60000 [================>.............] - ETA: 39s - loss: 0.2558 - categorical_accuracy: 0.9217
35520/60000 [================>.............] - ETA: 39s - loss: 0.2556 - categorical_accuracy: 0.9218
35552/60000 [================>.............] - ETA: 39s - loss: 0.2554 - categorical_accuracy: 0.9219
35584/60000 [================>.............] - ETA: 39s - loss: 0.2552 - categorical_accuracy: 0.9219
35616/60000 [================>.............] - ETA: 39s - loss: 0.2551 - categorical_accuracy: 0.9220
35648/60000 [================>.............] - ETA: 39s - loss: 0.2549 - categorical_accuracy: 0.9220
35680/60000 [================>.............] - ETA: 39s - loss: 0.2549 - categorical_accuracy: 0.9221
35712/60000 [================>.............] - ETA: 39s - loss: 0.2547 - categorical_accuracy: 0.9222
35744/60000 [================>.............] - ETA: 39s - loss: 0.2545 - categorical_accuracy: 0.9222
35808/60000 [================>.............] - ETA: 39s - loss: 0.2544 - categorical_accuracy: 0.9223
35840/60000 [================>.............] - ETA: 39s - loss: 0.2542 - categorical_accuracy: 0.9223
35872/60000 [================>.............] - ETA: 39s - loss: 0.2541 - categorical_accuracy: 0.9223
35936/60000 [================>.............] - ETA: 38s - loss: 0.2537 - categorical_accuracy: 0.9224
35968/60000 [================>.............] - ETA: 38s - loss: 0.2535 - categorical_accuracy: 0.9225
36000/60000 [=================>............] - ETA: 38s - loss: 0.2534 - categorical_accuracy: 0.9225
36064/60000 [=================>............] - ETA: 38s - loss: 0.2530 - categorical_accuracy: 0.9227
36128/60000 [=================>............] - ETA: 38s - loss: 0.2527 - categorical_accuracy: 0.9227
36160/60000 [=================>............] - ETA: 38s - loss: 0.2525 - categorical_accuracy: 0.9228
36192/60000 [=================>............] - ETA: 38s - loss: 0.2524 - categorical_accuracy: 0.9228
36224/60000 [=================>............] - ETA: 38s - loss: 0.2523 - categorical_accuracy: 0.9229
36256/60000 [=================>............] - ETA: 38s - loss: 0.2521 - categorical_accuracy: 0.9229
36288/60000 [=================>............] - ETA: 38s - loss: 0.2519 - categorical_accuracy: 0.9230
36320/60000 [=================>............] - ETA: 38s - loss: 0.2518 - categorical_accuracy: 0.9230
36352/60000 [=================>............] - ETA: 38s - loss: 0.2516 - categorical_accuracy: 0.9231
36416/60000 [=================>............] - ETA: 38s - loss: 0.2512 - categorical_accuracy: 0.9232
36480/60000 [=================>............] - ETA: 38s - loss: 0.2508 - categorical_accuracy: 0.9233
36512/60000 [=================>............] - ETA: 37s - loss: 0.2507 - categorical_accuracy: 0.9233
36544/60000 [=================>............] - ETA: 37s - loss: 0.2508 - categorical_accuracy: 0.9233
36576/60000 [=================>............] - ETA: 37s - loss: 0.2506 - categorical_accuracy: 0.9234
36608/60000 [=================>............] - ETA: 37s - loss: 0.2504 - categorical_accuracy: 0.9235
36640/60000 [=================>............] - ETA: 37s - loss: 0.2503 - categorical_accuracy: 0.9234
36672/60000 [=================>............] - ETA: 37s - loss: 0.2501 - categorical_accuracy: 0.9235
36704/60000 [=================>............] - ETA: 37s - loss: 0.2500 - categorical_accuracy: 0.9236
36736/60000 [=================>............] - ETA: 37s - loss: 0.2498 - categorical_accuracy: 0.9236
36768/60000 [=================>............] - ETA: 37s - loss: 0.2496 - categorical_accuracy: 0.9237
36800/60000 [=================>............] - ETA: 37s - loss: 0.2494 - categorical_accuracy: 0.9237
36832/60000 [=================>............] - ETA: 37s - loss: 0.2492 - categorical_accuracy: 0.9238
36864/60000 [=================>............] - ETA: 37s - loss: 0.2490 - categorical_accuracy: 0.9238
36896/60000 [=================>............] - ETA: 37s - loss: 0.2490 - categorical_accuracy: 0.9238
36928/60000 [=================>............] - ETA: 37s - loss: 0.2489 - categorical_accuracy: 0.9239
36960/60000 [=================>............] - ETA: 37s - loss: 0.2489 - categorical_accuracy: 0.9239
36992/60000 [=================>............] - ETA: 37s - loss: 0.2488 - categorical_accuracy: 0.9239
37024/60000 [=================>............] - ETA: 37s - loss: 0.2487 - categorical_accuracy: 0.9239
37088/60000 [=================>............] - ETA: 37s - loss: 0.2487 - categorical_accuracy: 0.9240
37120/60000 [=================>............] - ETA: 37s - loss: 0.2486 - categorical_accuracy: 0.9240
37184/60000 [=================>............] - ETA: 36s - loss: 0.2484 - categorical_accuracy: 0.9240
37216/60000 [=================>............] - ETA: 36s - loss: 0.2484 - categorical_accuracy: 0.9240
37248/60000 [=================>............] - ETA: 36s - loss: 0.2483 - categorical_accuracy: 0.9241
37280/60000 [=================>............] - ETA: 36s - loss: 0.2482 - categorical_accuracy: 0.9241
37312/60000 [=================>............] - ETA: 36s - loss: 0.2481 - categorical_accuracy: 0.9242
37344/60000 [=================>............] - ETA: 36s - loss: 0.2479 - categorical_accuracy: 0.9242
37376/60000 [=================>............] - ETA: 36s - loss: 0.2478 - categorical_accuracy: 0.9242
37440/60000 [=================>............] - ETA: 36s - loss: 0.2475 - categorical_accuracy: 0.9243
37472/60000 [=================>............] - ETA: 36s - loss: 0.2473 - categorical_accuracy: 0.9244
37504/60000 [=================>............] - ETA: 36s - loss: 0.2472 - categorical_accuracy: 0.9244
37536/60000 [=================>............] - ETA: 36s - loss: 0.2471 - categorical_accuracy: 0.9245
37568/60000 [=================>............] - ETA: 36s - loss: 0.2470 - categorical_accuracy: 0.9245
37600/60000 [=================>............] - ETA: 36s - loss: 0.2469 - categorical_accuracy: 0.9245
37632/60000 [=================>............] - ETA: 36s - loss: 0.2467 - categorical_accuracy: 0.9246
37664/60000 [=================>............] - ETA: 36s - loss: 0.2465 - categorical_accuracy: 0.9246
37696/60000 [=================>............] - ETA: 36s - loss: 0.2463 - categorical_accuracy: 0.9247
37728/60000 [=================>............] - ETA: 36s - loss: 0.2462 - categorical_accuracy: 0.9247
37760/60000 [=================>............] - ETA: 35s - loss: 0.2464 - categorical_accuracy: 0.9247
37824/60000 [=================>............] - ETA: 35s - loss: 0.2465 - categorical_accuracy: 0.9248
37856/60000 [=================>............] - ETA: 35s - loss: 0.2466 - categorical_accuracy: 0.9248
37888/60000 [=================>............] - ETA: 35s - loss: 0.2465 - categorical_accuracy: 0.9248
37952/60000 [=================>............] - ETA: 35s - loss: 0.2464 - categorical_accuracy: 0.9249
37984/60000 [=================>............] - ETA: 35s - loss: 0.2462 - categorical_accuracy: 0.9249
38016/60000 [==================>...........] - ETA: 35s - loss: 0.2461 - categorical_accuracy: 0.9250
38048/60000 [==================>...........] - ETA: 35s - loss: 0.2461 - categorical_accuracy: 0.9249
38112/60000 [==================>...........] - ETA: 35s - loss: 0.2460 - categorical_accuracy: 0.9249
38144/60000 [==================>...........] - ETA: 35s - loss: 0.2458 - categorical_accuracy: 0.9250
38176/60000 [==================>...........] - ETA: 35s - loss: 0.2457 - categorical_accuracy: 0.9250
38208/60000 [==================>...........] - ETA: 35s - loss: 0.2457 - categorical_accuracy: 0.9250
38240/60000 [==================>...........] - ETA: 35s - loss: 0.2455 - categorical_accuracy: 0.9251
38272/60000 [==================>...........] - ETA: 35s - loss: 0.2454 - categorical_accuracy: 0.9251
38336/60000 [==================>...........] - ETA: 35s - loss: 0.2454 - categorical_accuracy: 0.9251
38400/60000 [==================>...........] - ETA: 34s - loss: 0.2452 - categorical_accuracy: 0.9252
38464/60000 [==================>...........] - ETA: 34s - loss: 0.2450 - categorical_accuracy: 0.9253
38496/60000 [==================>...........] - ETA: 34s - loss: 0.2448 - categorical_accuracy: 0.9253
38528/60000 [==================>...........] - ETA: 34s - loss: 0.2447 - categorical_accuracy: 0.9254
38560/60000 [==================>...........] - ETA: 34s - loss: 0.2447 - categorical_accuracy: 0.9254
38624/60000 [==================>...........] - ETA: 34s - loss: 0.2444 - categorical_accuracy: 0.9254
38656/60000 [==================>...........] - ETA: 34s - loss: 0.2442 - categorical_accuracy: 0.9255
38688/60000 [==================>...........] - ETA: 34s - loss: 0.2441 - categorical_accuracy: 0.9255
38720/60000 [==================>...........] - ETA: 34s - loss: 0.2440 - categorical_accuracy: 0.9256
38752/60000 [==================>...........] - ETA: 34s - loss: 0.2440 - categorical_accuracy: 0.9256
38784/60000 [==================>...........] - ETA: 34s - loss: 0.2439 - categorical_accuracy: 0.9256
38816/60000 [==================>...........] - ETA: 34s - loss: 0.2438 - categorical_accuracy: 0.9256
38848/60000 [==================>...........] - ETA: 34s - loss: 0.2436 - categorical_accuracy: 0.9257
38880/60000 [==================>...........] - ETA: 34s - loss: 0.2435 - categorical_accuracy: 0.9257
38912/60000 [==================>...........] - ETA: 34s - loss: 0.2433 - categorical_accuracy: 0.9258
38944/60000 [==================>...........] - ETA: 34s - loss: 0.2432 - categorical_accuracy: 0.9258
38976/60000 [==================>...........] - ETA: 33s - loss: 0.2430 - categorical_accuracy: 0.9259
39008/60000 [==================>...........] - ETA: 33s - loss: 0.2429 - categorical_accuracy: 0.9259
39040/60000 [==================>...........] - ETA: 33s - loss: 0.2428 - categorical_accuracy: 0.9259
39072/60000 [==================>...........] - ETA: 33s - loss: 0.2427 - categorical_accuracy: 0.9259
39104/60000 [==================>...........] - ETA: 33s - loss: 0.2425 - categorical_accuracy: 0.9260
39136/60000 [==================>...........] - ETA: 33s - loss: 0.2423 - categorical_accuracy: 0.9261
39168/60000 [==================>...........] - ETA: 33s - loss: 0.2422 - categorical_accuracy: 0.9261
39232/60000 [==================>...........] - ETA: 33s - loss: 0.2420 - categorical_accuracy: 0.9262
39264/60000 [==================>...........] - ETA: 33s - loss: 0.2419 - categorical_accuracy: 0.9262
39328/60000 [==================>...........] - ETA: 33s - loss: 0.2416 - categorical_accuracy: 0.9263
39392/60000 [==================>...........] - ETA: 33s - loss: 0.2413 - categorical_accuracy: 0.9264
39424/60000 [==================>...........] - ETA: 33s - loss: 0.2412 - categorical_accuracy: 0.9264
39488/60000 [==================>...........] - ETA: 33s - loss: 0.2409 - categorical_accuracy: 0.9265
39520/60000 [==================>...........] - ETA: 33s - loss: 0.2408 - categorical_accuracy: 0.9265
39552/60000 [==================>...........] - ETA: 33s - loss: 0.2406 - categorical_accuracy: 0.9266
39616/60000 [==================>...........] - ETA: 32s - loss: 0.2403 - categorical_accuracy: 0.9267
39648/60000 [==================>...........] - ETA: 32s - loss: 0.2403 - categorical_accuracy: 0.9267
39680/60000 [==================>...........] - ETA: 32s - loss: 0.2402 - categorical_accuracy: 0.9267
39712/60000 [==================>...........] - ETA: 32s - loss: 0.2401 - categorical_accuracy: 0.9268
39744/60000 [==================>...........] - ETA: 32s - loss: 0.2399 - categorical_accuracy: 0.9268
39808/60000 [==================>...........] - ETA: 32s - loss: 0.2397 - categorical_accuracy: 0.9269
39872/60000 [==================>...........] - ETA: 32s - loss: 0.2394 - categorical_accuracy: 0.9270
39904/60000 [==================>...........] - ETA: 32s - loss: 0.2393 - categorical_accuracy: 0.9270
39968/60000 [==================>...........] - ETA: 32s - loss: 0.2391 - categorical_accuracy: 0.9271
40032/60000 [===================>..........] - ETA: 32s - loss: 0.2389 - categorical_accuracy: 0.9271
40064/60000 [===================>..........] - ETA: 32s - loss: 0.2387 - categorical_accuracy: 0.9272
40096/60000 [===================>..........] - ETA: 32s - loss: 0.2387 - categorical_accuracy: 0.9272
40128/60000 [===================>..........] - ETA: 32s - loss: 0.2385 - categorical_accuracy: 0.9273
40192/60000 [===================>..........] - ETA: 32s - loss: 0.2383 - categorical_accuracy: 0.9273
40224/60000 [===================>..........] - ETA: 31s - loss: 0.2383 - categorical_accuracy: 0.9274
40288/60000 [===================>..........] - ETA: 31s - loss: 0.2381 - categorical_accuracy: 0.9274
40352/60000 [===================>..........] - ETA: 31s - loss: 0.2378 - categorical_accuracy: 0.9275
40384/60000 [===================>..........] - ETA: 31s - loss: 0.2376 - categorical_accuracy: 0.9275
40416/60000 [===================>..........] - ETA: 31s - loss: 0.2376 - categorical_accuracy: 0.9275
40448/60000 [===================>..........] - ETA: 31s - loss: 0.2375 - categorical_accuracy: 0.9276
40480/60000 [===================>..........] - ETA: 31s - loss: 0.2374 - categorical_accuracy: 0.9276
40512/60000 [===================>..........] - ETA: 31s - loss: 0.2373 - categorical_accuracy: 0.9276
40544/60000 [===================>..........] - ETA: 31s - loss: 0.2372 - categorical_accuracy: 0.9276
40608/60000 [===================>..........] - ETA: 31s - loss: 0.2369 - categorical_accuracy: 0.9277
40672/60000 [===================>..........] - ETA: 31s - loss: 0.2367 - categorical_accuracy: 0.9277
40704/60000 [===================>..........] - ETA: 31s - loss: 0.2367 - categorical_accuracy: 0.9277
40736/60000 [===================>..........] - ETA: 31s - loss: 0.2366 - categorical_accuracy: 0.9278
40768/60000 [===================>..........] - ETA: 31s - loss: 0.2366 - categorical_accuracy: 0.9278
40832/60000 [===================>..........] - ETA: 30s - loss: 0.2363 - categorical_accuracy: 0.9279
40864/60000 [===================>..........] - ETA: 30s - loss: 0.2361 - categorical_accuracy: 0.9279
40928/60000 [===================>..........] - ETA: 30s - loss: 0.2360 - categorical_accuracy: 0.9279
40992/60000 [===================>..........] - ETA: 30s - loss: 0.2357 - categorical_accuracy: 0.9280
41024/60000 [===================>..........] - ETA: 30s - loss: 0.2356 - categorical_accuracy: 0.9281
41088/60000 [===================>..........] - ETA: 30s - loss: 0.2353 - categorical_accuracy: 0.9282
41120/60000 [===================>..........] - ETA: 30s - loss: 0.2351 - categorical_accuracy: 0.9282
41184/60000 [===================>..........] - ETA: 30s - loss: 0.2349 - categorical_accuracy: 0.9283
41216/60000 [===================>..........] - ETA: 30s - loss: 0.2348 - categorical_accuracy: 0.9283
41248/60000 [===================>..........] - ETA: 30s - loss: 0.2347 - categorical_accuracy: 0.9284
41312/60000 [===================>..........] - ETA: 30s - loss: 0.2344 - categorical_accuracy: 0.9284
41344/60000 [===================>..........] - ETA: 30s - loss: 0.2343 - categorical_accuracy: 0.9285
41408/60000 [===================>..........] - ETA: 30s - loss: 0.2341 - categorical_accuracy: 0.9285
41440/60000 [===================>..........] - ETA: 29s - loss: 0.2341 - categorical_accuracy: 0.9286
41472/60000 [===================>..........] - ETA: 29s - loss: 0.2342 - categorical_accuracy: 0.9286
41536/60000 [===================>..........] - ETA: 29s - loss: 0.2341 - categorical_accuracy: 0.9286
41600/60000 [===================>..........] - ETA: 29s - loss: 0.2339 - categorical_accuracy: 0.9287
41632/60000 [===================>..........] - ETA: 29s - loss: 0.2339 - categorical_accuracy: 0.9287
41664/60000 [===================>..........] - ETA: 29s - loss: 0.2337 - categorical_accuracy: 0.9287
41696/60000 [===================>..........] - ETA: 29s - loss: 0.2337 - categorical_accuracy: 0.9287
41728/60000 [===================>..........] - ETA: 29s - loss: 0.2336 - categorical_accuracy: 0.9287
41792/60000 [===================>..........] - ETA: 29s - loss: 0.2333 - categorical_accuracy: 0.9288
41824/60000 [===================>..........] - ETA: 29s - loss: 0.2332 - categorical_accuracy: 0.9288
41888/60000 [===================>..........] - ETA: 29s - loss: 0.2331 - categorical_accuracy: 0.9289
41920/60000 [===================>..........] - ETA: 29s - loss: 0.2331 - categorical_accuracy: 0.9289
41984/60000 [===================>..........] - ETA: 29s - loss: 0.2330 - categorical_accuracy: 0.9289
42048/60000 [====================>.........] - ETA: 29s - loss: 0.2328 - categorical_accuracy: 0.9290
42112/60000 [====================>.........] - ETA: 28s - loss: 0.2325 - categorical_accuracy: 0.9291
42176/60000 [====================>.........] - ETA: 28s - loss: 0.2323 - categorical_accuracy: 0.9292
42240/60000 [====================>.........] - ETA: 28s - loss: 0.2321 - categorical_accuracy: 0.9292
42272/60000 [====================>.........] - ETA: 28s - loss: 0.2319 - categorical_accuracy: 0.9293
42304/60000 [====================>.........] - ETA: 28s - loss: 0.2319 - categorical_accuracy: 0.9293
42336/60000 [====================>.........] - ETA: 28s - loss: 0.2317 - categorical_accuracy: 0.9293
42368/60000 [====================>.........] - ETA: 28s - loss: 0.2316 - categorical_accuracy: 0.9293
42400/60000 [====================>.........] - ETA: 28s - loss: 0.2316 - categorical_accuracy: 0.9293
42432/60000 [====================>.........] - ETA: 28s - loss: 0.2316 - categorical_accuracy: 0.9293
42464/60000 [====================>.........] - ETA: 28s - loss: 0.2314 - categorical_accuracy: 0.9294
42496/60000 [====================>.........] - ETA: 28s - loss: 0.2313 - categorical_accuracy: 0.9294
42528/60000 [====================>.........] - ETA: 28s - loss: 0.2311 - categorical_accuracy: 0.9294
42560/60000 [====================>.........] - ETA: 28s - loss: 0.2310 - categorical_accuracy: 0.9295
42592/60000 [====================>.........] - ETA: 28s - loss: 0.2308 - categorical_accuracy: 0.9295
42624/60000 [====================>.........] - ETA: 28s - loss: 0.2309 - categorical_accuracy: 0.9295
42656/60000 [====================>.........] - ETA: 28s - loss: 0.2310 - categorical_accuracy: 0.9295
42688/60000 [====================>.........] - ETA: 27s - loss: 0.2308 - categorical_accuracy: 0.9295
42720/60000 [====================>.........] - ETA: 27s - loss: 0.2309 - categorical_accuracy: 0.9295
42752/60000 [====================>.........] - ETA: 27s - loss: 0.2307 - categorical_accuracy: 0.9295
42784/60000 [====================>.........] - ETA: 27s - loss: 0.2306 - categorical_accuracy: 0.9296
42816/60000 [====================>.........] - ETA: 27s - loss: 0.2305 - categorical_accuracy: 0.9296
42848/60000 [====================>.........] - ETA: 27s - loss: 0.2304 - categorical_accuracy: 0.9296
42880/60000 [====================>.........] - ETA: 27s - loss: 0.2303 - categorical_accuracy: 0.9296
42912/60000 [====================>.........] - ETA: 27s - loss: 0.2303 - categorical_accuracy: 0.9296
42944/60000 [====================>.........] - ETA: 27s - loss: 0.2302 - categorical_accuracy: 0.9297
42976/60000 [====================>.........] - ETA: 27s - loss: 0.2301 - categorical_accuracy: 0.9297
43008/60000 [====================>.........] - ETA: 27s - loss: 0.2300 - categorical_accuracy: 0.9297
43040/60000 [====================>.........] - ETA: 27s - loss: 0.2300 - categorical_accuracy: 0.9297
43072/60000 [====================>.........] - ETA: 27s - loss: 0.2298 - categorical_accuracy: 0.9297
43104/60000 [====================>.........] - ETA: 27s - loss: 0.2297 - categorical_accuracy: 0.9298
43168/60000 [====================>.........] - ETA: 27s - loss: 0.2294 - categorical_accuracy: 0.9299
43200/60000 [====================>.........] - ETA: 27s - loss: 0.2293 - categorical_accuracy: 0.9299
43232/60000 [====================>.........] - ETA: 27s - loss: 0.2293 - categorical_accuracy: 0.9299
43264/60000 [====================>.........] - ETA: 27s - loss: 0.2292 - categorical_accuracy: 0.9300
43328/60000 [====================>.........] - ETA: 26s - loss: 0.2290 - categorical_accuracy: 0.9300
43392/60000 [====================>.........] - ETA: 26s - loss: 0.2288 - categorical_accuracy: 0.9301
43456/60000 [====================>.........] - ETA: 26s - loss: 0.2285 - categorical_accuracy: 0.9302
43488/60000 [====================>.........] - ETA: 26s - loss: 0.2284 - categorical_accuracy: 0.9302
43520/60000 [====================>.........] - ETA: 26s - loss: 0.2283 - categorical_accuracy: 0.9302
43552/60000 [====================>.........] - ETA: 26s - loss: 0.2281 - categorical_accuracy: 0.9302
43584/60000 [====================>.........] - ETA: 26s - loss: 0.2280 - categorical_accuracy: 0.9303
43648/60000 [====================>.........] - ETA: 26s - loss: 0.2278 - categorical_accuracy: 0.9304
43680/60000 [====================>.........] - ETA: 26s - loss: 0.2276 - categorical_accuracy: 0.9304
43712/60000 [====================>.........] - ETA: 26s - loss: 0.2275 - categorical_accuracy: 0.9305
43744/60000 [====================>.........] - ETA: 26s - loss: 0.2273 - categorical_accuracy: 0.9305
43776/60000 [====================>.........] - ETA: 26s - loss: 0.2272 - categorical_accuracy: 0.9306
43808/60000 [====================>.........] - ETA: 26s - loss: 0.2271 - categorical_accuracy: 0.9306
43840/60000 [====================>.........] - ETA: 26s - loss: 0.2270 - categorical_accuracy: 0.9306
43872/60000 [====================>.........] - ETA: 26s - loss: 0.2269 - categorical_accuracy: 0.9306
43904/60000 [====================>.........] - ETA: 25s - loss: 0.2268 - categorical_accuracy: 0.9307
43936/60000 [====================>.........] - ETA: 25s - loss: 0.2266 - categorical_accuracy: 0.9307
43968/60000 [====================>.........] - ETA: 25s - loss: 0.2266 - categorical_accuracy: 0.9307
44000/60000 [=====================>........] - ETA: 25s - loss: 0.2265 - categorical_accuracy: 0.9308
44064/60000 [=====================>........] - ETA: 25s - loss: 0.2264 - categorical_accuracy: 0.9308
44096/60000 [=====================>........] - ETA: 25s - loss: 0.2262 - categorical_accuracy: 0.9309
44128/60000 [=====================>........] - ETA: 25s - loss: 0.2263 - categorical_accuracy: 0.9309
44192/60000 [=====================>........] - ETA: 25s - loss: 0.2260 - categorical_accuracy: 0.9309
44224/60000 [=====================>........] - ETA: 25s - loss: 0.2258 - categorical_accuracy: 0.9310
44288/60000 [=====================>........] - ETA: 25s - loss: 0.2256 - categorical_accuracy: 0.9310
44352/60000 [=====================>........] - ETA: 25s - loss: 0.2254 - categorical_accuracy: 0.9311
44384/60000 [=====================>........] - ETA: 25s - loss: 0.2253 - categorical_accuracy: 0.9312
44448/60000 [=====================>........] - ETA: 25s - loss: 0.2251 - categorical_accuracy: 0.9312
44512/60000 [=====================>........] - ETA: 25s - loss: 0.2249 - categorical_accuracy: 0.9312
44544/60000 [=====================>........] - ETA: 24s - loss: 0.2248 - categorical_accuracy: 0.9313
44608/60000 [=====================>........] - ETA: 24s - loss: 0.2247 - categorical_accuracy: 0.9313
44672/60000 [=====================>........] - ETA: 24s - loss: 0.2245 - categorical_accuracy: 0.9313
44704/60000 [=====================>........] - ETA: 24s - loss: 0.2244 - categorical_accuracy: 0.9314
44736/60000 [=====================>........] - ETA: 24s - loss: 0.2243 - categorical_accuracy: 0.9314
44768/60000 [=====================>........] - ETA: 24s - loss: 0.2242 - categorical_accuracy: 0.9314
44832/60000 [=====================>........] - ETA: 24s - loss: 0.2241 - categorical_accuracy: 0.9315
44864/60000 [=====================>........] - ETA: 24s - loss: 0.2240 - categorical_accuracy: 0.9315
44896/60000 [=====================>........] - ETA: 24s - loss: 0.2240 - categorical_accuracy: 0.9315
44960/60000 [=====================>........] - ETA: 24s - loss: 0.2239 - categorical_accuracy: 0.9316
45024/60000 [=====================>........] - ETA: 24s - loss: 0.2239 - categorical_accuracy: 0.9315
45056/60000 [=====================>........] - ETA: 24s - loss: 0.2238 - categorical_accuracy: 0.9316
45120/60000 [=====================>........] - ETA: 24s - loss: 0.2236 - categorical_accuracy: 0.9316
45152/60000 [=====================>........] - ETA: 23s - loss: 0.2235 - categorical_accuracy: 0.9317
45184/60000 [=====================>........] - ETA: 23s - loss: 0.2233 - categorical_accuracy: 0.9317
45216/60000 [=====================>........] - ETA: 23s - loss: 0.2232 - categorical_accuracy: 0.9317
45280/60000 [=====================>........] - ETA: 23s - loss: 0.2229 - categorical_accuracy: 0.9318
45344/60000 [=====================>........] - ETA: 23s - loss: 0.2227 - categorical_accuracy: 0.9319
45408/60000 [=====================>........] - ETA: 23s - loss: 0.2226 - categorical_accuracy: 0.9320
45472/60000 [=====================>........] - ETA: 23s - loss: 0.2223 - categorical_accuracy: 0.9320
45504/60000 [=====================>........] - ETA: 23s - loss: 0.2222 - categorical_accuracy: 0.9321
45536/60000 [=====================>........] - ETA: 23s - loss: 0.2222 - categorical_accuracy: 0.9321
45600/60000 [=====================>........] - ETA: 23s - loss: 0.2222 - categorical_accuracy: 0.9321
45664/60000 [=====================>........] - ETA: 23s - loss: 0.2219 - categorical_accuracy: 0.9322
45696/60000 [=====================>........] - ETA: 23s - loss: 0.2218 - categorical_accuracy: 0.9322
45760/60000 [=====================>........] - ETA: 22s - loss: 0.2216 - categorical_accuracy: 0.9323
45792/60000 [=====================>........] - ETA: 22s - loss: 0.2215 - categorical_accuracy: 0.9323
45824/60000 [=====================>........] - ETA: 22s - loss: 0.2214 - categorical_accuracy: 0.9323
45856/60000 [=====================>........] - ETA: 22s - loss: 0.2213 - categorical_accuracy: 0.9323
45888/60000 [=====================>........] - ETA: 22s - loss: 0.2212 - categorical_accuracy: 0.9324
45920/60000 [=====================>........] - ETA: 22s - loss: 0.2211 - categorical_accuracy: 0.9324
45952/60000 [=====================>........] - ETA: 22s - loss: 0.2210 - categorical_accuracy: 0.9324
45984/60000 [=====================>........] - ETA: 22s - loss: 0.2209 - categorical_accuracy: 0.9325
46016/60000 [======================>.......] - ETA: 22s - loss: 0.2207 - categorical_accuracy: 0.9325
46048/60000 [======================>.......] - ETA: 22s - loss: 0.2206 - categorical_accuracy: 0.9325
46080/60000 [======================>.......] - ETA: 22s - loss: 0.2206 - categorical_accuracy: 0.9325
46112/60000 [======================>.......] - ETA: 22s - loss: 0.2206 - categorical_accuracy: 0.9325
46144/60000 [======================>.......] - ETA: 22s - loss: 0.2205 - categorical_accuracy: 0.9326
46176/60000 [======================>.......] - ETA: 22s - loss: 0.2205 - categorical_accuracy: 0.9326
46208/60000 [======================>.......] - ETA: 22s - loss: 0.2204 - categorical_accuracy: 0.9326
46272/60000 [======================>.......] - ETA: 22s - loss: 0.2203 - categorical_accuracy: 0.9326
46304/60000 [======================>.......] - ETA: 22s - loss: 0.2202 - categorical_accuracy: 0.9327
46368/60000 [======================>.......] - ETA: 22s - loss: 0.2200 - categorical_accuracy: 0.9328
46400/60000 [======================>.......] - ETA: 21s - loss: 0.2199 - categorical_accuracy: 0.9328
46432/60000 [======================>.......] - ETA: 21s - loss: 0.2198 - categorical_accuracy: 0.9328
46464/60000 [======================>.......] - ETA: 21s - loss: 0.2197 - categorical_accuracy: 0.9329
46496/60000 [======================>.......] - ETA: 21s - loss: 0.2196 - categorical_accuracy: 0.9329
46528/60000 [======================>.......] - ETA: 21s - loss: 0.2196 - categorical_accuracy: 0.9329
46560/60000 [======================>.......] - ETA: 21s - loss: 0.2195 - categorical_accuracy: 0.9329
46592/60000 [======================>.......] - ETA: 21s - loss: 0.2194 - categorical_accuracy: 0.9329
46624/60000 [======================>.......] - ETA: 21s - loss: 0.2192 - categorical_accuracy: 0.9330
46656/60000 [======================>.......] - ETA: 21s - loss: 0.2191 - categorical_accuracy: 0.9330
46688/60000 [======================>.......] - ETA: 21s - loss: 0.2190 - categorical_accuracy: 0.9330
46720/60000 [======================>.......] - ETA: 21s - loss: 0.2189 - categorical_accuracy: 0.9330
46752/60000 [======================>.......] - ETA: 21s - loss: 0.2188 - categorical_accuracy: 0.9331
46784/60000 [======================>.......] - ETA: 21s - loss: 0.2187 - categorical_accuracy: 0.9331
46816/60000 [======================>.......] - ETA: 21s - loss: 0.2187 - categorical_accuracy: 0.9331
46848/60000 [======================>.......] - ETA: 21s - loss: 0.2186 - categorical_accuracy: 0.9331
46912/60000 [======================>.......] - ETA: 21s - loss: 0.2186 - categorical_accuracy: 0.9332
46944/60000 [======================>.......] - ETA: 21s - loss: 0.2184 - categorical_accuracy: 0.9332
46976/60000 [======================>.......] - ETA: 21s - loss: 0.2184 - categorical_accuracy: 0.9333
47040/60000 [======================>.......] - ETA: 20s - loss: 0.2181 - categorical_accuracy: 0.9333
47072/60000 [======================>.......] - ETA: 20s - loss: 0.2182 - categorical_accuracy: 0.9333
47136/60000 [======================>.......] - ETA: 20s - loss: 0.2181 - categorical_accuracy: 0.9334
47168/60000 [======================>.......] - ETA: 20s - loss: 0.2180 - categorical_accuracy: 0.9334
47200/60000 [======================>.......] - ETA: 20s - loss: 0.2178 - categorical_accuracy: 0.9334
47232/60000 [======================>.......] - ETA: 20s - loss: 0.2177 - categorical_accuracy: 0.9335
47264/60000 [======================>.......] - ETA: 20s - loss: 0.2176 - categorical_accuracy: 0.9335
47296/60000 [======================>.......] - ETA: 20s - loss: 0.2175 - categorical_accuracy: 0.9335
47328/60000 [======================>.......] - ETA: 20s - loss: 0.2174 - categorical_accuracy: 0.9335
47392/60000 [======================>.......] - ETA: 20s - loss: 0.2172 - categorical_accuracy: 0.9336
47424/60000 [======================>.......] - ETA: 20s - loss: 0.2171 - categorical_accuracy: 0.9337
47456/60000 [======================>.......] - ETA: 20s - loss: 0.2171 - categorical_accuracy: 0.9337
47488/60000 [======================>.......] - ETA: 20s - loss: 0.2171 - categorical_accuracy: 0.9337
47552/60000 [======================>.......] - ETA: 20s - loss: 0.2169 - categorical_accuracy: 0.9337
47584/60000 [======================>.......] - ETA: 20s - loss: 0.2168 - categorical_accuracy: 0.9337
47616/60000 [======================>.......] - ETA: 19s - loss: 0.2167 - categorical_accuracy: 0.9338
47648/60000 [======================>.......] - ETA: 19s - loss: 0.2166 - categorical_accuracy: 0.9337
47680/60000 [======================>.......] - ETA: 19s - loss: 0.2166 - categorical_accuracy: 0.9338
47712/60000 [======================>.......] - ETA: 19s - loss: 0.2165 - categorical_accuracy: 0.9338
47744/60000 [======================>.......] - ETA: 19s - loss: 0.2163 - categorical_accuracy: 0.9339
47808/60000 [======================>.......] - ETA: 19s - loss: 0.2163 - categorical_accuracy: 0.9339
47872/60000 [======================>.......] - ETA: 19s - loss: 0.2161 - categorical_accuracy: 0.9339
47936/60000 [======================>.......] - ETA: 19s - loss: 0.2159 - categorical_accuracy: 0.9340
47968/60000 [======================>.......] - ETA: 19s - loss: 0.2158 - categorical_accuracy: 0.9340
48000/60000 [=======================>......] - ETA: 19s - loss: 0.2156 - categorical_accuracy: 0.9341
48032/60000 [=======================>......] - ETA: 19s - loss: 0.2155 - categorical_accuracy: 0.9341
48096/60000 [=======================>......] - ETA: 19s - loss: 0.2155 - categorical_accuracy: 0.9341
48128/60000 [=======================>......] - ETA: 19s - loss: 0.2153 - categorical_accuracy: 0.9342
48160/60000 [=======================>......] - ETA: 19s - loss: 0.2152 - categorical_accuracy: 0.9342
48192/60000 [=======================>......] - ETA: 19s - loss: 0.2151 - categorical_accuracy: 0.9342
48256/60000 [=======================>......] - ETA: 18s - loss: 0.2149 - categorical_accuracy: 0.9343
48288/60000 [=======================>......] - ETA: 18s - loss: 0.2148 - categorical_accuracy: 0.9343
48320/60000 [=======================>......] - ETA: 18s - loss: 0.2147 - categorical_accuracy: 0.9344
48352/60000 [=======================>......] - ETA: 18s - loss: 0.2147 - categorical_accuracy: 0.9344
48384/60000 [=======================>......] - ETA: 18s - loss: 0.2148 - categorical_accuracy: 0.9344
48416/60000 [=======================>......] - ETA: 18s - loss: 0.2147 - categorical_accuracy: 0.9344
48448/60000 [=======================>......] - ETA: 18s - loss: 0.2146 - categorical_accuracy: 0.9344
48512/60000 [=======================>......] - ETA: 18s - loss: 0.2144 - categorical_accuracy: 0.9345
48576/60000 [=======================>......] - ETA: 18s - loss: 0.2143 - categorical_accuracy: 0.9345
48608/60000 [=======================>......] - ETA: 18s - loss: 0.2143 - categorical_accuracy: 0.9345
48640/60000 [=======================>......] - ETA: 18s - loss: 0.2141 - categorical_accuracy: 0.9346
48672/60000 [=======================>......] - ETA: 18s - loss: 0.2141 - categorical_accuracy: 0.9346
48704/60000 [=======================>......] - ETA: 18s - loss: 0.2139 - categorical_accuracy: 0.9346
48736/60000 [=======================>......] - ETA: 18s - loss: 0.2138 - categorical_accuracy: 0.9347
48800/60000 [=======================>......] - ETA: 18s - loss: 0.2137 - categorical_accuracy: 0.9347
48832/60000 [=======================>......] - ETA: 18s - loss: 0.2136 - categorical_accuracy: 0.9348
48864/60000 [=======================>......] - ETA: 17s - loss: 0.2135 - categorical_accuracy: 0.9348
48928/60000 [=======================>......] - ETA: 17s - loss: 0.2135 - categorical_accuracy: 0.9348
48960/60000 [=======================>......] - ETA: 17s - loss: 0.2136 - categorical_accuracy: 0.9348
48992/60000 [=======================>......] - ETA: 17s - loss: 0.2135 - categorical_accuracy: 0.9348
49024/60000 [=======================>......] - ETA: 17s - loss: 0.2134 - categorical_accuracy: 0.9348
49088/60000 [=======================>......] - ETA: 17s - loss: 0.2131 - categorical_accuracy: 0.9349
49120/60000 [=======================>......] - ETA: 17s - loss: 0.2130 - categorical_accuracy: 0.9350
49152/60000 [=======================>......] - ETA: 17s - loss: 0.2130 - categorical_accuracy: 0.9350
49184/60000 [=======================>......] - ETA: 17s - loss: 0.2129 - categorical_accuracy: 0.9350
49216/60000 [=======================>......] - ETA: 17s - loss: 0.2128 - categorical_accuracy: 0.9350
49248/60000 [=======================>......] - ETA: 17s - loss: 0.2127 - categorical_accuracy: 0.9351
49280/60000 [=======================>......] - ETA: 17s - loss: 0.2127 - categorical_accuracy: 0.9351
49312/60000 [=======================>......] - ETA: 17s - loss: 0.2126 - categorical_accuracy: 0.9351
49344/60000 [=======================>......] - ETA: 17s - loss: 0.2125 - categorical_accuracy: 0.9351
49376/60000 [=======================>......] - ETA: 17s - loss: 0.2124 - categorical_accuracy: 0.9352
49440/60000 [=======================>......] - ETA: 17s - loss: 0.2124 - categorical_accuracy: 0.9352
49504/60000 [=======================>......] - ETA: 16s - loss: 0.2123 - categorical_accuracy: 0.9352
49568/60000 [=======================>......] - ETA: 16s - loss: 0.2122 - categorical_accuracy: 0.9352
49600/60000 [=======================>......] - ETA: 16s - loss: 0.2120 - categorical_accuracy: 0.9353
49632/60000 [=======================>......] - ETA: 16s - loss: 0.2120 - categorical_accuracy: 0.9353
49664/60000 [=======================>......] - ETA: 16s - loss: 0.2119 - categorical_accuracy: 0.9353
49696/60000 [=======================>......] - ETA: 16s - loss: 0.2119 - categorical_accuracy: 0.9353
49728/60000 [=======================>......] - ETA: 16s - loss: 0.2118 - categorical_accuracy: 0.9354
49760/60000 [=======================>......] - ETA: 16s - loss: 0.2117 - categorical_accuracy: 0.9354
49824/60000 [=======================>......] - ETA: 16s - loss: 0.2115 - categorical_accuracy: 0.9355
49856/60000 [=======================>......] - ETA: 16s - loss: 0.2115 - categorical_accuracy: 0.9355
49920/60000 [=======================>......] - ETA: 16s - loss: 0.2113 - categorical_accuracy: 0.9355
49984/60000 [=======================>......] - ETA: 16s - loss: 0.2112 - categorical_accuracy: 0.9356
50016/60000 [========================>.....] - ETA: 16s - loss: 0.2112 - categorical_accuracy: 0.9355
50048/60000 [========================>.....] - ETA: 16s - loss: 0.2111 - categorical_accuracy: 0.9356
50080/60000 [========================>.....] - ETA: 15s - loss: 0.2111 - categorical_accuracy: 0.9356
50112/60000 [========================>.....] - ETA: 15s - loss: 0.2110 - categorical_accuracy: 0.9356
50144/60000 [========================>.....] - ETA: 15s - loss: 0.2109 - categorical_accuracy: 0.9356
50176/60000 [========================>.....] - ETA: 15s - loss: 0.2108 - categorical_accuracy: 0.9356
50208/60000 [========================>.....] - ETA: 15s - loss: 0.2107 - categorical_accuracy: 0.9357
50240/60000 [========================>.....] - ETA: 15s - loss: 0.2106 - categorical_accuracy: 0.9357
50272/60000 [========================>.....] - ETA: 15s - loss: 0.2106 - categorical_accuracy: 0.9357
50336/60000 [========================>.....] - ETA: 15s - loss: 0.2106 - categorical_accuracy: 0.9357
50368/60000 [========================>.....] - ETA: 15s - loss: 0.2106 - categorical_accuracy: 0.9357
50400/60000 [========================>.....] - ETA: 15s - loss: 0.2104 - categorical_accuracy: 0.9357
50464/60000 [========================>.....] - ETA: 15s - loss: 0.2103 - categorical_accuracy: 0.9358
50496/60000 [========================>.....] - ETA: 15s - loss: 0.2102 - categorical_accuracy: 0.9358
50560/60000 [========================>.....] - ETA: 15s - loss: 0.2101 - categorical_accuracy: 0.9358
50592/60000 [========================>.....] - ETA: 15s - loss: 0.2101 - categorical_accuracy: 0.9358
50624/60000 [========================>.....] - ETA: 15s - loss: 0.2101 - categorical_accuracy: 0.9358
50656/60000 [========================>.....] - ETA: 15s - loss: 0.2101 - categorical_accuracy: 0.9359
50688/60000 [========================>.....] - ETA: 15s - loss: 0.2100 - categorical_accuracy: 0.9359
50720/60000 [========================>.....] - ETA: 14s - loss: 0.2099 - categorical_accuracy: 0.9359
50752/60000 [========================>.....] - ETA: 14s - loss: 0.2099 - categorical_accuracy: 0.9359
50784/60000 [========================>.....] - ETA: 14s - loss: 0.2098 - categorical_accuracy: 0.9359
50816/60000 [========================>.....] - ETA: 14s - loss: 0.2097 - categorical_accuracy: 0.9360
50848/60000 [========================>.....] - ETA: 14s - loss: 0.2096 - categorical_accuracy: 0.9360
50880/60000 [========================>.....] - ETA: 14s - loss: 0.2095 - categorical_accuracy: 0.9360
50912/60000 [========================>.....] - ETA: 14s - loss: 0.2095 - categorical_accuracy: 0.9360
50944/60000 [========================>.....] - ETA: 14s - loss: 0.2094 - categorical_accuracy: 0.9360
50976/60000 [========================>.....] - ETA: 14s - loss: 0.2092 - categorical_accuracy: 0.9361
51008/60000 [========================>.....] - ETA: 14s - loss: 0.2091 - categorical_accuracy: 0.9361
51040/60000 [========================>.....] - ETA: 14s - loss: 0.2091 - categorical_accuracy: 0.9361
51072/60000 [========================>.....] - ETA: 14s - loss: 0.2090 - categorical_accuracy: 0.9362
51104/60000 [========================>.....] - ETA: 14s - loss: 0.2089 - categorical_accuracy: 0.9362
51136/60000 [========================>.....] - ETA: 14s - loss: 0.2088 - categorical_accuracy: 0.9362
51168/60000 [========================>.....] - ETA: 14s - loss: 0.2087 - categorical_accuracy: 0.9362
51200/60000 [========================>.....] - ETA: 14s - loss: 0.2087 - categorical_accuracy: 0.9362
51232/60000 [========================>.....] - ETA: 14s - loss: 0.2087 - categorical_accuracy: 0.9362
51264/60000 [========================>.....] - ETA: 14s - loss: 0.2085 - categorical_accuracy: 0.9363
51296/60000 [========================>.....] - ETA: 14s - loss: 0.2087 - categorical_accuracy: 0.9363
51360/60000 [========================>.....] - ETA: 13s - loss: 0.2085 - categorical_accuracy: 0.9363
51392/60000 [========================>.....] - ETA: 13s - loss: 0.2084 - categorical_accuracy: 0.9364
51424/60000 [========================>.....] - ETA: 13s - loss: 0.2083 - categorical_accuracy: 0.9364
51488/60000 [========================>.....] - ETA: 13s - loss: 0.2082 - categorical_accuracy: 0.9365
51520/60000 [========================>.....] - ETA: 13s - loss: 0.2083 - categorical_accuracy: 0.9365
51584/60000 [========================>.....] - ETA: 13s - loss: 0.2081 - categorical_accuracy: 0.9365
51616/60000 [========================>.....] - ETA: 13s - loss: 0.2080 - categorical_accuracy: 0.9365
51648/60000 [========================>.....] - ETA: 13s - loss: 0.2080 - categorical_accuracy: 0.9365
51680/60000 [========================>.....] - ETA: 13s - loss: 0.2079 - categorical_accuracy: 0.9366
51712/60000 [========================>.....] - ETA: 13s - loss: 0.2077 - categorical_accuracy: 0.9366
51744/60000 [========================>.....] - ETA: 13s - loss: 0.2077 - categorical_accuracy: 0.9366
51776/60000 [========================>.....] - ETA: 13s - loss: 0.2076 - categorical_accuracy: 0.9367
51808/60000 [========================>.....] - ETA: 13s - loss: 0.2075 - categorical_accuracy: 0.9367
51840/60000 [========================>.....] - ETA: 13s - loss: 0.2074 - categorical_accuracy: 0.9367
51872/60000 [========================>.....] - ETA: 13s - loss: 0.2073 - categorical_accuracy: 0.9367
51904/60000 [========================>.....] - ETA: 13s - loss: 0.2072 - categorical_accuracy: 0.9367
51936/60000 [========================>.....] - ETA: 13s - loss: 0.2071 - categorical_accuracy: 0.9368
51968/60000 [========================>.....] - ETA: 12s - loss: 0.2071 - categorical_accuracy: 0.9368
52000/60000 [=========================>....] - ETA: 12s - loss: 0.2070 - categorical_accuracy: 0.9368
52032/60000 [=========================>....] - ETA: 12s - loss: 0.2070 - categorical_accuracy: 0.9369
52064/60000 [=========================>....] - ETA: 12s - loss: 0.2069 - categorical_accuracy: 0.9369
52096/60000 [=========================>....] - ETA: 12s - loss: 0.2068 - categorical_accuracy: 0.9369
52128/60000 [=========================>....] - ETA: 12s - loss: 0.2067 - categorical_accuracy: 0.9370
52160/60000 [=========================>....] - ETA: 12s - loss: 0.2068 - categorical_accuracy: 0.9370
52192/60000 [=========================>....] - ETA: 12s - loss: 0.2069 - categorical_accuracy: 0.9370
52224/60000 [=========================>....] - ETA: 12s - loss: 0.2069 - categorical_accuracy: 0.9370
52288/60000 [=========================>....] - ETA: 12s - loss: 0.2069 - categorical_accuracy: 0.9370
52320/60000 [=========================>....] - ETA: 12s - loss: 0.2068 - categorical_accuracy: 0.9370
52352/60000 [=========================>....] - ETA: 12s - loss: 0.2067 - categorical_accuracy: 0.9370
52384/60000 [=========================>....] - ETA: 12s - loss: 0.2066 - categorical_accuracy: 0.9370
52416/60000 [=========================>....] - ETA: 12s - loss: 0.2065 - categorical_accuracy: 0.9371
52448/60000 [=========================>....] - ETA: 12s - loss: 0.2066 - categorical_accuracy: 0.9370
52480/60000 [=========================>....] - ETA: 12s - loss: 0.2066 - categorical_accuracy: 0.9370
52512/60000 [=========================>....] - ETA: 12s - loss: 0.2066 - categorical_accuracy: 0.9370
52544/60000 [=========================>....] - ETA: 12s - loss: 0.2067 - categorical_accuracy: 0.9370
52576/60000 [=========================>....] - ETA: 11s - loss: 0.2066 - categorical_accuracy: 0.9370
52608/60000 [=========================>....] - ETA: 11s - loss: 0.2065 - categorical_accuracy: 0.9371
52672/60000 [=========================>....] - ETA: 11s - loss: 0.2065 - categorical_accuracy: 0.9371
52704/60000 [=========================>....] - ETA: 11s - loss: 0.2065 - categorical_accuracy: 0.9371
52736/60000 [=========================>....] - ETA: 11s - loss: 0.2065 - categorical_accuracy: 0.9371
52768/60000 [=========================>....] - ETA: 11s - loss: 0.2064 - categorical_accuracy: 0.9372
52800/60000 [=========================>....] - ETA: 11s - loss: 0.2063 - categorical_accuracy: 0.9372
52864/60000 [=========================>....] - ETA: 11s - loss: 0.2062 - categorical_accuracy: 0.9372
52896/60000 [=========================>....] - ETA: 11s - loss: 0.2061 - categorical_accuracy: 0.9373
52928/60000 [=========================>....] - ETA: 11s - loss: 0.2061 - categorical_accuracy: 0.9373
52960/60000 [=========================>....] - ETA: 11s - loss: 0.2060 - categorical_accuracy: 0.9373
52992/60000 [=========================>....] - ETA: 11s - loss: 0.2059 - categorical_accuracy: 0.9373
53024/60000 [=========================>....] - ETA: 11s - loss: 0.2058 - categorical_accuracy: 0.9373
53056/60000 [=========================>....] - ETA: 11s - loss: 0.2057 - categorical_accuracy: 0.9374
53088/60000 [=========================>....] - ETA: 11s - loss: 0.2056 - categorical_accuracy: 0.9374
53120/60000 [=========================>....] - ETA: 11s - loss: 0.2056 - categorical_accuracy: 0.9374
53152/60000 [=========================>....] - ETA: 11s - loss: 0.2055 - categorical_accuracy: 0.9374
53184/60000 [=========================>....] - ETA: 10s - loss: 0.2054 - categorical_accuracy: 0.9374
53248/60000 [=========================>....] - ETA: 10s - loss: 0.2055 - categorical_accuracy: 0.9374
53280/60000 [=========================>....] - ETA: 10s - loss: 0.2054 - categorical_accuracy: 0.9375
53312/60000 [=========================>....] - ETA: 10s - loss: 0.2053 - categorical_accuracy: 0.9375
53376/60000 [=========================>....] - ETA: 10s - loss: 0.2051 - categorical_accuracy: 0.9376
53408/60000 [=========================>....] - ETA: 10s - loss: 0.2051 - categorical_accuracy: 0.9376
53440/60000 [=========================>....] - ETA: 10s - loss: 0.2050 - categorical_accuracy: 0.9376
53504/60000 [=========================>....] - ETA: 10s - loss: 0.2049 - categorical_accuracy: 0.9376
53568/60000 [=========================>....] - ETA: 10s - loss: 0.2047 - categorical_accuracy: 0.9377
53632/60000 [=========================>....] - ETA: 10s - loss: 0.2047 - categorical_accuracy: 0.9377
53664/60000 [=========================>....] - ETA: 10s - loss: 0.2046 - categorical_accuracy: 0.9377
53696/60000 [=========================>....] - ETA: 10s - loss: 0.2045 - categorical_accuracy: 0.9377
53728/60000 [=========================>....] - ETA: 10s - loss: 0.2045 - categorical_accuracy: 0.9378
53760/60000 [=========================>....] - ETA: 10s - loss: 0.2045 - categorical_accuracy: 0.9378
53792/60000 [=========================>....] - ETA: 10s - loss: 0.2044 - categorical_accuracy: 0.9378
53856/60000 [=========================>....] - ETA: 9s - loss: 0.2044 - categorical_accuracy: 0.9378 
53888/60000 [=========================>....] - ETA: 9s - loss: 0.2043 - categorical_accuracy: 0.9379
53920/60000 [=========================>....] - ETA: 9s - loss: 0.2042 - categorical_accuracy: 0.9379
53952/60000 [=========================>....] - ETA: 9s - loss: 0.2042 - categorical_accuracy: 0.9379
53984/60000 [=========================>....] - ETA: 9s - loss: 0.2042 - categorical_accuracy: 0.9379
54016/60000 [==========================>...] - ETA: 9s - loss: 0.2041 - categorical_accuracy: 0.9379
54048/60000 [==========================>...] - ETA: 9s - loss: 0.2040 - categorical_accuracy: 0.9380
54112/60000 [==========================>...] - ETA: 9s - loss: 0.2039 - categorical_accuracy: 0.9379
54144/60000 [==========================>...] - ETA: 9s - loss: 0.2038 - categorical_accuracy: 0.9380
54176/60000 [==========================>...] - ETA: 9s - loss: 0.2037 - categorical_accuracy: 0.9380
54208/60000 [==========================>...] - ETA: 9s - loss: 0.2036 - categorical_accuracy: 0.9381
54240/60000 [==========================>...] - ETA: 9s - loss: 0.2037 - categorical_accuracy: 0.9381
54272/60000 [==========================>...] - ETA: 9s - loss: 0.2036 - categorical_accuracy: 0.9381
54304/60000 [==========================>...] - ETA: 9s - loss: 0.2037 - categorical_accuracy: 0.9380
54336/60000 [==========================>...] - ETA: 9s - loss: 0.2036 - categorical_accuracy: 0.9381
54368/60000 [==========================>...] - ETA: 9s - loss: 0.2035 - categorical_accuracy: 0.9381
54400/60000 [==========================>...] - ETA: 9s - loss: 0.2034 - categorical_accuracy: 0.9381
54432/60000 [==========================>...] - ETA: 8s - loss: 0.2033 - categorical_accuracy: 0.9381
54464/60000 [==========================>...] - ETA: 8s - loss: 0.2033 - categorical_accuracy: 0.9381
54496/60000 [==========================>...] - ETA: 8s - loss: 0.2032 - categorical_accuracy: 0.9382
54560/60000 [==========================>...] - ETA: 8s - loss: 0.2030 - categorical_accuracy: 0.9382
54592/60000 [==========================>...] - ETA: 8s - loss: 0.2029 - categorical_accuracy: 0.9382
54656/60000 [==========================>...] - ETA: 8s - loss: 0.2028 - categorical_accuracy: 0.9383
54688/60000 [==========================>...] - ETA: 8s - loss: 0.2027 - categorical_accuracy: 0.9383
54720/60000 [==========================>...] - ETA: 8s - loss: 0.2027 - categorical_accuracy: 0.9383
54752/60000 [==========================>...] - ETA: 8s - loss: 0.2026 - categorical_accuracy: 0.9383
54816/60000 [==========================>...] - ETA: 8s - loss: 0.2026 - categorical_accuracy: 0.9383
54880/60000 [==========================>...] - ETA: 8s - loss: 0.2025 - categorical_accuracy: 0.9383
54912/60000 [==========================>...] - ETA: 8s - loss: 0.2026 - categorical_accuracy: 0.9383
54944/60000 [==========================>...] - ETA: 8s - loss: 0.2026 - categorical_accuracy: 0.9384
54976/60000 [==========================>...] - ETA: 8s - loss: 0.2025 - categorical_accuracy: 0.9384
55008/60000 [==========================>...] - ETA: 8s - loss: 0.2024 - categorical_accuracy: 0.9384
55040/60000 [==========================>...] - ETA: 7s - loss: 0.2023 - categorical_accuracy: 0.9384
55072/60000 [==========================>...] - ETA: 7s - loss: 0.2022 - categorical_accuracy: 0.9385
55104/60000 [==========================>...] - ETA: 7s - loss: 0.2021 - categorical_accuracy: 0.9385
55136/60000 [==========================>...] - ETA: 7s - loss: 0.2020 - categorical_accuracy: 0.9385
55168/60000 [==========================>...] - ETA: 7s - loss: 0.2021 - categorical_accuracy: 0.9385
55200/60000 [==========================>...] - ETA: 7s - loss: 0.2020 - categorical_accuracy: 0.9385
55232/60000 [==========================>...] - ETA: 7s - loss: 0.2019 - categorical_accuracy: 0.9386
55264/60000 [==========================>...] - ETA: 7s - loss: 0.2018 - categorical_accuracy: 0.9386
55296/60000 [==========================>...] - ETA: 7s - loss: 0.2018 - categorical_accuracy: 0.9386
55360/60000 [==========================>...] - ETA: 7s - loss: 0.2016 - categorical_accuracy: 0.9386
55424/60000 [==========================>...] - ETA: 7s - loss: 0.2018 - categorical_accuracy: 0.9386
55456/60000 [==========================>...] - ETA: 7s - loss: 0.2017 - categorical_accuracy: 0.9386
55488/60000 [==========================>...] - ETA: 7s - loss: 0.2016 - categorical_accuracy: 0.9387
55552/60000 [==========================>...] - ETA: 7s - loss: 0.2015 - categorical_accuracy: 0.9387
55584/60000 [==========================>...] - ETA: 7s - loss: 0.2015 - categorical_accuracy: 0.9387
55616/60000 [==========================>...] - ETA: 7s - loss: 0.2014 - categorical_accuracy: 0.9388
55648/60000 [==========================>...] - ETA: 7s - loss: 0.2013 - categorical_accuracy: 0.9388
55712/60000 [==========================>...] - ETA: 6s - loss: 0.2011 - categorical_accuracy: 0.9388
55744/60000 [==========================>...] - ETA: 6s - loss: 0.2010 - categorical_accuracy: 0.9389
55776/60000 [==========================>...] - ETA: 6s - loss: 0.2009 - categorical_accuracy: 0.9389
55808/60000 [==========================>...] - ETA: 6s - loss: 0.2009 - categorical_accuracy: 0.9389
55840/60000 [==========================>...] - ETA: 6s - loss: 0.2008 - categorical_accuracy: 0.9389
55872/60000 [==========================>...] - ETA: 6s - loss: 0.2007 - categorical_accuracy: 0.9390
55936/60000 [==========================>...] - ETA: 6s - loss: 0.2005 - categorical_accuracy: 0.9390
55968/60000 [==========================>...] - ETA: 6s - loss: 0.2004 - categorical_accuracy: 0.9390
56032/60000 [===========================>..] - ETA: 6s - loss: 0.2002 - categorical_accuracy: 0.9391
56096/60000 [===========================>..] - ETA: 6s - loss: 0.2001 - categorical_accuracy: 0.9391
56128/60000 [===========================>..] - ETA: 6s - loss: 0.2000 - categorical_accuracy: 0.9391
56160/60000 [===========================>..] - ETA: 6s - loss: 0.1999 - categorical_accuracy: 0.9392
56192/60000 [===========================>..] - ETA: 6s - loss: 0.1998 - categorical_accuracy: 0.9392
56224/60000 [===========================>..] - ETA: 6s - loss: 0.1997 - categorical_accuracy: 0.9392
56256/60000 [===========================>..] - ETA: 6s - loss: 0.1997 - categorical_accuracy: 0.9392
56288/60000 [===========================>..] - ETA: 5s - loss: 0.1999 - categorical_accuracy: 0.9392
56352/60000 [===========================>..] - ETA: 5s - loss: 0.1998 - categorical_accuracy: 0.9393
56384/60000 [===========================>..] - ETA: 5s - loss: 0.1997 - categorical_accuracy: 0.9393
56416/60000 [===========================>..] - ETA: 5s - loss: 0.1996 - categorical_accuracy: 0.9393
56448/60000 [===========================>..] - ETA: 5s - loss: 0.1996 - categorical_accuracy: 0.9393
56480/60000 [===========================>..] - ETA: 5s - loss: 0.1997 - categorical_accuracy: 0.9393
56512/60000 [===========================>..] - ETA: 5s - loss: 0.1996 - categorical_accuracy: 0.9394
56544/60000 [===========================>..] - ETA: 5s - loss: 0.1995 - categorical_accuracy: 0.9394
56576/60000 [===========================>..] - ETA: 5s - loss: 0.1994 - categorical_accuracy: 0.9394
56608/60000 [===========================>..] - ETA: 5s - loss: 0.1994 - categorical_accuracy: 0.9395
56672/60000 [===========================>..] - ETA: 5s - loss: 0.1994 - categorical_accuracy: 0.9395
56736/60000 [===========================>..] - ETA: 5s - loss: 0.1993 - categorical_accuracy: 0.9395
56768/60000 [===========================>..] - ETA: 5s - loss: 0.1993 - categorical_accuracy: 0.9395
56800/60000 [===========================>..] - ETA: 5s - loss: 0.1993 - categorical_accuracy: 0.9395
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1992 - categorical_accuracy: 0.9395
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1991 - categorical_accuracy: 0.9396
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1990 - categorical_accuracy: 0.9396
56928/60000 [===========================>..] - ETA: 4s - loss: 0.1989 - categorical_accuracy: 0.9396
56992/60000 [===========================>..] - ETA: 4s - loss: 0.1987 - categorical_accuracy: 0.9397
57024/60000 [===========================>..] - ETA: 4s - loss: 0.1987 - categorical_accuracy: 0.9397
57056/60000 [===========================>..] - ETA: 4s - loss: 0.1986 - categorical_accuracy: 0.9397
57120/60000 [===========================>..] - ETA: 4s - loss: 0.1984 - categorical_accuracy: 0.9398
57184/60000 [===========================>..] - ETA: 4s - loss: 0.1984 - categorical_accuracy: 0.9398
57216/60000 [===========================>..] - ETA: 4s - loss: 0.1983 - categorical_accuracy: 0.9398
57248/60000 [===========================>..] - ETA: 4s - loss: 0.1983 - categorical_accuracy: 0.9398
57280/60000 [===========================>..] - ETA: 4s - loss: 0.1982 - categorical_accuracy: 0.9399
57312/60000 [===========================>..] - ETA: 4s - loss: 0.1982 - categorical_accuracy: 0.9399
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1981 - categorical_accuracy: 0.9399
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1983 - categorical_accuracy: 0.9399
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1982 - categorical_accuracy: 0.9399
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1980 - categorical_accuracy: 0.9400
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1979 - categorical_accuracy: 0.9400
57536/60000 [===========================>..] - ETA: 3s - loss: 0.1979 - categorical_accuracy: 0.9400
57600/60000 [===========================>..] - ETA: 3s - loss: 0.1977 - categorical_accuracy: 0.9400
57632/60000 [===========================>..] - ETA: 3s - loss: 0.1976 - categorical_accuracy: 0.9401
57664/60000 [===========================>..] - ETA: 3s - loss: 0.1976 - categorical_accuracy: 0.9401
57696/60000 [===========================>..] - ETA: 3s - loss: 0.1975 - categorical_accuracy: 0.9401
57760/60000 [===========================>..] - ETA: 3s - loss: 0.1974 - categorical_accuracy: 0.9401
57824/60000 [===========================>..] - ETA: 3s - loss: 0.1973 - categorical_accuracy: 0.9402
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1973 - categorical_accuracy: 0.9402
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1972 - categorical_accuracy: 0.9402
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1970 - categorical_accuracy: 0.9403
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1970 - categorical_accuracy: 0.9403
58016/60000 [============================>.] - ETA: 3s - loss: 0.1970 - categorical_accuracy: 0.9403
58080/60000 [============================>.] - ETA: 3s - loss: 0.1968 - categorical_accuracy: 0.9404
58144/60000 [============================>.] - ETA: 2s - loss: 0.1969 - categorical_accuracy: 0.9404
58176/60000 [============================>.] - ETA: 2s - loss: 0.1969 - categorical_accuracy: 0.9404
58208/60000 [============================>.] - ETA: 2s - loss: 0.1968 - categorical_accuracy: 0.9404
58240/60000 [============================>.] - ETA: 2s - loss: 0.1967 - categorical_accuracy: 0.9404
58272/60000 [============================>.] - ETA: 2s - loss: 0.1967 - categorical_accuracy: 0.9404
58304/60000 [============================>.] - ETA: 2s - loss: 0.1967 - categorical_accuracy: 0.9404
58336/60000 [============================>.] - ETA: 2s - loss: 0.1966 - categorical_accuracy: 0.9405
58368/60000 [============================>.] - ETA: 2s - loss: 0.1965 - categorical_accuracy: 0.9405
58400/60000 [============================>.] - ETA: 2s - loss: 0.1964 - categorical_accuracy: 0.9405
58432/60000 [============================>.] - ETA: 2s - loss: 0.1963 - categorical_accuracy: 0.9405
58464/60000 [============================>.] - ETA: 2s - loss: 0.1963 - categorical_accuracy: 0.9406
58528/60000 [============================>.] - ETA: 2s - loss: 0.1961 - categorical_accuracy: 0.9406
58592/60000 [============================>.] - ETA: 2s - loss: 0.1961 - categorical_accuracy: 0.9406
58656/60000 [============================>.] - ETA: 2s - loss: 0.1960 - categorical_accuracy: 0.9407
58720/60000 [============================>.] - ETA: 2s - loss: 0.1959 - categorical_accuracy: 0.9407
58752/60000 [============================>.] - ETA: 2s - loss: 0.1958 - categorical_accuracy: 0.9407
58784/60000 [============================>.] - ETA: 1s - loss: 0.1957 - categorical_accuracy: 0.9407
58848/60000 [============================>.] - ETA: 1s - loss: 0.1956 - categorical_accuracy: 0.9407
58912/60000 [============================>.] - ETA: 1s - loss: 0.1955 - categorical_accuracy: 0.9408
58944/60000 [============================>.] - ETA: 1s - loss: 0.1955 - categorical_accuracy: 0.9408
59008/60000 [============================>.] - ETA: 1s - loss: 0.1953 - categorical_accuracy: 0.9409
59072/60000 [============================>.] - ETA: 1s - loss: 0.1953 - categorical_accuracy: 0.9409
59104/60000 [============================>.] - ETA: 1s - loss: 0.1952 - categorical_accuracy: 0.9409
59168/60000 [============================>.] - ETA: 1s - loss: 0.1951 - categorical_accuracy: 0.9410
59232/60000 [============================>.] - ETA: 1s - loss: 0.1950 - categorical_accuracy: 0.9410
59264/60000 [============================>.] - ETA: 1s - loss: 0.1949 - categorical_accuracy: 0.9410
59296/60000 [============================>.] - ETA: 1s - loss: 0.1948 - categorical_accuracy: 0.9411
59328/60000 [============================>.] - ETA: 1s - loss: 0.1948 - categorical_accuracy: 0.9411
59360/60000 [============================>.] - ETA: 1s - loss: 0.1948 - categorical_accuracy: 0.9411
59392/60000 [============================>.] - ETA: 0s - loss: 0.1948 - categorical_accuracy: 0.9410
59424/60000 [============================>.] - ETA: 0s - loss: 0.1948 - categorical_accuracy: 0.9411
59456/60000 [============================>.] - ETA: 0s - loss: 0.1947 - categorical_accuracy: 0.9411
59488/60000 [============================>.] - ETA: 0s - loss: 0.1948 - categorical_accuracy: 0.9411
59520/60000 [============================>.] - ETA: 0s - loss: 0.1948 - categorical_accuracy: 0.9411
59552/60000 [============================>.] - ETA: 0s - loss: 0.1947 - categorical_accuracy: 0.9411
59616/60000 [============================>.] - ETA: 0s - loss: 0.1947 - categorical_accuracy: 0.9411
59680/60000 [============================>.] - ETA: 0s - loss: 0.1946 - categorical_accuracy: 0.9411
59712/60000 [============================>.] - ETA: 0s - loss: 0.1945 - categorical_accuracy: 0.9412
59744/60000 [============================>.] - ETA: 0s - loss: 0.1944 - categorical_accuracy: 0.9412
59808/60000 [============================>.] - ETA: 0s - loss: 0.1942 - categorical_accuracy: 0.9412
59872/60000 [============================>.] - ETA: 0s - loss: 0.1941 - categorical_accuracy: 0.9413
59904/60000 [============================>.] - ETA: 0s - loss: 0.1940 - categorical_accuracy: 0.9413
59936/60000 [============================>.] - ETA: 0s - loss: 0.1939 - categorical_accuracy: 0.9414
59968/60000 [============================>.] - ETA: 0s - loss: 0.1938 - categorical_accuracy: 0.9414
60000/60000 [==============================] - 100s 2ms/step - loss: 0.1938 - categorical_accuracy: 0.9414 - val_loss: 0.0517 - val_categorical_accuracy: 0.9839

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
 1664/10000 [===>..........................] - ETA: 2s
 1824/10000 [====>.........................] - ETA: 2s
 1984/10000 [====>.........................] - ETA: 2s
 2144/10000 [=====>........................] - ETA: 2s
 2336/10000 [======>.......................] - ETA: 2s
 2496/10000 [======>.......................] - ETA: 2s
 2656/10000 [======>.......................] - ETA: 2s
 2848/10000 [=======>......................] - ETA: 2s
 3040/10000 [========>.....................] - ETA: 2s
 3200/10000 [========>.....................] - ETA: 2s
 3360/10000 [=========>....................] - ETA: 2s
 3520/10000 [=========>....................] - ETA: 2s
 3712/10000 [==========>...................] - ETA: 2s
 3872/10000 [==========>...................] - ETA: 2s
 4032/10000 [===========>..................] - ETA: 1s
 4192/10000 [===========>..................] - ETA: 1s
 4384/10000 [============>.................] - ETA: 1s
 4544/10000 [============>.................] - ETA: 1s
 4704/10000 [=============>................] - ETA: 1s
 4864/10000 [=============>................] - ETA: 1s
 5024/10000 [==============>...............] - ETA: 1s
 5184/10000 [==============>...............] - ETA: 1s
 5344/10000 [===============>..............] - ETA: 1s
 5504/10000 [===============>..............] - ETA: 1s
 5664/10000 [===============>..............] - ETA: 1s
 5856/10000 [================>.............] - ETA: 1s
 6016/10000 [=================>............] - ETA: 1s
 6208/10000 [=================>............] - ETA: 1s
 6400/10000 [==================>...........] - ETA: 1s
 6560/10000 [==================>...........] - ETA: 1s
 6720/10000 [===================>..........] - ETA: 1s
 6880/10000 [===================>..........] - ETA: 1s
 7040/10000 [====================>.........] - ETA: 0s
 7200/10000 [====================>.........] - ETA: 0s
 7360/10000 [=====================>........] - ETA: 0s
 7520/10000 [=====================>........] - ETA: 0s
 7712/10000 [======================>.......] - ETA: 0s
 7904/10000 [======================>.......] - ETA: 0s
 8064/10000 [=======================>......] - ETA: 0s
 8224/10000 [=======================>......] - ETA: 0s
 8384/10000 [========================>.....] - ETA: 0s
 8544/10000 [========================>.....] - ETA: 0s
 8704/10000 [=========================>....] - ETA: 0s
 8896/10000 [=========================>....] - ETA: 0s
 9056/10000 [==========================>...] - ETA: 0s
 9248/10000 [==========================>...] - ETA: 0s
 9440/10000 [===========================>..] - ETA: 0s
 9600/10000 [===========================>..] - ETA: 0s
 9792/10000 [============================>.] - ETA: 0s
 9952/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 3s 326us/step
[[7.3025497e-08 3.1344079e-08 9.7775128e-06 ... 9.9998260e-01
  2.3535945e-07 4.9176147e-06]
 [7.0384162e-06 1.3459539e-05 9.9997497e-01 ... 6.6425789e-09
  2.8949776e-06 7.0340200e-10]
 [2.4543181e-06 9.9957484e-01 4.7364403e-05 ... 9.2420421e-05
  6.0946921e-05 3.8115302e-06]
 ...
 [2.4576612e-09 3.4445748e-07 1.1777670e-09 ... 2.5210268e-06
  1.2740310e-05 2.6155397e-04]
 [1.1882604e-05 7.6350199e-07 4.2969130e-08 ... 1.8629809e-06
  4.6952441e-03 8.1367079e-06]
 [4.3783108e-05 3.4243919e-07 1.2235118e-05 ... 2.4462268e-09
  1.1742113e-06 1.4812981e-07]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.05173122195682954, 'accuracy_test:': 0.9839000105857849}

  ('#### Save   #######################################################',) 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/charcnn/result'}

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
[master abbb5bd] ml_store
 1 file changed, 1709 insertions(+)
To github.com:arita37/mlmodels_store.git
   140db21..abbb5bd  master -> master





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
{'loss': 0.4814586900174618, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-12 16:33:19.176353: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
[master f59820e] ml_store
 1 file changed, 233 insertions(+)
To github.com:arita37/mlmodels_store.git
   abbb5bd..f59820e  master -> master





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
[master f83e341] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   f59820e..f83e341  master -> master





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
 40%|████      | 2/5 [00:21<00:32, 10.68s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.7774769016444215, 'learning_rate': 0.14531254103171726, 'min_data_in_leaf': 22, 'num_leaves': 59} and reward: 0.3916
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8\xe1\x17=>\xa9\x91X\r\x00\x00\x00learning_rateq\x02G?\xc2\x99\x99\xf1\xb7\x06nX\x10\x00\x00\x00min_data_in_leafq\x03K\x16X\n\x00\x00\x00num_leavesq\x04K;u.' and reward: 0.3916
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8\xe1\x17=>\xa9\x91X\r\x00\x00\x00learning_rateq\x02G?\xc2\x99\x99\xf1\xb7\x06nX\x10\x00\x00\x00min_data_in_leafq\x03K\x16X\n\x00\x00\x00num_leavesq\x04K;u.' and reward: 0.3916
 60%|██████    | 3/5 [00:49<00:32, 16.03s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.988172127412387, 'learning_rate': 0.00718745759649181, 'min_data_in_leaf': 24, 'num_leaves': 57} and reward: 0.3864
Finished Task with config: b"\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xef\x9f\x1b'A\xc2!X\r\x00\x00\x00learning_rateq\x02G?}p\x98u\x17\xd5\xc2X\x10\x00\x00\x00min_data_in_leafq\x03K\x18X\n\x00\x00\x00num_leavesq\x04K9u." and reward: 0.3864
Finished Task with config: b"\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xef\x9f\x1b'A\xc2!X\r\x00\x00\x00learning_rateq\x02G?}p\x98u\x17\xd5\xc2X\x10\x00\x00\x00min_data_in_leafq\x03K\x18X\n\x00\x00\x00num_leavesq\x04K9u." and reward: 0.3864
 80%|████████  | 4/5 [01:19<00:20, 20.21s/it] 80%|████████  | 4/5 [01:19<00:19, 19.96s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.7942561142317135, 'learning_rate': 0.11785175281610817, 'min_data_in_leaf': 15, 'num_leaves': 52} and reward: 0.3874
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9j\x8b\xcch\xbevX\r\x00\x00\x00learning_rateq\x02G?\xbe+\x88P\x1f\x18\tX\x10\x00\x00\x00min_data_in_leafq\x03K\x0fX\n\x00\x00\x00num_leavesq\x04K4u.' and reward: 0.3874
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9j\x8b\xcch\xbevX\r\x00\x00\x00learning_rateq\x02G?\xbe+\x88P\x1f\x18\tX\x10\x00\x00\x00min_data_in_leafq\x03K\x0fX\n\x00\x00\x00num_leavesq\x04K4u.' and reward: 0.3874
Time for Gradient Boosting hyperparameter optimization: 106.69513130187988
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.7774769016444215, 'learning_rate': 0.14531254103171726, 'min_data_in_leaf': 22, 'num_leaves': 59}
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
 40%|████      | 2/5 [00:54<01:22, 27.45s/it] 40%|████      | 2/5 [00:54<01:22, 27.46s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.28748604827400004, 'embedding_size_factor': 0.6592372238345126, 'layers.choice': 0, 'learning_rate': 0.0005504475339560686, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 0.000963576311622145} and reward: 0.359
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd2f+\xe1\xd9(\xacX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe5\x18x\xa9\x95\x9a\x1eX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?B\t}\x14\x08\\yX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?O\x93\x10_pd<u.' and reward: 0.359
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd2f+\xe1\xd9(\xacX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe5\x18x\xa9\x95\x9a\x1eX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?B\t}\x14\x08\\yX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?O\x93\x10_pd<u.' and reward: 0.359
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 111.40108489990234
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.75s of the -101.24s of remaining time.
Ensemble size: 64
Ensemble weights: 
[0.203125 0.25     0.203125 0.       0.09375  0.25    ]
	0.3998	 = Validation accuracy score
	1.56s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 222.86s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f2646296d68>

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
[master e0edea3] ml_store
 1 file changed, 198 insertions(+)
To github.com:arita37/mlmodels_store.git
   f83e341..e0edea3  master -> master





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
[master 6f02228] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   e0edea3..6f02228  master -> master





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
[master 3453ffd] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   6f02228..3453ffd  master -> master





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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7ff01990dfd0> 

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
[master 0add278] ml_store
 1 file changed, 108 insertions(+)
To github.com:arita37/mlmodels_store.git
   3453ffd..0add278  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 9.29250600e-01 -1.10657307e+00 -1.95816909e+00 -3.59224096e-01
  -1.21258781e+00  5.05381903e-01  5.42645295e-01  1.21794090e+00
  -1.94068096e+00  6.77807571e-01]
 [ 1.09488485e+00 -6.96245395e-02 -1.16444148e-01  3.53870427e-01
  -1.44189096e+00 -1.86955017e-01  1.29118890e+00 -1.53236162e-01
  -2.43250851e+00 -2.27729800e+00]
 [ 8.58774962e-01  2.29371761e+00 -1.47023709e+00 -8.30010986e-01
  -6.72049816e-01 -1.01951985e+00  5.99213235e-01 -2.14653842e-01
   1.02124813e+00  6.06403944e-01]
 [ 8.95623122e-01 -2.29820588e+00 -1.95225583e-02  1.45652739e+00
  -1.85064099e+00  3.16637236e-01  1.11337266e-01 -2.66412594e+00
  -4.26428618e-01 -8.39988915e-01]
 [ 1.32857949e+00 -5.63236604e-01 -1.06179676e+00  2.39014596e+00
  -1.68450770e+00  2.45422849e-01 -5.69148654e-01  1.15259914e+00
  -2.24235772e-01  1.32247779e-01]
 [ 1.12641981e+00 -6.29441604e-01  1.10100020e+00 -1.11343610e+00
   9.44595066e-01 -6.74100249e-02 -1.83400197e-01  1.16143998e+00
  -2.75293863e-02  7.80027135e-01]
 [ 1.39198128e+00 -1.90221025e-01 -5.37223024e-01 -4.48738033e-01
   7.04557071e-01 -6.72448039e-01 -7.01344426e-01 -5.57494722e-01
   9.39168744e-01  1.56263850e-01]
 [ 1.18559003e+00  8.64644065e-02  1.23289919e+00 -2.14246673e+00
   1.03334100e+00 -8.30168864e-01  3.67231814e-01  4.51615951e-01
   1.10417433e+00 -4.22856961e-01]
 [ 1.58463774e+00  5.71209961e-02 -1.77183179e-02 -7.99547491e-01
   1.32970299e+00 -2.91594596e-01 -1.10771250e+00 -2.58982853e-01
   1.89293198e-01 -1.71939447e+00]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 1.46893146e+00 -1.47115693e+00  5.85910431e-01 -8.30171895e-01
   1.03345052e+00 -8.80577600e-01 -9.55425262e-01 -2.79097722e-01
   1.62284909e+00  2.06578332e+00]
 [ 8.53355545e-01 -7.04350332e-01 -6.79383783e-01 -4.58666861e-02
  -1.29936179e+00 -2.18733459e-01  5.90039464e-01  1.53920701e+00
  -1.14870423e+00 -9.50909251e-01]
 [ 1.05936450e-01 -7.37289628e-01  6.50323214e-01  1.64665066e-01
  -1.53556118e+00  7.78174179e-01  5.03170861e-02  3.09816759e-01
   1.05132077e+00  6.06548400e-01]
 [ 8.15836116e-01 -1.39169388e+00  2.50598029e+00  4.50217742e-01
  -8.82869820e-01  6.27437083e-01 -1.19586151e+00  7.51337235e-01
   1.40395436e-01  1.91979229e+00]
 [ 1.14809657e+00 -7.33271604e-01  2.62467445e-01  8.36004719e-01
   1.17353145e+00  1.54335911e+00  2.84748111e-01  7.58805660e-01
   8.84908814e-01  2.76499305e-01]
 [ 1.18947778e+00 -6.80678141e-01 -5.68244809e-02 -8.45080274e-02
   8.21783210e-01 -2.97361883e-01 -1.86578994e-01  4.17302005e-01
   7.84770651e-01  4.92336556e-01]
 [ 8.88389445e-01  2.82995534e-01  1.79558917e-02  1.08030817e-01
  -8.49671873e-01  2.94176190e-02 -5.03973949e-01 -1.34793129e-01
   1.04921829e+00 -1.27046078e+00]
 [ 1.14377130e+00  7.27813500e-01  3.52494364e-01  5.15073614e-01
   1.17718111e+00 -2.78253447e+00 -1.94332341e+00  5.84646610e-01
   3.24274243e-01 -2.36436952e-01]
 [ 1.77547698e+00 -2.03394449e-01 -1.98837863e-01  2.42669441e-01
   9.64350564e-01  2.01830179e-01 -5.45774168e-01  6.61020288e-01
   1.79215821e+00 -7.00398505e-01]
 [ 8.61462558e-01  7.43205537e-02 -1.34501002e+00 -1.99560718e-01
  -1.47533915e+00 -6.54603169e-01 -3.14563862e-01  3.18014296e-01
  -8.90271552e-01 -1.29525789e+00]
 [ 6.25673373e-01  5.92472801e-01  6.74570707e-01  1.19783084e+00
   1.23187251e+00  1.70459417e+00 -7.67309826e-01  1.04008915e+00
  -9.18440038e-01  1.46089238e+00]
 [ 1.07258847e+00 -5.86523939e-01 -1.34267579e+00 -1.23685338e+00
   1.24328724e+00  8.75838928e-01 -3.26499498e-01  6.23362177e-01
  -4.34956683e-01  1.11438298e+00]
 [ 1.98519313e+00  6.74711526e-01 -1.39662042e+00  6.18539131e-01
   1.22382712e+00 -4.43171931e-01 -1.89148284e-03  1.81053491e+00
  -1.30572692e+00 -8.61316361e-01]
 [ 8.88838813e-01  1.03368687e+00 -4.97025792e-02  8.08844360e-01
   8.14051347e-01  1.78975468e+00  1.14690038e+00  4.51284016e-01
  -1.68405999e+00  4.66643267e-01]
 [ 7.75285326e-01  1.47016034e+00  1.03298378e+00 -8.70008223e-01
   7.86556511e-01  3.69190470e-01 -1.43195745e-01  8.53282186e-01
  -1.39711730e-01 -2.22414029e-01]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7f06ca527f60>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7f06e4896748> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 0.44118981  0.47985237 -0.1920037  -1.55269878 -1.88873982  0.57846442
   0.39859839 -0.9612636  -1.45832446 -3.05376438]
 [ 2.07582971 -1.40232915 -0.47918492  0.45112294  1.03436581 -0.6949209
  -0.4189379   0.5154138  -1.11487105 -1.95210529]
 [ 1.21619061 -0.01900052  0.86089124 -0.22676019 -1.36419132 -1.56450785
   1.63169151  0.93125568  0.94980882 -0.88018906]
 [ 0.93621125  0.20437739 -1.49419377  0.61223252 -0.98437725  0.74488454
   0.49434165 -0.03628129 -0.83239535 -0.4466992 ]
 [ 0.94781411 -1.13379204  0.64098587 -0.1905483  -1.23912256  0.23333913
  -0.3169012   0.43499832  0.9104236   1.21987438]
 [ 1.32720112 -0.16119832  0.6024509  -0.28638492 -0.5789623  -0.87088765
   1.37975819  0.50142959 -0.47861407 -0.89264667]
 [ 0.76170668 -1.48515645  1.30253554 -0.59246129 -1.64162479 -2.30490794
  -1.34869645 -0.03181717  0.11248774 -0.36261209]
 [ 1.16777676 -0.66575452 -1.23312074 -1.67419581  1.01313574  0.82502982
  -0.12046457 -0.49821356 -0.31098498 -1.18231813]
 [ 1.17867274 -0.59980453 -0.6946936   1.12341216  1.17899425  0.30526704
   0.01335268  1.3887794  -0.66134424  0.6218035 ]
 [ 1.16755486  0.0353601   0.7147896  -1.53879325  1.10863359 -0.44789518
  -1.75592564  0.61798553 -0.18417633  0.85270406]
 [ 1.09488485 -0.06962454 -0.11644415  0.35387043 -1.44189096 -0.18695502
   1.2911889  -0.15323616 -2.43250851 -2.277298  ]
 [ 0.96457205 -0.10679399  1.12232832  1.45142926  1.21828168 -0.61803685
   0.43816635 -2.03720123 -1.94258918 -0.9970198 ]
 [ 0.62368852  1.2066079   0.90399917 -0.28286355 -1.18913787 -0.26632688
   1.42361443  1.06897162  0.04037143  1.57546791]
 [ 1.58463774  0.057121   -0.01771832 -0.79954749  1.32970299 -0.2915946
  -1.1077125  -0.25898285  0.1892932  -1.71939447]
 [ 1.12641981 -0.6294416   1.1010002  -1.1134361   0.94459507 -0.06741002
  -0.1834002   1.16143998 -0.02752939  0.78002714]
 [ 0.98042741  1.93752881 -0.23083974  0.36633201  1.10018476 -1.04458938
  -0.34498721  2.05117344  0.585662   -2.793085  ]
 [ 0.6109426  -2.79099641 -1.33520272 -0.45611756 -0.94495995 -0.97989025
  -0.15699367  0.69257435 -0.47867236 -0.10646012]
 [ 0.88838944  0.28299553  0.01795589  0.10803082 -0.84967187  0.02941762
  -0.50397395 -0.13479313  1.04921829 -1.27046078]
 [ 1.34728643 -0.36453805  0.08075099 -0.45971768 -0.8894876   1.70548352
   0.09499611  0.24050555 -0.9994265  -0.76780375]
 [ 0.81583612 -1.39169388  2.50598029  0.45021774 -0.88286982  0.62743708
  -1.19586151  0.75133724  0.14039544  1.91979229]
 [ 0.10593645 -0.73728963  0.65032321  0.16466507 -1.53556118  0.77817418
   0.05031709  0.30981676  1.05132077  0.6065484 ]
 [ 0.44689516  0.38653915  1.35010682 -0.85145566  0.85063796  1.00088142
  -1.1601701  -0.38483225  1.45810824 -0.33128317]
 [ 1.77547698 -0.20339445 -0.19883786  0.24266944  0.96435056  0.20183018
  -0.54577417  0.66102029  1.79215821 -0.7003985 ]
 [ 0.85982375  0.17195713 -0.34898419  0.49056104 -1.15649503 -1.39528303
   0.61472628 -0.52235647 -0.3692559  -0.977773  ]
 [ 1.66752297  1.22372221 -0.4599301  -0.0593679  -0.493857    1.4489894
  -1.18110317 -0.47758085  0.02599999 -0.79079995]]
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
[[ 0.85335555 -0.70435033 -0.67938378 -0.04586669 -1.29936179 -0.21873346
   0.59003946  1.53920701 -1.14870423 -0.95090925]
 [ 1.32857949 -0.5632366  -1.06179676  2.39014596 -1.6845077   0.24542285
  -0.56914865  1.15259914 -0.22423577  0.13224778]
 [ 0.9292506  -1.10657307 -1.95816909 -0.3592241  -1.21258781  0.5053819
   0.54264529  1.2179409  -1.94068096  0.67780757]
 [ 0.78801845  0.30196005  0.70098212 -0.39468968 -1.20376927 -1.17181338
   0.75539203  0.98401224 -0.55968142 -0.19893745]
 [ 0.94781411 -1.13379204  0.64098587 -0.1905483  -1.23912256  0.23333913
  -0.3169012   0.43499832  0.9104236   1.21987438]
 [ 1.06702918 -0.42914228  0.35016716  1.20845633  0.75148062  1.1157018
  -0.4791571   0.84086156 -0.10288722  0.01716473]
 [ 1.66752297  1.22372221 -0.4599301  -0.0593679  -0.493857    1.4489894
  -1.18110317 -0.47758085  0.02599999 -0.79079995]
 [ 0.69211449 -0.06065249  2.05635552 -2.413503    1.17456965 -1.77756638
  -0.28173627 -0.77785883  1.11584111  1.76024923]
 [ 0.85729649  0.9561217  -0.82609743 -0.70584051  1.13872896  1.19268607
   0.28267571 -0.23794194  1.15528789  0.6210827 ]
 [ 1.14809657 -0.7332716   0.26246745  0.83600472  1.17353145  1.54335911
   0.28474811  0.75880566  0.88490881  0.2764993 ]
 [ 0.6675918  -0.45252497 -0.60598132  1.16128569 -1.44620987  1.06996554
   1.92381543 -1.04553425  0.35528451  1.80358898]
 [ 1.77547698 -0.20339445 -0.19883786  0.24266944  0.96435056  0.20183018
  -0.54577417  0.66102029  1.79215821 -0.7003985 ]
 [ 0.5630779  -1.17598267 -0.17418034  1.01012718  1.06796368  0.92001793
  -0.16819884 -0.19505734  0.80539342  0.4611641 ]
 [ 1.27991386 -0.87142207 -0.32403233 -0.86482994 -0.96853969  0.60874908
   0.50798434  0.5616381   1.51475038 -1.51107661]
 [ 0.79032389  1.61336137 -2.09424782 -0.37480469  0.91588404 -0.74996962
   0.31027229  2.0546241   0.05340954 -0.22876583]
 [ 1.09488485 -0.06962454 -0.11644415  0.35387043 -1.44189096 -0.18695502
   1.2911889  -0.15323616 -2.43250851 -2.277298  ]
 [ 0.89551051  0.92061512  0.79452824 -0.03536792  0.8780991   2.11060505
  -1.02188594 -1.30653407  0.07638048 -1.87316098]
 [ 0.87699465  1.23225307 -0.86778722 -0.25417987  0.89189141  1.39984394
  -0.87728152 -0.78191168 -0.43750898 -1.44087602]
 [ 0.87874071 -0.01923163  0.31965694  0.15001628 -1.46662161  0.46353432
  -0.89868319  0.39788042 -0.99601089  0.3181542 ]
 [ 1.32720112 -0.16119832  0.6024509  -0.28638492 -0.5789623  -0.87088765
   1.37975819  0.50142959 -0.47861407 -0.89264667]
 [ 0.97139534  0.71304905  1.76041518  1.30620607  1.0576549  -0.60460297
   0.12837699  0.63658341  1.40925339  0.96653925]
 [ 0.87226739 -2.51630386 -0.77507029 -0.59566788  1.02600767 -0.30912132
   1.74643509  0.51093777  1.71066184  0.14164054]
 [ 0.88861146  0.84958685 -0.03091142 -0.12215402 -1.14722826 -0.68085157
  -0.32606131 -1.06787658 -0.07667936  0.35571726]
 [ 1.838294    0.50274088  0.12910158  1.55880554  1.32551412  0.1094027
   1.40754    -1.2197444   2.44936865  1.6169496 ]
 [ 1.37661405 -0.60022533  0.72591685 -0.37951752 -0.62754626 -1.01480369
   0.96622086  0.4359862  -0.68748739  3.32107876]]
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
[master f671da7] ml_store
 1 file changed, 271 insertions(+)
To github.com:arita37/mlmodels_store.git
   0add278..f671da7  master -> master





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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140404187131856
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140404187131632
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140404187130400
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140404187129952
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140404187129448
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140404187129112

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
grad_step = 000000, loss = 1.581304
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 1.342467
grad_step = 000002, loss = 1.100579
grad_step = 000003, loss = 0.836999
grad_step = 000004, loss = 0.565454
grad_step = 000005, loss = 0.355614
grad_step = 000006, loss = 0.307599
grad_step = 000007, loss = 0.323313
grad_step = 000008, loss = 0.232878
grad_step = 000009, loss = 0.106548
grad_step = 000010, loss = 0.034793
grad_step = 000011, loss = 0.029220
grad_step = 000012, loss = 0.055215
grad_step = 000013, loss = 0.076367
grad_step = 000014, loss = 0.078564
grad_step = 000015, loss = 0.064112
grad_step = 000016, loss = 0.043906
grad_step = 000017, loss = 0.028520
grad_step = 000018, loss = 0.022824
grad_step = 000019, loss = 0.023330
grad_step = 000020, loss = 0.022713
grad_step = 000021, loss = 0.017727
grad_step = 000022, loss = 0.011975
grad_step = 000023, loss = 0.010305
grad_step = 000024, loss = 0.013539
grad_step = 000025, loss = 0.018423
grad_step = 000026, loss = 0.021316
grad_step = 000027, loss = 0.020757
grad_step = 000028, loss = 0.017524
grad_step = 000029, loss = 0.013523
grad_step = 000030, loss = 0.010510
grad_step = 000031, loss = 0.009093
grad_step = 000032, loss = 0.008576
grad_step = 000033, loss = 0.008004
grad_step = 000034, loss = 0.007190
grad_step = 000035, loss = 0.006702
grad_step = 000036, loss = 0.006965
grad_step = 000037, loss = 0.007696
grad_step = 000038, loss = 0.008294
grad_step = 000039, loss = 0.008265
grad_step = 000040, loss = 0.007603
grad_step = 000041, loss = 0.006709
grad_step = 000042, loss = 0.005998
grad_step = 000043, loss = 0.005651
grad_step = 000044, loss = 0.005562
grad_step = 000045, loss = 0.005505
grad_step = 000046, loss = 0.005377
grad_step = 000047, loss = 0.005261
grad_step = 000048, loss = 0.005269
grad_step = 000049, loss = 0.005372
grad_step = 000050, loss = 0.005411
grad_step = 000051, loss = 0.005272
grad_step = 000052, loss = 0.004998
grad_step = 000053, loss = 0.004731
grad_step = 000054, loss = 0.004569
grad_step = 000055, loss = 0.004494
grad_step = 000056, loss = 0.004426
grad_step = 000057, loss = 0.004331
grad_step = 000058, loss = 0.004260
grad_step = 000059, loss = 0.004258
grad_step = 000060, loss = 0.004297
grad_step = 000061, loss = 0.004303
grad_step = 000062, loss = 0.004237
grad_step = 000063, loss = 0.004126
grad_step = 000064, loss = 0.004020
grad_step = 000065, loss = 0.003941
grad_step = 000066, loss = 0.003869
grad_step = 000067, loss = 0.003787
grad_step = 000068, loss = 0.003716
grad_step = 000069, loss = 0.003678
grad_step = 000070, loss = 0.003663
grad_step = 000071, loss = 0.003636
grad_step = 000072, loss = 0.003582
grad_step = 000073, loss = 0.003518
grad_step = 000074, loss = 0.003462
grad_step = 000075, loss = 0.003409
grad_step = 000076, loss = 0.003340
grad_step = 000077, loss = 0.003276
grad_step = 000078, loss = 0.003227
grad_step = 000079, loss = 0.003191
grad_step = 000080, loss = 0.003154
grad_step = 000081, loss = 0.003106
grad_step = 000082, loss = 0.003054
grad_step = 000083, loss = 0.003003
grad_step = 000084, loss = 0.002948
grad_step = 000085, loss = 0.002890
grad_step = 000086, loss = 0.002840
grad_step = 000087, loss = 0.002798
grad_step = 000088, loss = 0.002753
grad_step = 000089, loss = 0.002707
grad_step = 000090, loss = 0.002664
grad_step = 000091, loss = 0.002616
grad_step = 000092, loss = 0.002566
grad_step = 000093, loss = 0.002522
grad_step = 000094, loss = 0.002481
grad_step = 000095, loss = 0.002439
grad_step = 000096, loss = 0.002407
grad_step = 000097, loss = 0.002368
grad_step = 000098, loss = 0.002335
grad_step = 000099, loss = 0.002298
grad_step = 000100, loss = 0.002264
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002232
grad_step = 000102, loss = 0.002199
grad_step = 000103, loss = 0.002169
grad_step = 000104, loss = 0.002136
grad_step = 000105, loss = 0.002106
grad_step = 000106, loss = 0.002074
grad_step = 000107, loss = 0.002042
grad_step = 000108, loss = 0.002009
grad_step = 000109, loss = 0.001976
grad_step = 000110, loss = 0.001943
grad_step = 000111, loss = 0.001909
grad_step = 000112, loss = 0.001874
grad_step = 000113, loss = 0.001839
grad_step = 000114, loss = 0.001803
grad_step = 000115, loss = 0.001768
grad_step = 000116, loss = 0.001733
grad_step = 000117, loss = 0.001699
grad_step = 000118, loss = 0.001666
grad_step = 000119, loss = 0.001631
grad_step = 000120, loss = 0.001588
grad_step = 000121, loss = 0.001546
grad_step = 000122, loss = 0.001509
grad_step = 000123, loss = 0.001476
grad_step = 000124, loss = 0.001439
grad_step = 000125, loss = 0.001398
grad_step = 000126, loss = 0.001360
grad_step = 000127, loss = 0.001327
grad_step = 000128, loss = 0.001299
grad_step = 000129, loss = 0.001273
grad_step = 000130, loss = 0.001242
grad_step = 000131, loss = 0.001213
grad_step = 000132, loss = 0.001188
grad_step = 000133, loss = 0.001171
grad_step = 000134, loss = 0.001157
grad_step = 000135, loss = 0.001146
grad_step = 000136, loss = 0.001130
grad_step = 000137, loss = 0.001114
grad_step = 000138, loss = 0.001099
grad_step = 000139, loss = 0.001088
grad_step = 000140, loss = 0.001081
grad_step = 000141, loss = 0.001078
grad_step = 000142, loss = 0.001077
grad_step = 000143, loss = 0.001069
grad_step = 000144, loss = 0.001046
grad_step = 000145, loss = 0.001020
grad_step = 000146, loss = 0.001005
grad_step = 000147, loss = 0.001001
grad_step = 000148, loss = 0.000990
grad_step = 000149, loss = 0.000970
grad_step = 000150, loss = 0.000956
grad_step = 000151, loss = 0.000952
grad_step = 000152, loss = 0.000950
grad_step = 000153, loss = 0.000940
grad_step = 000154, loss = 0.000923
grad_step = 000155, loss = 0.000909
grad_step = 000156, loss = 0.000903
grad_step = 000157, loss = 0.000903
grad_step = 000158, loss = 0.000901
grad_step = 000159, loss = 0.000894
grad_step = 000160, loss = 0.000882
grad_step = 000161, loss = 0.000863
grad_step = 000162, loss = 0.000846
grad_step = 000163, loss = 0.000842
grad_step = 000164, loss = 0.000846
grad_step = 000165, loss = 0.000843
grad_step = 000166, loss = 0.000829
grad_step = 000167, loss = 0.000811
grad_step = 000168, loss = 0.000799
grad_step = 000169, loss = 0.000793
grad_step = 000170, loss = 0.000790
grad_step = 000171, loss = 0.000789
grad_step = 000172, loss = 0.000786
grad_step = 000173, loss = 0.000777
grad_step = 000174, loss = 0.000764
grad_step = 000175, loss = 0.000750
grad_step = 000176, loss = 0.000740
grad_step = 000177, loss = 0.000736
grad_step = 000178, loss = 0.000735
grad_step = 000179, loss = 0.000736
grad_step = 000180, loss = 0.000737
grad_step = 000181, loss = 0.000739
grad_step = 000182, loss = 0.000732
grad_step = 000183, loss = 0.000719
grad_step = 000184, loss = 0.000697
grad_step = 000185, loss = 0.000683
grad_step = 000186, loss = 0.000676
grad_step = 000187, loss = 0.000675
grad_step = 000188, loss = 0.000676
grad_step = 000189, loss = 0.000677
grad_step = 000190, loss = 0.000683
grad_step = 000191, loss = 0.000685
grad_step = 000192, loss = 0.000686
grad_step = 000193, loss = 0.000665
grad_step = 000194, loss = 0.000637
grad_step = 000195, loss = 0.000623
grad_step = 000196, loss = 0.000629
grad_step = 000197, loss = 0.000639
grad_step = 000198, loss = 0.000633
grad_step = 000199, loss = 0.000619
grad_step = 000200, loss = 0.000607
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.000601
grad_step = 000202, loss = 0.000594
grad_step = 000203, loss = 0.000584
grad_step = 000204, loss = 0.000579
grad_step = 000205, loss = 0.000583
grad_step = 000206, loss = 0.000590
grad_step = 000207, loss = 0.000598
grad_step = 000208, loss = 0.000615
grad_step = 000209, loss = 0.000637
grad_step = 000210, loss = 0.000638
grad_step = 000211, loss = 0.000588
grad_step = 000212, loss = 0.000545
grad_step = 000213, loss = 0.000547
grad_step = 000214, loss = 0.000572
grad_step = 000215, loss = 0.000573
grad_step = 000216, loss = 0.000545
grad_step = 000217, loss = 0.000526
grad_step = 000218, loss = 0.000527
grad_step = 000219, loss = 0.000533
grad_step = 000220, loss = 0.000531
grad_step = 000221, loss = 0.000525
grad_step = 000222, loss = 0.000517
grad_step = 000223, loss = 0.000508
grad_step = 000224, loss = 0.000496
grad_step = 000225, loss = 0.000498
grad_step = 000226, loss = 0.000506
grad_step = 000227, loss = 0.000506
grad_step = 000228, loss = 0.000501
grad_step = 000229, loss = 0.000501
grad_step = 000230, loss = 0.000501
grad_step = 000231, loss = 0.000495
grad_step = 000232, loss = 0.000482
grad_step = 000233, loss = 0.000474
grad_step = 000234, loss = 0.000469
grad_step = 000235, loss = 0.000465
grad_step = 000236, loss = 0.000459
grad_step = 000237, loss = 0.000454
grad_step = 000238, loss = 0.000452
grad_step = 000239, loss = 0.000454
grad_step = 000240, loss = 0.000459
grad_step = 000241, loss = 0.000471
grad_step = 000242, loss = 0.000503
grad_step = 000243, loss = 0.000565
grad_step = 000244, loss = 0.000646
grad_step = 000245, loss = 0.000636
grad_step = 000246, loss = 0.000520
grad_step = 000247, loss = 0.000432
grad_step = 000248, loss = 0.000499
grad_step = 000249, loss = 0.000546
grad_step = 000250, loss = 0.000459
grad_step = 000251, loss = 0.000435
grad_step = 000252, loss = 0.000495
grad_step = 000253, loss = 0.000467
grad_step = 000254, loss = 0.000419
grad_step = 000255, loss = 0.000451
grad_step = 000256, loss = 0.000460
grad_step = 000257, loss = 0.000419
grad_step = 000258, loss = 0.000422
grad_step = 000259, loss = 0.000445
grad_step = 000260, loss = 0.000422
grad_step = 000261, loss = 0.000405
grad_step = 000262, loss = 0.000426
grad_step = 000263, loss = 0.000424
grad_step = 000264, loss = 0.000400
grad_step = 000265, loss = 0.000402
grad_step = 000266, loss = 0.000415
grad_step = 000267, loss = 0.000404
grad_step = 000268, loss = 0.000390
grad_step = 000269, loss = 0.000397
grad_step = 000270, loss = 0.000402
grad_step = 000271, loss = 0.000391
grad_step = 000272, loss = 0.000383
grad_step = 000273, loss = 0.000389
grad_step = 000274, loss = 0.000390
grad_step = 000275, loss = 0.000382
grad_step = 000276, loss = 0.000376
grad_step = 000277, loss = 0.000380
grad_step = 000278, loss = 0.000381
grad_step = 000279, loss = 0.000374
grad_step = 000280, loss = 0.000370
grad_step = 000281, loss = 0.000371
grad_step = 000282, loss = 0.000372
grad_step = 000283, loss = 0.000369
grad_step = 000284, loss = 0.000364
grad_step = 000285, loss = 0.000362
grad_step = 000286, loss = 0.000363
grad_step = 000287, loss = 0.000364
grad_step = 000288, loss = 0.000361
grad_step = 000289, loss = 0.000357
grad_step = 000290, loss = 0.000355
grad_step = 000291, loss = 0.000355
grad_step = 000292, loss = 0.000356
grad_step = 000293, loss = 0.000354
grad_step = 000294, loss = 0.000352
grad_step = 000295, loss = 0.000349
grad_step = 000296, loss = 0.000348
grad_step = 000297, loss = 0.000349
grad_step = 000298, loss = 0.000352
grad_step = 000299, loss = 0.000359
grad_step = 000300, loss = 0.000375
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000410
grad_step = 000302, loss = 0.000480
grad_step = 000303, loss = 0.000572
grad_step = 000304, loss = 0.000630
grad_step = 000305, loss = 0.000528
grad_step = 000306, loss = 0.000372
grad_step = 000307, loss = 0.000369
grad_step = 000308, loss = 0.000466
grad_step = 000309, loss = 0.000446
grad_step = 000310, loss = 0.000350
grad_step = 000311, loss = 0.000369
grad_step = 000312, loss = 0.000430
grad_step = 000313, loss = 0.000388
grad_step = 000314, loss = 0.000335
grad_step = 000315, loss = 0.000368
grad_step = 000316, loss = 0.000393
grad_step = 000317, loss = 0.000353
grad_step = 000318, loss = 0.000332
grad_step = 000319, loss = 0.000360
grad_step = 000320, loss = 0.000366
grad_step = 000321, loss = 0.000340
grad_step = 000322, loss = 0.000330
grad_step = 000323, loss = 0.000345
grad_step = 000324, loss = 0.000347
grad_step = 000325, loss = 0.000334
grad_step = 000326, loss = 0.000330
grad_step = 000327, loss = 0.000334
grad_step = 000328, loss = 0.000330
grad_step = 000329, loss = 0.000325
grad_step = 000330, loss = 0.000327
grad_step = 000331, loss = 0.000327
grad_step = 000332, loss = 0.000320
grad_step = 000333, loss = 0.000318
grad_step = 000334, loss = 0.000322
grad_step = 000335, loss = 0.000321
grad_step = 000336, loss = 0.000314
grad_step = 000337, loss = 0.000312
grad_step = 000338, loss = 0.000316
grad_step = 000339, loss = 0.000317
grad_step = 000340, loss = 0.000313
grad_step = 000341, loss = 0.000309
grad_step = 000342, loss = 0.000309
grad_step = 000343, loss = 0.000310
grad_step = 000344, loss = 0.000309
grad_step = 000345, loss = 0.000307
grad_step = 000346, loss = 0.000307
grad_step = 000347, loss = 0.000307
grad_step = 000348, loss = 0.000306
grad_step = 000349, loss = 0.000303
grad_step = 000350, loss = 0.000301
grad_step = 000351, loss = 0.000301
grad_step = 000352, loss = 0.000301
grad_step = 000353, loss = 0.000300
grad_step = 000354, loss = 0.000299
grad_step = 000355, loss = 0.000298
grad_step = 000356, loss = 0.000297
grad_step = 000357, loss = 0.000298
grad_step = 000358, loss = 0.000298
grad_step = 000359, loss = 0.000298
grad_step = 000360, loss = 0.000298
grad_step = 000361, loss = 0.000299
grad_step = 000362, loss = 0.000301
grad_step = 000363, loss = 0.000306
grad_step = 000364, loss = 0.000313
grad_step = 000365, loss = 0.000322
grad_step = 000366, loss = 0.000335
grad_step = 000367, loss = 0.000344
grad_step = 000368, loss = 0.000351
grad_step = 000369, loss = 0.000344
grad_step = 000370, loss = 0.000328
grad_step = 000371, loss = 0.000306
grad_step = 000372, loss = 0.000292
grad_step = 000373, loss = 0.000289
grad_step = 000374, loss = 0.000297
grad_step = 000375, loss = 0.000308
grad_step = 000376, loss = 0.000313
grad_step = 000377, loss = 0.000309
grad_step = 000378, loss = 0.000299
grad_step = 000379, loss = 0.000288
grad_step = 000380, loss = 0.000284
grad_step = 000381, loss = 0.000286
grad_step = 000382, loss = 0.000292
grad_step = 000383, loss = 0.000297
grad_step = 000384, loss = 0.000299
grad_step = 000385, loss = 0.000297
grad_step = 000386, loss = 0.000292
grad_step = 000387, loss = 0.000286
grad_step = 000388, loss = 0.000282
grad_step = 000389, loss = 0.000280
grad_step = 000390, loss = 0.000280
grad_step = 000391, loss = 0.000281
grad_step = 000392, loss = 0.000283
grad_step = 000393, loss = 0.000285
grad_step = 000394, loss = 0.000288
grad_step = 000395, loss = 0.000291
grad_step = 000396, loss = 0.000294
grad_step = 000397, loss = 0.000297
grad_step = 000398, loss = 0.000300
grad_step = 000399, loss = 0.000304
grad_step = 000400, loss = 0.000305
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000305
grad_step = 000402, loss = 0.000301
grad_step = 000403, loss = 0.000296
grad_step = 000404, loss = 0.000287
grad_step = 000405, loss = 0.000279
grad_step = 000406, loss = 0.000274
grad_step = 000407, loss = 0.000273
grad_step = 000408, loss = 0.000276
grad_step = 000409, loss = 0.000282
grad_step = 000410, loss = 0.000290
grad_step = 000411, loss = 0.000297
grad_step = 000412, loss = 0.000304
grad_step = 000413, loss = 0.000306
grad_step = 000414, loss = 0.000306
grad_step = 000415, loss = 0.000300
grad_step = 000416, loss = 0.000292
grad_step = 000417, loss = 0.000282
grad_step = 000418, loss = 0.000275
grad_step = 000419, loss = 0.000272
grad_step = 000420, loss = 0.000274
grad_step = 000421, loss = 0.000279
grad_step = 000422, loss = 0.000283
grad_step = 000423, loss = 0.000286
grad_step = 000424, loss = 0.000285
grad_step = 000425, loss = 0.000281
grad_step = 000426, loss = 0.000275
grad_step = 000427, loss = 0.000269
grad_step = 000428, loss = 0.000265
grad_step = 000429, loss = 0.000265
grad_step = 000430, loss = 0.000267
grad_step = 000431, loss = 0.000271
grad_step = 000432, loss = 0.000277
grad_step = 000433, loss = 0.000280
grad_step = 000434, loss = 0.000283
grad_step = 000435, loss = 0.000281
grad_step = 000436, loss = 0.000276
grad_step = 000437, loss = 0.000270
grad_step = 000438, loss = 0.000264
grad_step = 000439, loss = 0.000262
grad_step = 000440, loss = 0.000263
grad_step = 000441, loss = 0.000270
grad_step = 000442, loss = 0.000283
grad_step = 000443, loss = 0.000299
grad_step = 000444, loss = 0.000316
grad_step = 000445, loss = 0.000323
grad_step = 000446, loss = 0.000320
grad_step = 000447, loss = 0.000305
grad_step = 000448, loss = 0.000297
grad_step = 000449, loss = 0.000305
grad_step = 000450, loss = 0.000327
grad_step = 000451, loss = 0.000334
grad_step = 000452, loss = 0.000315
grad_step = 000453, loss = 0.000275
grad_step = 000454, loss = 0.000254
grad_step = 000455, loss = 0.000263
grad_step = 000456, loss = 0.000279
grad_step = 000457, loss = 0.000280
grad_step = 000458, loss = 0.000264
grad_step = 000459, loss = 0.000256
grad_step = 000460, loss = 0.000265
grad_step = 000461, loss = 0.000275
grad_step = 000462, loss = 0.000270
grad_step = 000463, loss = 0.000258
grad_step = 000464, loss = 0.000252
grad_step = 000465, loss = 0.000257
grad_step = 000466, loss = 0.000261
grad_step = 000467, loss = 0.000258
grad_step = 000468, loss = 0.000250
grad_step = 000469, loss = 0.000247
grad_step = 000470, loss = 0.000250
grad_step = 000471, loss = 0.000254
grad_step = 000472, loss = 0.000253
grad_step = 000473, loss = 0.000249
grad_step = 000474, loss = 0.000246
grad_step = 000475, loss = 0.000247
grad_step = 000476, loss = 0.000251
grad_step = 000477, loss = 0.000254
grad_step = 000478, loss = 0.000256
grad_step = 000479, loss = 0.000259
grad_step = 000480, loss = 0.000268
grad_step = 000481, loss = 0.000284
grad_step = 000482, loss = 0.000305
grad_step = 000483, loss = 0.000334
grad_step = 000484, loss = 0.000356
grad_step = 000485, loss = 0.000365
grad_step = 000486, loss = 0.000341
grad_step = 000487, loss = 0.000297
grad_step = 000488, loss = 0.000262
grad_step = 000489, loss = 0.000257
grad_step = 000490, loss = 0.000271
grad_step = 000491, loss = 0.000274
grad_step = 000492, loss = 0.000266
grad_step = 000493, loss = 0.000260
grad_step = 000494, loss = 0.000262
grad_step = 000495, loss = 0.000260
grad_step = 000496, loss = 0.000251
grad_step = 000497, loss = 0.000245
grad_step = 000498, loss = 0.000249
grad_step = 000499, loss = 0.000258
grad_step = 000500, loss = 0.000256
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000245
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
[[0.84101856 0.8560925  0.93028164 0.9520962  1.0152019 ]
 [0.85324484 0.90247124 0.96959835 1.0280722  0.994952  ]
 [0.8989238  0.9473556  1.0053128  0.99341947 0.96081215]
 [0.9247807  0.9867903  1.0131457  0.94579744 0.9219473 ]
 [0.9758899  0.9847399  0.96203315 0.9212699  0.88235307]
 [0.980739   0.94694805 0.9043923  0.8512877  0.86837196]
 [0.93942463 0.8988741  0.8616973  0.85971564 0.82694715]
 [0.8833118  0.84106505 0.85112655 0.8085425  0.8392597 ]
 [0.83775884 0.82511777 0.82219875 0.8525318  0.8476852 ]
 [0.81779826 0.81973743 0.8265046  0.8361739  0.8430387 ]
 [0.79746044 0.82883424 0.86746633 0.82276034 0.91695946]
 [0.8329021  0.83865905 0.8227398  0.9294531  0.95282906]
 [0.834606   0.8477032  0.92496073 0.95525444 1.014701  ]
 [0.8523514  0.9066994  0.9732157  1.0250393  0.9841621 ]
 [0.90670455 0.9550265  1.0072409  0.9854754  0.9444622 ]
 [0.9371617  0.9872458  1.0001236  0.9318045  0.9040872 ]
 [0.9783058  0.9744404  0.94081783 0.9010429  0.8630201 ]
 [0.9717376  0.9295113  0.8871093  0.83402544 0.8533921 ]
 [0.9316065  0.8842498  0.84852254 0.8523056  0.8230711 ]
 [0.8893796  0.8411665  0.8476954  0.81221986 0.8421043 ]
 [0.85316145 0.83263063 0.82738876 0.8600334  0.8575769 ]
 [0.8361604  0.8305274  0.831552   0.84901404 0.8508352 ]
 [0.8111187  0.83761084 0.87750995 0.8315504  0.92277807]
 [0.84094083 0.85228646 0.8341042  0.9294367  0.9503999 ]
 [0.84783274 0.8619914  0.9305964  0.95631254 1.022905  ]
 [0.8625338  0.90682733 0.97368515 1.0355903  1.009315  ]
 [0.9066593  0.9540961  1.0160186  1.0074178  0.97475815]
 [0.93453836 0.99670464 1.0286424  0.9586057  0.9349793 ]
 [0.9804949  0.99226815 0.97642744 0.9342735  0.8925036 ]
 [0.9890079  0.9559928  0.9154732  0.8576808  0.8767761 ]
 [0.9463651  0.90551805 0.86916965 0.8662795  0.835827  ]]

  #### Plot     ############################################### 
Saved image to ztest/model_tch/nbeats//n_beats_test.png.

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   f671da7..5ea0590  master     -> origin/master
Updating f671da7..5ea0590
Fast-forward
 ...-13_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py | 2404 ++++++++++++++++++++
 1 file changed, 2404 insertions(+)
 create mode 100644 log_benchmark/log_benchmark_2020-05-12-16-13_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py
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
[master abc9db6] ml_store
 1 file changed, 1128 insertions(+)
To github.com:arita37/mlmodels_store.git
   5ea0590..abc9db6  master -> master





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
[master 6d29d13] ml_store
 1 file changed, 37 insertions(+)
To github.com:arita37/mlmodels_store.git
   abc9db6..6d29d13  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//matchzoo_models.py 

  #### Loading params   ############################################## 

  {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'dataset_pars': {'data_pack': '', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'resample': True, 'sort': False, 'callbacks': 'PADDING'}, 'dataloader': '', 'dataloader_pars': {'device': 'cpu', 'dataset': 'None', 'stage': 'train', 'callback': 'PADDING'}, 'preprocess': {'train': {'transform': True, 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'stage': 'train', 'resample': True, 'sort': False, 'dataloader_callback': 'PADDING'}, 'test': {'transform': True, 'batch_size': 20, 'stage': 'dev', 'dataloader_callback': 'PADDING'}}} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Loading dataset   ############################################# 

  #### Model init   ################################################## 
  0%|          | 0/231508 [00:00<?, ?B/s] 23%|██▎       | 52224/231508 [00:00<00:00, 406559.59B/s] 53%|█████▎    | 121856/231508 [00:00<00:00, 438145.13B/s]100%|██████████| 231508/231508 [00:00<00:00, 883125.41B/s]
  0%|          | 0/433 [00:00<?, ?B/s]100%|██████████| 433/433 [00:00<00:00, 268460.26B/s]
  0%|          | 0/440473133 [00:00<?, ?B/s]  0%|          | 44032/440473133 [00:00<22:16, 329519.69B/s]  0%|          | 165888/440473133 [00:00<18:01, 406984.78B/s]  0%|          | 653312/440473133 [00:00<13:12, 554629.05B/s]  1%|          | 2540544/440473133 [00:00<09:22, 778972.81B/s]  1%|          | 5422080/440473133 [00:00<06:35, 1100071.75B/s]  2%|▏         | 8684544/440473133 [00:00<04:38, 1548180.40B/s]  3%|▎         | 12973056/440473133 [00:00<03:16, 2177988.17B/s]  4%|▎         | 16237568/440473133 [00:00<02:20, 3021615.16B/s]  5%|▍         | 20255744/440473133 [00:01<01:40, 4181809.32B/s]  5%|▌         | 23528448/440473133 [00:01<01:13, 5652256.91B/s]  6%|▋         | 27862016/440473133 [00:01<00:53, 7647141.15B/s]  7%|▋         | 31347712/440473133 [00:01<00:41, 9892953.47B/s]  8%|▊         | 35611648/440473133 [00:01<00:31, 12854547.40B/s]  9%|▉         | 39259136/440473133 [00:01<00:25, 15691842.73B/s] 10%|▉         | 43189248/440473133 [00:01<00:21, 18661931.03B/s] 11%|█         | 47112192/440473133 [00:01<00:17, 22144741.95B/s] 12%|█▏        | 50717696/440473133 [00:01<00:15, 24419085.20B/s] 13%|█▎        | 55087104/440473133 [00:02<00:13, 28143114.93B/s] 13%|█▎        | 58839040/440473133 [00:02<00:13, 29332993.92B/s] 14%|█▍        | 62718976/440473133 [00:02<00:12, 30352949.80B/s] 15%|█▌        | 66897920/440473133 [00:02<00:11, 33065533.87B/s] 16%|█▌        | 70610944/440473133 [00:02<00:11, 32941705.64B/s] 17%|█▋        | 74862592/440473133 [00:02<00:10, 35328041.37B/s] 18%|█▊        | 78643200/440473133 [00:02<00:10, 34361253.35B/s] 19%|█▊        | 82314240/440473133 [00:02<00:10, 33485737.67B/s] 20%|█▉        | 86078464/440473133 [00:02<00:10, 34632351.47B/s] 20%|██        | 89643008/440473133 [00:03<00:10, 33477749.26B/s] 21%|██▏       | 93902848/440473133 [00:03<00:09, 35774984.83B/s] 22%|██▏       | 97580032/440473133 [00:03<00:10, 34067228.22B/s] 23%|██▎       | 101073920/440473133 [00:03<00:10, 32679023.00B/s] 24%|██▍       | 105077760/440473133 [00:03<00:09, 34585256.42B/s] 25%|██▍       | 108617728/440473133 [00:03<00:10, 32907798.15B/s] 25%|██▌       | 111982592/440473133 [00:03<00:10, 31705469.62B/s] 26%|██▋       | 115937280/440473133 [00:03<00:09, 33710165.66B/s] 27%|██▋       | 119386112/440473133 [00:03<00:09, 32163278.65B/s] 28%|██▊       | 122930176/440473133 [00:04<00:10, 31535227.96B/s] 29%|██▉       | 126812160/440473133 [00:04<00:09, 33415526.33B/s] 30%|██▉       | 130220032/440473133 [00:04<00:09, 31887979.43B/s] 30%|███       | 133891072/440473133 [00:04<00:09, 31505225.08B/s] 31%|███▏      | 137895936/440473133 [00:04<00:08, 33658466.87B/s] 32%|███▏      | 141333504/440473133 [00:04<00:09, 32280472.27B/s] 33%|███▎      | 144999424/440473133 [00:04<00:09, 31613625.73B/s] 34%|███▍      | 148790272/440473133 [00:04<00:08, 33270549.77B/s] 35%|███▍      | 152173568/440473133 [00:04<00:09, 31734414.15B/s] 35%|███▌      | 155845632/440473133 [00:05<00:09, 31260929.48B/s] 36%|███▋      | 159771648/440473133 [00:05<00:08, 33295814.47B/s] 37%|███▋      | 163166208/440473133 [00:05<00:08, 31534116.74B/s] 38%|███▊      | 166675456/440473133 [00:05<00:08, 30881823.49B/s] 39%|███▊      | 170645504/440473133 [00:05<00:08, 33085567.98B/s] 40%|███▉      | 174031872/440473133 [00:05<00:08, 31480161.65B/s] 40%|████      | 177472512/440473133 [00:05<00:08, 30733527.10B/s] 41%|████      | 181486592/440473133 [00:05<00:07, 33057531.34B/s] 42%|████▏     | 184880128/440473133 [00:05<00:08, 31446961.26B/s] 43%|████▎     | 188367872/440473133 [00:06<00:08, 30782466.04B/s] 44%|████▎     | 192209920/440473133 [00:06<00:07, 32733764.33B/s] 44%|████▍     | 195557376/440473133 [00:06<00:07, 31188675.27B/s] 45%|████▌     | 199132160/440473133 [00:06<00:07, 31359236.69B/s] 46%|████▌     | 203113472/440473133 [00:06<00:07, 33490818.57B/s] 47%|████▋     | 206536704/440473133 [00:06<00:07, 31868547.15B/s] 48%|████▊     | 210404352/440473133 [00:06<00:07, 31932805.47B/s] 49%|████▊     | 214631424/440473133 [00:06<00:06, 34458194.72B/s] 50%|████▉     | 218168320/440473133 [00:06<00:06, 33268175.31B/s] 50%|█████     | 222413824/440473133 [00:07<00:06, 35576917.14B/s] 51%|█████▏    | 226067456/440473133 [00:07<00:06, 33931897.70B/s] 52%|█████▏    | 229543936/440473133 [00:07<00:06, 32930208.21B/s] 53%|█████▎    | 233615360/440473133 [00:07<00:05, 34933222.82B/s] 54%|█████▍    | 237187072/440473133 [00:07<00:06, 33577990.66B/s] 55%|█████▍    | 241189888/440473133 [00:07<00:05, 33339525.78B/s] 56%|█████▌    | 245285888/440473133 [00:07<00:05, 35309236.03B/s] 57%|█████▋    | 248881152/440473133 [00:07<00:05, 33706681.28B/s] 57%|█████▋    | 252691456/440473133 [00:07<00:05, 33031827.81B/s] 58%|█████▊    | 256649216/440473133 [00:08<00:05, 34755870.35B/s] 59%|█████▉    | 260180992/440473133 [00:08<00:05, 33316203.40B/s] 60%|█████▉    | 263968768/440473133 [00:08<00:05, 34546211.03B/s] 61%|██████    | 267471872/440473133 [00:08<00:05, 33940589.95B/s] 62%|██████▏   | 270959616/440473133 [00:08<00:05, 33226128.28B/s] 62%|██████▏   | 275000320/440473133 [00:08<00:04, 35096846.63B/s] 63%|██████▎   | 278558720/440473133 [00:08<00:04, 34089935.04B/s] 64%|██████▍   | 282920960/440473133 [00:08<00:04, 36479784.96B/s] 65%|██████▌   | 286644224/440473133 [00:08<00:04, 35076872.92B/s] 66%|██████▌   | 290522112/440473133 [00:09<00:04, 34567063.48B/s] 67%|██████▋   | 294844416/440473133 [00:09<00:03, 36775806.91B/s] 68%|██████▊   | 298593280/440473133 [00:09<00:04, 35221635.00B/s] 69%|██████▊   | 302795776/440473133 [00:09<00:03, 37018730.55B/s] 70%|██████▉   | 306567168/440473133 [00:09<00:03, 35490768.54B/s] 70%|███████   | 310232064/440473133 [00:09<00:03, 34328903.09B/s] 71%|███████▏  | 314588160/440473133 [00:09<00:03, 36658820.77B/s] 72%|███████▏  | 318333952/440473133 [00:09<00:03, 35148991.16B/s] 73%|███████▎  | 322683904/440473133 [00:09<00:03, 37293907.49B/s] 74%|███████▍  | 326498304/440473133 [00:09<00:03, 36324235.15B/s] 75%|███████▌  | 330515456/440473133 [00:10<00:03, 35939509.27B/s] 76%|███████▌  | 334808064/440473133 [00:10<00:02, 37780773.09B/s] 77%|███████▋  | 338643968/440473133 [00:10<00:02, 36686262.63B/s] 78%|███████▊  | 342688768/440473133 [00:10<00:02, 37737152.02B/s] 79%|███████▊  | 346503168/440473133 [00:10<00:02, 36977603.86B/s] 80%|███████▉  | 351011840/440473133 [00:10<00:02, 37100299.24B/s] 81%|████████  | 355566592/440473133 [00:10<00:02, 39285593.85B/s] 82%|████████▏ | 359545856/440473133 [00:10<00:02, 37864842.22B/s] 83%|████████▎ | 363931648/440473133 [00:10<00:01, 39481997.88B/s] 84%|████████▎ | 367930368/440473133 [00:11<00:01, 37747172.01B/s] 84%|████████▍ | 371757056/440473133 [00:11<00:01, 35985612.38B/s] 85%|████████▌ | 376195072/440473133 [00:11<00:01, 38148228.92B/s] 86%|████████▋ | 380083200/440473133 [00:11<00:01, 37431681.10B/s] 87%|████████▋ | 384635904/440473133 [00:11<00:01, 39540042.33B/s] 88%|████████▊ | 388660224/440473133 [00:11<00:01, 38420876.34B/s] 89%|████████▉ | 393020416/440473133 [00:11<00:01, 38158980.17B/s] 90%|█████████ | 397369344/440473133 [00:11<00:01, 39613479.53B/s] 91%|█████████ | 401373184/440473133 [00:11<00:01, 38535394.96B/s] 92%|█████████▏| 405954560/440473133 [00:12<00:00, 40463683.10B/s] 93%|█████████▎| 410051584/440473133 [00:12<00:00, 39185350.82B/s] 94%|█████████▍| 414467072/440473133 [00:12<00:00, 38625996.41B/s] 95%|█████████▌| 419015680/440473133 [00:12<00:00, 40455127.31B/s] 96%|█████████▌| 423107584/440473133 [00:12<00:00, 39348848.99B/s] 97%|█████████▋| 427658240/440473133 [00:12<00:00, 41013394.42B/s] 98%|█████████▊| 431805440/440473133 [00:12<00:00, 39823791.91B/s] 99%|█████████▉| 436275200/440473133 [00:12<00:00, 41168643.33B/s]100%|█████████▉| 440431616/440473133 [00:12<00:00, 39704057.01B/s]100%|██████████| 440473133/440473133 [00:12<00:00, 34120946.85B/s]Downloading data from https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip

   8192/7094233 [..............................] - ETA: 0s
4366336/7094233 [=================>............] - ETA: 0s
7094272/7094233 [==============================] - 0s 0us/step

Processing text_left with encode:   0%|          | 0/2118 [00:00<?, ?it/s]Processing text_left with encode:   3%|▎         | 71/2118 [00:00<00:02, 709.01it/s]Processing text_left with encode:  23%|██▎       | 492/2118 [00:00<00:01, 944.60it/s]Processing text_left with encode:  44%|████▍     | 928/2118 [00:00<00:00, 1234.69it/s]Processing text_left with encode:  62%|██████▏   | 1323/2118 [00:00<00:00, 1555.12it/s]Processing text_left with encode:  84%|████████▎ | 1773/2118 [00:00<00:00, 1934.63it/s]Processing text_left with encode: 100%|██████████| 2118/2118 [00:00<00:00, 3663.01it/s]
Processing text_right with encode:   0%|          | 0/18841 [00:00<?, ?it/s]Processing text_right with encode:   1%|          | 150/18841 [00:01<03:51, 80.84it/s]Processing text_right with encode:   1%|          | 160/18841 [00:02<04:47, 64.92it/s]Processing text_right with encode:   2%|▏         | 330/18841 [00:02<03:22, 91.24it/s]Processing text_right with encode:   3%|▎         | 495/18841 [00:02<02:24, 127.32it/s]Processing text_right with encode:   4%|▎         | 668/18841 [00:02<01:43, 176.31it/s]Processing text_right with encode:   4%|▍         | 831/18841 [00:02<01:14, 240.64it/s]Processing text_right with encode:   5%|▌         | 986/18841 [00:02<00:55, 322.28it/s]Processing text_right with encode:   6%|▌         | 1137/18841 [00:02<00:41, 421.65it/s]Processing text_right with encode:   7%|▋         | 1300/18841 [00:02<00:32, 542.13it/s]Processing text_right with encode:   8%|▊         | 1461/18841 [00:02<00:25, 676.44it/s]Processing text_right with encode:   9%|▊         | 1629/18841 [00:02<00:20, 823.32it/s]Processing text_right with encode:  10%|▉         | 1807/18841 [00:03<00:17, 981.26it/s]Processing text_right with encode:  10%|█         | 1971/18841 [00:03<00:15, 1115.05it/s]Processing text_right with encode:  11%|█▏        | 2137/18841 [00:03<00:13, 1234.09it/s]Processing text_right with encode:  12%|█▏        | 2309/18841 [00:03<00:12, 1346.69it/s]Processing text_right with encode:  13%|█▎        | 2475/18841 [00:03<00:11, 1421.90it/s]Processing text_right with encode:  14%|█▍        | 2646/18841 [00:03<00:10, 1497.06it/s]Processing text_right with encode:  15%|█▌        | 2845/18841 [00:03<00:09, 1616.83it/s]Processing text_right with encode:  16%|█▌        | 3022/18841 [00:03<00:09, 1603.37it/s]Processing text_right with encode:  17%|█▋        | 3193/18841 [00:03<00:09, 1616.36it/s]Processing text_right with encode:  18%|█▊        | 3368/18841 [00:04<00:09, 1654.05it/s]Processing text_right with encode:  19%|█▉        | 3539/18841 [00:04<00:09, 1627.13it/s]Processing text_right with encode:  20%|█▉        | 3706/18841 [00:04<00:09, 1583.48it/s]Processing text_right with encode:  21%|██        | 3868/18841 [00:04<00:09, 1573.56it/s]Processing text_right with encode:  21%|██▏       | 4045/18841 [00:04<00:09, 1625.39it/s]Processing text_right with encode:  22%|██▏       | 4210/18841 [00:04<00:09, 1609.27it/s]Processing text_right with encode:  23%|██▎       | 4379/18841 [00:04<00:08, 1632.21it/s]Processing text_right with encode:  24%|██▍       | 4544/18841 [00:04<00:08, 1628.62it/s]Processing text_right with encode:  25%|██▌       | 4711/18841 [00:04<00:08, 1637.04it/s]Processing text_right with encode:  26%|██▌       | 4890/18841 [00:04<00:08, 1677.43it/s]Processing text_right with encode:  27%|██▋       | 5076/18841 [00:05<00:07, 1724.22it/s]Processing text_right with encode:  28%|██▊       | 5254/18841 [00:05<00:07, 1739.90it/s]Processing text_right with encode:  29%|██▉       | 5429/18841 [00:05<00:08, 1675.32it/s]Processing text_right with encode:  30%|██▉       | 5598/18841 [00:05<00:08, 1562.63it/s]Processing text_right with encode:  31%|███       | 5764/18841 [00:05<00:08, 1589.76it/s]Processing text_right with encode:  31%|███▏      | 5925/18841 [00:05<00:08, 1593.96it/s]Processing text_right with encode:  32%|███▏      | 6095/18841 [00:05<00:07, 1624.07it/s]Processing text_right with encode:  33%|███▎      | 6259/18841 [00:05<00:07, 1587.86it/s]Processing text_right with encode:  34%|███▍      | 6430/18841 [00:05<00:07, 1621.04it/s]Processing text_right with encode:  35%|███▌      | 6599/18841 [00:05<00:07, 1639.00it/s]Processing text_right with encode:  36%|███▌      | 6765/18841 [00:06<00:07, 1644.00it/s]Processing text_right with encode:  37%|███▋      | 6932/18841 [00:06<00:07, 1649.09it/s]Processing text_right with encode:  38%|███▊      | 7101/18841 [00:06<00:07, 1659.94it/s]Processing text_right with encode:  39%|███▊      | 7275/18841 [00:06<00:06, 1681.37it/s]Processing text_right with encode:  40%|███▉      | 7448/18841 [00:06<00:06, 1695.26it/s]Processing text_right with encode:  40%|████      | 7618/18841 [00:06<00:06, 1658.70it/s]Processing text_right with encode:  41%|████▏     | 7789/18841 [00:06<00:06, 1669.72it/s]Processing text_right with encode:  42%|████▏     | 7964/18841 [00:06<00:06, 1685.75it/s]Processing text_right with encode:  43%|████▎     | 8133/18841 [00:06<00:06, 1679.82it/s]Processing text_right with encode:  44%|████▍     | 8306/18841 [00:07<00:06, 1692.67it/s]Processing text_right with encode:  45%|████▍     | 8476/18841 [00:07<00:06, 1679.83it/s]Processing text_right with encode:  46%|████▌     | 8645/18841 [00:07<00:06, 1650.28it/s]Processing text_right with encode:  47%|████▋     | 8811/18841 [00:07<00:06, 1642.36it/s]Processing text_right with encode:  48%|████▊     | 8976/18841 [00:07<00:06, 1604.88it/s]Processing text_right with encode:  49%|████▊     | 9141/18841 [00:07<00:06, 1616.59it/s]Processing text_right with encode:  49%|████▉     | 9304/18841 [00:07<00:05, 1619.23it/s]Processing text_right with encode:  50%|█████     | 9467/18841 [00:07<00:05, 1620.37it/s]Processing text_right with encode:  51%|█████     | 9634/18841 [00:07<00:05, 1629.03it/s]Processing text_right with encode:  52%|█████▏    | 9797/18841 [00:07<00:05, 1555.98it/s]Processing text_right with encode:  53%|█████▎    | 9970/18841 [00:08<00:05, 1604.31it/s]Processing text_right with encode:  54%|█████▍    | 10144/18841 [00:08<00:05, 1642.52it/s]Processing text_right with encode:  55%|█████▍    | 10310/18841 [00:08<00:05, 1605.20it/s]Processing text_right with encode:  56%|█████▌    | 10510/18841 [00:08<00:04, 1705.23it/s]Processing text_right with encode:  57%|█████▋    | 10683/18841 [00:08<00:04, 1667.84it/s]Processing text_right with encode:  58%|█████▊    | 10852/18841 [00:08<00:04, 1668.07it/s]Processing text_right with encode:  58%|█████▊    | 11021/18841 [00:08<00:04, 1653.47it/s]Processing text_right with encode:  59%|█████▉    | 11188/18841 [00:08<00:04, 1620.15it/s]Processing text_right with encode:  60%|██████    | 11351/18841 [00:08<00:04, 1562.93it/s]Processing text_right with encode:  61%|██████    | 11514/18841 [00:08<00:04, 1581.78it/s]Processing text_right with encode:  62%|██████▏   | 11684/18841 [00:09<00:04, 1610.94it/s]Processing text_right with encode:  63%|██████▎   | 11851/18841 [00:09<00:04, 1627.54it/s]Processing text_right with encode:  64%|██████▍   | 12015/18841 [00:09<00:04, 1608.87it/s]Processing text_right with encode:  65%|██████▍   | 12177/18841 [00:09<00:04, 1572.73it/s]Processing text_right with encode:  65%|██████▌   | 12340/18841 [00:09<00:04, 1585.95it/s]Processing text_right with encode:  66%|██████▋   | 12505/18841 [00:09<00:03, 1600.39it/s]Processing text_right with encode:  67%|██████▋   | 12666/18841 [00:09<00:03, 1598.56it/s]Processing text_right with encode:  68%|██████▊   | 12835/18841 [00:09<00:03, 1622.83it/s]Processing text_right with encode:  69%|██████▉   | 13009/18841 [00:09<00:03, 1654.08it/s]Processing text_right with encode:  70%|██████▉   | 13175/18841 [00:10<00:03, 1644.39it/s]Processing text_right with encode:  71%|███████   | 13352/18841 [00:10<00:03, 1679.64it/s]Processing text_right with encode:  72%|███████▏  | 13521/18841 [00:10<00:03, 1663.26it/s]Processing text_right with encode:  73%|███████▎  | 13695/18841 [00:10<00:03, 1685.26it/s]Processing text_right with encode:  74%|███████▎  | 13866/18841 [00:10<00:02, 1691.67it/s]Processing text_right with encode:  75%|███████▍  | 14039/18841 [00:10<00:02, 1702.40it/s]Processing text_right with encode:  75%|███████▌  | 14210/18841 [00:10<00:02, 1691.37it/s]Processing text_right with encode:  76%|███████▋  | 14380/18841 [00:10<00:02, 1688.72it/s]Processing text_right with encode:  77%|███████▋  | 14549/18841 [00:10<00:02, 1671.23it/s]Processing text_right with encode:  78%|███████▊  | 14727/18841 [00:10<00:02, 1698.58it/s]Processing text_right with encode:  79%|███████▉  | 14898/18841 [00:11<00:02, 1686.88it/s]Processing text_right with encode:  80%|████████  | 15074/18841 [00:11<00:02, 1707.01it/s]Processing text_right with encode:  81%|████████  | 15252/18841 [00:11<00:02, 1724.99it/s]Processing text_right with encode:  82%|████████▏ | 15425/18841 [00:11<00:01, 1718.95it/s]Processing text_right with encode:  83%|████████▎ | 15606/18841 [00:11<00:01, 1741.94it/s]Processing text_right with encode:  84%|████████▍ | 15787/18841 [00:11<00:01, 1758.96it/s]Processing text_right with encode:  85%|████████▍ | 15964/18841 [00:11<00:01, 1714.96it/s]Processing text_right with encode:  86%|████████▌ | 16136/18841 [00:11<00:01, 1683.50it/s]Processing text_right with encode:  87%|████████▋ | 16312/18841 [00:11<00:01, 1704.57it/s]Processing text_right with encode:  87%|████████▋ | 16483/18841 [00:11<00:01, 1694.37it/s]Processing text_right with encode:  88%|████████▊ | 16653/18841 [00:12<00:01, 1652.67it/s]Processing text_right with encode:  89%|████████▉ | 16821/18841 [00:12<00:01, 1659.81it/s]Processing text_right with encode:  90%|█████████ | 16988/18841 [00:12<00:01, 1629.36it/s]Processing text_right with encode:  91%|█████████ | 17153/18841 [00:12<00:01, 1635.05it/s]Processing text_right with encode:  92%|█████████▏| 17317/18841 [00:12<00:00, 1631.36it/s]Processing text_right with encode:  93%|█████████▎| 17481/18841 [00:12<00:00, 1590.20it/s]Processing text_right with encode:  94%|█████████▎| 17651/18841 [00:12<00:00, 1620.04it/s]Processing text_right with encode:  95%|█████████▍| 17824/18841 [00:12<00:00, 1648.00it/s]Processing text_right with encode:  96%|█████████▌| 18005/18841 [00:12<00:00, 1691.34it/s]Processing text_right with encode:  97%|█████████▋| 18185/18841 [00:12<00:00, 1721.18it/s]Processing text_right with encode:  97%|█████████▋| 18358/18841 [00:13<00:00, 1683.61it/s]Processing text_right with encode:  98%|█████████▊| 18535/18841 [00:13<00:00, 1706.43it/s]Processing text_right with encode:  99%|█████████▉| 18707/18841 [00:13<00:00, 1650.96it/s]Processing text_right with encode: 100%|██████████| 18841/18841 [00:13<00:00, 1408.89it/s]
Processing length_left with len:   0%|          | 0/2118 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 2118/2118 [00:00<00:00, 676531.56it/s]
Processing length_right with len:   0%|          | 0/18841 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 18841/18841 [00:00<00:00, 839627.30it/s]
Processing text_left with encode:   0%|          | 0/633 [00:00<?, ?it/s]Processing text_left with encode:  70%|███████   | 446/633 [00:00<00:00, 4452.99it/s]Processing text_left with encode: 100%|██████████| 633/633 [00:00<00:00, 4437.14it/s]
Processing text_right with encode:   0%|          | 0/5961 [00:00<?, ?it/s]Processing text_right with encode:   3%|▎         | 170/5961 [00:00<00:03, 1689.31it/s]Processing text_right with encode:   6%|▌         | 338/5961 [00:00<00:03, 1686.18it/s]Processing text_right with encode:   8%|▊         | 495/5961 [00:00<00:03, 1644.89it/s]Processing text_right with encode:  11%|█         | 669/5961 [00:00<00:03, 1667.93it/s]Processing text_right with encode:  14%|█▍        | 834/5961 [00:00<00:03, 1660.49it/s]Processing text_right with encode:  17%|█▋        | 1002/5961 [00:00<00:02, 1664.06it/s]Processing text_right with encode:  20%|█▉        | 1173/5961 [00:00<00:02, 1676.74it/s]Processing text_right with encode:  23%|██▎       | 1359/5961 [00:00<00:02, 1724.43it/s]Processing text_right with encode:  26%|██▌       | 1525/5961 [00:00<00:02, 1700.38it/s]Processing text_right with encode:  28%|██▊       | 1689/5961 [00:01<00:02, 1674.87it/s]Processing text_right with encode:  31%|███       | 1853/5961 [00:01<00:02, 1650.07it/s]Processing text_right with encode:  34%|███▍      | 2033/5961 [00:01<00:02, 1690.65it/s]Processing text_right with encode:  37%|███▋      | 2216/5961 [00:01<00:02, 1729.90it/s]Processing text_right with encode:  40%|████      | 2388/5961 [00:01<00:02, 1643.97it/s]Processing text_right with encode:  43%|████▎     | 2568/5961 [00:01<00:02, 1686.12it/s]Processing text_right with encode:  46%|████▌     | 2748/5961 [00:01<00:01, 1714.73it/s]Processing text_right with encode:  49%|████▉     | 2932/5961 [00:01<00:01, 1750.07it/s]Processing text_right with encode:  52%|█████▏    | 3108/5961 [00:01<00:01, 1722.10it/s]Processing text_right with encode:  55%|█████▌    | 3281/5961 [00:01<00:01, 1712.25it/s]Processing text_right with encode:  58%|█████▊    | 3453/5961 [00:02<00:01, 1687.30it/s]Processing text_right with encode:  61%|██████    | 3623/5961 [00:02<00:01, 1674.62it/s]Processing text_right with encode:  64%|██████▎   | 3791/5961 [00:02<00:01, 1622.85it/s]Processing text_right with encode:  67%|██████▋   | 3969/5961 [00:02<00:01, 1666.57it/s]Processing text_right with encode:  70%|██████▉   | 4144/5961 [00:02<00:01, 1689.74it/s]Processing text_right with encode:  72%|███████▏  | 4314/5961 [00:02<00:00, 1686.53it/s]Processing text_right with encode:  75%|███████▌  | 4484/5961 [00:02<00:00, 1686.93it/s]Processing text_right with encode:  78%|███████▊  | 4653/5961 [00:02<00:00, 1674.49it/s]Processing text_right with encode:  81%|████████  | 4821/5961 [00:02<00:00, 1660.35it/s]Processing text_right with encode:  84%|████████▎ | 4988/5961 [00:02<00:00, 1593.32it/s]Processing text_right with encode:  86%|████████▋ | 5153/5961 [00:03<00:00, 1609.65it/s]Processing text_right with encode:  89%|████████▉ | 5320/5961 [00:03<00:00, 1624.82it/s]Processing text_right with encode:  92%|█████████▏| 5486/5961 [00:03<00:00, 1631.94it/s]Processing text_right with encode:  95%|█████████▍| 5650/5961 [00:03<00:00, 1620.81it/s]Processing text_right with encode:  98%|█████████▊| 5813/5961 [00:03<00:00, 1605.63it/s]Processing text_right with encode: 100%|██████████| 5961/5961 [00:03<00:00, 1671.98it/s]
Processing length_left with len:   0%|          | 0/633 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 633/633 [00:00<00:00, 528041.85it/s]
Processing length_right with len:   0%|          | 0/5961 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 5961/5961 [00:00<00:00, 769347.23it/s]
  #### Model  fit   ############################################# 

  0%|          | 0/102 [00:00<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:28<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:28<?, ?it/s, loss=0.973]Epoch 1/1:   1%|          | 1/102 [00:28<47:10, 28.02s/it, loss=0.973]Epoch 1/1:   1%|          | 1/102 [01:10<47:10, 28.02s/it, loss=0.973]Epoch 1/1:   1%|          | 1/102 [01:10<47:10, 28.02s/it, loss=0.931]Epoch 1/1:   2%|▏         | 2/102 [01:10<54:06, 32.47s/it, loss=0.931]Epoch 1/1:   2%|▏         | 2/102 [01:28<54:06, 32.47s/it, loss=0.931]Epoch 1/1:   2%|▏         | 2/102 [01:28<54:06, 32.47s/it, loss=0.905]Epoch 1/1:   3%|▎         | 3/102 [01:28<45:59, 27.88s/it, loss=0.905]Epoch 1/1:   3%|▎         | 3/102 [02:28<45:59, 27.88s/it, loss=0.905]Epoch 1/1:   3%|▎         | 3/102 [02:28<45:59, 27.88s/it, loss=0.945]Epoch 1/1:   4%|▍         | 4/102 [02:28<1:01:17, 37.53s/it, loss=0.945]Epoch 1/1:   4%|▍         | 4/102 [03:07<1:01:17, 37.53s/it, loss=0.945]Epoch 1/1:   4%|▍         | 4/102 [03:07<1:01:17, 37.53s/it, loss=0.839]Epoch 1/1:   5%|▍         | 5/102 [03:07<1:01:39, 38.14s/it, loss=0.839]Epoch 1/1:   5%|▍         | 5/102 [06:25<1:01:39, 38.14s/it, loss=0.839]Epoch 1/1:   5%|▍         | 5/102 [06:25<1:01:39, 38.14s/it, loss=0.980]Epoch 1/1:   6%|▌         | 6/102 [06:25<2:17:28, 85.92s/it, loss=0.980]Epoch 1/1:   6%|▌         | 6/102 [08:47<2:17:28, 85.92s/it, loss=0.980]Epoch 1/1:   6%|▌         | 6/102 [08:47<2:17:28, 85.92s/it, loss=0.876]Epoch 1/1:   7%|▋         | 7/102 [08:47<2:42:51, 102.86s/it, loss=0.876]Epoch 1/1:   7%|▋         | 7/102 [10:45<2:42:51, 102.86s/it, loss=0.876]Epoch 1/1:   7%|▋         | 7/102 [10:45<2:42:51, 102.86s/it, loss=0.867]Epoch 1/1:   8%|▊         | 8/102 [10:45<2:48:24, 107.50s/it, loss=0.867]Epoch 1/1:   8%|▊         | 8/102 [13:56<2:48:24, 107.50s/it, loss=0.867]Epoch 1/1:   8%|▊         | 8/102 [13:56<2:48:24, 107.50s/it, loss=0.928]Epoch 1/1:   9%|▉         | 9/102 [13:56<3:25:29, 132.58s/it, loss=0.928]Epoch 1/1:   9%|▉         | 9/102 [16:53<3:25:29, 132.58s/it, loss=0.928]Epoch 1/1:   9%|▉         | 9/102 [16:53<3:25:29, 132.58s/it, loss=0.691]Epoch 1/1:  10%|▉         | 10/102 [16:53<3:43:24, 145.70s/it, loss=0.691]Epoch 1/1:  10%|▉         | 10/102 [19:23<3:43:24, 145.70s/it, loss=0.691]Epoch 1/1:  10%|▉         | 10/102 [19:23<3:43:24, 145.70s/it, loss=0.867]Epoch 1/1:  11%|█         | 11/102 [19:23<3:43:08, 147.12s/it, loss=0.867]Epoch 1/1:  11%|█         | 11/102 [21:18<3:43:08, 147.12s/it, loss=0.867]Epoch 1/1:  11%|█         | 11/102 [21:18<3:43:08, 147.12s/it, loss=0.713]Epoch 1/1:  12%|█▏        | 12/102 [21:18<3:26:05, 137.39s/it, loss=0.713]Epoch 1/1:  12%|█▏        | 12/102 [21:56<3:26:05, 137.39s/it, loss=0.713]Epoch 1/1:  12%|█▏        | 12/102 [21:56<3:26:05, 137.39s/it, loss=0.862]Epoch 1/1:  13%|█▎        | 13/102 [21:56<2:39:31, 107.55s/it, loss=0.862]Epoch 1/1:  13%|█▎        | 13/102 [23:06<2:39:31, 107.55s/it, loss=0.862]Epoch 1/1:  13%|█▎        | 13/102 [23:06<2:39:31, 107.55s/it, loss=0.665]Epoch 1/1:  14%|█▎        | 14/102 [23:06<2:21:11, 96.26s/it, loss=0.665] Epoch 1/1:  14%|█▎        | 14/102 [25:25<2:21:11, 96.26s/it, loss=0.665]Epoch 1/1:  14%|█▎        | 14/102 [25:25<2:21:11, 96.26s/it, loss=0.472]Epoch 1/1:  15%|█▍        | 15/102 [25:25<2:38:14, 109.13s/it, loss=0.472]Epoch 1/1:  15%|█▍        | 15/102 [28:18<2:38:14, 109.13s/it, loss=0.472]Epoch 1/1:  15%|█▍        | 15/102 [28:18<2:38:14, 109.13s/it, loss=0.622]Epoch 1/1:  16%|█▌        | 16/102 [28:18<3:03:56, 128.33s/it, loss=0.622]Epoch 1/1:  16%|█▌        | 16/102 [29:38<3:03:56, 128.33s/it, loss=0.622]Epoch 1/1:  16%|█▌        | 16/102 [29:38<3:03:56, 128.33s/it, loss=0.727]Epoch 1/1:  17%|█▋        | 17/102 [29:38<2:41:15, 113.83s/it, loss=0.727]Epoch 1/1:  17%|█▋        | 17/102 [32:13<2:41:15, 113.83s/it, loss=0.727]Epoch 1/1:  17%|█▋        | 17/102 [32:13<2:41:15, 113.83s/it, loss=0.576]Epoch 1/1:  18%|█▊        | 18/102 [32:13<2:56:37, 126.17s/it, loss=0.576]Epoch 1/1:  18%|█▊        | 18/102 [33:01<2:56:37, 126.17s/it, loss=0.576]Epoch 1/1:  18%|█▊        | 18/102 [33:01<2:56:37, 126.17s/it, loss=0.387]Epoch 1/1:  19%|█▊        | 19/102 [33:01<2:22:15, 102.84s/it, loss=0.387]Epoch 1/1:  19%|█▊        | 19/102 [36:20<2:22:15, 102.84s/it, loss=0.387]Epoch 1/1:  19%|█▊        | 19/102 [36:20<2:22:15, 102.84s/it, loss=0.452]Epoch 1/1:  20%|█▉        | 20/102 [36:20<2:59:53, 131.63s/it, loss=0.452]Epoch 1/1:  20%|█▉        | 20/102 [38:35<2:59:53, 131.63s/it, loss=0.452]Epoch 1/1:  20%|█▉        | 20/102 [38:35<2:59:53, 131.63s/it, loss=0.587]Epoch 1/1:  21%|██        | 21/102 [38:35<2:59:05, 132.66s/it, loss=0.587]Epoch 1/1:  21%|██        | 21/102 [40:18<2:59:05, 132.66s/it, loss=0.587]Epoch 1/1:  21%|██        | 21/102 [40:18<2:59:05, 132.66s/it, loss=0.525]Epoch 1/1:  22%|██▏       | 22/102 [40:18<2:45:03, 123.80s/it, loss=0.525]Epoch 1/1:  22%|██▏       | 22/102 [43:44<2:45:03, 123.80s/it, loss=0.525]Epoch 1/1:  22%|██▏       | 22/102 [43:44<2:45:03, 123.80s/it, loss=0.546]Epoch 1/1:  23%|██▎       | 23/102 [43:44<3:15:18, 148.34s/it, loss=0.546]Epoch 1/1:  23%|██▎       | 23/102 [46:07<3:15:18, 148.34s/it, loss=0.546]Epoch 1/1:  23%|██▎       | 23/102 [46:07<3:15:18, 148.34s/it, loss=0.619]Epoch 1/1:  24%|██▎       | 24/102 [46:07<3:10:48, 146.77s/it, loss=0.619]Epoch 1/1:  24%|██▎       | 24/102 [48:56<3:10:48, 146.77s/it, loss=0.619]Epoch 1/1:  24%|██▎       | 24/102 [48:56<3:10:48, 146.77s/it, loss=0.244]Epoch 1/1:  25%|██▍       | 25/102 [48:56<3:16:44, 153.31s/it, loss=0.244]Epoch 1/1:  25%|██▍       | 25/102 [50:02<3:16:44, 153.31s/it, loss=0.244]Epoch 1/1:  25%|██▍       | 25/102 [50:02<3:16:44, 153.31s/it, loss=0.266]Epoch 1/1:  25%|██▌       | 26/102 [50:02<2:41:11, 127.26s/it, loss=0.266]Epoch 1/1:  25%|██▌       | 26/102 [51:15<2:41:11, 127.26s/it, loss=0.266]Epoch 1/1:  25%|██▌       | 26/102 [51:15<2:41:11, 127.26s/it, loss=0.364]Epoch 1/1:  26%|██▋       | 27/102 [51:15<2:18:42, 110.97s/it, loss=0.364]Epoch 1/1:  26%|██▋       | 27/102 [54:04<2:18:42, 110.97s/it, loss=0.364]Epoch 1/1:  26%|██▋       | 27/102 [54:04<2:18:42, 110.97s/it, loss=0.357]Epoch 1/1:  27%|██▋       | 28/102 [54:04<2:38:18, 128.36s/it, loss=0.357]Epoch 1/1:  27%|██▋       | 28/102 [55:21<2:38:18, 128.36s/it, loss=0.357]Epoch 1/1:  27%|██▋       | 28/102 [55:21<2:38:18, 128.36s/it, loss=0.620]Epoch 1/1:  28%|██▊       | 29/102 [55:21<2:17:34, 113.07s/it, loss=0.620]Epoch 1/1:  28%|██▊       | 29/102 [57:40<2:17:34, 113.07s/it, loss=0.620]Epoch 1/1:  28%|██▊       | 29/102 [57:40<2:17:34, 113.07s/it, loss=0.284]Epoch 1/1:  29%|██▉       | 30/102 [57:40<2:24:58, 120.81s/it, loss=0.284]Epoch 1/1:  29%|██▉       | 30/102 [1:00:00<2:24:58, 120.81s/it, loss=0.284]Epoch 1/1:  29%|██▉       | 30/102 [1:00:00<2:24:58, 120.81s/it, loss=0.286]Epoch 1/1:  30%|███       | 31/102 [1:00:00<2:29:32, 126.37s/it, loss=0.286]Epoch 1/1:  30%|███       | 31/102 [1:00:55<2:29:32, 126.37s/it, loss=0.286]Epoch 1/1:  30%|███       | 31/102 [1:00:55<2:29:32, 126.37s/it, loss=0.661]Epoch 1/1:  31%|███▏      | 32/102 [1:00:55<2:02:42, 105.18s/it, loss=0.661]Epoch 1/1:  31%|███▏      | 32/102 [1:03:08<2:02:42, 105.18s/it, loss=0.661]Epoch 1/1:  31%|███▏      | 32/102 [1:03:08<2:02:42, 105.18s/it, loss=0.349]Epoch 1/1:  32%|███▏      | 33/102 [1:03:08<2:10:23, 113.39s/it, loss=0.349]Epoch 1/1:  32%|███▏      | 33/102 [1:03:51<2:10:23, 113.39s/it, loss=0.349]Epoch 1/1:  32%|███▏      | 33/102 [1:03:51<2:10:23, 113.39s/it, loss=0.440]Epoch 1/1:  33%|███▎      | 34/102 [1:03:51<1:44:36, 92.30s/it, loss=0.440] Epoch 1/1:  33%|███▎      | 34/102 [1:05:08<1:44:36, 92.30s/it, loss=0.440]Epoch 1/1:  33%|███▎      | 34/102 [1:05:08<1:44:36, 92.30s/it, loss=0.299]Epoch 1/1:  34%|███▍      | 35/102 [1:05:08<1:37:50, 87.63s/it, loss=0.299]Epoch 1/1:  34%|███▍      | 35/102 [1:08:11<1:37:50, 87.63s/it, loss=0.299]Epoch 1/1:  34%|███▍      | 35/102 [1:08:11<1:37:50, 87.63s/it, loss=0.658]Epoch 1/1:  35%|███▌      | 36/102 [1:08:11<2:07:59, 116.36s/it, loss=0.658]Killed

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   6d29d13..d62846d  master     -> origin/master
Updating 6d29d13..d62846d
Fast-forward
 ...-12_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py | 2451 ++++++++++++++++++++
 ...-10_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py |  620 +++++
 2 files changed, 3071 insertions(+)
 create mode 100644 log_benchmark/log_benchmark_2020-05-12-17-12_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py
 create mode 100644 log_pullrequest/log_pr_2020-05-12-17-10_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py
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
[master 661bfd8] ml_store
 1 file changed, 66 insertions(+)
To github.com:arita37/mlmodels_store.git
   d62846d..661bfd8  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py 

  #### Loading params   ############################################## 

  {'dataset': 'torchvision.datasets:MNIST', 'transform_uri': 'mlmodels.preprocess.image.py:torch_transform_mnist', '2nd___transform_uri': '/mnt/hgfs/d/gitdev/mlmodels/mlmodels/preprocess/image.py:torch_transform_mnist', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 4, 'test_batch_size': 1} {'checkpointdir': 'ztest/model_tch/torchhub/restnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18/'} 

  #### Loading dataset   ############################################# 
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:30, 328598.50it/s]  2%|▏         | 212992/9912422 [00:00<00:22, 424102.44it/s]  9%|▉         | 876544/9912422 [00:00<00:15, 586807.04it/s] 31%|███       | 3039232/9912422 [00:00<00:08, 826828.10it/s] 58%|█████▊    | 5750784/9912422 [00:00<00:03, 1163381.06it/s] 88%|████████▊ | 8724480/9912422 [00:00<00:00, 1629180.94it/s]9920512it [00:01, 9777820.71it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 144019.34it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 303042.25it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 395688.32it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 549095.91it/s]1654784it [00:00, 2889310.56it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 54405.84it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
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
[master d314f97] ml_store
 1 file changed, 74 insertions(+)
To github.com:arita37/mlmodels_store.git
   661bfd8..d314f97  master -> master





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
