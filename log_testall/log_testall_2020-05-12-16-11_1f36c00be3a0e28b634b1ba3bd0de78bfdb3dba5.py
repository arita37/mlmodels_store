
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
