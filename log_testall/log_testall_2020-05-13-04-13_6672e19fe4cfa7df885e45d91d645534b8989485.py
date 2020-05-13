
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
Warning: Permanently added the RSA host key for IP address '140.82.113.4' to the list of known hosts.
From github.com:arita37/mlmodels_store
   ed9bc82..ff43a54  master     -> origin/master
Updating ed9bc82..ff43a54
Fast-forward
 ...-10_6672e19fe4cfa7df885e45d91d645534b8989485.py | 622 +++++++++++++++++++++
 1 file changed, 622 insertions(+)
 create mode 100644 log_pullrequest/log_pr_2020-05-13-04-10_6672e19fe4cfa7df885e45d91d645534b8989485.py
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
[master f231422] ml_store
 1 file changed, 66 insertions(+)
 create mode 100644 log_testall/log_testall_2020-05-13-04-13_6672e19fe4cfa7df885e45d91d645534b8989485.py
To github.com:arita37/mlmodels_store.git
   ff43a54..f231422  master -> master





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
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
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
[master 754cb5a] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   f231422..754cb5a  master -> master





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
[master 590fb0e] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   754cb5a..590fb0e  master -> master





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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         16          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         4           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-13 04:13:45.407982: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 04:13:45.412836: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-13 04:13:45.413534: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ad95f56700 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 04:13:45.413554: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
100/500 [=====>........................] - ETA: 1s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 1ms/sample - loss: 0.2516 - binary_crossentropy: 0.7480 - val_loss: 0.2510 - val_binary_crossentropy: 0.7465

  #### metrics   #################################################### 
{'MSE': 0.25109249958240754}

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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         16          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         4           sparse_feature_2[0][0]           
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
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
Total params: 473
Trainable params: 473
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2557 - binary_crossentropy: 0.7049500/500 [==============================] - 1s 2ms/sample - loss: 0.2553 - binary_crossentropy: 0.7040 - val_loss: 0.2603 - val_binary_crossentropy: 0.7142

  #### metrics   #################################################### 
{'MSE': 0.25753380754474986}

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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
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
Total params: 473
Trainable params: 473
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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
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
Total params: 667
Trainable params: 667
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 3ms/sample - loss: 0.2505 - binary_crossentropy: 0.6942 - val_loss: 0.2498 - val_binary_crossentropy: 0.6927

  #### metrics   #################################################### 
{'MSE': 0.24980528503173877}

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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
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
Total params: 667
Trainable params: 667
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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         12          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_2[0][0]           
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
Total params: 393
Trainable params: 393
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.4700 - binary_crossentropy: 7.2497500/500 [==============================] - 2s 3ms/sample - loss: 0.4920 - binary_crossentropy: 7.5891 - val_loss: 0.5060 - val_binary_crossentropy: 7.8050

  #### metrics   #################################################### 
{'MSE': 0.499}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         12          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_2[0][0]           
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
Total params: 393
Trainable params: 393
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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_mean (E (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_max[0][0]               
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
Total params: 143
Trainable params: 143
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.4500 - binary_crossentropy: 6.9412500/500 [==============================] - 2s 4ms/sample - loss: 0.4780 - binary_crossentropy: 7.3731 - val_loss: 0.4920 - val_binary_crossentropy: 7.5891

  #### metrics   #################################################### 
{'MSE': 0.485}

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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_mean (E (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_max[0][0]               
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
Total params: 143
Trainable params: 143
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-13 04:15:14.620464: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 04:15:14.622661: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 04:15:14.628795: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-13 04:15:14.639802: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-13 04:15:14.641689: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 04:15:14.643401: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 04:15:14.645786: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2530 - val_binary_crossentropy: 0.6991
2020-05-13 04:15:16.119542: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 04:15:16.121630: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 04:15:16.126131: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-13 04:15:16.135576: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-13 04:15:16.137264: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 04:15:16.138698: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 04:15:16.139990: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.25380175423904}

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
2020-05-13 04:15:42.049740: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 04:15:42.051195: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 04:15:42.054793: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-13 04:15:42.061748: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-13 04:15:42.062858: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 04:15:42.063911: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 04:15:42.064853: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2512 - val_binary_crossentropy: 0.6956
2020-05-13 04:15:43.785472: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 04:15:43.786658: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 04:15:43.789320: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-13 04:15:43.794697: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-13 04:15:43.795643: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 04:15:43.796484: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 04:15:43.797233: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.25154047137759566}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-13 04:16:20.422732: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 04:16:20.428450: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 04:16:20.445299: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-13 04:16:20.473813: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-13 04:16:20.478806: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 04:16:20.483614: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 04:16:20.487993: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 6s 6s/sample - loss: 0.0914 - binary_crossentropy: 0.3601 - val_loss: 0.3340 - val_binary_crossentropy: 0.8977
2020-05-13 04:16:23.031817: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 04:16:23.036893: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 04:16:23.050647: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-13 04:16:23.079489: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-13 04:16:23.084205: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 04:16:23.088635: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 04:16:23.093129: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.23737581818359269}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_max[0][0]               
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
Total params: 680
Trainable params: 680
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.6000 - binary_crossentropy: 9.2550500/500 [==============================] - 5s 9ms/sample - loss: 0.4740 - binary_crossentropy: 7.3114 - val_loss: 0.5060 - val_binary_crossentropy: 7.8050

  #### metrics   #################################################### 
{'MSE': 0.49}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_max[0][0]               
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
Total params: 680
Trainable params: 680
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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 2)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 2)         6           sequence_mean[0][0]              
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         18          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         8           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         6           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         8           sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_mean[0][0]              
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
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_5[0][0]           
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
100/500 [=====>........................] - ETA: 7s - loss: 0.3876 - binary_crossentropy: 3.8521500/500 [==============================] - 5s 10ms/sample - loss: 0.3824 - binary_crossentropy: 4.0682 - val_loss: 0.4037 - val_binary_crossentropy: 4.2484

  #### metrics   #################################################### 
{'MSE': 0.39081547333674016}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 2)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 2)         6           sequence_mean[0][0]              
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         18          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         8           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         6           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         8           sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_mean[0][0]              
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
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_5[0][0]           
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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
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
Total params: 1,929
Trainable params: 1,929
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2644 - binary_crossentropy: 0.7229500/500 [==============================] - 5s 10ms/sample - loss: 0.2558 - binary_crossentropy: 0.7052 - val_loss: 0.2548 - val_binary_crossentropy: 0.7028

  #### metrics   #################################################### 
{'MSE': 0.25317698585822757}

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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
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
Total params: 1,929
Trainable params: 1,929
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
regionsequence_mean (InputLayer [(None, 6)]          0                                            
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
region_10sparse_seq_emb_regions (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 6, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 6, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 6, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 6, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 6, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 6, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 6, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 6, 1)         7           regionsequence_mean[0][0]        
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
Total params: 160
Trainable params: 160
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.4601 - binary_crossentropy: 7.0955500/500 [==============================] - 6s 12ms/sample - loss: 0.5041 - binary_crossentropy: 7.7742 - val_loss: 0.5161 - val_binary_crossentropy: 7.9593

  #### metrics   #################################################### 
{'MSE': 0.51}

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
regionsequence_mean (InputLayer [(None, 6)]          0                                            
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
region_10sparse_seq_emb_regions (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 6, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 6, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 6, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 6, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 6, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 6, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 6, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 6, 1)         7           regionsequence_mean[0][0]        
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         8           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         2           sequence_max[0][0]               
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
Total params: 1,387
Trainable params: 1,387
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.3177 - binary_crossentropy: 2.7754500/500 [==============================] - 6s 12ms/sample - loss: 0.3090 - binary_crossentropy: 2.4004 - val_loss: 0.3194 - val_binary_crossentropy: 2.5275

  #### metrics   #################################################### 
{'MSE': 0.31412696875454377}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         8           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         2           sequence_max[0][0]               
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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         36          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         28          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 1, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         28          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 1, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         28          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         28          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 1, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 1, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           hash_11[0][0]                    
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
Total params: 3,086
Trainable params: 3,006
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2556 - binary_crossentropy: 0.7058500/500 [==============================] - 7s 13ms/sample - loss: 0.2599 - binary_crossentropy: 0.7138 - val_loss: 0.2579 - val_binary_crossentropy: 0.7092

  #### metrics   #################################################### 
{'MSE': 0.2569856454687766}

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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         36          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         28          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 1, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         28          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 1, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         28          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         28          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 1, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 1, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           hash_11[0][0]                    
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
Total params: 3,086
Trainable params: 3,006
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
100/500 [=====>........................] - ETA: 8s - loss: 0.2540 - binary_crossentropy: 0.7014500/500 [==============================] - 6s 13ms/sample - loss: 0.2522 - binary_crossentropy: 0.6978 - val_loss: 0.2507 - val_binary_crossentropy: 0.6945

  #### metrics   #################################################### 
{'MSE': 0.250879513584138}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_44 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_max[0][0]               
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
Total params: 2,014
Trainable params: 2,014
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2550 - binary_crossentropy: 0.8335500/500 [==============================] - 7s 14ms/sample - loss: 0.2590 - binary_crossentropy: 0.8952 - val_loss: 0.2561 - val_binary_crossentropy: 0.8871

  #### metrics   #################################################### 
{'MSE': 0.2574320699282198}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_44 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_max[0][0]               
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
Total params: 2,014
Trainable params: 2,014
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_max[0][0]               
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
Total params: 316
Trainable params: 316
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2433 - binary_crossentropy: 0.6804500/500 [==============================] - 7s 14ms/sample - loss: 0.2639 - binary_crossentropy: 0.7222 - val_loss: 0.2578 - val_binary_crossentropy: 0.7098

  #### metrics   #################################################### 
{'MSE': 0.25997605720470635}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_max[0][0]               
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
Total params: 316
Trainable params: 316
Non-trainable params: 0
__________________________________________________________________________________________________

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
From github.com:arita37/mlmodels_store
   590fb0e..1399040  master     -> origin/master
Updating 590fb0e..1399040
Fast-forward
 error_list/20200513/list_log_jupyter_20200513.md | 2256 +++++++++++-----------
 error_list/20200513/list_log_testall_20200513.md |  795 +-------
 2 files changed, 1136 insertions(+), 1915 deletions(-)
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
[master 16838ac] ml_store
 1 file changed, 5668 insertions(+)
To github.com:arita37/mlmodels_store.git
   1399040..16838ac  master -> master





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
[master dc990d6] ml_store
 1 file changed, 50 insertions(+)
To github.com:arita37/mlmodels_store.git
   16838ac..dc990d6  master -> master





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
[master e0d5ef7] ml_store
 1 file changed, 46 insertions(+)
To github.com:arita37/mlmodels_store.git
   dc990d6..e0d5ef7  master -> master





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
[master 556f9f3] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   e0d5ef7..556f9f3  master -> master





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

2020-05-13 04:29:40.584495: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 04:29:40.590076: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-13 04:29:40.590250: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55efe545bd40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 04:29:40.590264: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 5s - loss: 1.3863
256/354 [====================>.........] - ETA: 2s - loss: 1.2866
354/354 [==============================] - 10s 28ms/step - loss: 1.2403 - val_loss: 1.7966

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
[master 3127bec] ml_store
 1 file changed, 149 insertions(+)
To github.com:arita37/mlmodels_store.git
   556f9f3..3127bec  master -> master





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
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
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
[master 9990dee] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   3127bec..9990dee  master -> master





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
[master d6f26b6] ml_store
 1 file changed, 44 insertions(+)
To github.com:arita37/mlmodels_store.git
   9990dee..d6f26b6  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
  737280/17464789 [>.............................] - ETA: 1s
 2809856/17464789 [===>..........................] - ETA: 0s
 5627904/17464789 [========>.....................] - ETA: 0s
 8478720/17464789 [=============>................] - ETA: 0s
12402688/17464789 [====================>.........] - ETA: 0s
17342464/17464789 [============================>.] - ETA: 0s
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
2020-05-13 04:30:38.068344: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 04:30:38.072842: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-13 04:30:38.073006: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561414ed4830 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 04:30:38.073021: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.5286 - accuracy: 0.5090
 2000/25000 [=>............................] - ETA: 8s - loss: 7.4366 - accuracy: 0.5150 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.5593 - accuracy: 0.5070
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6436 - accuracy: 0.5015
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5961 - accuracy: 0.5046
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6155 - accuracy: 0.5033
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6469 - accuracy: 0.5013
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6110 - accuracy: 0.5036
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6326 - accuracy: 0.5022
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6421 - accuracy: 0.5016
11000/25000 [============>.................] - ETA: 3s - loss: 7.6485 - accuracy: 0.5012
12000/25000 [=============>................] - ETA: 3s - loss: 7.6947 - accuracy: 0.4982
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7008 - accuracy: 0.4978
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7170 - accuracy: 0.4967
15000/25000 [=================>............] - ETA: 2s - loss: 7.7055 - accuracy: 0.4975
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7097 - accuracy: 0.4972
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6991 - accuracy: 0.4979
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6871 - accuracy: 0.4987
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6852 - accuracy: 0.4988
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6812 - accuracy: 0.4990
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6732 - accuracy: 0.4996
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6638 - accuracy: 0.5002
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6628 - accuracy: 0.5002
25000/25000 [==============================] - 7s 296us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7f6d8801cda0>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f6d5a748ef0> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.8660 - accuracy: 0.4870
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7893 - accuracy: 0.4920 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7995 - accuracy: 0.4913
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6705 - accuracy: 0.4997
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6881 - accuracy: 0.4986
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6513 - accuracy: 0.5010
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6951 - accuracy: 0.4981
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6647 - accuracy: 0.5001
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6820 - accuracy: 0.4990
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6697 - accuracy: 0.4998
11000/25000 [============>.................] - ETA: 3s - loss: 7.6541 - accuracy: 0.5008
12000/25000 [=============>................] - ETA: 3s - loss: 7.6487 - accuracy: 0.5012
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6513 - accuracy: 0.5010
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6327 - accuracy: 0.5022
15000/25000 [=================>............] - ETA: 2s - loss: 7.6441 - accuracy: 0.5015
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6369 - accuracy: 0.5019
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6396 - accuracy: 0.5018
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6504 - accuracy: 0.5011
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6247 - accuracy: 0.5027
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6260 - accuracy: 0.5027
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6192 - accuracy: 0.5031
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6346 - accuracy: 0.5021
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6560 - accuracy: 0.5007
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 7s 296us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 11s - loss: 7.6206 - accuracy: 0.5030
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6360 - accuracy: 0.5020 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.5848 - accuracy: 0.5053
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6513 - accuracy: 0.5010
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6789 - accuracy: 0.4992
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7126 - accuracy: 0.4970
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6798 - accuracy: 0.4991
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6896 - accuracy: 0.4985
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6581 - accuracy: 0.5006
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6237 - accuracy: 0.5028
11000/25000 [============>.................] - ETA: 3s - loss: 7.5927 - accuracy: 0.5048
12000/25000 [=============>................] - ETA: 3s - loss: 7.6181 - accuracy: 0.5032
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6808 - accuracy: 0.4991
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6951 - accuracy: 0.4981
15000/25000 [=================>............] - ETA: 2s - loss: 7.6952 - accuracy: 0.4981
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7002 - accuracy: 0.4978
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6783 - accuracy: 0.4992
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6581 - accuracy: 0.5006
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6569 - accuracy: 0.5006
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6682 - accuracy: 0.4999
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6557 - accuracy: 0.5007
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6624 - accuracy: 0.5003
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6686 - accuracy: 0.4999
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 8s 301us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
Warning: Permanently added the RSA host key for IP address '140.82.113.3' to the list of known hosts.
From github.com:arita37/mlmodels_store
   d6f26b6..2cbe563  master     -> origin/master
Updating d6f26b6..2cbe563
Fast-forward
 error_list/20200513/list_log_benchmark_20200513.md | 186 ++++++++++-----------
 error_list/20200513/list_log_testall_20200513.md   |  83 +++++++++
 2 files changed, 171 insertions(+), 98 deletions(-)
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
[master 047944b] ml_store
 1 file changed, 327 insertions(+)
To github.com:arita37/mlmodels_store.git
   2cbe563..047944b  master -> master





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

13/13 [==============================] - 2s 153ms/step - loss: nan
Epoch 2/10

13/13 [==============================] - 0s 5ms/step - loss: nan
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
[master 9c69c07] ml_store
 1 file changed, 125 insertions(+)
To github.com:arita37/mlmodels_store.git
   047944b..9c69c07  master -> master





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
 1245184/11490434 [==>...........................] - ETA: 0s
 3039232/11490434 [======>.......................] - ETA: 0s
 7200768/11490434 [=================>............] - ETA: 0s
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

   32/60000 [..............................] - ETA: 8:54 - loss: 2.3096 - categorical_accuracy: 0.1250
   64/60000 [..............................] - ETA: 5:18 - loss: 2.3056 - categorical_accuracy: 0.0938
  128/60000 [..............................] - ETA: 3:23 - loss: 2.2822 - categorical_accuracy: 0.1484
  192/60000 [..............................] - ETA: 2:44 - loss: 2.2287 - categorical_accuracy: 0.2448
  256/60000 [..............................] - ETA: 2:25 - loss: 2.1638 - categorical_accuracy: 0.2812
  320/60000 [..............................] - ETA: 2:14 - loss: 2.1230 - categorical_accuracy: 0.2875
  352/60000 [..............................] - ETA: 2:10 - loss: 2.1088 - categorical_accuracy: 0.2983
  416/60000 [..............................] - ETA: 2:04 - loss: 2.0746 - categorical_accuracy: 0.3149
  480/60000 [..............................] - ETA: 1:59 - loss: 1.9912 - categorical_accuracy: 0.3438
  544/60000 [..............................] - ETA: 1:55 - loss: 1.9288 - categorical_accuracy: 0.3566
  608/60000 [..............................] - ETA: 1:52 - loss: 1.8888 - categorical_accuracy: 0.3618
  672/60000 [..............................] - ETA: 1:49 - loss: 1.8353 - categorical_accuracy: 0.3869
  736/60000 [..............................] - ETA: 1:47 - loss: 1.7719 - categorical_accuracy: 0.4076
  800/60000 [..............................] - ETA: 1:46 - loss: 1.7119 - categorical_accuracy: 0.4313
  864/60000 [..............................] - ETA: 1:44 - loss: 1.6556 - categorical_accuracy: 0.4514
  928/60000 [..............................] - ETA: 1:43 - loss: 1.6200 - categorical_accuracy: 0.4591
  992/60000 [..............................] - ETA: 1:42 - loss: 1.5801 - categorical_accuracy: 0.4677
 1024/60000 [..............................] - ETA: 1:42 - loss: 1.5594 - categorical_accuracy: 0.4746
 1088/60000 [..............................] - ETA: 1:40 - loss: 1.5451 - categorical_accuracy: 0.4835
 1152/60000 [..............................] - ETA: 1:39 - loss: 1.5040 - categorical_accuracy: 0.4965
 1216/60000 [..............................] - ETA: 1:39 - loss: 1.4686 - categorical_accuracy: 0.5090
 1280/60000 [..............................] - ETA: 1:38 - loss: 1.4432 - categorical_accuracy: 0.5172
 1344/60000 [..............................] - ETA: 1:37 - loss: 1.4078 - categorical_accuracy: 0.5268
 1408/60000 [..............................] - ETA: 1:36 - loss: 1.3766 - categorical_accuracy: 0.5369
 1472/60000 [..............................] - ETA: 1:36 - loss: 1.3510 - categorical_accuracy: 0.5448
 1536/60000 [..............................] - ETA: 1:35 - loss: 1.3249 - categorical_accuracy: 0.5560
 1600/60000 [..............................] - ETA: 1:35 - loss: 1.3045 - categorical_accuracy: 0.5619
 1664/60000 [..............................] - ETA: 1:35 - loss: 1.2826 - categorical_accuracy: 0.5697
 1728/60000 [..............................] - ETA: 1:34 - loss: 1.2601 - categorical_accuracy: 0.5770
 1792/60000 [..............................] - ETA: 1:34 - loss: 1.2383 - categorical_accuracy: 0.5848
 1856/60000 [..............................] - ETA: 1:33 - loss: 1.2254 - categorical_accuracy: 0.5921
 1920/60000 [..............................] - ETA: 1:33 - loss: 1.2040 - categorical_accuracy: 0.5990
 1952/60000 [..............................] - ETA: 1:33 - loss: 1.1942 - categorical_accuracy: 0.6025
 2016/60000 [>.............................] - ETA: 1:33 - loss: 1.1704 - categorical_accuracy: 0.6086
 2080/60000 [>.............................] - ETA: 1:33 - loss: 1.1477 - categorical_accuracy: 0.6168
 2112/60000 [>.............................] - ETA: 1:33 - loss: 1.1377 - categorical_accuracy: 0.6198
 2176/60000 [>.............................] - ETA: 1:32 - loss: 1.1200 - categorical_accuracy: 0.6273
 2240/60000 [>.............................] - ETA: 1:32 - loss: 1.1041 - categorical_accuracy: 0.6335
 2304/60000 [>.............................] - ETA: 1:32 - loss: 1.0831 - categorical_accuracy: 0.6406
 2368/60000 [>.............................] - ETA: 1:31 - loss: 1.0782 - categorical_accuracy: 0.6432
 2432/60000 [>.............................] - ETA: 1:31 - loss: 1.0599 - categorical_accuracy: 0.6488
 2496/60000 [>.............................] - ETA: 1:31 - loss: 1.0451 - categorical_accuracy: 0.6542
 2560/60000 [>.............................] - ETA: 1:31 - loss: 1.0361 - categorical_accuracy: 0.6570
 2592/60000 [>.............................] - ETA: 1:31 - loss: 1.0262 - categorical_accuracy: 0.6601
 2656/60000 [>.............................] - ETA: 1:31 - loss: 1.0138 - categorical_accuracy: 0.6642
 2720/60000 [>.............................] - ETA: 1:31 - loss: 1.0002 - categorical_accuracy: 0.6699
 2752/60000 [>.............................] - ETA: 1:30 - loss: 0.9933 - categorical_accuracy: 0.6719
 2816/60000 [>.............................] - ETA: 1:30 - loss: 0.9760 - categorical_accuracy: 0.6779
 2848/60000 [>.............................] - ETA: 1:30 - loss: 0.9706 - categorical_accuracy: 0.6794
 2912/60000 [>.............................] - ETA: 1:30 - loss: 0.9573 - categorical_accuracy: 0.6844
 2976/60000 [>.............................] - ETA: 1:30 - loss: 0.9487 - categorical_accuracy: 0.6882
 3040/60000 [>.............................] - ETA: 1:30 - loss: 0.9358 - categorical_accuracy: 0.6924
 3104/60000 [>.............................] - ETA: 1:29 - loss: 0.9262 - categorical_accuracy: 0.6959
 3168/60000 [>.............................] - ETA: 1:29 - loss: 0.9140 - categorical_accuracy: 0.7001
 3232/60000 [>.............................] - ETA: 1:29 - loss: 0.9029 - categorical_accuracy: 0.7030
 3264/60000 [>.............................] - ETA: 1:29 - loss: 0.8999 - categorical_accuracy: 0.7044
 3328/60000 [>.............................] - ETA: 1:29 - loss: 0.8917 - categorical_accuracy: 0.7064
 3392/60000 [>.............................] - ETA: 1:28 - loss: 0.8881 - categorical_accuracy: 0.7081
 3456/60000 [>.............................] - ETA: 1:28 - loss: 0.8788 - categorical_accuracy: 0.7121
 3520/60000 [>.............................] - ETA: 1:28 - loss: 0.8704 - categorical_accuracy: 0.7151
 3552/60000 [>.............................] - ETA: 1:28 - loss: 0.8674 - categorical_accuracy: 0.7159
 3584/60000 [>.............................] - ETA: 1:28 - loss: 0.8655 - categorical_accuracy: 0.7168
 3648/60000 [>.............................] - ETA: 1:28 - loss: 0.8553 - categorical_accuracy: 0.7207
 3680/60000 [>.............................] - ETA: 1:28 - loss: 0.8504 - categorical_accuracy: 0.7226
 3744/60000 [>.............................] - ETA: 1:28 - loss: 0.8398 - categorical_accuracy: 0.7260
 3808/60000 [>.............................] - ETA: 1:27 - loss: 0.8314 - categorical_accuracy: 0.7285
 3872/60000 [>.............................] - ETA: 1:27 - loss: 0.8251 - categorical_accuracy: 0.7301
 3936/60000 [>.............................] - ETA: 1:27 - loss: 0.8164 - categorical_accuracy: 0.7327
 3968/60000 [>.............................] - ETA: 1:27 - loss: 0.8134 - categorical_accuracy: 0.7344
 4032/60000 [=>............................] - ETA: 1:27 - loss: 0.8086 - categorical_accuracy: 0.7366
 4096/60000 [=>............................] - ETA: 1:27 - loss: 0.8018 - categorical_accuracy: 0.7390
 4160/60000 [=>............................] - ETA: 1:26 - loss: 0.7939 - categorical_accuracy: 0.7416
 4224/60000 [=>............................] - ETA: 1:26 - loss: 0.7849 - categorical_accuracy: 0.7446
 4256/60000 [=>............................] - ETA: 1:26 - loss: 0.7806 - categorical_accuracy: 0.7460
 4320/60000 [=>............................] - ETA: 1:26 - loss: 0.7730 - categorical_accuracy: 0.7486
 4384/60000 [=>............................] - ETA: 1:26 - loss: 0.7685 - categorical_accuracy: 0.7511
 4448/60000 [=>............................] - ETA: 1:26 - loss: 0.7609 - categorical_accuracy: 0.7536
 4512/60000 [=>............................] - ETA: 1:26 - loss: 0.7556 - categorical_accuracy: 0.7547
 4576/60000 [=>............................] - ETA: 1:26 - loss: 0.7480 - categorical_accuracy: 0.7572
 4640/60000 [=>............................] - ETA: 1:25 - loss: 0.7399 - categorical_accuracy: 0.7601
 4704/60000 [=>............................] - ETA: 1:25 - loss: 0.7346 - categorical_accuracy: 0.7621
 4768/60000 [=>............................] - ETA: 1:25 - loss: 0.7292 - categorical_accuracy: 0.7641
 4800/60000 [=>............................] - ETA: 1:25 - loss: 0.7259 - categorical_accuracy: 0.7652
 4832/60000 [=>............................] - ETA: 1:25 - loss: 0.7223 - categorical_accuracy: 0.7661
 4896/60000 [=>............................] - ETA: 1:25 - loss: 0.7188 - categorical_accuracy: 0.7674
 4928/60000 [=>............................] - ETA: 1:25 - loss: 0.7166 - categorical_accuracy: 0.7681
 4992/60000 [=>............................] - ETA: 1:25 - loss: 0.7097 - categorical_accuracy: 0.7702
 5056/60000 [=>............................] - ETA: 1:24 - loss: 0.7054 - categorical_accuracy: 0.7722
 5120/60000 [=>............................] - ETA: 1:24 - loss: 0.6994 - categorical_accuracy: 0.7740
 5152/60000 [=>............................] - ETA: 1:24 - loss: 0.6962 - categorical_accuracy: 0.7750
 5184/60000 [=>............................] - ETA: 1:24 - loss: 0.6949 - categorical_accuracy: 0.7751
 5248/60000 [=>............................] - ETA: 1:24 - loss: 0.6892 - categorical_accuracy: 0.7771
 5312/60000 [=>............................] - ETA: 1:24 - loss: 0.6838 - categorical_accuracy: 0.7792
 5376/60000 [=>............................] - ETA: 1:24 - loss: 0.6794 - categorical_accuracy: 0.7801
 5440/60000 [=>............................] - ETA: 1:24 - loss: 0.6742 - categorical_accuracy: 0.7822
 5504/60000 [=>............................] - ETA: 1:24 - loss: 0.6689 - categorical_accuracy: 0.7838
 5536/60000 [=>............................] - ETA: 1:24 - loss: 0.6659 - categorical_accuracy: 0.7849
 5568/60000 [=>............................] - ETA: 1:24 - loss: 0.6646 - categorical_accuracy: 0.7854
 5600/60000 [=>............................] - ETA: 1:23 - loss: 0.6631 - categorical_accuracy: 0.7859
 5632/60000 [=>............................] - ETA: 1:23 - loss: 0.6632 - categorical_accuracy: 0.7864
 5696/60000 [=>............................] - ETA: 1:23 - loss: 0.6586 - categorical_accuracy: 0.7883
 5728/60000 [=>............................] - ETA: 1:23 - loss: 0.6557 - categorical_accuracy: 0.7891
 5792/60000 [=>............................] - ETA: 1:23 - loss: 0.6529 - categorical_accuracy: 0.7901
 5856/60000 [=>............................] - ETA: 1:23 - loss: 0.6480 - categorical_accuracy: 0.7917
 5920/60000 [=>............................] - ETA: 1:23 - loss: 0.6444 - categorical_accuracy: 0.7927
 5952/60000 [=>............................] - ETA: 1:23 - loss: 0.6430 - categorical_accuracy: 0.7930
 6016/60000 [==>...........................] - ETA: 1:23 - loss: 0.6375 - categorical_accuracy: 0.7949
 6080/60000 [==>...........................] - ETA: 1:23 - loss: 0.6338 - categorical_accuracy: 0.7962
 6144/60000 [==>...........................] - ETA: 1:23 - loss: 0.6306 - categorical_accuracy: 0.7975
 6176/60000 [==>...........................] - ETA: 1:23 - loss: 0.6278 - categorical_accuracy: 0.7984
 6208/60000 [==>...........................] - ETA: 1:23 - loss: 0.6257 - categorical_accuracy: 0.7990
 6272/60000 [==>...........................] - ETA: 1:22 - loss: 0.6213 - categorical_accuracy: 0.8004
 6336/60000 [==>...........................] - ETA: 1:22 - loss: 0.6199 - categorical_accuracy: 0.8011
 6368/60000 [==>...........................] - ETA: 1:22 - loss: 0.6182 - categorical_accuracy: 0.8018
 6432/60000 [==>...........................] - ETA: 1:22 - loss: 0.6142 - categorical_accuracy: 0.8032
 6496/60000 [==>...........................] - ETA: 1:22 - loss: 0.6103 - categorical_accuracy: 0.8043
 6560/60000 [==>...........................] - ETA: 1:22 - loss: 0.6073 - categorical_accuracy: 0.8052
 6592/60000 [==>...........................] - ETA: 1:22 - loss: 0.6067 - categorical_accuracy: 0.8052
 6656/60000 [==>...........................] - ETA: 1:22 - loss: 0.6038 - categorical_accuracy: 0.8059
 6720/60000 [==>...........................] - ETA: 1:21 - loss: 0.6011 - categorical_accuracy: 0.8068
 6784/60000 [==>...........................] - ETA: 1:21 - loss: 0.5967 - categorical_accuracy: 0.8082
 6816/60000 [==>...........................] - ETA: 1:21 - loss: 0.5949 - categorical_accuracy: 0.8088
 6880/60000 [==>...........................] - ETA: 1:21 - loss: 0.5917 - categorical_accuracy: 0.8100
 6944/60000 [==>...........................] - ETA: 1:21 - loss: 0.5879 - categorical_accuracy: 0.8113
 6976/60000 [==>...........................] - ETA: 1:21 - loss: 0.5862 - categorical_accuracy: 0.8118
 7040/60000 [==>...........................] - ETA: 1:21 - loss: 0.5835 - categorical_accuracy: 0.8129
 7104/60000 [==>...........................] - ETA: 1:21 - loss: 0.5811 - categorical_accuracy: 0.8138
 7168/60000 [==>...........................] - ETA: 1:21 - loss: 0.5770 - categorical_accuracy: 0.8150
 7232/60000 [==>...........................] - ETA: 1:21 - loss: 0.5738 - categorical_accuracy: 0.8160
 7264/60000 [==>...........................] - ETA: 1:20 - loss: 0.5732 - categorical_accuracy: 0.8164
 7328/60000 [==>...........................] - ETA: 1:20 - loss: 0.5698 - categorical_accuracy: 0.8171
 7392/60000 [==>...........................] - ETA: 1:20 - loss: 0.5668 - categorical_accuracy: 0.8182
 7456/60000 [==>...........................] - ETA: 1:20 - loss: 0.5635 - categorical_accuracy: 0.8192
 7520/60000 [==>...........................] - ETA: 1:20 - loss: 0.5625 - categorical_accuracy: 0.8198
 7552/60000 [==>...........................] - ETA: 1:20 - loss: 0.5609 - categorical_accuracy: 0.8203
 7584/60000 [==>...........................] - ETA: 1:20 - loss: 0.5595 - categorical_accuracy: 0.8208
 7616/60000 [==>...........................] - ETA: 1:20 - loss: 0.5591 - categorical_accuracy: 0.8210
 7680/60000 [==>...........................] - ETA: 1:20 - loss: 0.5563 - categorical_accuracy: 0.8219
 7744/60000 [==>...........................] - ETA: 1:20 - loss: 0.5539 - categorical_accuracy: 0.8227
 7808/60000 [==>...........................] - ETA: 1:20 - loss: 0.5508 - categorical_accuracy: 0.8236
 7872/60000 [==>...........................] - ETA: 1:19 - loss: 0.5491 - categorical_accuracy: 0.8244
 7936/60000 [==>...........................] - ETA: 1:19 - loss: 0.5456 - categorical_accuracy: 0.8256
 8000/60000 [===>..........................] - ETA: 1:19 - loss: 0.5462 - categorical_accuracy: 0.8260
 8064/60000 [===>..........................] - ETA: 1:19 - loss: 0.5442 - categorical_accuracy: 0.8269
 8128/60000 [===>..........................] - ETA: 1:19 - loss: 0.5428 - categorical_accuracy: 0.8274
 8192/60000 [===>..........................] - ETA: 1:19 - loss: 0.5404 - categorical_accuracy: 0.8282
 8256/60000 [===>..........................] - ETA: 1:19 - loss: 0.5384 - categorical_accuracy: 0.8289
 8320/60000 [===>..........................] - ETA: 1:19 - loss: 0.5368 - categorical_accuracy: 0.8294
 8384/60000 [===>..........................] - ETA: 1:19 - loss: 0.5343 - categorical_accuracy: 0.8304
 8416/60000 [===>..........................] - ETA: 1:19 - loss: 0.5335 - categorical_accuracy: 0.8306
 8480/60000 [===>..........................] - ETA: 1:19 - loss: 0.5310 - categorical_accuracy: 0.8313
 8544/60000 [===>..........................] - ETA: 1:18 - loss: 0.5280 - categorical_accuracy: 0.8322
 8608/60000 [===>..........................] - ETA: 1:18 - loss: 0.5267 - categorical_accuracy: 0.8328
 8672/60000 [===>..........................] - ETA: 1:18 - loss: 0.5244 - categorical_accuracy: 0.8336
 8736/60000 [===>..........................] - ETA: 1:18 - loss: 0.5218 - categorical_accuracy: 0.8345
 8800/60000 [===>..........................] - ETA: 1:18 - loss: 0.5206 - categorical_accuracy: 0.8351
 8864/60000 [===>..........................] - ETA: 1:18 - loss: 0.5177 - categorical_accuracy: 0.8360
 8928/60000 [===>..........................] - ETA: 1:18 - loss: 0.5164 - categorical_accuracy: 0.8364
 8992/60000 [===>..........................] - ETA: 1:18 - loss: 0.5146 - categorical_accuracy: 0.8369
 9056/60000 [===>..........................] - ETA: 1:17 - loss: 0.5122 - categorical_accuracy: 0.8376
 9120/60000 [===>..........................] - ETA: 1:17 - loss: 0.5107 - categorical_accuracy: 0.8380
 9184/60000 [===>..........................] - ETA: 1:17 - loss: 0.5083 - categorical_accuracy: 0.8387
 9216/60000 [===>..........................] - ETA: 1:17 - loss: 0.5075 - categorical_accuracy: 0.8392
 9280/60000 [===>..........................] - ETA: 1:17 - loss: 0.5053 - categorical_accuracy: 0.8399
 9344/60000 [===>..........................] - ETA: 1:17 - loss: 0.5030 - categorical_accuracy: 0.8405
 9408/60000 [===>..........................] - ETA: 1:17 - loss: 0.5027 - categorical_accuracy: 0.8409
 9440/60000 [===>..........................] - ETA: 1:17 - loss: 0.5014 - categorical_accuracy: 0.8413
 9504/60000 [===>..........................] - ETA: 1:17 - loss: 0.4993 - categorical_accuracy: 0.8419
 9568/60000 [===>..........................] - ETA: 1:17 - loss: 0.4971 - categorical_accuracy: 0.8428
 9632/60000 [===>..........................] - ETA: 1:17 - loss: 0.4948 - categorical_accuracy: 0.8438
 9696/60000 [===>..........................] - ETA: 1:16 - loss: 0.4922 - categorical_accuracy: 0.8445
 9760/60000 [===>..........................] - ETA: 1:16 - loss: 0.4897 - categorical_accuracy: 0.8454
 9792/60000 [===>..........................] - ETA: 1:16 - loss: 0.4899 - categorical_accuracy: 0.8455
 9856/60000 [===>..........................] - ETA: 1:16 - loss: 0.4876 - categorical_accuracy: 0.8460
 9920/60000 [===>..........................] - ETA: 1:16 - loss: 0.4858 - categorical_accuracy: 0.8466
 9984/60000 [===>..........................] - ETA: 1:16 - loss: 0.4833 - categorical_accuracy: 0.8475
10016/60000 [====>.........................] - ETA: 1:16 - loss: 0.4823 - categorical_accuracy: 0.8477
10080/60000 [====>.........................] - ETA: 1:16 - loss: 0.4807 - categorical_accuracy: 0.8483
10144/60000 [====>.........................] - ETA: 1:16 - loss: 0.4796 - categorical_accuracy: 0.8490
10208/60000 [====>.........................] - ETA: 1:15 - loss: 0.4782 - categorical_accuracy: 0.8493
10272/60000 [====>.........................] - ETA: 1:15 - loss: 0.4778 - categorical_accuracy: 0.8493
10336/60000 [====>.........................] - ETA: 1:15 - loss: 0.4766 - categorical_accuracy: 0.8497
10400/60000 [====>.........................] - ETA: 1:15 - loss: 0.4749 - categorical_accuracy: 0.8502
10464/60000 [====>.........................] - ETA: 1:15 - loss: 0.4731 - categorical_accuracy: 0.8507
10528/60000 [====>.........................] - ETA: 1:15 - loss: 0.4720 - categorical_accuracy: 0.8511
10592/60000 [====>.........................] - ETA: 1:15 - loss: 0.4706 - categorical_accuracy: 0.8513
10656/60000 [====>.........................] - ETA: 1:15 - loss: 0.4688 - categorical_accuracy: 0.8516
10720/60000 [====>.........................] - ETA: 1:15 - loss: 0.4670 - categorical_accuracy: 0.8521
10784/60000 [====>.........................] - ETA: 1:14 - loss: 0.4649 - categorical_accuracy: 0.8528
10848/60000 [====>.........................] - ETA: 1:14 - loss: 0.4632 - categorical_accuracy: 0.8532
10912/60000 [====>.........................] - ETA: 1:14 - loss: 0.4614 - categorical_accuracy: 0.8539
10944/60000 [====>.........................] - ETA: 1:14 - loss: 0.4608 - categorical_accuracy: 0.8540
11008/60000 [====>.........................] - ETA: 1:14 - loss: 0.4594 - categorical_accuracy: 0.8544
11072/60000 [====>.........................] - ETA: 1:14 - loss: 0.4580 - categorical_accuracy: 0.8549
11136/60000 [====>.........................] - ETA: 1:14 - loss: 0.4568 - categorical_accuracy: 0.8554
11200/60000 [====>.........................] - ETA: 1:14 - loss: 0.4564 - categorical_accuracy: 0.8557
11264/60000 [====>.........................] - ETA: 1:14 - loss: 0.4565 - categorical_accuracy: 0.8559
11328/60000 [====>.........................] - ETA: 1:13 - loss: 0.4548 - categorical_accuracy: 0.8566
11392/60000 [====>.........................] - ETA: 1:13 - loss: 0.4529 - categorical_accuracy: 0.8573
11456/60000 [====>.........................] - ETA: 1:13 - loss: 0.4510 - categorical_accuracy: 0.8578
11520/60000 [====>.........................] - ETA: 1:13 - loss: 0.4511 - categorical_accuracy: 0.8579
11584/60000 [====>.........................] - ETA: 1:13 - loss: 0.4496 - categorical_accuracy: 0.8583
11648/60000 [====>.........................] - ETA: 1:13 - loss: 0.4480 - categorical_accuracy: 0.8588
11712/60000 [====>.........................] - ETA: 1:13 - loss: 0.4467 - categorical_accuracy: 0.8592
11776/60000 [====>.........................] - ETA: 1:13 - loss: 0.4452 - categorical_accuracy: 0.8596
11840/60000 [====>.........................] - ETA: 1:13 - loss: 0.4434 - categorical_accuracy: 0.8602
11872/60000 [====>.........................] - ETA: 1:13 - loss: 0.4427 - categorical_accuracy: 0.8604
11904/60000 [====>.........................] - ETA: 1:13 - loss: 0.4419 - categorical_accuracy: 0.8607
11968/60000 [====>.........................] - ETA: 1:12 - loss: 0.4399 - categorical_accuracy: 0.8613
12032/60000 [=====>........................] - ETA: 1:12 - loss: 0.4391 - categorical_accuracy: 0.8617
12064/60000 [=====>........................] - ETA: 1:12 - loss: 0.4383 - categorical_accuracy: 0.8619
12096/60000 [=====>........................] - ETA: 1:12 - loss: 0.4374 - categorical_accuracy: 0.8621
12128/60000 [=====>........................] - ETA: 1:12 - loss: 0.4369 - categorical_accuracy: 0.8623
12192/60000 [=====>........................] - ETA: 1:12 - loss: 0.4353 - categorical_accuracy: 0.8629
12256/60000 [=====>........................] - ETA: 1:12 - loss: 0.4341 - categorical_accuracy: 0.8633
12320/60000 [=====>........................] - ETA: 1:12 - loss: 0.4324 - categorical_accuracy: 0.8639
12352/60000 [=====>........................] - ETA: 1:12 - loss: 0.4318 - categorical_accuracy: 0.8642
12416/60000 [=====>........................] - ETA: 1:12 - loss: 0.4304 - categorical_accuracy: 0.8645
12480/60000 [=====>........................] - ETA: 1:12 - loss: 0.4295 - categorical_accuracy: 0.8649
12512/60000 [=====>........................] - ETA: 1:12 - loss: 0.4293 - categorical_accuracy: 0.8649
12544/60000 [=====>........................] - ETA: 1:12 - loss: 0.4287 - categorical_accuracy: 0.8651
12576/60000 [=====>........................] - ETA: 1:12 - loss: 0.4283 - categorical_accuracy: 0.8651
12640/60000 [=====>........................] - ETA: 1:12 - loss: 0.4267 - categorical_accuracy: 0.8657
12704/60000 [=====>........................] - ETA: 1:11 - loss: 0.4264 - categorical_accuracy: 0.8660
12768/60000 [=====>........................] - ETA: 1:11 - loss: 0.4255 - categorical_accuracy: 0.8663
12832/60000 [=====>........................] - ETA: 1:11 - loss: 0.4240 - categorical_accuracy: 0.8667
12864/60000 [=====>........................] - ETA: 1:11 - loss: 0.4231 - categorical_accuracy: 0.8670
12928/60000 [=====>........................] - ETA: 1:11 - loss: 0.4219 - categorical_accuracy: 0.8673
12992/60000 [=====>........................] - ETA: 1:11 - loss: 0.4206 - categorical_accuracy: 0.8677
13056/60000 [=====>........................] - ETA: 1:11 - loss: 0.4191 - categorical_accuracy: 0.8683
13120/60000 [=====>........................] - ETA: 1:11 - loss: 0.4178 - categorical_accuracy: 0.8686
13184/60000 [=====>........................] - ETA: 1:11 - loss: 0.4175 - categorical_accuracy: 0.8689
13248/60000 [=====>........................] - ETA: 1:11 - loss: 0.4164 - categorical_accuracy: 0.8691
13312/60000 [=====>........................] - ETA: 1:10 - loss: 0.4155 - categorical_accuracy: 0.8694
13344/60000 [=====>........................] - ETA: 1:10 - loss: 0.4161 - categorical_accuracy: 0.8694
13408/60000 [=====>........................] - ETA: 1:10 - loss: 0.4151 - categorical_accuracy: 0.8697
13440/60000 [=====>........................] - ETA: 1:10 - loss: 0.4148 - categorical_accuracy: 0.8699
13504/60000 [=====>........................] - ETA: 1:10 - loss: 0.4135 - categorical_accuracy: 0.8703
13568/60000 [=====>........................] - ETA: 1:10 - loss: 0.4121 - categorical_accuracy: 0.8707
13632/60000 [=====>........................] - ETA: 1:10 - loss: 0.4106 - categorical_accuracy: 0.8713
13696/60000 [=====>........................] - ETA: 1:10 - loss: 0.4089 - categorical_accuracy: 0.8718
13760/60000 [=====>........................] - ETA: 1:10 - loss: 0.4076 - categorical_accuracy: 0.8722
13824/60000 [=====>........................] - ETA: 1:10 - loss: 0.4074 - categorical_accuracy: 0.8724
13888/60000 [=====>........................] - ETA: 1:09 - loss: 0.4065 - categorical_accuracy: 0.8728
13952/60000 [=====>........................] - ETA: 1:09 - loss: 0.4051 - categorical_accuracy: 0.8732
14016/60000 [======>.......................] - ETA: 1:09 - loss: 0.4046 - categorical_accuracy: 0.8737
14080/60000 [======>.......................] - ETA: 1:09 - loss: 0.4048 - categorical_accuracy: 0.8739
14144/60000 [======>.......................] - ETA: 1:09 - loss: 0.4036 - categorical_accuracy: 0.8744
14208/60000 [======>.......................] - ETA: 1:09 - loss: 0.4027 - categorical_accuracy: 0.8746
14272/60000 [======>.......................] - ETA: 1:09 - loss: 0.4016 - categorical_accuracy: 0.8749
14336/60000 [======>.......................] - ETA: 1:09 - loss: 0.4013 - categorical_accuracy: 0.8751
14368/60000 [======>.......................] - ETA: 1:09 - loss: 0.4007 - categorical_accuracy: 0.8753
14432/60000 [======>.......................] - ETA: 1:09 - loss: 0.3995 - categorical_accuracy: 0.8758
14464/60000 [======>.......................] - ETA: 1:09 - loss: 0.3992 - categorical_accuracy: 0.8758
14528/60000 [======>.......................] - ETA: 1:08 - loss: 0.3982 - categorical_accuracy: 0.8761
14592/60000 [======>.......................] - ETA: 1:08 - loss: 0.3971 - categorical_accuracy: 0.8765
14656/60000 [======>.......................] - ETA: 1:08 - loss: 0.3956 - categorical_accuracy: 0.8770
14720/60000 [======>.......................] - ETA: 1:08 - loss: 0.3945 - categorical_accuracy: 0.8773
14784/60000 [======>.......................] - ETA: 1:08 - loss: 0.3937 - categorical_accuracy: 0.8776
14848/60000 [======>.......................] - ETA: 1:08 - loss: 0.3928 - categorical_accuracy: 0.8780
14912/60000 [======>.......................] - ETA: 1:08 - loss: 0.3915 - categorical_accuracy: 0.8785
14976/60000 [======>.......................] - ETA: 1:08 - loss: 0.3907 - categorical_accuracy: 0.8787
15040/60000 [======>.......................] - ETA: 1:08 - loss: 0.3896 - categorical_accuracy: 0.8791
15104/60000 [======>.......................] - ETA: 1:08 - loss: 0.3890 - categorical_accuracy: 0.8792
15136/60000 [======>.......................] - ETA: 1:07 - loss: 0.3884 - categorical_accuracy: 0.8794
15200/60000 [======>.......................] - ETA: 1:07 - loss: 0.3875 - categorical_accuracy: 0.8797
15264/60000 [======>.......................] - ETA: 1:07 - loss: 0.3865 - categorical_accuracy: 0.8800
15328/60000 [======>.......................] - ETA: 1:07 - loss: 0.3865 - categorical_accuracy: 0.8802
15392/60000 [======>.......................] - ETA: 1:07 - loss: 0.3856 - categorical_accuracy: 0.8803
15424/60000 [======>.......................] - ETA: 1:07 - loss: 0.3852 - categorical_accuracy: 0.8804
15456/60000 [======>.......................] - ETA: 1:07 - loss: 0.3846 - categorical_accuracy: 0.8806
15520/60000 [======>.......................] - ETA: 1:07 - loss: 0.3837 - categorical_accuracy: 0.8808
15584/60000 [======>.......................] - ETA: 1:07 - loss: 0.3833 - categorical_accuracy: 0.8810
15648/60000 [======>.......................] - ETA: 1:07 - loss: 0.3821 - categorical_accuracy: 0.8814
15712/60000 [======>.......................] - ETA: 1:07 - loss: 0.3810 - categorical_accuracy: 0.8817
15776/60000 [======>.......................] - ETA: 1:06 - loss: 0.3804 - categorical_accuracy: 0.8820
15840/60000 [======>.......................] - ETA: 1:06 - loss: 0.3789 - categorical_accuracy: 0.8824
15904/60000 [======>.......................] - ETA: 1:06 - loss: 0.3784 - categorical_accuracy: 0.8826
15936/60000 [======>.......................] - ETA: 1:06 - loss: 0.3782 - categorical_accuracy: 0.8827
16000/60000 [=======>......................] - ETA: 1:06 - loss: 0.3776 - categorical_accuracy: 0.8829
16064/60000 [=======>......................] - ETA: 1:06 - loss: 0.3768 - categorical_accuracy: 0.8832
16128/60000 [=======>......................] - ETA: 1:06 - loss: 0.3758 - categorical_accuracy: 0.8835
16192/60000 [=======>......................] - ETA: 1:06 - loss: 0.3751 - categorical_accuracy: 0.8838
16256/60000 [=======>......................] - ETA: 1:06 - loss: 0.3744 - categorical_accuracy: 0.8840
16320/60000 [=======>......................] - ETA: 1:06 - loss: 0.3736 - categorical_accuracy: 0.8843
16384/60000 [=======>......................] - ETA: 1:06 - loss: 0.3734 - categorical_accuracy: 0.8845
16448/60000 [=======>......................] - ETA: 1:05 - loss: 0.3723 - categorical_accuracy: 0.8848
16512/60000 [=======>......................] - ETA: 1:05 - loss: 0.3723 - categorical_accuracy: 0.8848
16576/60000 [=======>......................] - ETA: 1:05 - loss: 0.3723 - categorical_accuracy: 0.8850
16640/60000 [=======>......................] - ETA: 1:05 - loss: 0.3714 - categorical_accuracy: 0.8852
16704/60000 [=======>......................] - ETA: 1:05 - loss: 0.3706 - categorical_accuracy: 0.8854
16768/60000 [=======>......................] - ETA: 1:05 - loss: 0.3695 - categorical_accuracy: 0.8858
16832/60000 [=======>......................] - ETA: 1:05 - loss: 0.3690 - categorical_accuracy: 0.8860
16896/60000 [=======>......................] - ETA: 1:05 - loss: 0.3684 - categorical_accuracy: 0.8861
16960/60000 [=======>......................] - ETA: 1:05 - loss: 0.3675 - categorical_accuracy: 0.8863
17024/60000 [=======>......................] - ETA: 1:04 - loss: 0.3663 - categorical_accuracy: 0.8867
17088/60000 [=======>......................] - ETA: 1:04 - loss: 0.3662 - categorical_accuracy: 0.8868
17152/60000 [=======>......................] - ETA: 1:04 - loss: 0.3661 - categorical_accuracy: 0.8869
17216/60000 [=======>......................] - ETA: 1:04 - loss: 0.3654 - categorical_accuracy: 0.8871
17280/60000 [=======>......................] - ETA: 1:04 - loss: 0.3646 - categorical_accuracy: 0.8873
17344/60000 [=======>......................] - ETA: 1:04 - loss: 0.3640 - categorical_accuracy: 0.8875
17408/60000 [=======>......................] - ETA: 1:04 - loss: 0.3635 - categorical_accuracy: 0.8876
17472/60000 [=======>......................] - ETA: 1:04 - loss: 0.3630 - categorical_accuracy: 0.8878
17536/60000 [=======>......................] - ETA: 1:04 - loss: 0.3623 - categorical_accuracy: 0.8879
17600/60000 [=======>......................] - ETA: 1:04 - loss: 0.3617 - categorical_accuracy: 0.8880
17632/60000 [=======>......................] - ETA: 1:03 - loss: 0.3615 - categorical_accuracy: 0.8880
17696/60000 [=======>......................] - ETA: 1:03 - loss: 0.3604 - categorical_accuracy: 0.8884
17760/60000 [=======>......................] - ETA: 1:03 - loss: 0.3599 - categorical_accuracy: 0.8885
17824/60000 [=======>......................] - ETA: 1:03 - loss: 0.3588 - categorical_accuracy: 0.8887
17888/60000 [=======>......................] - ETA: 1:03 - loss: 0.3580 - categorical_accuracy: 0.8890
17952/60000 [=======>......................] - ETA: 1:03 - loss: 0.3572 - categorical_accuracy: 0.8893
18016/60000 [========>.....................] - ETA: 1:03 - loss: 0.3565 - categorical_accuracy: 0.8894
18048/60000 [========>.....................] - ETA: 1:03 - loss: 0.3564 - categorical_accuracy: 0.8895
18112/60000 [========>.....................] - ETA: 1:03 - loss: 0.3562 - categorical_accuracy: 0.8895
18144/60000 [========>.....................] - ETA: 1:03 - loss: 0.3558 - categorical_accuracy: 0.8896
18176/60000 [========>.....................] - ETA: 1:03 - loss: 0.3554 - categorical_accuracy: 0.8897
18240/60000 [========>.....................] - ETA: 1:03 - loss: 0.3544 - categorical_accuracy: 0.8900
18304/60000 [========>.....................] - ETA: 1:02 - loss: 0.3546 - categorical_accuracy: 0.8900
18368/60000 [========>.....................] - ETA: 1:02 - loss: 0.3536 - categorical_accuracy: 0.8904
18432/60000 [========>.....................] - ETA: 1:02 - loss: 0.3527 - categorical_accuracy: 0.8907
18496/60000 [========>.....................] - ETA: 1:02 - loss: 0.3519 - categorical_accuracy: 0.8908
18528/60000 [========>.....................] - ETA: 1:02 - loss: 0.3515 - categorical_accuracy: 0.8910
18592/60000 [========>.....................] - ETA: 1:02 - loss: 0.3512 - categorical_accuracy: 0.8912
18656/60000 [========>.....................] - ETA: 1:02 - loss: 0.3509 - categorical_accuracy: 0.8913
18720/60000 [========>.....................] - ETA: 1:02 - loss: 0.3502 - categorical_accuracy: 0.8916
18784/60000 [========>.....................] - ETA: 1:02 - loss: 0.3499 - categorical_accuracy: 0.8917
18848/60000 [========>.....................] - ETA: 1:01 - loss: 0.3495 - categorical_accuracy: 0.8919
18880/60000 [========>.....................] - ETA: 1:01 - loss: 0.3490 - categorical_accuracy: 0.8921
18944/60000 [========>.....................] - ETA: 1:01 - loss: 0.3483 - categorical_accuracy: 0.8922
19008/60000 [========>.....................] - ETA: 1:01 - loss: 0.3476 - categorical_accuracy: 0.8924
19072/60000 [========>.....................] - ETA: 1:01 - loss: 0.3473 - categorical_accuracy: 0.8924
19136/60000 [========>.....................] - ETA: 1:01 - loss: 0.3464 - categorical_accuracy: 0.8927
19200/60000 [========>.....................] - ETA: 1:01 - loss: 0.3458 - categorical_accuracy: 0.8929
19232/60000 [========>.....................] - ETA: 1:01 - loss: 0.3453 - categorical_accuracy: 0.8930
19296/60000 [========>.....................] - ETA: 1:01 - loss: 0.3445 - categorical_accuracy: 0.8933
19360/60000 [========>.....................] - ETA: 1:01 - loss: 0.3437 - categorical_accuracy: 0.8936
19424/60000 [========>.....................] - ETA: 1:01 - loss: 0.3428 - categorical_accuracy: 0.8939
19488/60000 [========>.....................] - ETA: 1:00 - loss: 0.3423 - categorical_accuracy: 0.8940
19552/60000 [========>.....................] - ETA: 1:00 - loss: 0.3414 - categorical_accuracy: 0.8943
19584/60000 [========>.....................] - ETA: 1:00 - loss: 0.3413 - categorical_accuracy: 0.8943
19648/60000 [========>.....................] - ETA: 1:00 - loss: 0.3405 - categorical_accuracy: 0.8945
19712/60000 [========>.....................] - ETA: 1:00 - loss: 0.3399 - categorical_accuracy: 0.8947
19776/60000 [========>.....................] - ETA: 1:00 - loss: 0.3391 - categorical_accuracy: 0.8950
19840/60000 [========>.....................] - ETA: 1:00 - loss: 0.3389 - categorical_accuracy: 0.8952
19904/60000 [========>.....................] - ETA: 1:00 - loss: 0.3386 - categorical_accuracy: 0.8952
19968/60000 [========>.....................] - ETA: 1:00 - loss: 0.3377 - categorical_accuracy: 0.8955
20032/60000 [=========>....................] - ETA: 1:00 - loss: 0.3370 - categorical_accuracy: 0.8957
20096/60000 [=========>....................] - ETA: 1:00 - loss: 0.3363 - categorical_accuracy: 0.8958
20128/60000 [=========>....................] - ETA: 59s - loss: 0.3366 - categorical_accuracy: 0.8958 
20160/60000 [=========>....................] - ETA: 59s - loss: 0.3362 - categorical_accuracy: 0.8959
20224/60000 [=========>....................] - ETA: 59s - loss: 0.3355 - categorical_accuracy: 0.8962
20256/60000 [=========>....................] - ETA: 59s - loss: 0.3350 - categorical_accuracy: 0.8963
20320/60000 [=========>....................] - ETA: 59s - loss: 0.3349 - categorical_accuracy: 0.8964
20384/60000 [=========>....................] - ETA: 59s - loss: 0.3344 - categorical_accuracy: 0.8965
20448/60000 [=========>....................] - ETA: 59s - loss: 0.3338 - categorical_accuracy: 0.8967
20512/60000 [=========>....................] - ETA: 59s - loss: 0.3336 - categorical_accuracy: 0.8968
20576/60000 [=========>....................] - ETA: 59s - loss: 0.3330 - categorical_accuracy: 0.8969
20608/60000 [=========>....................] - ETA: 59s - loss: 0.3325 - categorical_accuracy: 0.8970
20672/60000 [=========>....................] - ETA: 59s - loss: 0.3319 - categorical_accuracy: 0.8972
20736/60000 [=========>....................] - ETA: 59s - loss: 0.3322 - categorical_accuracy: 0.8972
20800/60000 [=========>....................] - ETA: 58s - loss: 0.3313 - categorical_accuracy: 0.8975
20864/60000 [=========>....................] - ETA: 58s - loss: 0.3307 - categorical_accuracy: 0.8977
20928/60000 [=========>....................] - ETA: 58s - loss: 0.3300 - categorical_accuracy: 0.8979
20992/60000 [=========>....................] - ETA: 58s - loss: 0.3295 - categorical_accuracy: 0.8981
21056/60000 [=========>....................] - ETA: 58s - loss: 0.3293 - categorical_accuracy: 0.8982
21088/60000 [=========>....................] - ETA: 58s - loss: 0.3290 - categorical_accuracy: 0.8983
21152/60000 [=========>....................] - ETA: 58s - loss: 0.3288 - categorical_accuracy: 0.8985
21216/60000 [=========>....................] - ETA: 58s - loss: 0.3282 - categorical_accuracy: 0.8987
21280/60000 [=========>....................] - ETA: 58s - loss: 0.3274 - categorical_accuracy: 0.8990
21344/60000 [=========>....................] - ETA: 58s - loss: 0.3269 - categorical_accuracy: 0.8991
21408/60000 [=========>....................] - ETA: 57s - loss: 0.3264 - categorical_accuracy: 0.8992
21472/60000 [=========>....................] - ETA: 57s - loss: 0.3257 - categorical_accuracy: 0.8994
21536/60000 [=========>....................] - ETA: 57s - loss: 0.3250 - categorical_accuracy: 0.8996
21600/60000 [=========>....................] - ETA: 57s - loss: 0.3243 - categorical_accuracy: 0.8998
21664/60000 [=========>....................] - ETA: 57s - loss: 0.3241 - categorical_accuracy: 0.8999
21728/60000 [=========>....................] - ETA: 57s - loss: 0.3240 - categorical_accuracy: 0.8999
21792/60000 [=========>....................] - ETA: 57s - loss: 0.3232 - categorical_accuracy: 0.9002
21856/60000 [=========>....................] - ETA: 57s - loss: 0.3229 - categorical_accuracy: 0.9004
21920/60000 [=========>....................] - ETA: 57s - loss: 0.3224 - categorical_accuracy: 0.9005
21984/60000 [=========>....................] - ETA: 57s - loss: 0.3219 - categorical_accuracy: 0.9006
22048/60000 [==========>...................] - ETA: 56s - loss: 0.3211 - categorical_accuracy: 0.9009
22112/60000 [==========>...................] - ETA: 56s - loss: 0.3207 - categorical_accuracy: 0.9011
22176/60000 [==========>...................] - ETA: 56s - loss: 0.3200 - categorical_accuracy: 0.9013
22240/60000 [==========>...................] - ETA: 56s - loss: 0.3196 - categorical_accuracy: 0.9015
22272/60000 [==========>...................] - ETA: 56s - loss: 0.3193 - categorical_accuracy: 0.9016
22336/60000 [==========>...................] - ETA: 56s - loss: 0.3186 - categorical_accuracy: 0.9018
22400/60000 [==========>...................] - ETA: 56s - loss: 0.3183 - categorical_accuracy: 0.9019
22464/60000 [==========>...................] - ETA: 56s - loss: 0.3179 - categorical_accuracy: 0.9020
22528/60000 [==========>...................] - ETA: 56s - loss: 0.3172 - categorical_accuracy: 0.9023
22592/60000 [==========>...................] - ETA: 56s - loss: 0.3166 - categorical_accuracy: 0.9024
22656/60000 [==========>...................] - ETA: 55s - loss: 0.3161 - categorical_accuracy: 0.9025
22720/60000 [==========>...................] - ETA: 55s - loss: 0.3154 - categorical_accuracy: 0.9026
22784/60000 [==========>...................] - ETA: 55s - loss: 0.3150 - categorical_accuracy: 0.9029
22848/60000 [==========>...................] - ETA: 55s - loss: 0.3144 - categorical_accuracy: 0.9031
22912/60000 [==========>...................] - ETA: 55s - loss: 0.3137 - categorical_accuracy: 0.9033
22976/60000 [==========>...................] - ETA: 55s - loss: 0.3135 - categorical_accuracy: 0.9034
23040/60000 [==========>...................] - ETA: 55s - loss: 0.3130 - categorical_accuracy: 0.9036
23072/60000 [==========>...................] - ETA: 55s - loss: 0.3125 - categorical_accuracy: 0.9037
23136/60000 [==========>...................] - ETA: 55s - loss: 0.3119 - categorical_accuracy: 0.9039
23200/60000 [==========>...................] - ETA: 55s - loss: 0.3112 - categorical_accuracy: 0.9041
23264/60000 [==========>...................] - ETA: 55s - loss: 0.3106 - categorical_accuracy: 0.9043
23328/60000 [==========>...................] - ETA: 54s - loss: 0.3102 - categorical_accuracy: 0.9043
23360/60000 [==========>...................] - ETA: 54s - loss: 0.3098 - categorical_accuracy: 0.9045
23424/60000 [==========>...................] - ETA: 54s - loss: 0.3093 - categorical_accuracy: 0.9046
23456/60000 [==========>...................] - ETA: 54s - loss: 0.3094 - categorical_accuracy: 0.9046
23520/60000 [==========>...................] - ETA: 54s - loss: 0.3088 - categorical_accuracy: 0.9048
23552/60000 [==========>...................] - ETA: 54s - loss: 0.3085 - categorical_accuracy: 0.9049
23584/60000 [==========>...................] - ETA: 54s - loss: 0.3084 - categorical_accuracy: 0.9049
23648/60000 [==========>...................] - ETA: 54s - loss: 0.3080 - categorical_accuracy: 0.9051
23712/60000 [==========>...................] - ETA: 54s - loss: 0.3074 - categorical_accuracy: 0.9053
23776/60000 [==========>...................] - ETA: 54s - loss: 0.3073 - categorical_accuracy: 0.9054
23840/60000 [==========>...................] - ETA: 54s - loss: 0.3065 - categorical_accuracy: 0.9056
23872/60000 [==========>...................] - ETA: 54s - loss: 0.3064 - categorical_accuracy: 0.9056
23936/60000 [==========>...................] - ETA: 54s - loss: 0.3058 - categorical_accuracy: 0.9059
24000/60000 [===========>..................] - ETA: 53s - loss: 0.3053 - categorical_accuracy: 0.9060
24064/60000 [===========>..................] - ETA: 53s - loss: 0.3056 - categorical_accuracy: 0.9060
24128/60000 [===========>..................] - ETA: 53s - loss: 0.3051 - categorical_accuracy: 0.9062
24192/60000 [===========>..................] - ETA: 53s - loss: 0.3044 - categorical_accuracy: 0.9064
24256/60000 [===========>..................] - ETA: 53s - loss: 0.3038 - categorical_accuracy: 0.9066
24288/60000 [===========>..................] - ETA: 53s - loss: 0.3035 - categorical_accuracy: 0.9067
24320/60000 [===========>..................] - ETA: 53s - loss: 0.3032 - categorical_accuracy: 0.9068
24384/60000 [===========>..................] - ETA: 53s - loss: 0.3025 - categorical_accuracy: 0.9070
24448/60000 [===========>..................] - ETA: 53s - loss: 0.3018 - categorical_accuracy: 0.9072
24512/60000 [===========>..................] - ETA: 53s - loss: 0.3018 - categorical_accuracy: 0.9072
24576/60000 [===========>..................] - ETA: 53s - loss: 0.3012 - categorical_accuracy: 0.9075
24640/60000 [===========>..................] - ETA: 53s - loss: 0.3005 - categorical_accuracy: 0.9077
24704/60000 [===========>..................] - ETA: 52s - loss: 0.3003 - categorical_accuracy: 0.9077
24736/60000 [===========>..................] - ETA: 52s - loss: 0.3000 - categorical_accuracy: 0.9077
24800/60000 [===========>..................] - ETA: 52s - loss: 0.3002 - categorical_accuracy: 0.9078
24832/60000 [===========>..................] - ETA: 52s - loss: 0.3002 - categorical_accuracy: 0.9078
24896/60000 [===========>..................] - ETA: 52s - loss: 0.2999 - categorical_accuracy: 0.9079
24960/60000 [===========>..................] - ETA: 52s - loss: 0.2993 - categorical_accuracy: 0.9080
25024/60000 [===========>..................] - ETA: 52s - loss: 0.2989 - categorical_accuracy: 0.9081
25088/60000 [===========>..................] - ETA: 52s - loss: 0.2982 - categorical_accuracy: 0.9083
25152/60000 [===========>..................] - ETA: 52s - loss: 0.2978 - categorical_accuracy: 0.9084
25216/60000 [===========>..................] - ETA: 52s - loss: 0.2973 - categorical_accuracy: 0.9086
25280/60000 [===========>..................] - ETA: 52s - loss: 0.2967 - categorical_accuracy: 0.9087
25344/60000 [===========>..................] - ETA: 51s - loss: 0.2965 - categorical_accuracy: 0.9089
25408/60000 [===========>..................] - ETA: 51s - loss: 0.2959 - categorical_accuracy: 0.9091
25472/60000 [===========>..................] - ETA: 51s - loss: 0.2953 - categorical_accuracy: 0.9092
25536/60000 [===========>..................] - ETA: 51s - loss: 0.2948 - categorical_accuracy: 0.9094
25600/60000 [===========>..................] - ETA: 51s - loss: 0.2946 - categorical_accuracy: 0.9095
25664/60000 [===========>..................] - ETA: 51s - loss: 0.2941 - categorical_accuracy: 0.9096
25728/60000 [===========>..................] - ETA: 51s - loss: 0.2936 - categorical_accuracy: 0.9097
25792/60000 [===========>..................] - ETA: 51s - loss: 0.2932 - categorical_accuracy: 0.9098
25856/60000 [===========>..................] - ETA: 51s - loss: 0.2926 - categorical_accuracy: 0.9100
25920/60000 [===========>..................] - ETA: 51s - loss: 0.2921 - categorical_accuracy: 0.9101
25984/60000 [===========>..................] - ETA: 50s - loss: 0.2919 - categorical_accuracy: 0.9102
26048/60000 [============>.................] - ETA: 50s - loss: 0.2916 - categorical_accuracy: 0.9102
26112/60000 [============>.................] - ETA: 50s - loss: 0.2911 - categorical_accuracy: 0.9104
26144/60000 [============>.................] - ETA: 50s - loss: 0.2909 - categorical_accuracy: 0.9105
26208/60000 [============>.................] - ETA: 50s - loss: 0.2906 - categorical_accuracy: 0.9106
26272/60000 [============>.................] - ETA: 50s - loss: 0.2900 - categorical_accuracy: 0.9108
26336/60000 [============>.................] - ETA: 50s - loss: 0.2895 - categorical_accuracy: 0.9109
26368/60000 [============>.................] - ETA: 50s - loss: 0.2892 - categorical_accuracy: 0.9110
26432/60000 [============>.................] - ETA: 50s - loss: 0.2887 - categorical_accuracy: 0.9111
26496/60000 [============>.................] - ETA: 50s - loss: 0.2885 - categorical_accuracy: 0.9111
26560/60000 [============>.................] - ETA: 50s - loss: 0.2880 - categorical_accuracy: 0.9112
26624/60000 [============>.................] - ETA: 49s - loss: 0.2873 - categorical_accuracy: 0.9114
26688/60000 [============>.................] - ETA: 49s - loss: 0.2871 - categorical_accuracy: 0.9114
26752/60000 [============>.................] - ETA: 49s - loss: 0.2869 - categorical_accuracy: 0.9115
26816/60000 [============>.................] - ETA: 49s - loss: 0.2871 - categorical_accuracy: 0.9115
26880/60000 [============>.................] - ETA: 49s - loss: 0.2869 - categorical_accuracy: 0.9116
26944/60000 [============>.................] - ETA: 49s - loss: 0.2866 - categorical_accuracy: 0.9116
27008/60000 [============>.................] - ETA: 49s - loss: 0.2864 - categorical_accuracy: 0.9117
27072/60000 [============>.................] - ETA: 49s - loss: 0.2859 - categorical_accuracy: 0.9118
27136/60000 [============>.................] - ETA: 49s - loss: 0.2854 - categorical_accuracy: 0.9119
27168/60000 [============>.................] - ETA: 49s - loss: 0.2851 - categorical_accuracy: 0.9120
27200/60000 [============>.................] - ETA: 49s - loss: 0.2851 - categorical_accuracy: 0.9121
27264/60000 [============>.................] - ETA: 49s - loss: 0.2846 - categorical_accuracy: 0.9122
27328/60000 [============>.................] - ETA: 48s - loss: 0.2843 - categorical_accuracy: 0.9123
27392/60000 [============>.................] - ETA: 48s - loss: 0.2840 - categorical_accuracy: 0.9124
27456/60000 [============>.................] - ETA: 48s - loss: 0.2836 - categorical_accuracy: 0.9125
27520/60000 [============>.................] - ETA: 48s - loss: 0.2832 - categorical_accuracy: 0.9126
27584/60000 [============>.................] - ETA: 48s - loss: 0.2828 - categorical_accuracy: 0.9127
27648/60000 [============>.................] - ETA: 48s - loss: 0.2823 - categorical_accuracy: 0.9128
27712/60000 [============>.................] - ETA: 48s - loss: 0.2817 - categorical_accuracy: 0.9130
27776/60000 [============>.................] - ETA: 48s - loss: 0.2813 - categorical_accuracy: 0.9131
27840/60000 [============>.................] - ETA: 48s - loss: 0.2811 - categorical_accuracy: 0.9132
27904/60000 [============>.................] - ETA: 48s - loss: 0.2806 - categorical_accuracy: 0.9134
27968/60000 [============>.................] - ETA: 47s - loss: 0.2803 - categorical_accuracy: 0.9135
28000/60000 [=============>................] - ETA: 47s - loss: 0.2800 - categorical_accuracy: 0.9136
28064/60000 [=============>................] - ETA: 47s - loss: 0.2800 - categorical_accuracy: 0.9136
28128/60000 [=============>................] - ETA: 47s - loss: 0.2796 - categorical_accuracy: 0.9136
28192/60000 [=============>................] - ETA: 47s - loss: 0.2793 - categorical_accuracy: 0.9136
28256/60000 [=============>................] - ETA: 47s - loss: 0.2788 - categorical_accuracy: 0.9138
28320/60000 [=============>................] - ETA: 47s - loss: 0.2783 - categorical_accuracy: 0.9139
28384/60000 [=============>................] - ETA: 47s - loss: 0.2779 - categorical_accuracy: 0.9139
28448/60000 [=============>................] - ETA: 47s - loss: 0.2774 - categorical_accuracy: 0.9141
28480/60000 [=============>................] - ETA: 47s - loss: 0.2771 - categorical_accuracy: 0.9142
28512/60000 [=============>................] - ETA: 47s - loss: 0.2770 - categorical_accuracy: 0.9142
28576/60000 [=============>................] - ETA: 47s - loss: 0.2767 - categorical_accuracy: 0.9143
28640/60000 [=============>................] - ETA: 46s - loss: 0.2762 - categorical_accuracy: 0.9145
28704/60000 [=============>................] - ETA: 46s - loss: 0.2759 - categorical_accuracy: 0.9146
28768/60000 [=============>................] - ETA: 46s - loss: 0.2753 - categorical_accuracy: 0.9148
28832/60000 [=============>................] - ETA: 46s - loss: 0.2749 - categorical_accuracy: 0.9150
28896/60000 [=============>................] - ETA: 46s - loss: 0.2744 - categorical_accuracy: 0.9150
28960/60000 [=============>................] - ETA: 46s - loss: 0.2741 - categorical_accuracy: 0.9151
29024/60000 [=============>................] - ETA: 46s - loss: 0.2737 - categorical_accuracy: 0.9152
29088/60000 [=============>................] - ETA: 46s - loss: 0.2732 - categorical_accuracy: 0.9154
29152/60000 [=============>................] - ETA: 46s - loss: 0.2729 - categorical_accuracy: 0.9154
29216/60000 [=============>................] - ETA: 46s - loss: 0.2726 - categorical_accuracy: 0.9155
29280/60000 [=============>................] - ETA: 45s - loss: 0.2729 - categorical_accuracy: 0.9154
29344/60000 [=============>................] - ETA: 45s - loss: 0.2726 - categorical_accuracy: 0.9155
29408/60000 [=============>................] - ETA: 45s - loss: 0.2721 - categorical_accuracy: 0.9156
29472/60000 [=============>................] - ETA: 45s - loss: 0.2719 - categorical_accuracy: 0.9157
29536/60000 [=============>................] - ETA: 45s - loss: 0.2716 - categorical_accuracy: 0.9158
29568/60000 [=============>................] - ETA: 45s - loss: 0.2714 - categorical_accuracy: 0.9158
29632/60000 [=============>................] - ETA: 45s - loss: 0.2710 - categorical_accuracy: 0.9159
29664/60000 [=============>................] - ETA: 45s - loss: 0.2707 - categorical_accuracy: 0.9160
29728/60000 [=============>................] - ETA: 45s - loss: 0.2705 - categorical_accuracy: 0.9161
29792/60000 [=============>................] - ETA: 45s - loss: 0.2699 - categorical_accuracy: 0.9163
29824/60000 [=============>................] - ETA: 45s - loss: 0.2697 - categorical_accuracy: 0.9164
29888/60000 [=============>................] - ETA: 45s - loss: 0.2693 - categorical_accuracy: 0.9165
29952/60000 [=============>................] - ETA: 44s - loss: 0.2690 - categorical_accuracy: 0.9166
30016/60000 [==============>...............] - ETA: 44s - loss: 0.2691 - categorical_accuracy: 0.9166
30080/60000 [==============>...............] - ETA: 44s - loss: 0.2687 - categorical_accuracy: 0.9168
30144/60000 [==============>...............] - ETA: 44s - loss: 0.2682 - categorical_accuracy: 0.9169
30208/60000 [==============>...............] - ETA: 44s - loss: 0.2680 - categorical_accuracy: 0.9170
30272/60000 [==============>...............] - ETA: 44s - loss: 0.2678 - categorical_accuracy: 0.9171
30336/60000 [==============>...............] - ETA: 44s - loss: 0.2676 - categorical_accuracy: 0.9171
30400/60000 [==============>...............] - ETA: 44s - loss: 0.2672 - categorical_accuracy: 0.9172
30464/60000 [==============>...............] - ETA: 44s - loss: 0.2669 - categorical_accuracy: 0.9173
30528/60000 [==============>...............] - ETA: 44s - loss: 0.2669 - categorical_accuracy: 0.9174
30560/60000 [==============>...............] - ETA: 44s - loss: 0.2667 - categorical_accuracy: 0.9174
30624/60000 [==============>...............] - ETA: 43s - loss: 0.2665 - categorical_accuracy: 0.9175
30688/60000 [==============>...............] - ETA: 43s - loss: 0.2662 - categorical_accuracy: 0.9177
30752/60000 [==============>...............] - ETA: 43s - loss: 0.2660 - categorical_accuracy: 0.9178
30784/60000 [==============>...............] - ETA: 43s - loss: 0.2658 - categorical_accuracy: 0.9178
30816/60000 [==============>...............] - ETA: 43s - loss: 0.2658 - categorical_accuracy: 0.9179
30848/60000 [==============>...............] - ETA: 43s - loss: 0.2655 - categorical_accuracy: 0.9180
30880/60000 [==============>...............] - ETA: 43s - loss: 0.2654 - categorical_accuracy: 0.9180
30912/60000 [==============>...............] - ETA: 43s - loss: 0.2652 - categorical_accuracy: 0.9180
30976/60000 [==============>...............] - ETA: 43s - loss: 0.2648 - categorical_accuracy: 0.9181
31040/60000 [==============>...............] - ETA: 43s - loss: 0.2643 - categorical_accuracy: 0.9183
31104/60000 [==============>...............] - ETA: 43s - loss: 0.2643 - categorical_accuracy: 0.9183
31168/60000 [==============>...............] - ETA: 43s - loss: 0.2639 - categorical_accuracy: 0.9184
31232/60000 [==============>...............] - ETA: 43s - loss: 0.2638 - categorical_accuracy: 0.9184
31296/60000 [==============>...............] - ETA: 42s - loss: 0.2635 - categorical_accuracy: 0.9185
31360/60000 [==============>...............] - ETA: 42s - loss: 0.2636 - categorical_accuracy: 0.9186
31424/60000 [==============>...............] - ETA: 42s - loss: 0.2634 - categorical_accuracy: 0.9186
31456/60000 [==============>...............] - ETA: 42s - loss: 0.2634 - categorical_accuracy: 0.9186
31520/60000 [==============>...............] - ETA: 42s - loss: 0.2630 - categorical_accuracy: 0.9188
31584/60000 [==============>...............] - ETA: 42s - loss: 0.2627 - categorical_accuracy: 0.9189
31648/60000 [==============>...............] - ETA: 42s - loss: 0.2623 - categorical_accuracy: 0.9190
31712/60000 [==============>...............] - ETA: 42s - loss: 0.2619 - categorical_accuracy: 0.9191
31776/60000 [==============>...............] - ETA: 42s - loss: 0.2614 - categorical_accuracy: 0.9193
31840/60000 [==============>...............] - ETA: 42s - loss: 0.2611 - categorical_accuracy: 0.9193
31904/60000 [==============>...............] - ETA: 42s - loss: 0.2607 - categorical_accuracy: 0.9195
31968/60000 [==============>...............] - ETA: 41s - loss: 0.2603 - categorical_accuracy: 0.9196
32000/60000 [===============>..............] - ETA: 41s - loss: 0.2601 - categorical_accuracy: 0.9197
32064/60000 [===============>..............] - ETA: 41s - loss: 0.2598 - categorical_accuracy: 0.9198
32128/60000 [===============>..............] - ETA: 41s - loss: 0.2594 - categorical_accuracy: 0.9199
32192/60000 [===============>..............] - ETA: 41s - loss: 0.2589 - categorical_accuracy: 0.9200
32256/60000 [===============>..............] - ETA: 41s - loss: 0.2588 - categorical_accuracy: 0.9201
32320/60000 [===============>..............] - ETA: 41s - loss: 0.2586 - categorical_accuracy: 0.9201
32384/60000 [===============>..............] - ETA: 41s - loss: 0.2583 - categorical_accuracy: 0.9202
32448/60000 [===============>..............] - ETA: 41s - loss: 0.2579 - categorical_accuracy: 0.9203
32512/60000 [===============>..............] - ETA: 41s - loss: 0.2575 - categorical_accuracy: 0.9205
32576/60000 [===============>..............] - ETA: 40s - loss: 0.2572 - categorical_accuracy: 0.9206
32608/60000 [===============>..............] - ETA: 40s - loss: 0.2570 - categorical_accuracy: 0.9206
32672/60000 [===============>..............] - ETA: 40s - loss: 0.2567 - categorical_accuracy: 0.9207
32736/60000 [===============>..............] - ETA: 40s - loss: 0.2564 - categorical_accuracy: 0.9208
32800/60000 [===============>..............] - ETA: 40s - loss: 0.2562 - categorical_accuracy: 0.9209
32864/60000 [===============>..............] - ETA: 40s - loss: 0.2560 - categorical_accuracy: 0.9210
32928/60000 [===============>..............] - ETA: 40s - loss: 0.2555 - categorical_accuracy: 0.9211
32960/60000 [===============>..............] - ETA: 40s - loss: 0.2557 - categorical_accuracy: 0.9211
33024/60000 [===============>..............] - ETA: 40s - loss: 0.2554 - categorical_accuracy: 0.9212
33088/60000 [===============>..............] - ETA: 40s - loss: 0.2551 - categorical_accuracy: 0.9214
33120/60000 [===============>..............] - ETA: 40s - loss: 0.2549 - categorical_accuracy: 0.9214
33184/60000 [===============>..............] - ETA: 40s - loss: 0.2546 - categorical_accuracy: 0.9216
33216/60000 [===============>..............] - ETA: 40s - loss: 0.2546 - categorical_accuracy: 0.9216
33280/60000 [===============>..............] - ETA: 39s - loss: 0.2544 - categorical_accuracy: 0.9216
33344/60000 [===============>..............] - ETA: 39s - loss: 0.2542 - categorical_accuracy: 0.9217
33408/60000 [===============>..............] - ETA: 39s - loss: 0.2543 - categorical_accuracy: 0.9217
33472/60000 [===============>..............] - ETA: 39s - loss: 0.2540 - categorical_accuracy: 0.9218
33536/60000 [===============>..............] - ETA: 39s - loss: 0.2536 - categorical_accuracy: 0.9220
33600/60000 [===============>..............] - ETA: 39s - loss: 0.2533 - categorical_accuracy: 0.9221
33664/60000 [===============>..............] - ETA: 39s - loss: 0.2528 - categorical_accuracy: 0.9223
33728/60000 [===============>..............] - ETA: 39s - loss: 0.2525 - categorical_accuracy: 0.9223
33792/60000 [===============>..............] - ETA: 39s - loss: 0.2523 - categorical_accuracy: 0.9224
33856/60000 [===============>..............] - ETA: 39s - loss: 0.2519 - categorical_accuracy: 0.9225
33920/60000 [===============>..............] - ETA: 38s - loss: 0.2515 - categorical_accuracy: 0.9227
33984/60000 [===============>..............] - ETA: 38s - loss: 0.2510 - categorical_accuracy: 0.9228
34048/60000 [================>.............] - ETA: 38s - loss: 0.2511 - categorical_accuracy: 0.9228
34112/60000 [================>.............] - ETA: 38s - loss: 0.2510 - categorical_accuracy: 0.9229
34176/60000 [================>.............] - ETA: 38s - loss: 0.2508 - categorical_accuracy: 0.9230
34240/60000 [================>.............] - ETA: 38s - loss: 0.2509 - categorical_accuracy: 0.9230
34304/60000 [================>.............] - ETA: 38s - loss: 0.2506 - categorical_accuracy: 0.9231
34368/60000 [================>.............] - ETA: 38s - loss: 0.2503 - categorical_accuracy: 0.9232
34432/60000 [================>.............] - ETA: 38s - loss: 0.2501 - categorical_accuracy: 0.9232
34496/60000 [================>.............] - ETA: 38s - loss: 0.2498 - categorical_accuracy: 0.9233
34560/60000 [================>.............] - ETA: 38s - loss: 0.2497 - categorical_accuracy: 0.9234
34624/60000 [================>.............] - ETA: 37s - loss: 0.2495 - categorical_accuracy: 0.9235
34688/60000 [================>.............] - ETA: 37s - loss: 0.2493 - categorical_accuracy: 0.9236
34752/60000 [================>.............] - ETA: 37s - loss: 0.2492 - categorical_accuracy: 0.9236
34784/60000 [================>.............] - ETA: 37s - loss: 0.2491 - categorical_accuracy: 0.9236
34848/60000 [================>.............] - ETA: 37s - loss: 0.2490 - categorical_accuracy: 0.9237
34912/60000 [================>.............] - ETA: 37s - loss: 0.2487 - categorical_accuracy: 0.9238
34976/60000 [================>.............] - ETA: 37s - loss: 0.2484 - categorical_accuracy: 0.9239
35040/60000 [================>.............] - ETA: 37s - loss: 0.2480 - categorical_accuracy: 0.9240
35072/60000 [================>.............] - ETA: 37s - loss: 0.2479 - categorical_accuracy: 0.9240
35136/60000 [================>.............] - ETA: 37s - loss: 0.2475 - categorical_accuracy: 0.9241
35200/60000 [================>.............] - ETA: 37s - loss: 0.2472 - categorical_accuracy: 0.9242
35264/60000 [================>.............] - ETA: 36s - loss: 0.2469 - categorical_accuracy: 0.9243
35328/60000 [================>.............] - ETA: 36s - loss: 0.2466 - categorical_accuracy: 0.9244
35392/60000 [================>.............] - ETA: 36s - loss: 0.2463 - categorical_accuracy: 0.9245
35424/60000 [================>.............] - ETA: 36s - loss: 0.2464 - categorical_accuracy: 0.9245
35488/60000 [================>.............] - ETA: 36s - loss: 0.2464 - categorical_accuracy: 0.9245
35552/60000 [================>.............] - ETA: 36s - loss: 0.2462 - categorical_accuracy: 0.9246
35616/60000 [================>.............] - ETA: 36s - loss: 0.2460 - categorical_accuracy: 0.9246
35680/60000 [================>.............] - ETA: 36s - loss: 0.2458 - categorical_accuracy: 0.9247
35744/60000 [================>.............] - ETA: 36s - loss: 0.2456 - categorical_accuracy: 0.9247
35808/60000 [================>.............] - ETA: 36s - loss: 0.2455 - categorical_accuracy: 0.9247
35872/60000 [================>.............] - ETA: 36s - loss: 0.2454 - categorical_accuracy: 0.9248
35936/60000 [================>.............] - ETA: 35s - loss: 0.2451 - categorical_accuracy: 0.9249
36000/60000 [=================>............] - ETA: 35s - loss: 0.2450 - categorical_accuracy: 0.9249
36032/60000 [=================>............] - ETA: 35s - loss: 0.2448 - categorical_accuracy: 0.9249
36064/60000 [=================>............] - ETA: 35s - loss: 0.2447 - categorical_accuracy: 0.9250
36128/60000 [=================>............] - ETA: 35s - loss: 0.2444 - categorical_accuracy: 0.9251
36192/60000 [=================>............] - ETA: 35s - loss: 0.2441 - categorical_accuracy: 0.9251
36256/60000 [=================>............] - ETA: 35s - loss: 0.2443 - categorical_accuracy: 0.9251
36320/60000 [=================>............] - ETA: 35s - loss: 0.2440 - categorical_accuracy: 0.9252
36384/60000 [=================>............] - ETA: 35s - loss: 0.2438 - categorical_accuracy: 0.9252
36448/60000 [=================>............] - ETA: 35s - loss: 0.2437 - categorical_accuracy: 0.9253
36480/60000 [=================>............] - ETA: 35s - loss: 0.2436 - categorical_accuracy: 0.9254
36512/60000 [=================>............] - ETA: 35s - loss: 0.2435 - categorical_accuracy: 0.9254
36544/60000 [=================>............] - ETA: 35s - loss: 0.2433 - categorical_accuracy: 0.9255
36576/60000 [=================>............] - ETA: 34s - loss: 0.2431 - categorical_accuracy: 0.9255
36640/60000 [=================>............] - ETA: 34s - loss: 0.2427 - categorical_accuracy: 0.9257
36704/60000 [=================>............] - ETA: 34s - loss: 0.2425 - categorical_accuracy: 0.9257
36736/60000 [=================>............] - ETA: 34s - loss: 0.2424 - categorical_accuracy: 0.9257
36800/60000 [=================>............] - ETA: 34s - loss: 0.2420 - categorical_accuracy: 0.9258
36864/60000 [=================>............] - ETA: 34s - loss: 0.2420 - categorical_accuracy: 0.9259
36928/60000 [=================>............] - ETA: 34s - loss: 0.2418 - categorical_accuracy: 0.9259
36992/60000 [=================>............] - ETA: 34s - loss: 0.2415 - categorical_accuracy: 0.9260
37056/60000 [=================>............] - ETA: 34s - loss: 0.2411 - categorical_accuracy: 0.9261
37120/60000 [=================>............] - ETA: 34s - loss: 0.2409 - categorical_accuracy: 0.9261
37184/60000 [=================>............] - ETA: 34s - loss: 0.2408 - categorical_accuracy: 0.9261
37248/60000 [=================>............] - ETA: 33s - loss: 0.2409 - categorical_accuracy: 0.9262
37280/60000 [=================>............] - ETA: 33s - loss: 0.2408 - categorical_accuracy: 0.9262
37312/60000 [=================>............] - ETA: 33s - loss: 0.2407 - categorical_accuracy: 0.9262
37344/60000 [=================>............] - ETA: 33s - loss: 0.2405 - categorical_accuracy: 0.9262
37408/60000 [=================>............] - ETA: 33s - loss: 0.2403 - categorical_accuracy: 0.9263
37472/60000 [=================>............] - ETA: 33s - loss: 0.2400 - categorical_accuracy: 0.9264
37536/60000 [=================>............] - ETA: 33s - loss: 0.2398 - categorical_accuracy: 0.9264
37600/60000 [=================>............] - ETA: 33s - loss: 0.2397 - categorical_accuracy: 0.9265
37664/60000 [=================>............] - ETA: 33s - loss: 0.2396 - categorical_accuracy: 0.9265
37728/60000 [=================>............] - ETA: 33s - loss: 0.2393 - categorical_accuracy: 0.9267
37792/60000 [=================>............] - ETA: 33s - loss: 0.2389 - categorical_accuracy: 0.9268
37856/60000 [=================>............] - ETA: 33s - loss: 0.2388 - categorical_accuracy: 0.9268
37920/60000 [=================>............] - ETA: 32s - loss: 0.2385 - categorical_accuracy: 0.9269
37984/60000 [=================>............] - ETA: 32s - loss: 0.2383 - categorical_accuracy: 0.9270
38048/60000 [==================>...........] - ETA: 32s - loss: 0.2381 - categorical_accuracy: 0.9270
38112/60000 [==================>...........] - ETA: 32s - loss: 0.2379 - categorical_accuracy: 0.9271
38176/60000 [==================>...........] - ETA: 32s - loss: 0.2377 - categorical_accuracy: 0.9272
38240/60000 [==================>...........] - ETA: 32s - loss: 0.2374 - categorical_accuracy: 0.9272
38304/60000 [==================>...........] - ETA: 32s - loss: 0.2374 - categorical_accuracy: 0.9273
38368/60000 [==================>...........] - ETA: 32s - loss: 0.2371 - categorical_accuracy: 0.9274
38432/60000 [==================>...........] - ETA: 32s - loss: 0.2369 - categorical_accuracy: 0.9274
38496/60000 [==================>...........] - ETA: 32s - loss: 0.2366 - categorical_accuracy: 0.9275
38560/60000 [==================>...........] - ETA: 32s - loss: 0.2363 - categorical_accuracy: 0.9276
38592/60000 [==================>...........] - ETA: 31s - loss: 0.2361 - categorical_accuracy: 0.9277
38656/60000 [==================>...........] - ETA: 31s - loss: 0.2357 - categorical_accuracy: 0.9278
38720/60000 [==================>...........] - ETA: 31s - loss: 0.2356 - categorical_accuracy: 0.9278
38784/60000 [==================>...........] - ETA: 31s - loss: 0.2354 - categorical_accuracy: 0.9279
38816/60000 [==================>...........] - ETA: 31s - loss: 0.2352 - categorical_accuracy: 0.9279
38880/60000 [==================>...........] - ETA: 31s - loss: 0.2349 - categorical_accuracy: 0.9281
38944/60000 [==================>...........] - ETA: 31s - loss: 0.2345 - categorical_accuracy: 0.9282
39008/60000 [==================>...........] - ETA: 31s - loss: 0.2344 - categorical_accuracy: 0.9282
39072/60000 [==================>...........] - ETA: 31s - loss: 0.2342 - categorical_accuracy: 0.9283
39136/60000 [==================>...........] - ETA: 31s - loss: 0.2342 - categorical_accuracy: 0.9282
39200/60000 [==================>...........] - ETA: 31s - loss: 0.2341 - categorical_accuracy: 0.9283
39264/60000 [==================>...........] - ETA: 30s - loss: 0.2342 - categorical_accuracy: 0.9283
39328/60000 [==================>...........] - ETA: 30s - loss: 0.2340 - categorical_accuracy: 0.9283
39392/60000 [==================>...........] - ETA: 30s - loss: 0.2338 - categorical_accuracy: 0.9284
39456/60000 [==================>...........] - ETA: 30s - loss: 0.2335 - categorical_accuracy: 0.9285
39520/60000 [==================>...........] - ETA: 30s - loss: 0.2334 - categorical_accuracy: 0.9285
39584/60000 [==================>...........] - ETA: 30s - loss: 0.2330 - categorical_accuracy: 0.9287
39648/60000 [==================>...........] - ETA: 30s - loss: 0.2328 - categorical_accuracy: 0.9287
39712/60000 [==================>...........] - ETA: 30s - loss: 0.2327 - categorical_accuracy: 0.9287
39744/60000 [==================>...........] - ETA: 30s - loss: 0.2326 - categorical_accuracy: 0.9287
39808/60000 [==================>...........] - ETA: 30s - loss: 0.2323 - categorical_accuracy: 0.9288
39872/60000 [==================>...........] - ETA: 30s - loss: 0.2320 - categorical_accuracy: 0.9289
39904/60000 [==================>...........] - ETA: 30s - loss: 0.2320 - categorical_accuracy: 0.9289
39968/60000 [==================>...........] - ETA: 29s - loss: 0.2319 - categorical_accuracy: 0.9290
40032/60000 [===================>..........] - ETA: 29s - loss: 0.2319 - categorical_accuracy: 0.9290
40096/60000 [===================>..........] - ETA: 29s - loss: 0.2317 - categorical_accuracy: 0.9291
40160/60000 [===================>..........] - ETA: 29s - loss: 0.2313 - categorical_accuracy: 0.9292
40192/60000 [===================>..........] - ETA: 29s - loss: 0.2314 - categorical_accuracy: 0.9291
40256/60000 [===================>..........] - ETA: 29s - loss: 0.2315 - categorical_accuracy: 0.9292
40320/60000 [===================>..........] - ETA: 29s - loss: 0.2313 - categorical_accuracy: 0.9292
40384/60000 [===================>..........] - ETA: 29s - loss: 0.2315 - categorical_accuracy: 0.9292
40448/60000 [===================>..........] - ETA: 29s - loss: 0.2313 - categorical_accuracy: 0.9292
40512/60000 [===================>..........] - ETA: 29s - loss: 0.2310 - categorical_accuracy: 0.9293
40576/60000 [===================>..........] - ETA: 29s - loss: 0.2307 - categorical_accuracy: 0.9294
40640/60000 [===================>..........] - ETA: 28s - loss: 0.2306 - categorical_accuracy: 0.9294
40704/60000 [===================>..........] - ETA: 28s - loss: 0.2303 - categorical_accuracy: 0.9295
40768/60000 [===================>..........] - ETA: 28s - loss: 0.2305 - categorical_accuracy: 0.9295
40832/60000 [===================>..........] - ETA: 28s - loss: 0.2303 - categorical_accuracy: 0.9296
40864/60000 [===================>..........] - ETA: 28s - loss: 0.2303 - categorical_accuracy: 0.9296
40896/60000 [===================>..........] - ETA: 28s - loss: 0.2302 - categorical_accuracy: 0.9296
40960/60000 [===================>..........] - ETA: 28s - loss: 0.2299 - categorical_accuracy: 0.9297
40992/60000 [===================>..........] - ETA: 28s - loss: 0.2299 - categorical_accuracy: 0.9297
41024/60000 [===================>..........] - ETA: 28s - loss: 0.2297 - categorical_accuracy: 0.9298
41088/60000 [===================>..........] - ETA: 28s - loss: 0.2297 - categorical_accuracy: 0.9297
41120/60000 [===================>..........] - ETA: 28s - loss: 0.2295 - categorical_accuracy: 0.9298
41184/60000 [===================>..........] - ETA: 28s - loss: 0.2293 - categorical_accuracy: 0.9299
41216/60000 [===================>..........] - ETA: 28s - loss: 0.2291 - categorical_accuracy: 0.9299
41280/60000 [===================>..........] - ETA: 27s - loss: 0.2289 - categorical_accuracy: 0.9300
41344/60000 [===================>..........] - ETA: 27s - loss: 0.2286 - categorical_accuracy: 0.9301
41408/60000 [===================>..........] - ETA: 27s - loss: 0.2283 - categorical_accuracy: 0.9301
41472/60000 [===================>..........] - ETA: 27s - loss: 0.2281 - categorical_accuracy: 0.9302
41536/60000 [===================>..........] - ETA: 27s - loss: 0.2278 - categorical_accuracy: 0.9303
41600/60000 [===================>..........] - ETA: 27s - loss: 0.2275 - categorical_accuracy: 0.9304
41664/60000 [===================>..........] - ETA: 27s - loss: 0.2272 - categorical_accuracy: 0.9304
41728/60000 [===================>..........] - ETA: 27s - loss: 0.2271 - categorical_accuracy: 0.9305
41792/60000 [===================>..........] - ETA: 27s - loss: 0.2270 - categorical_accuracy: 0.9305
41856/60000 [===================>..........] - ETA: 27s - loss: 0.2268 - categorical_accuracy: 0.9306
41920/60000 [===================>..........] - ETA: 27s - loss: 0.2266 - categorical_accuracy: 0.9307
41984/60000 [===================>..........] - ETA: 26s - loss: 0.2264 - categorical_accuracy: 0.9307
42048/60000 [====================>.........] - ETA: 26s - loss: 0.2262 - categorical_accuracy: 0.9307
42112/60000 [====================>.........] - ETA: 26s - loss: 0.2260 - categorical_accuracy: 0.9308
42176/60000 [====================>.........] - ETA: 26s - loss: 0.2257 - categorical_accuracy: 0.9309
42240/60000 [====================>.........] - ETA: 26s - loss: 0.2255 - categorical_accuracy: 0.9309
42304/60000 [====================>.........] - ETA: 26s - loss: 0.2253 - categorical_accuracy: 0.9310
42368/60000 [====================>.........] - ETA: 26s - loss: 0.2252 - categorical_accuracy: 0.9311
42432/60000 [====================>.........] - ETA: 26s - loss: 0.2250 - categorical_accuracy: 0.9311
42496/60000 [====================>.........] - ETA: 26s - loss: 0.2249 - categorical_accuracy: 0.9312
42560/60000 [====================>.........] - ETA: 26s - loss: 0.2247 - categorical_accuracy: 0.9312
42624/60000 [====================>.........] - ETA: 25s - loss: 0.2248 - categorical_accuracy: 0.9313
42688/60000 [====================>.........] - ETA: 25s - loss: 0.2245 - categorical_accuracy: 0.9313
42752/60000 [====================>.........] - ETA: 25s - loss: 0.2245 - categorical_accuracy: 0.9314
42816/60000 [====================>.........] - ETA: 25s - loss: 0.2244 - categorical_accuracy: 0.9314
42880/60000 [====================>.........] - ETA: 25s - loss: 0.2243 - categorical_accuracy: 0.9314
42944/60000 [====================>.........] - ETA: 25s - loss: 0.2241 - categorical_accuracy: 0.9315
43008/60000 [====================>.........] - ETA: 25s - loss: 0.2239 - categorical_accuracy: 0.9315
43072/60000 [====================>.........] - ETA: 25s - loss: 0.2238 - categorical_accuracy: 0.9315
43136/60000 [====================>.........] - ETA: 25s - loss: 0.2236 - categorical_accuracy: 0.9316
43200/60000 [====================>.........] - ETA: 25s - loss: 0.2234 - categorical_accuracy: 0.9317
43264/60000 [====================>.........] - ETA: 24s - loss: 0.2232 - categorical_accuracy: 0.9317
43296/60000 [====================>.........] - ETA: 24s - loss: 0.2232 - categorical_accuracy: 0.9317
43328/60000 [====================>.........] - ETA: 24s - loss: 0.2232 - categorical_accuracy: 0.9317
43360/60000 [====================>.........] - ETA: 24s - loss: 0.2230 - categorical_accuracy: 0.9317
43424/60000 [====================>.........] - ETA: 24s - loss: 0.2228 - categorical_accuracy: 0.9318
43488/60000 [====================>.........] - ETA: 24s - loss: 0.2228 - categorical_accuracy: 0.9318
43552/60000 [====================>.........] - ETA: 24s - loss: 0.2226 - categorical_accuracy: 0.9319
43584/60000 [====================>.........] - ETA: 24s - loss: 0.2225 - categorical_accuracy: 0.9319
43648/60000 [====================>.........] - ETA: 24s - loss: 0.2223 - categorical_accuracy: 0.9320
43712/60000 [====================>.........] - ETA: 24s - loss: 0.2220 - categorical_accuracy: 0.9321
43776/60000 [====================>.........] - ETA: 24s - loss: 0.2217 - categorical_accuracy: 0.9322
43840/60000 [====================>.........] - ETA: 24s - loss: 0.2217 - categorical_accuracy: 0.9322
43904/60000 [====================>.........] - ETA: 24s - loss: 0.2215 - categorical_accuracy: 0.9322
43968/60000 [====================>.........] - ETA: 23s - loss: 0.2214 - categorical_accuracy: 0.9323
44032/60000 [=====================>........] - ETA: 23s - loss: 0.2213 - categorical_accuracy: 0.9323
44096/60000 [=====================>........] - ETA: 23s - loss: 0.2210 - categorical_accuracy: 0.9324
44160/60000 [=====================>........] - ETA: 23s - loss: 0.2208 - categorical_accuracy: 0.9325
44224/60000 [=====================>........] - ETA: 23s - loss: 0.2205 - categorical_accuracy: 0.9325
44288/60000 [=====================>........] - ETA: 23s - loss: 0.2208 - categorical_accuracy: 0.9325
44352/60000 [=====================>........] - ETA: 23s - loss: 0.2207 - categorical_accuracy: 0.9325
44416/60000 [=====================>........] - ETA: 23s - loss: 0.2206 - categorical_accuracy: 0.9325
44480/60000 [=====================>........] - ETA: 23s - loss: 0.2203 - categorical_accuracy: 0.9326
44544/60000 [=====================>........] - ETA: 23s - loss: 0.2203 - categorical_accuracy: 0.9326
44608/60000 [=====================>........] - ETA: 22s - loss: 0.2201 - categorical_accuracy: 0.9326
44672/60000 [=====================>........] - ETA: 22s - loss: 0.2199 - categorical_accuracy: 0.9326
44736/60000 [=====================>........] - ETA: 22s - loss: 0.2199 - categorical_accuracy: 0.9326
44800/60000 [=====================>........] - ETA: 22s - loss: 0.2196 - categorical_accuracy: 0.9327
44864/60000 [=====================>........] - ETA: 22s - loss: 0.2194 - categorical_accuracy: 0.9327
44928/60000 [=====================>........] - ETA: 22s - loss: 0.2195 - categorical_accuracy: 0.9327
44992/60000 [=====================>........] - ETA: 22s - loss: 0.2193 - categorical_accuracy: 0.9328
45024/60000 [=====================>........] - ETA: 22s - loss: 0.2191 - categorical_accuracy: 0.9328
45088/60000 [=====================>........] - ETA: 22s - loss: 0.2191 - categorical_accuracy: 0.9328
45152/60000 [=====================>........] - ETA: 22s - loss: 0.2189 - categorical_accuracy: 0.9328
45216/60000 [=====================>........] - ETA: 22s - loss: 0.2187 - categorical_accuracy: 0.9329
45280/60000 [=====================>........] - ETA: 21s - loss: 0.2185 - categorical_accuracy: 0.9329
45344/60000 [=====================>........] - ETA: 21s - loss: 0.2183 - categorical_accuracy: 0.9330
45408/60000 [=====================>........] - ETA: 21s - loss: 0.2181 - categorical_accuracy: 0.9331
45472/60000 [=====================>........] - ETA: 21s - loss: 0.2179 - categorical_accuracy: 0.9332
45536/60000 [=====================>........] - ETA: 21s - loss: 0.2177 - categorical_accuracy: 0.9332
45600/60000 [=====================>........] - ETA: 21s - loss: 0.2175 - categorical_accuracy: 0.9333
45664/60000 [=====================>........] - ETA: 21s - loss: 0.2174 - categorical_accuracy: 0.9333
45696/60000 [=====================>........] - ETA: 21s - loss: 0.2173 - categorical_accuracy: 0.9333
45760/60000 [=====================>........] - ETA: 21s - loss: 0.2172 - categorical_accuracy: 0.9333
45824/60000 [=====================>........] - ETA: 21s - loss: 0.2170 - categorical_accuracy: 0.9334
45888/60000 [=====================>........] - ETA: 21s - loss: 0.2169 - categorical_accuracy: 0.9334
45952/60000 [=====================>........] - ETA: 20s - loss: 0.2167 - categorical_accuracy: 0.9335
46016/60000 [======================>.......] - ETA: 20s - loss: 0.2166 - categorical_accuracy: 0.9335
46080/60000 [======================>.......] - ETA: 20s - loss: 0.2165 - categorical_accuracy: 0.9336
46144/60000 [======================>.......] - ETA: 20s - loss: 0.2163 - categorical_accuracy: 0.9336
46208/60000 [======================>.......] - ETA: 20s - loss: 0.2161 - categorical_accuracy: 0.9336
46272/60000 [======================>.......] - ETA: 20s - loss: 0.2161 - categorical_accuracy: 0.9337
46336/60000 [======================>.......] - ETA: 20s - loss: 0.2159 - categorical_accuracy: 0.9337
46400/60000 [======================>.......] - ETA: 20s - loss: 0.2157 - categorical_accuracy: 0.9338
46432/60000 [======================>.......] - ETA: 20s - loss: 0.2156 - categorical_accuracy: 0.9338
46496/60000 [======================>.......] - ETA: 20s - loss: 0.2154 - categorical_accuracy: 0.9339
46560/60000 [======================>.......] - ETA: 20s - loss: 0.2152 - categorical_accuracy: 0.9339
46624/60000 [======================>.......] - ETA: 19s - loss: 0.2151 - categorical_accuracy: 0.9339
46656/60000 [======================>.......] - ETA: 19s - loss: 0.2149 - categorical_accuracy: 0.9340
46688/60000 [======================>.......] - ETA: 19s - loss: 0.2148 - categorical_accuracy: 0.9340
46720/60000 [======================>.......] - ETA: 19s - loss: 0.2149 - categorical_accuracy: 0.9341
46784/60000 [======================>.......] - ETA: 19s - loss: 0.2148 - categorical_accuracy: 0.9341
46848/60000 [======================>.......] - ETA: 19s - loss: 0.2147 - categorical_accuracy: 0.9341
46912/60000 [======================>.......] - ETA: 19s - loss: 0.2145 - categorical_accuracy: 0.9342
46976/60000 [======================>.......] - ETA: 19s - loss: 0.2143 - categorical_accuracy: 0.9343
47040/60000 [======================>.......] - ETA: 19s - loss: 0.2140 - categorical_accuracy: 0.9343
47072/60000 [======================>.......] - ETA: 19s - loss: 0.2139 - categorical_accuracy: 0.9344
47104/60000 [======================>.......] - ETA: 19s - loss: 0.2138 - categorical_accuracy: 0.9344
47136/60000 [======================>.......] - ETA: 19s - loss: 0.2137 - categorical_accuracy: 0.9344
47200/60000 [======================>.......] - ETA: 19s - loss: 0.2137 - categorical_accuracy: 0.9345
47264/60000 [======================>.......] - ETA: 19s - loss: 0.2134 - categorical_accuracy: 0.9346
47296/60000 [======================>.......] - ETA: 18s - loss: 0.2134 - categorical_accuracy: 0.9346
47360/60000 [======================>.......] - ETA: 18s - loss: 0.2132 - categorical_accuracy: 0.9347
47424/60000 [======================>.......] - ETA: 18s - loss: 0.2130 - categorical_accuracy: 0.9347
47488/60000 [======================>.......] - ETA: 18s - loss: 0.2129 - categorical_accuracy: 0.9348
47520/60000 [======================>.......] - ETA: 18s - loss: 0.2128 - categorical_accuracy: 0.9348
47584/60000 [======================>.......] - ETA: 18s - loss: 0.2125 - categorical_accuracy: 0.9349
47648/60000 [======================>.......] - ETA: 18s - loss: 0.2125 - categorical_accuracy: 0.9349
47712/60000 [======================>.......] - ETA: 18s - loss: 0.2127 - categorical_accuracy: 0.9349
47776/60000 [======================>.......] - ETA: 18s - loss: 0.2125 - categorical_accuracy: 0.9349
47840/60000 [======================>.......] - ETA: 18s - loss: 0.2124 - categorical_accuracy: 0.9350
47904/60000 [======================>.......] - ETA: 18s - loss: 0.2122 - categorical_accuracy: 0.9350
47968/60000 [======================>.......] - ETA: 17s - loss: 0.2119 - categorical_accuracy: 0.9351
48032/60000 [=======================>......] - ETA: 17s - loss: 0.2117 - categorical_accuracy: 0.9352
48096/60000 [=======================>......] - ETA: 17s - loss: 0.2115 - categorical_accuracy: 0.9353
48160/60000 [=======================>......] - ETA: 17s - loss: 0.2113 - categorical_accuracy: 0.9353
48224/60000 [=======================>......] - ETA: 17s - loss: 0.2112 - categorical_accuracy: 0.9353
48288/60000 [=======================>......] - ETA: 17s - loss: 0.2110 - categorical_accuracy: 0.9354
48352/60000 [=======================>......] - ETA: 17s - loss: 0.2108 - categorical_accuracy: 0.9355
48416/60000 [=======================>......] - ETA: 17s - loss: 0.2106 - categorical_accuracy: 0.9355
48480/60000 [=======================>......] - ETA: 17s - loss: 0.2103 - categorical_accuracy: 0.9356
48544/60000 [=======================>......] - ETA: 17s - loss: 0.2101 - categorical_accuracy: 0.9357
48608/60000 [=======================>......] - ETA: 17s - loss: 0.2099 - categorical_accuracy: 0.9357
48672/60000 [=======================>......] - ETA: 16s - loss: 0.2098 - categorical_accuracy: 0.9358
48736/60000 [=======================>......] - ETA: 16s - loss: 0.2097 - categorical_accuracy: 0.9358
48800/60000 [=======================>......] - ETA: 16s - loss: 0.2095 - categorical_accuracy: 0.9359
48864/60000 [=======================>......] - ETA: 16s - loss: 0.2093 - categorical_accuracy: 0.9359
48928/60000 [=======================>......] - ETA: 16s - loss: 0.2091 - categorical_accuracy: 0.9360
48992/60000 [=======================>......] - ETA: 16s - loss: 0.2091 - categorical_accuracy: 0.9360
49056/60000 [=======================>......] - ETA: 16s - loss: 0.2089 - categorical_accuracy: 0.9360
49120/60000 [=======================>......] - ETA: 16s - loss: 0.2089 - categorical_accuracy: 0.9360
49152/60000 [=======================>......] - ETA: 16s - loss: 0.2089 - categorical_accuracy: 0.9360
49216/60000 [=======================>......] - ETA: 16s - loss: 0.2086 - categorical_accuracy: 0.9361
49280/60000 [=======================>......] - ETA: 15s - loss: 0.2086 - categorical_accuracy: 0.9361
49344/60000 [=======================>......] - ETA: 15s - loss: 0.2084 - categorical_accuracy: 0.9362
49408/60000 [=======================>......] - ETA: 15s - loss: 0.2085 - categorical_accuracy: 0.9362
49472/60000 [=======================>......] - ETA: 15s - loss: 0.2083 - categorical_accuracy: 0.9363
49504/60000 [=======================>......] - ETA: 15s - loss: 0.2082 - categorical_accuracy: 0.9363
49568/60000 [=======================>......] - ETA: 15s - loss: 0.2080 - categorical_accuracy: 0.9364
49632/60000 [=======================>......] - ETA: 15s - loss: 0.2080 - categorical_accuracy: 0.9364
49664/60000 [=======================>......] - ETA: 15s - loss: 0.2081 - categorical_accuracy: 0.9364
49696/60000 [=======================>......] - ETA: 15s - loss: 0.2080 - categorical_accuracy: 0.9364
49760/60000 [=======================>......] - ETA: 15s - loss: 0.2079 - categorical_accuracy: 0.9364
49824/60000 [=======================>......] - ETA: 15s - loss: 0.2078 - categorical_accuracy: 0.9364
49888/60000 [=======================>......] - ETA: 15s - loss: 0.2076 - categorical_accuracy: 0.9365
49952/60000 [=======================>......] - ETA: 14s - loss: 0.2075 - categorical_accuracy: 0.9365
50016/60000 [========================>.....] - ETA: 14s - loss: 0.2073 - categorical_accuracy: 0.9366
50048/60000 [========================>.....] - ETA: 14s - loss: 0.2072 - categorical_accuracy: 0.9366
50112/60000 [========================>.....] - ETA: 14s - loss: 0.2070 - categorical_accuracy: 0.9366
50176/60000 [========================>.....] - ETA: 14s - loss: 0.2068 - categorical_accuracy: 0.9367
50208/60000 [========================>.....] - ETA: 14s - loss: 0.2067 - categorical_accuracy: 0.9368
50240/60000 [========================>.....] - ETA: 14s - loss: 0.2065 - categorical_accuracy: 0.9368
50272/60000 [========================>.....] - ETA: 14s - loss: 0.2065 - categorical_accuracy: 0.9368
50336/60000 [========================>.....] - ETA: 14s - loss: 0.2062 - categorical_accuracy: 0.9369
50400/60000 [========================>.....] - ETA: 14s - loss: 0.2061 - categorical_accuracy: 0.9369
50464/60000 [========================>.....] - ETA: 14s - loss: 0.2059 - categorical_accuracy: 0.9370
50528/60000 [========================>.....] - ETA: 14s - loss: 0.2057 - categorical_accuracy: 0.9371
50592/60000 [========================>.....] - ETA: 14s - loss: 0.2060 - categorical_accuracy: 0.9370
50656/60000 [========================>.....] - ETA: 13s - loss: 0.2058 - categorical_accuracy: 0.9371
50720/60000 [========================>.....] - ETA: 13s - loss: 0.2056 - categorical_accuracy: 0.9372
50752/60000 [========================>.....] - ETA: 13s - loss: 0.2055 - categorical_accuracy: 0.9372
50784/60000 [========================>.....] - ETA: 13s - loss: 0.2055 - categorical_accuracy: 0.9372
50848/60000 [========================>.....] - ETA: 13s - loss: 0.2054 - categorical_accuracy: 0.9372
50912/60000 [========================>.....] - ETA: 13s - loss: 0.2053 - categorical_accuracy: 0.9373
50976/60000 [========================>.....] - ETA: 13s - loss: 0.2053 - categorical_accuracy: 0.9372
51008/60000 [========================>.....] - ETA: 13s - loss: 0.2051 - categorical_accuracy: 0.9373
51072/60000 [========================>.....] - ETA: 13s - loss: 0.2049 - categorical_accuracy: 0.9373
51136/60000 [========================>.....] - ETA: 13s - loss: 0.2048 - categorical_accuracy: 0.9373
51168/60000 [========================>.....] - ETA: 13s - loss: 0.2047 - categorical_accuracy: 0.9374
51200/60000 [========================>.....] - ETA: 13s - loss: 0.2046 - categorical_accuracy: 0.9374
51264/60000 [========================>.....] - ETA: 13s - loss: 0.2045 - categorical_accuracy: 0.9375
51328/60000 [========================>.....] - ETA: 12s - loss: 0.2043 - categorical_accuracy: 0.9375
51392/60000 [========================>.....] - ETA: 12s - loss: 0.2042 - categorical_accuracy: 0.9375
51456/60000 [========================>.....] - ETA: 12s - loss: 0.2039 - categorical_accuracy: 0.9376
51520/60000 [========================>.....] - ETA: 12s - loss: 0.2038 - categorical_accuracy: 0.9376
51584/60000 [========================>.....] - ETA: 12s - loss: 0.2035 - categorical_accuracy: 0.9377
51648/60000 [========================>.....] - ETA: 12s - loss: 0.2033 - categorical_accuracy: 0.9378
51712/60000 [========================>.....] - ETA: 12s - loss: 0.2031 - categorical_accuracy: 0.9378
51776/60000 [========================>.....] - ETA: 12s - loss: 0.2029 - categorical_accuracy: 0.9379
51808/60000 [========================>.....] - ETA: 12s - loss: 0.2028 - categorical_accuracy: 0.9379
51872/60000 [========================>.....] - ETA: 12s - loss: 0.2027 - categorical_accuracy: 0.9379
51936/60000 [========================>.....] - ETA: 12s - loss: 0.2027 - categorical_accuracy: 0.9379
52000/60000 [=========================>....] - ETA: 11s - loss: 0.2025 - categorical_accuracy: 0.9380
52064/60000 [=========================>....] - ETA: 11s - loss: 0.2023 - categorical_accuracy: 0.9380
52128/60000 [=========================>....] - ETA: 11s - loss: 0.2021 - categorical_accuracy: 0.9381
52192/60000 [=========================>....] - ETA: 11s - loss: 0.2019 - categorical_accuracy: 0.9382
52256/60000 [=========================>....] - ETA: 11s - loss: 0.2017 - categorical_accuracy: 0.9382
52320/60000 [=========================>....] - ETA: 11s - loss: 0.2016 - categorical_accuracy: 0.9383
52352/60000 [=========================>....] - ETA: 11s - loss: 0.2015 - categorical_accuracy: 0.9383
52416/60000 [=========================>....] - ETA: 11s - loss: 0.2014 - categorical_accuracy: 0.9383
52480/60000 [=========================>....] - ETA: 11s - loss: 0.2012 - categorical_accuracy: 0.9384
52544/60000 [=========================>....] - ETA: 11s - loss: 0.2012 - categorical_accuracy: 0.9384
52608/60000 [=========================>....] - ETA: 11s - loss: 0.2011 - categorical_accuracy: 0.9385
52672/60000 [=========================>....] - ETA: 10s - loss: 0.2011 - categorical_accuracy: 0.9385
52736/60000 [=========================>....] - ETA: 10s - loss: 0.2009 - categorical_accuracy: 0.9386
52800/60000 [=========================>....] - ETA: 10s - loss: 0.2007 - categorical_accuracy: 0.9386
52864/60000 [=========================>....] - ETA: 10s - loss: 0.2005 - categorical_accuracy: 0.9387
52928/60000 [=========================>....] - ETA: 10s - loss: 0.2003 - categorical_accuracy: 0.9387
52992/60000 [=========================>....] - ETA: 10s - loss: 0.2001 - categorical_accuracy: 0.9388
53024/60000 [=========================>....] - ETA: 10s - loss: 0.2000 - categorical_accuracy: 0.9388
53088/60000 [=========================>....] - ETA: 10s - loss: 0.1999 - categorical_accuracy: 0.9389
53152/60000 [=========================>....] - ETA: 10s - loss: 0.1997 - categorical_accuracy: 0.9389
53216/60000 [=========================>....] - ETA: 10s - loss: 0.1998 - categorical_accuracy: 0.9389
53248/60000 [=========================>....] - ETA: 10s - loss: 0.1997 - categorical_accuracy: 0.9389
53312/60000 [=========================>....] - ETA: 9s - loss: 0.1997 - categorical_accuracy: 0.9390 
53344/60000 [=========================>....] - ETA: 9s - loss: 0.1996 - categorical_accuracy: 0.9390
53376/60000 [=========================>....] - ETA: 9s - loss: 0.1995 - categorical_accuracy: 0.9390
53440/60000 [=========================>....] - ETA: 9s - loss: 0.1994 - categorical_accuracy: 0.9391
53504/60000 [=========================>....] - ETA: 9s - loss: 0.1992 - categorical_accuracy: 0.9391
53568/60000 [=========================>....] - ETA: 9s - loss: 0.1990 - categorical_accuracy: 0.9392
53632/60000 [=========================>....] - ETA: 9s - loss: 0.1990 - categorical_accuracy: 0.9392
53664/60000 [=========================>....] - ETA: 9s - loss: 0.1989 - categorical_accuracy: 0.9392
53728/60000 [=========================>....] - ETA: 9s - loss: 0.1988 - categorical_accuracy: 0.9392
53792/60000 [=========================>....] - ETA: 9s - loss: 0.1987 - categorical_accuracy: 0.9393
53824/60000 [=========================>....] - ETA: 9s - loss: 0.1985 - categorical_accuracy: 0.9393
53856/60000 [=========================>....] - ETA: 9s - loss: 0.1984 - categorical_accuracy: 0.9393
53920/60000 [=========================>....] - ETA: 9s - loss: 0.1983 - categorical_accuracy: 0.9394
53984/60000 [=========================>....] - ETA: 8s - loss: 0.1983 - categorical_accuracy: 0.9394
54048/60000 [==========================>...] - ETA: 8s - loss: 0.1981 - categorical_accuracy: 0.9395
54112/60000 [==========================>...] - ETA: 8s - loss: 0.1981 - categorical_accuracy: 0.9395
54176/60000 [==========================>...] - ETA: 8s - loss: 0.1980 - categorical_accuracy: 0.9395
54208/60000 [==========================>...] - ETA: 8s - loss: 0.1980 - categorical_accuracy: 0.9395
54240/60000 [==========================>...] - ETA: 8s - loss: 0.1980 - categorical_accuracy: 0.9395
54272/60000 [==========================>...] - ETA: 8s - loss: 0.1980 - categorical_accuracy: 0.9395
54336/60000 [==========================>...] - ETA: 8s - loss: 0.1978 - categorical_accuracy: 0.9396
54400/60000 [==========================>...] - ETA: 8s - loss: 0.1976 - categorical_accuracy: 0.9397
54464/60000 [==========================>...] - ETA: 8s - loss: 0.1975 - categorical_accuracy: 0.9397
54528/60000 [==========================>...] - ETA: 8s - loss: 0.1976 - categorical_accuracy: 0.9396
54592/60000 [==========================>...] - ETA: 8s - loss: 0.1976 - categorical_accuracy: 0.9396
54624/60000 [==========================>...] - ETA: 8s - loss: 0.1975 - categorical_accuracy: 0.9397
54688/60000 [==========================>...] - ETA: 7s - loss: 0.1974 - categorical_accuracy: 0.9397
54752/60000 [==========================>...] - ETA: 7s - loss: 0.1972 - categorical_accuracy: 0.9397
54816/60000 [==========================>...] - ETA: 7s - loss: 0.1971 - categorical_accuracy: 0.9398
54880/60000 [==========================>...] - ETA: 7s - loss: 0.1970 - categorical_accuracy: 0.9398
54944/60000 [==========================>...] - ETA: 7s - loss: 0.1968 - categorical_accuracy: 0.9398
55008/60000 [==========================>...] - ETA: 7s - loss: 0.1967 - categorical_accuracy: 0.9399
55040/60000 [==========================>...] - ETA: 7s - loss: 0.1966 - categorical_accuracy: 0.9399
55104/60000 [==========================>...] - ETA: 7s - loss: 0.1965 - categorical_accuracy: 0.9400
55168/60000 [==========================>...] - ETA: 7s - loss: 0.1964 - categorical_accuracy: 0.9400
55232/60000 [==========================>...] - ETA: 7s - loss: 0.1963 - categorical_accuracy: 0.9400
55264/60000 [==========================>...] - ETA: 7s - loss: 0.1962 - categorical_accuracy: 0.9400
55296/60000 [==========================>...] - ETA: 7s - loss: 0.1962 - categorical_accuracy: 0.9400
55360/60000 [==========================>...] - ETA: 6s - loss: 0.1960 - categorical_accuracy: 0.9401
55424/60000 [==========================>...] - ETA: 6s - loss: 0.1959 - categorical_accuracy: 0.9401
55488/60000 [==========================>...] - ETA: 6s - loss: 0.1957 - categorical_accuracy: 0.9402
55552/60000 [==========================>...] - ETA: 6s - loss: 0.1958 - categorical_accuracy: 0.9402
55616/60000 [==========================>...] - ETA: 6s - loss: 0.1957 - categorical_accuracy: 0.9402
55680/60000 [==========================>...] - ETA: 6s - loss: 0.1956 - categorical_accuracy: 0.9402
55744/60000 [==========================>...] - ETA: 6s - loss: 0.1954 - categorical_accuracy: 0.9402
55808/60000 [==========================>...] - ETA: 6s - loss: 0.1952 - categorical_accuracy: 0.9403
55872/60000 [==========================>...] - ETA: 6s - loss: 0.1951 - categorical_accuracy: 0.9403
55936/60000 [==========================>...] - ETA: 6s - loss: 0.1950 - categorical_accuracy: 0.9404
56000/60000 [===========================>..] - ETA: 5s - loss: 0.1951 - categorical_accuracy: 0.9404
56064/60000 [===========================>..] - ETA: 5s - loss: 0.1949 - categorical_accuracy: 0.9404
56128/60000 [===========================>..] - ETA: 5s - loss: 0.1947 - categorical_accuracy: 0.9405
56192/60000 [===========================>..] - ETA: 5s - loss: 0.1945 - categorical_accuracy: 0.9406
56256/60000 [===========================>..] - ETA: 5s - loss: 0.1944 - categorical_accuracy: 0.9406
56320/60000 [===========================>..] - ETA: 5s - loss: 0.1943 - categorical_accuracy: 0.9407
56384/60000 [===========================>..] - ETA: 5s - loss: 0.1943 - categorical_accuracy: 0.9406
56448/60000 [===========================>..] - ETA: 5s - loss: 0.1942 - categorical_accuracy: 0.9407
56512/60000 [===========================>..] - ETA: 5s - loss: 0.1942 - categorical_accuracy: 0.9407
56576/60000 [===========================>..] - ETA: 5s - loss: 0.1942 - categorical_accuracy: 0.9407
56640/60000 [===========================>..] - ETA: 5s - loss: 0.1941 - categorical_accuracy: 0.9407
56672/60000 [===========================>..] - ETA: 4s - loss: 0.1940 - categorical_accuracy: 0.9407
56736/60000 [===========================>..] - ETA: 4s - loss: 0.1938 - categorical_accuracy: 0.9408
56800/60000 [===========================>..] - ETA: 4s - loss: 0.1937 - categorical_accuracy: 0.9409
56864/60000 [===========================>..] - ETA: 4s - loss: 0.1935 - categorical_accuracy: 0.9409
56928/60000 [===========================>..] - ETA: 4s - loss: 0.1935 - categorical_accuracy: 0.9409
56992/60000 [===========================>..] - ETA: 4s - loss: 0.1934 - categorical_accuracy: 0.9410
57024/60000 [===========================>..] - ETA: 4s - loss: 0.1934 - categorical_accuracy: 0.9410
57088/60000 [===========================>..] - ETA: 4s - loss: 0.1935 - categorical_accuracy: 0.9410
57152/60000 [===========================>..] - ETA: 4s - loss: 0.1934 - categorical_accuracy: 0.9410
57216/60000 [===========================>..] - ETA: 4s - loss: 0.1937 - categorical_accuracy: 0.9410
57280/60000 [===========================>..] - ETA: 4s - loss: 0.1936 - categorical_accuracy: 0.9410
57344/60000 [===========================>..] - ETA: 3s - loss: 0.1934 - categorical_accuracy: 0.9411
57376/60000 [===========================>..] - ETA: 3s - loss: 0.1934 - categorical_accuracy: 0.9411
57408/60000 [===========================>..] - ETA: 3s - loss: 0.1934 - categorical_accuracy: 0.9411
57472/60000 [===========================>..] - ETA: 3s - loss: 0.1932 - categorical_accuracy: 0.9411
57536/60000 [===========================>..] - ETA: 3s - loss: 0.1931 - categorical_accuracy: 0.9411
57600/60000 [===========================>..] - ETA: 3s - loss: 0.1929 - categorical_accuracy: 0.9412
57664/60000 [===========================>..] - ETA: 3s - loss: 0.1928 - categorical_accuracy: 0.9412
57728/60000 [===========================>..] - ETA: 3s - loss: 0.1926 - categorical_accuracy: 0.9413
57792/60000 [===========================>..] - ETA: 3s - loss: 0.1927 - categorical_accuracy: 0.9413
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1925 - categorical_accuracy: 0.9413
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1923 - categorical_accuracy: 0.9414
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1923 - categorical_accuracy: 0.9414
58048/60000 [============================>.] - ETA: 2s - loss: 0.1922 - categorical_accuracy: 0.9415
58112/60000 [============================>.] - ETA: 2s - loss: 0.1923 - categorical_accuracy: 0.9414
58176/60000 [============================>.] - ETA: 2s - loss: 0.1922 - categorical_accuracy: 0.9415
58240/60000 [============================>.] - ETA: 2s - loss: 0.1921 - categorical_accuracy: 0.9415
58304/60000 [============================>.] - ETA: 2s - loss: 0.1921 - categorical_accuracy: 0.9415
58368/60000 [============================>.] - ETA: 2s - loss: 0.1921 - categorical_accuracy: 0.9415
58432/60000 [============================>.] - ETA: 2s - loss: 0.1921 - categorical_accuracy: 0.9416
58496/60000 [============================>.] - ETA: 2s - loss: 0.1920 - categorical_accuracy: 0.9416
58560/60000 [============================>.] - ETA: 2s - loss: 0.1920 - categorical_accuracy: 0.9416
58592/60000 [============================>.] - ETA: 2s - loss: 0.1919 - categorical_accuracy: 0.9416
58656/60000 [============================>.] - ETA: 2s - loss: 0.1917 - categorical_accuracy: 0.9416
58720/60000 [============================>.] - ETA: 1s - loss: 0.1916 - categorical_accuracy: 0.9417
58784/60000 [============================>.] - ETA: 1s - loss: 0.1915 - categorical_accuracy: 0.9417
58848/60000 [============================>.] - ETA: 1s - loss: 0.1914 - categorical_accuracy: 0.9417
58912/60000 [============================>.] - ETA: 1s - loss: 0.1913 - categorical_accuracy: 0.9418
58944/60000 [============================>.] - ETA: 1s - loss: 0.1912 - categorical_accuracy: 0.9418
58976/60000 [============================>.] - ETA: 1s - loss: 0.1911 - categorical_accuracy: 0.9418
59040/60000 [============================>.] - ETA: 1s - loss: 0.1909 - categorical_accuracy: 0.9419
59104/60000 [============================>.] - ETA: 1s - loss: 0.1908 - categorical_accuracy: 0.9419
59168/60000 [============================>.] - ETA: 1s - loss: 0.1906 - categorical_accuracy: 0.9420
59232/60000 [============================>.] - ETA: 1s - loss: 0.1907 - categorical_accuracy: 0.9420
59296/60000 [============================>.] - ETA: 1s - loss: 0.1906 - categorical_accuracy: 0.9420
59360/60000 [============================>.] - ETA: 0s - loss: 0.1905 - categorical_accuracy: 0.9420
59424/60000 [============================>.] - ETA: 0s - loss: 0.1905 - categorical_accuracy: 0.9421
59456/60000 [============================>.] - ETA: 0s - loss: 0.1904 - categorical_accuracy: 0.9421
59520/60000 [============================>.] - ETA: 0s - loss: 0.1902 - categorical_accuracy: 0.9421
59552/60000 [============================>.] - ETA: 0s - loss: 0.1906 - categorical_accuracy: 0.9421
59584/60000 [============================>.] - ETA: 0s - loss: 0.1905 - categorical_accuracy: 0.9421
59648/60000 [============================>.] - ETA: 0s - loss: 0.1904 - categorical_accuracy: 0.9422
59712/60000 [============================>.] - ETA: 0s - loss: 0.1904 - categorical_accuracy: 0.9422
59776/60000 [============================>.] - ETA: 0s - loss: 0.1902 - categorical_accuracy: 0.9423
59808/60000 [============================>.] - ETA: 0s - loss: 0.1901 - categorical_accuracy: 0.9423
59872/60000 [============================>.] - ETA: 0s - loss: 0.1899 - categorical_accuracy: 0.9424
59936/60000 [============================>.] - ETA: 0s - loss: 0.1898 - categorical_accuracy: 0.9424
60000/60000 [==============================] - 93s 2ms/step - loss: 0.1897 - categorical_accuracy: 0.9424 - val_loss: 0.0523 - val_categorical_accuracy: 0.9829

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 16s
  224/10000 [..............................] - ETA: 4s 
  416/10000 [>.............................] - ETA: 3s
  576/10000 [>.............................] - ETA: 3s
  768/10000 [=>............................] - ETA: 3s
  960/10000 [=>............................] - ETA: 3s
 1152/10000 [==>...........................] - ETA: 2s
 1344/10000 [===>..........................] - ETA: 2s
 1536/10000 [===>..........................] - ETA: 2s
 1728/10000 [====>.........................] - ETA: 2s
 1920/10000 [====>.........................] - ETA: 2s
 2112/10000 [=====>........................] - ETA: 2s
 2272/10000 [=====>........................] - ETA: 2s
 2432/10000 [======>.......................] - ETA: 2s
 2592/10000 [======>.......................] - ETA: 2s
 2752/10000 [=======>......................] - ETA: 2s
 2944/10000 [=======>......................] - ETA: 2s
 3136/10000 [========>.....................] - ETA: 2s
 3328/10000 [========>.....................] - ETA: 2s
 3520/10000 [=========>....................] - ETA: 2s
 3712/10000 [==========>...................] - ETA: 1s
 3904/10000 [==========>...................] - ETA: 1s
 4096/10000 [===========>..................] - ETA: 1s
 4288/10000 [===========>..................] - ETA: 1s
 4480/10000 [============>.................] - ETA: 1s
 4672/10000 [=============>................] - ETA: 1s
 4864/10000 [=============>................] - ETA: 1s
 5056/10000 [==============>...............] - ETA: 1s
 5248/10000 [==============>...............] - ETA: 1s
 5408/10000 [===============>..............] - ETA: 1s
 5568/10000 [===============>..............] - ETA: 1s
 5760/10000 [================>.............] - ETA: 1s
 5952/10000 [================>.............] - ETA: 1s
 6144/10000 [=================>............] - ETA: 1s
 6336/10000 [==================>...........] - ETA: 1s
 6528/10000 [==================>...........] - ETA: 1s
 6720/10000 [===================>..........] - ETA: 1s
 6880/10000 [===================>..........] - ETA: 0s
 7072/10000 [====================>.........] - ETA: 0s
 7264/10000 [====================>.........] - ETA: 0s
 7456/10000 [=====================>........] - ETA: 0s
 7648/10000 [=====================>........] - ETA: 0s
 7808/10000 [======================>.......] - ETA: 0s
 8000/10000 [=======================>......] - ETA: 0s
 8192/10000 [=======================>......] - ETA: 0s
 8384/10000 [========================>.....] - ETA: 0s
 8576/10000 [========================>.....] - ETA: 0s
 8768/10000 [=========================>....] - ETA: 0s
 8960/10000 [=========================>....] - ETA: 0s
 9152/10000 [==========================>...] - ETA: 0s
 9344/10000 [===========================>..] - ETA: 0s
 9536/10000 [===========================>..] - ETA: 0s
 9728/10000 [============================>.] - ETA: 0s
 9920/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 3s 305us/step
[[6.91386930e-08 4.35699761e-08 4.61326806e-07 ... 9.99997020e-01
  4.50642723e-09 2.18924265e-06]
 [1.09181847e-05 8.45534450e-05 9.99895453e-01 ... 1.09348875e-07
  1.64530366e-06 1.31175542e-08]
 [2.83839938e-07 9.99968052e-01 2.01811122e-06 ... 8.62374418e-06
  3.47673290e-06 1.23914026e-06]
 ...
 [1.06133804e-08 1.28811348e-06 6.06388451e-09 ... 1.21435846e-06
  2.37017730e-06 3.47460627e-05]
 [7.50761610e-05 8.77008802e-07 4.09343642e-07 ... 1.61491153e-06
  5.05211996e-03 3.12555967e-05]
 [5.03904266e-06 1.92612532e-07 4.86051931e-06 ... 8.41430126e-09
  7.61757406e-07 1.08109440e-07]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.052288835537922566, 'accuracy_test:': 0.9829000234603882}

  ('#### Save   #######################################################',) 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/charcnn/result'}

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   9c69c07..34e3603  master     -> origin/master
Updating 9c69c07..34e3603
Fast-forward
 error_list/20200513/list_log_json_20200513.md    |  276 ++--
 error_list/20200513/list_log_jupyter_20200513.md | 1664 +++++++++++-----------
 error_list/20200513/list_log_testall_20200513.md |  435 ++++++
 3 files changed, 1400 insertions(+), 975 deletions(-)
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
[master c2adda3] ml_store
 1 file changed, 1191 insertions(+)
To github.com:arita37/mlmodels_store.git
   34e3603..c2adda3  master -> master





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
{'loss': 0.5523540079593658, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-13 04:33:58.730108: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
[master 3f6525f] ml_store
 1 file changed, 233 insertions(+)
To github.com:arita37/mlmodels_store.git
   c2adda3..3f6525f  master -> master





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
[master 3c8d13c] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   3f6525f..3c8d13c  master -> master





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
 40%|████      | 2/5 [00:22<00:33, 11.16s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.7573870217116389, 'learning_rate': 0.05531900905781489, 'min_data_in_leaf': 13, 'num_leaves': 61} and reward: 0.3952
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8<\x83\xb5\x15R\x95X\r\x00\x00\x00learning_rateq\x02G?\xacR\xc5\xed\x80:\x07X\x10\x00\x00\x00min_data_in_leafq\x03K\rX\n\x00\x00\x00num_leavesq\x04K=u.' and reward: 0.3952
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8<\x83\xb5\x15R\x95X\r\x00\x00\x00learning_rateq\x02G?\xacR\xc5\xed\x80:\x07X\x10\x00\x00\x00min_data_in_leafq\x03K\rX\n\x00\x00\x00num_leavesq\x04K=u.' and reward: 0.3952
 60%|██████    | 3/5 [00:54<00:35, 17.59s/it] 60%|██████    | 3/5 [00:54<00:36, 18.31s/it]
Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.7557309752167448, 'learning_rate': 0.009714646003910667, 'min_data_in_leaf': 27, 'num_leaves': 58} and reward: 0.3868
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8.\xf2\xb9\xe4)\xd3X\r\x00\x00\x00learning_rateq\x02G?\x83\xe5E\xb7\x088.X\x10\x00\x00\x00min_data_in_leafq\x03K\x1bX\n\x00\x00\x00num_leavesq\x04K:u.' and reward: 0.3868
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8.\xf2\xb9\xe4)\xd3X\r\x00\x00\x00learning_rateq\x02G?\x83\xe5E\xb7\x088.X\x10\x00\x00\x00min_data_in_leafq\x03K\x1bX\n\x00\x00\x00num_leavesq\x04K:u.' and reward: 0.3868
Time for Gradient Boosting hyperparameter optimization: 85.99614143371582
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.7573870217116389, 'learning_rate': 0.05531900905781489, 'min_data_in_leaf': 13, 'num_leaves': 61}
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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_3_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.39
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.39
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.39
 40%|████      | 2/5 [00:54<01:21, 27.30s/it] 40%|████      | 2/5 [00:54<01:21, 27.30s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.4696709294059357, 'embedding_size_factor': 0.7655330832260723, 'layers.choice': 2, 'learning_rate': 0.0003976338172652067, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 0.007359529582685722} and reward: 0.3392
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xde\x0f\x16\xa8k\x8b\xfaX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe8\x7f?<\x8e\xc8\xecX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?:\x0f0=\xae\x17\xe2X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?~%\x06\xad\xf1\xfd\x8fu.' and reward: 0.3392
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xde\x0f\x16\xa8k\x8b\xfaX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe8\x7f?<\x8e\xc8\xecX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?:\x0f0=\xae\x17\xe2X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?~%\x06\xad\xf1\xfd\x8fu.' and reward: 0.3392
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 110.58554315567017
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_3_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.75s of the -80.54s of remaining time.
Ensemble size: 73
Ensemble weights: 
[0.60273973 0.         0.04109589 0.30136986 0.05479452]
	0.3982	 = Validation accuracy score
	1.32s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 201.9s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_3_tabularNN.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7fec648bd668>

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   3c8d13c..f850237  master     -> origin/master
Updating 3c8d13c..f850237
Fast-forward
 error_list/20200513/list_log_json_20200513.md      |  276 +--
 error_list/20200513/list_log_jupyter_20200513.md   | 1664 ++++++-------
 error_list/20200513/list_log_testall_20200513.md   |  169 ++
 ...-14_6672e19fe4cfa7df885e45d91d645534b8989485.py | 2476 ++++++++++++++++++++
 4 files changed, 3620 insertions(+), 965 deletions(-)
 create mode 100644 log_benchmark/log_benchmark_2020-05-13-04-14_6672e19fe4cfa7df885e45d91d645534b8989485.py
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
[master eed2afd] ml_store
 1 file changed, 207 insertions(+)
To github.com:arita37/mlmodels_store.git
   f850237..eed2afd  master -> master





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
[master a0710ab] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   eed2afd..a0710ab  master -> master





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
[master 1cc2cc6] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   a0710ab..1cc2cc6  master -> master





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
{'roc_auc_score': 0.9615384615384616}

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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7f9d1defea90> 

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
[master 7963527] ml_store
 1 file changed, 108 insertions(+)
To github.com:arita37/mlmodels_store.git
   1cc2cc6..7963527  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 0.10593645 -0.73728963  0.65032321  0.16466507 -1.53556118  0.77817418
   0.05031709  0.30981676  1.05132077  0.6065484 ]
 [ 0.87122579 -0.20975294 -0.45698786  0.93514778 -0.87353582  1.81252782
   0.92550121  0.14010988 -1.41914878  1.06898597]
 [ 0.96703727  0.38271517 -0.80618482 -0.28899734  0.90852604 -0.39181624
   1.62091229  0.68400133 -0.35340998 -0.25167421]
 [ 1.37661405 -0.60022533  0.72591685 -0.37951752 -0.62754626 -1.01480369
   0.96622086  0.4359862  -0.68748739  3.32107876]
 [ 1.16755486  0.0353601   0.7147896  -1.53879325  1.10863359 -0.44789518
  -1.75592564  0.61798553 -0.18417633  0.85270406]
 [ 0.6675918  -0.45252497 -0.60598132  1.16128569 -1.44620987  1.06996554
   1.92381543 -1.04553425  0.35528451  1.80358898]
 [ 0.78344054 -0.05118845  0.82458463 -0.72559712  0.9317172  -0.86776868
   3.03085711 -0.13597733 -0.79726979  0.65458015]
 [ 0.97139534  0.71304905  1.76041518  1.30620607  1.0576549  -0.60460297
   0.12837699  0.63658341  1.40925339  0.96653925]
 [ 0.76170668 -1.48515645  1.30253554 -0.59246129 -1.64162479 -2.30490794
  -1.34869645 -0.03181717  0.11248774 -0.36261209]
 [ 1.16777676 -0.66575452 -1.23312074 -1.67419581  1.01313574  0.82502982
  -0.12046457 -0.49821356 -0.31098498 -1.18231813]
 [ 1.21619061 -0.01900052  0.86089124 -0.22676019 -1.36419132 -1.56450785
   1.63169151  0.93125568  0.94980882 -0.88018906]
 [ 1.39198128 -0.19022103 -0.53722302 -0.44873803  0.70455707 -0.67244804
  -0.70134443 -0.55749472  0.93916874  0.15626385]
 [ 1.27991386 -0.87142207 -0.32403233 -0.86482994 -0.96853969  0.60874908
   0.50798434  0.5616381   1.51475038 -1.51107661]
 [ 1.34740825  0.73302323  0.83863475 -1.89881206 -0.54245992 -1.11711069
  -1.09715436 -0.50897228 -0.16648595 -1.03918232]
 [ 0.8786438   1.03703898 -0.47712421  0.67261975 -1.04948638  2.42887697
   0.52475049  1.00568668  0.35356722 -0.03599018]
 [ 2.07582971 -1.40232915 -0.47918492  0.45112294  1.03436581 -0.6949209
  -0.4189379   0.5154138  -1.11487105 -1.95210529]
 [ 1.13545112  0.8616231   0.04906169 -2.08639057 -1.1146902   0.36180164
  -0.80617821  0.42592018  0.0490804  -0.59608633]
 [ 0.85982375  0.17195713 -0.34898419  0.49056104 -1.15649503 -1.39528303
   0.61472628 -0.52235647 -0.3692559  -0.977773  ]
 [ 0.55853873 -0.51634791 -0.51814555  0.3511169   0.82550695 -0.06877046
  -0.9520621  -1.34776494  1.47073986 -1.4614036 ]
 [ 0.88883881  1.03368687 -0.04970258  0.80884436  0.81405135  1.78975468
   1.14690038  0.45128402 -1.68405999  0.46664327]
 [ 1.838294    0.50274088  0.12910158  1.55880554  1.32551412  0.1094027
   1.40754    -1.2197444   2.44936865  1.6169496 ]
 [ 0.89891716  0.55743945 -0.75806733  0.18103874  0.84146721  1.10717545
   0.69336623  1.44287693 -0.53968156 -0.8088472 ]
 [ 1.25704434 -1.82391985 -0.61240697  1.16707517 -0.62373281 -0.0396687
   0.81604368  0.8858258   0.18986165  0.39310924]
 [ 1.01177337  0.09574677  0.73140252  1.0334508  -1.42203164 -0.14627327
  -0.01745495 -0.85749682 -0.93418184  0.95449567]
 [ 1.01195228 -1.88141087  1.70018815  0.4972691  -0.91766462  0.2373327
  -1.09033833 -2.14444405 -0.36956243  0.60878366]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7fbfe1924f28>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7fbffbcab6d8> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 1.22867367  0.13437312 -0.18242041 -0.2683713  -1.73963799 -0.13167563
  -0.92687194  1.01855247  1.2305582  -0.49112514]
 [ 0.69211449 -0.06065249  2.05635552 -2.413503    1.17456965 -1.77756638
  -0.28173627 -0.77785883  1.11584111  1.76024923]
 [ 1.4468218   0.80745592  1.49810818  0.31223869 -0.68243019 -0.19332164
   0.28807817 -2.07680202  0.94750117 -0.30097615]
 [ 1.27991386 -0.87142207 -0.32403233 -0.86482994 -0.96853969  0.60874908
   0.50798434  0.5616381   1.51475038 -1.51107661]
 [ 0.96457205 -0.10679399  1.12232832  1.45142926  1.21828168 -0.61803685
   0.43816635 -2.03720123 -1.94258918 -0.9970198 ]
 [ 0.69174373  1.00978733 -1.21333813 -1.55694156 -1.20257258 -0.61244213
  -2.69836174 -0.13935181 -0.72853749  0.0722519 ]
 [ 0.85729649  0.9561217  -0.82609743 -0.70584051  1.13872896  1.19268607
   0.28267571 -0.23794194  1.15528789  0.6210827 ]
 [ 1.06040861  0.5103076   0.50172511 -0.91579185 -0.90731836 -0.40725204
  -0.17961229  0.98495167  1.07125243 -0.59334375]
 [ 0.87874071 -0.01923163  0.31965694  0.15001628 -1.46662161  0.46353432
  -0.89868319  0.39788042 -0.99601089  0.3181542 ]
 [ 1.07258847 -0.58652394 -1.34267579 -1.23685338  1.24328724  0.87583893
  -0.3264995   0.62336218 -0.43495668  1.11438298]
 [ 1.1437713   0.7278135   0.35249436  0.51507361  1.17718111 -2.78253447
  -1.94332341  0.58464661  0.32427424 -0.23643695]
 [ 0.77528533  1.47016034  1.03298378 -0.87000822  0.78655651  0.36919047
  -0.14319575  0.85328219 -0.13971173 -0.22241403]
 [ 0.92686981  0.39233491 -0.4234783   0.44838065 -1.09230828  1.1253235
  -0.94843966  0.10405339  0.52800342  1.00796648]
 [ 0.94781411 -1.13379204  0.64098587 -0.1905483  -1.23912256  0.23333913
  -0.3169012   0.43499832  0.9104236   1.21987438]
 [ 1.64661853 -1.52568032 -0.6069984   0.79502609  1.08480038 -0.37443832
   0.42952614  0.1340482   1.20205486  0.10622272]
 [ 1.25704434 -1.82391985 -0.61240697  1.16707517 -0.62373281 -0.0396687
   0.81604368  0.8858258   0.18986165  0.39310924]
 [ 0.8786438   1.03703898 -0.47712421  0.67261975 -1.04948638  2.42887697
   0.52475049  1.00568668  0.35356722 -0.03599018]
 [ 0.6675918  -0.45252497 -0.60598132  1.16128569 -1.44620987  1.06996554
   1.92381543 -1.04553425  0.35528451  1.80358898]
 [ 0.98379959 -0.40724002  0.93272141  0.16056499 -1.278618   -0.12014998
   0.19975956  0.38560229  0.71829074 -0.5301198 ]
 [ 0.5630779  -1.17598267 -0.17418034  1.01012718  1.06796368  0.92001793
  -0.16819884 -0.19505734  0.80539342  0.4611641 ]
 [ 0.89551051  0.92061512  0.79452824 -0.03536792  0.8780991   2.11060505
  -1.02188594 -1.30653407  0.07638048 -1.87316098]
 [ 0.79032389  1.61336137 -2.09424782 -0.37480469  0.91588404 -0.74996962
   0.31027229  2.0546241   0.05340954 -0.22876583]
 [ 1.66752297  1.22372221 -0.4599301  -0.0593679  -0.493857    1.4489894
  -1.18110317 -0.47758085  0.02599999 -0.79079995]
 [ 0.72297801  0.18553562  0.91549927  0.39442803 -0.84983074  0.72552256
  -0.15050433  1.49588477  0.67545381 -0.43820027]
 [ 1.17867274 -0.59980453 -0.6946936   1.12341216  1.17899425  0.30526704
   0.01335268  1.3887794  -0.66134424  0.6218035 ]]
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
[[ 1.14377130e+00  7.27813500e-01  3.52494364e-01  5.15073614e-01
   1.17718111e+00 -2.78253447e+00 -1.94332341e+00  5.84646610e-01
   3.24274243e-01 -2.36436952e-01]
 [ 8.98917161e-01  5.57439453e-01 -7.58067329e-01  1.81038744e-01
   8.41467206e-01  1.10717545e+00  6.93366226e-01  1.44287693e+00
  -5.39681562e-01 -8.08847196e-01]
 [ 3.45715997e-01 -4.13029310e-01 -4.68673816e-01  1.83471763e+00
   7.71514409e-01  5.64382855e-01  2.18628366e-02  2.13782807e+00
  -7.85533997e-01  8.53281222e-01]
 [ 1.01195228e+00 -1.88141087e+00  1.70018815e+00  4.97269099e-01
  -9.17664624e-01  2.37332699e-01 -1.09033833e+00 -2.14444405e+00
  -3.69562425e-01  6.08783659e-01]
 [ 4.73307772e-01 -9.73267585e-01 -2.28140691e-01  1.75167729e-01
  -1.01366961e+00 -5.34836927e-02  3.93787731e-01 -1.83061987e-01
  -2.21028902e-01  5.80330113e-01]
 [ 9.83799588e-01 -4.07240024e-01  9.32721414e-01  1.60564992e-01
  -1.27861800e+00 -1.20149976e-01  1.99759555e-01  3.85602292e-01
   7.18290736e-01 -5.30119800e-01]
 [ 1.12641981e+00 -6.29441604e-01  1.10100020e+00 -1.11343610e+00
   9.44595066e-01 -6.74100249e-02 -1.83400197e-01  1.16143998e+00
  -2.75293863e-02  7.80027135e-01]
 [ 8.78643802e-01  1.03703898e+00 -4.77124206e-01  6.72619748e-01
  -1.04948638e+00  2.42887697e+00  5.24750492e-01  1.00568668e+00
   3.53567216e-01 -3.59901817e-02]
 [ 8.48069266e-01  4.51946037e-01  6.30195671e-01 -1.57915629e+00
   8.27987371e-01 -8.28627979e-01 -1.05344713e-01  5.28879746e-01
  -2.23708651e+00 -4.14846901e-01]
 [ 9.36211246e-01  2.04377395e-01 -1.49419377e+00  6.12232523e-01
  -9.84377246e-01  7.44884536e-01  4.94341651e-01 -3.62812886e-02
  -8.32395348e-01 -4.46699203e-01]
 [ 8.61462558e-01  7.43205537e-02 -1.34501002e+00 -1.99560718e-01
  -1.47533915e+00 -6.54603169e-01 -3.14563862e-01  3.18014296e-01
  -8.90271552e-01 -1.29525789e+00]
 [ 8.59823751e-01  1.71957132e-01 -3.48984191e-01  4.90561044e-01
  -1.15649503e+00 -1.39528303e+00  6.14726276e-01 -5.22356465e-01
  -3.69255902e-01 -9.77773002e-01]
 [ 8.57296491e-01  9.56121704e-01 -8.26097432e-01 -7.05840507e-01
   1.13872896e+00  1.19268607e+00  2.82675712e-01 -2.37941936e-01
   1.15528789e+00  6.21082701e-01]
 [ 1.58463774e+00  5.71209961e-02 -1.77183179e-02 -7.99547491e-01
   1.32970299e+00 -2.91594596e-01 -1.10771250e+00 -2.58982853e-01
   1.89293198e-01 -1.71939447e+00]
 [ 1.98519313e+00  6.74711526e-01 -1.39662042e+00  6.18539131e-01
   1.22382712e+00 -4.43171931e-01 -1.89148284e-03  1.81053491e+00
  -1.30572692e+00 -8.61316361e-01]
 [ 7.22978007e-01  1.85535621e-01  9.15499268e-01  3.94428030e-01
  -8.49830738e-01  7.25522558e-01 -1.50504326e-01  1.49588477e+00
   6.75453809e-01 -4.38200267e-01]
 [ 8.72267394e-01 -2.51630386e+00 -7.75070287e-01 -5.95667881e-01
   1.02600767e+00 -3.09121319e-01  1.74643509e+00  5.10937774e-01
   1.71066184e+00  1.41640538e-01]
 [ 1.16755486e+00  3.53600971e-02  7.14789597e-01 -1.53879325e+00
   1.10863359e+00 -4.47895185e-01 -1.75592564e+00  6.17985534e-01
  -1.84176326e-01  8.52704062e-01]
 [ 1.39198128e+00 -1.90221025e-01 -5.37223024e-01 -4.48738033e-01
   7.04557071e-01 -6.72448039e-01 -7.01344426e-01 -5.57494722e-01
   9.39168744e-01  1.56263850e-01]
 [ 1.36586461e+00  3.95860270e+00  5.48129585e-01  6.48643644e-01
   8.49176066e-01  1.07343294e-01  1.38631426e+00 -1.39881282e+00
   8.17678188e-02 -1.63744959e+00]
 [ 8.88838813e-01  1.03368687e+00 -4.97025792e-02  8.08844360e-01
   8.14051347e-01  1.78975468e+00  1.14690038e+00  4.51284016e-01
  -1.68405999e+00  4.66643267e-01]
 [ 8.95623122e-01 -2.29820588e+00 -1.95225583e-02  1.45652739e+00
  -1.85064099e+00  3.16637236e-01  1.11337266e-01 -2.66412594e+00
  -4.26428618e-01 -8.39988915e-01]
 [ 4.67397905e-01 -2.37875265e-01 -1.54491194e-01 -7.55662765e-01
  -5.47062239e-01  1.85143789e+00 -1.46405357e+00  2.09096677e-01
   1.55501599e+00 -9.24323185e-02]
 [ 7.61706684e-01 -1.48515645e+00  1.30253554e+00 -5.92461285e-01
  -1.64162479e+00 -2.30490794e+00 -1.34869645e+00 -3.18171727e-02
   1.12487742e-01 -3.62612088e-01]
 [ 1.77547698e+00 -2.03394449e-01 -1.98837863e-01  2.42669441e-01
   9.64350564e-01  2.01830179e-01 -5.45774168e-01  6.61020288e-01
   1.79215821e+00 -7.00398505e-01]]
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
[master b8321c0] ml_store
 1 file changed, 271 insertions(+)
To github.com:arita37/mlmodels_store.git
   7963527..b8321c0  master -> master





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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @139679189958608
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @139679189958384
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @139679189957152
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @139679189956704
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @139679189956200
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @139679189955864

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
grad_step = 000000, loss = 1.370522
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 1.164485
grad_step = 000002, loss = 1.007432
grad_step = 000003, loss = 0.829131
grad_step = 000004, loss = 0.619551
grad_step = 000005, loss = 0.387403
grad_step = 000006, loss = 0.187852
grad_step = 000007, loss = 0.113037
grad_step = 000008, loss = 0.186860
grad_step = 000009, loss = 0.224149
grad_step = 000010, loss = 0.148616
grad_step = 000011, loss = 0.057850
grad_step = 000012, loss = 0.012328
grad_step = 000013, loss = 0.013159
grad_step = 000014, loss = 0.036186
grad_step = 000015, loss = 0.059717
grad_step = 000016, loss = 0.073222
grad_step = 000017, loss = 0.074715
grad_step = 000018, loss = 0.066503
grad_step = 000019, loss = 0.052883
grad_step = 000020, loss = 0.040350
grad_step = 000021, loss = 0.033131
grad_step = 000022, loss = 0.030754
grad_step = 000023, loss = 0.028759
grad_step = 000024, loss = 0.023927
grad_step = 000025, loss = 0.016836
grad_step = 000026, loss = 0.010417
grad_step = 000027, loss = 0.007045
grad_step = 000028, loss = 0.006483
grad_step = 000029, loss = 0.008014
grad_step = 000030, loss = 0.010510
grad_step = 000031, loss = 0.012704
grad_step = 000032, loss = 0.013747
grad_step = 000033, loss = 0.013523
grad_step = 000034, loss = 0.012503
grad_step = 000035, loss = 0.011193
grad_step = 000036, loss = 0.009881
grad_step = 000037, loss = 0.008564
grad_step = 000038, loss = 0.007247
grad_step = 000039, loss = 0.006047
grad_step = 000040, loss = 0.005100
grad_step = 000041, loss = 0.004495
grad_step = 000042, loss = 0.004309
grad_step = 000043, loss = 0.004562
grad_step = 000044, loss = 0.005111
grad_step = 000045, loss = 0.005676
grad_step = 000046, loss = 0.005985
grad_step = 000047, loss = 0.005930
grad_step = 000048, loss = 0.005610
grad_step = 000049, loss = 0.005219
grad_step = 000050, loss = 0.004896
grad_step = 000051, loss = 0.004642
grad_step = 000052, loss = 0.004385
grad_step = 000053, loss = 0.004099
grad_step = 000054, loss = 0.003843
grad_step = 000055, loss = 0.003714
grad_step = 000056, loss = 0.003722
grad_step = 000057, loss = 0.003819
grad_step = 000058, loss = 0.003924
grad_step = 000059, loss = 0.003997
grad_step = 000060, loss = 0.003983
grad_step = 000061, loss = 0.003939
grad_step = 000062, loss = 0.003866
grad_step = 000063, loss = 0.003772
grad_step = 000064, loss = 0.003670
grad_step = 000065, loss = 0.003569
grad_step = 000066, loss = 0.003481
grad_step = 000067, loss = 0.003420
grad_step = 000068, loss = 0.003396
grad_step = 000069, loss = 0.003402
grad_step = 000070, loss = 0.003414
grad_step = 000071, loss = 0.003413
grad_step = 000072, loss = 0.003395
grad_step = 000073, loss = 0.003369
grad_step = 000074, loss = 0.003341
grad_step = 000075, loss = 0.003311
grad_step = 000076, loss = 0.003273
grad_step = 000077, loss = 0.003231
grad_step = 000078, loss = 0.003191
grad_step = 000079, loss = 0.003159
grad_step = 000080, loss = 0.003136
grad_step = 000081, loss = 0.003119
grad_step = 000082, loss = 0.003105
grad_step = 000083, loss = 0.003088
grad_step = 000084, loss = 0.003068
grad_step = 000085, loss = 0.003047
grad_step = 000086, loss = 0.003023
grad_step = 000087, loss = 0.002998
grad_step = 000088, loss = 0.002970
grad_step = 000089, loss = 0.002939
grad_step = 000090, loss = 0.002909
grad_step = 000091, loss = 0.002880
grad_step = 000092, loss = 0.002853
grad_step = 000093, loss = 0.002827
grad_step = 000094, loss = 0.002801
grad_step = 000095, loss = 0.002774
grad_step = 000096, loss = 0.002745
grad_step = 000097, loss = 0.002715
grad_step = 000098, loss = 0.002683
grad_step = 000099, loss = 0.002649
grad_step = 000100, loss = 0.002615
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002580
grad_step = 000102, loss = 0.002544
grad_step = 000103, loss = 0.002509
grad_step = 000104, loss = 0.002473
grad_step = 000105, loss = 0.002437
grad_step = 000106, loss = 0.002400
grad_step = 000107, loss = 0.002363
grad_step = 000108, loss = 0.002324
grad_step = 000109, loss = 0.002286
grad_step = 000110, loss = 0.002248
grad_step = 000111, loss = 0.002210
grad_step = 000112, loss = 0.002173
grad_step = 000113, loss = 0.002136
grad_step = 000114, loss = 0.002100
grad_step = 000115, loss = 0.002064
grad_step = 000116, loss = 0.002030
grad_step = 000117, loss = 0.001998
grad_step = 000118, loss = 0.001967
grad_step = 000119, loss = 0.001938
grad_step = 000120, loss = 0.001911
grad_step = 000121, loss = 0.001887
grad_step = 000122, loss = 0.001865
grad_step = 000123, loss = 0.001845
grad_step = 000124, loss = 0.001828
grad_step = 000125, loss = 0.001817
grad_step = 000126, loss = 0.001813
grad_step = 000127, loss = 0.001792
grad_step = 000128, loss = 0.001773
grad_step = 000129, loss = 0.001771
grad_step = 000130, loss = 0.001746
grad_step = 000131, loss = 0.001735
grad_step = 000132, loss = 0.001719
grad_step = 000133, loss = 0.001697
grad_step = 000134, loss = 0.001685
grad_step = 000135, loss = 0.001661
grad_step = 000136, loss = 0.001646
grad_step = 000137, loss = 0.001626
grad_step = 000138, loss = 0.001607
grad_step = 000139, loss = 0.001590
grad_step = 000140, loss = 0.001569
grad_step = 000141, loss = 0.001551
grad_step = 000142, loss = 0.001531
grad_step = 000143, loss = 0.001512
grad_step = 000144, loss = 0.001492
grad_step = 000145, loss = 0.001473
grad_step = 000146, loss = 0.001452
grad_step = 000147, loss = 0.001432
grad_step = 000148, loss = 0.001412
grad_step = 000149, loss = 0.001390
grad_step = 000150, loss = 0.001371
grad_step = 000151, loss = 0.001350
grad_step = 000152, loss = 0.001329
grad_step = 000153, loss = 0.001308
grad_step = 000154, loss = 0.001287
grad_step = 000155, loss = 0.001266
grad_step = 000156, loss = 0.001245
grad_step = 000157, loss = 0.001224
grad_step = 000158, loss = 0.001203
grad_step = 000159, loss = 0.001183
grad_step = 000160, loss = 0.001164
grad_step = 000161, loss = 0.001146
grad_step = 000162, loss = 0.001127
grad_step = 000163, loss = 0.001105
grad_step = 000164, loss = 0.001083
grad_step = 000165, loss = 0.001062
grad_step = 000166, loss = 0.001045
grad_step = 000167, loss = 0.001029
grad_step = 000168, loss = 0.001013
grad_step = 000169, loss = 0.000998
grad_step = 000170, loss = 0.000981
grad_step = 000171, loss = 0.000958
grad_step = 000172, loss = 0.000936
grad_step = 000173, loss = 0.000916
grad_step = 000174, loss = 0.000899
grad_step = 000175, loss = 0.000885
grad_step = 000176, loss = 0.000872
grad_step = 000177, loss = 0.000859
grad_step = 000178, loss = 0.000848
grad_step = 000179, loss = 0.000836
grad_step = 000180, loss = 0.000813
grad_step = 000181, loss = 0.000788
grad_step = 000182, loss = 0.000770
grad_step = 000183, loss = 0.000760
grad_step = 000184, loss = 0.000752
grad_step = 000185, loss = 0.000740
grad_step = 000186, loss = 0.000725
grad_step = 000187, loss = 0.000706
grad_step = 000188, loss = 0.000691
grad_step = 000189, loss = 0.000681
grad_step = 000190, loss = 0.000675
grad_step = 000191, loss = 0.000671
grad_step = 000192, loss = 0.000669
grad_step = 000193, loss = 0.000674
grad_step = 000194, loss = 0.000676
grad_step = 000195, loss = 0.000672
grad_step = 000196, loss = 0.000652
grad_step = 000197, loss = 0.000628
grad_step = 000198, loss = 0.000610
grad_step = 000199, loss = 0.000614
grad_step = 000200, loss = 0.000623
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.000618
grad_step = 000202, loss = 0.000606
grad_step = 000203, loss = 0.000591
grad_step = 000204, loss = 0.000583
grad_step = 000205, loss = 0.000586
grad_step = 000206, loss = 0.000590
grad_step = 000207, loss = 0.000589
grad_step = 000208, loss = 0.000585
grad_step = 000209, loss = 0.000578
grad_step = 000210, loss = 0.000570
grad_step = 000211, loss = 0.000563
grad_step = 000212, loss = 0.000557
grad_step = 000213, loss = 0.000550
grad_step = 000214, loss = 0.000546
grad_step = 000215, loss = 0.000544
grad_step = 000216, loss = 0.000541
grad_step = 000217, loss = 0.000541
grad_step = 000218, loss = 0.000545
grad_step = 000219, loss = 0.000553
grad_step = 000220, loss = 0.000569
grad_step = 000221, loss = 0.000587
grad_step = 000222, loss = 0.000598
grad_step = 000223, loss = 0.000582
grad_step = 000224, loss = 0.000542
grad_step = 000225, loss = 0.000515
grad_step = 000226, loss = 0.000521
grad_step = 000227, loss = 0.000542
grad_step = 000228, loss = 0.000543
grad_step = 000229, loss = 0.000523
grad_step = 000230, loss = 0.000502
grad_step = 000231, loss = 0.000504
grad_step = 000232, loss = 0.000516
grad_step = 000233, loss = 0.000518
grad_step = 000234, loss = 0.000506
grad_step = 000235, loss = 0.000491
grad_step = 000236, loss = 0.000489
grad_step = 000237, loss = 0.000495
grad_step = 000238, loss = 0.000499
grad_step = 000239, loss = 0.000494
grad_step = 000240, loss = 0.000484
grad_step = 000241, loss = 0.000477
grad_step = 000242, loss = 0.000476
grad_step = 000243, loss = 0.000479
grad_step = 000244, loss = 0.000481
grad_step = 000245, loss = 0.000482
grad_step = 000246, loss = 0.000478
grad_step = 000247, loss = 0.000473
grad_step = 000248, loss = 0.000467
grad_step = 000249, loss = 0.000461
grad_step = 000250, loss = 0.000458
grad_step = 000251, loss = 0.000456
grad_step = 000252, loss = 0.000454
grad_step = 000253, loss = 0.000454
grad_step = 000254, loss = 0.000456
grad_step = 000255, loss = 0.000460
grad_step = 000256, loss = 0.000469
grad_step = 000257, loss = 0.000487
grad_step = 000258, loss = 0.000510
grad_step = 000259, loss = 0.000535
grad_step = 000260, loss = 0.000533
grad_step = 000261, loss = 0.000494
grad_step = 000262, loss = 0.000446
grad_step = 000263, loss = 0.000438
grad_step = 000264, loss = 0.000464
grad_step = 000265, loss = 0.000484
grad_step = 000266, loss = 0.000468
grad_step = 000267, loss = 0.000437
grad_step = 000268, loss = 0.000426
grad_step = 000269, loss = 0.000441
grad_step = 000270, loss = 0.000457
grad_step = 000271, loss = 0.000451
grad_step = 000272, loss = 0.000430
grad_step = 000273, loss = 0.000417
grad_step = 000274, loss = 0.000423
grad_step = 000275, loss = 0.000434
grad_step = 000276, loss = 0.000433
grad_step = 000277, loss = 0.000422
grad_step = 000278, loss = 0.000410
grad_step = 000279, loss = 0.000408
grad_step = 000280, loss = 0.000415
grad_step = 000281, loss = 0.000420
grad_step = 000282, loss = 0.000419
grad_step = 000283, loss = 0.000412
grad_step = 000284, loss = 0.000404
grad_step = 000285, loss = 0.000398
grad_step = 000286, loss = 0.000396
grad_step = 000287, loss = 0.000397
grad_step = 000288, loss = 0.000398
grad_step = 000289, loss = 0.000399
grad_step = 000290, loss = 0.000399
grad_step = 000291, loss = 0.000398
grad_step = 000292, loss = 0.000396
grad_step = 000293, loss = 0.000393
grad_step = 000294, loss = 0.000390
grad_step = 000295, loss = 0.000387
grad_step = 000296, loss = 0.000385
grad_step = 000297, loss = 0.000383
grad_step = 000298, loss = 0.000382
grad_step = 000299, loss = 0.000383
grad_step = 000300, loss = 0.000384
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000390
grad_step = 000302, loss = 0.000400
grad_step = 000303, loss = 0.000420
grad_step = 000304, loss = 0.000449
grad_step = 000305, loss = 0.000473
grad_step = 000306, loss = 0.000477
grad_step = 000307, loss = 0.000433
grad_step = 000308, loss = 0.000384
grad_step = 000309, loss = 0.000365
grad_step = 000310, loss = 0.000383
grad_step = 000311, loss = 0.000408
grad_step = 000312, loss = 0.000404
grad_step = 000313, loss = 0.000380
grad_step = 000314, loss = 0.000360
grad_step = 000315, loss = 0.000363
grad_step = 000316, loss = 0.000378
grad_step = 000317, loss = 0.000385
grad_step = 000318, loss = 0.000375
grad_step = 000319, loss = 0.000358
grad_step = 000320, loss = 0.000351
grad_step = 000321, loss = 0.000356
grad_step = 000322, loss = 0.000364
grad_step = 000323, loss = 0.000367
grad_step = 000324, loss = 0.000361
grad_step = 000325, loss = 0.000351
grad_step = 000326, loss = 0.000344
grad_step = 000327, loss = 0.000343
grad_step = 000328, loss = 0.000346
grad_step = 000329, loss = 0.000350
grad_step = 000330, loss = 0.000352
grad_step = 000331, loss = 0.000351
grad_step = 000332, loss = 0.000346
grad_step = 000333, loss = 0.000341
grad_step = 000334, loss = 0.000336
grad_step = 000335, loss = 0.000333
grad_step = 000336, loss = 0.000333
grad_step = 000337, loss = 0.000334
grad_step = 000338, loss = 0.000335
grad_step = 000339, loss = 0.000336
grad_step = 000340, loss = 0.000337
grad_step = 000341, loss = 0.000337
grad_step = 000342, loss = 0.000336
grad_step = 000343, loss = 0.000334
grad_step = 000344, loss = 0.000332
grad_step = 000345, loss = 0.000330
grad_step = 000346, loss = 0.000328
grad_step = 000347, loss = 0.000325
grad_step = 000348, loss = 0.000323
grad_step = 000349, loss = 0.000321
grad_step = 000350, loss = 0.000319
grad_step = 000351, loss = 0.000317
grad_step = 000352, loss = 0.000316
grad_step = 000353, loss = 0.000314
grad_step = 000354, loss = 0.000313
grad_step = 000355, loss = 0.000313
grad_step = 000356, loss = 0.000313
grad_step = 000357, loss = 0.000317
grad_step = 000358, loss = 0.000327
grad_step = 000359, loss = 0.000349
grad_step = 000360, loss = 0.000392
grad_step = 000361, loss = 0.000472
grad_step = 000362, loss = 0.000514
grad_step = 000363, loss = 0.000496
grad_step = 000364, loss = 0.000360
grad_step = 000365, loss = 0.000308
grad_step = 000366, loss = 0.000374
grad_step = 000367, loss = 0.000403
grad_step = 000368, loss = 0.000341
grad_step = 000369, loss = 0.000305
grad_step = 000370, loss = 0.000351
grad_step = 000371, loss = 0.000365
grad_step = 000372, loss = 0.000314
grad_step = 000373, loss = 0.000305
grad_step = 000374, loss = 0.000339
grad_step = 000375, loss = 0.000332
grad_step = 000376, loss = 0.000300
grad_step = 000377, loss = 0.000302
grad_step = 000378, loss = 0.000322
grad_step = 000379, loss = 0.000315
grad_step = 000380, loss = 0.000293
grad_step = 000381, loss = 0.000294
grad_step = 000382, loss = 0.000309
grad_step = 000383, loss = 0.000306
grad_step = 000384, loss = 0.000291
grad_step = 000385, loss = 0.000285
grad_step = 000386, loss = 0.000293
grad_step = 000387, loss = 0.000299
grad_step = 000388, loss = 0.000293
grad_step = 000389, loss = 0.000284
grad_step = 000390, loss = 0.000280
grad_step = 000391, loss = 0.000283
grad_step = 000392, loss = 0.000288
grad_step = 000393, loss = 0.000286
grad_step = 000394, loss = 0.000280
grad_step = 000395, loss = 0.000275
grad_step = 000396, loss = 0.000274
grad_step = 000397, loss = 0.000278
grad_step = 000398, loss = 0.000278
grad_step = 000399, loss = 0.000276
grad_step = 000400, loss = 0.000272
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000270
grad_step = 000402, loss = 0.000269
grad_step = 000403, loss = 0.000269
grad_step = 000404, loss = 0.000269
grad_step = 000405, loss = 0.000270
grad_step = 000406, loss = 0.000268
grad_step = 000407, loss = 0.000266
grad_step = 000408, loss = 0.000264
grad_step = 000409, loss = 0.000263
grad_step = 000410, loss = 0.000262
grad_step = 000411, loss = 0.000261
grad_step = 000412, loss = 0.000260
grad_step = 000413, loss = 0.000260
grad_step = 000414, loss = 0.000261
grad_step = 000415, loss = 0.000261
grad_step = 000416, loss = 0.000261
grad_step = 000417, loss = 0.000261
grad_step = 000418, loss = 0.000261
grad_step = 000419, loss = 0.000261
grad_step = 000420, loss = 0.000261
grad_step = 000421, loss = 0.000259
grad_step = 000422, loss = 0.000256
grad_step = 000423, loss = 0.000254
grad_step = 000424, loss = 0.000253
grad_step = 000425, loss = 0.000252
grad_step = 000426, loss = 0.000251
grad_step = 000427, loss = 0.000249
grad_step = 000428, loss = 0.000248
grad_step = 000429, loss = 0.000248
grad_step = 000430, loss = 0.000248
grad_step = 000431, loss = 0.000249
grad_step = 000432, loss = 0.000251
grad_step = 000433, loss = 0.000255
grad_step = 000434, loss = 0.000260
grad_step = 000435, loss = 0.000268
grad_step = 000436, loss = 0.000276
grad_step = 000437, loss = 0.000285
grad_step = 000438, loss = 0.000287
grad_step = 000439, loss = 0.000283
grad_step = 000440, loss = 0.000269
grad_step = 000441, loss = 0.000255
grad_step = 000442, loss = 0.000246
grad_step = 000443, loss = 0.000243
grad_step = 000444, loss = 0.000239
grad_step = 000445, loss = 0.000240
grad_step = 000446, loss = 0.000246
grad_step = 000447, loss = 0.000254
grad_step = 000448, loss = 0.000258
grad_step = 000449, loss = 0.000258
grad_step = 000450, loss = 0.000256
grad_step = 000451, loss = 0.000252
grad_step = 000452, loss = 0.000245
grad_step = 000453, loss = 0.000236
grad_step = 000454, loss = 0.000231
grad_step = 000455, loss = 0.000228
grad_step = 000456, loss = 0.000227
grad_step = 000457, loss = 0.000227
grad_step = 000458, loss = 0.000229
grad_step = 000459, loss = 0.000233
grad_step = 000460, loss = 0.000236
grad_step = 000461, loss = 0.000240
grad_step = 000462, loss = 0.000245
grad_step = 000463, loss = 0.000251
grad_step = 000464, loss = 0.000253
grad_step = 000465, loss = 0.000256
grad_step = 000466, loss = 0.000254
grad_step = 000467, loss = 0.000249
grad_step = 000468, loss = 0.000240
grad_step = 000469, loss = 0.000231
grad_step = 000470, loss = 0.000222
grad_step = 000471, loss = 0.000217
grad_step = 000472, loss = 0.000216
grad_step = 000473, loss = 0.000217
grad_step = 000474, loss = 0.000220
grad_step = 000475, loss = 0.000224
grad_step = 000476, loss = 0.000230
grad_step = 000477, loss = 0.000235
grad_step = 000478, loss = 0.000242
grad_step = 000479, loss = 0.000249
grad_step = 000480, loss = 0.000257
grad_step = 000481, loss = 0.000260
grad_step = 000482, loss = 0.000260
grad_step = 000483, loss = 0.000248
grad_step = 000484, loss = 0.000232
grad_step = 000485, loss = 0.000216
grad_step = 000486, loss = 0.000209
grad_step = 000487, loss = 0.000210
grad_step = 000488, loss = 0.000217
grad_step = 000489, loss = 0.000224
grad_step = 000490, loss = 0.000229
grad_step = 000491, loss = 0.000232
grad_step = 000492, loss = 0.000232
grad_step = 000493, loss = 0.000229
grad_step = 000494, loss = 0.000223
grad_step = 000495, loss = 0.000216
grad_step = 000496, loss = 0.000209
grad_step = 000497, loss = 0.000206
grad_step = 000498, loss = 0.000204
grad_step = 000499, loss = 0.000203
grad_step = 000500, loss = 0.000204
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000205
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
[[0.8391049  0.8683721  0.9354136  0.96073145 1.0067403 ]
 [0.84070295 0.9203332  0.9470961  1.0235584  0.9735024 ]
 [0.9062614  0.9316501  1.0120666  0.9950213  0.9460311 ]
 [0.9256667  0.9898792  0.99064744 0.9556607  0.9207146 ]
 [0.996308   0.9915278  0.95202875 0.9094171  0.85695815]
 [0.99181795 0.9494798  0.9136671  0.86345786 0.8564298 ]
 [0.94287086 0.908536   0.8548856  0.84718937 0.81991935]
 [0.8899098  0.8309107  0.8495815  0.8155796  0.8416626 ]
 [0.83267623 0.8325379  0.81647575 0.8449271  0.8575051 ]
 [0.827035   0.8105985  0.8394998  0.85528183 0.8092929 ]
 [0.7962716  0.8184432  0.85406756 0.8299625  0.9131402 ]
 [0.8071555  0.8224133  0.8215678  0.92838234 0.95254433]
 [0.8303585  0.86230123 0.9344274  0.9650563  1.004011  ]
 [0.8449212  0.9240528  0.94591    1.0163817  0.9632871 ]
 [0.9204018  0.93977165 1.0087765  0.9795166  0.92917645]
 [0.93416387 0.99434054 0.98049295 0.93829435 0.8989346 ]
 [1.0002885  0.98582816 0.9358161  0.88515997 0.83736825]
 [0.9771076  0.9247191  0.89091873 0.84065026 0.8437189 ]
 [0.9257996  0.89333415 0.83736587 0.8390352  0.81876034]
 [0.8941468  0.83182716 0.8475346  0.8147871  0.8529206 ]
 [0.8444485  0.8408323  0.8214388  0.8472011  0.8736777 ]
 [0.84409255 0.8218134  0.8484503  0.8649422  0.82138544]
 [0.8106446  0.8251533  0.8626548  0.8377483  0.91364396]
 [0.8168375  0.83492607 0.8279419  0.9269984  0.95391095]
 [0.8460912  0.8731114  0.9355554  0.9620023  1.0178318 ]
 [0.85055363 0.928292   0.95208716 1.0303829  0.9882678 ]
 [0.9182049  0.9411803  1.019004   1.004705   0.9572872 ]
 [0.94037694 0.9999033  1.0011845  0.9627587  0.93129283]
 [1.0047188  1.0041094  0.96483886 0.91520536 0.86418295]
 [1.0018575  0.95988953 0.9227946  0.8701643  0.8644253 ]
 [0.9495523  0.91484886 0.8626659  0.8512658  0.828789  ]]

  #### Plot     ############################################### 
Saved image to ztest/model_tch/nbeats//n_beats_test.png.

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Warning: Permanently added the RSA host key for IP address '140.82.118.4' to the list of known hosts.
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
[master bab7ddd] ml_store
 1 file changed, 1123 insertions(+)
To github.com:arita37/mlmodels_store.git
   b8321c0..bab7ddd  master -> master





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
[master 74a8de2] ml_store
 1 file changed, 37 insertions(+)
To github.com:arita37/mlmodels_store.git
   bab7ddd..74a8de2  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//matchzoo_models.py 

  #### Loading params   ############################################## 

  {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'dataset_pars': {'data_pack': '', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'resample': True, 'sort': False, 'callbacks': 'PADDING'}, 'dataloader': '', 'dataloader_pars': {'device': 'cpu', 'dataset': 'None', 'stage': 'train', 'callback': 'PADDING'}, 'preprocess': {'train': {'transform': True, 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'stage': 'train', 'resample': True, 'sort': False, 'dataloader_callback': 'PADDING'}, 'test': {'transform': True, 'batch_size': 20, 'stage': 'dev', 'dataloader_callback': 'PADDING'}}} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Loading dataset   ############################################# 

  #### Model init   ################################################## 
  0%|          | 0/231508 [00:00<?, ?B/s]100%|██████████| 231508/231508 [00:00<00:00, 22200716.32B/s]
  0%|          | 0/433 [00:00<?, ?B/s]100%|██████████| 433/433 [00:00<00:00, 336258.77B/s]
  0%|          | 0/440473133 [00:00<?, ?B/s]  1%|          | 4517888/440473133 [00:00<00:09, 45177845.96B/s]  2%|▏         | 10938368/440473133 [00:00<00:08, 49585471.54B/s]  4%|▍         | 17395712/440473133 [00:00<00:07, 53295198.54B/s]  5%|▌         | 23707648/440473133 [00:00<00:07, 55904707.02B/s]  7%|▋         | 30071808/440473133 [00:00<00:07, 58019090.99B/s]  8%|▊         | 36136960/440473133 [00:00<00:06, 58783711.09B/s] 10%|▉         | 42507264/440473133 [00:00<00:06, 60177370.37B/s] 11%|█         | 48994304/440473133 [00:00<00:06, 61511289.47B/s] 13%|█▎        | 55361536/440473133 [00:00<00:06, 62142059.56B/s] 14%|█▍        | 61937664/440473133 [00:01<00:05, 63184964.30B/s] 15%|█▌        | 68215808/440473133 [00:01<00:05, 63061976.97B/s] 17%|█▋        | 74748928/440473133 [00:01<00:05, 63722211.90B/s] 18%|█▊        | 81231872/440473133 [00:01<00:05, 64048505.84B/s] 20%|█▉        | 87606272/440473133 [00:01<00:05, 63922553.92B/s] 21%|██▏       | 93977600/440473133 [00:01<00:05, 63471010.74B/s] 23%|██▎       | 100311040/440473133 [00:01<00:05, 62556961.28B/s] 24%|██▍       | 106854400/440473133 [00:01<00:05, 63392223.28B/s] 26%|██▌       | 113426432/440473133 [00:01<00:05, 64071020.61B/s] 27%|██▋       | 119834624/440473133 [00:01<00:05, 63711930.63B/s] 29%|██▊       | 126434304/440473133 [00:02<00:04, 64379814.60B/s] 30%|███       | 132875264/440473133 [00:02<00:04, 64165285.51B/s] 32%|███▏      | 139411456/440473133 [00:02<00:04, 64518301.47B/s] 33%|███▎      | 145875968/440473133 [00:02<00:04, 64554837.68B/s] 35%|███▍      | 152333312/440473133 [00:02<00:04, 63794878.72B/s] 36%|███▌      | 158874624/440473133 [00:02<00:04, 64271086.28B/s] 38%|███▊      | 165305344/440473133 [00:02<00:04, 63563069.88B/s] 39%|███▉      | 171909120/440473133 [00:02<00:04, 64284232.20B/s] 41%|████      | 178411520/440473133 [00:02<00:04, 64501661.39B/s] 42%|████▏     | 184865792/440473133 [00:02<00:03, 64186388.60B/s] 43%|████▎     | 191287296/440473133 [00:03<00:04, 61962248.09B/s] 45%|████▍     | 197580800/440473133 [00:03<00:03, 62244677.66B/s] 46%|████▋     | 204191744/440473133 [00:03<00:03, 63353181.56B/s] 48%|████▊     | 210674688/440473133 [00:03<00:03, 63787308.02B/s] 49%|████▉     | 217282560/440473133 [00:03<00:03, 64457797.79B/s] 51%|█████     | 223762432/440473133 [00:03<00:03, 64554241.29B/s] 52%|█████▏    | 230224896/440473133 [00:03<00:03, 64549804.79B/s] 54%|█████▎    | 236685312/440473133 [00:03<00:03, 63748914.67B/s] 55%|█████▌    | 243065856/440473133 [00:03<00:03, 63374570.63B/s] 57%|█████▋    | 249581568/440473133 [00:03<00:02, 63897190.03B/s] 58%|█████▊    | 255975424/440473133 [00:04<00:02, 63905821.55B/s] 60%|█████▉    | 262392832/440473133 [00:04<00:02, 63983708.76B/s] 61%|██████    | 268876800/440473133 [00:04<00:02, 64236997.92B/s] 63%|██████▎   | 275302400/440473133 [00:04<00:02, 62946770.43B/s] 64%|██████▍   | 281643008/440473133 [00:04<00:02, 63080783.64B/s] 65%|██████▌   | 287993856/440473133 [00:04<00:02, 63207019.60B/s] 67%|██████▋   | 294476800/440473133 [00:04<00:02, 63678058.55B/s] 68%|██████▊   | 300848128/440473133 [00:04<00:02, 62819673.36B/s] 70%|██████▉   | 307408896/440473133 [00:04<00:02, 63629390.86B/s] 71%|███████   | 313778176/440473133 [00:04<00:01, 63416116.97B/s] 73%|███████▎  | 320124928/440473133 [00:05<00:01, 61931617.41B/s] 74%|███████▍  | 326329344/440473133 [00:05<00:01, 60563999.62B/s] 75%|███████▌  | 332400640/440473133 [00:05<00:01, 60081901.10B/s] 77%|███████▋  | 338419712/440473133 [00:05<00:01, 59518644.61B/s] 78%|███████▊  | 344380416/440473133 [00:05<00:01, 58950698.36B/s] 80%|███████▉  | 350332928/440473133 [00:05<00:01, 59121412.20B/s] 81%|████████  | 356250624/440473133 [00:05<00:01, 59038195.88B/s] 82%|████████▏ | 362159104/440473133 [00:05<00:01, 58743993.89B/s] 84%|████████▎ | 368079872/440473133 [00:05<00:01, 58881933.25B/s] 85%|████████▍ | 374062080/440473133 [00:05<00:01, 59158114.75B/s] 86%|████████▋ | 379986944/440473133 [00:06<00:01, 59183315.03B/s] 88%|████████▊ | 385906688/440473133 [00:06<00:00, 57597107.31B/s] 89%|████████▉ | 391698432/440473133 [00:06<00:00, 57690672.39B/s] 90%|█████████ | 397658112/440473133 [00:06<00:00, 58246831.33B/s] 92%|█████████▏| 403489792/440473133 [00:06<00:00, 58058784.25B/s] 93%|█████████▎| 409332736/440473133 [00:06<00:00, 58168757.26B/s] 94%|█████████▍| 415153152/440473133 [00:06<00:00, 58095944.83B/s] 96%|█████████▌| 421068800/440473133 [00:06<00:00, 58408086.35B/s] 97%|█████████▋| 427021312/440473133 [00:06<00:00, 58737169.27B/s] 98%|█████████▊| 432897024/440473133 [00:06<00:00, 58685174.32B/s]100%|█████████▉| 438767616/440473133 [00:07<00:00, 58450829.96B/s]100%|██████████| 440473133/440473133 [00:07<00:00, 61875410.15B/s]Downloading data from https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip

   8192/7094233 [..............................] - ETA: 0s
  65536/7094233 [..............................] - ETA: 6s
 385024/7094233 [>.............................] - ETA: 1s
1638400/7094233 [=====>........................] - ETA: 0s
7094272/7094233 [==============================] - 0s 0us/step

Processing text_left with encode:   0%|          | 0/2118 [00:00<?, ?it/s]Processing text_left with encode:  14%|█▍        | 295/2118 [00:00<00:00, 2945.67it/s]Processing text_left with encode:  35%|███▍      | 734/2118 [00:00<00:00, 3267.80it/s]Processing text_left with encode:  54%|█████▎    | 1137/2118 [00:00<00:00, 3462.01it/s]Processing text_left with encode:  75%|███████▍  | 1578/2118 [00:00<00:00, 3700.43it/s]Processing text_left with encode:  95%|█████████▍| 2005/2118 [00:00<00:00, 3854.33it/s]Processing text_left with encode: 100%|██████████| 2118/2118 [00:00<00:00, 4017.79it/s]
Processing text_right with encode:   0%|          | 0/18841 [00:00<?, ?it/s]Processing text_right with encode:   1%|          | 151/18841 [00:00<00:12, 1508.78it/s]Processing text_right with encode:   2%|▏         | 319/18841 [00:00<00:11, 1555.59it/s]Processing text_right with encode:   3%|▎         | 485/18841 [00:00<00:11, 1583.59it/s]Processing text_right with encode:   3%|▎         | 657/18841 [00:00<00:11, 1618.62it/s]Processing text_right with encode:   4%|▍         | 818/18841 [00:00<00:11, 1615.21it/s]Processing text_right with encode:   5%|▌         | 977/18841 [00:00<00:11, 1603.81it/s]Processing text_right with encode:   6%|▌         | 1134/18841 [00:00<00:11, 1592.25it/s]Processing text_right with encode:   7%|▋         | 1294/18841 [00:00<00:11, 1594.28it/s]Processing text_right with encode:   8%|▊         | 1448/18841 [00:00<00:11, 1575.34it/s]Processing text_right with encode:   9%|▊         | 1612/18841 [00:01<00:10, 1593.97it/s]Processing text_right with encode:   9%|▉         | 1786/18841 [00:01<00:10, 1632.81it/s]Processing text_right with encode:  10%|█         | 1947/18841 [00:01<00:10, 1614.85it/s]Processing text_right with encode:  11%|█         | 2107/18841 [00:01<00:10, 1600.31it/s]Processing text_right with encode:  12%|█▏        | 2278/18841 [00:01<00:10, 1628.95it/s]Processing text_right with encode:  13%|█▎        | 2445/18841 [00:01<00:09, 1640.66it/s]Processing text_right with encode:  14%|█▍        | 2609/18841 [00:01<00:09, 1629.57it/s]Processing text_right with encode:  15%|█▍        | 2790/18841 [00:01<00:09, 1677.66it/s]Processing text_right with encode:  16%|█▌        | 2958/18841 [00:01<00:09, 1635.59it/s]Processing text_right with encode:  17%|█▋        | 3122/18841 [00:01<00:09, 1600.06it/s]Processing text_right with encode:  17%|█▋        | 3283/18841 [00:02<00:09, 1560.58it/s]Processing text_right with encode:  18%|█▊        | 3440/18841 [00:02<00:09, 1547.41it/s]Processing text_right with encode:  19%|█▉        | 3597/18841 [00:02<00:09, 1553.55it/s]Processing text_right with encode:  20%|█▉        | 3759/18841 [00:02<00:09, 1569.43it/s]Processing text_right with encode:  21%|██        | 3917/18841 [00:02<00:09, 1538.90it/s]Processing text_right with encode:  22%|██▏       | 4085/18841 [00:02<00:09, 1574.73it/s]Processing text_right with encode:  23%|██▎       | 4243/18841 [00:02<00:09, 1565.47it/s]Processing text_right with encode:  23%|██▎       | 4403/18841 [00:02<00:09, 1574.32it/s]Processing text_right with encode:  24%|██▍       | 4561/18841 [00:02<00:09, 1560.94it/s]Processing text_right with encode:  25%|██▌       | 4725/18841 [00:02<00:08, 1578.25it/s]Processing text_right with encode:  26%|██▌       | 4895/18841 [00:03<00:08, 1612.13it/s]Processing text_right with encode:  27%|██▋       | 5067/18841 [00:03<00:08, 1641.02it/s]Processing text_right with encode:  28%|██▊       | 5248/18841 [00:03<00:08, 1687.37it/s]Processing text_right with encode:  29%|██▉       | 5418/18841 [00:03<00:08, 1660.90it/s]Processing text_right with encode:  30%|██▉       | 5585/18841 [00:03<00:08, 1623.96it/s]Processing text_right with encode:  31%|███       | 5748/18841 [00:03<00:08, 1615.64it/s]Processing text_right with encode:  31%|███▏      | 5911/18841 [00:03<00:07, 1619.81it/s]Processing text_right with encode:  32%|███▏      | 6074/18841 [00:03<00:07, 1609.83it/s]Processing text_right with encode:  33%|███▎      | 6236/18841 [00:03<00:07, 1602.73it/s]Processing text_right with encode:  34%|███▍      | 6397/18841 [00:03<00:07, 1601.40it/s]Processing text_right with encode:  35%|███▍      | 6559/18841 [00:04<00:07, 1605.53it/s]Processing text_right with encode:  36%|███▌      | 6722/18841 [00:04<00:07, 1612.00it/s]Processing text_right with encode:  37%|███▋      | 6885/18841 [00:04<00:07, 1615.15it/s]Processing text_right with encode:  37%|███▋      | 7049/18841 [00:04<00:07, 1621.40it/s]Processing text_right with encode:  38%|███▊      | 7214/18841 [00:04<00:07, 1627.27it/s]Processing text_right with encode:  39%|███▉      | 7391/18841 [00:04<00:06, 1666.41it/s]Processing text_right with encode:  40%|████      | 7558/18841 [00:04<00:06, 1647.47it/s]Processing text_right with encode:  41%|████      | 7723/18841 [00:04<00:06, 1645.69it/s]Processing text_right with encode:  42%|████▏     | 7888/18841 [00:04<00:06, 1640.77it/s]Processing text_right with encode:  43%|████▎     | 8053/18841 [00:04<00:06, 1610.82it/s]Processing text_right with encode:  44%|████▎     | 8219/18841 [00:05<00:06, 1622.70it/s]Processing text_right with encode:  45%|████▍     | 8393/18841 [00:05<00:06, 1655.66it/s]Processing text_right with encode:  45%|████▌     | 8559/18841 [00:05<00:06, 1617.06it/s]Processing text_right with encode:  46%|████▋     | 8730/18841 [00:05<00:06, 1643.32it/s]Processing text_right with encode:  47%|████▋     | 8903/18841 [00:05<00:05, 1668.29it/s]Processing text_right with encode:  48%|████▊     | 9071/18841 [00:05<00:06, 1614.03it/s]Processing text_right with encode:  49%|████▉     | 9238/18841 [00:05<00:05, 1630.16it/s]Processing text_right with encode:  50%|████▉     | 9402/18841 [00:05<00:05, 1605.71it/s]Processing text_right with encode:  51%|█████     | 9575/18841 [00:05<00:05, 1639.11it/s]Processing text_right with encode:  52%|█████▏    | 9740/18841 [00:06<00:05, 1595.26it/s]Processing text_right with encode:  53%|█████▎    | 9908/18841 [00:06<00:05, 1619.34it/s]Processing text_right with encode:  54%|█████▎    | 10085/18841 [00:06<00:05, 1661.60it/s]Processing text_right with encode:  54%|█████▍    | 10252/18841 [00:06<00:05, 1621.81it/s]Processing text_right with encode:  55%|█████▌    | 10445/18841 [00:06<00:04, 1699.74it/s]Processing text_right with encode:  56%|█████▋    | 10617/18841 [00:06<00:04, 1680.83it/s]Processing text_right with encode:  57%|█████▋    | 10787/18841 [00:06<00:04, 1675.35it/s]Processing text_right with encode:  58%|█████▊    | 10956/18841 [00:06<00:04, 1644.25it/s]Processing text_right with encode:  59%|█████▉    | 11122/18841 [00:06<00:04, 1596.57it/s]Processing text_right with encode:  60%|█████▉    | 11283/18841 [00:06<00:04, 1590.15it/s]Processing text_right with encode:  61%|██████    | 11446/18841 [00:07<00:04, 1600.70it/s]Processing text_right with encode:  62%|██████▏   | 11607/18841 [00:07<00:04, 1577.52it/s]Processing text_right with encode:  62%|██████▏   | 11774/18841 [00:07<00:04, 1603.30it/s]Processing text_right with encode:  63%|██████▎   | 11935/18841 [00:07<00:04, 1595.60it/s]Processing text_right with encode:  64%|██████▍   | 12095/18841 [00:07<00:04, 1595.46it/s]Processing text_right with encode:  65%|██████▌   | 12255/18841 [00:07<00:04, 1540.67it/s]Processing text_right with encode:  66%|██████▌   | 12410/18841 [00:07<00:04, 1534.47it/s]Processing text_right with encode:  67%|██████▋   | 12569/18841 [00:07<00:04, 1550.08it/s]Processing text_right with encode:  68%|██████▊   | 12736/18841 [00:07<00:03, 1582.81it/s]Processing text_right with encode:  68%|██████▊   | 12895/18841 [00:07<00:03, 1577.56it/s]Processing text_right with encode:  69%|██████▉   | 13065/18841 [00:08<00:03, 1611.57it/s]Processing text_right with encode:  70%|███████   | 13227/18841 [00:08<00:03, 1591.36it/s]Processing text_right with encode:  71%|███████   | 13393/18841 [00:08<00:03, 1610.85it/s]Processing text_right with encode:  72%|███████▏  | 13564/18841 [00:08<00:03, 1636.90it/s]Processing text_right with encode:  73%|███████▎  | 13743/18841 [00:08<00:03, 1679.64it/s]Processing text_right with encode:  74%|███████▍  | 13912/18841 [00:08<00:03, 1638.32it/s]Processing text_right with encode:  75%|███████▍  | 14079/18841 [00:08<00:02, 1645.13it/s]Processing text_right with encode:  76%|███████▌  | 14244/18841 [00:08<00:02, 1624.06it/s]Processing text_right with encode:  76%|███████▋  | 14409/18841 [00:08<00:02, 1630.61it/s]Processing text_right with encode:  77%|███████▋  | 14573/18841 [00:09<00:02, 1623.72it/s]Processing text_right with encode:  78%|███████▊  | 14752/18841 [00:09<00:02, 1670.25it/s]Processing text_right with encode:  79%|███████▉  | 14920/18841 [00:09<00:02, 1664.82it/s]Processing text_right with encode:  80%|████████  | 15087/18841 [00:09<00:02, 1663.24it/s]Processing text_right with encode:  81%|████████  | 15254/18841 [00:09<00:02, 1663.72it/s]Processing text_right with encode:  82%|████████▏ | 15421/18841 [00:09<00:02, 1642.20it/s]Processing text_right with encode:  83%|████████▎ | 15590/18841 [00:09<00:01, 1652.70it/s]Processing text_right with encode:  84%|████████▎ | 15765/18841 [00:09<00:01, 1679.67it/s]Processing text_right with encode:  85%|████████▍ | 15934/18841 [00:09<00:01, 1565.91it/s]Processing text_right with encode:  85%|████████▌ | 16094/18841 [00:09<00:01, 1575.68it/s]Processing text_right with encode:  86%|████████▋ | 16255/18841 [00:10<00:01, 1584.54it/s]Processing text_right with encode:  87%|████████▋ | 16420/18841 [00:10<00:01, 1603.40it/s]Processing text_right with encode:  88%|████████▊ | 16581/18841 [00:10<00:01, 1556.53it/s]Processing text_right with encode:  89%|████████▉ | 16743/18841 [00:10<00:01, 1574.55it/s]Processing text_right with encode:  90%|████████▉ | 16902/18841 [00:10<00:01, 1565.45it/s]Processing text_right with encode:  91%|█████████ | 17064/18841 [00:10<00:01, 1577.48it/s]Processing text_right with encode:  91%|█████████▏| 17224/18841 [00:10<00:01, 1580.83it/s]Processing text_right with encode:  92%|█████████▏| 17387/18841 [00:10<00:00, 1595.25it/s]Processing text_right with encode:  93%|█████████▎| 17547/18841 [00:10<00:00, 1595.31it/s]Processing text_right with encode:  94%|█████████▍| 17719/18841 [00:10<00:00, 1630.50it/s]Processing text_right with encode:  95%|█████████▍| 17883/18841 [00:11<00:00, 1611.46it/s]Processing text_right with encode:  96%|█████████▌| 18065/18841 [00:11<00:00, 1666.25it/s]Processing text_right with encode:  97%|█████████▋| 18233/18841 [00:11<00:00, 1658.87it/s]Processing text_right with encode:  98%|█████████▊| 18401/18841 [00:11<00:00, 1663.17it/s]Processing text_right with encode:  99%|█████████▊| 18578/18841 [00:11<00:00, 1692.33it/s]Processing text_right with encode: 100%|█████████▉| 18748/18841 [00:11<00:00, 1677.40it/s]Processing text_right with encode: 100%|██████████| 18841/18841 [00:11<00:00, 1619.20it/s]
Processing length_left with len:   0%|          | 0/2118 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 2118/2118 [00:00<00:00, 476610.11it/s]
Processing length_right with len:   0%|          | 0/18841 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 18841/18841 [00:00<00:00, 703663.08it/s]
Processing text_left with encode:   0%|          | 0/633 [00:00<?, ?it/s]Processing text_left with encode:  69%|██████▊   | 435/633 [00:00<00:00, 4349.79it/s]Processing text_left with encode: 100%|██████████| 633/633 [00:00<00:00, 4204.23it/s]
Processing text_right with encode:   0%|          | 0/5961 [00:00<?, ?it/s]Processing text_right with encode:   3%|▎         | 164/5961 [00:00<00:03, 1635.36it/s]Processing text_right with encode:   6%|▌         | 335/5961 [00:00<00:03, 1656.82it/s]Processing text_right with encode:   8%|▊         | 489/5961 [00:00<00:03, 1617.09it/s]Processing text_right with encode:  11%|█         | 658/5961 [00:00<00:03, 1638.11it/s]Processing text_right with encode:  14%|█▍        | 826/5961 [00:00<00:03, 1649.70it/s]Processing text_right with encode:  17%|█▋        | 1000/5961 [00:00<00:02, 1674.53it/s]Processing text_right with encode:  20%|█▉        | 1169/5961 [00:00<00:02, 1678.52it/s]Processing text_right with encode:  23%|██▎       | 1348/5961 [00:00<00:02, 1709.06it/s]Processing text_right with encode:  25%|██▌       | 1519/5961 [00:00<00:02, 1705.02it/s]Processing text_right with encode:  28%|██▊       | 1684/5961 [00:01<00:02, 1677.63it/s]Processing text_right with encode:  31%|███       | 1848/5961 [00:01<00:02, 1645.65it/s]Processing text_right with encode:  34%|███▍      | 2017/5961 [00:01<00:02, 1657.03it/s]Processing text_right with encode:  37%|███▋      | 2194/5961 [00:01<00:02, 1685.86it/s]Processing text_right with encode:  40%|███▉      | 2362/5961 [00:01<00:02, 1651.59it/s]Processing text_right with encode:  42%|████▏     | 2527/5961 [00:01<00:02, 1630.05it/s]Processing text_right with encode:  45%|████▌     | 2695/5961 [00:01<00:01, 1642.45it/s]Processing text_right with encode:  48%|████▊     | 2879/5961 [00:01<00:01, 1693.13it/s]Processing text_right with encode:  51%|█████     | 3049/5961 [00:01<00:01, 1689.79it/s]Processing text_right with encode:  54%|█████▍    | 3219/5961 [00:01<00:01, 1684.65it/s]Processing text_right with encode:  57%|█████▋    | 3388/5961 [00:02<00:01, 1630.72it/s]Processing text_right with encode:  60%|█████▉    | 3552/5961 [00:02<00:01, 1620.44it/s]Processing text_right with encode:  62%|██████▏   | 3716/5961 [00:02<00:01, 1624.85it/s]Processing text_right with encode:  65%|██████▌   | 3890/5961 [00:02<00:01, 1655.50it/s]Processing text_right with encode:  68%|██████▊   | 4065/5961 [00:02<00:01, 1679.97it/s]Processing text_right with encode:  71%|███████   | 4234/5961 [00:02<00:01, 1665.73it/s]Processing text_right with encode:  74%|███████▍  | 4403/5961 [00:02<00:00, 1672.05it/s]Processing text_right with encode:  77%|███████▋  | 4576/5961 [00:02<00:00, 1686.97it/s]Processing text_right with encode:  80%|███████▉  | 4745/5961 [00:02<00:00, 1666.66it/s]Processing text_right with encode:  82%|████████▏ | 4912/5961 [00:02<00:00, 1618.90it/s]Processing text_right with encode:  85%|████████▌ | 5075/5961 [00:03<00:00, 1617.06it/s]Processing text_right with encode:  88%|████████▊ | 5246/5961 [00:03<00:00, 1642.10it/s]Processing text_right with encode:  91%|█████████ | 5411/5961 [00:03<00:00, 1605.09it/s]Processing text_right with encode:  93%|█████████▎| 5572/5961 [00:03<00:00, 1577.59it/s]Processing text_right with encode:  96%|█████████▌| 5731/5961 [00:03<00:00, 1547.60it/s]Processing text_right with encode:  99%|█████████▉| 5908/5961 [00:03<00:00, 1605.36it/s]Processing text_right with encode: 100%|██████████| 5961/5961 [00:03<00:00, 1652.53it/s]
Processing length_left with len:   0%|          | 0/633 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 633/633 [00:00<00:00, 430237.31it/s]
Processing length_right with len:   0%|          | 0/5961 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 5961/5961 [00:00<00:00, 716992.52it/s]
  #### Model  fit   ############################################# 

  0%|          | 0/102 [00:00<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:18<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:18<?, ?it/s, loss=0.969]Epoch 1/1:   1%|          | 1/102 [00:18<30:47, 18.29s/it, loss=0.969]Killed

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   74a8de2..f1fc5db  master     -> origin/master
Updating 74a8de2..f1fc5db
Fast-forward
 error_list/20200513/list_log_test_cli_20200513.md | 138 +++++++++++-----------
 error_list/20200513/list_log_testall_20200513.md  |  33 ++++++
 2 files changed, 102 insertions(+), 69 deletions(-)
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
[master 609e9df] ml_store
 1 file changed, 66 insertions(+)
To github.com:arita37/mlmodels_store.git
   f1fc5db..609e9df  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py 

  #### Loading params   ############################################## 

  {'dataset': 'torchvision.datasets:MNIST', 'transform_uri': 'mlmodels.preprocess.image.py:torch_transform_mnist', '2nd___transform_uri': '/mnt/hgfs/d/gitdev/mlmodels/mlmodels/preprocess/image.py:torch_transform_mnist', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 4, 'test_batch_size': 1} {'checkpointdir': 'ztest/model_tch/torchhub/restnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18/'} 

  #### Loading dataset   ############################################# 

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:10, 140696.93it/s] 57%|█████▋    | 5685248/9912422 [00:00<00:21, 200782.05it/s]9920512it [00:00, 39277650.50it/s]                           
0it [00:00, ?it/s]32768it [00:00, 584471.06it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1012153.73it/s]1654784it [00:00, 12675026.66it/s]                           
0it [00:00, ?it/s]8192it [00:00, 227620.29it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
[master 2fcbd55] ml_store
 1 file changed, 84 insertions(+)
To github.com:arita37/mlmodels_store.git
   609e9df..2fcbd55  master -> master





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
[master d4b1734] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   2fcbd55..d4b1734  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_sentence.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Epoch:   0%|          | 0/1 [00:00<?, ?it/s]
Iteration:   0%|          | 0/29440 [00:00<?, ?it/s][A
Iteration:   0%|          | 1/29440 [00:09<77:08:48,  9.43s/it][A
Iteration:   0%|          | 2/29440 [00:35<117:59:09, 14.43s/it][A
Iteration:   0%|          | 3/29440 [00:44<104:55:16, 12.83s/it][A
Iteration:   0%|          | 4/29440 [01:02<117:04:21, 14.32s/it][A
Iteration:   0%|          | 5/29440 [02:53<354:17:54, 43.33s/it][A
Iteration:   0%|          | 6/29440 [03:32<344:28:00, 42.13s/it][A
Iteration:   0%|          | 7/29440 [04:13<341:06:59, 41.72s/it][A
Iteration:   0%|          | 8/29440 [04:26<270:28:12, 33.08s/it][A
Iteration:   0%|          | 9/29440 [05:01<275:06:30, 33.65s/it][A
Iteration:   0%|          | 10/29440 [07:12<513:45:59, 62.85s/it][A
Iteration:   0%|          | 11/29440 [07:23<385:55:50, 47.21s/it][A
Iteration:   0%|          | 12/29440 [08:27<429:12:14, 52.51s/it][A
Iteration:   0%|          | 13/29440 [08:48<351:54:05, 43.05s/it][A
Iteration:   0%|          | 14/29440 [08:58<269:16:34, 32.94s/it][A
Iteration:   0%|          | 15/29440 [09:15<230:54:51, 28.25s/it][A
Iteration:   0%|          | 16/29440 [09:25<184:55:12, 22.62s/it][A
Iteration:   0%|          | 17/29440 [10:56<354:31:15, 43.38s/it][A
Iteration:   0%|          | 18/29440 [12:32<482:32:44, 59.04s/it][AKilled

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   d4b1734..4a0ccd9  master     -> origin/master
Updating d4b1734..4a0ccd9
Fast-forward
 error_list/20200513/list_log_jupyter_20200513.md   | 1674 ++++++++---------
 error_list/20200513/list_log_testall_20200513.md   |   28 +
 ...-14_6672e19fe4cfa7df885e45d91d645534b8989485.py | 1954 ++++++++++++++++++++
 ...-10_6672e19fe4cfa7df885e45d91d645534b8989485.py |  611 ++++++
 4 files changed, 3430 insertions(+), 837 deletions(-)
 create mode 100644 log_jupyter/log_jupyter_2020-05-13-05-14_6672e19fe4cfa7df885e45d91d645534b8989485.py
 create mode 100644 log_pullrequest/log_pr_2020-05-13-05-10_6672e19fe4cfa7df885e45d91d645534b8989485.py
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
[master 4cbff41] ml_store
 1 file changed, 69 insertions(+)
To github.com:arita37/mlmodels_store.git
   4a0ccd9..4cbff41  master -> master





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
[master 21e7109] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   4cbff41..21e7109  master -> master





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
   21e7109..97e0033  master     -> origin/master
Updating 21e7109..97e0033
Fast-forward
 error_list/20200513/list_log_jupyter_20200513.md  | 2264 ++++++++++-----------
 error_list/20200513/list_log_test_cli_20200513.md |  138 +-
 error_list/20200513/list_log_testall_20200513.md  |    6 +
 3 files changed, 1203 insertions(+), 1205 deletions(-)
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
[master 05a8160] ml_store
 1 file changed, 52 insertions(+)
To github.com:arita37/mlmodels_store.git
   97e0033..05a8160  master -> master





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
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=20db5b2b3012f7dd178bdcd36dfb4e134773f7418a8b38afeea8799a281cd112
  Stored in directory: /tmp/pip-ephem-wheel-cache-09s3odrt/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
From github.com:arita37/mlmodels_store
   05a8160..c3ca4b7  master     -> origin/master
Updating 05a8160..c3ca4b7
Fast-forward
 error_list/20200513/list_log_benchmark_20200513.md |  186 +-
 error_list/20200513/list_log_jupyter_20200513.md   | 2264 ++++++++++----------
 error_list/20200513/list_log_test_cli_20200513.md  |  138 +-
 3 files changed, 1303 insertions(+), 1285 deletions(-)
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
[master 6fb4d83] ml_store
 1 file changed, 111 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.118.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   c3ca4b7..6fb4d83  master -> master





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
