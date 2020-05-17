
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
Warning: Permanently added the RSA host key for IP address '140.82.113.4' to the list of known hosts.
Already up to date.
[master 0a661ac] ml_store
 2 files changed, 74 insertions(+), 10992 deletions(-)
 rewrite log_testall/log_testall.py (99%)
To github.com:arita37/mlmodels_store.git
   9e1c7a3..0a661ac  master -> master





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
[master c5eb180] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   0a661ac..c5eb180  master -> master





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
[master 0866ca1] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   c5eb180..0866ca1  master -> master





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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         12          sequence_mean[0][0]              
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
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-17 20:11:53.930134: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-17 20:11:53.934169: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294690000 Hz
2020-05-17 20:11:53.934326: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5566fe638460 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-17 20:11:53.934341: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
100/500 [=====>........................] - ETA: 1s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 1s 1ms/sample - loss: 0.2536 - binary_crossentropy: 0.8042 - val_loss: 0.2535 - val_binary_crossentropy: 0.8039

  #### metrics   #################################################### 
{'MSE': 0.25331366334080846}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         12          sequence_mean[0][0]              
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
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         8           sparse_feature_2[0][0]           
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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_max[0][0]               
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
Total params: 453
Trainable params: 453
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2591 - binary_crossentropy: 0.7115500/500 [==============================] - 1s 1ms/sample - loss: 0.2554 - binary_crossentropy: 0.7041 - val_loss: 0.2506 - val_binary_crossentropy: 0.6944

  #### metrics   #################################################### 
{'MSE': 0.2527727249129429}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_max[0][0]               
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
Total params: 453
Trainable params: 453
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
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
Total params: 607
Trainable params: 607
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 1s 2ms/sample - loss: 0.2504 - binary_crossentropy: 0.7197 - val_loss: 0.2499 - val_binary_crossentropy: 0.7182

  #### metrics   #################################################### 
{'MSE': 0.25001857898448054}

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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
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
Total params: 607
Trainable params: 607
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
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
Total params: 393
Trainable params: 393
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2554 - binary_crossentropy: 0.7035500/500 [==============================] - 1s 2ms/sample - loss: 0.2637 - binary_crossentropy: 0.7224 - val_loss: 0.2519 - val_binary_crossentropy: 0.6971

  #### metrics   #################################################### 
{'MSE': 0.2554833316412683}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_max[0][0]               
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
Total params: 183
Trainable params: 183
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2955 - binary_crossentropy: 0.7946500/500 [==============================] - 1s 3ms/sample - loss: 0.2715 - binary_crossentropy: 0.8458 - val_loss: 0.2729 - val_binary_crossentropy: 0.8208

  #### metrics   #################################################### 
{'MSE': 0.26969683218150825}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_max[0][0]               
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
Total params: 183
Trainable params: 183
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-17 20:12:57.252896: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 20:12:57.254659: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 20:12:57.259933: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-17 20:12:57.268627: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-17 20:12:57.270171: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-17 20:12:57.271708: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 20:12:57.273127: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2512 - val_binary_crossentropy: 0.6955
2020-05-17 20:12:58.285339: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 20:12:58.287143: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 20:12:58.291079: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-17 20:12:58.298810: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-17 20:12:58.300149: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-17 20:12:58.301388: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 20:12:58.302558: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.25149605238898065}

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
2020-05-17 20:13:17.627513: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 20:13:17.628683: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 20:13:17.632173: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-17 20:13:17.637923: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-17 20:13:17.638878: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-17 20:13:17.639747: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 20:13:17.640565: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2480 - val_binary_crossentropy: 0.6891
2020-05-17 20:13:18.838621: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 20:13:18.839617: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 20:13:18.841992: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-17 20:13:18.846649: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-17 20:13:18.847517: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-17 20:13:18.848319: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 20:13:18.849019: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24711965668376715}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-17 20:13:47.206000: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 20:13:47.210242: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 20:13:47.222440: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-17 20:13:47.243476: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-17 20:13:47.247162: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-17 20:13:47.250549: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 20:13:47.253922: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 4s 4s/sample - loss: 0.0141 - binary_crossentropy: 0.1266 - val_loss: 0.3933 - val_binary_crossentropy: 1.1190
2020-05-17 20:13:49.072701: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 20:13:49.076704: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 20:13:49.087060: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-17 20:13:49.107655: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-17 20:13:49.111207: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-17 20:13:49.114422: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-17 20:13:49.117850: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.26713909686130677}

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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
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
100/500 [=====>........................] - ETA: 5s - loss: 0.3036 - binary_crossentropy: 0.8185500/500 [==============================] - 4s 7ms/sample - loss: 0.3077 - binary_crossentropy: 0.9317 - val_loss: 0.2986 - val_binary_crossentropy: 0.8330

  #### metrics   #################################################### 
{'MSE': 0.3020173101750465}

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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 2)         18          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 2)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 2)         6           sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         8           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         18          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         2           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         10          sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         3           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_5[0][0]           
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
Total params: 245
Trainable params: 245
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 5s - loss: 0.2877 - binary_crossentropy: 0.7840500/500 [==============================] - 4s 8ms/sample - loss: 0.2873 - binary_crossentropy: 0.7823 - val_loss: 0.2670 - val_binary_crossentropy: 0.7305

  #### metrics   #################################################### 
{'MSE': 0.26835556564899476}

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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 2)         18          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 2)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 2)         6           sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         8           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         18          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         2           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         10          sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         3           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_5[0][0]           
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
Total params: 245
Trainable params: 245
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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
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
Total params: 1,914
Trainable params: 1,914
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 5s - loss: 0.2991 - binary_crossentropy: 0.8090500/500 [==============================] - 4s 8ms/sample - loss: 0.2985 - binary_crossentropy: 0.8058 - val_loss: 0.2761 - val_binary_crossentropy: 0.7542

  #### metrics   #################################################### 
{'MSE': 0.2848576664379038}

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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
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
Total params: 1,914
Trainable params: 1,914
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
regionsequence_sum (InputLayer) [(None, 3)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 3)]          0                                            
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
region_10sparse_seq_emb_regions (None, 3, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         2           regionsequence_mean[0][0]        
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
100/500 [=====>........................] - ETA: 7s - loss: 0.2495 - binary_crossentropy: 0.6920500/500 [==============================] - 5s 10ms/sample - loss: 0.2506 - binary_crossentropy: 0.6943 - val_loss: 0.2515 - val_binary_crossentropy: 0.6961

  #### metrics   #################################################### 
{'MSE': 0.2509576113002975}

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
regionsequence_sum (InputLayer) [(None, 3)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 3)]          0                                            
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
region_10sparse_seq_emb_regions (None, 3, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         2           regionsequence_mean[0][0]        
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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
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
Total params: 1,422
Trainable params: 1,422
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2577 - binary_crossentropy: 0.7089500/500 [==============================] - 5s 10ms/sample - loss: 0.2536 - binary_crossentropy: 0.7006 - val_loss: 0.2504 - val_binary_crossentropy: 0.6939

  #### metrics   #################################################### 
{'MSE': 0.25130136213162213}

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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
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
Total params: 1,422
Trainable params: 1,422
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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         12          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         24          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         12          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         12          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         12          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 5, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 5, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 5, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 5, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           hash_10[0][0]                    
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
Total params: 2,848
Trainable params: 2,768
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.5500 - binary_crossentropy: 8.4837500/500 [==============================] - 5s 11ms/sample - loss: 0.5300 - binary_crossentropy: 8.1752 - val_loss: 0.5000 - val_binary_crossentropy: 7.7125

  #### metrics   #################################################### 
{'MSE': 0.515}

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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         12          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         24          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         12          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         12          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         12          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 5, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 5, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 5, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 5, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           hash_10[0][0]                    
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
Total params: 2,848
Trainable params: 2,768
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
From github.com:arita37/mlmodels_store
   0866ca1..88e0a33  master     -> origin/master
Updating 0866ca1..88e0a33
Fast-forward
 error_list/20200517/list_log_benchmark_20200517.md |  182 +-
 error_list/20200517/list_log_json_20200517.md      | 1146 ++++++-------
 error_list/20200517/list_log_jupyter_20200517.md   | 1749 ++++++++++----------
 error_list/20200517/list_log_test_cli_20200517.md  |  364 ++--
 4 files changed, 1725 insertions(+), 1716 deletions(-)
[master 92aebaf] ml_store
 1 file changed, 4955 insertions(+)
To github.com:arita37/mlmodels_store.git
   88e0a33..92aebaf  master -> master





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
[master 2be3719] ml_store
 1 file changed, 51 insertions(+)
To github.com:arita37/mlmodels_store.git
   92aebaf..2be3719  master -> master





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
[master dea3141] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   2be3719..dea3141  master -> master





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
[master 32f6778] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   dea3141..32f6778  master -> master





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

2020-05-17 20:21:45.501653: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-17 20:21:45.506866: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294690000 Hz
2020-05-17 20:21:45.507023: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5640543b0da0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-17 20:21:45.507038: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 8s - loss: 1.3849
256/354 [====================>.........] - ETA: 3s - loss: 1.2692
354/354 [==============================] - 14s 40ms/step - loss: 1.3921 - val_loss: 1.9967

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
Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
Already up to date.
[master 1dbddf5] ml_store
 1 file changed, 151 insertions(+)
To github.com:arita37/mlmodels_store.git
   32f6778..1dbddf5  master -> master





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
[master 9e619a1] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   1dbddf5..9e619a1  master -> master





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
[master e362efe] ml_store
 1 file changed, 45 insertions(+)
To github.com:arita37/mlmodels_store.git
   9e619a1..e362efe  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2842624/17464789 [===>..........................] - ETA: 0s
11288576/17464789 [==================>...........] - ETA: 0s
16310272/17464789 [===========================>..] - ETA: 0s
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
2020-05-17 20:22:44.376193: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-17 20:22:44.379770: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294690000 Hz
2020-05-17 20:22:44.379893: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f032070a50 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-17 20:22:44.379907: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 10s - loss: 8.0040 - accuracy: 0.4780
 2000/25000 [=>............................] - ETA: 8s - loss: 7.8736 - accuracy: 0.4865 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.8660 - accuracy: 0.4870
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.8736 - accuracy: 0.4865
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.9120 - accuracy: 0.4840
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.8685 - accuracy: 0.4868
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.8287 - accuracy: 0.4894
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.8065 - accuracy: 0.4909
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.8063 - accuracy: 0.4909
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7540 - accuracy: 0.4943
11000/25000 [============>.................] - ETA: 3s - loss: 7.7210 - accuracy: 0.4965
12000/25000 [=============>................] - ETA: 3s - loss: 7.7062 - accuracy: 0.4974
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7055 - accuracy: 0.4975
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7082 - accuracy: 0.4973
15000/25000 [=================>............] - ETA: 2s - loss: 7.6963 - accuracy: 0.4981
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6887 - accuracy: 0.4986
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6874 - accuracy: 0.4986
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6871 - accuracy: 0.4987
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6868 - accuracy: 0.4987
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6682 - accuracy: 0.4999
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6564 - accuracy: 0.5007
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6548 - accuracy: 0.5008
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6693 - accuracy: 0.4998
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6602 - accuracy: 0.5004
25000/25000 [==============================] - 8s 327us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7f229725eb00>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f228cd779b0> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 7.7280 - accuracy: 0.4960
 2000/25000 [=>............................] - ETA: 8s - loss: 7.8736 - accuracy: 0.4865 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6564 - accuracy: 0.5007
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6206 - accuracy: 0.5030
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6206 - accuracy: 0.5030
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5848 - accuracy: 0.5053
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5856 - accuracy: 0.5053
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5842 - accuracy: 0.5054
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5797 - accuracy: 0.5057
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5900 - accuracy: 0.5050
11000/25000 [============>.................] - ETA: 3s - loss: 7.5621 - accuracy: 0.5068
12000/25000 [=============>................] - ETA: 3s - loss: 7.5848 - accuracy: 0.5053
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6018 - accuracy: 0.5042
14000/25000 [===============>..............] - ETA: 2s - loss: 7.5834 - accuracy: 0.5054
15000/25000 [=================>............] - ETA: 2s - loss: 7.5838 - accuracy: 0.5054
16000/25000 [==================>...........] - ETA: 2s - loss: 7.5804 - accuracy: 0.5056
17000/25000 [===================>..........] - ETA: 2s - loss: 7.5981 - accuracy: 0.5045
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6130 - accuracy: 0.5035
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6279 - accuracy: 0.5025
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6429 - accuracy: 0.5016
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6287 - accuracy: 0.5025
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6478 - accuracy: 0.5012
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6506 - accuracy: 0.5010
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6513 - accuracy: 0.5010
25000/25000 [==============================] - 8s 326us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 11s - loss: 7.3600 - accuracy: 0.5200
 2000/25000 [=>............................] - ETA: 8s - loss: 7.3906 - accuracy: 0.5180 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.4877 - accuracy: 0.5117
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6206 - accuracy: 0.5030
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6114 - accuracy: 0.5036
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6206 - accuracy: 0.5030
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6053 - accuracy: 0.5040
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6053 - accuracy: 0.5040
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5848 - accuracy: 0.5053
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6053 - accuracy: 0.5040
11000/25000 [============>.................] - ETA: 3s - loss: 7.6262 - accuracy: 0.5026
12000/25000 [=============>................] - ETA: 3s - loss: 7.6321 - accuracy: 0.5023
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6525 - accuracy: 0.5009
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6469 - accuracy: 0.5013
15000/25000 [=================>............] - ETA: 2s - loss: 7.6574 - accuracy: 0.5006
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6436 - accuracy: 0.5015
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6224 - accuracy: 0.5029
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6036 - accuracy: 0.5041
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6182 - accuracy: 0.5032
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6360 - accuracy: 0.5020
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6674 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6638 - accuracy: 0.5002
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6733 - accuracy: 0.4996
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
25000/25000 [==============================] - 8s 328us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
Already up to date.
[master 0d9a89a] ml_store
 1 file changed, 319 insertions(+)
To github.com:arita37/mlmodels_store.git
   e362efe..0d9a89a  master -> master





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

13/13 [==============================] - 1s 97ms/step - loss: nan
Epoch 2/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 3/10

13/13 [==============================] - 0s 3ms/step - loss: nan
Epoch 4/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 5/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 6/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 7/10

13/13 [==============================] - 0s 5ms/step - loss: nan
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
[master c3562e2] ml_store
 1 file changed, 126 insertions(+)
To github.com:arita37/mlmodels_store.git
   0d9a89a..c3562e2  master -> master





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
 3375104/11490434 [=======>......................] - ETA: 0s
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

   32/60000 [..............................] - ETA: 6:20 - loss: 2.3302 - categorical_accuracy: 0.0000e+00
   64/60000 [..............................] - ETA: 3:59 - loss: 2.2657 - categorical_accuracy: 0.1406    
   96/60000 [..............................] - ETA: 3:12 - loss: 2.2638 - categorical_accuracy: 0.1458
  160/60000 [..............................] - ETA: 2:31 - loss: 2.2371 - categorical_accuracy: 0.1813
  192/60000 [..............................] - ETA: 2:22 - loss: 2.2107 - categorical_accuracy: 0.2031
  256/60000 [..............................] - ETA: 2:09 - loss: 2.1604 - categorical_accuracy: 0.2344
  288/60000 [..............................] - ETA: 2:06 - loss: 2.1204 - categorical_accuracy: 0.2535
  352/60000 [..............................] - ETA: 2:00 - loss: 2.0912 - categorical_accuracy: 0.2670
  384/60000 [..............................] - ETA: 1:58 - loss: 2.0507 - categorical_accuracy: 0.2917
  416/60000 [..............................] - ETA: 1:57 - loss: 2.0036 - categorical_accuracy: 0.3149
  448/60000 [..............................] - ETA: 1:55 - loss: 1.9685 - categorical_accuracy: 0.3326
  480/60000 [..............................] - ETA: 1:55 - loss: 1.9633 - categorical_accuracy: 0.3375
  512/60000 [..............................] - ETA: 1:54 - loss: 1.9451 - categorical_accuracy: 0.3477
  544/60000 [..............................] - ETA: 1:53 - loss: 1.9200 - categorical_accuracy: 0.3585
  576/60000 [..............................] - ETA: 1:53 - loss: 1.8980 - categorical_accuracy: 0.3594
  608/60000 [..............................] - ETA: 1:52 - loss: 1.8778 - categorical_accuracy: 0.3602
  640/60000 [..............................] - ETA: 1:52 - loss: 1.8541 - categorical_accuracy: 0.3672
  672/60000 [..............................] - ETA: 1:51 - loss: 1.8199 - categorical_accuracy: 0.3824
  736/60000 [..............................] - ETA: 1:49 - loss: 1.7741 - categorical_accuracy: 0.3995
  800/60000 [..............................] - ETA: 1:48 - loss: 1.7115 - categorical_accuracy: 0.4200
  832/60000 [..............................] - ETA: 1:47 - loss: 1.6915 - categorical_accuracy: 0.4267
  864/60000 [..............................] - ETA: 1:47 - loss: 1.6638 - categorical_accuracy: 0.4398
  896/60000 [..............................] - ETA: 1:47 - loss: 1.6401 - categorical_accuracy: 0.4464
  928/60000 [..............................] - ETA: 1:47 - loss: 1.6063 - categorical_accuracy: 0.4569
  960/60000 [..............................] - ETA: 1:46 - loss: 1.5884 - categorical_accuracy: 0.4625
 1024/60000 [..............................] - ETA: 1:45 - loss: 1.5326 - categorical_accuracy: 0.4844
 1088/60000 [..............................] - ETA: 1:44 - loss: 1.4985 - categorical_accuracy: 0.4963
 1120/60000 [..............................] - ETA: 1:44 - loss: 1.4779 - categorical_accuracy: 0.5036
 1184/60000 [..............................] - ETA: 1:43 - loss: 1.4436 - categorical_accuracy: 0.5152
 1248/60000 [..............................] - ETA: 1:42 - loss: 1.4210 - categorical_accuracy: 0.5240
 1280/60000 [..............................] - ETA: 1:42 - loss: 1.4040 - categorical_accuracy: 0.5289
 1344/60000 [..............................] - ETA: 1:42 - loss: 1.3643 - categorical_accuracy: 0.5432
 1376/60000 [..............................] - ETA: 1:41 - loss: 1.3414 - categorical_accuracy: 0.5501
 1408/60000 [..............................] - ETA: 1:41 - loss: 1.3275 - categorical_accuracy: 0.5547
 1440/60000 [..............................] - ETA: 1:41 - loss: 1.3168 - categorical_accuracy: 0.5569
 1472/60000 [..............................] - ETA: 1:41 - loss: 1.3065 - categorical_accuracy: 0.5598
 1504/60000 [..............................] - ETA: 1:41 - loss: 1.2931 - categorical_accuracy: 0.5645
 1536/60000 [..............................] - ETA: 1:41 - loss: 1.2791 - categorical_accuracy: 0.5703
 1568/60000 [..............................] - ETA: 1:41 - loss: 1.2677 - categorical_accuracy: 0.5753
 1600/60000 [..............................] - ETA: 1:41 - loss: 1.2568 - categorical_accuracy: 0.5800
 1664/60000 [..............................] - ETA: 1:40 - loss: 1.2361 - categorical_accuracy: 0.5865
 1728/60000 [..............................] - ETA: 1:40 - loss: 1.2083 - categorical_accuracy: 0.5938
 1792/60000 [..............................] - ETA: 1:39 - loss: 1.1871 - categorical_accuracy: 0.6004
 1824/60000 [..............................] - ETA: 1:39 - loss: 1.1782 - categorical_accuracy: 0.6036
 1888/60000 [..............................] - ETA: 1:38 - loss: 1.1552 - categorical_accuracy: 0.6123
 1920/60000 [..............................] - ETA: 1:38 - loss: 1.1449 - categorical_accuracy: 0.6161
 1984/60000 [..............................] - ETA: 1:38 - loss: 1.1232 - categorical_accuracy: 0.6235
 2048/60000 [>.............................] - ETA: 1:37 - loss: 1.1014 - categorical_accuracy: 0.6318
 2080/60000 [>.............................] - ETA: 1:37 - loss: 1.0881 - categorical_accuracy: 0.6361
 2144/60000 [>.............................] - ETA: 1:37 - loss: 1.0711 - categorical_accuracy: 0.6423
 2208/60000 [>.............................] - ETA: 1:36 - loss: 1.0540 - categorical_accuracy: 0.6467
 2272/60000 [>.............................] - ETA: 1:36 - loss: 1.0420 - categorical_accuracy: 0.6532
 2336/60000 [>.............................] - ETA: 1:36 - loss: 1.0249 - categorical_accuracy: 0.6588
 2400/60000 [>.............................] - ETA: 1:35 - loss: 1.0088 - categorical_accuracy: 0.6646
 2432/60000 [>.............................] - ETA: 1:35 - loss: 0.9977 - categorical_accuracy: 0.6686
 2464/60000 [>.............................] - ETA: 1:35 - loss: 0.9908 - categorical_accuracy: 0.6717
 2496/60000 [>.............................] - ETA: 1:35 - loss: 0.9866 - categorical_accuracy: 0.6735
 2560/60000 [>.............................] - ETA: 1:35 - loss: 0.9792 - categorical_accuracy: 0.6766
 2592/60000 [>.............................] - ETA: 1:35 - loss: 0.9702 - categorical_accuracy: 0.6798
 2656/60000 [>.............................] - ETA: 1:35 - loss: 0.9575 - categorical_accuracy: 0.6849
 2688/60000 [>.............................] - ETA: 1:35 - loss: 0.9522 - categorical_accuracy: 0.6875
 2752/60000 [>.............................] - ETA: 1:35 - loss: 0.9434 - categorical_accuracy: 0.6911
 2816/60000 [>.............................] - ETA: 1:34 - loss: 0.9313 - categorical_accuracy: 0.6946
 2880/60000 [>.............................] - ETA: 1:34 - loss: 0.9211 - categorical_accuracy: 0.6972
 2944/60000 [>.............................] - ETA: 1:34 - loss: 0.9152 - categorical_accuracy: 0.7001
 3008/60000 [>.............................] - ETA: 1:33 - loss: 0.9032 - categorical_accuracy: 0.7038
 3040/60000 [>.............................] - ETA: 1:33 - loss: 0.8984 - categorical_accuracy: 0.7046
 3072/60000 [>.............................] - ETA: 1:33 - loss: 0.8916 - categorical_accuracy: 0.7067
 3104/60000 [>.............................] - ETA: 1:33 - loss: 0.8841 - categorical_accuracy: 0.7091
 3136/60000 [>.............................] - ETA: 1:33 - loss: 0.8782 - categorical_accuracy: 0.7108
 3200/60000 [>.............................] - ETA: 1:33 - loss: 0.8665 - categorical_accuracy: 0.7147
 3232/60000 [>.............................] - ETA: 1:33 - loss: 0.8615 - categorical_accuracy: 0.7163
 3296/60000 [>.............................] - ETA: 1:33 - loss: 0.8524 - categorical_accuracy: 0.7194
 3328/60000 [>.............................] - ETA: 1:33 - loss: 0.8485 - categorical_accuracy: 0.7203
 3360/60000 [>.............................] - ETA: 1:33 - loss: 0.8435 - categorical_accuracy: 0.7217
 3392/60000 [>.............................] - ETA: 1:33 - loss: 0.8383 - categorical_accuracy: 0.7229
 3456/60000 [>.............................] - ETA: 1:32 - loss: 0.8273 - categorical_accuracy: 0.7271
 3488/60000 [>.............................] - ETA: 1:32 - loss: 0.8238 - categorical_accuracy: 0.7282
 3520/60000 [>.............................] - ETA: 1:32 - loss: 0.8213 - categorical_accuracy: 0.7293
 3552/60000 [>.............................] - ETA: 1:32 - loss: 0.8186 - categorical_accuracy: 0.7303
 3616/60000 [>.............................] - ETA: 1:32 - loss: 0.8074 - categorical_accuracy: 0.7340
 3648/60000 [>.............................] - ETA: 1:32 - loss: 0.8056 - categorical_accuracy: 0.7357
 3680/60000 [>.............................] - ETA: 1:32 - loss: 0.8010 - categorical_accuracy: 0.7375
 3744/60000 [>.............................] - ETA: 1:32 - loss: 0.7930 - categorical_accuracy: 0.7404
 3776/60000 [>.............................] - ETA: 1:32 - loss: 0.7882 - categorical_accuracy: 0.7418
 3840/60000 [>.............................] - ETA: 1:31 - loss: 0.7809 - categorical_accuracy: 0.7437
 3904/60000 [>.............................] - ETA: 1:31 - loss: 0.7703 - categorical_accuracy: 0.7477
 3968/60000 [>.............................] - ETA: 1:31 - loss: 0.7681 - categorical_accuracy: 0.7490
 4000/60000 [=>............................] - ETA: 1:31 - loss: 0.7685 - categorical_accuracy: 0.7492
 4032/60000 [=>............................] - ETA: 1:31 - loss: 0.7659 - categorical_accuracy: 0.7505
 4096/60000 [=>............................] - ETA: 1:31 - loss: 0.7607 - categorical_accuracy: 0.7527
 4160/60000 [=>............................] - ETA: 1:31 - loss: 0.7535 - categorical_accuracy: 0.7553
 4224/60000 [=>............................] - ETA: 1:30 - loss: 0.7472 - categorical_accuracy: 0.7571
 4256/60000 [=>............................] - ETA: 1:30 - loss: 0.7463 - categorical_accuracy: 0.7573
 4320/60000 [=>............................] - ETA: 1:30 - loss: 0.7406 - categorical_accuracy: 0.7595
 4352/60000 [=>............................] - ETA: 1:30 - loss: 0.7378 - categorical_accuracy: 0.7601
 4384/60000 [=>............................] - ETA: 1:30 - loss: 0.7361 - categorical_accuracy: 0.7607
 4416/60000 [=>............................] - ETA: 1:30 - loss: 0.7332 - categorical_accuracy: 0.7615
 4448/60000 [=>............................] - ETA: 1:30 - loss: 0.7300 - categorical_accuracy: 0.7628
 4512/60000 [=>............................] - ETA: 1:30 - loss: 0.7251 - categorical_accuracy: 0.7651
 4544/60000 [=>............................] - ETA: 1:30 - loss: 0.7226 - categorical_accuracy: 0.7658
 4576/60000 [=>............................] - ETA: 1:30 - loss: 0.7205 - categorical_accuracy: 0.7668
 4608/60000 [=>............................] - ETA: 1:30 - loss: 0.7193 - categorical_accuracy: 0.7674
 4640/60000 [=>............................] - ETA: 1:30 - loss: 0.7158 - categorical_accuracy: 0.7685
 4672/60000 [=>............................] - ETA: 1:30 - loss: 0.7129 - categorical_accuracy: 0.7695
 4736/60000 [=>............................] - ETA: 1:30 - loss: 0.7076 - categorical_accuracy: 0.7713
 4768/60000 [=>............................] - ETA: 1:30 - loss: 0.7051 - categorical_accuracy: 0.7727
 4800/60000 [=>............................] - ETA: 1:30 - loss: 0.7030 - categorical_accuracy: 0.7733
 4832/60000 [=>............................] - ETA: 1:30 - loss: 0.6997 - categorical_accuracy: 0.7744
 4864/60000 [=>............................] - ETA: 1:29 - loss: 0.6965 - categorical_accuracy: 0.7755
 4928/60000 [=>............................] - ETA: 1:29 - loss: 0.6895 - categorical_accuracy: 0.7780
 4960/60000 [=>............................] - ETA: 1:29 - loss: 0.6869 - categorical_accuracy: 0.7786
 5024/60000 [=>............................] - ETA: 1:29 - loss: 0.6816 - categorical_accuracy: 0.7805
 5056/60000 [=>............................] - ETA: 1:29 - loss: 0.6809 - categorical_accuracy: 0.7805
 5120/60000 [=>............................] - ETA: 1:29 - loss: 0.6749 - categorical_accuracy: 0.7824
 5184/60000 [=>............................] - ETA: 1:29 - loss: 0.6727 - categorical_accuracy: 0.7834
 5248/60000 [=>............................] - ETA: 1:29 - loss: 0.6700 - categorical_accuracy: 0.7849
 5280/60000 [=>............................] - ETA: 1:28 - loss: 0.6679 - categorical_accuracy: 0.7858
 5344/60000 [=>............................] - ETA: 1:28 - loss: 0.6635 - categorical_accuracy: 0.7869
 5376/60000 [=>............................] - ETA: 1:28 - loss: 0.6616 - categorical_accuracy: 0.7878
 5440/60000 [=>............................] - ETA: 1:28 - loss: 0.6562 - categorical_accuracy: 0.7895
 5504/60000 [=>............................] - ETA: 1:28 - loss: 0.6499 - categorical_accuracy: 0.7918
 5568/60000 [=>............................] - ETA: 1:28 - loss: 0.6438 - categorical_accuracy: 0.7940
 5600/60000 [=>............................] - ETA: 1:28 - loss: 0.6418 - categorical_accuracy: 0.7948
 5632/60000 [=>............................] - ETA: 1:28 - loss: 0.6399 - categorical_accuracy: 0.7951
 5664/60000 [=>............................] - ETA: 1:28 - loss: 0.6369 - categorical_accuracy: 0.7961
 5696/60000 [=>............................] - ETA: 1:28 - loss: 0.6346 - categorical_accuracy: 0.7967
 5728/60000 [=>............................] - ETA: 1:28 - loss: 0.6330 - categorical_accuracy: 0.7971
 5792/60000 [=>............................] - ETA: 1:27 - loss: 0.6268 - categorical_accuracy: 0.7992
 5824/60000 [=>............................] - ETA: 1:27 - loss: 0.6241 - categorical_accuracy: 0.8001
 5888/60000 [=>............................] - ETA: 1:27 - loss: 0.6208 - categorical_accuracy: 0.8015
 5920/60000 [=>............................] - ETA: 1:27 - loss: 0.6198 - categorical_accuracy: 0.8019
 5952/60000 [=>............................] - ETA: 1:27 - loss: 0.6175 - categorical_accuracy: 0.8026
 6016/60000 [==>...........................] - ETA: 1:27 - loss: 0.6164 - categorical_accuracy: 0.8034
 6080/60000 [==>...........................] - ETA: 1:27 - loss: 0.6136 - categorical_accuracy: 0.8043
 6144/60000 [==>...........................] - ETA: 1:27 - loss: 0.6105 - categorical_accuracy: 0.8057
 6208/60000 [==>...........................] - ETA: 1:26 - loss: 0.6069 - categorical_accuracy: 0.8069
 6240/60000 [==>...........................] - ETA: 1:26 - loss: 0.6046 - categorical_accuracy: 0.8074
 6304/60000 [==>...........................] - ETA: 1:26 - loss: 0.6001 - categorical_accuracy: 0.8085
 6368/60000 [==>...........................] - ETA: 1:26 - loss: 0.5970 - categorical_accuracy: 0.8094
 6400/60000 [==>...........................] - ETA: 1:26 - loss: 0.5957 - categorical_accuracy: 0.8095
 6432/60000 [==>...........................] - ETA: 1:26 - loss: 0.5955 - categorical_accuracy: 0.8097
 6464/60000 [==>...........................] - ETA: 1:26 - loss: 0.5933 - categorical_accuracy: 0.8102
 6496/60000 [==>...........................] - ETA: 1:26 - loss: 0.5909 - categorical_accuracy: 0.8110
 6528/60000 [==>...........................] - ETA: 1:26 - loss: 0.5896 - categorical_accuracy: 0.8111
 6560/60000 [==>...........................] - ETA: 1:26 - loss: 0.5876 - categorical_accuracy: 0.8119
 6624/60000 [==>...........................] - ETA: 1:26 - loss: 0.5835 - categorical_accuracy: 0.8133
 6688/60000 [==>...........................] - ETA: 1:26 - loss: 0.5803 - categorical_accuracy: 0.8140
 6720/60000 [==>...........................] - ETA: 1:26 - loss: 0.5779 - categorical_accuracy: 0.8147
 6752/60000 [==>...........................] - ETA: 1:26 - loss: 0.5762 - categorical_accuracy: 0.8152
 6784/60000 [==>...........................] - ETA: 1:26 - loss: 0.5745 - categorical_accuracy: 0.8159
 6816/60000 [==>...........................] - ETA: 1:26 - loss: 0.5732 - categorical_accuracy: 0.8165
 6848/60000 [==>...........................] - ETA: 1:26 - loss: 0.5718 - categorical_accuracy: 0.8167
 6880/60000 [==>...........................] - ETA: 1:26 - loss: 0.5697 - categorical_accuracy: 0.8173
 6944/60000 [==>...........................] - ETA: 1:25 - loss: 0.5672 - categorical_accuracy: 0.8178
 6976/60000 [==>...........................] - ETA: 1:25 - loss: 0.5657 - categorical_accuracy: 0.8182
 7040/60000 [==>...........................] - ETA: 1:25 - loss: 0.5619 - categorical_accuracy: 0.8196
 7104/60000 [==>...........................] - ETA: 1:25 - loss: 0.5579 - categorical_accuracy: 0.8211
 7136/60000 [==>...........................] - ETA: 1:25 - loss: 0.5575 - categorical_accuracy: 0.8217
 7200/60000 [==>...........................] - ETA: 1:25 - loss: 0.5560 - categorical_accuracy: 0.8225
 7264/60000 [==>...........................] - ETA: 1:25 - loss: 0.5524 - categorical_accuracy: 0.8237
 7296/60000 [==>...........................] - ETA: 1:25 - loss: 0.5507 - categorical_accuracy: 0.8243
 7360/60000 [==>...........................] - ETA: 1:24 - loss: 0.5480 - categorical_accuracy: 0.8251
 7424/60000 [==>...........................] - ETA: 1:24 - loss: 0.5439 - categorical_accuracy: 0.8264
 7488/60000 [==>...........................] - ETA: 1:24 - loss: 0.5407 - categorical_accuracy: 0.8273
 7520/60000 [==>...........................] - ETA: 1:24 - loss: 0.5395 - categorical_accuracy: 0.8277
 7552/60000 [==>...........................] - ETA: 1:24 - loss: 0.5383 - categorical_accuracy: 0.8280
 7584/60000 [==>...........................] - ETA: 1:24 - loss: 0.5368 - categorical_accuracy: 0.8285
 7648/60000 [==>...........................] - ETA: 1:24 - loss: 0.5333 - categorical_accuracy: 0.8296
 7680/60000 [==>...........................] - ETA: 1:24 - loss: 0.5324 - categorical_accuracy: 0.8301
 7744/60000 [==>...........................] - ETA: 1:24 - loss: 0.5308 - categorical_accuracy: 0.8308
 7776/60000 [==>...........................] - ETA: 1:24 - loss: 0.5293 - categorical_accuracy: 0.8314
 7808/60000 [==>...........................] - ETA: 1:24 - loss: 0.5286 - categorical_accuracy: 0.8316
 7840/60000 [==>...........................] - ETA: 1:24 - loss: 0.5272 - categorical_accuracy: 0.8320
 7872/60000 [==>...........................] - ETA: 1:24 - loss: 0.5269 - categorical_accuracy: 0.8322
 7936/60000 [==>...........................] - ETA: 1:23 - loss: 0.5236 - categorical_accuracy: 0.8332
 8000/60000 [===>..........................] - ETA: 1:23 - loss: 0.5242 - categorical_accuracy: 0.8332
 8032/60000 [===>..........................] - ETA: 1:23 - loss: 0.5229 - categorical_accuracy: 0.8334
 8064/60000 [===>..........................] - ETA: 1:23 - loss: 0.5220 - categorical_accuracy: 0.8338
 8096/60000 [===>..........................] - ETA: 1:23 - loss: 0.5205 - categorical_accuracy: 0.8344
 8128/60000 [===>..........................] - ETA: 1:23 - loss: 0.5197 - categorical_accuracy: 0.8348
 8160/60000 [===>..........................] - ETA: 1:23 - loss: 0.5180 - categorical_accuracy: 0.8353
 8224/60000 [===>..........................] - ETA: 1:23 - loss: 0.5150 - categorical_accuracy: 0.8362
 8288/60000 [===>..........................] - ETA: 1:23 - loss: 0.5130 - categorical_accuracy: 0.8369
 8352/60000 [===>..........................] - ETA: 1:23 - loss: 0.5114 - categorical_accuracy: 0.8373
 8384/60000 [===>..........................] - ETA: 1:23 - loss: 0.5100 - categorical_accuracy: 0.8378
 8448/60000 [===>..........................] - ETA: 1:23 - loss: 0.5073 - categorical_accuracy: 0.8387
 8480/60000 [===>..........................] - ETA: 1:23 - loss: 0.5066 - categorical_accuracy: 0.8389
 8544/60000 [===>..........................] - ETA: 1:22 - loss: 0.5037 - categorical_accuracy: 0.8400
 8608/60000 [===>..........................] - ETA: 1:22 - loss: 0.5019 - categorical_accuracy: 0.8406
 8672/60000 [===>..........................] - ETA: 1:22 - loss: 0.5009 - categorical_accuracy: 0.8411
 8736/60000 [===>..........................] - ETA: 1:22 - loss: 0.4991 - categorical_accuracy: 0.8412
 8768/60000 [===>..........................] - ETA: 1:22 - loss: 0.4976 - categorical_accuracy: 0.8418
 8800/60000 [===>..........................] - ETA: 1:22 - loss: 0.4965 - categorical_accuracy: 0.8422
 8832/60000 [===>..........................] - ETA: 1:22 - loss: 0.4953 - categorical_accuracy: 0.8426
 8864/60000 [===>..........................] - ETA: 1:22 - loss: 0.4942 - categorical_accuracy: 0.8428
 8896/60000 [===>..........................] - ETA: 1:22 - loss: 0.4932 - categorical_accuracy: 0.8432
 8928/60000 [===>..........................] - ETA: 1:22 - loss: 0.4936 - categorical_accuracy: 0.8433
 8960/60000 [===>..........................] - ETA: 1:22 - loss: 0.4929 - categorical_accuracy: 0.8436
 8992/60000 [===>..........................] - ETA: 1:22 - loss: 0.4918 - categorical_accuracy: 0.8441
 9024/60000 [===>..........................] - ETA: 1:22 - loss: 0.4907 - categorical_accuracy: 0.8443
 9056/60000 [===>..........................] - ETA: 1:22 - loss: 0.4894 - categorical_accuracy: 0.8447
 9088/60000 [===>..........................] - ETA: 1:22 - loss: 0.4885 - categorical_accuracy: 0.8452
 9152/60000 [===>..........................] - ETA: 1:22 - loss: 0.4874 - categorical_accuracy: 0.8452
 9216/60000 [===>..........................] - ETA: 1:21 - loss: 0.4856 - categorical_accuracy: 0.8458
 9280/60000 [===>..........................] - ETA: 1:21 - loss: 0.4837 - categorical_accuracy: 0.8466
 9344/60000 [===>..........................] - ETA: 1:21 - loss: 0.4816 - categorical_accuracy: 0.8474
 9376/60000 [===>..........................] - ETA: 1:21 - loss: 0.4807 - categorical_accuracy: 0.8476
 9408/60000 [===>..........................] - ETA: 1:21 - loss: 0.4794 - categorical_accuracy: 0.8480
 9472/60000 [===>..........................] - ETA: 1:21 - loss: 0.4774 - categorical_accuracy: 0.8488
 9504/60000 [===>..........................] - ETA: 1:21 - loss: 0.4761 - categorical_accuracy: 0.8492
 9568/60000 [===>..........................] - ETA: 1:21 - loss: 0.4757 - categorical_accuracy: 0.8496
 9632/60000 [===>..........................] - ETA: 1:21 - loss: 0.4746 - categorical_accuracy: 0.8499
 9696/60000 [===>..........................] - ETA: 1:20 - loss: 0.4722 - categorical_accuracy: 0.8508
 9728/60000 [===>..........................] - ETA: 1:20 - loss: 0.4713 - categorical_accuracy: 0.8510
 9792/60000 [===>..........................] - ETA: 1:20 - loss: 0.4695 - categorical_accuracy: 0.8514
 9856/60000 [===>..........................] - ETA: 1:20 - loss: 0.4690 - categorical_accuracy: 0.8519
 9888/60000 [===>..........................] - ETA: 1:20 - loss: 0.4679 - categorical_accuracy: 0.8522
 9920/60000 [===>..........................] - ETA: 1:20 - loss: 0.4674 - categorical_accuracy: 0.8522
 9952/60000 [===>..........................] - ETA: 1:20 - loss: 0.4661 - categorical_accuracy: 0.8527
 9984/60000 [===>..........................] - ETA: 1:20 - loss: 0.4652 - categorical_accuracy: 0.8531
10048/60000 [====>.........................] - ETA: 1:20 - loss: 0.4644 - categorical_accuracy: 0.8534
10080/60000 [====>.........................] - ETA: 1:20 - loss: 0.4636 - categorical_accuracy: 0.8534
10144/60000 [====>.........................] - ETA: 1:20 - loss: 0.4616 - categorical_accuracy: 0.8540
10208/60000 [====>.........................] - ETA: 1:20 - loss: 0.4604 - categorical_accuracy: 0.8542
10272/60000 [====>.........................] - ETA: 1:19 - loss: 0.4590 - categorical_accuracy: 0.8546
10304/60000 [====>.........................] - ETA: 1:19 - loss: 0.4580 - categorical_accuracy: 0.8548
10368/60000 [====>.........................] - ETA: 1:19 - loss: 0.4559 - categorical_accuracy: 0.8554
10400/60000 [====>.........................] - ETA: 1:19 - loss: 0.4550 - categorical_accuracy: 0.8557
10432/60000 [====>.........................] - ETA: 1:19 - loss: 0.4541 - categorical_accuracy: 0.8560
10496/60000 [====>.........................] - ETA: 1:19 - loss: 0.4522 - categorical_accuracy: 0.8568
10560/60000 [====>.........................] - ETA: 1:19 - loss: 0.4508 - categorical_accuracy: 0.8572
10624/60000 [====>.........................] - ETA: 1:19 - loss: 0.4500 - categorical_accuracy: 0.8576
10656/60000 [====>.........................] - ETA: 1:19 - loss: 0.4489 - categorical_accuracy: 0.8579
10688/60000 [====>.........................] - ETA: 1:19 - loss: 0.4477 - categorical_accuracy: 0.8583
10720/60000 [====>.........................] - ETA: 1:19 - loss: 0.4473 - categorical_accuracy: 0.8585
10784/60000 [====>.........................] - ETA: 1:19 - loss: 0.4468 - categorical_accuracy: 0.8586
10816/60000 [====>.........................] - ETA: 1:19 - loss: 0.4464 - categorical_accuracy: 0.8588
10880/60000 [====>.........................] - ETA: 1:18 - loss: 0.4459 - categorical_accuracy: 0.8592
10944/60000 [====>.........................] - ETA: 1:18 - loss: 0.4442 - categorical_accuracy: 0.8598
10976/60000 [====>.........................] - ETA: 1:18 - loss: 0.4435 - categorical_accuracy: 0.8601
11008/60000 [====>.........................] - ETA: 1:18 - loss: 0.4434 - categorical_accuracy: 0.8602
11040/60000 [====>.........................] - ETA: 1:18 - loss: 0.4426 - categorical_accuracy: 0.8603
11072/60000 [====>.........................] - ETA: 1:18 - loss: 0.4426 - categorical_accuracy: 0.8605
11104/60000 [====>.........................] - ETA: 1:18 - loss: 0.4415 - categorical_accuracy: 0.8607
11136/60000 [====>.........................] - ETA: 1:18 - loss: 0.4415 - categorical_accuracy: 0.8608
11168/60000 [====>.........................] - ETA: 1:18 - loss: 0.4407 - categorical_accuracy: 0.8610
11200/60000 [====>.........................] - ETA: 1:18 - loss: 0.4396 - categorical_accuracy: 0.8613
11232/60000 [====>.........................] - ETA: 1:18 - loss: 0.4391 - categorical_accuracy: 0.8615
11264/60000 [====>.........................] - ETA: 1:18 - loss: 0.4382 - categorical_accuracy: 0.8618
11296/60000 [====>.........................] - ETA: 1:18 - loss: 0.4380 - categorical_accuracy: 0.8618
11328/60000 [====>.........................] - ETA: 1:18 - loss: 0.4373 - categorical_accuracy: 0.8620
11360/60000 [====>.........................] - ETA: 1:18 - loss: 0.4365 - categorical_accuracy: 0.8623
11424/60000 [====>.........................] - ETA: 1:18 - loss: 0.4353 - categorical_accuracy: 0.8627
11488/60000 [====>.........................] - ETA: 1:17 - loss: 0.4345 - categorical_accuracy: 0.8629
11552/60000 [====>.........................] - ETA: 1:17 - loss: 0.4329 - categorical_accuracy: 0.8635
11616/60000 [====>.........................] - ETA: 1:17 - loss: 0.4312 - categorical_accuracy: 0.8641
11648/60000 [====>.........................] - ETA: 1:17 - loss: 0.4304 - categorical_accuracy: 0.8644
11680/60000 [====>.........................] - ETA: 1:17 - loss: 0.4294 - categorical_accuracy: 0.8646
11744/60000 [====>.........................] - ETA: 1:17 - loss: 0.4278 - categorical_accuracy: 0.8650
11808/60000 [====>.........................] - ETA: 1:17 - loss: 0.4263 - categorical_accuracy: 0.8656
11872/60000 [====>.........................] - ETA: 1:17 - loss: 0.4254 - categorical_accuracy: 0.8658
11904/60000 [====>.........................] - ETA: 1:17 - loss: 0.4251 - categorical_accuracy: 0.8659
11936/60000 [====>.........................] - ETA: 1:17 - loss: 0.4247 - categorical_accuracy: 0.8660
11968/60000 [====>.........................] - ETA: 1:17 - loss: 0.4243 - categorical_accuracy: 0.8661
12000/60000 [=====>........................] - ETA: 1:17 - loss: 0.4233 - categorical_accuracy: 0.8665
12032/60000 [=====>........................] - ETA: 1:17 - loss: 0.4222 - categorical_accuracy: 0.8669
12064/60000 [=====>........................] - ETA: 1:17 - loss: 0.4213 - categorical_accuracy: 0.8672
12096/60000 [=====>........................] - ETA: 1:16 - loss: 0.4204 - categorical_accuracy: 0.8675
12128/60000 [=====>........................] - ETA: 1:16 - loss: 0.4195 - categorical_accuracy: 0.8677
12160/60000 [=====>........................] - ETA: 1:16 - loss: 0.4192 - categorical_accuracy: 0.8678
12192/60000 [=====>........................] - ETA: 1:16 - loss: 0.4190 - categorical_accuracy: 0.8679
12224/60000 [=====>........................] - ETA: 1:16 - loss: 0.4183 - categorical_accuracy: 0.8682
12288/60000 [=====>........................] - ETA: 1:16 - loss: 0.4168 - categorical_accuracy: 0.8686
12320/60000 [=====>........................] - ETA: 1:16 - loss: 0.4160 - categorical_accuracy: 0.8687
12352/60000 [=====>........................] - ETA: 1:16 - loss: 0.4153 - categorical_accuracy: 0.8690
12384/60000 [=====>........................] - ETA: 1:16 - loss: 0.4144 - categorical_accuracy: 0.8693
12448/60000 [=====>........................] - ETA: 1:16 - loss: 0.4141 - categorical_accuracy: 0.8696
12512/60000 [=====>........................] - ETA: 1:16 - loss: 0.4137 - categorical_accuracy: 0.8700
12544/60000 [=====>........................] - ETA: 1:16 - loss: 0.4131 - categorical_accuracy: 0.8703
12576/60000 [=====>........................] - ETA: 1:16 - loss: 0.4125 - categorical_accuracy: 0.8705
12608/60000 [=====>........................] - ETA: 1:16 - loss: 0.4118 - categorical_accuracy: 0.8708
12640/60000 [=====>........................] - ETA: 1:16 - loss: 0.4112 - categorical_accuracy: 0.8709
12704/60000 [=====>........................] - ETA: 1:16 - loss: 0.4106 - categorical_accuracy: 0.8711
12768/60000 [=====>........................] - ETA: 1:15 - loss: 0.4091 - categorical_accuracy: 0.8716
12800/60000 [=====>........................] - ETA: 1:15 - loss: 0.4082 - categorical_accuracy: 0.8720
12864/60000 [=====>........................] - ETA: 1:15 - loss: 0.4066 - categorical_accuracy: 0.8725
12928/60000 [=====>........................] - ETA: 1:15 - loss: 0.4048 - categorical_accuracy: 0.8731
12992/60000 [=====>........................] - ETA: 1:15 - loss: 0.4044 - categorical_accuracy: 0.8735
13024/60000 [=====>........................] - ETA: 1:15 - loss: 0.4040 - categorical_accuracy: 0.8735
13088/60000 [=====>........................] - ETA: 1:15 - loss: 0.4027 - categorical_accuracy: 0.8739
13120/60000 [=====>........................] - ETA: 1:15 - loss: 0.4022 - categorical_accuracy: 0.8740
13152/60000 [=====>........................] - ETA: 1:15 - loss: 0.4021 - categorical_accuracy: 0.8739
13184/60000 [=====>........................] - ETA: 1:15 - loss: 0.4013 - categorical_accuracy: 0.8742
13216/60000 [=====>........................] - ETA: 1:15 - loss: 0.4008 - categorical_accuracy: 0.8743
13248/60000 [=====>........................] - ETA: 1:15 - loss: 0.4005 - categorical_accuracy: 0.8744
13312/60000 [=====>........................] - ETA: 1:15 - loss: 0.3993 - categorical_accuracy: 0.8748
13344/60000 [=====>........................] - ETA: 1:14 - loss: 0.3989 - categorical_accuracy: 0.8750
13408/60000 [=====>........................] - ETA: 1:14 - loss: 0.3978 - categorical_accuracy: 0.8754
13472/60000 [=====>........................] - ETA: 1:14 - loss: 0.3964 - categorical_accuracy: 0.8760
13536/60000 [=====>........................] - ETA: 1:14 - loss: 0.3952 - categorical_accuracy: 0.8763
13568/60000 [=====>........................] - ETA: 1:14 - loss: 0.3950 - categorical_accuracy: 0.8764
13600/60000 [=====>........................] - ETA: 1:14 - loss: 0.3947 - categorical_accuracy: 0.8765
13664/60000 [=====>........................] - ETA: 1:14 - loss: 0.3935 - categorical_accuracy: 0.8768
13728/60000 [=====>........................] - ETA: 1:14 - loss: 0.3924 - categorical_accuracy: 0.8772
13792/60000 [=====>........................] - ETA: 1:14 - loss: 0.3916 - categorical_accuracy: 0.8775
13856/60000 [=====>........................] - ETA: 1:14 - loss: 0.3902 - categorical_accuracy: 0.8780
13920/60000 [=====>........................] - ETA: 1:13 - loss: 0.3893 - categorical_accuracy: 0.8782
13952/60000 [=====>........................] - ETA: 1:13 - loss: 0.3886 - categorical_accuracy: 0.8784
14016/60000 [======>.......................] - ETA: 1:13 - loss: 0.3886 - categorical_accuracy: 0.8785
14048/60000 [======>.......................] - ETA: 1:13 - loss: 0.3883 - categorical_accuracy: 0.8786
14112/60000 [======>.......................] - ETA: 1:13 - loss: 0.3869 - categorical_accuracy: 0.8790
14144/60000 [======>.......................] - ETA: 1:13 - loss: 0.3865 - categorical_accuracy: 0.8791
14176/60000 [======>.......................] - ETA: 1:13 - loss: 0.3859 - categorical_accuracy: 0.8792
14208/60000 [======>.......................] - ETA: 1:13 - loss: 0.3854 - categorical_accuracy: 0.8793
14240/60000 [======>.......................] - ETA: 1:13 - loss: 0.3858 - categorical_accuracy: 0.8795
14304/60000 [======>.......................] - ETA: 1:13 - loss: 0.3851 - categorical_accuracy: 0.8796
14368/60000 [======>.......................] - ETA: 1:13 - loss: 0.3837 - categorical_accuracy: 0.8800
14400/60000 [======>.......................] - ETA: 1:13 - loss: 0.3833 - categorical_accuracy: 0.8801
14432/60000 [======>.......................] - ETA: 1:13 - loss: 0.3827 - categorical_accuracy: 0.8803
14464/60000 [======>.......................] - ETA: 1:13 - loss: 0.3823 - categorical_accuracy: 0.8804
14528/60000 [======>.......................] - ETA: 1:12 - loss: 0.3812 - categorical_accuracy: 0.8808
14560/60000 [======>.......................] - ETA: 1:12 - loss: 0.3805 - categorical_accuracy: 0.8810
14592/60000 [======>.......................] - ETA: 1:12 - loss: 0.3798 - categorical_accuracy: 0.8812
14624/60000 [======>.......................] - ETA: 1:12 - loss: 0.3800 - categorical_accuracy: 0.8812
14688/60000 [======>.......................] - ETA: 1:12 - loss: 0.3797 - categorical_accuracy: 0.8815
14752/60000 [======>.......................] - ETA: 1:12 - loss: 0.3791 - categorical_accuracy: 0.8818
14816/60000 [======>.......................] - ETA: 1:12 - loss: 0.3777 - categorical_accuracy: 0.8823
14848/60000 [======>.......................] - ETA: 1:12 - loss: 0.3774 - categorical_accuracy: 0.8824
14880/60000 [======>.......................] - ETA: 1:12 - loss: 0.3768 - categorical_accuracy: 0.8827
14912/60000 [======>.......................] - ETA: 1:12 - loss: 0.3762 - categorical_accuracy: 0.8828
14944/60000 [======>.......................] - ETA: 1:12 - loss: 0.3759 - categorical_accuracy: 0.8829
14976/60000 [======>.......................] - ETA: 1:12 - loss: 0.3758 - categorical_accuracy: 0.8829
15008/60000 [======>.......................] - ETA: 1:12 - loss: 0.3753 - categorical_accuracy: 0.8831
15040/60000 [======>.......................] - ETA: 1:12 - loss: 0.3746 - categorical_accuracy: 0.8833
15072/60000 [======>.......................] - ETA: 1:12 - loss: 0.3740 - categorical_accuracy: 0.8835
15136/60000 [======>.......................] - ETA: 1:11 - loss: 0.3732 - categorical_accuracy: 0.8838
15168/60000 [======>.......................] - ETA: 1:11 - loss: 0.3730 - categorical_accuracy: 0.8839
15200/60000 [======>.......................] - ETA: 1:11 - loss: 0.3726 - categorical_accuracy: 0.8839
15232/60000 [======>.......................] - ETA: 1:11 - loss: 0.3720 - categorical_accuracy: 0.8842
15264/60000 [======>.......................] - ETA: 1:11 - loss: 0.3713 - categorical_accuracy: 0.8844
15296/60000 [======>.......................] - ETA: 1:11 - loss: 0.3713 - categorical_accuracy: 0.8844
15328/60000 [======>.......................] - ETA: 1:11 - loss: 0.3709 - categorical_accuracy: 0.8845
15360/60000 [======>.......................] - ETA: 1:11 - loss: 0.3704 - categorical_accuracy: 0.8847
15392/60000 [======>.......................] - ETA: 1:11 - loss: 0.3700 - categorical_accuracy: 0.8847
15424/60000 [======>.......................] - ETA: 1:11 - loss: 0.3697 - categorical_accuracy: 0.8849
15456/60000 [======>.......................] - ETA: 1:11 - loss: 0.3690 - categorical_accuracy: 0.8851
15488/60000 [======>.......................] - ETA: 1:11 - loss: 0.3687 - categorical_accuracy: 0.8851
15520/60000 [======>.......................] - ETA: 1:11 - loss: 0.3684 - categorical_accuracy: 0.8852
15584/60000 [======>.......................] - ETA: 1:11 - loss: 0.3675 - categorical_accuracy: 0.8855
15616/60000 [======>.......................] - ETA: 1:11 - loss: 0.3670 - categorical_accuracy: 0.8856
15648/60000 [======>.......................] - ETA: 1:11 - loss: 0.3668 - categorical_accuracy: 0.8856
15680/60000 [======>.......................] - ETA: 1:11 - loss: 0.3662 - categorical_accuracy: 0.8858
15712/60000 [======>.......................] - ETA: 1:11 - loss: 0.3658 - categorical_accuracy: 0.8860
15744/60000 [======>.......................] - ETA: 1:11 - loss: 0.3654 - categorical_accuracy: 0.8861
15776/60000 [======>.......................] - ETA: 1:11 - loss: 0.3649 - categorical_accuracy: 0.8863
15808/60000 [======>.......................] - ETA: 1:11 - loss: 0.3643 - categorical_accuracy: 0.8865
15872/60000 [======>.......................] - ETA: 1:10 - loss: 0.3644 - categorical_accuracy: 0.8866
15904/60000 [======>.......................] - ETA: 1:10 - loss: 0.3640 - categorical_accuracy: 0.8868
15968/60000 [======>.......................] - ETA: 1:10 - loss: 0.3630 - categorical_accuracy: 0.8871
16000/60000 [=======>......................] - ETA: 1:10 - loss: 0.3623 - categorical_accuracy: 0.8873
16032/60000 [=======>......................] - ETA: 1:10 - loss: 0.3618 - categorical_accuracy: 0.8873
16064/60000 [=======>......................] - ETA: 1:10 - loss: 0.3613 - categorical_accuracy: 0.8874
16096/60000 [=======>......................] - ETA: 1:10 - loss: 0.3607 - categorical_accuracy: 0.8876
16128/60000 [=======>......................] - ETA: 1:10 - loss: 0.3600 - categorical_accuracy: 0.8878
16160/60000 [=======>......................] - ETA: 1:10 - loss: 0.3594 - categorical_accuracy: 0.8881
16192/60000 [=======>......................] - ETA: 1:10 - loss: 0.3591 - categorical_accuracy: 0.8881
16224/60000 [=======>......................] - ETA: 1:10 - loss: 0.3584 - categorical_accuracy: 0.8883
16256/60000 [=======>......................] - ETA: 1:10 - loss: 0.3579 - categorical_accuracy: 0.8884
16288/60000 [=======>......................] - ETA: 1:10 - loss: 0.3579 - categorical_accuracy: 0.8884
16320/60000 [=======>......................] - ETA: 1:10 - loss: 0.3575 - categorical_accuracy: 0.8885
16384/60000 [=======>......................] - ETA: 1:10 - loss: 0.3565 - categorical_accuracy: 0.8887
16448/60000 [=======>......................] - ETA: 1:10 - loss: 0.3557 - categorical_accuracy: 0.8890
16480/60000 [=======>......................] - ETA: 1:10 - loss: 0.3551 - categorical_accuracy: 0.8892
16512/60000 [=======>......................] - ETA: 1:10 - loss: 0.3545 - categorical_accuracy: 0.8894
16544/60000 [=======>......................] - ETA: 1:09 - loss: 0.3544 - categorical_accuracy: 0.8893
16576/60000 [=======>......................] - ETA: 1:09 - loss: 0.3539 - categorical_accuracy: 0.8894
16608/60000 [=======>......................] - ETA: 1:09 - loss: 0.3537 - categorical_accuracy: 0.8896
16640/60000 [=======>......................] - ETA: 1:09 - loss: 0.3533 - categorical_accuracy: 0.8896
16672/60000 [=======>......................] - ETA: 1:09 - loss: 0.3529 - categorical_accuracy: 0.8898
16704/60000 [=======>......................] - ETA: 1:09 - loss: 0.3524 - categorical_accuracy: 0.8899
16768/60000 [=======>......................] - ETA: 1:09 - loss: 0.3515 - categorical_accuracy: 0.8902
16832/60000 [=======>......................] - ETA: 1:09 - loss: 0.3508 - categorical_accuracy: 0.8905
16864/60000 [=======>......................] - ETA: 1:09 - loss: 0.3502 - categorical_accuracy: 0.8907
16896/60000 [=======>......................] - ETA: 1:09 - loss: 0.3496 - categorical_accuracy: 0.8909
16960/60000 [=======>......................] - ETA: 1:09 - loss: 0.3488 - categorical_accuracy: 0.8912
16992/60000 [=======>......................] - ETA: 1:09 - loss: 0.3484 - categorical_accuracy: 0.8914
17024/60000 [=======>......................] - ETA: 1:09 - loss: 0.3479 - categorical_accuracy: 0.8915
17056/60000 [=======>......................] - ETA: 1:09 - loss: 0.3473 - categorical_accuracy: 0.8917
17088/60000 [=======>......................] - ETA: 1:09 - loss: 0.3468 - categorical_accuracy: 0.8919
17152/60000 [=======>......................] - ETA: 1:09 - loss: 0.3458 - categorical_accuracy: 0.8922
17184/60000 [=======>......................] - ETA: 1:08 - loss: 0.3454 - categorical_accuracy: 0.8923
17248/60000 [=======>......................] - ETA: 1:08 - loss: 0.3448 - categorical_accuracy: 0.8926
17280/60000 [=======>......................] - ETA: 1:08 - loss: 0.3444 - categorical_accuracy: 0.8927
17312/60000 [=======>......................] - ETA: 1:08 - loss: 0.3439 - categorical_accuracy: 0.8928
17344/60000 [=======>......................] - ETA: 1:08 - loss: 0.3435 - categorical_accuracy: 0.8929
17376/60000 [=======>......................] - ETA: 1:08 - loss: 0.3434 - categorical_accuracy: 0.8930
17408/60000 [=======>......................] - ETA: 1:08 - loss: 0.3428 - categorical_accuracy: 0.8932
17440/60000 [=======>......................] - ETA: 1:08 - loss: 0.3424 - categorical_accuracy: 0.8933
17472/60000 [=======>......................] - ETA: 1:08 - loss: 0.3424 - categorical_accuracy: 0.8933
17504/60000 [=======>......................] - ETA: 1:08 - loss: 0.3419 - categorical_accuracy: 0.8934
17536/60000 [=======>......................] - ETA: 1:08 - loss: 0.3414 - categorical_accuracy: 0.8936
17568/60000 [=======>......................] - ETA: 1:08 - loss: 0.3411 - categorical_accuracy: 0.8936
17632/60000 [=======>......................] - ETA: 1:08 - loss: 0.3406 - categorical_accuracy: 0.8938
17664/60000 [=======>......................] - ETA: 1:08 - loss: 0.3401 - categorical_accuracy: 0.8940
17696/60000 [=======>......................] - ETA: 1:08 - loss: 0.3397 - categorical_accuracy: 0.8940
17728/60000 [=======>......................] - ETA: 1:08 - loss: 0.3393 - categorical_accuracy: 0.8941
17760/60000 [=======>......................] - ETA: 1:08 - loss: 0.3392 - categorical_accuracy: 0.8941
17792/60000 [=======>......................] - ETA: 1:08 - loss: 0.3390 - categorical_accuracy: 0.8942
17824/60000 [=======>......................] - ETA: 1:07 - loss: 0.3385 - categorical_accuracy: 0.8944
17856/60000 [=======>......................] - ETA: 1:07 - loss: 0.3379 - categorical_accuracy: 0.8946
17888/60000 [=======>......................] - ETA: 1:07 - loss: 0.3377 - categorical_accuracy: 0.8946
17920/60000 [=======>......................] - ETA: 1:07 - loss: 0.3372 - categorical_accuracy: 0.8948
17984/60000 [=======>......................] - ETA: 1:07 - loss: 0.3364 - categorical_accuracy: 0.8950
18016/60000 [========>.....................] - ETA: 1:07 - loss: 0.3366 - categorical_accuracy: 0.8950
18048/60000 [========>.....................] - ETA: 1:07 - loss: 0.3362 - categorical_accuracy: 0.8952
18080/60000 [========>.....................] - ETA: 1:07 - loss: 0.3361 - categorical_accuracy: 0.8952
18112/60000 [========>.....................] - ETA: 1:07 - loss: 0.3356 - categorical_accuracy: 0.8954
18144/60000 [========>.....................] - ETA: 1:07 - loss: 0.3350 - categorical_accuracy: 0.8956
18176/60000 [========>.....................] - ETA: 1:07 - loss: 0.3348 - categorical_accuracy: 0.8957
18208/60000 [========>.....................] - ETA: 1:07 - loss: 0.3345 - categorical_accuracy: 0.8958
18240/60000 [========>.....................] - ETA: 1:07 - loss: 0.3340 - categorical_accuracy: 0.8959
18272/60000 [========>.....................] - ETA: 1:07 - loss: 0.3342 - categorical_accuracy: 0.8959
18304/60000 [========>.....................] - ETA: 1:07 - loss: 0.3344 - categorical_accuracy: 0.8959
18336/60000 [========>.....................] - ETA: 1:07 - loss: 0.3340 - categorical_accuracy: 0.8959
18368/60000 [========>.....................] - ETA: 1:07 - loss: 0.3340 - categorical_accuracy: 0.8960
18400/60000 [========>.....................] - ETA: 1:07 - loss: 0.3334 - categorical_accuracy: 0.8961
18432/60000 [========>.....................] - ETA: 1:07 - loss: 0.3332 - categorical_accuracy: 0.8961
18464/60000 [========>.....................] - ETA: 1:07 - loss: 0.3327 - categorical_accuracy: 0.8963
18496/60000 [========>.....................] - ETA: 1:07 - loss: 0.3323 - categorical_accuracy: 0.8965
18528/60000 [========>.....................] - ETA: 1:06 - loss: 0.3321 - categorical_accuracy: 0.8966
18592/60000 [========>.....................] - ETA: 1:06 - loss: 0.3317 - categorical_accuracy: 0.8966
18624/60000 [========>.....................] - ETA: 1:06 - loss: 0.3313 - categorical_accuracy: 0.8967
18656/60000 [========>.....................] - ETA: 1:06 - loss: 0.3310 - categorical_accuracy: 0.8969
18688/60000 [========>.....................] - ETA: 1:06 - loss: 0.3308 - categorical_accuracy: 0.8969
18752/60000 [========>.....................] - ETA: 1:06 - loss: 0.3305 - categorical_accuracy: 0.8971
18784/60000 [========>.....................] - ETA: 1:06 - loss: 0.3306 - categorical_accuracy: 0.8971
18848/60000 [========>.....................] - ETA: 1:06 - loss: 0.3298 - categorical_accuracy: 0.8974
18912/60000 [========>.....................] - ETA: 1:06 - loss: 0.3293 - categorical_accuracy: 0.8976
18944/60000 [========>.....................] - ETA: 1:06 - loss: 0.3291 - categorical_accuracy: 0.8976
18976/60000 [========>.....................] - ETA: 1:06 - loss: 0.3290 - categorical_accuracy: 0.8977
19008/60000 [========>.....................] - ETA: 1:06 - loss: 0.3289 - categorical_accuracy: 0.8977
19040/60000 [========>.....................] - ETA: 1:06 - loss: 0.3286 - categorical_accuracy: 0.8978
19072/60000 [========>.....................] - ETA: 1:06 - loss: 0.3280 - categorical_accuracy: 0.8980
19136/60000 [========>.....................] - ETA: 1:05 - loss: 0.3281 - categorical_accuracy: 0.8980
19168/60000 [========>.....................] - ETA: 1:05 - loss: 0.3280 - categorical_accuracy: 0.8981
19200/60000 [========>.....................] - ETA: 1:05 - loss: 0.3278 - categorical_accuracy: 0.8981
19232/60000 [========>.....................] - ETA: 1:05 - loss: 0.3277 - categorical_accuracy: 0.8982
19264/60000 [========>.....................] - ETA: 1:05 - loss: 0.3272 - categorical_accuracy: 0.8984
19296/60000 [========>.....................] - ETA: 1:05 - loss: 0.3269 - categorical_accuracy: 0.8984
19328/60000 [========>.....................] - ETA: 1:05 - loss: 0.3265 - categorical_accuracy: 0.8986
19360/60000 [========>.....................] - ETA: 1:05 - loss: 0.3260 - categorical_accuracy: 0.8988
19424/60000 [========>.....................] - ETA: 1:05 - loss: 0.3253 - categorical_accuracy: 0.8989
19456/60000 [========>.....................] - ETA: 1:05 - loss: 0.3251 - categorical_accuracy: 0.8990
19520/60000 [========>.....................] - ETA: 1:05 - loss: 0.3242 - categorical_accuracy: 0.8993
19552/60000 [========>.....................] - ETA: 1:05 - loss: 0.3238 - categorical_accuracy: 0.8994
19584/60000 [========>.....................] - ETA: 1:05 - loss: 0.3233 - categorical_accuracy: 0.8996
19616/60000 [========>.....................] - ETA: 1:05 - loss: 0.3236 - categorical_accuracy: 0.8995
19648/60000 [========>.....................] - ETA: 1:05 - loss: 0.3232 - categorical_accuracy: 0.8996
19680/60000 [========>.....................] - ETA: 1:05 - loss: 0.3231 - categorical_accuracy: 0.8996
19712/60000 [========>.....................] - ETA: 1:05 - loss: 0.3231 - categorical_accuracy: 0.8996
19744/60000 [========>.....................] - ETA: 1:05 - loss: 0.3227 - categorical_accuracy: 0.8998
19776/60000 [========>.....................] - ETA: 1:05 - loss: 0.3223 - categorical_accuracy: 0.8998
19808/60000 [========>.....................] - ETA: 1:05 - loss: 0.3221 - categorical_accuracy: 0.8998
19840/60000 [========>.....................] - ETA: 1:04 - loss: 0.3218 - categorical_accuracy: 0.8999
19872/60000 [========>.....................] - ETA: 1:04 - loss: 0.3213 - categorical_accuracy: 0.9001
19904/60000 [========>.....................] - ETA: 1:04 - loss: 0.3209 - categorical_accuracy: 0.9002
19936/60000 [========>.....................] - ETA: 1:04 - loss: 0.3205 - categorical_accuracy: 0.9003
19968/60000 [========>.....................] - ETA: 1:04 - loss: 0.3201 - categorical_accuracy: 0.9005
20000/60000 [=========>....................] - ETA: 1:04 - loss: 0.3197 - categorical_accuracy: 0.9007
20032/60000 [=========>....................] - ETA: 1:04 - loss: 0.3197 - categorical_accuracy: 0.9007
20064/60000 [=========>....................] - ETA: 1:04 - loss: 0.3196 - categorical_accuracy: 0.9007
20096/60000 [=========>....................] - ETA: 1:04 - loss: 0.3197 - categorical_accuracy: 0.9007
20128/60000 [=========>....................] - ETA: 1:04 - loss: 0.3194 - categorical_accuracy: 0.9008
20160/60000 [=========>....................] - ETA: 1:04 - loss: 0.3192 - categorical_accuracy: 0.9008
20192/60000 [=========>....................] - ETA: 1:04 - loss: 0.3187 - categorical_accuracy: 0.9010
20224/60000 [=========>....................] - ETA: 1:04 - loss: 0.3186 - categorical_accuracy: 0.9010
20256/60000 [=========>....................] - ETA: 1:04 - loss: 0.3189 - categorical_accuracy: 0.9010
20288/60000 [=========>....................] - ETA: 1:04 - loss: 0.3188 - categorical_accuracy: 0.9011
20320/60000 [=========>....................] - ETA: 1:04 - loss: 0.3186 - categorical_accuracy: 0.9011
20352/60000 [=========>....................] - ETA: 1:04 - loss: 0.3185 - categorical_accuracy: 0.9012
20384/60000 [=========>....................] - ETA: 1:04 - loss: 0.3182 - categorical_accuracy: 0.9013
20416/60000 [=========>....................] - ETA: 1:04 - loss: 0.3179 - categorical_accuracy: 0.9014
20448/60000 [=========>....................] - ETA: 1:04 - loss: 0.3175 - categorical_accuracy: 0.9016
20480/60000 [=========>....................] - ETA: 1:04 - loss: 0.3173 - categorical_accuracy: 0.9016
20512/60000 [=========>....................] - ETA: 1:03 - loss: 0.3173 - categorical_accuracy: 0.9016
20544/60000 [=========>....................] - ETA: 1:03 - loss: 0.3173 - categorical_accuracy: 0.9015
20576/60000 [=========>....................] - ETA: 1:03 - loss: 0.3171 - categorical_accuracy: 0.9016
20608/60000 [=========>....................] - ETA: 1:03 - loss: 0.3168 - categorical_accuracy: 0.9017
20640/60000 [=========>....................] - ETA: 1:03 - loss: 0.3163 - categorical_accuracy: 0.9018
20672/60000 [=========>....................] - ETA: 1:03 - loss: 0.3166 - categorical_accuracy: 0.9018
20704/60000 [=========>....................] - ETA: 1:03 - loss: 0.3166 - categorical_accuracy: 0.9018
20736/60000 [=========>....................] - ETA: 1:03 - loss: 0.3164 - categorical_accuracy: 0.9018
20768/60000 [=========>....................] - ETA: 1:03 - loss: 0.3162 - categorical_accuracy: 0.9019
20800/60000 [=========>....................] - ETA: 1:03 - loss: 0.3161 - categorical_accuracy: 0.9019
20832/60000 [=========>....................] - ETA: 1:03 - loss: 0.3161 - categorical_accuracy: 0.9019
20864/60000 [=========>....................] - ETA: 1:03 - loss: 0.3158 - categorical_accuracy: 0.9021
20896/60000 [=========>....................] - ETA: 1:03 - loss: 0.3160 - categorical_accuracy: 0.9020
20928/60000 [=========>....................] - ETA: 1:03 - loss: 0.3157 - categorical_accuracy: 0.9021
20960/60000 [=========>....................] - ETA: 1:03 - loss: 0.3152 - categorical_accuracy: 0.9023
20992/60000 [=========>....................] - ETA: 1:03 - loss: 0.3151 - categorical_accuracy: 0.9023
21024/60000 [=========>....................] - ETA: 1:03 - loss: 0.3150 - categorical_accuracy: 0.9023
21056/60000 [=========>....................] - ETA: 1:03 - loss: 0.3147 - categorical_accuracy: 0.9024
21088/60000 [=========>....................] - ETA: 1:03 - loss: 0.3145 - categorical_accuracy: 0.9025
21120/60000 [=========>....................] - ETA: 1:03 - loss: 0.3142 - categorical_accuracy: 0.9026
21152/60000 [=========>....................] - ETA: 1:03 - loss: 0.3138 - categorical_accuracy: 0.9028
21184/60000 [=========>....................] - ETA: 1:03 - loss: 0.3135 - categorical_accuracy: 0.9029
21216/60000 [=========>....................] - ETA: 1:02 - loss: 0.3130 - categorical_accuracy: 0.9030
21248/60000 [=========>....................] - ETA: 1:02 - loss: 0.3128 - categorical_accuracy: 0.9031
21280/60000 [=========>....................] - ETA: 1:02 - loss: 0.3125 - categorical_accuracy: 0.9032
21312/60000 [=========>....................] - ETA: 1:02 - loss: 0.3121 - categorical_accuracy: 0.9033
21344/60000 [=========>....................] - ETA: 1:02 - loss: 0.3121 - categorical_accuracy: 0.9033
21376/60000 [=========>....................] - ETA: 1:02 - loss: 0.3118 - categorical_accuracy: 0.9034
21408/60000 [=========>....................] - ETA: 1:02 - loss: 0.3116 - categorical_accuracy: 0.9034
21440/60000 [=========>....................] - ETA: 1:02 - loss: 0.3112 - categorical_accuracy: 0.9036
21472/60000 [=========>....................] - ETA: 1:02 - loss: 0.3109 - categorical_accuracy: 0.9037
21504/60000 [=========>....................] - ETA: 1:02 - loss: 0.3105 - categorical_accuracy: 0.9038
21536/60000 [=========>....................] - ETA: 1:02 - loss: 0.3105 - categorical_accuracy: 0.9039
21568/60000 [=========>....................] - ETA: 1:02 - loss: 0.3104 - categorical_accuracy: 0.9038
21600/60000 [=========>....................] - ETA: 1:02 - loss: 0.3102 - categorical_accuracy: 0.9038
21632/60000 [=========>....................] - ETA: 1:02 - loss: 0.3099 - categorical_accuracy: 0.9040
21664/60000 [=========>....................] - ETA: 1:02 - loss: 0.3095 - categorical_accuracy: 0.9041
21696/60000 [=========>....................] - ETA: 1:02 - loss: 0.3091 - categorical_accuracy: 0.9043
21728/60000 [=========>....................] - ETA: 1:02 - loss: 0.3088 - categorical_accuracy: 0.9044
21760/60000 [=========>....................] - ETA: 1:02 - loss: 0.3085 - categorical_accuracy: 0.9044
21792/60000 [=========>....................] - ETA: 1:02 - loss: 0.3086 - categorical_accuracy: 0.9044
21824/60000 [=========>....................] - ETA: 1:02 - loss: 0.3085 - categorical_accuracy: 0.9044
21856/60000 [=========>....................] - ETA: 1:02 - loss: 0.3081 - categorical_accuracy: 0.9045
21888/60000 [=========>....................] - ETA: 1:01 - loss: 0.3078 - categorical_accuracy: 0.9045
21920/60000 [=========>....................] - ETA: 1:01 - loss: 0.3074 - categorical_accuracy: 0.9047
21952/60000 [=========>....................] - ETA: 1:01 - loss: 0.3070 - categorical_accuracy: 0.9048
21984/60000 [=========>....................] - ETA: 1:01 - loss: 0.3067 - categorical_accuracy: 0.9049
22016/60000 [==========>...................] - ETA: 1:01 - loss: 0.3064 - categorical_accuracy: 0.9051
22080/60000 [==========>...................] - ETA: 1:01 - loss: 0.3063 - categorical_accuracy: 0.9051
22144/60000 [==========>...................] - ETA: 1:01 - loss: 0.3061 - categorical_accuracy: 0.9051
22208/60000 [==========>...................] - ETA: 1:01 - loss: 0.3056 - categorical_accuracy: 0.9052
22272/60000 [==========>...................] - ETA: 1:01 - loss: 0.3054 - categorical_accuracy: 0.9053
22304/60000 [==========>...................] - ETA: 1:01 - loss: 0.3052 - categorical_accuracy: 0.9054
22336/60000 [==========>...................] - ETA: 1:01 - loss: 0.3050 - categorical_accuracy: 0.9055
22368/60000 [==========>...................] - ETA: 1:01 - loss: 0.3050 - categorical_accuracy: 0.9055
22400/60000 [==========>...................] - ETA: 1:01 - loss: 0.3048 - categorical_accuracy: 0.9055
22432/60000 [==========>...................] - ETA: 1:01 - loss: 0.3046 - categorical_accuracy: 0.9055
22464/60000 [==========>...................] - ETA: 1:01 - loss: 0.3044 - categorical_accuracy: 0.9055
22496/60000 [==========>...................] - ETA: 1:00 - loss: 0.3043 - categorical_accuracy: 0.9056
22528/60000 [==========>...................] - ETA: 1:00 - loss: 0.3040 - categorical_accuracy: 0.9058
22592/60000 [==========>...................] - ETA: 1:00 - loss: 0.3034 - categorical_accuracy: 0.9059
22624/60000 [==========>...................] - ETA: 1:00 - loss: 0.3030 - categorical_accuracy: 0.9060
22688/60000 [==========>...................] - ETA: 1:00 - loss: 0.3028 - categorical_accuracy: 0.9061
22752/60000 [==========>...................] - ETA: 1:00 - loss: 0.3024 - categorical_accuracy: 0.9062
22784/60000 [==========>...................] - ETA: 1:00 - loss: 0.3020 - categorical_accuracy: 0.9064
22816/60000 [==========>...................] - ETA: 1:00 - loss: 0.3017 - categorical_accuracy: 0.9065
22848/60000 [==========>...................] - ETA: 1:00 - loss: 0.3013 - categorical_accuracy: 0.9066
22880/60000 [==========>...................] - ETA: 1:00 - loss: 0.3014 - categorical_accuracy: 0.9066
22912/60000 [==========>...................] - ETA: 1:00 - loss: 0.3011 - categorical_accuracy: 0.9067
22944/60000 [==========>...................] - ETA: 1:00 - loss: 0.3007 - categorical_accuracy: 0.9069
22976/60000 [==========>...................] - ETA: 1:00 - loss: 0.3004 - categorical_accuracy: 0.9070
23008/60000 [==========>...................] - ETA: 1:00 - loss: 0.3005 - categorical_accuracy: 0.9070
23040/60000 [==========>...................] - ETA: 1:00 - loss: 0.3005 - categorical_accuracy: 0.9070
23072/60000 [==========>...................] - ETA: 1:00 - loss: 0.3002 - categorical_accuracy: 0.9071
23104/60000 [==========>...................] - ETA: 59s - loss: 0.3000 - categorical_accuracy: 0.9072 
23136/60000 [==========>...................] - ETA: 59s - loss: 0.2997 - categorical_accuracy: 0.9072
23200/60000 [==========>...................] - ETA: 59s - loss: 0.2993 - categorical_accuracy: 0.9074
23232/60000 [==========>...................] - ETA: 59s - loss: 0.2989 - categorical_accuracy: 0.9075
23264/60000 [==========>...................] - ETA: 59s - loss: 0.2985 - categorical_accuracy: 0.9076
23328/60000 [==========>...................] - ETA: 59s - loss: 0.2982 - categorical_accuracy: 0.9077
23360/60000 [==========>...................] - ETA: 59s - loss: 0.2979 - categorical_accuracy: 0.9078
23392/60000 [==========>...................] - ETA: 59s - loss: 0.2976 - categorical_accuracy: 0.9079
23424/60000 [==========>...................] - ETA: 59s - loss: 0.2972 - categorical_accuracy: 0.9080
23456/60000 [==========>...................] - ETA: 59s - loss: 0.2970 - categorical_accuracy: 0.9080
23488/60000 [==========>...................] - ETA: 59s - loss: 0.2967 - categorical_accuracy: 0.9082
23520/60000 [==========>...................] - ETA: 59s - loss: 0.2964 - categorical_accuracy: 0.9082
23552/60000 [==========>...................] - ETA: 59s - loss: 0.2961 - categorical_accuracy: 0.9084
23584/60000 [==========>...................] - ETA: 59s - loss: 0.2958 - categorical_accuracy: 0.9085
23616/60000 [==========>...................] - ETA: 59s - loss: 0.2956 - categorical_accuracy: 0.9086
23648/60000 [==========>...................] - ETA: 59s - loss: 0.2952 - categorical_accuracy: 0.9087
23680/60000 [==========>...................] - ETA: 59s - loss: 0.2953 - categorical_accuracy: 0.9087
23712/60000 [==========>...................] - ETA: 59s - loss: 0.2951 - categorical_accuracy: 0.9088
23744/60000 [==========>...................] - ETA: 59s - loss: 0.2947 - categorical_accuracy: 0.9089
23776/60000 [==========>...................] - ETA: 58s - loss: 0.2945 - categorical_accuracy: 0.9089
23808/60000 [==========>...................] - ETA: 58s - loss: 0.2943 - categorical_accuracy: 0.9090
23840/60000 [==========>...................] - ETA: 58s - loss: 0.2941 - categorical_accuracy: 0.9091
23872/60000 [==========>...................] - ETA: 58s - loss: 0.2940 - categorical_accuracy: 0.9091
23904/60000 [==========>...................] - ETA: 58s - loss: 0.2939 - categorical_accuracy: 0.9091
23936/60000 [==========>...................] - ETA: 58s - loss: 0.2937 - categorical_accuracy: 0.9091
23968/60000 [==========>...................] - ETA: 58s - loss: 0.2934 - categorical_accuracy: 0.9092
24000/60000 [===========>..................] - ETA: 58s - loss: 0.2931 - categorical_accuracy: 0.9093
24032/60000 [===========>..................] - ETA: 58s - loss: 0.2930 - categorical_accuracy: 0.9093
24064/60000 [===========>..................] - ETA: 58s - loss: 0.2928 - categorical_accuracy: 0.9094
24096/60000 [===========>..................] - ETA: 58s - loss: 0.2925 - categorical_accuracy: 0.9095
24128/60000 [===========>..................] - ETA: 58s - loss: 0.2923 - categorical_accuracy: 0.9095
24160/60000 [===========>..................] - ETA: 58s - loss: 0.2920 - categorical_accuracy: 0.9096
24192/60000 [===========>..................] - ETA: 58s - loss: 0.2917 - categorical_accuracy: 0.9097
24224/60000 [===========>..................] - ETA: 58s - loss: 0.2913 - categorical_accuracy: 0.9098
24256/60000 [===========>..................] - ETA: 58s - loss: 0.2910 - categorical_accuracy: 0.9100
24288/60000 [===========>..................] - ETA: 58s - loss: 0.2907 - categorical_accuracy: 0.9100
24320/60000 [===========>..................] - ETA: 58s - loss: 0.2904 - categorical_accuracy: 0.9101
24352/60000 [===========>..................] - ETA: 58s - loss: 0.2901 - categorical_accuracy: 0.9102
24384/60000 [===========>..................] - ETA: 58s - loss: 0.2899 - categorical_accuracy: 0.9102
24416/60000 [===========>..................] - ETA: 57s - loss: 0.2896 - categorical_accuracy: 0.9103
24448/60000 [===========>..................] - ETA: 57s - loss: 0.2893 - categorical_accuracy: 0.9103
24480/60000 [===========>..................] - ETA: 57s - loss: 0.2891 - categorical_accuracy: 0.9104
24512/60000 [===========>..................] - ETA: 57s - loss: 0.2888 - categorical_accuracy: 0.9105
24544/60000 [===========>..................] - ETA: 57s - loss: 0.2886 - categorical_accuracy: 0.9106
24576/60000 [===========>..................] - ETA: 57s - loss: 0.2882 - categorical_accuracy: 0.9107
24608/60000 [===========>..................] - ETA: 57s - loss: 0.2880 - categorical_accuracy: 0.9108
24640/60000 [===========>..................] - ETA: 57s - loss: 0.2878 - categorical_accuracy: 0.9108
24672/60000 [===========>..................] - ETA: 57s - loss: 0.2879 - categorical_accuracy: 0.9109
24704/60000 [===========>..................] - ETA: 57s - loss: 0.2880 - categorical_accuracy: 0.9109
24736/60000 [===========>..................] - ETA: 57s - loss: 0.2878 - categorical_accuracy: 0.9110
24768/60000 [===========>..................] - ETA: 57s - loss: 0.2877 - categorical_accuracy: 0.9110
24800/60000 [===========>..................] - ETA: 57s - loss: 0.2874 - categorical_accuracy: 0.9111
24832/60000 [===========>..................] - ETA: 57s - loss: 0.2870 - categorical_accuracy: 0.9112
24864/60000 [===========>..................] - ETA: 57s - loss: 0.2868 - categorical_accuracy: 0.9113
24896/60000 [===========>..................] - ETA: 57s - loss: 0.2867 - categorical_accuracy: 0.9113
24928/60000 [===========>..................] - ETA: 57s - loss: 0.2866 - categorical_accuracy: 0.9113
24960/60000 [===========>..................] - ETA: 57s - loss: 0.2863 - categorical_accuracy: 0.9115
24992/60000 [===========>..................] - ETA: 57s - loss: 0.2859 - categorical_accuracy: 0.9116
25024/60000 [===========>..................] - ETA: 57s - loss: 0.2856 - categorical_accuracy: 0.9117
25056/60000 [===========>..................] - ETA: 57s - loss: 0.2854 - categorical_accuracy: 0.9118
25088/60000 [===========>..................] - ETA: 56s - loss: 0.2850 - categorical_accuracy: 0.9119
25120/60000 [===========>..................] - ETA: 56s - loss: 0.2849 - categorical_accuracy: 0.9119
25152/60000 [===========>..................] - ETA: 56s - loss: 0.2848 - categorical_accuracy: 0.9120
25184/60000 [===========>..................] - ETA: 56s - loss: 0.2844 - categorical_accuracy: 0.9121
25216/60000 [===========>..................] - ETA: 56s - loss: 0.2841 - categorical_accuracy: 0.9122
25248/60000 [===========>..................] - ETA: 56s - loss: 0.2841 - categorical_accuracy: 0.9122
25280/60000 [===========>..................] - ETA: 56s - loss: 0.2838 - categorical_accuracy: 0.9123
25312/60000 [===========>..................] - ETA: 56s - loss: 0.2835 - categorical_accuracy: 0.9124
25344/60000 [===========>..................] - ETA: 56s - loss: 0.2832 - categorical_accuracy: 0.9124
25376/60000 [===========>..................] - ETA: 56s - loss: 0.2829 - categorical_accuracy: 0.9126
25408/60000 [===========>..................] - ETA: 56s - loss: 0.2826 - categorical_accuracy: 0.9126
25440/60000 [===========>..................] - ETA: 56s - loss: 0.2825 - categorical_accuracy: 0.9127
25472/60000 [===========>..................] - ETA: 56s - loss: 0.2822 - categorical_accuracy: 0.9128
25504/60000 [===========>..................] - ETA: 56s - loss: 0.2820 - categorical_accuracy: 0.9128
25536/60000 [===========>..................] - ETA: 56s - loss: 0.2818 - categorical_accuracy: 0.9129
25568/60000 [===========>..................] - ETA: 56s - loss: 0.2815 - categorical_accuracy: 0.9130
25600/60000 [===========>..................] - ETA: 56s - loss: 0.2815 - categorical_accuracy: 0.9130
25632/60000 [===========>..................] - ETA: 56s - loss: 0.2812 - categorical_accuracy: 0.9131
25664/60000 [===========>..................] - ETA: 56s - loss: 0.2811 - categorical_accuracy: 0.9131
25696/60000 [===========>..................] - ETA: 56s - loss: 0.2810 - categorical_accuracy: 0.9132
25728/60000 [===========>..................] - ETA: 55s - loss: 0.2809 - categorical_accuracy: 0.9132
25760/60000 [===========>..................] - ETA: 55s - loss: 0.2808 - categorical_accuracy: 0.9132
25792/60000 [===========>..................] - ETA: 55s - loss: 0.2805 - categorical_accuracy: 0.9133
25824/60000 [===========>..................] - ETA: 55s - loss: 0.2805 - categorical_accuracy: 0.9134
25856/60000 [===========>..................] - ETA: 55s - loss: 0.2803 - categorical_accuracy: 0.9134
25888/60000 [===========>..................] - ETA: 55s - loss: 0.2801 - categorical_accuracy: 0.9135
25920/60000 [===========>..................] - ETA: 55s - loss: 0.2798 - categorical_accuracy: 0.9135
25952/60000 [===========>..................] - ETA: 55s - loss: 0.2795 - categorical_accuracy: 0.9136
25984/60000 [===========>..................] - ETA: 55s - loss: 0.2795 - categorical_accuracy: 0.9136
26016/60000 [============>.................] - ETA: 55s - loss: 0.2799 - categorical_accuracy: 0.9135
26048/60000 [============>.................] - ETA: 55s - loss: 0.2799 - categorical_accuracy: 0.9135
26080/60000 [============>.................] - ETA: 55s - loss: 0.2797 - categorical_accuracy: 0.9136
26112/60000 [============>.................] - ETA: 55s - loss: 0.2794 - categorical_accuracy: 0.9137
26144/60000 [============>.................] - ETA: 55s - loss: 0.2793 - categorical_accuracy: 0.9137
26176/60000 [============>.................] - ETA: 55s - loss: 0.2796 - categorical_accuracy: 0.9135
26208/60000 [============>.................] - ETA: 55s - loss: 0.2794 - categorical_accuracy: 0.9136
26240/60000 [============>.................] - ETA: 55s - loss: 0.2794 - categorical_accuracy: 0.9136
26272/60000 [============>.................] - ETA: 55s - loss: 0.2791 - categorical_accuracy: 0.9137
26304/60000 [============>.................] - ETA: 55s - loss: 0.2790 - categorical_accuracy: 0.9137
26336/60000 [============>.................] - ETA: 55s - loss: 0.2790 - categorical_accuracy: 0.9138
26368/60000 [============>.................] - ETA: 55s - loss: 0.2788 - categorical_accuracy: 0.9138
26400/60000 [============>.................] - ETA: 54s - loss: 0.2785 - categorical_accuracy: 0.9139
26432/60000 [============>.................] - ETA: 54s - loss: 0.2783 - categorical_accuracy: 0.9140
26464/60000 [============>.................] - ETA: 54s - loss: 0.2780 - categorical_accuracy: 0.9141
26496/60000 [============>.................] - ETA: 54s - loss: 0.2779 - categorical_accuracy: 0.9141
26528/60000 [============>.................] - ETA: 54s - loss: 0.2776 - categorical_accuracy: 0.9142
26560/60000 [============>.................] - ETA: 54s - loss: 0.2774 - categorical_accuracy: 0.9142
26592/60000 [============>.................] - ETA: 54s - loss: 0.2772 - categorical_accuracy: 0.9143
26624/60000 [============>.................] - ETA: 54s - loss: 0.2773 - categorical_accuracy: 0.9143
26656/60000 [============>.................] - ETA: 54s - loss: 0.2771 - categorical_accuracy: 0.9144
26688/60000 [============>.................] - ETA: 54s - loss: 0.2770 - categorical_accuracy: 0.9144
26720/60000 [============>.................] - ETA: 54s - loss: 0.2769 - categorical_accuracy: 0.9144
26752/60000 [============>.................] - ETA: 54s - loss: 0.2766 - categorical_accuracy: 0.9145
26784/60000 [============>.................] - ETA: 54s - loss: 0.2766 - categorical_accuracy: 0.9145
26816/60000 [============>.................] - ETA: 54s - loss: 0.2768 - categorical_accuracy: 0.9145
26848/60000 [============>.................] - ETA: 54s - loss: 0.2766 - categorical_accuracy: 0.9145
26880/60000 [============>.................] - ETA: 54s - loss: 0.2765 - categorical_accuracy: 0.9145
26912/60000 [============>.................] - ETA: 54s - loss: 0.2763 - categorical_accuracy: 0.9145
26944/60000 [============>.................] - ETA: 54s - loss: 0.2760 - categorical_accuracy: 0.9146
26976/60000 [============>.................] - ETA: 54s - loss: 0.2758 - categorical_accuracy: 0.9146
27008/60000 [============>.................] - ETA: 54s - loss: 0.2757 - categorical_accuracy: 0.9147
27040/60000 [============>.................] - ETA: 53s - loss: 0.2755 - categorical_accuracy: 0.9148
27104/60000 [============>.................] - ETA: 53s - loss: 0.2751 - categorical_accuracy: 0.9149
27136/60000 [============>.................] - ETA: 53s - loss: 0.2749 - categorical_accuracy: 0.9149
27168/60000 [============>.................] - ETA: 53s - loss: 0.2748 - categorical_accuracy: 0.9150
27200/60000 [============>.................] - ETA: 53s - loss: 0.2747 - categorical_accuracy: 0.9150
27232/60000 [============>.................] - ETA: 53s - loss: 0.2746 - categorical_accuracy: 0.9151
27264/60000 [============>.................] - ETA: 53s - loss: 0.2743 - categorical_accuracy: 0.9152
27296/60000 [============>.................] - ETA: 53s - loss: 0.2742 - categorical_accuracy: 0.9152
27328/60000 [============>.................] - ETA: 53s - loss: 0.2743 - categorical_accuracy: 0.9152
27360/60000 [============>.................] - ETA: 53s - loss: 0.2740 - categorical_accuracy: 0.9153
27392/60000 [============>.................] - ETA: 53s - loss: 0.2739 - categorical_accuracy: 0.9153
27424/60000 [============>.................] - ETA: 53s - loss: 0.2738 - categorical_accuracy: 0.9154
27456/60000 [============>.................] - ETA: 53s - loss: 0.2737 - categorical_accuracy: 0.9154
27488/60000 [============>.................] - ETA: 53s - loss: 0.2734 - categorical_accuracy: 0.9155
27520/60000 [============>.................] - ETA: 53s - loss: 0.2731 - categorical_accuracy: 0.9156
27552/60000 [============>.................] - ETA: 53s - loss: 0.2728 - categorical_accuracy: 0.9157
27584/60000 [============>.................] - ETA: 53s - loss: 0.2725 - categorical_accuracy: 0.9158
27616/60000 [============>.................] - ETA: 53s - loss: 0.2724 - categorical_accuracy: 0.9158
27648/60000 [============>.................] - ETA: 53s - loss: 0.2724 - categorical_accuracy: 0.9158
27680/60000 [============>.................] - ETA: 52s - loss: 0.2724 - categorical_accuracy: 0.9158
27712/60000 [============>.................] - ETA: 52s - loss: 0.2721 - categorical_accuracy: 0.9159
27744/60000 [============>.................] - ETA: 52s - loss: 0.2719 - categorical_accuracy: 0.9160
27776/60000 [============>.................] - ETA: 52s - loss: 0.2716 - categorical_accuracy: 0.9160
27808/60000 [============>.................] - ETA: 52s - loss: 0.2716 - categorical_accuracy: 0.9161
27840/60000 [============>.................] - ETA: 52s - loss: 0.2713 - categorical_accuracy: 0.9162
27904/60000 [============>.................] - ETA: 52s - loss: 0.2709 - categorical_accuracy: 0.9162
27968/60000 [============>.................] - ETA: 52s - loss: 0.2707 - categorical_accuracy: 0.9163
28000/60000 [=============>................] - ETA: 52s - loss: 0.2705 - categorical_accuracy: 0.9164
28032/60000 [=============>................] - ETA: 52s - loss: 0.2706 - categorical_accuracy: 0.9163
28064/60000 [=============>................] - ETA: 52s - loss: 0.2703 - categorical_accuracy: 0.9164
28096/60000 [=============>................] - ETA: 52s - loss: 0.2704 - categorical_accuracy: 0.9164
28128/60000 [=============>................] - ETA: 52s - loss: 0.2704 - categorical_accuracy: 0.9164
28160/60000 [=============>................] - ETA: 52s - loss: 0.2704 - categorical_accuracy: 0.9164
28192/60000 [=============>................] - ETA: 52s - loss: 0.2704 - categorical_accuracy: 0.9165
28256/60000 [=============>................] - ETA: 52s - loss: 0.2705 - categorical_accuracy: 0.9165
28288/60000 [=============>................] - ETA: 51s - loss: 0.2706 - categorical_accuracy: 0.9165
28320/60000 [=============>................] - ETA: 51s - loss: 0.2705 - categorical_accuracy: 0.9166
28352/60000 [=============>................] - ETA: 51s - loss: 0.2704 - categorical_accuracy: 0.9165
28384/60000 [=============>................] - ETA: 51s - loss: 0.2703 - categorical_accuracy: 0.9166
28416/60000 [=============>................] - ETA: 51s - loss: 0.2703 - categorical_accuracy: 0.9166
28448/60000 [=============>................] - ETA: 51s - loss: 0.2701 - categorical_accuracy: 0.9166
28480/60000 [=============>................] - ETA: 51s - loss: 0.2700 - categorical_accuracy: 0.9166
28512/60000 [=============>................] - ETA: 51s - loss: 0.2698 - categorical_accuracy: 0.9166
28544/60000 [=============>................] - ETA: 51s - loss: 0.2696 - categorical_accuracy: 0.9167
28576/60000 [=============>................] - ETA: 51s - loss: 0.2695 - categorical_accuracy: 0.9168
28608/60000 [=============>................] - ETA: 51s - loss: 0.2693 - categorical_accuracy: 0.9168
28640/60000 [=============>................] - ETA: 51s - loss: 0.2692 - categorical_accuracy: 0.9169
28672/60000 [=============>................] - ETA: 51s - loss: 0.2693 - categorical_accuracy: 0.9169
28736/60000 [=============>................] - ETA: 51s - loss: 0.2689 - categorical_accuracy: 0.9170
28768/60000 [=============>................] - ETA: 51s - loss: 0.2688 - categorical_accuracy: 0.9170
28800/60000 [=============>................] - ETA: 51s - loss: 0.2686 - categorical_accuracy: 0.9170
28832/60000 [=============>................] - ETA: 51s - loss: 0.2684 - categorical_accuracy: 0.9170
28864/60000 [=============>................] - ETA: 51s - loss: 0.2684 - categorical_accuracy: 0.9170
28896/60000 [=============>................] - ETA: 51s - loss: 0.2682 - categorical_accuracy: 0.9170
28928/60000 [=============>................] - ETA: 50s - loss: 0.2680 - categorical_accuracy: 0.9171
28960/60000 [=============>................] - ETA: 50s - loss: 0.2678 - categorical_accuracy: 0.9172
28992/60000 [=============>................] - ETA: 50s - loss: 0.2677 - categorical_accuracy: 0.9173
29056/60000 [=============>................] - ETA: 50s - loss: 0.2676 - categorical_accuracy: 0.9173
29088/60000 [=============>................] - ETA: 50s - loss: 0.2674 - categorical_accuracy: 0.9174
29120/60000 [=============>................] - ETA: 50s - loss: 0.2671 - categorical_accuracy: 0.9174
29152/60000 [=============>................] - ETA: 50s - loss: 0.2669 - categorical_accuracy: 0.9175
29184/60000 [=============>................] - ETA: 50s - loss: 0.2671 - categorical_accuracy: 0.9174
29216/60000 [=============>................] - ETA: 50s - loss: 0.2669 - categorical_accuracy: 0.9174
29248/60000 [=============>................] - ETA: 50s - loss: 0.2668 - categorical_accuracy: 0.9175
29280/60000 [=============>................] - ETA: 50s - loss: 0.2666 - categorical_accuracy: 0.9176
29312/60000 [=============>................] - ETA: 50s - loss: 0.2664 - categorical_accuracy: 0.9176
29344/60000 [=============>................] - ETA: 50s - loss: 0.2662 - categorical_accuracy: 0.9177
29376/60000 [=============>................] - ETA: 50s - loss: 0.2660 - categorical_accuracy: 0.9178
29408/60000 [=============>................] - ETA: 50s - loss: 0.2659 - categorical_accuracy: 0.9178
29440/60000 [=============>................] - ETA: 50s - loss: 0.2658 - categorical_accuracy: 0.9179
29472/60000 [=============>................] - ETA: 50s - loss: 0.2657 - categorical_accuracy: 0.9179
29504/60000 [=============>................] - ETA: 50s - loss: 0.2654 - categorical_accuracy: 0.9180
29536/60000 [=============>................] - ETA: 49s - loss: 0.2654 - categorical_accuracy: 0.9180
29568/60000 [=============>................] - ETA: 49s - loss: 0.2653 - categorical_accuracy: 0.9181
29600/60000 [=============>................] - ETA: 49s - loss: 0.2652 - categorical_accuracy: 0.9181
29632/60000 [=============>................] - ETA: 49s - loss: 0.2653 - categorical_accuracy: 0.9182
29664/60000 [=============>................] - ETA: 49s - loss: 0.2653 - categorical_accuracy: 0.9181
29696/60000 [=============>................] - ETA: 49s - loss: 0.2651 - categorical_accuracy: 0.9182
29728/60000 [=============>................] - ETA: 49s - loss: 0.2649 - categorical_accuracy: 0.9183
29760/60000 [=============>................] - ETA: 49s - loss: 0.2646 - categorical_accuracy: 0.9184
29792/60000 [=============>................] - ETA: 49s - loss: 0.2644 - categorical_accuracy: 0.9185
29824/60000 [=============>................] - ETA: 49s - loss: 0.2644 - categorical_accuracy: 0.9185
29856/60000 [=============>................] - ETA: 49s - loss: 0.2643 - categorical_accuracy: 0.9185
29888/60000 [=============>................] - ETA: 49s - loss: 0.2640 - categorical_accuracy: 0.9186
29920/60000 [=============>................] - ETA: 49s - loss: 0.2642 - categorical_accuracy: 0.9185
29952/60000 [=============>................] - ETA: 49s - loss: 0.2640 - categorical_accuracy: 0.9186
29984/60000 [=============>................] - ETA: 49s - loss: 0.2637 - categorical_accuracy: 0.9187
30016/60000 [==============>...............] - ETA: 49s - loss: 0.2635 - categorical_accuracy: 0.9188
30048/60000 [==============>...............] - ETA: 49s - loss: 0.2633 - categorical_accuracy: 0.9189
30080/60000 [==============>...............] - ETA: 49s - loss: 0.2633 - categorical_accuracy: 0.9189
30144/60000 [==============>...............] - ETA: 49s - loss: 0.2631 - categorical_accuracy: 0.9189
30176/60000 [==============>...............] - ETA: 48s - loss: 0.2628 - categorical_accuracy: 0.9190
30208/60000 [==============>...............] - ETA: 48s - loss: 0.2626 - categorical_accuracy: 0.9191
30240/60000 [==============>...............] - ETA: 48s - loss: 0.2625 - categorical_accuracy: 0.9190
30272/60000 [==============>...............] - ETA: 48s - loss: 0.2623 - categorical_accuracy: 0.9191
30304/60000 [==============>...............] - ETA: 48s - loss: 0.2621 - categorical_accuracy: 0.9192
30336/60000 [==============>...............] - ETA: 48s - loss: 0.2620 - categorical_accuracy: 0.9192
30368/60000 [==============>...............] - ETA: 48s - loss: 0.2618 - categorical_accuracy: 0.9193
30400/60000 [==============>...............] - ETA: 48s - loss: 0.2621 - categorical_accuracy: 0.9192
30432/60000 [==============>...............] - ETA: 48s - loss: 0.2619 - categorical_accuracy: 0.9192
30464/60000 [==============>...............] - ETA: 48s - loss: 0.2617 - categorical_accuracy: 0.9193
30496/60000 [==============>...............] - ETA: 48s - loss: 0.2615 - categorical_accuracy: 0.9194
30528/60000 [==============>...............] - ETA: 48s - loss: 0.2614 - categorical_accuracy: 0.9194
30560/60000 [==============>...............] - ETA: 48s - loss: 0.2612 - categorical_accuracy: 0.9195
30592/60000 [==============>...............] - ETA: 48s - loss: 0.2609 - categorical_accuracy: 0.9196
30624/60000 [==============>...............] - ETA: 48s - loss: 0.2607 - categorical_accuracy: 0.9196
30656/60000 [==============>...............] - ETA: 48s - loss: 0.2605 - categorical_accuracy: 0.9197
30688/60000 [==============>...............] - ETA: 48s - loss: 0.2602 - categorical_accuracy: 0.9198
30720/60000 [==============>...............] - ETA: 48s - loss: 0.2600 - categorical_accuracy: 0.9198
30752/60000 [==============>...............] - ETA: 48s - loss: 0.2597 - categorical_accuracy: 0.9199
30784/60000 [==============>...............] - ETA: 48s - loss: 0.2596 - categorical_accuracy: 0.9200
30848/60000 [==============>...............] - ETA: 47s - loss: 0.2591 - categorical_accuracy: 0.9201
30880/60000 [==============>...............] - ETA: 47s - loss: 0.2591 - categorical_accuracy: 0.9201
30912/60000 [==============>...............] - ETA: 47s - loss: 0.2588 - categorical_accuracy: 0.9202
30944/60000 [==============>...............] - ETA: 47s - loss: 0.2586 - categorical_accuracy: 0.9203
30976/60000 [==============>...............] - ETA: 47s - loss: 0.2585 - categorical_accuracy: 0.9203
31040/60000 [==============>...............] - ETA: 47s - loss: 0.2581 - categorical_accuracy: 0.9204
31072/60000 [==============>...............] - ETA: 47s - loss: 0.2580 - categorical_accuracy: 0.9204
31104/60000 [==============>...............] - ETA: 47s - loss: 0.2578 - categorical_accuracy: 0.9205
31136/60000 [==============>...............] - ETA: 47s - loss: 0.2577 - categorical_accuracy: 0.9205
31168/60000 [==============>...............] - ETA: 47s - loss: 0.2575 - categorical_accuracy: 0.9206
31200/60000 [==============>...............] - ETA: 47s - loss: 0.2573 - categorical_accuracy: 0.9207
31232/60000 [==============>...............] - ETA: 47s - loss: 0.2573 - categorical_accuracy: 0.9207
31264/60000 [==============>...............] - ETA: 47s - loss: 0.2572 - categorical_accuracy: 0.9207
31296/60000 [==============>...............] - ETA: 47s - loss: 0.2571 - categorical_accuracy: 0.9208
31328/60000 [==============>...............] - ETA: 47s - loss: 0.2569 - categorical_accuracy: 0.9208
31360/60000 [==============>...............] - ETA: 47s - loss: 0.2567 - categorical_accuracy: 0.9208
31392/60000 [==============>...............] - ETA: 47s - loss: 0.2567 - categorical_accuracy: 0.9208
31424/60000 [==============>...............] - ETA: 46s - loss: 0.2567 - categorical_accuracy: 0.9208
31456/60000 [==============>...............] - ETA: 46s - loss: 0.2566 - categorical_accuracy: 0.9208
31488/60000 [==============>...............] - ETA: 46s - loss: 0.2565 - categorical_accuracy: 0.9209
31552/60000 [==============>...............] - ETA: 46s - loss: 0.2562 - categorical_accuracy: 0.9210
31616/60000 [==============>...............] - ETA: 46s - loss: 0.2559 - categorical_accuracy: 0.9211
31648/60000 [==============>...............] - ETA: 46s - loss: 0.2558 - categorical_accuracy: 0.9211
31680/60000 [==============>...............] - ETA: 46s - loss: 0.2556 - categorical_accuracy: 0.9211
31712/60000 [==============>...............] - ETA: 46s - loss: 0.2554 - categorical_accuracy: 0.9212
31744/60000 [==============>...............] - ETA: 46s - loss: 0.2553 - categorical_accuracy: 0.9212
31776/60000 [==============>...............] - ETA: 46s - loss: 0.2551 - categorical_accuracy: 0.9213
31808/60000 [==============>...............] - ETA: 46s - loss: 0.2548 - categorical_accuracy: 0.9214
31872/60000 [==============>...............] - ETA: 46s - loss: 0.2546 - categorical_accuracy: 0.9215
31936/60000 [==============>...............] - ETA: 46s - loss: 0.2541 - categorical_accuracy: 0.9216
31968/60000 [==============>...............] - ETA: 46s - loss: 0.2539 - categorical_accuracy: 0.9217
32000/60000 [===============>..............] - ETA: 46s - loss: 0.2538 - categorical_accuracy: 0.9217
32032/60000 [===============>..............] - ETA: 45s - loss: 0.2536 - categorical_accuracy: 0.9218
32064/60000 [===============>..............] - ETA: 45s - loss: 0.2534 - categorical_accuracy: 0.9218
32096/60000 [===============>..............] - ETA: 45s - loss: 0.2532 - categorical_accuracy: 0.9219
32128/60000 [===============>..............] - ETA: 45s - loss: 0.2532 - categorical_accuracy: 0.9219
32160/60000 [===============>..............] - ETA: 45s - loss: 0.2530 - categorical_accuracy: 0.9220
32192/60000 [===============>..............] - ETA: 45s - loss: 0.2529 - categorical_accuracy: 0.9220
32224/60000 [===============>..............] - ETA: 45s - loss: 0.2526 - categorical_accuracy: 0.9221
32256/60000 [===============>..............] - ETA: 45s - loss: 0.2528 - categorical_accuracy: 0.9221
32288/60000 [===============>..............] - ETA: 45s - loss: 0.2526 - categorical_accuracy: 0.9221
32352/60000 [===============>..............] - ETA: 45s - loss: 0.2526 - categorical_accuracy: 0.9221
32384/60000 [===============>..............] - ETA: 45s - loss: 0.2524 - categorical_accuracy: 0.9222
32416/60000 [===============>..............] - ETA: 45s - loss: 0.2527 - categorical_accuracy: 0.9222
32448/60000 [===============>..............] - ETA: 45s - loss: 0.2525 - categorical_accuracy: 0.9222
32480/60000 [===============>..............] - ETA: 45s - loss: 0.2524 - categorical_accuracy: 0.9222
32512/60000 [===============>..............] - ETA: 45s - loss: 0.2521 - categorical_accuracy: 0.9223
32544/60000 [===============>..............] - ETA: 45s - loss: 0.2521 - categorical_accuracy: 0.9224
32576/60000 [===============>..............] - ETA: 45s - loss: 0.2519 - categorical_accuracy: 0.9224
32608/60000 [===============>..............] - ETA: 45s - loss: 0.2520 - categorical_accuracy: 0.9224
32672/60000 [===============>..............] - ETA: 44s - loss: 0.2518 - categorical_accuracy: 0.9225
32704/60000 [===============>..............] - ETA: 44s - loss: 0.2516 - categorical_accuracy: 0.9225
32736/60000 [===============>..............] - ETA: 44s - loss: 0.2515 - categorical_accuracy: 0.9226
32768/60000 [===============>..............] - ETA: 44s - loss: 0.2514 - categorical_accuracy: 0.9226
32832/60000 [===============>..............] - ETA: 44s - loss: 0.2512 - categorical_accuracy: 0.9227
32896/60000 [===============>..............] - ETA: 44s - loss: 0.2508 - categorical_accuracy: 0.9228
32960/60000 [===============>..............] - ETA: 44s - loss: 0.2505 - categorical_accuracy: 0.9229
32992/60000 [===============>..............] - ETA: 44s - loss: 0.2503 - categorical_accuracy: 0.9230
33024/60000 [===============>..............] - ETA: 44s - loss: 0.2501 - categorical_accuracy: 0.9230
33056/60000 [===============>..............] - ETA: 44s - loss: 0.2499 - categorical_accuracy: 0.9231
33088/60000 [===============>..............] - ETA: 44s - loss: 0.2498 - categorical_accuracy: 0.9231
33120/60000 [===============>..............] - ETA: 44s - loss: 0.2496 - categorical_accuracy: 0.9232
33152/60000 [===============>..............] - ETA: 44s - loss: 0.2494 - categorical_accuracy: 0.9232
33184/60000 [===============>..............] - ETA: 44s - loss: 0.2492 - categorical_accuracy: 0.9233
33248/60000 [===============>..............] - ETA: 44s - loss: 0.2488 - categorical_accuracy: 0.9234
33312/60000 [===============>..............] - ETA: 43s - loss: 0.2485 - categorical_accuracy: 0.9235
33376/60000 [===============>..............] - ETA: 43s - loss: 0.2483 - categorical_accuracy: 0.9236
33408/60000 [===============>..............] - ETA: 43s - loss: 0.2481 - categorical_accuracy: 0.9236
33440/60000 [===============>..............] - ETA: 43s - loss: 0.2482 - categorical_accuracy: 0.9237
33472/60000 [===============>..............] - ETA: 43s - loss: 0.2483 - categorical_accuracy: 0.9237
33504/60000 [===============>..............] - ETA: 43s - loss: 0.2483 - categorical_accuracy: 0.9237
33568/60000 [===============>..............] - ETA: 43s - loss: 0.2479 - categorical_accuracy: 0.9238
33600/60000 [===============>..............] - ETA: 43s - loss: 0.2478 - categorical_accuracy: 0.9238
33632/60000 [===============>..............] - ETA: 43s - loss: 0.2478 - categorical_accuracy: 0.9238
33696/60000 [===============>..............] - ETA: 43s - loss: 0.2474 - categorical_accuracy: 0.9240
33728/60000 [===============>..............] - ETA: 43s - loss: 0.2472 - categorical_accuracy: 0.9240
33760/60000 [===============>..............] - ETA: 43s - loss: 0.2471 - categorical_accuracy: 0.9240
33792/60000 [===============>..............] - ETA: 43s - loss: 0.2469 - categorical_accuracy: 0.9240
33856/60000 [===============>..............] - ETA: 43s - loss: 0.2468 - categorical_accuracy: 0.9241
33920/60000 [===============>..............] - ETA: 42s - loss: 0.2464 - categorical_accuracy: 0.9242
33952/60000 [===============>..............] - ETA: 42s - loss: 0.2464 - categorical_accuracy: 0.9243
34016/60000 [================>.............] - ETA: 42s - loss: 0.2460 - categorical_accuracy: 0.9244
34048/60000 [================>.............] - ETA: 42s - loss: 0.2460 - categorical_accuracy: 0.9244
34080/60000 [================>.............] - ETA: 42s - loss: 0.2458 - categorical_accuracy: 0.9245
34112/60000 [================>.............] - ETA: 42s - loss: 0.2457 - categorical_accuracy: 0.9245
34176/60000 [================>.............] - ETA: 42s - loss: 0.2454 - categorical_accuracy: 0.9246
34240/60000 [================>.............] - ETA: 42s - loss: 0.2450 - categorical_accuracy: 0.9248
34272/60000 [================>.............] - ETA: 42s - loss: 0.2449 - categorical_accuracy: 0.9247
34304/60000 [================>.............] - ETA: 42s - loss: 0.2450 - categorical_accuracy: 0.9247
34336/60000 [================>.............] - ETA: 42s - loss: 0.2448 - categorical_accuracy: 0.9248
34368/60000 [================>.............] - ETA: 42s - loss: 0.2447 - categorical_accuracy: 0.9248
34400/60000 [================>.............] - ETA: 42s - loss: 0.2445 - categorical_accuracy: 0.9249
34432/60000 [================>.............] - ETA: 42s - loss: 0.2443 - categorical_accuracy: 0.9249
34464/60000 [================>.............] - ETA: 42s - loss: 0.2444 - categorical_accuracy: 0.9249
34496/60000 [================>.............] - ETA: 41s - loss: 0.2442 - categorical_accuracy: 0.9249
34528/60000 [================>.............] - ETA: 41s - loss: 0.2441 - categorical_accuracy: 0.9250
34560/60000 [================>.............] - ETA: 41s - loss: 0.2442 - categorical_accuracy: 0.9250
34592/60000 [================>.............] - ETA: 41s - loss: 0.2440 - categorical_accuracy: 0.9250
34624/60000 [================>.............] - ETA: 41s - loss: 0.2438 - categorical_accuracy: 0.9251
34688/60000 [================>.............] - ETA: 41s - loss: 0.2436 - categorical_accuracy: 0.9251
34720/60000 [================>.............] - ETA: 41s - loss: 0.2435 - categorical_accuracy: 0.9251
34752/60000 [================>.............] - ETA: 41s - loss: 0.2436 - categorical_accuracy: 0.9251
34784/60000 [================>.............] - ETA: 41s - loss: 0.2434 - categorical_accuracy: 0.9252
34816/60000 [================>.............] - ETA: 41s - loss: 0.2433 - categorical_accuracy: 0.9252
34848/60000 [================>.............] - ETA: 41s - loss: 0.2431 - categorical_accuracy: 0.9253
34880/60000 [================>.............] - ETA: 41s - loss: 0.2433 - categorical_accuracy: 0.9253
34944/60000 [================>.............] - ETA: 41s - loss: 0.2431 - categorical_accuracy: 0.9253
34976/60000 [================>.............] - ETA: 41s - loss: 0.2429 - categorical_accuracy: 0.9253
35008/60000 [================>.............] - ETA: 41s - loss: 0.2429 - categorical_accuracy: 0.9253
35072/60000 [================>.............] - ETA: 41s - loss: 0.2431 - categorical_accuracy: 0.9253
35104/60000 [================>.............] - ETA: 40s - loss: 0.2430 - categorical_accuracy: 0.9253
35136/60000 [================>.............] - ETA: 40s - loss: 0.2430 - categorical_accuracy: 0.9253
35168/60000 [================>.............] - ETA: 40s - loss: 0.2428 - categorical_accuracy: 0.9253
35200/60000 [================>.............] - ETA: 40s - loss: 0.2429 - categorical_accuracy: 0.9253
35232/60000 [================>.............] - ETA: 40s - loss: 0.2429 - categorical_accuracy: 0.9253
35296/60000 [================>.............] - ETA: 40s - loss: 0.2426 - categorical_accuracy: 0.9253
35328/60000 [================>.............] - ETA: 40s - loss: 0.2424 - categorical_accuracy: 0.9254
35392/60000 [================>.............] - ETA: 40s - loss: 0.2421 - categorical_accuracy: 0.9255
35424/60000 [================>.............] - ETA: 40s - loss: 0.2420 - categorical_accuracy: 0.9255
35456/60000 [================>.............] - ETA: 40s - loss: 0.2419 - categorical_accuracy: 0.9256
35488/60000 [================>.............] - ETA: 40s - loss: 0.2417 - categorical_accuracy: 0.9256
35520/60000 [================>.............] - ETA: 40s - loss: 0.2415 - categorical_accuracy: 0.9256
35552/60000 [================>.............] - ETA: 40s - loss: 0.2413 - categorical_accuracy: 0.9257
35584/60000 [================>.............] - ETA: 40s - loss: 0.2412 - categorical_accuracy: 0.9258
35648/60000 [================>.............] - ETA: 40s - loss: 0.2409 - categorical_accuracy: 0.9259
35712/60000 [================>.............] - ETA: 39s - loss: 0.2406 - categorical_accuracy: 0.9260
35744/60000 [================>.............] - ETA: 39s - loss: 0.2405 - categorical_accuracy: 0.9260
35776/60000 [================>.............] - ETA: 39s - loss: 0.2406 - categorical_accuracy: 0.9260
35840/60000 [================>.............] - ETA: 39s - loss: 0.2404 - categorical_accuracy: 0.9261
35904/60000 [================>.............] - ETA: 39s - loss: 0.2400 - categorical_accuracy: 0.9262
35936/60000 [================>.............] - ETA: 39s - loss: 0.2399 - categorical_accuracy: 0.9262
35968/60000 [================>.............] - ETA: 39s - loss: 0.2398 - categorical_accuracy: 0.9262
36000/60000 [=================>............] - ETA: 39s - loss: 0.2397 - categorical_accuracy: 0.9263
36032/60000 [=================>............] - ETA: 39s - loss: 0.2395 - categorical_accuracy: 0.9263
36064/60000 [=================>............] - ETA: 39s - loss: 0.2393 - categorical_accuracy: 0.9264
36096/60000 [=================>............] - ETA: 39s - loss: 0.2391 - categorical_accuracy: 0.9264
36128/60000 [=================>............] - ETA: 39s - loss: 0.2390 - categorical_accuracy: 0.9265
36160/60000 [=================>............] - ETA: 39s - loss: 0.2389 - categorical_accuracy: 0.9265
36192/60000 [=================>............] - ETA: 39s - loss: 0.2389 - categorical_accuracy: 0.9265
36224/60000 [=================>............] - ETA: 39s - loss: 0.2388 - categorical_accuracy: 0.9265
36256/60000 [=================>............] - ETA: 39s - loss: 0.2386 - categorical_accuracy: 0.9266
36288/60000 [=================>............] - ETA: 39s - loss: 0.2392 - categorical_accuracy: 0.9266
36352/60000 [=================>............] - ETA: 38s - loss: 0.2390 - categorical_accuracy: 0.9266
36384/60000 [=================>............] - ETA: 38s - loss: 0.2388 - categorical_accuracy: 0.9267
36416/60000 [=================>............] - ETA: 38s - loss: 0.2386 - categorical_accuracy: 0.9268
36448/60000 [=================>............] - ETA: 38s - loss: 0.2385 - categorical_accuracy: 0.9268
36480/60000 [=================>............] - ETA: 38s - loss: 0.2383 - categorical_accuracy: 0.9269
36512/60000 [=================>............] - ETA: 38s - loss: 0.2381 - categorical_accuracy: 0.9269
36544/60000 [=================>............] - ETA: 38s - loss: 0.2380 - categorical_accuracy: 0.9270
36576/60000 [=================>............] - ETA: 38s - loss: 0.2378 - categorical_accuracy: 0.9270
36640/60000 [=================>............] - ETA: 38s - loss: 0.2376 - categorical_accuracy: 0.9270
36672/60000 [=================>............] - ETA: 38s - loss: 0.2376 - categorical_accuracy: 0.9271
36704/60000 [=================>............] - ETA: 38s - loss: 0.2378 - categorical_accuracy: 0.9270
36768/60000 [=================>............] - ETA: 38s - loss: 0.2375 - categorical_accuracy: 0.9271
36800/60000 [=================>............] - ETA: 38s - loss: 0.2373 - categorical_accuracy: 0.9272
36832/60000 [=================>............] - ETA: 38s - loss: 0.2374 - categorical_accuracy: 0.9272
36864/60000 [=================>............] - ETA: 38s - loss: 0.2372 - categorical_accuracy: 0.9273
36896/60000 [=================>............] - ETA: 38s - loss: 0.2371 - categorical_accuracy: 0.9273
36928/60000 [=================>............] - ETA: 37s - loss: 0.2369 - categorical_accuracy: 0.9274
36960/60000 [=================>............] - ETA: 37s - loss: 0.2367 - categorical_accuracy: 0.9275
36992/60000 [=================>............] - ETA: 37s - loss: 0.2368 - categorical_accuracy: 0.9274
37056/60000 [=================>............] - ETA: 37s - loss: 0.2368 - categorical_accuracy: 0.9275
37088/60000 [=================>............] - ETA: 37s - loss: 0.2367 - categorical_accuracy: 0.9275
37152/60000 [=================>............] - ETA: 37s - loss: 0.2363 - categorical_accuracy: 0.9276
37184/60000 [=================>............] - ETA: 37s - loss: 0.2363 - categorical_accuracy: 0.9276
37248/60000 [=================>............] - ETA: 37s - loss: 0.2361 - categorical_accuracy: 0.9277
37280/60000 [=================>............] - ETA: 37s - loss: 0.2361 - categorical_accuracy: 0.9277
37312/60000 [=================>............] - ETA: 37s - loss: 0.2359 - categorical_accuracy: 0.9277
37344/60000 [=================>............] - ETA: 37s - loss: 0.2357 - categorical_accuracy: 0.9278
37408/60000 [=================>............] - ETA: 37s - loss: 0.2356 - categorical_accuracy: 0.9278
37440/60000 [=================>............] - ETA: 37s - loss: 0.2354 - categorical_accuracy: 0.9278
37472/60000 [=================>............] - ETA: 37s - loss: 0.2353 - categorical_accuracy: 0.9278
37504/60000 [=================>............] - ETA: 37s - loss: 0.2353 - categorical_accuracy: 0.9279
37536/60000 [=================>............] - ETA: 36s - loss: 0.2351 - categorical_accuracy: 0.9279
37568/60000 [=================>............] - ETA: 36s - loss: 0.2350 - categorical_accuracy: 0.9279
37600/60000 [=================>............] - ETA: 36s - loss: 0.2348 - categorical_accuracy: 0.9280
37632/60000 [=================>............] - ETA: 36s - loss: 0.2347 - categorical_accuracy: 0.9281
37696/60000 [=================>............] - ETA: 36s - loss: 0.2344 - categorical_accuracy: 0.9281
37760/60000 [=================>............] - ETA: 36s - loss: 0.2342 - categorical_accuracy: 0.9282
37792/60000 [=================>............] - ETA: 36s - loss: 0.2342 - categorical_accuracy: 0.9282
37824/60000 [=================>............] - ETA: 36s - loss: 0.2340 - categorical_accuracy: 0.9282
37856/60000 [=================>............] - ETA: 36s - loss: 0.2340 - categorical_accuracy: 0.9282
37920/60000 [=================>............] - ETA: 36s - loss: 0.2337 - categorical_accuracy: 0.9283
37952/60000 [=================>............] - ETA: 36s - loss: 0.2339 - categorical_accuracy: 0.9283
37984/60000 [=================>............] - ETA: 36s - loss: 0.2338 - categorical_accuracy: 0.9283
38016/60000 [==================>...........] - ETA: 36s - loss: 0.2338 - categorical_accuracy: 0.9282
38048/60000 [==================>...........] - ETA: 36s - loss: 0.2337 - categorical_accuracy: 0.9282
38112/60000 [==================>...........] - ETA: 36s - loss: 0.2334 - categorical_accuracy: 0.9284
38144/60000 [==================>...........] - ETA: 35s - loss: 0.2332 - categorical_accuracy: 0.9284
38208/60000 [==================>...........] - ETA: 35s - loss: 0.2329 - categorical_accuracy: 0.9285
38240/60000 [==================>...........] - ETA: 35s - loss: 0.2330 - categorical_accuracy: 0.9285
38272/60000 [==================>...........] - ETA: 35s - loss: 0.2328 - categorical_accuracy: 0.9286
38304/60000 [==================>...........] - ETA: 35s - loss: 0.2327 - categorical_accuracy: 0.9286
38368/60000 [==================>...........] - ETA: 35s - loss: 0.2323 - categorical_accuracy: 0.9287
38400/60000 [==================>...........] - ETA: 35s - loss: 0.2322 - categorical_accuracy: 0.9287
38464/60000 [==================>...........] - ETA: 35s - loss: 0.2321 - categorical_accuracy: 0.9288
38496/60000 [==================>...........] - ETA: 35s - loss: 0.2321 - categorical_accuracy: 0.9287
38528/60000 [==================>...........] - ETA: 35s - loss: 0.2319 - categorical_accuracy: 0.9288
38560/60000 [==================>...........] - ETA: 35s - loss: 0.2320 - categorical_accuracy: 0.9288
38592/60000 [==================>...........] - ETA: 35s - loss: 0.2321 - categorical_accuracy: 0.9288
38624/60000 [==================>...........] - ETA: 35s - loss: 0.2320 - categorical_accuracy: 0.9288
38656/60000 [==================>...........] - ETA: 35s - loss: 0.2319 - categorical_accuracy: 0.9288
38688/60000 [==================>...........] - ETA: 35s - loss: 0.2318 - categorical_accuracy: 0.9289
38720/60000 [==================>...........] - ETA: 35s - loss: 0.2319 - categorical_accuracy: 0.9288
38752/60000 [==================>...........] - ETA: 34s - loss: 0.2317 - categorical_accuracy: 0.9289
38816/60000 [==================>...........] - ETA: 34s - loss: 0.2315 - categorical_accuracy: 0.9290
38848/60000 [==================>...........] - ETA: 34s - loss: 0.2314 - categorical_accuracy: 0.9290
38880/60000 [==================>...........] - ETA: 34s - loss: 0.2313 - categorical_accuracy: 0.9290
38912/60000 [==================>...........] - ETA: 34s - loss: 0.2313 - categorical_accuracy: 0.9290
38944/60000 [==================>...........] - ETA: 34s - loss: 0.2312 - categorical_accuracy: 0.9291
38976/60000 [==================>...........] - ETA: 34s - loss: 0.2310 - categorical_accuracy: 0.9291
39040/60000 [==================>...........] - ETA: 34s - loss: 0.2309 - categorical_accuracy: 0.9292
39072/60000 [==================>...........] - ETA: 34s - loss: 0.2308 - categorical_accuracy: 0.9292
39104/60000 [==================>...........] - ETA: 34s - loss: 0.2307 - categorical_accuracy: 0.9292
39136/60000 [==================>...........] - ETA: 34s - loss: 0.2306 - categorical_accuracy: 0.9293
39168/60000 [==================>...........] - ETA: 34s - loss: 0.2305 - categorical_accuracy: 0.9293
39200/60000 [==================>...........] - ETA: 34s - loss: 0.2304 - categorical_accuracy: 0.9294
39232/60000 [==================>...........] - ETA: 34s - loss: 0.2303 - categorical_accuracy: 0.9294
39296/60000 [==================>...........] - ETA: 34s - loss: 0.2301 - categorical_accuracy: 0.9294
39328/60000 [==================>...........] - ETA: 34s - loss: 0.2300 - categorical_accuracy: 0.9294
39360/60000 [==================>...........] - ETA: 33s - loss: 0.2299 - categorical_accuracy: 0.9294
39392/60000 [==================>...........] - ETA: 33s - loss: 0.2298 - categorical_accuracy: 0.9295
39424/60000 [==================>...........] - ETA: 33s - loss: 0.2299 - categorical_accuracy: 0.9294
39456/60000 [==================>...........] - ETA: 33s - loss: 0.2299 - categorical_accuracy: 0.9294
39488/60000 [==================>...........] - ETA: 33s - loss: 0.2297 - categorical_accuracy: 0.9294
39520/60000 [==================>...........] - ETA: 33s - loss: 0.2296 - categorical_accuracy: 0.9295
39552/60000 [==================>...........] - ETA: 33s - loss: 0.2294 - categorical_accuracy: 0.9296
39584/60000 [==================>...........] - ETA: 33s - loss: 0.2292 - categorical_accuracy: 0.9296
39616/60000 [==================>...........] - ETA: 33s - loss: 0.2290 - categorical_accuracy: 0.9297
39648/60000 [==================>...........] - ETA: 33s - loss: 0.2289 - categorical_accuracy: 0.9297
39680/60000 [==================>...........] - ETA: 33s - loss: 0.2288 - categorical_accuracy: 0.9298
39744/60000 [==================>...........] - ETA: 33s - loss: 0.2286 - categorical_accuracy: 0.9298
39776/60000 [==================>...........] - ETA: 33s - loss: 0.2287 - categorical_accuracy: 0.9298
39840/60000 [==================>...........] - ETA: 33s - loss: 0.2286 - categorical_accuracy: 0.9298
39872/60000 [==================>...........] - ETA: 33s - loss: 0.2285 - categorical_accuracy: 0.9299
39904/60000 [==================>...........] - ETA: 33s - loss: 0.2284 - categorical_accuracy: 0.9299
39968/60000 [==================>...........] - ETA: 32s - loss: 0.2281 - categorical_accuracy: 0.9299
40032/60000 [===================>..........] - ETA: 32s - loss: 0.2281 - categorical_accuracy: 0.9299
40064/60000 [===================>..........] - ETA: 32s - loss: 0.2280 - categorical_accuracy: 0.9300
40128/60000 [===================>..........] - ETA: 32s - loss: 0.2277 - categorical_accuracy: 0.9301
40160/60000 [===================>..........] - ETA: 32s - loss: 0.2276 - categorical_accuracy: 0.9301
40192/60000 [===================>..........] - ETA: 32s - loss: 0.2275 - categorical_accuracy: 0.9301
40256/60000 [===================>..........] - ETA: 32s - loss: 0.2274 - categorical_accuracy: 0.9302
40288/60000 [===================>..........] - ETA: 32s - loss: 0.2273 - categorical_accuracy: 0.9302
40320/60000 [===================>..........] - ETA: 32s - loss: 0.2271 - categorical_accuracy: 0.9303
40352/60000 [===================>..........] - ETA: 32s - loss: 0.2270 - categorical_accuracy: 0.9303
40384/60000 [===================>..........] - ETA: 32s - loss: 0.2269 - categorical_accuracy: 0.9303
40416/60000 [===================>..........] - ETA: 32s - loss: 0.2268 - categorical_accuracy: 0.9303
40448/60000 [===================>..........] - ETA: 32s - loss: 0.2266 - categorical_accuracy: 0.9304
40480/60000 [===================>..........] - ETA: 32s - loss: 0.2265 - categorical_accuracy: 0.9304
40512/60000 [===================>..........] - ETA: 32s - loss: 0.2263 - categorical_accuracy: 0.9305
40576/60000 [===================>..........] - ETA: 31s - loss: 0.2261 - categorical_accuracy: 0.9306
40608/60000 [===================>..........] - ETA: 31s - loss: 0.2260 - categorical_accuracy: 0.9306
40640/60000 [===================>..........] - ETA: 31s - loss: 0.2259 - categorical_accuracy: 0.9306
40672/60000 [===================>..........] - ETA: 31s - loss: 0.2257 - categorical_accuracy: 0.9307
40704/60000 [===================>..........] - ETA: 31s - loss: 0.2256 - categorical_accuracy: 0.9307
40736/60000 [===================>..........] - ETA: 31s - loss: 0.2254 - categorical_accuracy: 0.9308
40768/60000 [===================>..........] - ETA: 31s - loss: 0.2253 - categorical_accuracy: 0.9308
40800/60000 [===================>..........] - ETA: 31s - loss: 0.2251 - categorical_accuracy: 0.9309
40832/60000 [===================>..........] - ETA: 31s - loss: 0.2250 - categorical_accuracy: 0.9309
40864/60000 [===================>..........] - ETA: 31s - loss: 0.2249 - categorical_accuracy: 0.9309
40928/60000 [===================>..........] - ETA: 31s - loss: 0.2246 - categorical_accuracy: 0.9310
40960/60000 [===================>..........] - ETA: 31s - loss: 0.2245 - categorical_accuracy: 0.9310
40992/60000 [===================>..........] - ETA: 31s - loss: 0.2243 - categorical_accuracy: 0.9311
41024/60000 [===================>..........] - ETA: 31s - loss: 0.2242 - categorical_accuracy: 0.9311
41056/60000 [===================>..........] - ETA: 31s - loss: 0.2241 - categorical_accuracy: 0.9312
41088/60000 [===================>..........] - ETA: 31s - loss: 0.2240 - categorical_accuracy: 0.9312
41120/60000 [===================>..........] - ETA: 31s - loss: 0.2238 - categorical_accuracy: 0.9312
41152/60000 [===================>..........] - ETA: 31s - loss: 0.2237 - categorical_accuracy: 0.9313
41184/60000 [===================>..........] - ETA: 30s - loss: 0.2236 - categorical_accuracy: 0.9313
41216/60000 [===================>..........] - ETA: 30s - loss: 0.2235 - categorical_accuracy: 0.9313
41248/60000 [===================>..........] - ETA: 30s - loss: 0.2234 - categorical_accuracy: 0.9314
41280/60000 [===================>..........] - ETA: 30s - loss: 0.2233 - categorical_accuracy: 0.9314
41312/60000 [===================>..........] - ETA: 30s - loss: 0.2233 - categorical_accuracy: 0.9314
41376/60000 [===================>..........] - ETA: 30s - loss: 0.2230 - categorical_accuracy: 0.9315
41408/60000 [===================>..........] - ETA: 30s - loss: 0.2230 - categorical_accuracy: 0.9315
41472/60000 [===================>..........] - ETA: 30s - loss: 0.2229 - categorical_accuracy: 0.9315
41504/60000 [===================>..........] - ETA: 30s - loss: 0.2228 - categorical_accuracy: 0.9315
41536/60000 [===================>..........] - ETA: 30s - loss: 0.2227 - categorical_accuracy: 0.9315
41600/60000 [===================>..........] - ETA: 30s - loss: 0.2228 - categorical_accuracy: 0.9315
41632/60000 [===================>..........] - ETA: 30s - loss: 0.2226 - categorical_accuracy: 0.9316
41664/60000 [===================>..........] - ETA: 30s - loss: 0.2225 - categorical_accuracy: 0.9316
41696/60000 [===================>..........] - ETA: 30s - loss: 0.2226 - categorical_accuracy: 0.9316
41728/60000 [===================>..........] - ETA: 30s - loss: 0.2225 - categorical_accuracy: 0.9316
41760/60000 [===================>..........] - ETA: 30s - loss: 0.2224 - categorical_accuracy: 0.9316
41792/60000 [===================>..........] - ETA: 29s - loss: 0.2223 - categorical_accuracy: 0.9317
41824/60000 [===================>..........] - ETA: 29s - loss: 0.2222 - categorical_accuracy: 0.9317
41856/60000 [===================>..........] - ETA: 29s - loss: 0.2220 - categorical_accuracy: 0.9318
41888/60000 [===================>..........] - ETA: 29s - loss: 0.2221 - categorical_accuracy: 0.9318
41920/60000 [===================>..........] - ETA: 29s - loss: 0.2219 - categorical_accuracy: 0.9318
41952/60000 [===================>..........] - ETA: 29s - loss: 0.2218 - categorical_accuracy: 0.9319
41984/60000 [===================>..........] - ETA: 29s - loss: 0.2217 - categorical_accuracy: 0.9319
42048/60000 [====================>.........] - ETA: 29s - loss: 0.2216 - categorical_accuracy: 0.9320
42080/60000 [====================>.........] - ETA: 29s - loss: 0.2215 - categorical_accuracy: 0.9320
42144/60000 [====================>.........] - ETA: 29s - loss: 0.2212 - categorical_accuracy: 0.9321
42176/60000 [====================>.........] - ETA: 29s - loss: 0.2213 - categorical_accuracy: 0.9321
42208/60000 [====================>.........] - ETA: 29s - loss: 0.2212 - categorical_accuracy: 0.9321
42240/60000 [====================>.........] - ETA: 29s - loss: 0.2210 - categorical_accuracy: 0.9321
42304/60000 [====================>.........] - ETA: 29s - loss: 0.2210 - categorical_accuracy: 0.9322
42368/60000 [====================>.........] - ETA: 29s - loss: 0.2207 - categorical_accuracy: 0.9323
42432/60000 [====================>.........] - ETA: 28s - loss: 0.2205 - categorical_accuracy: 0.9323
42464/60000 [====================>.........] - ETA: 28s - loss: 0.2203 - categorical_accuracy: 0.9324
42496/60000 [====================>.........] - ETA: 28s - loss: 0.2203 - categorical_accuracy: 0.9324
42528/60000 [====================>.........] - ETA: 28s - loss: 0.2202 - categorical_accuracy: 0.9324
42560/60000 [====================>.........] - ETA: 28s - loss: 0.2201 - categorical_accuracy: 0.9324
42624/60000 [====================>.........] - ETA: 28s - loss: 0.2201 - categorical_accuracy: 0.9324
42656/60000 [====================>.........] - ETA: 28s - loss: 0.2200 - categorical_accuracy: 0.9325
42688/60000 [====================>.........] - ETA: 28s - loss: 0.2200 - categorical_accuracy: 0.9325
42720/60000 [====================>.........] - ETA: 28s - loss: 0.2200 - categorical_accuracy: 0.9325
42752/60000 [====================>.........] - ETA: 28s - loss: 0.2200 - categorical_accuracy: 0.9325
42784/60000 [====================>.........] - ETA: 28s - loss: 0.2199 - categorical_accuracy: 0.9325
42816/60000 [====================>.........] - ETA: 28s - loss: 0.2198 - categorical_accuracy: 0.9325
42848/60000 [====================>.........] - ETA: 28s - loss: 0.2197 - categorical_accuracy: 0.9326
42880/60000 [====================>.........] - ETA: 28s - loss: 0.2195 - categorical_accuracy: 0.9326
42912/60000 [====================>.........] - ETA: 28s - loss: 0.2194 - categorical_accuracy: 0.9326
42976/60000 [====================>.........] - ETA: 28s - loss: 0.2192 - categorical_accuracy: 0.9327
43008/60000 [====================>.........] - ETA: 27s - loss: 0.2190 - categorical_accuracy: 0.9328
43040/60000 [====================>.........] - ETA: 27s - loss: 0.2189 - categorical_accuracy: 0.9328
43072/60000 [====================>.........] - ETA: 27s - loss: 0.2187 - categorical_accuracy: 0.9329
43136/60000 [====================>.........] - ETA: 27s - loss: 0.2184 - categorical_accuracy: 0.9330
43168/60000 [====================>.........] - ETA: 27s - loss: 0.2183 - categorical_accuracy: 0.9330
43200/60000 [====================>.........] - ETA: 27s - loss: 0.2183 - categorical_accuracy: 0.9330
43232/60000 [====================>.........] - ETA: 27s - loss: 0.2182 - categorical_accuracy: 0.9330
43296/60000 [====================>.........] - ETA: 27s - loss: 0.2180 - categorical_accuracy: 0.9331
43328/60000 [====================>.........] - ETA: 27s - loss: 0.2179 - categorical_accuracy: 0.9331
43360/60000 [====================>.........] - ETA: 27s - loss: 0.2179 - categorical_accuracy: 0.9331
43392/60000 [====================>.........] - ETA: 27s - loss: 0.2178 - categorical_accuracy: 0.9331
43424/60000 [====================>.........] - ETA: 27s - loss: 0.2177 - categorical_accuracy: 0.9331
43456/60000 [====================>.........] - ETA: 27s - loss: 0.2177 - categorical_accuracy: 0.9332
43488/60000 [====================>.........] - ETA: 27s - loss: 0.2176 - categorical_accuracy: 0.9332
43520/60000 [====================>.........] - ETA: 27s - loss: 0.2176 - categorical_accuracy: 0.9332
43584/60000 [====================>.........] - ETA: 27s - loss: 0.2175 - categorical_accuracy: 0.9332
43616/60000 [====================>.........] - ETA: 26s - loss: 0.2174 - categorical_accuracy: 0.9332
43648/60000 [====================>.........] - ETA: 26s - loss: 0.2173 - categorical_accuracy: 0.9333
43680/60000 [====================>.........] - ETA: 26s - loss: 0.2172 - categorical_accuracy: 0.9333
43712/60000 [====================>.........] - ETA: 26s - loss: 0.2172 - categorical_accuracy: 0.9333
43744/60000 [====================>.........] - ETA: 26s - loss: 0.2171 - categorical_accuracy: 0.9333
43776/60000 [====================>.........] - ETA: 26s - loss: 0.2170 - categorical_accuracy: 0.9333
43840/60000 [====================>.........] - ETA: 26s - loss: 0.2169 - categorical_accuracy: 0.9334
43904/60000 [====================>.........] - ETA: 26s - loss: 0.2167 - categorical_accuracy: 0.9334
43936/60000 [====================>.........] - ETA: 26s - loss: 0.2167 - categorical_accuracy: 0.9334
43968/60000 [====================>.........] - ETA: 26s - loss: 0.2166 - categorical_accuracy: 0.9335
44032/60000 [=====================>........] - ETA: 26s - loss: 0.2164 - categorical_accuracy: 0.9335
44064/60000 [=====================>........] - ETA: 26s - loss: 0.2163 - categorical_accuracy: 0.9336
44096/60000 [=====================>........] - ETA: 26s - loss: 0.2163 - categorical_accuracy: 0.9336
44128/60000 [=====================>........] - ETA: 26s - loss: 0.2162 - categorical_accuracy: 0.9336
44192/60000 [=====================>........] - ETA: 26s - loss: 0.2162 - categorical_accuracy: 0.9336
44256/60000 [=====================>........] - ETA: 25s - loss: 0.2160 - categorical_accuracy: 0.9337
44320/60000 [=====================>........] - ETA: 25s - loss: 0.2158 - categorical_accuracy: 0.9337
44352/60000 [=====================>........] - ETA: 25s - loss: 0.2156 - categorical_accuracy: 0.9338
44384/60000 [=====================>........] - ETA: 25s - loss: 0.2155 - categorical_accuracy: 0.9338
44416/60000 [=====================>........] - ETA: 25s - loss: 0.2153 - categorical_accuracy: 0.9339
44448/60000 [=====================>........] - ETA: 25s - loss: 0.2152 - categorical_accuracy: 0.9339
44512/60000 [=====================>........] - ETA: 25s - loss: 0.2149 - categorical_accuracy: 0.9340
44576/60000 [=====================>........] - ETA: 25s - loss: 0.2149 - categorical_accuracy: 0.9340
44608/60000 [=====================>........] - ETA: 25s - loss: 0.2147 - categorical_accuracy: 0.9340
44640/60000 [=====================>........] - ETA: 25s - loss: 0.2146 - categorical_accuracy: 0.9341
44672/60000 [=====================>........] - ETA: 25s - loss: 0.2145 - categorical_accuracy: 0.9341
44704/60000 [=====================>........] - ETA: 25s - loss: 0.2144 - categorical_accuracy: 0.9341
44736/60000 [=====================>........] - ETA: 25s - loss: 0.2143 - categorical_accuracy: 0.9341
44768/60000 [=====================>........] - ETA: 25s - loss: 0.2142 - categorical_accuracy: 0.9341
44832/60000 [=====================>........] - ETA: 24s - loss: 0.2140 - categorical_accuracy: 0.9342
44864/60000 [=====================>........] - ETA: 24s - loss: 0.2139 - categorical_accuracy: 0.9342
44896/60000 [=====================>........] - ETA: 24s - loss: 0.2141 - categorical_accuracy: 0.9342
44960/60000 [=====================>........] - ETA: 24s - loss: 0.2141 - categorical_accuracy: 0.9342
44992/60000 [=====================>........] - ETA: 24s - loss: 0.2142 - categorical_accuracy: 0.9342
45024/60000 [=====================>........] - ETA: 24s - loss: 0.2141 - categorical_accuracy: 0.9342
45088/60000 [=====================>........] - ETA: 24s - loss: 0.2140 - categorical_accuracy: 0.9342
45120/60000 [=====================>........] - ETA: 24s - loss: 0.2139 - categorical_accuracy: 0.9343
45152/60000 [=====================>........] - ETA: 24s - loss: 0.2138 - categorical_accuracy: 0.9343
45184/60000 [=====================>........] - ETA: 24s - loss: 0.2137 - categorical_accuracy: 0.9344
45216/60000 [=====================>........] - ETA: 24s - loss: 0.2136 - categorical_accuracy: 0.9344
45248/60000 [=====================>........] - ETA: 24s - loss: 0.2134 - categorical_accuracy: 0.9344
45280/60000 [=====================>........] - ETA: 24s - loss: 0.2135 - categorical_accuracy: 0.9344
45312/60000 [=====================>........] - ETA: 24s - loss: 0.2134 - categorical_accuracy: 0.9345
45344/60000 [=====================>........] - ETA: 24s - loss: 0.2132 - categorical_accuracy: 0.9345
45376/60000 [=====================>........] - ETA: 24s - loss: 0.2131 - categorical_accuracy: 0.9345
45408/60000 [=====================>........] - ETA: 24s - loss: 0.2130 - categorical_accuracy: 0.9346
45440/60000 [=====================>........] - ETA: 24s - loss: 0.2129 - categorical_accuracy: 0.9346
45472/60000 [=====================>........] - ETA: 23s - loss: 0.2130 - categorical_accuracy: 0.9346
45504/60000 [=====================>........] - ETA: 23s - loss: 0.2129 - categorical_accuracy: 0.9346
45536/60000 [=====================>........] - ETA: 23s - loss: 0.2128 - categorical_accuracy: 0.9346
45568/60000 [=====================>........] - ETA: 23s - loss: 0.2127 - categorical_accuracy: 0.9346
45600/60000 [=====================>........] - ETA: 23s - loss: 0.2126 - categorical_accuracy: 0.9347
45632/60000 [=====================>........] - ETA: 23s - loss: 0.2126 - categorical_accuracy: 0.9347
45664/60000 [=====================>........] - ETA: 23s - loss: 0.2126 - categorical_accuracy: 0.9347
45696/60000 [=====================>........] - ETA: 23s - loss: 0.2125 - categorical_accuracy: 0.9347
45728/60000 [=====================>........] - ETA: 23s - loss: 0.2123 - categorical_accuracy: 0.9348
45792/60000 [=====================>........] - ETA: 23s - loss: 0.2122 - categorical_accuracy: 0.9348
45824/60000 [=====================>........] - ETA: 23s - loss: 0.2122 - categorical_accuracy: 0.9348
45856/60000 [=====================>........] - ETA: 23s - loss: 0.2122 - categorical_accuracy: 0.9348
45920/60000 [=====================>........] - ETA: 23s - loss: 0.2120 - categorical_accuracy: 0.9349
45952/60000 [=====================>........] - ETA: 23s - loss: 0.2119 - categorical_accuracy: 0.9349
46016/60000 [======================>.......] - ETA: 23s - loss: 0.2117 - categorical_accuracy: 0.9350
46048/60000 [======================>.......] - ETA: 23s - loss: 0.2116 - categorical_accuracy: 0.9350
46112/60000 [======================>.......] - ETA: 22s - loss: 0.2114 - categorical_accuracy: 0.9350
46144/60000 [======================>.......] - ETA: 22s - loss: 0.2113 - categorical_accuracy: 0.9351
46176/60000 [======================>.......] - ETA: 22s - loss: 0.2112 - categorical_accuracy: 0.9351
46208/60000 [======================>.......] - ETA: 22s - loss: 0.2111 - categorical_accuracy: 0.9351
46272/60000 [======================>.......] - ETA: 22s - loss: 0.2109 - categorical_accuracy: 0.9352
46336/60000 [======================>.......] - ETA: 22s - loss: 0.2107 - categorical_accuracy: 0.9353
46368/60000 [======================>.......] - ETA: 22s - loss: 0.2106 - categorical_accuracy: 0.9353
46400/60000 [======================>.......] - ETA: 22s - loss: 0.2105 - categorical_accuracy: 0.9353
46432/60000 [======================>.......] - ETA: 22s - loss: 0.2105 - categorical_accuracy: 0.9353
46464/60000 [======================>.......] - ETA: 22s - loss: 0.2104 - categorical_accuracy: 0.9354
46528/60000 [======================>.......] - ETA: 22s - loss: 0.2101 - categorical_accuracy: 0.9355
46592/60000 [======================>.......] - ETA: 22s - loss: 0.2100 - categorical_accuracy: 0.9355
46624/60000 [======================>.......] - ETA: 22s - loss: 0.2100 - categorical_accuracy: 0.9355
46656/60000 [======================>.......] - ETA: 22s - loss: 0.2098 - categorical_accuracy: 0.9356
46688/60000 [======================>.......] - ETA: 21s - loss: 0.2098 - categorical_accuracy: 0.9356
46720/60000 [======================>.......] - ETA: 21s - loss: 0.2097 - categorical_accuracy: 0.9356
46752/60000 [======================>.......] - ETA: 21s - loss: 0.2097 - categorical_accuracy: 0.9356
46784/60000 [======================>.......] - ETA: 21s - loss: 0.2096 - categorical_accuracy: 0.9356
46816/60000 [======================>.......] - ETA: 21s - loss: 0.2095 - categorical_accuracy: 0.9357
46848/60000 [======================>.......] - ETA: 21s - loss: 0.2094 - categorical_accuracy: 0.9357
46880/60000 [======================>.......] - ETA: 21s - loss: 0.2095 - categorical_accuracy: 0.9357
46912/60000 [======================>.......] - ETA: 21s - loss: 0.2093 - categorical_accuracy: 0.9357
46976/60000 [======================>.......] - ETA: 21s - loss: 0.2091 - categorical_accuracy: 0.9358
47040/60000 [======================>.......] - ETA: 21s - loss: 0.2091 - categorical_accuracy: 0.9358
47072/60000 [======================>.......] - ETA: 21s - loss: 0.2089 - categorical_accuracy: 0.9358
47104/60000 [======================>.......] - ETA: 21s - loss: 0.2089 - categorical_accuracy: 0.9358
47136/60000 [======================>.......] - ETA: 21s - loss: 0.2088 - categorical_accuracy: 0.9358
47168/60000 [======================>.......] - ETA: 21s - loss: 0.2087 - categorical_accuracy: 0.9358
47232/60000 [======================>.......] - ETA: 21s - loss: 0.2087 - categorical_accuracy: 0.9359
47264/60000 [======================>.......] - ETA: 21s - loss: 0.2085 - categorical_accuracy: 0.9359
47296/60000 [======================>.......] - ETA: 20s - loss: 0.2084 - categorical_accuracy: 0.9359
47328/60000 [======================>.......] - ETA: 20s - loss: 0.2083 - categorical_accuracy: 0.9360
47360/60000 [======================>.......] - ETA: 20s - loss: 0.2083 - categorical_accuracy: 0.9360
47392/60000 [======================>.......] - ETA: 20s - loss: 0.2082 - categorical_accuracy: 0.9360
47424/60000 [======================>.......] - ETA: 20s - loss: 0.2081 - categorical_accuracy: 0.9360
47456/60000 [======================>.......] - ETA: 20s - loss: 0.2080 - categorical_accuracy: 0.9360
47488/60000 [======================>.......] - ETA: 20s - loss: 0.2079 - categorical_accuracy: 0.9361
47520/60000 [======================>.......] - ETA: 20s - loss: 0.2078 - categorical_accuracy: 0.9361
47584/60000 [======================>.......] - ETA: 20s - loss: 0.2077 - categorical_accuracy: 0.9361
47616/60000 [======================>.......] - ETA: 20s - loss: 0.2077 - categorical_accuracy: 0.9361
47648/60000 [======================>.......] - ETA: 20s - loss: 0.2076 - categorical_accuracy: 0.9362
47680/60000 [======================>.......] - ETA: 20s - loss: 0.2074 - categorical_accuracy: 0.9362
47712/60000 [======================>.......] - ETA: 20s - loss: 0.2074 - categorical_accuracy: 0.9362
47744/60000 [======================>.......] - ETA: 20s - loss: 0.2073 - categorical_accuracy: 0.9362
47808/60000 [======================>.......] - ETA: 20s - loss: 0.2073 - categorical_accuracy: 0.9362
47840/60000 [======================>.......] - ETA: 20s - loss: 0.2072 - categorical_accuracy: 0.9362
47872/60000 [======================>.......] - ETA: 20s - loss: 0.2071 - categorical_accuracy: 0.9363
47936/60000 [======================>.......] - ETA: 19s - loss: 0.2069 - categorical_accuracy: 0.9364
47968/60000 [======================>.......] - ETA: 19s - loss: 0.2070 - categorical_accuracy: 0.9364
48000/60000 [=======================>......] - ETA: 19s - loss: 0.2069 - categorical_accuracy: 0.9364
48032/60000 [=======================>......] - ETA: 19s - loss: 0.2068 - categorical_accuracy: 0.9364
48064/60000 [=======================>......] - ETA: 19s - loss: 0.2067 - categorical_accuracy: 0.9364
48096/60000 [=======================>......] - ETA: 19s - loss: 0.2066 - categorical_accuracy: 0.9365
48128/60000 [=======================>......] - ETA: 19s - loss: 0.2065 - categorical_accuracy: 0.9365
48192/60000 [=======================>......] - ETA: 19s - loss: 0.2064 - categorical_accuracy: 0.9365
48224/60000 [=======================>......] - ETA: 19s - loss: 0.2063 - categorical_accuracy: 0.9366
48256/60000 [=======================>......] - ETA: 19s - loss: 0.2062 - categorical_accuracy: 0.9366
48288/60000 [=======================>......] - ETA: 19s - loss: 0.2061 - categorical_accuracy: 0.9366
48320/60000 [=======================>......] - ETA: 19s - loss: 0.2060 - categorical_accuracy: 0.9367
48352/60000 [=======================>......] - ETA: 19s - loss: 0.2059 - categorical_accuracy: 0.9367
48384/60000 [=======================>......] - ETA: 19s - loss: 0.2058 - categorical_accuracy: 0.9367
48416/60000 [=======================>......] - ETA: 19s - loss: 0.2057 - categorical_accuracy: 0.9368
48448/60000 [=======================>......] - ETA: 19s - loss: 0.2056 - categorical_accuracy: 0.9368
48480/60000 [=======================>......] - ETA: 19s - loss: 0.2056 - categorical_accuracy: 0.9368
48544/60000 [=======================>......] - ETA: 18s - loss: 0.2054 - categorical_accuracy: 0.9368
48576/60000 [=======================>......] - ETA: 18s - loss: 0.2054 - categorical_accuracy: 0.9368
48608/60000 [=======================>......] - ETA: 18s - loss: 0.2053 - categorical_accuracy: 0.9369
48640/60000 [=======================>......] - ETA: 18s - loss: 0.2051 - categorical_accuracy: 0.9369
48672/60000 [=======================>......] - ETA: 18s - loss: 0.2050 - categorical_accuracy: 0.9370
48704/60000 [=======================>......] - ETA: 18s - loss: 0.2049 - categorical_accuracy: 0.9370
48736/60000 [=======================>......] - ETA: 18s - loss: 0.2049 - categorical_accuracy: 0.9370
48768/60000 [=======================>......] - ETA: 18s - loss: 0.2049 - categorical_accuracy: 0.9370
48800/60000 [=======================>......] - ETA: 18s - loss: 0.2048 - categorical_accuracy: 0.9370
48832/60000 [=======================>......] - ETA: 18s - loss: 0.2048 - categorical_accuracy: 0.9370
48864/60000 [=======================>......] - ETA: 18s - loss: 0.2047 - categorical_accuracy: 0.9371
48928/60000 [=======================>......] - ETA: 18s - loss: 0.2045 - categorical_accuracy: 0.9371
48960/60000 [=======================>......] - ETA: 18s - loss: 0.2045 - categorical_accuracy: 0.9371
48992/60000 [=======================>......] - ETA: 18s - loss: 0.2044 - categorical_accuracy: 0.9371
49024/60000 [=======================>......] - ETA: 18s - loss: 0.2044 - categorical_accuracy: 0.9371
49056/60000 [=======================>......] - ETA: 18s - loss: 0.2043 - categorical_accuracy: 0.9372
49088/60000 [=======================>......] - ETA: 18s - loss: 0.2042 - categorical_accuracy: 0.9372
49120/60000 [=======================>......] - ETA: 17s - loss: 0.2042 - categorical_accuracy: 0.9372
49152/60000 [=======================>......] - ETA: 17s - loss: 0.2042 - categorical_accuracy: 0.9372
49184/60000 [=======================>......] - ETA: 17s - loss: 0.2042 - categorical_accuracy: 0.9372
49216/60000 [=======================>......] - ETA: 17s - loss: 0.2041 - categorical_accuracy: 0.9372
49280/60000 [=======================>......] - ETA: 17s - loss: 0.2041 - categorical_accuracy: 0.9372
49312/60000 [=======================>......] - ETA: 17s - loss: 0.2040 - categorical_accuracy: 0.9373
49344/60000 [=======================>......] - ETA: 17s - loss: 0.2040 - categorical_accuracy: 0.9373
49376/60000 [=======================>......] - ETA: 17s - loss: 0.2039 - categorical_accuracy: 0.9373
49408/60000 [=======================>......] - ETA: 17s - loss: 0.2041 - categorical_accuracy: 0.9373
49440/60000 [=======================>......] - ETA: 17s - loss: 0.2041 - categorical_accuracy: 0.9373
49472/60000 [=======================>......] - ETA: 17s - loss: 0.2040 - categorical_accuracy: 0.9373
49504/60000 [=======================>......] - ETA: 17s - loss: 0.2039 - categorical_accuracy: 0.9373
49536/60000 [=======================>......] - ETA: 17s - loss: 0.2039 - categorical_accuracy: 0.9373
49568/60000 [=======================>......] - ETA: 17s - loss: 0.2038 - categorical_accuracy: 0.9373
49600/60000 [=======================>......] - ETA: 17s - loss: 0.2038 - categorical_accuracy: 0.9373
49632/60000 [=======================>......] - ETA: 17s - loss: 0.2037 - categorical_accuracy: 0.9374
49664/60000 [=======================>......] - ETA: 17s - loss: 0.2036 - categorical_accuracy: 0.9374
49696/60000 [=======================>......] - ETA: 17s - loss: 0.2035 - categorical_accuracy: 0.9374
49728/60000 [=======================>......] - ETA: 16s - loss: 0.2034 - categorical_accuracy: 0.9375
49760/60000 [=======================>......] - ETA: 16s - loss: 0.2035 - categorical_accuracy: 0.9375
49792/60000 [=======================>......] - ETA: 16s - loss: 0.2034 - categorical_accuracy: 0.9375
49824/60000 [=======================>......] - ETA: 16s - loss: 0.2033 - categorical_accuracy: 0.9376
49856/60000 [=======================>......] - ETA: 16s - loss: 0.2032 - categorical_accuracy: 0.9376
49888/60000 [=======================>......] - ETA: 16s - loss: 0.2031 - categorical_accuracy: 0.9376
49920/60000 [=======================>......] - ETA: 16s - loss: 0.2030 - categorical_accuracy: 0.9376
49952/60000 [=======================>......] - ETA: 16s - loss: 0.2029 - categorical_accuracy: 0.9377
49984/60000 [=======================>......] - ETA: 16s - loss: 0.2028 - categorical_accuracy: 0.9377
50016/60000 [========================>.....] - ETA: 16s - loss: 0.2027 - categorical_accuracy: 0.9377
50048/60000 [========================>.....] - ETA: 16s - loss: 0.2027 - categorical_accuracy: 0.9377
50080/60000 [========================>.....] - ETA: 16s - loss: 0.2025 - categorical_accuracy: 0.9378
50144/60000 [========================>.....] - ETA: 16s - loss: 0.2025 - categorical_accuracy: 0.9378
50176/60000 [========================>.....] - ETA: 16s - loss: 0.2024 - categorical_accuracy: 0.9378
50240/60000 [========================>.....] - ETA: 16s - loss: 0.2023 - categorical_accuracy: 0.9378
50272/60000 [========================>.....] - ETA: 16s - loss: 0.2021 - categorical_accuracy: 0.9379
50304/60000 [========================>.....] - ETA: 16s - loss: 0.2020 - categorical_accuracy: 0.9379
50336/60000 [========================>.....] - ETA: 15s - loss: 0.2020 - categorical_accuracy: 0.9379
50368/60000 [========================>.....] - ETA: 15s - loss: 0.2020 - categorical_accuracy: 0.9379
50432/60000 [========================>.....] - ETA: 15s - loss: 0.2019 - categorical_accuracy: 0.9380
50464/60000 [========================>.....] - ETA: 15s - loss: 0.2018 - categorical_accuracy: 0.9380
50496/60000 [========================>.....] - ETA: 15s - loss: 0.2017 - categorical_accuracy: 0.9380
50528/60000 [========================>.....] - ETA: 15s - loss: 0.2017 - categorical_accuracy: 0.9380
50560/60000 [========================>.....] - ETA: 15s - loss: 0.2017 - categorical_accuracy: 0.9380
50592/60000 [========================>.....] - ETA: 15s - loss: 0.2016 - categorical_accuracy: 0.9381
50624/60000 [========================>.....] - ETA: 15s - loss: 0.2015 - categorical_accuracy: 0.9381
50656/60000 [========================>.....] - ETA: 15s - loss: 0.2014 - categorical_accuracy: 0.9381
50688/60000 [========================>.....] - ETA: 15s - loss: 0.2013 - categorical_accuracy: 0.9381
50720/60000 [========================>.....] - ETA: 15s - loss: 0.2012 - categorical_accuracy: 0.9381
50784/60000 [========================>.....] - ETA: 15s - loss: 0.2013 - categorical_accuracy: 0.9381
50816/60000 [========================>.....] - ETA: 15s - loss: 0.2012 - categorical_accuracy: 0.9381
50848/60000 [========================>.....] - ETA: 15s - loss: 0.2011 - categorical_accuracy: 0.9382
50880/60000 [========================>.....] - ETA: 15s - loss: 0.2010 - categorical_accuracy: 0.9382
50912/60000 [========================>.....] - ETA: 15s - loss: 0.2011 - categorical_accuracy: 0.9382
50944/60000 [========================>.....] - ETA: 14s - loss: 0.2009 - categorical_accuracy: 0.9382
50976/60000 [========================>.....] - ETA: 14s - loss: 0.2009 - categorical_accuracy: 0.9382
51008/60000 [========================>.....] - ETA: 14s - loss: 0.2008 - categorical_accuracy: 0.9382
51040/60000 [========================>.....] - ETA: 14s - loss: 0.2007 - categorical_accuracy: 0.9383
51072/60000 [========================>.....] - ETA: 14s - loss: 0.2006 - categorical_accuracy: 0.9383
51104/60000 [========================>.....] - ETA: 14s - loss: 0.2005 - categorical_accuracy: 0.9383
51168/60000 [========================>.....] - ETA: 14s - loss: 0.2003 - categorical_accuracy: 0.9384
51200/60000 [========================>.....] - ETA: 14s - loss: 0.2003 - categorical_accuracy: 0.9384
51232/60000 [========================>.....] - ETA: 14s - loss: 0.2003 - categorical_accuracy: 0.9384
51264/60000 [========================>.....] - ETA: 14s - loss: 0.2002 - categorical_accuracy: 0.9385
51296/60000 [========================>.....] - ETA: 14s - loss: 0.2001 - categorical_accuracy: 0.9385
51360/60000 [========================>.....] - ETA: 14s - loss: 0.1999 - categorical_accuracy: 0.9386
51392/60000 [========================>.....] - ETA: 14s - loss: 0.1998 - categorical_accuracy: 0.9386
51456/60000 [========================>.....] - ETA: 14s - loss: 0.1999 - categorical_accuracy: 0.9386
51488/60000 [========================>.....] - ETA: 14s - loss: 0.1999 - categorical_accuracy: 0.9386
51520/60000 [========================>.....] - ETA: 14s - loss: 0.1998 - categorical_accuracy: 0.9386
51552/60000 [========================>.....] - ETA: 13s - loss: 0.1996 - categorical_accuracy: 0.9387
51616/60000 [========================>.....] - ETA: 13s - loss: 0.1995 - categorical_accuracy: 0.9387
51648/60000 [========================>.....] - ETA: 13s - loss: 0.1994 - categorical_accuracy: 0.9388
51680/60000 [========================>.....] - ETA: 13s - loss: 0.1993 - categorical_accuracy: 0.9388
51712/60000 [========================>.....] - ETA: 13s - loss: 0.1992 - categorical_accuracy: 0.9388
51744/60000 [========================>.....] - ETA: 13s - loss: 0.1991 - categorical_accuracy: 0.9389
51808/60000 [========================>.....] - ETA: 13s - loss: 0.1990 - categorical_accuracy: 0.9389
51840/60000 [========================>.....] - ETA: 13s - loss: 0.1988 - categorical_accuracy: 0.9389
51872/60000 [========================>.....] - ETA: 13s - loss: 0.1988 - categorical_accuracy: 0.9390
51904/60000 [========================>.....] - ETA: 13s - loss: 0.1988 - categorical_accuracy: 0.9390
51936/60000 [========================>.....] - ETA: 13s - loss: 0.1990 - categorical_accuracy: 0.9389
51968/60000 [========================>.....] - ETA: 13s - loss: 0.1990 - categorical_accuracy: 0.9389
52000/60000 [=========================>....] - ETA: 13s - loss: 0.1989 - categorical_accuracy: 0.9389
52032/60000 [=========================>....] - ETA: 13s - loss: 0.1989 - categorical_accuracy: 0.9389
52064/60000 [=========================>....] - ETA: 13s - loss: 0.1988 - categorical_accuracy: 0.9389
52096/60000 [=========================>....] - ETA: 13s - loss: 0.1987 - categorical_accuracy: 0.9389
52128/60000 [=========================>....] - ETA: 13s - loss: 0.1986 - categorical_accuracy: 0.9390
52192/60000 [=========================>....] - ETA: 12s - loss: 0.1984 - categorical_accuracy: 0.9390
52256/60000 [=========================>....] - ETA: 12s - loss: 0.1983 - categorical_accuracy: 0.9391
52288/60000 [=========================>....] - ETA: 12s - loss: 0.1982 - categorical_accuracy: 0.9391
52320/60000 [=========================>....] - ETA: 12s - loss: 0.1981 - categorical_accuracy: 0.9391
52352/60000 [=========================>....] - ETA: 12s - loss: 0.1980 - categorical_accuracy: 0.9391
52384/60000 [=========================>....] - ETA: 12s - loss: 0.1980 - categorical_accuracy: 0.9392
52416/60000 [=========================>....] - ETA: 12s - loss: 0.1979 - categorical_accuracy: 0.9392
52448/60000 [=========================>....] - ETA: 12s - loss: 0.1979 - categorical_accuracy: 0.9392
52480/60000 [=========================>....] - ETA: 12s - loss: 0.1978 - categorical_accuracy: 0.9392
52512/60000 [=========================>....] - ETA: 12s - loss: 0.1978 - categorical_accuracy: 0.9392
52544/60000 [=========================>....] - ETA: 12s - loss: 0.1978 - categorical_accuracy: 0.9392
52576/60000 [=========================>....] - ETA: 12s - loss: 0.1977 - categorical_accuracy: 0.9392
52608/60000 [=========================>....] - ETA: 12s - loss: 0.1976 - categorical_accuracy: 0.9392
52640/60000 [=========================>....] - ETA: 12s - loss: 0.1975 - categorical_accuracy: 0.9393
52672/60000 [=========================>....] - ETA: 12s - loss: 0.1974 - categorical_accuracy: 0.9393
52704/60000 [=========================>....] - ETA: 12s - loss: 0.1973 - categorical_accuracy: 0.9394
52736/60000 [=========================>....] - ETA: 12s - loss: 0.1972 - categorical_accuracy: 0.9394
52768/60000 [=========================>....] - ETA: 11s - loss: 0.1972 - categorical_accuracy: 0.9394
52832/60000 [=========================>....] - ETA: 11s - loss: 0.1970 - categorical_accuracy: 0.9394
52864/60000 [=========================>....] - ETA: 11s - loss: 0.1969 - categorical_accuracy: 0.9395
52896/60000 [=========================>....] - ETA: 11s - loss: 0.1969 - categorical_accuracy: 0.9395
52928/60000 [=========================>....] - ETA: 11s - loss: 0.1967 - categorical_accuracy: 0.9395
52992/60000 [=========================>....] - ETA: 11s - loss: 0.1966 - categorical_accuracy: 0.9396
53024/60000 [=========================>....] - ETA: 11s - loss: 0.1965 - categorical_accuracy: 0.9396
53056/60000 [=========================>....] - ETA: 11s - loss: 0.1964 - categorical_accuracy: 0.9396
53088/60000 [=========================>....] - ETA: 11s - loss: 0.1964 - categorical_accuracy: 0.9396
53152/60000 [=========================>....] - ETA: 11s - loss: 0.1962 - categorical_accuracy: 0.9397
53184/60000 [=========================>....] - ETA: 11s - loss: 0.1962 - categorical_accuracy: 0.9397
53216/60000 [=========================>....] - ETA: 11s - loss: 0.1962 - categorical_accuracy: 0.9397
53280/60000 [=========================>....] - ETA: 11s - loss: 0.1961 - categorical_accuracy: 0.9397
53344/60000 [=========================>....] - ETA: 11s - loss: 0.1960 - categorical_accuracy: 0.9397
53376/60000 [=========================>....] - ETA: 10s - loss: 0.1959 - categorical_accuracy: 0.9398
53440/60000 [=========================>....] - ETA: 10s - loss: 0.1958 - categorical_accuracy: 0.9398
53504/60000 [=========================>....] - ETA: 10s - loss: 0.1956 - categorical_accuracy: 0.9399
53536/60000 [=========================>....] - ETA: 10s - loss: 0.1955 - categorical_accuracy: 0.9399
53568/60000 [=========================>....] - ETA: 10s - loss: 0.1954 - categorical_accuracy: 0.9399
53632/60000 [=========================>....] - ETA: 10s - loss: 0.1954 - categorical_accuracy: 0.9399
53664/60000 [=========================>....] - ETA: 10s - loss: 0.1953 - categorical_accuracy: 0.9399
53696/60000 [=========================>....] - ETA: 10s - loss: 0.1954 - categorical_accuracy: 0.9399
53728/60000 [=========================>....] - ETA: 10s - loss: 0.1953 - categorical_accuracy: 0.9400
53760/60000 [=========================>....] - ETA: 10s - loss: 0.1953 - categorical_accuracy: 0.9399
53792/60000 [=========================>....] - ETA: 10s - loss: 0.1952 - categorical_accuracy: 0.9400
53856/60000 [=========================>....] - ETA: 10s - loss: 0.1952 - categorical_accuracy: 0.9400
53888/60000 [=========================>....] - ETA: 10s - loss: 0.1951 - categorical_accuracy: 0.9400
53952/60000 [=========================>....] - ETA: 10s - loss: 0.1950 - categorical_accuracy: 0.9401
53984/60000 [=========================>....] - ETA: 9s - loss: 0.1950 - categorical_accuracy: 0.9401 
54016/60000 [==========================>...] - ETA: 9s - loss: 0.1949 - categorical_accuracy: 0.9401
54048/60000 [==========================>...] - ETA: 9s - loss: 0.1948 - categorical_accuracy: 0.9401
54080/60000 [==========================>...] - ETA: 9s - loss: 0.1947 - categorical_accuracy: 0.9401
54112/60000 [==========================>...] - ETA: 9s - loss: 0.1946 - categorical_accuracy: 0.9402
54144/60000 [==========================>...] - ETA: 9s - loss: 0.1945 - categorical_accuracy: 0.9402
54176/60000 [==========================>...] - ETA: 9s - loss: 0.1944 - categorical_accuracy: 0.9402
54208/60000 [==========================>...] - ETA: 9s - loss: 0.1943 - categorical_accuracy: 0.9403
54240/60000 [==========================>...] - ETA: 9s - loss: 0.1942 - categorical_accuracy: 0.9403
54272/60000 [==========================>...] - ETA: 9s - loss: 0.1942 - categorical_accuracy: 0.9403
54304/60000 [==========================>...] - ETA: 9s - loss: 0.1941 - categorical_accuracy: 0.9403
54336/60000 [==========================>...] - ETA: 9s - loss: 0.1941 - categorical_accuracy: 0.9403
54368/60000 [==========================>...] - ETA: 9s - loss: 0.1940 - categorical_accuracy: 0.9403
54400/60000 [==========================>...] - ETA: 9s - loss: 0.1939 - categorical_accuracy: 0.9403
54464/60000 [==========================>...] - ETA: 9s - loss: 0.1938 - categorical_accuracy: 0.9404
54496/60000 [==========================>...] - ETA: 9s - loss: 0.1938 - categorical_accuracy: 0.9404
54528/60000 [==========================>...] - ETA: 9s - loss: 0.1937 - categorical_accuracy: 0.9404
54560/60000 [==========================>...] - ETA: 9s - loss: 0.1937 - categorical_accuracy: 0.9405
54592/60000 [==========================>...] - ETA: 8s - loss: 0.1936 - categorical_accuracy: 0.9405
54624/60000 [==========================>...] - ETA: 8s - loss: 0.1935 - categorical_accuracy: 0.9405
54656/60000 [==========================>...] - ETA: 8s - loss: 0.1935 - categorical_accuracy: 0.9405
54688/60000 [==========================>...] - ETA: 8s - loss: 0.1934 - categorical_accuracy: 0.9405
54720/60000 [==========================>...] - ETA: 8s - loss: 0.1933 - categorical_accuracy: 0.9406
54752/60000 [==========================>...] - ETA: 8s - loss: 0.1934 - categorical_accuracy: 0.9405
54784/60000 [==========================>...] - ETA: 8s - loss: 0.1934 - categorical_accuracy: 0.9405
54816/60000 [==========================>...] - ETA: 8s - loss: 0.1933 - categorical_accuracy: 0.9405
54848/60000 [==========================>...] - ETA: 8s - loss: 0.1932 - categorical_accuracy: 0.9406
54880/60000 [==========================>...] - ETA: 8s - loss: 0.1931 - categorical_accuracy: 0.9406
54912/60000 [==========================>...] - ETA: 8s - loss: 0.1931 - categorical_accuracy: 0.9406
54976/60000 [==========================>...] - ETA: 8s - loss: 0.1930 - categorical_accuracy: 0.9407
55040/60000 [==========================>...] - ETA: 8s - loss: 0.1930 - categorical_accuracy: 0.9407
55104/60000 [==========================>...] - ETA: 8s - loss: 0.1928 - categorical_accuracy: 0.9408
55136/60000 [==========================>...] - ETA: 8s - loss: 0.1927 - categorical_accuracy: 0.9408
55168/60000 [==========================>...] - ETA: 7s - loss: 0.1927 - categorical_accuracy: 0.9408
55200/60000 [==========================>...] - ETA: 7s - loss: 0.1926 - categorical_accuracy: 0.9408
55232/60000 [==========================>...] - ETA: 7s - loss: 0.1926 - categorical_accuracy: 0.9408
55296/60000 [==========================>...] - ETA: 7s - loss: 0.1924 - categorical_accuracy: 0.9409
55328/60000 [==========================>...] - ETA: 7s - loss: 0.1924 - categorical_accuracy: 0.9409
55360/60000 [==========================>...] - ETA: 7s - loss: 0.1924 - categorical_accuracy: 0.9409
55392/60000 [==========================>...] - ETA: 7s - loss: 0.1923 - categorical_accuracy: 0.9409
55424/60000 [==========================>...] - ETA: 7s - loss: 0.1922 - categorical_accuracy: 0.9409
55456/60000 [==========================>...] - ETA: 7s - loss: 0.1922 - categorical_accuracy: 0.9409
55488/60000 [==========================>...] - ETA: 7s - loss: 0.1921 - categorical_accuracy: 0.9410
55520/60000 [==========================>...] - ETA: 7s - loss: 0.1921 - categorical_accuracy: 0.9410
55552/60000 [==========================>...] - ETA: 7s - loss: 0.1920 - categorical_accuracy: 0.9410
55584/60000 [==========================>...] - ETA: 7s - loss: 0.1920 - categorical_accuracy: 0.9410
55616/60000 [==========================>...] - ETA: 7s - loss: 0.1919 - categorical_accuracy: 0.9410
55648/60000 [==========================>...] - ETA: 7s - loss: 0.1918 - categorical_accuracy: 0.9411
55680/60000 [==========================>...] - ETA: 7s - loss: 0.1918 - categorical_accuracy: 0.9411
55712/60000 [==========================>...] - ETA: 7s - loss: 0.1918 - categorical_accuracy: 0.9411
55744/60000 [==========================>...] - ETA: 7s - loss: 0.1917 - categorical_accuracy: 0.9411
55776/60000 [==========================>...] - ETA: 6s - loss: 0.1916 - categorical_accuracy: 0.9411
55808/60000 [==========================>...] - ETA: 6s - loss: 0.1916 - categorical_accuracy: 0.9411
55872/60000 [==========================>...] - ETA: 6s - loss: 0.1914 - categorical_accuracy: 0.9412
55904/60000 [==========================>...] - ETA: 6s - loss: 0.1914 - categorical_accuracy: 0.9412
55968/60000 [==========================>...] - ETA: 6s - loss: 0.1913 - categorical_accuracy: 0.9412
56032/60000 [===========================>..] - ETA: 6s - loss: 0.1911 - categorical_accuracy: 0.9412
56064/60000 [===========================>..] - ETA: 6s - loss: 0.1911 - categorical_accuracy: 0.9413
56096/60000 [===========================>..] - ETA: 6s - loss: 0.1910 - categorical_accuracy: 0.9413
56160/60000 [===========================>..] - ETA: 6s - loss: 0.1909 - categorical_accuracy: 0.9413
56224/60000 [===========================>..] - ETA: 6s - loss: 0.1908 - categorical_accuracy: 0.9414
56256/60000 [===========================>..] - ETA: 6s - loss: 0.1907 - categorical_accuracy: 0.9414
56288/60000 [===========================>..] - ETA: 6s - loss: 0.1906 - categorical_accuracy: 0.9414
56320/60000 [===========================>..] - ETA: 6s - loss: 0.1906 - categorical_accuracy: 0.9414
56352/60000 [===========================>..] - ETA: 6s - loss: 0.1906 - categorical_accuracy: 0.9414
56384/60000 [===========================>..] - ETA: 5s - loss: 0.1905 - categorical_accuracy: 0.9415
56416/60000 [===========================>..] - ETA: 5s - loss: 0.1904 - categorical_accuracy: 0.9415
56448/60000 [===========================>..] - ETA: 5s - loss: 0.1903 - categorical_accuracy: 0.9415
56480/60000 [===========================>..] - ETA: 5s - loss: 0.1903 - categorical_accuracy: 0.9415
56512/60000 [===========================>..] - ETA: 5s - loss: 0.1903 - categorical_accuracy: 0.9415
56544/60000 [===========================>..] - ETA: 5s - loss: 0.1903 - categorical_accuracy: 0.9415
56576/60000 [===========================>..] - ETA: 5s - loss: 0.1902 - categorical_accuracy: 0.9415
56608/60000 [===========================>..] - ETA: 5s - loss: 0.1901 - categorical_accuracy: 0.9416
56640/60000 [===========================>..] - ETA: 5s - loss: 0.1900 - categorical_accuracy: 0.9416
56672/60000 [===========================>..] - ETA: 5s - loss: 0.1899 - categorical_accuracy: 0.9416
56704/60000 [===========================>..] - ETA: 5s - loss: 0.1899 - categorical_accuracy: 0.9416
56736/60000 [===========================>..] - ETA: 5s - loss: 0.1898 - categorical_accuracy: 0.9417
56768/60000 [===========================>..] - ETA: 5s - loss: 0.1897 - categorical_accuracy: 0.9417
56800/60000 [===========================>..] - ETA: 5s - loss: 0.1899 - categorical_accuracy: 0.9417
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1898 - categorical_accuracy: 0.9417
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1898 - categorical_accuracy: 0.9417
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1897 - categorical_accuracy: 0.9417
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1897 - categorical_accuracy: 0.9417
56960/60000 [===========================>..] - ETA: 5s - loss: 0.1896 - categorical_accuracy: 0.9417
56992/60000 [===========================>..] - ETA: 4s - loss: 0.1895 - categorical_accuracy: 0.9417
57024/60000 [===========================>..] - ETA: 4s - loss: 0.1895 - categorical_accuracy: 0.9417
57056/60000 [===========================>..] - ETA: 4s - loss: 0.1895 - categorical_accuracy: 0.9417
57088/60000 [===========================>..] - ETA: 4s - loss: 0.1895 - categorical_accuracy: 0.9417
57152/60000 [===========================>..] - ETA: 4s - loss: 0.1895 - categorical_accuracy: 0.9418
57216/60000 [===========================>..] - ETA: 4s - loss: 0.1893 - categorical_accuracy: 0.9418
57248/60000 [===========================>..] - ETA: 4s - loss: 0.1892 - categorical_accuracy: 0.9418
57280/60000 [===========================>..] - ETA: 4s - loss: 0.1891 - categorical_accuracy: 0.9419
57312/60000 [===========================>..] - ETA: 4s - loss: 0.1891 - categorical_accuracy: 0.9419
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1891 - categorical_accuracy: 0.9419
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1890 - categorical_accuracy: 0.9419
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1890 - categorical_accuracy: 0.9419
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1889 - categorical_accuracy: 0.9420
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1888 - categorical_accuracy: 0.9420
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1887 - categorical_accuracy: 0.9420
57600/60000 [===========================>..] - ETA: 3s - loss: 0.1886 - categorical_accuracy: 0.9420
57632/60000 [===========================>..] - ETA: 3s - loss: 0.1887 - categorical_accuracy: 0.9420
57664/60000 [===========================>..] - ETA: 3s - loss: 0.1888 - categorical_accuracy: 0.9420
57696/60000 [===========================>..] - ETA: 3s - loss: 0.1887 - categorical_accuracy: 0.9420
57728/60000 [===========================>..] - ETA: 3s - loss: 0.1886 - categorical_accuracy: 0.9421
57760/60000 [===========================>..] - ETA: 3s - loss: 0.1885 - categorical_accuracy: 0.9421
57792/60000 [===========================>..] - ETA: 3s - loss: 0.1884 - categorical_accuracy: 0.9421
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1883 - categorical_accuracy: 0.9421
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1882 - categorical_accuracy: 0.9422
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1882 - categorical_accuracy: 0.9422
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1881 - categorical_accuracy: 0.9422
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1881 - categorical_accuracy: 0.9422
58016/60000 [============================>.] - ETA: 3s - loss: 0.1880 - categorical_accuracy: 0.9423
58048/60000 [============================>.] - ETA: 3s - loss: 0.1879 - categorical_accuracy: 0.9423
58080/60000 [============================>.] - ETA: 3s - loss: 0.1878 - categorical_accuracy: 0.9423
58112/60000 [============================>.] - ETA: 3s - loss: 0.1877 - categorical_accuracy: 0.9423
58144/60000 [============================>.] - ETA: 3s - loss: 0.1876 - categorical_accuracy: 0.9424
58176/60000 [============================>.] - ETA: 3s - loss: 0.1875 - categorical_accuracy: 0.9424
58208/60000 [============================>.] - ETA: 2s - loss: 0.1875 - categorical_accuracy: 0.9424
58272/60000 [============================>.] - ETA: 2s - loss: 0.1874 - categorical_accuracy: 0.9424
58336/60000 [============================>.] - ETA: 2s - loss: 0.1874 - categorical_accuracy: 0.9424
58400/60000 [============================>.] - ETA: 2s - loss: 0.1873 - categorical_accuracy: 0.9425
58464/60000 [============================>.] - ETA: 2s - loss: 0.1872 - categorical_accuracy: 0.9425
58528/60000 [============================>.] - ETA: 2s - loss: 0.1870 - categorical_accuracy: 0.9425
58592/60000 [============================>.] - ETA: 2s - loss: 0.1871 - categorical_accuracy: 0.9426
58656/60000 [============================>.] - ETA: 2s - loss: 0.1869 - categorical_accuracy: 0.9426
58688/60000 [============================>.] - ETA: 2s - loss: 0.1870 - categorical_accuracy: 0.9426
58720/60000 [============================>.] - ETA: 2s - loss: 0.1869 - categorical_accuracy: 0.9427
58752/60000 [============================>.] - ETA: 2s - loss: 0.1868 - categorical_accuracy: 0.9427
58784/60000 [============================>.] - ETA: 2s - loss: 0.1868 - categorical_accuracy: 0.9427
58816/60000 [============================>.] - ETA: 1s - loss: 0.1867 - categorical_accuracy: 0.9427
58848/60000 [============================>.] - ETA: 1s - loss: 0.1867 - categorical_accuracy: 0.9428
58912/60000 [============================>.] - ETA: 1s - loss: 0.1866 - categorical_accuracy: 0.9428
58976/60000 [============================>.] - ETA: 1s - loss: 0.1865 - categorical_accuracy: 0.9428
59040/60000 [============================>.] - ETA: 1s - loss: 0.1863 - categorical_accuracy: 0.9429
59104/60000 [============================>.] - ETA: 1s - loss: 0.1862 - categorical_accuracy: 0.9429
59168/60000 [============================>.] - ETA: 1s - loss: 0.1861 - categorical_accuracy: 0.9429
59232/60000 [============================>.] - ETA: 1s - loss: 0.1859 - categorical_accuracy: 0.9430
59264/60000 [============================>.] - ETA: 1s - loss: 0.1859 - categorical_accuracy: 0.9430
59296/60000 [============================>.] - ETA: 1s - loss: 0.1858 - categorical_accuracy: 0.9430
59360/60000 [============================>.] - ETA: 1s - loss: 0.1858 - categorical_accuracy: 0.9430
59424/60000 [============================>.] - ETA: 0s - loss: 0.1857 - categorical_accuracy: 0.9431
59488/60000 [============================>.] - ETA: 0s - loss: 0.1856 - categorical_accuracy: 0.9431
59520/60000 [============================>.] - ETA: 0s - loss: 0.1855 - categorical_accuracy: 0.9431
59552/60000 [============================>.] - ETA: 0s - loss: 0.1855 - categorical_accuracy: 0.9431
59616/60000 [============================>.] - ETA: 0s - loss: 0.1853 - categorical_accuracy: 0.9432
59648/60000 [============================>.] - ETA: 0s - loss: 0.1853 - categorical_accuracy: 0.9432
59680/60000 [============================>.] - ETA: 0s - loss: 0.1853 - categorical_accuracy: 0.9432
59712/60000 [============================>.] - ETA: 0s - loss: 0.1852 - categorical_accuracy: 0.9432
59744/60000 [============================>.] - ETA: 0s - loss: 0.1852 - categorical_accuracy: 0.9432
59776/60000 [============================>.] - ETA: 0s - loss: 0.1851 - categorical_accuracy: 0.9432
59808/60000 [============================>.] - ETA: 0s - loss: 0.1850 - categorical_accuracy: 0.9433
59872/60000 [============================>.] - ETA: 0s - loss: 0.1849 - categorical_accuracy: 0.9433
59904/60000 [============================>.] - ETA: 0s - loss: 0.1848 - categorical_accuracy: 0.9433
59936/60000 [============================>.] - ETA: 0s - loss: 0.1847 - categorical_accuracy: 0.9434
60000/60000 [==============================] - 102s 2ms/step - loss: 0.1846 - categorical_accuracy: 0.9434 - val_loss: 0.0499 - val_categorical_accuracy: 0.9832

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 14s
  224/10000 [..............................] - ETA: 4s 
  384/10000 [>.............................] - ETA: 3s
  576/10000 [>.............................] - ETA: 3s
  768/10000 [=>............................] - ETA: 3s
  960/10000 [=>............................] - ETA: 3s
 1152/10000 [==>...........................] - ETA: 2s
 1344/10000 [===>..........................] - ETA: 2s
 1536/10000 [===>..........................] - ETA: 2s
 1728/10000 [====>.........................] - ETA: 2s
 1920/10000 [====>.........................] - ETA: 2s
 2112/10000 [=====>........................] - ETA: 2s
 2304/10000 [=====>........................] - ETA: 2s
 2496/10000 [======>.......................] - ETA: 2s
 2688/10000 [=======>......................] - ETA: 2s
 2880/10000 [=======>......................] - ETA: 2s
 3072/10000 [========>.....................] - ETA: 2s
 3232/10000 [========>.....................] - ETA: 2s
 3392/10000 [=========>....................] - ETA: 2s
 3584/10000 [=========>....................] - ETA: 2s
 3744/10000 [==========>...................] - ETA: 1s
 3936/10000 [==========>...................] - ETA: 1s
 4128/10000 [===========>..................] - ETA: 1s
 4320/10000 [===========>..................] - ETA: 1s
 4480/10000 [============>.................] - ETA: 1s
 4672/10000 [=============>................] - ETA: 1s
 4864/10000 [=============>................] - ETA: 1s
 5056/10000 [==============>...............] - ETA: 1s
 5248/10000 [==============>...............] - ETA: 1s
 5440/10000 [===============>..............] - ETA: 1s
 5632/10000 [===============>..............] - ETA: 1s
 5824/10000 [================>.............] - ETA: 1s
 5984/10000 [================>.............] - ETA: 1s
 6176/10000 [=================>............] - ETA: 1s
 6368/10000 [==================>...........] - ETA: 1s
 6560/10000 [==================>...........] - ETA: 1s
 6752/10000 [===================>..........] - ETA: 1s
 6944/10000 [===================>..........] - ETA: 0s
 7136/10000 [====================>.........] - ETA: 0s
 7296/10000 [====================>.........] - ETA: 0s
 7488/10000 [=====================>........] - ETA: 0s
 7680/10000 [======================>.......] - ETA: 0s
 7872/10000 [======================>.......] - ETA: 0s
 8064/10000 [=======================>......] - ETA: 0s
 8224/10000 [=======================>......] - ETA: 0s
 8416/10000 [========================>.....] - ETA: 0s
 8608/10000 [========================>.....] - ETA: 0s
 8768/10000 [=========================>....] - ETA: 0s
 8896/10000 [=========================>....] - ETA: 0s
 9056/10000 [==========================>...] - ETA: 0s
 9216/10000 [==========================>...] - ETA: 0s
 9408/10000 [===========================>..] - ETA: 0s
 9600/10000 [===========================>..] - ETA: 0s
 9792/10000 [============================>.] - ETA: 0s
 9984/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 3s 310us/step
[[1.8924922e-09 1.9004647e-09 7.1789479e-08 ... 9.9999976e-01
  7.7125217e-10 1.5022705e-07]
 [8.5856693e-05 5.0015306e-06 9.9981040e-01 ... 8.8626946e-09
  3.5712094e-06 2.0826288e-10]
 [2.1052770e-07 9.9993038e-01 8.1173021e-06 ... 2.6068326e-05
  5.7164661e-06 1.9404747e-07]
 ...
 [1.3258482e-10 1.5297994e-07 3.2510312e-09 ... 9.1300029e-07
  6.3021861e-07 3.3267549e-06]
 [3.1759995e-05 5.4429256e-07 1.1573924e-07 ... 1.0747978e-06
  5.8198934e-03 2.0531957e-05]
 [4.2825101e-07 4.4615714e-08 2.9237123e-07 ... 1.5597598e-10
  1.0971195e-08 2.1264128e-09]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.04985158218088327, 'accuracy_test:': 0.9832000136375427}

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
   c3562e2..1764d42  master     -> origin/master
Updating c3562e2..1764d42
Fast-forward
 error_list/20200517/list_log_jupyter_20200517.md  | 1749 +++++++++++----------
 error_list/20200517/list_log_test_cli_20200517.md |  138 +-
 error_list/20200517/list_log_testall_20200517.md  |  386 ++---
 3 files changed, 1117 insertions(+), 1156 deletions(-)
[master 2c92eab] ml_store
 1 file changed, 1713 insertions(+)
To github.com:arita37/mlmodels_store.git
   1764d42..2c92eab  master -> master





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
{'loss': 0.39880717173218727, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-17 20:26:09.571280: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
[master cc83c0c] ml_store
 1 file changed, 234 insertions(+)
To github.com:arita37/mlmodels_store.git
   2c92eab..cc83c0c  master -> master





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
[master 88aeb60] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   cc83c0c..88aeb60  master -> master





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
	Data preprocessing and feature engineering runtime = 0.26s ...
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
 40%|████      | 2/5 [00:20<00:31, 10.44s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.9698348262280712, 'learning_rate': 0.08179848346396598, 'min_data_in_leaf': 16, 'num_leaves': 64} and reward: 0.3888
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xef\x08\xe3\x0b\xa5|.X\r\x00\x00\x00learning_rateq\x02G?\xb4\xf0\xbe\xd3W\x12\xceX\x10\x00\x00\x00min_data_in_leafq\x03K\x10X\n\x00\x00\x00num_leavesq\x04K@u.' and reward: 0.3888
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xef\x08\xe3\x0b\xa5|.X\r\x00\x00\x00learning_rateq\x02G?\xb4\xf0\xbe\xd3W\x12\xceX\x10\x00\x00\x00min_data_in_leafq\x03K\x10X\n\x00\x00\x00num_leavesq\x04K@u.' and reward: 0.3888
 60%|██████    | 3/5 [00:48<00:31, 15.69s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.9474693845749683, 'learning_rate': 0.01265784494158128, 'min_data_in_leaf': 17, 'num_leaves': 57} and reward: 0.3896
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeeQ\xabP\x96\xbeYX\r\x00\x00\x00learning_rateq\x02G?\x89\xec[0~\xd5|X\x10\x00\x00\x00min_data_in_leafq\x03K\x11X\n\x00\x00\x00num_leavesq\x04K9u.' and reward: 0.3896
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeeQ\xabP\x96\xbeYX\r\x00\x00\x00learning_rateq\x02G?\x89\xec[0~\xd5|X\x10\x00\x00\x00min_data_in_leafq\x03K\x11X\n\x00\x00\x00num_leavesq\x04K9u.' and reward: 0.3896
 80%|████████  | 4/5 [01:13<00:18, 18.48s/it] 80%|████████  | 4/5 [01:13<00:18, 18.45s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.7565665481800391, 'learning_rate': 0.05183771007056573, 'min_data_in_leaf': 18, 'num_leaves': 47} and reward: 0.3934
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe85\xcb\x0c\xb5\xc9\xbdX\r\x00\x00\x00learning_rateq\x02G?\xaa\x8ax\xea\xe7\xbctX\x10\x00\x00\x00min_data_in_leafq\x03K\x12X\n\x00\x00\x00num_leavesq\x04K/u.' and reward: 0.3934
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe85\xcb\x0c\xb5\xc9\xbdX\r\x00\x00\x00learning_rateq\x02G?\xaa\x8ax\xea\xe7\xbctX\x10\x00\x00\x00min_data_in_leafq\x03K\x12X\n\x00\x00\x00num_leavesq\x04K/u.' and reward: 0.3934
Time for Gradient Boosting hyperparameter optimization: 95.43663334846497
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.7565665481800391, 'learning_rate': 0.05183771007056573, 'min_data_in_leaf': 18, 'num_leaves': 47}
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
 40%|████      | 2/5 [00:44<01:06, 22.00s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.3130947077300143, 'embedding_size_factor': 1.0100004653101298, 'layers.choice': 1, 'learning_rate': 0.002355898784786277, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 1.5019196538711226e-06} and reward: 0.3724
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd4\t\xbeb\x90\x11\xdbX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0(\xf6?w:\x88X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?cL\xad\x87w\x8c*X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xb92\xb2\x1f\x98;\x08u.' and reward: 0.3724
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd4\t\xbeb\x90\x11\xdbX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0(\xf6?w:\x88X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?cL\xad\x87w\x8c*X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xb92\xb2\x1f\x98;\x08u.' and reward: 0.3724
 60%|██████    | 3/5 [02:04<01:18, 39.44s/it] 60%|██████    | 3/5 [02:04<01:22, 41.38s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.4713805265174775, 'embedding_size_factor': 1.2252450236681995, 'layers.choice': 1, 'learning_rate': 0.0012475835004927753, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 5.101679260023631e-10} and reward: 0.3708
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xde+\x19:WH\xf0X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3\x9a\x9a\x86\xa3\xdd\xdaX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?Tp\xbe\x95Zs^X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\x01\x87|\nl\xa5ru.' and reward: 0.3708
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xde+\x19:WH\xf0X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3\x9a\x9a\x86\xa3\xdd\xdaX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?Tp\xbe\x95Zs^X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\x01\x87|\nl\xa5ru.' and reward: 0.3708
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 199.20248651504517
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
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.74s of the -179.14s of remaining time.
Ensemble size: 25
Ensemble weights: 
[0.56 0.04 0.16 0.16 0.04 0.   0.04]
	0.3978	 = Validation accuracy score
	1.75s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 300.95s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f6f7c32ec88>

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
   88aeb60..de2e12f  master     -> origin/master
Updating 88aeb60..de2e12f
Fast-forward
 .../20200517/list_log_pullrequest_20200517.md      |   2 +-
 error_list/20200517/list_log_testall_20200517.md   | 386 ++++++++++++---------
 2 files changed, 214 insertions(+), 174 deletions(-)
